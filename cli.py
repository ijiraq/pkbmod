from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import deque
import gc
import logging
import numpy as np
import os
import sys
import textwrap
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from data_models import StackParams
from data_models import read_flag_list_from_file
from load_budget import LoadMemoryBudget, stack_inputs_array_nbytes
import stack

try:
    import psutil
except ImportError:
    psutil = None

APP_NAME = 'pkbmod'
EXTENSION_WITH_WCS = 1
_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_FLAGKEYS = _PKG_DIR / "data" / "flagkeys_nh.dat"

logger = logging.getLogger(__name__)


def default_fakes_collection(day_obs: int) -> str:
    return f"fakes/master-fakes/{day_obs}"


def resolve_flagkeys_path(flagkeys: str) -> Path:
    """Resolve --flagkeys relative to cwd, then the package directory."""
    path = Path(flagkeys)
    if path.is_file():
        return path.resolve()
    pkg_path = _PKG_DIR / path
    if pkg_path.is_file():
        return pkg_path
    return path


def load_badflags(flagkeys: str) -> list[str]:
    """Load mask plane names from a file path or comma-separated list."""
    flagkeys_path = resolve_flagkeys_path(flagkeys)
    if flagkeys_path.is_file():
        return read_flag_list_from_file(flagkeys_path)
    if "/" in flagkeys or flagkeys.endswith(".dat"):
        logger.warning(
            "Flagkeys file not found: %s (cwd=%s, package=%s)",
            flagkeys,
            os.getcwd(),
            _PKG_DIR,
        )
    return [f.strip() for f in flagkeys.split(",") if f.strip()]


def configure_cli_logging(level_name: str, filename: str, *, no_tty: bool = False) -> None:
    """Configure the *root* logger so library ``logging.getLogger(__name__)`` records propagate.

    - **File**: receives every record at or above ``--log-level`` (the usual behavior).
    - **stderr**: receives INFO and above, or only stricter levels if ``--log-level`` is
      above INFO (e.g. WARNING → stderr shows WARNING+ only). Disabled with ``--no-tty``.
    """
    file_level = getattr(logging, level_name.upper(), logging.INFO)
    # At least INFO on the console unless user chose something stricter
    stream_level = max(logging.INFO, file_level)

    file_formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)d %(module)-12s: %(levelname)-8s %(message)s"
    )
    stream_formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)d %(module)s.%(funcName)s: %(levelname)-8s %(message)s"
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename, mode="a", encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(file_formatter)
    root.addHandler(fh)

    if not no_tty:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(stream_level)
        sh.setFormatter(stream_formatter)
        root.addHandler(sh)


def _apply_stack_params_from_args(stack_params: StackParams, args) -> None:
    stack_params.use_negative_well = not args.dontUseNegativeWell
    stack_params.kernel_width = args.kernel_width
    stack_params.use_gaussian_kernel = args.use_gaussian_kernel
    stack_params.min_snr = args.min_snr
    stack_params.rate_fwhm_grid_step = args.rate_fwhm_grid_step
    stack_params.n_keep = args.n_keep
    stack_params.dist_lim = args.clust_dist_lim
    stack_params.min_samp = args.clust_min_samp
    stack_params.trim_snr = args.trim_snr
    stack_params.dist_lim_x = 4
    stack_params.dist_lim_y = 6
    stack_params.peak_offset_max = args.peak_offset_max
    stack_params.dist_max = args.dist_max
    stack_params.dist_rate_max = args.dist_rate_max
    stack_params.variance_trim = args.variance_trim
    stack_params.sat_dilate_pixels = args.sat_dilate_pixels
    stack_params.badflags = args.badflags


def _load_and_pack_butler_patch(
    *,
    day_obs: int,
    collections: str,
    dataset_type: str,
    butler: str,
    skymap: str,
    tract: int,
    patch: int,
    band: str,
    instrument: str,
    psf_dataset_type: str,
    data_dtype: np.dtype,
    variance_trim: float,
    injection_catalog_collections: str,
) -> tuple[dict, str, str, str, int, int]:
    """Blocking I/O + CPU prep in a worker thread. Returns stack_inputs, paths, ref count, nbytes."""
    from butler_data_model import ButlerDataModel

    output_path = "/".join(
        [butler, collections, APP_NAME, str(day_obs), str(tract), str(patch)]
    )
    os.makedirs(output_path, exist_ok=True)
    results_basename = f"sns_{day_obs}_{band}_{tract}_{patch}_detections.txt"
    results_filename = f"{output_path}/{results_basename}"
    plants_match_filename = f"{output_path}/plant_matches.txt"
    params_filename = f"{output_path}/params.json"

    dm = ButlerDataModel(
        butler=butler,
        collections=collections,
        day_obs=day_obs,
        skymap=skymap,
        tract=tract,
        patch=patch,
        band=band,
        instrument=instrument,
        dataset_type=dataset_type,
        data_dtype=data_dtype,
        psf_dataset_type=psf_dataset_type,
        injection_catalog_collections=injection_catalog_collections,
    )
    dm.mask_variance(variance_trim)
    dm.pack_inputs()
    n_refs = len(dm.refs)
    stack_inputs = dm.stack_inputs
    nbytes = stack_inputs_array_nbytes(stack_inputs)
    del dm
    gc.collect()
    return stack_inputs, results_filename, plants_match_filename, params_filename, n_refs, nbytes


def run_butler_patches_pipeline(args, badflags: list[str], run_dtype: type) -> None:
    """Load patches in parallel (thread pool) with RAM back-pressure; stack.run sequentially."""
    from lsst.daf.butler import Butler

    from butler_data_model import count_datasets_for_patch

    patches = list(args.patches)
    max_ram = float(args.max_ram_percent)
    max_parallel = max(1, int(args.max_parallel_loads))
    if psutil is None:
        logger.warning(
            "psutil not installed — RSS gate disabled; "
            "reservation vs --container-ram-gib still applies. "
            "Install psutil for full --max-ram-percent behavior."
        )

    data_dtype = np.float16 if run_dtype == np.float16 else np.float32

    container_bytes = int(float(args.container_ram_gib) * (1024**3))
    bytes_per_exposure = float(args.bytes_per_exposure_mib) * (1024**2)
    budget = LoadMemoryBudget(
        container_cap_bytes=container_bytes,
        max_ram_fraction=max_ram / 100.0,
        bytes_per_exposure=bytes_per_exposure,
        ema_alpha=float(args.budget_ema),
    )

    shared_butler = Butler(args.butler, collections=args.collections)
    fakes_collection = getattr(args, 'fakes_collection', None) or default_fakes_collection(
        args.day_obs
    )

    def submit_if_possible(ex: ThreadPoolExecutor, pending: deque, futures: dict) -> None:
        while pending:
            patch = pending[0]
            try:
                n_refs = count_datasets_for_patch(
                    shared_butler,
                    collections=args.collections,
                    dataset_type=args.dataset_type,
                    instrument=args.instrument,
                    day_obs=args.day_obs,
                    skymap=args.skymap,
                    tract=args.tract,
                    patch=patch,
                    band=args.band,
                )
            except Exception:
                logger.exception("Ref query failed for patch %s", patch)
                pending.popleft()
                continue

            estimate = budget.estimate_patch_bytes(n_refs)
            if not budget.can_start_load(
                estimate,
                n_running_loads=len(futures),
                max_parallel=max_parallel,
            ):
                return

            pending.popleft()
            budget.reserve(estimate)
            fut = ex.submit(
                _load_and_pack_butler_patch,
                day_obs=args.day_obs,
                collections=args.collections,
                dataset_type=args.dataset_type,
                butler=args.butler,
                skymap=args.skymap,
                tract=args.tract,
                patch=patch,
                band=args.band,
                instrument=args.instrument,
                psf_dataset_type=args.psf_dataset_type,
                data_dtype=data_dtype,
                variance_trim=args.variance_trim,
                injection_catalog_collections=fakes_collection,
            )
            futures[fut] = (patch, estimate)
            logger.info(
                "Submitted background load for patch %s (n_refs=%d, est ~%.2f GiB, RAM ~%.1f%%, "
                "reserved ~%.2f GiB, bytes/ref ~%.1f MiB)",
                patch,
                n_refs,
                estimate / (1024**3),
                budget.memory_percent(),
                budget.reserved_bytes / (1024**3),
                budget.bytes_per_exposure / (1024**2),
            )

    pending = deque(patches)
    futures = {}

    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        submit_if_possible(ex, pending, futures)

        while pending or futures:
            if not futures:
                if not pending:
                    break
                # Always try to submit first. Do not gate on RSS before submit — that caused
                # deadlock: with no futures, nothing completes to lower RSS, yet we never called
                # submit_if_possible (LoadMemoryBudget.can_start_load handles admission).
                submit_if_possible(ex, pending, futures)
                if not futures and pending:
                    if psutil is not None and budget.memory_percent() >= max_ram:
                        logger.info(
                            "Could not start next load yet (RAM ~%.1f%%, cap %.1f%%); retrying…",
                            budget.memory_percent(),
                            max_ram,
                        )
                        time.sleep(0.4)
                    else:
                        time.sleep(0.1)
                continue

            done, _ = wait(futures.keys(), timeout=2.0, return_when=FIRST_COMPLETED)
            if not done:
                submit_if_possible(ex, pending, futures)
                continue

            for fut in done:
                patch, reserved_estimate = futures.pop(fut)
                try:
                    (
                        stack_inputs,
                        results_filename,
                        plants_match_filename,
                        params_filename,
                        n_refs,
                        array_nbytes,
                    ) = fut.result()
                except Exception:
                    budget.release(reserved_estimate)
                    logger.exception("Load failed for patch %s", patch)
                    continue

                budget.release(reserved_estimate)
                budget.observe_completed_load(n_refs, array_nbytes)

                stack_params = StackParams(params_filename)
                _apply_stack_params_from_args(stack_params, args)
                stack_params.save()

                logger.info(
                    "Running stack for patch %s (RAM ~%.1f%%)",
                    patch,
                    budget.memory_percent(),
                )
                try:
                    stack.run(
                        stack_inputs=stack_inputs,
                        stack_params=dict(stack_params),
                        results_filename=results_filename,
                        plant_matches_filename=plants_match_filename,
                        low_mem=args.low_mem,
                        low_mem_tile_w=args.low_mem_tile_w,
                        dtype=run_dtype,
                    )
                finally:
                    del stack_inputs
                    gc.collect()
                    logger.info(
                        "Finished patch %s; RAM ~%.1f%%",
                        patch,
                        budget.memory_percent(),
                    )

            submit_if_possible(ex, pending, futures)


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log-level', default='INFO',
                        type=str,
                        help="Configure the logging level.",
                        choices=logging.getLevelNamesMapping().keys())
    parser.add_argument('--no-tty', default=False, action='store_true')
    parser.add_argument(
        '--flagkeys',
        default=str(_DEFAULT_FLAGKEYS),
        type=str,
        help='File with list or comma-separated mask plane names.',
    )
    parser.add_argument(
        '--clust-dist-lim',
        default=4.0,
        help="maximum distance between candidate and linear motion",
        type=float)
    parser.add_argument(
        '--clust-min-samp',
        default=2,
        type=int,
        help="minimum number of clustered detections required")
    parser.add_argument('--dontUseNegativeWell',
                        action='store_true',
                        default=False,
                        help="Use negative well as detection criterion")
    parser.add_argument('--kernel-width',
                        type=int,
                        default=14,
                        help="Width of the psf kernel",)
    parser.add_argument('--min_snr',
                        type=float,
                        default=4.5,
                        help="Minimum SNR to be considered a detection")
    parser.add_argument(
        '--trim-snr',
        help="After clustering, trim candidates with SNR below this value",
        default=5.5, type=float)
    parser.add_argument(
        '--n-keep',
        help="For each pixel examine the n-keep rates with highest SNR",
        default=4, type=int)
    parser.add_argument(
        '--peak-offset-max',
        help="max distance between peak and centre of stamp",
        default=4.0, type=float)
    parser.add_argument(
        '--dist-max',
        type=float,
        default=4.0,
        help=(
            "Max pixel distance (planted x0/y0 vs detection) "
            "for a plant-detection association and det_* match flags."
        ),
    )
    parser.add_argument(
        '--dist-rate-max',
        type=float,
        default=60.0,
        help=(
            "Max Euclidean separation in rate space (planted rate vs "
            "detection rate grid) for association; units match StackParams.rate_x/y."
        ),
    )
    parser.add_argument(
        '--rate_fwhm_grid_step',
        help="width of rate grid steps in units of FWHM",
        default=0.75, type=float)
    parser.add_argument(
        '--read-from-params',
        action='store_true',
        default=False,
        help=(f'Read from ROOT_DIR/{APP_NAME}/params.txt and '
              'ignore command line inputs'))
    parser.add_argument(
        '--use-gaussian-kernel',
        action='store_true',
        default=False,
        help="Don't use a PSF model file, build kernel using guassian.")
    parser.add_argument(
        '--low-mem',
        action='store_true',
        default=False,
        help="Use low-memory shift-and-stack implementation.")
    parser.add_argument(
        '--low-mem-tile-w',
        type=int,
        default=512,
        help="X-axis tile width used by low-memory top-k merge.")
    parser.add_argument('--variance-trim', default=1.3, type=float,
                        help="factor above median variance to mask pixels",
                        )
    parser.add_argument(
        '--sat-dilate-pixels',
        type=int,
        default=2,
        help="Grow SAT mask by N pixels in all directions; 0 disables.",
    )
    parser.add_argument(
        '--float-precision',
        type=int,
        choices=[16, 32],
        default=32,
        help="Floating-point precision for CLI workflow arrays/tensors.")
    parser.add_argument(
        '--max-ram-percent',
        type=float,
        default=50.0,
        help=(
            "Butler mode (multi-patch): fraction of --container-ram-gib used as the ceiling "
            "for process RSS and for the sum of in-flight load reservations (parallel loads). "
            "RSS gate requires psutil."
        ),
    )
    parser.add_argument(
        '--max-parallel-loads',
        type=int,
        default=5,
        help=(
            "Butler mode: maximum concurrent ButlerDataModel loads (I/O threads). "
            "stack.run still runs one at a time on the main thread."
        ),
    )
    parser.add_argument(
        '--container-ram-gib',
        type=float,
        default=32.0,
        help=(
            "Butler mode: total RAM of the environment (GiB) for RSS %% and reservation cap "
            "(e.g. cgroup limit), not necessarily host physical RAM."
        ),
    )
    parser.add_argument(
        '--bytes-per-exposure-mib',
        type=float,
        default=200.0,
        help=(
            "Butler mode: initial heap estimate per exposure ref for load scheduling; "
            "refined from measured stack_inputs size (EMA)."
        ),
    )
    parser.add_argument(
        '--budget-ema',
        type=float,
        default=0.15,
        help="Butler mode: EMA weight when updating bytes-per-exposure after each load.",
    )

    sp = parser.add_subparsers(dest='mode')

    fsargs = sp.add_parser('filesystem', help="Find inputs in pre-defined filesystem paths",
                           formatter_class=ArgumentDefaultsHelpFormatter,
                           )
            # If --rt is used, {APP_NAME} will be replaced with rt{APP_NAME}
    fsargs.add_argument('day_obs', type=int, help='day_obs', default=20240811)
    fsargs.add_argument('chip', type=int, help='Chip')
    fsargs.add_argument('--collections',
                        type=str,
                        help="name of collection/sub-dir with warps to stack")
    fsargs.add_argument('--dataset-type', type=str,
                        help="dataset type of difference images to stack")
    fsargs.add_argument('--rt', action='store_true',
                        default=False,
                        help='Run on the reverse time diff images instead.')
    fsargs.add_argument('--base-dir',
                        default="/arc/projects/NewHorizons/HSC_2024/",
                        help=textwrap.dedent(f"""\
            BASE_DIR is the file system path to the data storage directory.
            Path for inputs and outputs are logically given by .......\n
            warps: BASE_DIR/COLLECTIONS/DAY_OBS/CHIP ........... \n
            properties: BASE_DIR/COLLECTIONS/DAY_OBS/CHIP ........... \n
            results: BASE_DIR/{APP_NAME}/DAY_OBS/CHIP/sns_DAY_OBS_cCHIP_detections.txt ........... \n
            inputs: BASE_DIR/{APP_NAME}/DAY_OBS/CHIP/params.json ........... \n
            log: BASE_DIR/{APP_NAME}/DAY_OBS/CHIP/log.txt files to ........
            """),
                        metavar='BASE_DIR')
    fsargs.set_defaults(collections="DIFFS",
                        dataset_type="diff_directWarp")
    fsargs.add_argument('--bitmask-filename', type=str,
                        help=('The bitmask used with these data. '
                              '(ommit to read keys from mask extension.)'))

    btargs = sp.add_parser('butler', help="Find inputs using the LSST Butler",
                           formatter_class=ArgumentDefaultsHelpFormatter,
                           description=textwrap.dedent(f"""
                           Use the LSST Butler to find data to stack, also assumes
                           the bitmask flag name to value map is in the header of image"""),
                           )
    btargs.add_argument("butler",
                        help="LSST Butler path",
                        default="/arc/projects/NewHorizons/HSC_2024/PG2_BUTLER")
    btargs.add_argument('day_obs', type=int, help='day_obs')
    btargs.add_argument('skymap', type=str, help='skymap name')
    btargs.add_argument('tract', type=int, help='Tract')
    btargs.add_argument('patches', type=int, nargs='+', help='Patch id(s), one or more')
    btargs.add_argument('--collections',
                        type=str,
                        help="name of collection/sub-dir with warps to stack")
    btargs.add_argument('--dataset-type', type=str,
                        help="dataset type of difference images to stack")
    btargs.set_defaults(collections="u/NH/coadd",
                        dataset_type="injected_diff_directWarp")
    btargs.add_argument('--band', type=str, help='Band', default='gri')
    btargs.add_argument('--instrument', type=str, help='Instrument', default='HSC')
    btargs.add_argument('--psf-dataset-type', type=str, help='What dataset to get PSF from',
                        default='injected_calexp')
    btargs.add_argument(
        '--fakes-collection',
        type=str,
        default=None,
        help='Butler collection for injection_catalog (default: fakes/master-fakes/{day_obs})',
    )

    stampsargs = sp.add_parser(
        'stamps-butler',
        help=(
            "Build CFHT-style motion-compensated stamp pickles from Butler warps "
            "and sns_*_detections.txt."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    stampsargs.add_argument('butler', help='LSST Butler repository path')
    stampsargs.add_argument('day_obs', type=int, help='day_obs (must match detection run)')
    stampsargs.add_argument('skymap', type=str, help='skymap name')
    stampsargs.add_argument('tract', type=int, help='Tract')
    stampsargs.add_argument('patch', type=int, help='Patch id')
    stampsargs.add_argument('--collections', type=str, default='u/NH/coadd')
    stampsargs.add_argument(
        '--dataset-type',
        type=str,
        default='injected_diff_directWarp',
    )
    stampsargs.add_argument('--band', type=str, default='gri')
    stampsargs.add_argument('--instrument', type=str, default='HSC')
    stampsargs.add_argument('--psf-dataset-type', type=str, default='injected_calexp')
    stampsargs.add_argument(
        '--detections-file',
        type=str,
        default=None,
        help=(
            "Path to sns_*_detections.txt. Default: "
            "{butler}/{collections}/pkbmod/{day_obs}/{tract}/{patch}/"
            "sns_{day_obs}_{band}_{tract}_{patch}_detections.txt"
        ),
    )
    stampsargs.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for pickle outputs (default: same dir as detections file)',
    )
    stampsargs.add_argument(
        '--variance-trim',
        type=float,
        default=None,
        help='Override variance trim (default: params.json next to detections, else 1.3)',
    )
    stampsargs.add_argument(
        '--mask-planes',
        type=str,
        default=None,
        dest='stamps_mask_planes',
        metavar='PLANES',
        help='Comma-separated mask plane names (override params.json / --flagkeys)',
    )
    stampsargs.add_argument(
        '--no-wide',
        action='store_true',
        help='Only write 21x21 cutouts (omit 43x43 wide stamps)',
    )

    args = parser.parse_args()

    # what level of floating point to use (float16 to lower memory footprint)
    run_dtype = np.float16 if args.float_precision == 16 else np.float32

    badflags = load_badflags(args.flagkeys)
    args.badflags = badflags

    # add mode specific args and set the model
    if args.mode == 'filesystem':
        from data_models import ExtractedDataModel as DataModel
        data_model_args = {'day_obs': args.day_obs,
                           'collections': args.collections,
                           'dataset_type': args.dataset_type,
                           'bitmask_filename': args.bitmask_filename,
                           'data_dtype': run_dtype,
                           'chip': args.chip,
                           'base_dir': args.base_dir,
                           }
        output_path = "/".join([f"{data_model_args['base_dir']}",
                                f"{APP_NAME}",
                                f"{data_model_args['day_obs']}",
                                f"results_{data_model_args['chip']}"])
        os.makedirs(output_path, exist_ok=True)
        results_basename = f"sns_{args.day_obs}_c{args.chip}_detections.txt"

        logfilname = f"{output_path}/log.txt"
        configure_cli_logging(args.log_level, logfilname, no_tty=args.no_tty)
        logger.debug("Args: %r", args)
        params_filename = f"{output_path}/params.json"
        results_filename = f"{output_path}/{results_basename}"
        plants_match_filename = f"{output_path}/plant_matches.txt"
        logger.info("Saving parameters to %s", params_filename)
        logger.info("Saving results to %s", results_filename)
        logger.info("Saving matched plants to %s", plants_match_filename)

        stack_params = StackParams(params_filename)
        _apply_stack_params_from_args(stack_params, args)
        stack_params.badflags = badflags
        stack_params.save()

        logger.info("Saving log to %s", logfilname)

        data_model = DataModel(**data_model_args)
        data_model.mask_variance(stack_params.variance_trim)
        data_model.pack_inputs()

        stack.run(stack_inputs=data_model.stack_inputs,
                  stack_params=dict(stack_params),
                  results_filename=results_filename,
                  plant_matches_filename=plants_match_filename,
                  low_mem=args.low_mem,
                  low_mem_tile_w=args.low_mem_tile_w,
                  dtype=run_dtype)
        return

    if args.mode == 'stamps-butler':
        from motion_stamps_butler import (
            CUTOUT_NARROW,
            CUTOUT_WIDE,
            resolve_variance_trim_and_badflags,
            run_motion_stamps,
        )

        if args.detections_file:
            results_filename = args.detections_file
        else:
            results_filename = "/".join(
                [
                    args.butler,
                    args.collections,
                    APP_NAME,
                    str(args.day_obs),
                    str(args.tract),
                    str(args.patch),
                    f"sns_{args.day_obs}_{args.band}_{args.tract}_{args.patch}_detections.txt",
                ]
            )
        output_path = "/".join(
            [
                args.butler,
                args.collections,
                APP_NAME,
                str(args.day_obs),
                str(args.tract),
                str(args.patch),
            ]
        )
        os.makedirs(output_path, exist_ok=True)
        logfilname = f"{output_path}/log.txt"
        configure_cli_logging(args.log_level, logfilname, no_tty=args.no_tty)
        logger.debug("stamps-butler args: %r", args)

        default_bf = load_badflags(args.flagkeys)
        params_path = Path(results_filename).parent / "params.json"
        bf_override = (
            args.stamps_mask_planes.split(",")
            if getattr(args, "stamps_mask_planes", None)
            else None
        )
        var_trim, badf = resolve_variance_trim_and_badflags(
            params_path=params_path,
            variance_trim_cli=args.variance_trim,
            badflags_cli=bf_override,
            default_badflags=default_bf,
        )
        cutouts = (CUTOUT_NARROW,) if args.no_wide else (CUTOUT_NARROW, CUTOUT_WIDE)
        run_motion_stamps(
            butler=args.butler,
            collections=args.collections,
            day_obs=args.day_obs,
            skymap=args.skymap,
            tract=args.tract,
            patch=args.patch,
            band=args.band,
            dataset_type=args.dataset_type,
            instrument=args.instrument,
            psf_dataset_type=args.psf_dataset_type,
            detections_path=results_filename,
            output_dir=args.output_dir,
            flag_keys=badf,
            variance_trim=var_trim,
            data_dtype=run_dtype,
            cutout_sizes=cutouts,
        )
        return

    if args.mode == 'butler':
        # Log to first patch output dir (each patch also logs via root logger 
        # to same file if same handler — we reconfigure once)
        first_patch = args.patches[0]
        output_path = "/".join([
            args.butler,
            args.collections,
            APP_NAME,
            str(args.day_obs),
            str(args.tract),
            str(first_patch),
        ])
        os.makedirs(output_path, exist_ok=True)
        logfilname = f"{output_path}/log.txt"
        configure_cli_logging(args.log_level, logfilname, no_tty=args.no_tty)
        logger.debug("Args: %r", args)
        args.fakes_collection = (
            args.fakes_collection
            or default_fakes_collection(args.day_obs)
        )
        logger.info(
            "Butler pipeline: %d patch(es) %s; collections=%s fakes_collection=%s; "
            "max_parallel_loads=%s max_ram_percent=%s "
            "container_ram_gib=%s bytes_per_exposure_mib=%s budget_ema=%s",
            len(args.patches),
            list(args.patches),
            args.collections,
            args.fakes_collection,
            args.max_parallel_loads,
            args.max_ram_percent if psutil else "n/a",
            args.container_ram_gib,
            args.bytes_per_exposure_mib,
            args.budget_ema,
        )

        run_butler_patches_pipeline(args, badflags, run_dtype)
        return

    parser.error("Choose a mode: filesystem, butler, or stamps-butler")


if __name__ == '__main__':
    main()
