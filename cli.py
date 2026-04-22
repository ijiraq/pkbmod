from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataclasses import asdict
import inspect
import gc
from pathlib import Path
import logging
import numpy as np
import os
import sys
import textwrap
from data_model import DataModelLoader
from datargs import make_parser
import stack
from pkbmod_storage import pkbmod_sns_catalog

APP_NAME = 'pkbmod'
EXTENSION_WITH_WCS = 1

logger = logging.getLogger(__name__)


def _init_kwargs_for_data_model(data_model_cls, args) -> dict:
    """Pass only parameters accepted by ``DataModel.__init__``, with dtype mapping."""
    params = inspect.signature(data_model_cls.__init__).parameters
    names = set(params) - {"self"}
    va = vars(args)
    kw = {k: va[k] for k in names if k in va}
    if "data_dtype" in names and "data_dtype" not in kw and "float_precision" in va:
        fp = int(va["float_precision"])
        if (data_model_cls.__module__ or "").endswith("butler.data_model"):
            kw["data_dtype"] = fp
        else:
            kw["data_dtype"] = np.float32 if fp == 32 else np.float16
    return kw


def configure_cli_logging(level_name: str, filename: str, *, no_tty: bool = False) -> None:
    """Configure the *root* logger so library ``logging.getLogger(__name__)`` records propagate.

    - **File**: receives every record at or above ``--log-level`` (the usual behavior).
    - **stderr**: receives INFO and above, or only stricter levels if ``--log-level`` is
      above INFO (e.g. WARNING → stderr shows WARNING+ only). Disabled with ``--no-tty``.
    """
    file_level = getattr(logging, level_name.upper(), logging.INFO)
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


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log-level', default='INFO',
                        type=str,
                        help="Configure the logging level.",
                        choices=logging.getLevelNamesMapping().keys())
    parser.add_argument('--no-tty', default=False, action='store_true')
    parser.add_argument(
        '--low-mem',
        action='store_true',
        default=False,
        help="Use low-memory shift-and-stack implementation.")
    parser.add_argument(
        '--low-mem-tile-w',
        type=int,
        default=128,
        help="X-axis tile width used by low-memory top-k merge.")
    parser.add_argument(
        '--float-precision',
        type=int,
        choices=[16, 32],
        default=32,
        help="Floating-point precision for CLI workflow arrays/tensors.")
    
    parser.add_argument('--output-dir',
                        default='./',
                        type=str,
                        help='filesystem location to write results to')
    data_model_loader = DataModelLoader()
    data_model_loader.add_subparsers(parser, dest='mode')
    # Use make_parser, not parse(): parse() would call StackParams(**all_namespace_keys)
    # and fail on parent flags (log_level, output_dir, …). make_parser only adds fields.
    make_parser(stack.StackParams, parser=parser)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sp_obj = stack.StackParams.from_dict(vars(args))
    stack_params = asdict(sp_obj)
    stack_params["dist_lim"] = stack_params["clust_dist_lim"]

    dm_cls = data_model_loader.get_data_model(args.mode)
    dm = dm_cls(**_init_kwargs_for_data_model(dm_cls, args))
    dm.mask_variance(stack_params["variance_trim"])
    dm.pack_inputs()
    metadata = dm.data_id
    results_basename = f"{APP_NAME}_{'_'.join(str(v) for v in metadata.values())}"
    logfilename = f"{args.output_dir}/{results_basename}.log"
    configure_cli_logging(args.log_level, logfilename, no_tty=args.no_tty)
    logger.debug("Args: %r", args)

    results_fits_filename = f"{args.output_dir}/{results_basename}.fits"
    stack_inputs = dm.stack_inputs
    del dm

    run_dtype = np.float32 if args.float_precision == 32 else np.float16
    result = stack.run(
        stack_inputs=stack_inputs,
        stack_params=stack_params,
        low_mem=args.low_mem,
        low_mem_tile_w=args.low_mem_tile_w,
        dtype=run_dtype,
    )

    catalog = pkbmod_sns_catalog.from_stack_result(result, metadata)
    catalog.writeFits(results_fits_filename)

if __name__ == '__main__':
    main()
