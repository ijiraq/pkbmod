from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import numpy as np
import os
import sys
import textwrap
from data_models import StackParams
from data_models import read_flag_list_from_file
import stack

APP_NAME = 'pkbmod'
EXTENSION_WITH_WCS = 1

def get_logging_handlers_and_level(level: str, filename, no_tty=False):
    level = getattr(logging, level)
    # Create a StreamHandler and set its level and format
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)  # Set the desired level for the console
    stream_formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)d %(module)s.%(funcName)s: %(levelname)-8s %(message)s')
    stream_handler.setFormatter(stream_formatter)

    # Create a FileHandler and set its level and format
    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(level)  # Set the desired level for the file
    file_formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)d %(module)-12s: %(levelname)-8s %(message)s')
    file_handler.setFormatter(file_formatter)

    handlers = [file_handler]
    if not no_tty:
        handlers.append(stream_handler)
    return handlers


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log-level', default='INFO',
                        type=str,
                        help="Configure the logging level.",
                        choices=logging.getLevelNamesMapping().keys())
    parser.add_argument('--no-tty', default=False, action='store_true')
    parser.add_argument('--flagkeys', default='data/flagkeys_nh.dat', type=str,
                        help='File with list or , seperated list of keys to mask.')
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
                        default=15,
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
        default=256,
        help="X-axis tile width used by low-memory top-k merge.")
    parser.add_argument('--variance-trim', default=1.3, type=float,
                        help="factor above median variance to mask pixels",
                        )
    parser.add_argument(
        '--float-precision',
        type=int,
        choices=[16, 32],
        default=32,
        help="Floating-point precision for CLI workflow arrays/tensors.")


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
            results: BASE_DIR/{APP_NAME}/DAY_OBS/CHIP/results.txt ........... \n
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
    btargs.add_argument('patch', type=int, help='Patch')
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
    
    args = parser.parse_args()

    # what level of floating point to use (float16 to lower memory footprint)
    run_dtype = np.float16 if args.float_precision == 16 else np.float32

    # get the flag names for pixels to mask from a file or string on command line
    if os.access(args.flagkeys, os.R_OK):
        badflags = read_flag_list_from_file(args.flagkeys)
    else:
        badflags = args.flagskeys.split(",")


    # add mode specific args and set the model
    if args.mode == 'filesystem':
       from data_models import ExtractedDataModel as DataModel
       data_model_args = {'day_obs': args.day_obs,
                          'collections': args.collections,
                          'dataset_type': args.dataset_type,
                          'bitmask_filename': args.bitmask_filename,
                          'data_dtype': run_dtype,
                          'chip': args.chip,
                          'base_dir':  args.base_dir,
                          }
       output_path = "/".join([f"{data_model_args['base_dir']}",
                     f"{APP_NAME}",
                     f"{data_model_args['day_obs']}",
                     f"results_{data_model_args['chip']}"])
       os.makedirs(output_path, exist_ok=True)

    if args.mode == 'butler':
        from butler_data_model import ButlerDataModel as DataModel
        data_model_args = {'day_obs': args.day_obs,
                           'collections': args.collections,
                           'dataset_type': args.dataset_type,
                           'butler': args.butler,
                           'skymap': args.skymap,
                           'tract': args.tract,
                           'patch': args.patch,
                           'band': args.band,
                           'instrument':  args.instrument,
                           'psf_dataset_type': args.psf_dataset_type,
                           }
        output_path = "/".join([f"{data_model_args['butler']}",
                                f"{data_model_args['collections']}",
                                f"{APP_NAME}",
                                f"{data_model_args['day_obs']}",
                                f"{data_model_args['tract']}",
                                f"{data_model_args['patch']}"])
        os.makedirs(output_path, exist_ok=True)


    logfilname = f'{output_path}/log.txt'
    logger = logging.getLogger(__name__)
    handlers, level = get_logging_handlers_and_level(args.log_level,
                                                     logfilname,
                                                     args.no_tty)

 .  logger.addHandlers(handlers)
    logger.setLevel(level)
    logging.error(f"Logger set to: {logging.getLogger().getEffectiveLevel()}")
    logging.debug("Args: {args}")
    params_filename = f"{output_path}/params.json"
    results_filename = f"{output_path}/results_.txt"
    plants_match_filename = f"{output_path}/plant_matches.txt"
    logging.info(f"Saving parameters to {params_filename}")
    logging.info(f"Saving results to {results_filename}")
    logging.info(f"Saving matched plants to {plants_match_filename}")


    # Stacking Parameters
    stack_params = StackParams(params_filename)
    stack_params.use_negative_well = not args.dontUseNegativeWell
    stack_params.min_snr = args.min_snr
    stack_params.rate_fwhm_grid_step = args.rate_fwhm_grid_step
    stack_params.n_keep = args.n_keep
    stack_params.dist_lim = args.clust_dist_lim
    stack_params.min_samp = args.clust_min_samp
    stack_params.trim_snr = args.trim_snr
    stack_params.dist_lim_x = 4
    stack_params.dist_lim_y = 6
    stack_params.peak_offset_max = args.peak_offset_max
    stack_params.variance_trim = args.variance_trim
    stack_params.badflags = badflags
    stack_params.save()

    # common arguments used by DataModel class builders


    logging.info(f"Saving log to {logfilname}")

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


if __name__ == '__main__':
    main()
