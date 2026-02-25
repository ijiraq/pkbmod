from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import logger
import os
import textwrap

from data_models import ExtractedDataModel
from data_models import StackParams
from data_models import read_flag_list_from_file
from stack import run

APP_NAME = 'pkbmod'
EXTENSION_WITH_WCS = 1


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'day_obs',
        help="The day-obs directory in {BASE_DIR}/{COLLECTIONS} to process",
        default='20240811')
    parser.add_argument('chip',
                        help="sub-directory of VISIT to process",
                        default='00')
    parser.add_argument('--log-level', default='INFO',
                        type=str,
                        help="Configure the logging level.",
                        choices=logging.getLevelNamesMapping().keys())
    parser.add_argument('--no-tty', default=False, action='store_true')
    parser.add_argument('--bitmask', type=str,
                        help=('The bitmask used with these data. '
                              '(ommit to read keys from mask extension.)'))
    parser.add_argument('--flagkeys', default='flagkeys_nh.dat', type=str,
                        help='File with list of keys to mask.')
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
    parser.add_argument('--variance-trim', default=1.3, type=float,
                        help="factor above median variance to mask pixels",
                        )
    parser.add_argument('--rt', action='store_true',
                        default=False,
                        help='Run on the reverse time diff images instead.')
    parser.add_argument('--collections',
                        type=str,
                        default='DIFFS',
                        help="Sub-directory of BASER_DIR with warps to stack")
    parser.add_argument('--dataset-type', type=str,
                        default="diff_directWarp",
                        help="dataset type of difference images to stack")
    parser.add_argument(
        '--base-dir',
        default='/arc/projects/NewHorizons/HSC_2024',
        help=textwrap.dedent(f"""
            Root path for inputs and outputs:
                warps: BASE_DIR/COLLECTIONS/DAY_OBS/CHIP,
                properties: BASE_DIR/COLLECTIONS/DAY_OBS/CHIP,
                results: BASE_DIR/{APP_NAME}/DAY_OBS/CHIP/results.txt,'
                inputs: BASE_DIR/{APP_NAME}/DAY_OBS/CHIP/params.json,
                log: BASE_DIR/{APP_NAME}/DAY_OBS/CHIP/log.txt files to.
            If --rt is used, {APP_NAME} will be replaced with rt{APP_NAME}"""))
    args = parser.parse_args()

    rt = '' if not args.rt else 'rt'
    base_dir = args.base_dir
    collections = args.collections
    dataset_type = args.dataset_type
    day_obs = args.day_obs
    chip = args.chip
    bitmask_filename = args.bitmask
    flaglist_filename = args.flagkeys

    path = "/".join([base_dir,
                     f"{rt}{APP_NAME}",
                     day_obs,
                     f"results_{chip}"])
    os.makedirs(path, exist_ok=True)
    logfilname = f'{path}/log.txt'
    _ = logger.config_logging(args.log_level,
                              logfilname,
                              args.no_tty)
    params_filename = f"{path}/params.json"
    results_filename = f"{path}/results_.txt"
    plants_match_filename = f"{path}/plant_matches.txt"

    logging.info(f"Saving log to {logfilname}")
    logging.info(f"Saving parameters to {params_filename}")
    logging.info(f"Saving results to {results_filename}")
    logging.info(f"Saving matched plants to {plants_match_filename}")
    
    badflags = read_flag_list_from_file(flaglist_filename)

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

    data_model = ExtractedDataModel(base_dir, collections,
                                    day_obs, chip, dataset_type,
                                    bitmask_filename=bitmask_filename)

    data_model.mask_variance(stack_params.variance_trim)
    data_model.pack_inputs()

    run(stack_inputs=data_model.stack_inputs,
        stack_params=dict(stack_params),
        results_filename=results_filename,
        plant_matches_filename=plants_match_filename)


if __name__ == '__main__':
    main()
