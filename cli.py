from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import logger
import os
import textwrap

from stack import DataLoad, StackParams, run
from data_models import ExtractedDataModel

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
    parser.add_argument('--bitmask', default=None, type=str,
                        help=('The bitmask used with these data. '
                              '(ommit to read keys from mask extension.)'))
    parser.add_argument('--flagkeys', default='flagkeys_nh.dat', type=str,
                        help='File with list of keys to mask.')
    parser.add_argument('--clust-dist-lim', default=4.0, type=float)
    parser.add_argument('--clust-min-samp', default=2, type=int)
    parser.add_argument('--dontUseNegativeWell',
                        help="Use negative well as detection criterion",
                        default=False,
                        action='store_true')
    parser.add_argument('--kernel-width', default=15, type=int)
    parser.add_argument('--min_snr', default=4.5, type=float)
    parser.add_argument('--trim-snr', default=5.5, type=float)
    parser.add_argument('--n-keep', default=4, type=int)
    parser.add_argument('--peak-offset-max', default=4.0, type=float)
    parser.add_argument('--rate_fwhm_grid_step', default=0.75, type=float)
    parser.add_argument('--read-from-params', action='store_true',
                        default=True,
                        help=(f'Read from ROOT_DIR/{APP_NAME}/params.txt and '
                              'ignore command line inputs'))
    parser.add_argument('--use-gaussian-kernel', action='store_true',
                        default=False)
    parser.add_argument('--variance-trim', default=1.3, type=float)
    parser.add_argument('--rt', action='store_true',
                        default=False,
                        help='Run on the rt diff images instead.')
    parser.add_argument('--collections',
                        type=str,
                        default='DIFFS',
                        help="Sub-directory of BASER_DIR with warps to stack")
    parser.add_argument('--dataset-type', type=str,
                        default="diff_directWarp")
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
    bitmask = args.bitmask
    flagkeys = args.flagkeys

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

    data_model = ExtractedDataModel(base_dir, collections,
                                    day_obs, chip, dataset_type,
                                    bitmask, flagkeys)

    dataload = DataLoad(
        warps=data_model.warps,
        psfs=data_model.psfs,
        properties=data_model.properties,
        plants=data_model.plants,
        results_filename=results_filename,
        bitmask=data_model.bitmask,
        plant_matches_filename=plants_match_filename)

    # Stacking Parameters
    stackparams = StackParams(params_filename)
    stackparams.use_negative_well = not args.dontUseNegativeWell
    stackparams.min_snr = args.min_snr
    stackparams.rate_fwhm_grid_step = args.rate_fwhm_grid_step
    stackparams.n_keep = args.n_keep
    stackparams.dist_lim = args.clust_dist_lim
    stackparams.min_samp = args.clust_min_samp
    stackparams.trim_snr = args.trim_snr
    stackparams.dist_lim_x = 4
    stackparams.dist_lim_y = 6
    stackparams.peak_offset_max = args.peak_offset_max
    stackparams.variance_trim = args.variance_trim
    stackparams.badflags = data_model.badflags
    stackparams.save()

    run(dataload=dataload, stackparams=stackparams)


if __name__ == '__main__':
    main()
