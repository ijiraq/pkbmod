from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
import textwrap

def add_parser(sp, name):
    fsargs = sp.add_parser(name,
                           help="Find inputs in pre-defined filesystem paths",
                           formatter_class=ArgumentDefaultsHelpFormatter,
                           )
    fsargs.add_argument('base-dir',
                        default="/arc/projects/NewHorizons/HSC_2024/",
                        help=textwrap.dedent("""\
                        BASE_DIR is the file system path to the data storage directory.
                        Path for inputs and outputs are logically given by .......\n
                        warps: BASE_DIR/COLLECTIONS/DAY_OBS/CHIP ........... \n
                        properties: BASE_DIR/COLLECTIONS/DAY_OBS/CHIP ........... \n
                        results: BASE_DIR/APPNAME/DAY_OBS/CHIP/sns_DAY_OBS_cCHIP_detections.txt ......... \n
                        inputs: BASE_DIR/APPNAME/DAY_OBS/CHIP/params.json ........... \n
                        log: BASE_DIR/APP_NAME/DAY_OBS/CHIP/log.txt files to ........
                        """),
                        metavar='BASE_DIR')
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
    fsargs.set_defaults(collections="DIFFS",
                        dataset_type="diff_directWarp")
    fsargs.add_argument('--bitmask-filename', type=str,
                        help=('The bitmask used with these data. '
                              '(ommit to read keys from mask extension.)'))
