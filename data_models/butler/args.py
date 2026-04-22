from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
import textwrap

def add_parser(sp, name):
    btargs = sp.add_parser(name,
                           help="Find inputs using the LSST Butler (single patch; use cli_multi for many)",
                           formatter_class=ArgumentDefaultsHelpFormatter,
                           description=textwrap.dedent(f"""
                           Use the LSST Butler to find data to and stack based on
                           skymap/tract/patch warps rather than individual expsosures,
                           the bitmask flag name to value map is taken from the
                           header of image"""),
                           )
    btargs.add_argument("butler",
                        help="LSST Butler path",
                        default="/arc/projects/NewHorizons/HSC_2024/PG2_BUTLER")
    btargs.add_argument('day_obs',
                        type=int,
                        help='day_obs')
    btargs.add_argument('skymap',
                        type=str,
                        help='skymap name')
    btargs.add_argument('tract',
                        type=int,
                        help='Tract')
    btargs.add_argument('patch',
                        type=int,
                        help='Single patch id')
    btargs.add_argument('--collections',
                        type=str,
                        help="name of collection/sub-dir with warps to stack")
    btargs.add_argument('--dataset-type',
                        type=str,
                        help="dataset type of difference images to stack")
    btargs.set_defaults(collections="u/NH/coadd",
                        dataset_type="injected_diff_directWarp")
    btargs.add_argument('--instrument',
                        type=str,
                        help='Instrument',
                        default='HSC')
    btargs.add_argument('--psf-dataset-type',
                        type=str,
                        help='What dataset to get PSF from',
                        default='injected_calexp')
