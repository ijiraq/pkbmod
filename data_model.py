from dataclasses import dataclass, field, fields, asdict
import importlib
import logging
import os
from pathlib import Path
import re
from typing import List

DATA_MODELS_DIR = 'data_models'

logger = logging.getLogger(__name__)

class DataModelLoader:

    def __init__(self, data_models_dir=DATA_MODELS_DIR):
        self._dmd = Path(data_models_dir)
        self.data_models = {}
        self.data_args = {}
        self._discover_data_models()

    def _discover_data_models(self):
        if not self._dmd.exists(): return
        data_models = [ x for x in self._dmd.iterdir() if x.is_dir() and not x.name.startswith('__') ]
        for data_model in data_models:
            import_base = '.'.join(data_model.parts)
            self.data_models[data_model.name] = f"{import_base}"

    def get_data_model(self, data_model):
        if data_model in self.data_models:
            data_model = importlib.import_module(f"{self.data_models[data_model]}.data_model").DataModel
            return data_model
        raise ImportError(f"DataModel {data_model} not found")

    def list_data_models(self):
        return self.data_models.keys()

    def add_subparsers(self, parser, dest='mode'):
        sp = parser.add_subparsers(dest='mode')
        for data_model in self.data_models:
            importlib.import_module(f"{self.data_models[data_model]}.args").add_parser(sp, data_model)
        


def read_flag_list_from_file(flags_fn) -> [str]:
    """Read the list of flags to mask
    """
    flag_keys = []
    with open(flags_fn) as han:
        for line in han.readlines():
            if line.startswith('#'):
                continue
            key = line.split()[0]
            flag_keys.append(key)

    logger.debug(f"FLAG_KEYS: {flag_keys}")
    return flag_keys


'''
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
    parser.add_argument('--skip-use-negative-well',
                        action='store_true',
                        default=False,
                        help="Don't use negative well as detection criterion")
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
        "--rate-min",
        help="min rate to search (in arcsec/hour)",
        default=0.4,
        type=float)
    parser.add_argument(
        "--rate-max",
        help="max rates to search (in arcsec/hour)",
        default=5.5,
        type=float)
    parser.add_argument(
        "--angle-width",
        help="opening angle width of seach cone in degrees.",
        type=float,
        default=45)
    parser.add_argument(
        '--use-gaussian-kernel',
        action='store_true',
        default=False,
        help="Don't use a PSF model file, build kernel using guassian.")
    parser.add_argument(
        '--variance-trim',
        default=1.3,
        type=float,
        help="factor above median variance to mask pixels",
    )
    parser.add_argument(
        '--read-from-params',
        action='store_true',
        default=False,
        help=(f'Read from ROOT_DIR/{APP_NAME}/params.txt and '
              'ignore command line inputs'))
'''



