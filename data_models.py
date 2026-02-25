from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from dataclasses import dataclass, field, asdict
import json
from glob import glob
import logging
import numpy as np
import re
from typing import List


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

    logging.debug(f"FLAG_KEYS: {flag_keys}")
    return flag_keys


@dataclass
class StackParams(object):
    params_filename: str  # filename to save parameters to
    badflags: List[str] = field(default_factory=list)  # bad flags
    dist_lim: float = 5.0  # candidate-line distance in clustering routine
    dist_lim_x: int = 4  # maximum cluster distance in x
    dist_lim_y: int = 6  # maximum cluster distance in y
    dist_max: float = 4.0  # maximum spatial sep planted/detected link
    dist_rate_max: float = 60.0  # maximum rate sep  for plan/detected link
    kernel_width: int = 14  # width of kernel in pixels
    min_samp: int = 3  # minimum number of clustered detections required
    min_snr: float = 4.5  # Minimum SNR for a detection
    n_keep: int = 10000  # number of sources to keep after initial serach
    peak_offset_max: float = 4  # max distance between peak and centre of stamp
    rate_fwhm_grid_step: float = 0.75  # width of steps in units of FWHM
    trim_snr: float = 5.5  # min SNR of sources to keep after clustering
    use_gaussian_kernel: bool = False  # use a guassian kernel instead of a PSF
    use_negative_well: bool = True  # use the negative well for detection.
    variance_trim: float = 1.3  # factor above median variance to mask pixels

    def save(self) -> None:
        logging.info(f"Saving params to {self.params_filename}")
        with open(self.params_filename, 'w+') as han:
            json.dump(asdict(self), han)

    def __iter__(self) -> dict:
        yield from asdict(self).items()


class ExtractedDataModel(object):
    WCS_EXT = 1
    IMAGE_EXT = 1
    MASK_EXT = 2
    VARIANCE_EXT = 3
    MASK_PREFIX = "MP_"
    MAX_PIX_VALUE = 8000
    MIN_PIX_VALUE = -10000
    VARIANCE_BITMASK = "SAT"

    def __init__(self, base_dir, collections, day_obs, chip, dataset_type,
                 bitmask_filename=None):
        self.base_dir = base_dir
        self.collections = collections
        self.day_obs = day_obs
        self.chip = chip
        self.dataset_type = dataset_type
        self.properties_dataset_type = "properties"
        self.bitmask_filename = bitmask_filename
        self._warps = None
        self._psfs = None
        self._properties = None
        self._ref_wcs = None
        self._ref_visit = None
        self._ref_header = None
        self._bitmask = None
        self._plants = None
        self._stack_inputs = None

    # mask high, low, nan and inf pixels and
    # remove images that are fully masked
    def mask_variance(self, variance_trim) -> {'str': [np.array]}:
        """set the var_trim_keyword mask to ON if data exceeds variance_trim
        fraction of variance.

        Args:
            variance_trim (float): variance threshold fraction
            var_trim_keyword (str, optional): mask string. Default 'SAT'
        """
        datas = self.stack_inputs['datas']
        variances = self.stack_inputs['variances']
        masks = self.stack_inputs['masks']
        dmjds = self.stack_inputs['dmjds']
        im_nums = self.stack_inputs['im_nums']
        for idx in range(len(datas)):
            w = np.where((np.isinf(variances[idx])) |
                         (np.isinf(datas[idx])) |
                         (np.isnan(datas[idx])) |
                         (datas[idx] > self.MAX_PIX_VALUE) |
                         (datas[idx] < self.MIN_PIX_VALUE))
            masks[idx][w] |= 2**self.bitmask[self.VARIANCE_BITMASK]
            variances[idx][w] = np.nan
            datas[idx][w] = 0.0
            nan_med_variance = np.nanmedian(variances[idx])
            logging.debug((f"{im_nums[idx]} {dmjds[idx]} {nan_med_variance}"))
            if np.isnan(nan_med_variance):
                logging.debug('Skipping image {im_nums[idx]} due to nans.')
                for key in self.stack_inputs:
                    _ = self._stack_inputs[key].pop(idx)
            else:
                w = np.where(
                    variances[idx] >
                    variance_trim*nan_med_variance)
                masks[idx][w] |= 2**self.bitmask[self.VARIANCE_BITMASK]

    def pack_inputs(self) -> None:
        """convert list of arrays in stack_inputs into 3d arrays

        """
        for key in ['datas', 'masks', 'variances',
                    'dmjds', 'psfs', 'fwhms', 'im_nums']:
            self.stack_inputs[key] = np.array(self.stack_inputs[key])
        # im_nums should be int arrays
        self.stack_inputs['im_nums'] = (
            self.stack_inputs['im_nums'].astype('int'))
        return self.stack_inputs

    @property
    def stack_inputs(self) -> {str: [np]}:
        """ pack data into a dictionary of numpy arrays
            for use in shift-and-stack code.
        """
        if self._stack_inputs is not None:
            return self._stack_inputs
        PSF_DATA_EXTNO = 0
        DATA_EXTNO = 1
        MASK_EXTNO = 2
        VARIANCE_EXTNO = 3
        datas, masks, variances = [], [], []
        dmjds, psfs, fwhms, im_nums = [], [], [], []
        logging.debug("Creating numpy data lists to pack data onto GPU with.")
        for im_num in self.warps:
            hdul = self.warps[im_num]
            datas.append(hdul[DATA_EXTNO].data)
            masks.append(hdul[MASK_EXTNO].data)
            variances.append(hdul[VARIANCE_EXTNO].data)
            dmjds.append(self.properties[im_num]['dmjd'])
            fwhms.append(self.properties[im_num]['fwhm'])
            psf_data = self.psfs[im_num][PSF_DATA_EXTNO].data
            psfs.append(psf_data/np.sum(psf_data))
            im_nums.append(im_num)
        logging.debug(f"Using {len(datas)} images.")
        self._stack_inputs = {
            'datas': datas,
            'masks': masks,
            'variances': variances,
            'dmjds': dmjds,
            'psfs': psfs,
            'fwhms': fwhms,
            'im_nums': im_nums,
            'plants': self.plants,
            'bitmask': self.bitmask}
        return self.stack_inputs

    @property
    def path(self):
        return "/".join([self.base_dir,
                         self.collections,
                         self.day_obs,
                         self.chip])

    @property
    def filename_pattern(self) -> str:
        return f"{self.path}/{self.dataset_type}_??????_{self.chip}"

    def visit_number_from_filename(self, filename) -> int:
        visit_re = re.compile('_([0-9]{6})_')
        return int(visit_re.search(filename).group(1))

    @property
    def bitmask(self) -> {}:
        if self._bitmask is not None:
            return self._bitmask
        if self.bitmask_filename is not None:
            logging.debug("LOADNIG BITMASK FROM {self.bitmask_filename}")
            with open(self.bitmask_filename) as han:
                self._bitmask = {}
                for line in han.readlines():
                    if line.startswith('#'):
                        continue
                    s = line.split(': ')
                    key, val = s[0], int(float(s[1]))
                    self._bitmask[key] = val
            logging.debug(f"FILE BITMASK: {self._bitmask}")
        else:
            logging.debug(f"LOADING BITMASK FROM {self.ref_visit} HEADER")
            header = self.warps[self.ref_visit][self.MASK_EXT].header
            # get all keywords that start with MASK_PREFIX and then
            # strip the prefix from keyword to map to common usage
            # e.g. MP_BAD => BAD
            bitmask = header[f"{self.MASK_PREFIX}*"]
            self._bitmask = {x.removeprefix(self.MASK_PREFIX): bitmask[x]
                             for x in bitmask}
            logging.debug(f"HEADER BITMASK: {self._bitmask}")
        return self._bitmask

    @property
    def properties(self) -> {}:
        """Load the image propertes (mjd, fwhm, exposuer_time) from storage
        """
        if self._properties is not None:
            return self._properties
        prop_filename = (f"{self.path}/"
                         f"{self.properties_dataset_type}_{self.chip}.txt")
        logging.debug(f"Loading properties from {prop_filename}")
        table = Table.read(prop_filename, format='ascii.commented_header')
        # compute delta mjd as the time since the first exposure plus
        # 1/2 exposure time.
        mjd0 = table[table['visit'] == self.ref_visit]['mjd'][0]
        table['dmjd'] = table['mjd'] - mjd0
        table['dmjd'] += table['exposure_time']/2.0/3600/24.0
        properties = {}
        for visit in set(table['visit']):
            w = table['visit'] == visit
            row = table[w][0]
            properties[visit] = row['dmjd', 'fwhm']
        logging.debug(f"Loaded {len(properties)} property records")
        self._properties = properties
        return self._properties

    @property
    def psfs(self):
        """load the psfs from storage"""
        if self._psfs is not None:
            return self._psfs
        filelist = glob(self.filename_pattern+".psf.fits")
        filelist.sort()
        self._psfs = {self.visit_number_from_filename(x): fits.open(x)
                      for x in filelist}
        logging.info(f"Loaded {len(self._psfs)} psfs")
        return self._psfs

    @property
    def ref_header(self):
        if self._ref_header is None:
            self._ref_header = self.warps[self.ref_visit][self.WCS_EXT].header
        return self._ref_header

    @property
    def ref_wcs(self):
        """Get the WCS of the reference visit"""
        if self._ref_wcs is None:
            self._ref_wcs = WCS(self.ref_header)
        return self._ref_wcs

    @property
    def ref_visit(self):
        """The visit number of the reference visit,
        in this case the key of the first
        entry in the warps dictionary"""
        if self._ref_visit is None:
            self._ref_visit = next(iter(self.warps.keys()))
        return self._ref_visit

    @property
    def warps(self):
        """load warped diffs from storage"""
        if self._warps is not None:
            return self._warps
        filelist = glob(self.filename_pattern+".fits")
        filelist.sort()
        self._warps = {self.visit_number_from_filename(x): fits.open(x)
                       for x in filelist}
        logging.info(f"Loaded {len(self._warps)} warped difference images")
        return self._warps

    @property
    def plants(self) -> Table:
        """Load a list of sources injected into the reference image.
        x0: x pixel location on first image
        y0: y pixel locaiton on first image
        rate_x: rate of motion in x direction in pixels/hour
        rate_y: rate of motion in y direction in pixels/hour
        mag: the magnitude of planted soruce.
        4 blank columns, expected to hold the information from the
            detection process:
                det_shift, det_filt, det_clust, det_final, num_match
        """
        if self._plants is not None:
            return self._plants
        plant_filename = "_".join([self.dataset_type,
                                   str(self.ref_visit),
                                   str(self.chip)])
        plant_filename = f"{self.path}/{plant_filename}.plantList"
        plants = Table.read(plant_filename, format='ascii.commented_header')
        x0, y0 = self.ref_wcs.all_world2pix(plants['ra'], plants['dec'], 0)
        plants['x0'] = x0
        plants['y0'] = y0
        ra1 = plants['ra'] + plants['rate_ra']/3600.0
        dec1 = plants['dec'] + plants['rate_dec']/3600.0
        x1, y1 = self.ref_wcs.all_world2pix(ra1, dec1, 0)
        plants['rate_x'] = (x1-x0)/24.0
        plants['rate_y'] = (y1-y0)/24.0
        plants.sort('mag')
        self._plants = plants
        return self._plants
