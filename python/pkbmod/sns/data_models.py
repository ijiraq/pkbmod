from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from glob import glob
import logging
import re


def _read_flag_list(flags_fn) -> [str]:
    """Read the list of flags to mask
    """
    flag_keys = []
    with open(flags_fn) as han:
        for line in han.readline():
            if line.startswith('#'):
                continue
            key = line.split()[0]
            flag_keys.append(key)

    logging.debug(f"FLAG_KEYS: {flag_keys}")
    return flag_keys


class ExtractedDataModel(object):

    WCS_EXT = 1
    IMAGE_EXT = 1
    MASK_EXT = 2
    VARIANCE_EXT = 3
    MASK_PREFIX = "MP_"

    def __init__(self, base_dir, collections, day_obs, chip, dataset_type,
                 bitmask_filename=None, badflags: list[str] | str = ['BAD']):
        self.base_dir = base_dir
        self.collections = collections
        self.day_obs = day_obs
        self.chip = chip
        self.dataset_type = dataset_type
        self.properties_dataset_type = "properties"
        self.bitmask_filename = bitmask_filename
        self.badflags = badflags
        self._warps = None
        self._psfs = None
        self._properties = None
        self._ref_wcs = None
        self._ref_visit = None
        self._ref_header = None
        self._bitmask = None
        self._plants = None

    @property
    def badflags(self) -> [str]:
        return self._badflags
    
    @badflags.setter
    def badflags(self, badflags):
        if isinstance(badflags, list):
            self._badflags = badflags
        # treate a string to laod from file
        flag_keys = []
        with open(badflags) as han:
            for line in han.readlines():
                if line.startswith('#'):
                    continue
                key = line.split()[0]
                flag_keys.append(key)
        logging.debug(f"FLAG_KEY: {flag_keys}")
        self._badflags = flag_keys

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
