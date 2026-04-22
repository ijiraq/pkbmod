class ExtractedDataModel(object):
    WCS_EXT = 1
    IMAGE_EXT = 1
    MASK_EXT = 2
    VARIANCE_EXT = 3
    MASK_PREFIX = "MP_"
    MAX_PIX_VALUE = 8000
    MIN_PIX_VALUE = -10000
    VARIANCE_BITMASK = "SAT"

    def args_parser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_subparsers(dest='filesystem')
        group.add_argument('--chip',
                        help="sub-directory of VISIT to process",
                        default='00')
        return parser

    def __init__(self, base_dir, collections, day_obs, chip, dataset_type,
                 bitmask_filename=None, data_dtype=np.float32):
        self.base_dir = base_dir
        self.collections = collections
        self.day_obs = day_obs
        self.chip = chip
        self.dataset_type = dataset_type
        self.properties_dataset_type = "properties"
        self.bitmask_filename = bitmask_filename
        self.data_dtype = data_dtype
        self._warps = None
        self._psfs = None
        self._properties = None
        self._ref_wcs = None
        self._ref_visit = None
        self._ref_header = None
        self._bitmask = None
        self._plants = None
        self._stack_inputs = None
        logger.info(f"reading data from {self.path}")

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
            mask_bit = np.uint32(1) << np.uint32(self.bitmask[self.VARIANCE_BITMASK])
            w = np.where((np.isinf(variances[idx])) |
                         (np.isinf(datas[idx])) |
                         (np.isnan(datas[idx])) |
                         (datas[idx] > self.MAX_PIX_VALUE) |
                         (datas[idx] < self.MIN_PIX_VALUE))
            masks[idx][w] |= mask_bit
            variances[idx][w] = np.nan
            datas[idx][w] = 0.0
            nan_med_variance = np.nanmedian(variances[idx])
            logger.debug((f"{im_nums[idx]} {dmjds[idx]} {nan_med_variance}"))
            if np.isnan(nan_med_variance):
                logger.debug('Skipping image {im_nums[idx]} due to nans.')
                if idx == 0:
                    logger.warning(
                        "Removing the first stacked image (visit %s); "
                        "detection_frame WCS still refers to the original "
                        "reference warp — sky coordinates may be inconsistent.",
                        im_nums[idx],
                    )
                for key in self.stack_inputs:
                    _ = self._stack_inputs[key].pop(idx)
            else:
                w = np.where(
                    variances[idx] >
                    variance_trim*nan_med_variance)
                masks[idx][w] |= mask_bit

    def pack_inputs(self) -> None:
        """convert list of arrays in stack_inputs into 3d arrays

        """
        for key in ['datas', 'variances', 'psfs', 'dmjds', 'fwhms']:
            self.stack_inputs[key] = np.array(self.stack_inputs[key],
                                              dtype=self.data_dtype)
        self.stack_inputs['masks'] = np.array(self.stack_inputs['masks'],
                                              dtype=np.uint32)
        self.stack_inputs['im_nums'] = np.array(self.stack_inputs['im_nums'],
                                                dtype=np.int32)
        # im_nums should be int arrays
        # self.stack_inputs['im_nums'] = (
        #    self.stack_inputs['im_nums'].astype('int'))
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
        logger.debug("Creating numpy data lists to pack data onto GPU with.")
        for im_num in self.warps:
            hdul = self.warps[im_num]
            datas.append(np.asarray(hdul[DATA_EXTNO].data,
                                    dtype=self.data_dtype))
            masks.append(np.asarray(hdul[MASK_EXTNO].data, dtype=np.uint32))
            variances.append(np.asarray(hdul[VARIANCE_EXTNO].data,
                                        dtype=self.data_dtype))
            dmjds.append(self.properties[im_num]['dmjd'])
            fwhms.append(self.properties[im_num]['fwhm'])
            psf_data = np.asarray(self.psfs[im_num][PSF_DATA_EXTNO].data,
                                  dtype=self.data_dtype)
            psfs.append(psf_data/np.sum(psf_data))
            im_nums.append(im_num)
        logger.debug(f"Using {len(datas)} images.")
        ref_hdr = self.ref_header.copy() if hasattr(self.ref_header, "copy") else self.ref_header
        wcs_astropy = WCS(ref_hdr)
        self._stack_inputs = {
            'datas': datas,
            'masks': masks,
            'variances': variances,
            'dmjds': dmjds,
            'psfs': psfs,
            'fwhms': fwhms,
            'im_nums': im_nums,
            'bitmask': self.bitmask,
            'detection_frame': {
                'source': 'fits_warp',
                'reference_visit': int(self.ref_visit),
                'parent_origin_xy': (0.0, 0.0),
                'wcs_astropy': wcs_astropy,
                'fits_header_text': str(ref_hdr),
            },
        }
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
            logger.debug("LOADNIG BITMASK FROM {self.bitmask_filename}")
            with open(self.bitmask_filename) as han:
                self._bitmask = {}
                for line in han.readlines():
                    if line.startswith('#'):
                        continue
                    s = line.split(': ')
                    key, val = s[0], int(float(s[1]))
                    self._bitmask[key] = val
            logger.debug(f"FILE BITMASK: {self._bitmask}")
        else:
            logging.debug(f"LOADING BITMASK FROM {self.ref_visit} HEADER")
            header = self.warps[self.ref_visit][self.MASK_EXT].header
            # get all keywords that start with MASK_PREFIX and then
            # strip the prefix from keyword to map to common usage
            # e.g. MP_BAD => BAD
            bitmask = header[f"{self.MASK_PREFIX}*"]
            self._bitmask = {x.removeprefix(self.MASK_PREFIX): bitmask[x]
                             for x in bitmask}
            logger.debug(f"HEADER BITMASK: {self._bitmask}")
        return self._bitmask

    @property
    def properties(self) -> {}:
        """Load the image propertes (mjd, fwhm, exposuer_time) from storage
        """
        if self._properties is not None:
            return self._properties
        prop_filename = (f"{self.path}/"
                         f"{self.properties_dataset_type}_{self.chip}.txt")
        logger.debug(f"Loading properties from {prop_filename}")
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
        logger.debug(f"Loaded {len(properties)} property records")
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
        logger.info(f"Loaded {len(self._psfs)} psfs")
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
        logger.info(f"Loaded {len(self._warps)} warped difference images")
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
        plants_list = []
        path = "/".join([self.base_dir,
                         self.collections,
                         self.day_obs,
                         "*"])
        plant_pattern = "_".join([self.dataset_type,
                                  str(self.ref_visit),
                                  "*"])
        plant_pattern = f"{path}/{plant_pattern}.plantList"
        for plant_filename in glob(plant_pattern):
            plants = Table.read(plant_filename, format='ascii.commented_header')
            x0, y0 = self.ref_wcs.all_world2pix(plants['ra'], plants['dec'], 0)
            plants['x0'] = x0
            plants['y0'] = y0
            ra1 = plants['ra'] + plants['rate_ra']/3600.0
            dec1 = plants['dec'] + plants['rate_dec']/3600.0
            x1, y1 = self.ref_wcs.all_world2pix(ra1, dec1, 0)
            plants['rate_x'] = (x1-x0)*24.0
            plants['rate_y'] = (y1-y0)*24.0
            plants.sort('mag')
            plants_list.append(plants)
        self._plants = vstack(plants_list, metadata_conflicts='silent')
        self._plants.sort('mag')
        return self._plants
