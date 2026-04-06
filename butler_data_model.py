"""Load shift-and-stack inputs from an LSST Gen3 Butler repository.

Requires the LSST Science Pipelines in the active Python environment. 

Typical dataset types for warped difference images include
``injected_diff_directWarp``. PSFs come from ``injected_calexp`` (see
``psf_dataset_type``): a good pixel near the warp center is converted to
RA/Dec, then the per-visit calexp whose detector footprint contains that sky
position is selected to provide the PSF.
"""
import argparse
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.table import join
from lsst.afw.image import ImageOrigin
from lsst.daf.base import DateTime  # noqa: F401  # type: ignore[import-untyped]
from lsst.daf.butler import Butler
from lsst.geom import Point2D, SpherePoint
from lsst.meas.algorithms import installGaussianPsf
import logging
import numpy as np  
from typing import Any, Mapping, Sequence


def _mask_plane_bitmask(mask) -> dict[str, int]:
    """Map LSST mask plane names to bit indices for :mod:`stack` compatibility."""
    out: dict[str, int] = {}
    for name, bit in mask.getMaskPlaneDict().items():
        out[name] = int(bit)
    return out


def _bad_mask_bits(mi) -> int:
    """Bit mask of planes treated as unusable for 'good' pixel tests."""
    bad = 0
    for plane in ("BAD", "NO_DATA", "SUSPECT", "EDGE"):
        try:
            bad |= mi.mask.getPlaneBitMask(plane)
        except Exception:
            continue
    return bad


def _good_point_near_image_center(image) -> "Any":
    """Pick a parent pixel that is unmasked and closest to the image bbox center."""

    mi = image.maskedImage
    wcs = image.getWcs()
    marr = mi.mask.array
    bad = _bad_mask_bits(mi)
    good = (marr & bad) == 0
    bbox = image.getBBox(ImageOrigin.LOCAL)
    cx = bbox.getCenter().getX()
    cy = bbox.getCenter().getY()
    if not np.any(good):
        raise ValueError(f"No unmasked pixels in image")
    ys, xs = np.where(good)
    d2 = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
    j = int(np.argmin(d2))
    p=Point2D(float(xs[j])+image.getX0(), float(ys[j])+image.getY0())
    return wcs.pixelToSky(p)


def _exposure_to_arrays(exposure, dtype: np.dtype):
    """Return image, variance, mask (uint32) arrays from an Exposure."""
    mi = exposure.maskedImage
    data = np.asarray(mi.image.array, dtype=dtype)
    variance = np.asarray(mi.variance.array, dtype=dtype)
    mask = np.asarray(mi.mask.array, dtype=np.uint32)
    return data, variance, mask


def _visit_mjd_mid(exposure, DateTime) -> float:
    """MJD at mid-exposure (absolute)."""
    vi = exposure.getInfo().getVisitInfo()
    start = vi.getDate().get(DateTime.MJD)
    et = vi.getExposureTime()
    return float(start + et * 0.5 / 86400.0)


class ButlerDataModel:
    """Build the same ``stack_inputs`` dict as :class:`ExtractedDataModel`, from Butler queries."""

    MAX_PIX_VALUE = 8000
    MIN_PIX_VALUE = -10000
    VARIANCE_BITMASK = "SAT"


    def __init__(
        self,
        butler: str,
        collections: str | Sequence[str],
        day_obs: int,
        skymap: str,
        tract: int,
        patch: int,
        band: str = 'gri',
        dataset_type: str = "injected_diff_directWarp",
        instrument: str = 'HSC',
        data_dtype: np.dtype = np.float32,
        psf_dataset_type: str = "injected_calexp",
        injected_catalog_dataset_type: str = "injected_calexp_catalog",
    ) -> None:
        """
        Args:
            butler: An open ``lsst.daf.butler.Butler`` instance.
            collections: Collection name(s) passed to the registry query.
            dataset_type: Butler dataset type string (e.g. injected diff warps).
            where: Optional ``registry.queryDatasets`` WHERE string. 
            plants: Injection/plant table; if None, :func:`minimal_plants_table` is used.
            data_dtype: Array dtype for science data and PSFs.
            psf_dataset_type: Calexp dataset used for PSF. The detector is chosen by
                mapping a warp pixel (good and near the warp center) to RA/Dec and
                selecting the calexp whose footprint contains that sky position.
        """
        self._refs = None
        self.butler = Butler(butler, collections=collections)
        self.collections = collections
        self.dataset_type = dataset_type
        self.skymap = skymap
        self.tract = tract
        self.patch = patch
        self.band = band
        self.day_obs = day_obs
        self.instrument = instrument
        self.data_dtype = np.dtype(data_dtype)
        self.psf_dataset_type = psf_dataset_type
        self.injected_catalog_dataset_type = injected_catalog_dataset_type
        self._stack_inputs: dict | None = None
        self._bitmask: dict | None = None
        self._plants: Table | None = None

    @property
    def plants(self) -> Table:
        if self._plants is None:
            self._plants = self._get_injected_source_catalog()
        return self._plants

    @property
    def bitmask(self) -> dict[str, int]:
        if self._bitmask is None:
            self._load_from_butler()
        return self._bitmask

    @property
    def refs(self):
        if self._refs is None:
            self._refs = self._query_refs()
        return self._refs

    def _query_refs(self):
        where = (f"instrument='{self.instrument}' "
                 f"AND day_obs={self.day_obs} "
                 f"AND skymap='{self.skymap}' "
                 f"AND tract={self.tract} "
                 f"AND patch={self.patch} "
                 f"AND band='{self.band}'")
        limit = logging.getLogger().getEffectiveLevel() <= logging.DEBUG and 10 or None
        logging.debug(f"Getting {self.dataset_type} datasets using\n where:{where}\n limit:{limit}")
        refs = sorted(
            self.butler.query_datasets(
                self.dataset_type,
                collections=self.collections,
                where=where,
                limit=limit,
            ),
            key=lambda r: r.dataId["visit"],
        )
        if not refs:
            raise ValueError(f"No datasets of type {self.dataset_type} for collections={self.collections} where={where}")
        return refs

    def _get_psf_at_sky(self, dataId: dict, instrument: str, point: SpherePoint):
        dec = point.getDec().asDegrees()
        ra = point.getRa().asDegrees()
        where=f"instrument='{instrument}' AND visit_detector_region.region OVERLAPS POINT({ra}, {dec})"
        dataset_ref = self.butler.query_datasets('injected_calexp', 
                                            collections='u/NH/coadd',
                                            data_id=dataId,
                                            where=where)
        injected_calexp = self.butler.get(dataset_ref[0])
        wcs = injected_calexp.getWcs()
        p = wcs.skyToPixel(point)
        psf = injected_calexp.getPsf()
        fwhm = psf.computeShape(p).getDeterminantRadius()*installGaussianPsf.FwhmPerSigma
        kernel = psf.computeKernelImage(p).array
        return kernel, fwhm

    def _get_injected_source_catalog(self):
        """Get the injected source catalog from the Butler.
        
        The injected source catalog is a table of sources that were injected into the difference images.
        It is stored in the Butler as a dataset type of ``injected_calexp_catalog``.
        The catalog is stored for each detector in the difference image, and we combine them into a single table.

        There is an error in the source injection code that causes the rate_ra and rate_dec columns to be incorrect
        so we ignore them and compute the rates from the differences x1 and y0 between the first and last visit.

        """
        data_id = {'initial': self.refs[0].dataId,
                   'final': self.refs[-1].dataId}
        # in lsst science pipeline the WCS can have a different x0/y0 compared to the numpy array 
        # the WCS returns x, y set in the x0/y0 reference and we must remove those to be in the 
        # np.array from of the image.
        diff = self.butler.get(self.refs[0])
        x0, y0 = diff.getXY0()
        wcs = diff.getWcs()
        injected_source_catalogs = {}
        for epoch in data_id:
            injected_source_catalogs[epoch] = []
            dataset_refs = self.butler.query_datasets(self.injected_catalog_dataset_type,
                                                      data_id=data_id[epoch])
            for ref in dataset_refs:
                cat = self.butler.get(ref)
                # convert ra/dec of input into x/y locations using the diff WCS
                x, y = wcs.skyToPixelArray(cat['ra'], cat['dec'], degrees=True)
                cat['X0'] = x - x0
                cat['Y0'] = y - y0 
                injected_source_catalogs[epoch].append(cat)
            injected_source_catalogs[epoch] = vstack(injected_source_catalogs[epoch])
        t1 = Time(injected_source_catalogs['initial'].meta['day_obs'])
        t2 = Time(injected_source_catalogs['final'].meta['day_obs'])
        dt = (t2-t1).to('hour').value
        cat = join(injected_source_catalogs['initial'], 
                   injected_source_catalogs['final'], 
                   keys=['injection_id'])
        cat['rate_x'] = (cat['X0_2']-cat['X0_1'])/dt
        cat['rate_y'] = (cat['Y0_2']-cat['Y0_1'])/dt
        cat = cat['injection_id', 'X0_1','Y0_1','rate_x', 'rate_y', 'mag_1']
        logging.debug(f"Full injected source catalog:\n{cat}")
        cat['injection_id'].name = 'id'
        cat['X0_1'].name = 'x0'
        cat['Y0_1'].name = 'y0'
        cat['mag_1'].name = 'mag'
        return cat['id','x0','y0', 'rate_x', 'rate_y', 'mag']

    def _load_from_butler(self) -> dict[str, Any]:
        """Load the data from the butler and return a dictionary of arrays for stacking."""
        refs = self.refs
        datas: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        psfs: list[np.ndarray] = []
        dmjds: list[float] = []
        fwhms: list[float] = []
        im_nums: list[int] = []

        mjd0: float | None = None
        for ref in refs:
            exposure = self.butler.get(ref)
            data, variance, mask = _exposure_to_arrays(exposure, self.data_dtype)
            # get bitmask from exposure mask plane of first exposure
            if self._bitmask is None:
                self._bitmask = _mask_plane_bitmask(exposure.maskedImage.mask)
                logging.debug("Bitmask from exposure mask planes: %s", self._bitmask)

            point = _good_point_near_image_center(exposure)
            visit = int(ref.dataId["visit"])
            instrument = ref.dataId["instrument"]
            logging.debug(
                f"Warp sky location for PSF lookup: {point}"
            )
            psf_arr, fwhm = self._get_psf_at_sky(ref.dataId, instrument, point)
            mjd_mid = _visit_mjd_mid(exposure, DateTime)
            # first exposure mjd is the reference mjd
            if mjd0 is None:
                mjd0 = mjd_mid
            dmjd = mjd_mid - mjd0

            datas.append(data)
            masks.append(mask)
            variances.append(variance)
            psfs.append(psf_arr)
            dmjds.append(float(dmjd))
            fwhms.append(fwhm)
            im_nums.append(visit)

        return {
            "datas": datas,
            "masks": masks,
            "variances": variances,
            "psfs": psfs,
            "dmjds": dmjds,
            "fwhms": fwhms,
            "im_nums": im_nums,
            "plants": self.plants,
            "bitmask": self.bitmask,
        }

    @property
    def stack_inputs(self) -> dict[str, Any]:
        if self._stack_inputs is None:
            self._stack_inputs = self._load_from_butler()
        return self._stack_inputs

    def mask_variance(self, variance_trim: float) -> None:
        """mask variance outliers and remove images that are fully masked"""
        if self._stack_inputs is None:
            _ = self.stack_inputs
        assert self._stack_inputs is not None
        datas = self._stack_inputs["datas"]
        variances = self._stack_inputs["variances"]
        masks = self._stack_inputs["masks"]
        dmjds = self._stack_inputs["dmjds"]
        im_nums = self._stack_inputs["im_nums"]
        bm = self.bitmask
        if self.VARIANCE_BITMASK not in bm:
            logging.warning(
                "No mask plane %r in bitmask %s; skipping variance trim mask bit",
                self.VARIANCE_BITMASK,
                list(bm.keys()),
            )
            mask_bit = np.uint32(0)
        else:
            mask_bit = np.uint32(1) << np.uint32(bm[self.VARIANCE_BITMASK])

        per_visit_keys = (
            "datas",
            "masks",
            "variances",
            "dmjds",
            "psfs",
            "fwhms",
            "im_nums",
        )
        for idx in range(len(datas) - 1, -1, -1):
            w = np.where(
                (np.isinf(variances[idx]))
                | (np.isinf(datas[idx]))
                | (np.isnan(datas[idx]))
                | (datas[idx] > self.MAX_PIX_VALUE)
                | (datas[idx] < self.MIN_PIX_VALUE)
            )
            masks[idx][w] |= mask_bit
            variances[idx][w] = np.nan
            datas[idx][w] = 0.0
            nan_med_variance = np.nanmedian(variances[idx])
            logging.debug("%s %s %s", im_nums[idx], dmjds[idx], nan_med_variance)
            if np.isnan(nan_med_variance):
                logging.debug("Skipping image %s due to nans.", im_nums[idx])
                for key in per_visit_keys:
                    self._stack_inputs[key].pop(idx)
            else:
                w2 = np.where(variances[idx] > variance_trim * nan_med_variance)
                masks[idx][w2] |= mask_bit

    def pack_inputs(self) -> dict[str, Any]:
        """Convert lists of arrays to stacked arrays."""
        if self._stack_inputs is None:
            # this triggers _load_from_butler() if not already loaded
            _ = self.stack_inputs
        assert self._stack_inputs is not None
        for key in ["datas", "variances", "psfs", "dmjds", "fwhms"]:
            self._stack_inputs[key] = np.array(
                self._stack_inputs[key], dtype=self.data_dtype
            )
        self._stack_inputs["masks"] = np.array(
            self._stack_inputs["masks"], dtype=np.uint32
        )
        self._stack_inputs["im_nums"] = np.array(
            self._stack_inputs["im_nums"], dtype=np.int32
        )
        return self._stack_inputs
