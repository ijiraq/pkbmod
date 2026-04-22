"""Load shift-and-stack inputs from an LSST Gen3 Butler repository.

Requires the LSST Science Pipelines in the active Python environment. 

Typical dataset types for warped difference images include
``injected_diff_directWarp``. PSFs come from ``injected_calexp`` (see
``psf_dataset_type``): a good pixel near the warp center is converted to
RA/Dec, then the per-visit calexp whose detector footprint contains that sky
position is selected to provide the PSF.
"""
import astropy.units as u
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

from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
from astropy.wcs.utils import proj_plane_pixel_area

logger = logging.getLogger(__name__)

def load(
    butler: str | Butler,
    collections: str | Sequence[str],
    day_obs: int,
    skymap: str,
    tract: int,
    patch: int,
    band: str,
    instrument: str,
    dataset_type: str,
    data_dtype: np.dtype,
    psf_dataset_type: str,
    variance_trim: float,
    injected_catalog_dataset_type: str = "injected_calexp_catalog",
) -> dict[str, Any]:
    """Load from Butler, apply variance mask, pack arrays.

    Returns ``(stack_inputs, n_refs, nbytes_heap_estimate)``. No output paths —
    callers (e.g. :mod:`cli`) own filenames and formats.
    """
    import gc

    dm = DataModel(
        butler=butler,
        collections=collections,
        day_obs=day_obs,
        skymap=skymap,
        tract=tract,
        patch=patch,
        band=band,
        instrument=instrument,
        dataset_type=dataset_type,
        data_dtype=data_dtype,
        psf_dataset_type=psf_dataset_type,
        injected_catalog_dataset_type=injected_catalog_dataset_type,
    )
    dm.mask_variance(variance_trim)
    dm.pack_inputs()
    stack_inputs = dm.stack_inputs
    del dm
    gc.collect()
    return stack_inputs, n_refs, nbytes



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


def _skywcs_metadata_to_astropy_wcs(md: Any) -> "AstropyWCS | None":
    """Best-effort FITS header from LSST ``getFitsMetadata()`` for Astropy."""
    try:
        hdr = fits.Header()
        try:
            names = list(md.names())
        except Exception:
            names = list(md.paramNames(False))  # type: ignore[attr-defined]
        for name in names:
            hdr[name] = md.get(name)
        return AstropyWCS(hdr)
    except Exception as exc:
        logger.warning("Could not build Astropy WCS from LSST metadata: %s", exc)
        return None


def _detection_frame_from_exposure(
    exposure: Any, reference_data_id: Mapping[str, Any], dataset_type: str
) -> dict[str, Any]:
    """Frame for converting numpy detection x/y to sky (matches plant pixel convention)."""
    skywcs = exposure.getWcs()
    x0 = float(exposure.getX0())
    y0 = float(exposure.getY0())
    md = skywcs.getFitsMetadata(precise=False)
    awcs = _skywcs_metadata_to_astropy_wcs(md)
    # sqrt(area) is deg/pixel; sns_rates expects arcsec/pixel for rate_lims in "/hour.
    pixel_scale = float(np.sqrt(proj_plane_pixel_area(awcs))) * 3600.0
    ref = {str(k): _json_safe_data_id(v) for k, v in reference_data_id.items()}
    out: dict[str, Any] = {
        "source": "lsst_butler",
        "dataset_type": dataset_type,
        "reference_data_id": ref,
        "parent_origin_xy": (x0, y0),
        "lsst_sky_wcs": skywcs,
        "wcs_astropy": awcs,
        "pixel_scale": pixel_scale,
    }
    if awcs is not None:
        out["fits_header_text"] = str(awcs.to_header())
    return out


def _json_safe_data_id(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _visit_mjd_mid(exposure, DateTime) -> float:
    """MJD at mid-exposure (absolute)."""
    vi = exposure.getInfo().getVisitInfo()
    start = vi.getDate().get(DateTime.MJD)
    et = vi.getExposureTime()
    return float(start + et * 0.5 / 86400.0)


class DataModel:
    """Build the same ``stack_inputs`` dict as :class:`ExtractedDataModel`, from Butler queries.

    Plant ``rate_x`` / ``rate_y`` are pixels per day (see :attr:`plants`).
    """

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
        data_dtype:int = 32,
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
        self.data_dtype = np.float16 if data_dtype == 16 else np.float32
        self.data_dtype = np.dtype(self.data_dtype)
        self.psf_dataset_type = psf_dataset_type
        self.injected_catalog_dataset_type = injected_catalog_dataset_type
        self._stack_inputs: dict | None = None
        self._bitmask: dict | None = None
        logger.info(
            "Stacking patches from LSST Data Butler using dimensions:\n"
            f"day_obs:{self.day_obs}\n"
            f"band: {self.band}\n"
            f"skymap: {self.skymap}\n"
            f"tract: {self.tract}\n"
            f"patch: {self.patch}\n"
        )

    @property
    def data_id(self) -> dict:
        """a disctionary defining the resulting dataset"""
        return dict(day_obs=self.day_obs,
                    band=self.band,
                    skymap=self.skymap,
                    tract=self.tract,
                    patch=self.patch)

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
        limit = None
        # limit = 10 if logger.isEnabledFor(logging.DEBUG) else None
        logger.info(
            "Getting %s datasets using\n where:%s\n limit:%s",
            self.dataset_type,
            where,
            limit,
        )
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
        logger.info(f"Loading {len(refs)} datasets from {self.butler}")
        detection_frame: dict[str, Any] | None = None
        for ref in refs:
            exposure = self.butler.get(ref)
            if detection_frame is None:
                detection_frame = _detection_frame_from_exposure(
                    exposure, ref.dataId.required, self.dataset_type
                )
            data, variance, mask = _exposure_to_arrays(exposure, self.data_dtype)
            # get bitmask from exposure mask plane of first exposure
            if self._bitmask is None:
                self._bitmask = _mask_plane_bitmask(exposure.maskedImage.mask)
                logger.debug("Bitmask from exposure mask planes: %s", self._bitmask)

            point = _good_point_near_image_center(exposure)
            visit = int(ref.dataId["visit"])
            instrument = ref.dataId["instrument"]
            logger.debug(
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
            "bitmask": self.bitmask,
            "detection_frame": detection_frame,
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
            logger.debug("%s %s %s", im_nums[idx], dmjds[idx], nan_med_variance)
            if np.isnan(nan_med_variance):
                logger.debug("Skipping image %s due to nans.", im_nums[idx])
                if idx == 0:
                    logger.warning(
                        "Removing the first stacked image (visit %s); "
                        "detection_frame WCS still refers to the original "
                        "reference warp — sky coordinates may be inconsistent.",
                        im_nums[idx],
                    )
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

