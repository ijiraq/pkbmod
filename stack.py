from astropy.wcs import WCS
from astropy.table import Table
from calc_ecliptic_angle import calc_ecliptic_angle
from dataclasses import dataclass, field, fields, asdict
import gc
import logging
import numpy as np
import sns_data_nh as data
import sns_utils as utils
from sns_rates import get_shift_rates_from_angle
from torch.nn import functional
import torch
from typing import Any, Mapping, List

logger = logging.getLogger(__name__)

EXTENSION_WITH_WCS = 1
VARIANCE_MASK = "VARIANCE"

@dataclass
class StackParams(object):
    
    badflags: tuple[str, ...] = field(default=("BAD","BRIGHT_OBJECT","INTRP",
                                               "NO_DATA","SAT","STREAK","UNMASKEDNAN"),
                                       metadata=dict(help="List of mask bitflag names to flag as bad."))
    clust_dist_lim: float = field(default=5.0, metadata=dict(help="maximum distance between candidate and linear motion"))
    kernel_width: int = field(default=14, metadata=dict(help="width of kernel in pixels"))
    min_samp: int = field(default=3, metadata=dict(help="minimum number of clustered detections required"))
    min_snr: float = field(default=4.5, metadata=dict(help="Minimum SNR for a detection"))
    n_keep: int = field(default=12, metadata=dict(help="number of sources to keep after initial serach"))
    peak_offset_max: float = field(default=4, metadata=dict(help="max distance between peak and centre of stamp"))
    rate_fwhm_grid_step: float = field(default=0.75, metadata=dict(help="width of steps in units of FWHM"))
    rate_min: float = field(default=0.5, metadata=dict(help="Slowest rate (''/hour) to search"))
    rate_max: float = field(default=5.5, metadata=dict(help="Fastest rate (''/hour) to search"))
    angle_width: float = field(default=45.0, metadata=dict(help="Opening angle of search grid, relative to ecliptic"))
    trim_snr: float = field(default=5.5, metadata=dict(help="min SNR of sources to keep after clustering"))
    use_gaussian_kernel: bool = field(default=False, metadata=dict(help="use a guassian kernel instead of a PSF"))
    skip_use_negative_well: bool = field(default=False, metadata=dict(help="skip use of the negative well for detection"))
    variance_trim: float = field(default=1.3, metadata=dict(help="factor above median variance to mask pixels"))

    def __iter__(self) -> dict:
        yield from asdict(self).items()

    @classmethod
    def from_dict(cls, data):
        field_names = {f.name for f in fields(cls)}
        valid_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**valid_data)



@dataclass
class StackRunResult:
    """In-memory outputs from :func:`run`. Persist paths and formats in the caller (e.g. :mod:`cli`).

    This dataclass holds all the results from a stack run and passes them back to the caller who then
    writes the data to disk or what ever is wanted.  This seperates the process of running the
    stacking code from reading and writing the files.

    For the format of passing the data into stack.run look at the '*_data_model.py' files which
    are responsible for getting data from storage.

    """

    detections: Table
    wcs: WCS
    stamps: np.ndarray
    params: dict



def run(
    stack_inputs: dict,
    stack_params: dict,
    *,
    low_mem: bool = False,
    low_mem_tile_w: int = 256,
    dtype=np.float16,
) -> StackRunResult:
    """Run shift-and-stack search. Rates use ecliptic-angle based grid instead of
    a list of injected sources.  This runner does not know about injected sources.

    To match with injected sources use a different method to load the detection
    and injection lists and compare them.
    
    Returns :class:`StackRunResult` only; no filesystem I/O.
    """
    datas = stack_inputs["datas"]
    masks = stack_inputs["masks"]
    variances = stack_inputs["variances"]
    dmjds = stack_inputs["dmjds"]
    fwhms = stack_inputs["fwhms"]
    im_nums = stack_inputs["im_nums"]
    psfs = stack_inputs["psfs"]
    bitmask = stack_inputs["bitmask"]
    detection_frame = stack_inputs.get("detection_frame")

    badflags = stack_params["badflags"]
    rate_fwhm_grid_step = stack_params["rate_fwhm_grid_step"]
    n_keep = stack_params["n_keep"]
    kernel_width = stack_params["kernel_width"]
    khw = kernel_width // 2
    use_gaussian_kernel = stack_params["use_gaussian_kernel"]
    use_negative_well = not stack_params["skip_use_negative_well"]
    peak_offset_max = stack_params["peak_offset_max"]
    dist_lim = stack_params["dist_lim"]
    min_samp = stack_params["min_samp"]
    min_snr = stack_params["min_snr"]
    trim_snr = stack_params["trim_snr"]
    rate_max = stack_params["rate_max"]
    rate_min = stack_params["rate_min"]
    angle_width = stack_params["angle_width"]

    dtype = np.dtype(dtype)
    if dtype == np.float16:
        torch_dtype = torch.float16
    elif dtype == np.float32:
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype {dtype}; use np.float16 or np.float32")

    wcs_astropy = None
    if detection_frame is not None:
        wcs_astropy = detection_frame.get("wcs_astropy")
        pixel_scale = detection_frame.get("pixel_scale")
    if wcs_astropy is None or pixel_scale is None:
        raise ValueError(
            "stack_inputs['detection_frame'] with key 'wcs_astropy' is required "
            "for ecliptic rate grid (no planted-source rates)."
        )

    fwhm = np.median(fwhms)
    rate_lims = [rate_min/pixel_scale, rate_max/pixel_scale]
    ecl_ang = calc_ecliptic_angle(wcs_astropy, 1000, 1000)
    rates = get_shift_rates_from_angle(
        ecl_ang,
        dmjds,
        rate_lims,
        [-angle_width, angle_width],
        fwhm,
        rate_fwhm_grid_step,
    )

    logger.debug(
        "Creating the convolution kernel: Use Guassian:%s", use_gaussian_kernel
    )
    kernel = data.create_kernel(
        psfs=psfs,
        dmjds=dmjds,
        rates=rates,
        useNegativeWell=use_negative_well,
        useGaussianKernel=use_gaussian_kernel,
        kernel_width=kernel_width,
        im_nums=im_nums,
        dtype=dtype,
    )

    np_datas = np.expand_dims(np.expand_dims(np.asarray(datas, dtype=dtype), 0), 0)
    np_variances = np.asarray(variances, dtype=dtype)
    np.reciprocal(np_variances, out=np_variances, where=np_variances != 0)
    np_inv_variances = np.expand_dims(np.expand_dims(np_variances, 0), 0)
    del np_variances
    np_masks = np.expand_dims(
        np.expand_dims(np.asarray(masks, dtype=np.uint32), 0), 0
    )

    logging.debug("Masking the np arrays that will be used for stacking.")
    badflags = np.array([2 ** bitmask[flag] for flag in badflags]).sum()
    bad_pixels = (np_masks & badflags) != 0
    bad_pixels |= ~np.isfinite(np_datas)
    np.copyto(np_datas, 0.0, where=bad_pixels)
    np.copyto(np_inv_variances, 0.0, where=bad_pixels)
    np_masks[bad_pixels] = np.uint32(0)
    del bad_pixels
    np.clip(np_masks, 0, 1, out=np_masks)
    np_masks = np_masks.astype(np.uint8, copy=False)

    post_torch_dtype = torch.float16
    logger.debug(
        "Dtypes: initial_shift=%s, post_shift=%s, low_mem_tile_w=%s",
        torch_dtype,
        post_torch_dtype,
        low_mem_tile_w,
    )

    device = data.get_device()
    logger.debug("Loading data onto %s", device)
    datas_t = torch.as_tensor(np_datas, dtype=torch_dtype, device=device)
    inv_variances = torch.as_tensor(np_inv_variances, dtype=torch_dtype, device=device)
    n_im = int(datas_t.size()[2])

    _ = torch.rot90(kernel, k=2, dims=(3, 4))

    logger.info("Convolving %d images and variances with kernel", n_im)
    for ir in range(n_im):
        datas_t[0, 0, ir, :, :] = torch.conv2d(
            datas_t[:, :, ir, :, :] * inv_variances[:, :, ir, :, :],
            kernel[:, :, ir, :, :],
            padding="same",
        )
        inv_variances[0, 0, ir, :, :] = torch.conv2d(
            inv_variances[:, :, ir, :, :],
            kernel[:, :, ir, :, :] * kernel[:, :, ir, :, :],
            padding="same",
        )

    if n_keep > len(rates):
        logger.warning(
            "Number of stack rate: %d is smaller than request n_keep %d. "
            "Only keeping %d detections per pixel",
            len(rates),
            n_keep,
            len(rates),
        )
        n_keep = min(n_keep, len(rates))

    logger.info("Using low-memory initial shift-and-stack stage")
    top_snr, top_alpha, top_rate_idx = utils.run_shifts_topk(
        datas=datas_t,
        inv_variances=inv_variances,
        rates=rates,
        dmjds=dmjds,
        min_snr=min_snr,
        n_keep=n_keep,
        tile_w=low_mem_tile_w,
        work_dtype=torch_dtype,
        output_dtype=torch_dtype,
    )
    detections = utils.topk_to_detections(
        top_snr=top_snr,
        top_alpha=top_alpha,
        top_rate_idx=top_rate_idx,
        rates=rates,
        dtype=dtype,
    )
    del top_snr, top_alpha, top_rate_idx
    gc.collect()

    del datas_t
    del inv_variances
    gc.collect()
    torch.cuda.empty_cache()
    detections = utils.trim_negative_flux(detections)
    detections_idx = np.arange(len(detections))

    logger.debug("Post-shift tensor dtype: %s", post_torch_dtype)
    im_datas = functional.pad(
        torch.as_tensor(np_datas, dtype=post_torch_dtype, device=device),
        (khw, khw, khw, khw),
    )
    del np_datas
    gc.collect()
    inv_vars = functional.pad(
        torch.as_tensor(
            np.asarray(0.5, dtype=dtype) * np_inv_variances,
            dtype=post_torch_dtype,
            device=device,
        ),
        (khw, khw, khw, khw),
    )
    del np_inv_variances
    gc.collect()

    c = torch.zeros_like(im_datas)
    c[0, 0, 0] = im_datas[0, 0, 0]
    cv = torch.zeros_like(im_datas)
    cv[0, 0, 0] = inv_vars[0, 0, 0]

    keeps = utils.brightness_filter_fast(
        im_datas,
        inv_vars,
        c,
        cv,
        kernel,
        dmjds,
        rates,
        detections,
        khw,
        n_im,
        n_bright_test=10,
        test_high=1.15,
        test_low=0.85,
        word_dtype=post_torch_dtype,
    )

    logger.info("Number of detections: %d", len(detections))
    logger.info("Number kept: %d", len(keeps))
    filt_detections_idx = detections_idx[keeps]
    del keeps

    im_masks = functional.pad(
        torch.as_tensor(np_masks, dtype=post_torch_dtype, device=device),
        (khw, khw, khw, khw),
    )
    del np_masks

    mean_stamps = utils.create_stamps(
        im_datas,
        im_masks,
        c,
        cv,
        dmjds,
        rates,
        detections[filt_detections_idx],
        khw,
    )
    del im_masks
    gc.collect()
    torch.cuda.empty_cache()

    stamps = mean_stamps
    peak_keep = utils.peak_offset_filter(
        stamps, detections[filt_detections_idx], peak_offset_max
    )
    stamps = stamps[peak_keep]
    filt_detections_idx = filt_detections_idx[peak_keep]

    clust_filt_idx = utils.predictive_line_cluster_indices(
        detections[filt_detections_idx],
        dmjds,
        dist_lim,
        min_samp,
        init_select_proc_distance=60,
    )
    gc.collect()

    logger.info(
        "Number of sources kept after brightness and peak location filtering: %d.",
        len(clust_filt_idx),
    )

    clust_master_idx = filt_detections_idx[clust_filt_idx]
    snr_trim = np.where(detections[clust_master_idx, 5] >= trim_snr)[0]
    clust_filt_idx = clust_filt_idx[snr_trim]
    clust_master_idx = clust_master_idx[snr_trim]
    logger.info(
        "Number of sources kept after final SNR trim: %d.", len(clust_master_idx)
    )

    final_detection_indices = clust_master_idx
    final_stamps = stamps[clust_filt_idx]

    logger.info("Number of candidates %d", len(final_detection_indices))

    det_shift_idx = detections_idx
    det_filt_idx = filt_detections_idx
    det_gird_idx = clust_master_idx
    det_final_idx = np.copy(final_detection_indices)
    
    det_lvl = np.array(['shift']*len(det_shift_idx))
    det_lvl[det_filt_idx] = ['peak']
    det_lvl[det_gird_idx] = ['clust']
    det_lvl[det_final_idx] = ['line']

    order = np.argsort(detections[final_detection_indices, 5])[::-1]
    final_stamps_idx  = final_detection_indices[order]
    final_stamps = final_stamps[order]
    

    # represent that detection list as an astropy Table object
    names = ['idx','x','y',
             'flux','snr','rate_x','rate_y',
             'ra_deg','dec_deg','det_lvl', 'stamp_idx']
    coords = wcs_astropy.pixel_to_world(
        detections[det_shift_idx, 0],
        detections[det_shift_idx, 1],
    )
    idx = det_shift_idx
    stamp_idx=np.zeros(len(idx))-1
    stamp_idx[final_stamps_idx] = range(len(final_stamps_idx))
    table_cols = [
        idx,
        detections[idx, 0],
        detections[idx, 1],
        detections[idx, 4],
        detections[idx, 5],
        rates[np.round(detections[idx, 2]).astype('int'),0],
        rates[np.round(detections[idx, 2]).astype('int'),1],
        coords.ra.degree,
        coords.dec.degree,
        det_lvl[idx],
        stamp_idx,
    ]
    detections_table = Table(table_cols, names=names)
            
    return StackRunResult(
        detections=detections_table,
        wcs=detection_frame['wcs_astropy'],
        stamps=final_stamps,
        params=stack_params,
    )
