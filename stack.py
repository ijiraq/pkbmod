import gc
import logging
import numpy as np
from astropy.table import Table, vstack
from pathlib import Path
from torch.nn import functional
import torch

import sns_data_nh as data
import sns_utils as utils

logger = logging.getLogger(__name__)

EXTENSION_WITH_WCS = 1
VARIANCE_MASK = 'VARIANCE'


def _detection_rates(detections: np.ndarray,
                     rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y rates of motion based on index stored in a 
    detection table."""
    rate_idx = np.round(detections[:, 2]).astype("int")
    return rates[rate_idx, 0], rates[rate_idx, 1]


def match_detections_to_plants(plants: Table,
                               detections: np.ndarray,
                               rates: np.ndarray,
                               dist_max: float,
                               dist_rate_max: float,
                               detection_type: str) -> Table:
    """Create a join table between plants and matched detections."""
    columns = {
        'plant_index': [],
        'detection_index': [],
        'detection_type': [],
        'dist_r': [],
        'dist_v': [],
        'plant_x0': [],
        'plant_y0': [],
        'plant_rate_x': [],
        'plant_rate_y': [],
        'det_x': [],
        'det_y': [],
        'det_rate_x': [],
        'det_rate_y': [],
        'det_flux': [],
        'det_snr': [],
    }

    if len(detections) == 0 or len(plants) == 0:
        return Table(columns)

    det_rx, det_ry = _detection_rates(detections, rates)
    det_x = detections[:, 0]
    det_y = detections[:, 1]

    for idx in range(len(plants)):
        plant_index = plants['plant_id'][idx]
        dist_sq = ((plants['x0'][idx] - det_x)**2 +
                   (plants['y0'][idx] - det_y)**2)
        dist_rate_sq = ((plants['rate_x'][idx] - det_rx)**2 +
                        (plants['rate_y'][idx] - det_ry)**2)
        matched = np.where((dist_sq < dist_max**2) &
                           (dist_rate_sq < dist_rate_max**2))[0]
        for detection_index in matched:
            columns['plant_index'].append(plant_index)
            columns['detection_index'].append(int(detection_index))
            columns['detection_type'].append(detection_type)
            columns['dist_r'].append(float(np.sqrt(dist_sq[detection_index])))
            columns['dist_v'].append(
                float(np.sqrt(dist_rate_sq[detection_index])))
            columns['plant_x0'].append(float(plants['x0'][idx]))
            columns['plant_y0'].append(float(plants['y0'][idx]))
            columns['plant_rate_x'].append(float(plants['rate_x'][idx]))
            columns['plant_rate_y'].append(float(plants['rate_y'][idx]))
            columns['det_x'].append(float(det_x[detection_index]))
            columns['det_y'].append(float(det_y[detection_index]))
            columns['det_rate_x'].append(float(det_rx[detection_index]))
            columns['det_rate_y'].append(float(det_ry[detection_index]))
            columns['det_flux'].append(float(detections[detection_index, 4]))
            columns['det_snr'].append(float(detections[detection_index, 5]))

    return Table(columns)


def summarize_plant_matches(plants: Table,
                            detection_types: dict[str, np.ndarray],
                            rates: np.ndarray,
                            dist_max: float,
                            dist_rate_max: float) -> tuple[Table, Table]:
    """Annotate plants with per-stage match flags and return a join table."""
    for column in ["min_dist_r", "min_dist_v"]:
        plants[column] = np.nan
    for column in detection_types:
        plants[column] = 0
    plants['num_match'] = 0

    match_tables = []
    final_detection_type = next(reversed(detection_types))
    final_matches = None

    for detection_type, detections in detection_types.items():
        match_table = match_detections_to_plants(
            plants=plants,
            detections=detections,
            rates=rates,
            dist_max=dist_max,
            dist_rate_max=dist_rate_max,
            detection_type=detection_type)
        if len(match_table) > 0:
            match_tables.append(match_table)
            matched_plants = np.unique(match_table['plant_index'])
        else:
            matched_plants = np.array([], dtype=int)
        plants[detection_type][:] = 0
        if len(matched_plants) > 0:
            plants[detection_type][matched_plants] = 1
        if detection_type == final_detection_type:
            final_matches = match_table

    if final_matches is None:
        final_matches = Table({
            'plant_index': [],
            'detection_index': [],
            'detection_type': [],
            'dist_r': [],
            'dist_v': [],
            'plant_x0': [],
            'plant_y0': [],
            'plant_rate_x': [],
            'plant_rate_y': [],
            'det_x': [],
            'det_y': [],
            'det_rate_x': [],
            'det_rate_y': [],
            'det_flux': [],
            'det_snr': [],
        })

    final_detections = detection_types[final_detection_type]
    if len(final_detections) > 0:
        final_rx, final_ry = _detection_rates(final_detections, rates)
        for idx in range(len(plants)):
            plant_index = plants['plant_id'][idx]
            dist_sq = ((plants['x0'][idx] - final_detections[:, 0])**2 +
                       (plants['y0'][idx] - final_detections[:, 1])**2)
            dist_rate_sq = ((plants['rate_x'][idx] - final_rx)**2 +
                            (plants['rate_y'][idx] - final_ry)**2)
            plants['min_dist_r'][idx] = np.min(dist_sq)**0.5
            plants['min_dist_v'][idx] = np.min(dist_rate_sq)**0.5
            if len(final_matches) > 0:
                matched = final_matches['plant_index'] == plant_index
                plants['num_match'][idx] = int(np.sum(matched))
    all_matches = vstack(match_tables, metadata_conflicts='silent') if match_tables else final_matches.copy()
    return plants, all_matches


def run(stack_inputs: dict, stack_params: dict,
        results_filename: str, plant_matches_filename: str,
        low_mem: bool = False, low_mem_tile_w: int = 256,
        dtype=np.float16):
    """Given the data load and stacking parameters run the shift-and-stack
    search.

    Args:
        stack_inputs (dict): dictionary of np.arrays for stacking: datas, etc.
        stack_params (dict): parameters to use for the sns search
    """
    # now map to the array variables used in rest of code
    datas = stack_inputs['datas']
    masks = stack_inputs['masks']
    variances = stack_inputs['variances']
    dmjds = stack_inputs['dmjds']
    fwhms = stack_inputs['fwhms']
    im_nums = stack_inputs['im_nums']
    psfs = stack_inputs['psfs']
    plants = stack_inputs['plants']
    results_filename = results_filename
    bitmask = stack_inputs['bitmask']

    # now define the parameters based on stackparams dictionary.
    badflags = stack_params['badflags']
    rate_fwhm_grid_step = stack_params['rate_fwhm_grid_step']
    n_keep = stack_params['n_keep']
    kernel_width = stack_params['kernel_width']
    khw = kernel_width//2
    use_gaussian_kernel = stack_params['use_gaussian_kernel']
    use_negative_well = stack_params['use_negative_well']
    peak_offset_max = stack_params['peak_offset_max']
    dist_lim = stack_params['dist_lim']
    min_samp = stack_params['min_samp']
    min_snr = stack_params['min_snr']
    trim_snr = stack_params['trim_snr']
    dist_max = stack_params['dist_max']
    dist_rate_max = stack_params['dist_rate_max']
    dtype = np.dtype(dtype)
    if dtype == np.float16:
        torch_dtype = torch.float16
    elif dtype == np.float32:
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype {dtype}; use np.float16 or np.float32")

    rates = data.get_shift_rates(
        plants=plants,
        fwhms=fwhms,
        dmjds=dmjds,
        rate_fwhm_grid_step=rate_fwhm_grid_step)

    logger.debug(("Creating the convolution kernel:"
                   f" Use Guassian:{use_gaussian_kernel}"))
    kernel = data.create_kernel(
        psfs=psfs,
        dmjds=dmjds,
        rates=rates,
        useNegativeWell=use_negative_well,
        useGaussianKernel=use_gaussian_kernel,
        kernel_width=kernel_width,
        im_nums=im_nums,
        dtype=dtype)

    (A, B) = datas[0].shape

    np_datas = np.expand_dims(np.expand_dims(
        np.asarray(datas, dtype=dtype), 0), 0)
    np_variances = np.asarray(variances, dtype=dtype)
    np.reciprocal(np_variances, out=np_variances, where=np_variances != 0)
    np_inv_variances = np.expand_dims(np.expand_dims(np_variances, 0), 0)
    del np_variances
    np_masks = np.expand_dims(np.expand_dims(
        np.asarray(masks, dtype=np.uint32), 0), 0)

    # (np_masks & badflags) == 0 is FALSE when masks matches a badflag value
    # ~((np_masks & badflags) == 0) is TRUE when mask matches a badflag value
    # ~((np_masks & badflags) == 0) | np.isnan(datas)
    # is TRUE when a mask matches a badflag or is nans

    # Set masked/nan pixels values to 0 to ignore in shift-and-stack
    # below is original line based on logic in comment above
    # which appears to be wrong?
    # w = np.where(~((np_masks & badmask) == 0) | np.isnan(datas))
    logging.debug("Masking the np arrays that will be used for stacking.")
    badflags = np.array([2**bitmask[flag] for flag in badflags]).sum()
    bad_pixels = (np_masks & badflags) != 0
    bad_pixels |= ~np.isfinite(np_datas)
    np.copyto(np_datas, 0.0, where=bad_pixels)
    np.copyto(np_inv_variances, 0.0, where=bad_pixels)
    np_masks[bad_pixels] = np.uint32(0)
    del bad_pixels
    # masks with 1 are GOOD pixels, 0 are BAD pixels
    np.clip(np_masks, 0, 1, out=np_masks)
    np_masks = np_masks.astype(np.uint8, copy=False)

    # using logical & value of mask > 0 if mask holds value in bits
    # w = (np_masks & badvalue > 0) | np.isnan(datas)
    # np_datas[w] = 0.0
    # np_inv_variances[w] = 0.0
    # for shift-and-stack routines masks with 1 are GOOD pixels
    # np_masks[w] = 0
    # np_masks = np.clip(np_masks, 0, 1)

    # Always use the low-memory shift-and-stack path. Keep the post-shift
    # stages in fp16 as well to reduce resident GPU memory.
    post_torch_dtype = torch.float16
    logger.debug(("Dtypes: initial_shift=%s, post_shift=%s, low_mem_tile_w=%s"),
                  torch_dtype, post_torch_dtype, low_mem_tile_w)

    # set device value based on gpu availability.
    device = data.get_device()
    # push the data to the device for tourch shift-and-stack
    logger.debug(f"Loading data onto {device}")
    datas = torch.as_tensor(np_datas, dtype=torch_dtype, device=device)
    inv_variances = torch.as_tensor(np_inv_variances,
                                    dtype=torch_dtype, device=device)
    n_im = int(datas.size()[2])

    _ = torch.rot90(kernel, k=2, dims=(3, 4))

    # convolve pixels and variances with the kernels.
    logger.info(f"Convolving {n_im} images and variances with kernel")
    for ir in range(n_im):
        datas[0, 0, ir, :, :] = torch.conv2d(
            datas[:, :, ir, :, :]*inv_variances[:, :, ir, :, :],
            kernel[:, :, ir, :, :], padding='same')
        inv_variances[0, 0, ir, :, :] = torch.conv2d(
            inv_variances[:, :, ir, :, :],
            kernel[:, :, ir, :, :]*kernel[:, :, ir, :, :], padding='same')

    if n_keep > len(rates):
        logger.warning((f"Number of stack rate: {len(rates)}"
                      f"is smaller than request n_keep {n_keep}. "
                      f"Only keeping {len(rates)} detections per pixel"))
        n_keep = min(n_keep, len(rates))

    logger.info("Using low-memory initial shift-and-stack stage")
    logger.debug("run_shifts_topk dtype: work=%s output=%s",
                  torch_dtype, torch_dtype)
    top_snr, top_alpha, top_rate_idx = utils.run_shifts_topk(
        datas=datas,
        inv_variances=inv_variances,
        rates=rates,
        dmjds=dmjds,
        min_snr=min_snr,
        n_keep=n_keep,
        tile_w=low_mem_tile_w,
        work_dtype=torch_dtype,
        output_dtype=torch_dtype)
    detections = utils.topk_to_detections(
        top_snr=top_snr,
        top_alpha=top_alpha,
        top_rate_idx=top_rate_idx,
        rates=rates,
        dtype=dtype)
    del top_snr, top_alpha, top_rate_idx
    gc.collect()

    del datas
    del inv_variances
    gc.collect()
    torch.cuda.empty_cache()
    # trim the flux negative sources
    detections = utils.trim_negative_flux(detections)

    # now apply the brightness filter.
    # Check n_bright_test values between test_low and
    # test_high fraction of the estimated value
    # pad the data and variance arrays
    logger.debug("Post-shift tensor dtype: %s", post_torch_dtype)
    logger.debug(f"Creating im_datas with shape {np_datas.shape}")
    im_datas = functional.pad(torch.as_tensor(np_datas,
                                              dtype=post_torch_dtype,
                                              device=device),
                              (khw, khw, khw, khw))
    del np_datas  # I don't think this is used again.
    gc.collect()
    logger.debug(f"Creating inv_vars with shape {np_inv_variances.shape}")
    inv_vars = functional.pad(
        torch.as_tensor(
            np.asarray(0.5, dtype=dtype) * np_inv_variances,
            dtype=post_torch_dtype,
            device=device), (khw, khw, khw, khw))
    del np_inv_variances
    gc.collect()

    c = torch.zeros_like(im_datas)
    c[0, 0, 0] = im_datas[0, 0, 0]
    cv = torch.zeros_like(im_datas)
    cv[0, 0, 0] = inv_vars[0, 0, 0]

    keeps = utils.brightness_filter_fast(im_datas, inv_vars, c, cv, kernel,
                                         dmjds, rates, detections, khw, n_im,
                                         n_bright_test=10,
                                         test_high=1.15,
                                         test_low=0.85,
                                         word_dtype=post_torch_dtype)

    logger.info(f"Number of detections: {len(detections)}")
    logger.info(f"Number kept: {len(keeps)}")
    filt_detections = np.copy(detections[keeps])
    del keeps

    im_masks = functional.pad(
        torch.as_tensor(np_masks, dtype=post_torch_dtype, device=device),
        (khw, khw, khw, khw))
    del np_masks

    # create the stamps
    mean_stamps = utils.create_stamps(im_datas, im_masks,
                                      c, cv, dmjds, rates,
                                      filt_detections, khw)
    del im_masks
    gc.collect()
    torch.cuda.empty_cache()

    stamps = mean_stamps
    # trim the candidates with peak offset more than peak_offset_max pixels
    stamps, filt_detections = utils.peak_offset_filter(stamps,
                                                       filt_detections,
                                                       peak_offset_max)

    save_filt_detections = False
    if save_filt_detections:
        with open('filt_detections.npy', 'wb') as han:
            np.save(han, filt_detections)

    # apply predictive clustering
    clust_detections, clust_stamps = utils.predictive_line_cluster(
        filt_detections, stamps, dmjds, dist_lim, min_samp,
        init_select_proc_distance=60)
    del stamps
    gc.collect()

    n_det = len(clust_detections)
    logger.info(("Number of sources kept after "
                  f"brightness and peak location filtering: {n_det}."))

    w = np.where(clust_detections[:, 5] >= trim_snr)
    clust_detections = clust_detections[w]
    clust_stamps = clust_stamps[w]
    n_det = len(clust_detections)
    logger.info(("Number of sources kept after "
                  f"final SNR trim: {n_det}."))

    clust_detection_matches = match_detections_to_plants(
        plants=plants,
        detections=clust_detections,
        rates=rates,
        dist_max=dist_max,
        dist_rate_max=dist_rate_max,
        detection_type='det_clust')
    logger.info("Clustered detection/plant matches before position filter: %d",
                 len(clust_detection_matches))
    debug_detection_indices = None
    debug_output_dir = None
    if len(clust_detection_matches) > 0:
        debug_detection_indices = np.unique(
            np.asarray(clust_detection_matches['detection_index'], dtype=int))
        debug_output_dir = (
            Path(plant_matches_filename).with_suffix('').parent /
            f"{Path(plant_matches_filename).with_suffix('').name}.position_filter_debug"
        )

    cv[0, 0, 0] = inv_vars[0, 0, 0]
    
    if False:
        # Just skip the position filter for now

        grid_detections, grid_stamps = utils.position_filter(
            clust_detections, clust_stamps, im_datas, inv_vars,
            c, cv, kernel, dmjds, rates, khw, n_offsets=11,
            debug_detection_indices=debug_detection_indices,
            debug_output_dir=debug_output_dir)

        w = np.where(grid_detections[:, 5] >= trim_snr)
        final_stamps = grid_stamps[w]
        final_detections = grid_detections[w]
        del grid_detections, grid_stamps
    else:
        w = np.where(clust_detections[:, 5] >= trim_snr)
        final_detections = clust_detections[w]
        final_stamps = None
    n_det = len(final_detections)
    # clust_stamps = clust_stamps[w]
    logger.info(f'Number of candidates {n_det}')
    # remove these memory cleanups as they aren't needed at this point
    # del im_datas, inv_vars, c, cv, kernel
    # gc.collect()
    # torch.cuda.empty_cache()

    # columns to add to the plant table to track matched detections
    detection_types = {'det_shift': detections,
                       'det_filt': filt_detections,
                       'det_gird': clust_detections,
                       'det_final': final_detections}
    plants, detection_matches = summarize_plant_matches(
        plants=plants,
        detection_types=detection_types,
        rates=rates,
        dist_max=dist_max,
        dist_rate_max=dist_rate_max)

    logger.info(f"Numer of plants found {(plants['num_match'] > 0).sum()}")
    plants.write(plant_matches_filename,
                 format='ascii.commented_header',
                 overwrite=True)
    detection_matches_filename = (
        plant_matches_filename.rsplit('.', 1)[0] + '.detection_matches.txt'
    )
    detection_matches.write(detection_matches_filename,
                            format='ascii.commented_header',
                            overwrite=True)
    logger.info("Wrote plant/detection join table to: %s",
                 detection_matches_filename)

    args = np.argsort(final_detections[:, 5])[::-1]
    final_detections = final_detections[args]
    if final_stamps is not None:
        final_stamps = final_stamps[args]

    logger.info(f"Saving to: {results_filename}")
    with open(results_filename, 'w') as han:
        for i in range(len(final_detections)):
            rx = rates[round(final_detections[i, 2]), 0]
            ry = rates[round(final_detections[i, 2]), 1]
            (x, y, f, snr) = (final_detections[i, 0],
                              final_detections[i, 1],
                              final_detections[i, 4],
                              final_detections[i, 5])
            row = f'snr: {snr} flux: {f} x: {x} y: {y} x_v: {rx} y_v: {ry}\n'
            han.write(row)
