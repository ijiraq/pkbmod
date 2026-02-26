import gc
import logging
import numpy as np
from torch.nn import functional
import torch

import sns_data_nh as data
import sns_utils as utils

EXTENSION_WITH_WCS = 1
VARIANCE_MASK = 'VARIANCE'


def run(stack_inputs: dict, stack_params: dict,
        results_filename: str, plant_matches_filename: str,
        low_mem: bool = False, low_mem_tile_w: int = 256):
    """Given the data load and stacking parameters run the shift-and-stack
    search.

    Args:
        stack_inputs (dict): dictionary of np.arrays for stacking: datas, etc.
        stack_params (dict): parameters to use for the sns search
    """
    # this routine is setup to store the index in the detection tensor rather
    # than the rates.
    use_index = True

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

    rates = data.get_shift_rates(
        plants=plants,
        fwhms=fwhms,
        dmjds=dmjds,
        rate_fwhm_grid_step=rate_fwhm_grid_step)

    logging.debug(("Creating the convolution kernel:"
                   f" Use Guassian:{use_gaussian_kernel}"))
    kernel = data.create_kernel(
        psfs=psfs,
        dmjds=dmjds,
        rates=rates,
        useNegativeWell=use_negative_well,
        useGaussianKernel=use_gaussian_kernel,
        kernel_width=kernel_width,
        im_nums=im_nums)

    (A, B) = datas[0].shape

    np_datas = np.expand_dims(np.expand_dims(
        np.array(datas, dtype='float32'), 0), 0)
    np_inv_variances = np.expand_dims(np.expand_dims(
        1.0/np.array(variances, dtype='float32'), 0), 0)
    np_masks = np.expand_dims(np.expand_dims(
        np.array(masks, dtype='int'), 0), 0)

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
    # where pixels are bad
    w = np.where(~((np_masks & badflags) == 0) | np.isnan(datas))
    np_datas[w] = 0.0
    np_inv_variances[w] = 0.0
    np_masks[w] = 0
    # masks with 1 are GOOD pixels, 0 are BAD pixels
    np_masks = np.clip(np_masks, 0, 1)

    # using logical & value of mask > 0 if mask holds value in bits
    # w = (np_masks & badvalue > 0) | np.isnan(datas)
    # np_datas[w] = 0.0
    # np_inv_variances[w] = 0.0
    # for shift-and-stack routines masks with 1 are GOOD pixels
    # np_masks[w] = 0
    # np_masks = np.clip(np_masks, 0, 1)

    # set device value based on gpu availability.
    device = data.get_device()
    # push the data to the device for tourch shift-and-stack
    logging.debug(f"Loading data onto {device}")
    datas = torch.tensor(np_datas).to(device)
    inv_variances = torch.tensor(np_inv_variances).to(device)
    n_im = int(torch.tensor(float(datas.size()[2])).to(device).item())

    _ = torch.rot90(kernel, k=2, dims=(3, 4))

    # convolve pixels and variances with the kernels.
    logging.info(f"Convolving {n_im} images and variances with kernel")
    for ir in range(n_im):
        datas[0, 0, ir, :, :] = torch.conv2d(
            datas[:, :, ir, :, :]*inv_variances[:, :, ir, :, :],
            kernel[:, :, ir, :, :], padding='same')
        inv_variances[0, 0, ir, :, :] = torch.conv2d(
            inv_variances[:, :, ir, :, :],
            kernel[:, :, ir, :, :]*kernel[:, :, ir, :, :], padding='same')

    if n_keep > len(rates):
        logging.warn((f"Number of stack rate: {len(rates)}"
                      f"is smaller than request n_keep {n_keep}. "
                      f"Only keeping {len(rates)} detections per pixel"))
        n_keep = min(n_keep, len(rates))

    if low_mem:
        logging.info("Using low-memory initial shift-and-stack stage")
        top_snr, top_alpha, top_rate_idx = utils.run_shifts_topk(
            datas=datas,
            inv_variances=inv_variances,
            rates=rates,
            dmjds=dmjds,
            min_snr=min_snr,
            n_keep=n_keep,
            tile_w=low_mem_tile_w)
        detections = utils.topk_to_detections(
            top_snr=top_snr,
            top_alpha=top_alpha,
            top_rate_idx=top_rate_idx,
            rates=rates,
            use_index=use_index)
        del top_snr, top_alpha, top_rate_idx
        gc.collect()
    else:
        logging.info("Using original shift-and-stack, high memory")
        # do the shift-stacking
        snr_image, alpha_image = utils.run_shifts(datas, inv_variances, rates,
                                                  dmjds,
                                                  min_snr,
                                                  writeTestImages=False)

        # sort and keep the top n_keep detections,
        sort_inds = torch.zeros((1, 1, n_keep, A, B),
                                dtype=torch.int64, device='cpu')
        logging.debug(f'Packing {snr_image.shape} into {sort_inds.shape}')

        # sort on the SNR index (2) and select the top n_keep
        # these shift rates of those top SNR are selected as the detection
        sort_step = 1000
        a = 0
        b = sort_step
        while b < B:
            b = min(a+sort_step, B)
            logging.debug(f' Sorting {a} to {b} of {B}...')
            sort_inds_wedge = torch.sort(
                snr_image[:, :, :, :, a:b].to(device), 2, descending=True)[1]
            sort_inds[:, :, :, :, a:b] = sort_inds_wedge[:, :, :n_keep, :, :]
            a += sort_step
            logging.debug('Done')

        # trim the negative SNR sources. The reason these show up is
        # because the likelihood formalism sucks
        detections = utils.trim_negative_snr(snr_image, alpha_image, sort_inds,
                                             n_keep, rates, A, B,
                                             use_index=use_index)
        del snr_image, alpha_image, sort_inds
        gc.collect()
        torch.cuda.empty_cache()

    # trim the flux negative sources
    detections = utils.trim_negative_flux(detections)

    # now apply the brightness filter.
    # Check n_bright_test values between test_low and
    # test_high fraction of the estimated value
    # pad the data and variance arrays
    im_datas = functional.pad(torch.tensor(np_datas).to(device),
                              (khw, khw, khw, khw))
    inv_vars = functional.pad(torch.tensor(0.5*np_inv_variances).to(device),
                              (khw, khw, khw, khw))
    #
    del np_datas  # I don't think this is used again.
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
                                         exact_check=False,
                                         use_index=use_index)

    logging.info(f"Number of detections: {len(detections)}")
    logging.info(f"Number kept: {len(keeps)}")
    filt_detections = np.copy(detections[keeps])
    del keeps

    # some cleanup
    del inv_vars
    gc.collect()
    torch.cuda.empty_cache()

    im_masks = functional.pad(torch.tensor(np_masks),
                              (khw, khw, khw, khw)).to(device)
    del np_masks

    # create the stamps
    mean_stamps = utils.create_stamps(im_datas, im_masks,
                                      c, cv, dmjds, rates,
                                      filt_detections, khw,
                                      use_index=use_index)
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
    logging.info(("Number of sources kept after "
                  f"brightness and peak location filtering: {n_det}."))

    w = np.where(clust_detections[:, 5] >= trim_snr)
    clust_detections = clust_detections[w]
    clust_stamps = clust_stamps[w]
    n_det = len(clust_detections)
    logging.info(("Number of sources kept after "
                  f"final SNR trim: {n_det}."))

    inv_vars = functional.pad(
        torch.tensor(0.5*np_inv_variances).to(device), (khw, khw, khw, khw))
    cv[0, 0, 0] = inv_vars[0, 0, 0]

    grid_detections, grid_stamps = utils.position_filter(
        clust_detections, clust_stamps, im_datas, inv_vars,
        c, cv, kernel, dmjds, rates, khw, use_index=use_index)

    w = np.where(grid_detections[:, 5] >= trim_snr)
    final_detections = grid_detections[w]
    final_stamps = grid_stamps[w]
    n_det = len(final_detections)
    # clust_stamps = clust_stamps[w]
    logging.info(f'Number of candidates {n_det}')

    # columns to add to the plant table to track matched detections
    match_columns = ["min_dist_r", "min_dist_v",
                     "det_shift", "det_filt", "det_clust", "det_final",
                     "num_match"]
    for column in match_columns:
        plants[column] = np.nan
    detection_types = {'det_shift': detections,
                       'det_filt': filt_detections,
                       'det_clust': clust_detections,
                       'det_final': final_detections}
    for i in range(len(plants)):
        for detection_type in detection_types:
            det = detection_types[detection_type]
            dist_sq = ((plants['x0'][i] - det[:, 0])**2 +
                       (plants['y0'][i] - det[:, 1])**2)
            # lookup the rate in the rates array if use_index
            if use_index:
                rx = rates[np.round(det[:, 2]).astype("int"), 0]
                ry = rates[np.round(det[:, 2]).astype("int"), 1]
            else:
                rx = final_detections[:, 2]
                ry = final_detections[:, 3]
            dist_rate_sq = ((plants['rate_x'][i] - rx**2) +
                            (plants['rate_y'][i] - ry**2))
            w = (dist_sq < dist_max**2) & (dist_rate_sq < dist_rate_max**2)
            plants[detection_type][i] = w.sum() > 0
        plants['min_dist_r'][i] = np.min(dist_sq)**0.5
        plants['min_dist_v'][i] = np.min(dist_rate_sq)**0.5
        plants['num_match'][i] = w.sum()

    logging.info(f"Numer of plants found {(plants['num_match'] > 0).sum()}")
    plants.write(plant_matches_filename,
                 format='ascii.commented_header',
                 overwrite=True)

    args = np.argsort(final_detections[:, 5])[::-1]
    final_detections = final_detections[args]
    final_stamps = final_stamps[args]

    logging.info(f"Saving to: {results_filename}")
    with open(results_filename, 'w+') as han:
        for i in range(len(final_detections)):
            # lookup the rate in the rates if use_index
            if use_index:
                rx = rates[round(final_detections[i, 2]), 0]
                ry = rates[round(final_detections[i, 2]), 1]
            else:
                rx = final_detections[i, 2]
                ry = final_detections[i, 3]
            (x, y, f, snr) = (final_detections[i, 0],
                              final_detections[i, 1],
                              final_detections[i, 4],
                              final_detections[i, 5])
            row = f'snr: {snr} flux: {f} x: {x} y: {y} x_v: {rx} y_v: {ry}\n'
            han.write(row)
