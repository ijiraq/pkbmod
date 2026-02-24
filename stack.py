from astropy.io import fits
from dataclasses import dataclass, field, asdict
import gc
import json
import logging
import numpy as np
from torch.nn import functional
import torch
from typing import Dict, List

import sns_data_nh as data
import sns_utils as utils

EXTENSION_WITH_WCS = 1
VARIANCE_MASK = 'VARIANCE'


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

    def save(self):
        logging.info(f"Saving params to {self.params_filename}")
        with open(self.params_filename, 'w+') as han:
            json.dump(asdict(self), han)


@dataclass
class DataLoad(object):
    """All the data objects that will be needed by shift and stacking routine

    stack_inputs becomes a dictionary holding the various data structures that
    the shift-and-stack wants.  These are loaded into the correct format using
    subroutines in the sns_data_nh.pack_data method.

    """
    warps: Dict[int, fits.HDUList]
    psfs: Dict[int, fits.PrimaryHDU]
    properties: Dict[int, List]
    plants: np.array  # location of injecetd suorces in ref_im
    results_filename: str  # file to save detections to
    plant_matches_filename: str  # store matched plants here
    bitmask: Dict[str, int]
    scrambled: bool = False  # Time scrambled prefix


def oom_observer(device, alloc, device_alloc, device_free):
    logging.debug("Out of memory! Saving snapshot...")
    snapshot = torch.cuda.memory._snapshot()
    from pickle import dump
    with open('oom_snapshot.pkl', 'wb') as f:
        dump(snapshot, f)


def run(dataload: DataLoad, stackparams: StackParams):
    """Given the data load and stacking parameters run the shift-and-stack
    search.

    Args:
        dataload (DataLoad): diffs, masks, variance, psfs, plants, etc
        stackparams (StackParams): parameters to use for the sns search
    """
    
    # mask high, low, nan and inf pixels and
    # remove images that are fully masked
    stack_inputs = data.pack_data(dataload.warps,
                                  dataload.psfs,
                                  dataload.properties)
    stack_inputs = data.mask_variance(stack_inputs, 
                                      bitmask=dataload.bitmask,
                                      variance_trim=stackparams.variance_trim)

    # convert all stack_inputs into numpy arrays
    for key in stack_inputs:
        stack_inputs[key] = np.array(stack_inputs[key])
    # im_nums should be int arrays
    stack_inputs['im_nums'] = stack_inputs['im_nums'].astype('int')

    # now map to the array variables used in rest of code
    datas = stack_inputs['datas']
    masks = stack_inputs['masks']
    variances = stack_inputs['variances']
    dmjds = stack_inputs['dmjds']
    fwhms = stack_inputs['fwhms']
    im_nums = stack_inputs['im_nums']
    psfs = stack_inputs['psfs']
    plants = dataload.plants
    results_filename = dataload.results_filename
    plant_matches_filename = dataload.plant_matches_filename
    bitmask = dataload.bitmask

    # now define the parameters based on stackparams object.
    badflags = stackparams.badflags
    rate_fwhm_grid_step = stackparams.rate_fwhm_grid_step
    n_keep = stackparams.n_keep
    kernel_width = stackparams.kernel_width
    khw = kernel_width//2
    use_gaussian_kernel = stackparams.use_gaussian_kernel
    use_negative_well = stackparams.use_negative_well
    peak_offset_max = stackparams.peak_offset_max
    dist_lim = stackparams.dist_lim
    min_samp = stackparams.min_samp
    min_snr = stackparams.min_snr
    trim_snr = stackparams.trim_snr
    dist_max = stackparams.dist_max
    dist_rate_max = stackparams.dist_rate_max

    rates = data.get_shift_rates(
        plants=plants,
        fwhms=fwhms,
        dmjds=dmjds,
        rate_fwhm_grid_step=rate_fwhm_grid_step)

    logging.debug(f"Creating the convolution kernel: Use Guassian:{use_gaussian_kernel}")
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
    for ir in range(n_im):
        datas[0, 0, ir, :, :] = torch.conv2d(
            datas[:, :, ir, :, :]*inv_variances[:, :, ir, :, :],
            kernel[:, :, ir, :, :], padding='same')
        inv_variances[0, 0, ir, :, :] = torch.conv2d(
            inv_variances[:, :, ir, :, :],
            kernel[:, :, ir, :, :]*kernel[:, :, ir, :, :], padding='same')

    # do the shift-stacking
    snr_image, alpha_image = utils.run_shifts(datas, inv_variances, rates,
                                              dmjds,
                                              min_snr,
                                              writeTestImages=False)
    if n_keep > len(rates):
        logging.warn((f"Number of stack rate: {len(rates)}"
                      f"is smaller than request n_keep {n_keep}. "
                      f"Only keeping {len(rates)} detections per pixel"))
        n_keep = min(n_keep, len(rates))

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
                                         n_keep, rates, A, B)
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

    keeps = utils.brightness_filter_fast(im_datas, inv_vars, c, cv, kernel, dmjds,
                                         rates, detections, khw, n_im,
                                         n_bright_test=10,
                                         test_high=1.15,
                                         test_low=0.85, 
                                         exact_check=False)

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
        c, cv, kernel, dmjds, rates, khw, exact_check=False)

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
            dist_rate_sq = ((plants['rate_x'][i] - rates[np.round(det[:, 2]).astype("int"), 0])**2 +
                            (plants['rate_y'][i] - rates[np.round(det[:, 2]).astype("int"), 1])**2)
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
            (x, y, rx, ry, f, snr) = (final_detections[i, 0],
                                      final_detections[i, 1], 
                                      rates[round(final_detections[i, 2]), 0], 
                                      rates[round(final_detections[i, 2]), 1],
                                      final_detections[i, 4],
                                      final_detections[i, 5])
            row = {"snr": snr, "x": x, "y": y, "x_v": rx, "y_v": ry}
            han.write(" ".join([f"{key}: {row[key]}" for key in row])+"\n")
