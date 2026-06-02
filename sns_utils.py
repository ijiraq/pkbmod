import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from scipy import ndimage

from sns_data_nh import get_device

logger = logging.getLogger(__name__)


def dilate_sat_bitmask(masks: np.ndarray, sat_bit: np.uint32, n_pixels: int) -> None:
    """In-place: OR ``sat_bit`` onto pixels within ``n_pixels`` of any SAT pixel.

    Uses square/Chebyshev dilation (same kernel size as :func:`dilate_sat_mask`).
    Accepts mask cubes shaped ``(n_im, H, W)`` or ``(1, 1, n_im, H, W)``.
    """
    if n_pixels <= 0:
        return
    sat_bit = np.uint32(sat_bit)
    struct = np.ones((2 * n_pixels + 1, 2 * n_pixels + 1), dtype=bool)
    if masks.ndim == 5:
        planes = masks[0, 0]
    elif masks.ndim == 3:
        planes = masks
    else:
        raise ValueError(f"expected mask ndim 3 or 5, got {masks.ndim}")
    for ir in range(planes.shape[0]):
        plane = planes[ir]
        sat = (plane & sat_bit) != 0
        if not np.any(sat):
            continue
        grown = ndimage.binary_dilation(sat, structure=struct)
        plane[grown] |= sat_bit


def dilate_sat_mask(mask, bitmask=2**1, n_pixels=2):
    """
    Dilates the SAT mask by exactly n_pixels using PyTorch.

    For CPU uint32 mask planes, prefer :func:`dilate_sat_bitmask`.

    Args:
        cutout_image: The LSST Exposure or MaskedImage object.
        n_pixels (int): The number of pixels to grow the mask in all directions.
    """
    bitmask = torch.tensor(bitmask, dtype=mask.dtype, device=mask.device)
    mask = (mask & bitmask) > 0
    
    # 2. Reshape for PyTorch pooling: (Batch=1, Channel=1, Height, Width)
    mask_4d = mask.unsqueeze(0).unsqueeze(0).float()
    
    # 3. Calculate kernel properties dynamically
    k_size = 2 * n_pixels + 1
    pad = n_pixels
    
    # 4. Run MaxPool to perform exact mathematical dilation
    dilated_4d = torch.functional.max_pool2d(mask_4d, kernel_size=k_size, stride=1, padding=pad)
    
    # 5. Squeeze back down to a 2D boolean tensor (1700x1700)
    final_mask = dilated_4d.squeeze() > 0
    
    return final_mask


def _flux_snr_from_moments(sum_flux, sum_flux_sq, n_im, work_dtype):
    """SNR of the mean flux estimate alpha = mean(PSI/PHI).

    alpha is already an average over N images, so its uncertainty scales as
    std(PSI/PHI) / sqrt(N).  Equivalently:

        SNR = alpha / (std / sqrt(N)) = alpha * sqrt(N) / std
    """
    n = float(n_im)
    alpha = sum_flux / n
    mean_sq = sum_flux_sq / n
    var = torch.clamp(mean_sq - alpha * alpha, min=0.0)
    flux_std = torch.sqrt(var)
    sqrt_n = torch.sqrt(
        torch.tensor(n, device=sum_flux.device, dtype=torch.float32)
    )
    flux_snr = torch.where(
        flux_std > 0,
        (alpha / flux_std) * sqrt_n,
        torch.zeros_like(alpha),
    )
    return alpha.to(work_dtype), flux_snr.to(work_dtype)


def _flux_snr_from_inv_var(alpha, sum_inv_phi, n_im, work_dtype):
    """SNR of alpha = mean(PSI/PHI) using propagated MF weights.

    Var(alpha) ≈ (1/N²) Σ (1/PHI_i), so SNR = alpha × N / sqrt(Σ 1/PHI_i).
    """
    n = float(n_im)
    sqrt_sum_inv = torch.sqrt(torch.clamp(sum_inv_phi, min=0.0))
    sqrt_n = torch.sqrt(
        torch.tensor(n, device=alpha.device, dtype=torch.float32)
    )
    iv_flux_snr = torch.where(
        sqrt_sum_inv > 0,
        alpha * sqrt_n / sqrt_sum_inv,
        torch.zeros_like(alpha),
    )
    return iv_flux_snr.to(work_dtype)


def recompute_pixel_statistics(
    datas,
    inv_variances,
    dmjds,
    rate,
    y,
    x,
    n_im,
    work_dtype=torch.float16,
):
    """Recompute matched-filter stack statistics at one pixel for one rate."""
    device = datas.device
    sum_psi = torch.zeros((), dtype=torch.float32, device=device)
    sum_phi = torch.zeros((), dtype=torch.float32, device=device)
    sum_inv_phi = torch.zeros((), dtype=torch.float32, device=device)
    per_image_flux = []

    for idx in range(n_im):
        if idx == 0:
            psi = datas[0, 0, idx, y, x].to(torch.float32)
            phi = inv_variances[0, 0, idx, y, x].to(torch.float32)
        else:
            shifts = (-round(dmjds[idx] * rate[1]), -round(dmjds[idx] * rate[0]))
            psi = torch.roll(datas[0, 0, idx], shifts=shifts, dims=[0, 1])[y, x].to(
                torch.float32
            )
            phi = torch.roll(inv_variances[0, 0, idx], shifts=shifts, dims=[0, 1])[
                y, x
            ].to(torch.float32)

        sum_psi = sum_psi + psi
        sum_phi = sum_phi + phi
        if phi > 0:
            per_image_flux.append(float((psi / phi).cpu()))
            sum_inv_phi = sum_inv_phi + 1.0 / phi

    sum_psi_f = float(sum_psi.cpu())
    sum_phi_f = float(sum_phi.cpu())
    sum_inv_phi_f = float(sum_inv_phi.cpu())
    sqrt_phi = float(np.sqrt(sum_phi_f)) if sum_phi_f > 0 else 0.0
    nu = sum_psi_f / sqrt_phi if sqrt_phi > 0 else -1.0
    alpha = float(np.mean(per_image_flux)) if per_image_flux else 0.0
    alpha_snr = alpha * sqrt_phi if sqrt_phi > 0 else 0.0
    n_used = len(per_image_flux)
    flux_std = float(np.std(per_image_flux)) if n_used > 1 else 0.0
    flux_snr = (
        (alpha / flux_std) * np.sqrt(n_used) if flux_std > 0 else 0.0
    )
    iv_flux_snr = (
        alpha * np.sqrt(n_used) / np.sqrt(sum_inv_phi_f)
        if sum_inv_phi_f > 0
        else 0.0
    )

    return {
        "sum_psi": sum_psi_f,
        "sum_phi": sum_phi_f,
        "sum_inv_phi": sum_inv_phi_f,
        "sqrt_sum_phi": sqrt_phi,
        "nu": nu,
        "alpha": alpha,
        "alpha_times_sqrt_phi": alpha_snr,
        "flux_std": flux_std,
        "flux_snr": flux_snr,
        "iv_flux_snr": iv_flux_snr,
        "n_images_with_phi": len(per_image_flux),
    }


def log_snr_debug_samples(
    detections,
    datas,
    inv_variances,
    dmjds,
    rates,
    n_im,
    khw,
    stamps=None,
    sample_size=24,
    stage="initial",
    run_id="pre-fix",
    recompute_convolved=True,
):
    """Log SNR diagnostics for a stratified sample of detections."""
    if len(detections) == 0:
        return

    order = np.argsort(detections[:, 5])
    if len(order) <= sample_size:
        sample_idx = order
    else:
        low = order[: sample_size // 3]
        mid = order[len(order) // 2 - sample_size // 6 : len(order) // 2 + sample_size // 6]
        high = order[-sample_size // 3 :]
        sample_idx = np.unique(np.concatenate([low, mid, high]))

    mean_rate = np.mean(rates, axis=0)

    for det_i in sample_idx:
        x, y, rate_idx, flux_snr_col, flux, det_snr = detections[det_i, :6]
        iv_flux_snr_col = (
            float(detections[det_i, 6]) if detections.shape[1] > 6 else np.nan
        )
        x_i, y_i = int(x), int(y)
        rate_idx_i = int(round(rate_idx))
        assigned_rate = rates[rate_idx_i]

        entry = {
            "stage": stage,
            "det_table_index": int(det_i),
            "x": float(x),
            "y": float(y),
            "rate_idx": rate_idx_i,
            "reported_det_snr": float(det_snr),
            "reported_flux_snr": float(flux_snr_col),
            "reported_iv_flux_snr": iv_flux_snr_col,
            "reported_flux": float(flux),
            "assigned_rate": [float(assigned_rate[0]), float(assigned_rate[1])],
        }

        if recompute_convolved:
            assigned_stats = recompute_pixel_statistics(
                datas, inv_variances, dmjds, assigned_rate, y_i, x_i, n_im
            )
            wrong_stats = recompute_pixel_statistics(
                datas, inv_variances, dmjds, mean_rate, y_i, x_i, n_im
            )
            entry.update(
                {
                    "assigned_nu": assigned_stats["nu"],
                    "assigned_alpha": assigned_stats["alpha"],
                    "assigned_alpha_sqrt_phi": assigned_stats["alpha_times_sqrt_phi"],
                    "assigned_flux_snr": assigned_stats["flux_snr"],
                    "assigned_iv_flux_snr": assigned_stats["iv_flux_snr"],
                    "assigned_flux_std": assigned_stats["flux_std"],
                    "assigned_sqrt_sum_phi": assigned_stats["sqrt_sum_phi"],
                    "assigned_sum_inv_phi": assigned_stats["sum_inv_phi"],
                    "wrong_rate_nu": wrong_stats["nu"],
                    "nu_minus_reported": assigned_stats["nu"] - float(det_snr),
                    "flux_snr_minus_reported": assigned_stats["flux_snr"] - float(flux_snr_col),
                    "iv_flux_snr_minus_reported": assigned_stats["iv_flux_snr"] - iv_flux_snr_col,
                }
            )

        if stamps is not None:
            stamp = stamps[det_i] if det_i < len(stamps) else None
            if stamp is not None and np.any(np.isfinite(stamp)):
                peak = float(np.nanmax(stamp))
                center = float(stamp[khw, khw])
                std = float(np.nanstd(stamp))
                entry["stamp_peak"] = peak
                entry["stamp_center"] = center
                entry["stamp_std"] = std
                entry["stamp_peak_over_std"] = peak / std if std > 0 else np.nan
                entry["stamp_center_over_std"] = center / std if std > 0 else np.nan

        logger.debug("SNR debug sample (%s, run_id=%s): %s", stage, run_id, entry)


def run_shifts(
    datas,
    inv_variances,
    rates,
    dmjds,
    min_snr,
    n_keep=4,
    writeTestImages=False,
    tile_w=256,
    word_dtype=torch.float16,
):
    n_im = len(datas[0, 0, :])
    logger.debug(f"NUM IM {n_im}")
    c = torch.zeros_like(datas)
    c[0, 0, 0] = datas[0, 0, 0]
    cv = torch.zeros_like(datas)
    cv[0, 0, 0] = inv_variances[0, 0, 0]

    snr_image = torch.zeros(
        (1, 1, len(rates), datas.size()[3], datas.size()[4]), dtype=word_dtype
    )
    flux_snr_image = torch.zeros(
        (1, 1, len(rates), datas.size()[3], datas.size()[4]), dtype=word_dtype
    )
    iv_flux_snr_image = torch.zeros(
        (1, 1, len(rates), datas.size()[3], datas.size()[4]), dtype=word_dtype
    )
    alpha_image = torch.zeros(
        (1, 1, len(rates), datas.size()[3], datas.size()[4]), dtype=word_dtype
    )

    # rates=[[-200.6777606841237, 78.88276756451387]]
    for ir in range(len(rates)):
        for idx in range(1, n_im):
            shifts = (
                -round(dmjds[idx] * rates[ir][1]),
                -round(dmjds[idx] * rates[ir][0]),
            )
            c[
                0,
                0,
                idx,
            ] = torch.roll(datas[0, 0, idx], shifts=shifts, dims=[0, 1])
            cv[0, 0, idx] = torch.roll(
                inv_variances[0, 0, idx], shifts=shifts, dims=[0, 1]
            )
        # C = functional.conv3d(c, kernel)
        # sums = torch.sum(functional.conv3d(c, ones,padding='same'), 2)

        # these are set abov
        PSI = c
        PHI = cv

        # median_alpha = torch.median(PSI/PHI, dim=2)[0]
        # doesn't seem to be used for anything

        flux = torch.nan_to_num(PSI / PHI, nan=0.0, posinf=0.0, neginf=0.0)
        sum_flux = torch.nansum(flux, 2)
        sum_flux_sq = torch.nansum(flux * flux, 2)
        alpha, flux_snr = _flux_snr_from_moments(
            sum_flux[0, 0], sum_flux_sq[0, 0], n_im, word_dtype
        )
        sum_inv_phi = torch.nansum(
            torch.where(PHI > 0, 1.0 / PHI, torch.zeros_like(PHI)), dim=2
        )[0, 0]
        iv_flux_snr = _flux_snr_from_inv_var(
            alpha, sum_inv_phi, n_im, word_dtype
        )
        alpha = alpha.unsqueeze(0).unsqueeze(0)
        flux_snr = flux_snr.unsqueeze(0).unsqueeze(0)
        iv_flux_snr = iv_flux_snr.unsqueeze(0).unsqueeze(0)

        # Matched-filter detection statistic (used only for candidate gating).
        nu = torch.sum(PSI, 2) / torch.pow(torch.sum(PHI, 2), 0.5)
        nu = torch.nan_to_num(nu, -1.0)
        nu[nu == float("Inf")] = 0

        where = nu[0, 0] > min_snr
        inds = where.nonzero()

        snr_image[0, 0, ir, inds[:, 0], inds[:, 1]] = (
            nu[0, 0, inds[:, 0], inds[:, 1]].cpu().to(word_dtype)
        )
        flux_snr_image[0, 0, ir, inds[:, 0], inds[:, 1]] = (
            flux_snr[0, 0, inds[:, 0], inds[:, 1]].cpu().to(word_dtype)
        )
        iv_flux_snr_image[0, 0, ir, inds[:, 0], inds[:, 1]] = (
            iv_flux_snr[0, 0, inds[:, 0], inds[:, 1]].cpu().to(word_dtype)
        )
        alpha_image[0, 0, ir, inds[:, 0], inds[:, 1]] = (
            alpha[0, 0, inds[:, 0], inds[:, 1]].cpu().to(word_dtype)
        )

        gc.collect()
        torch.cuda.empty_cache()

    logger.debug(f"Max per image flux of candidates: {torch.max(alpha_image)}")
    logger.debug(f"Max per image snr of candidates: {torch.max(snr_image)}")

    del c, cv, datas, inv_variances, PSI, PHI
    gc.collect()
    torch.cuda.empty_cache()

    return snr_image, alpha_image, flux_snr_image, iv_flux_snr_image


def trim_negative_snr(
    snr_image, alpha_image, sort_inds, n_keep, A, B, dtype=np.float16,
    flux_snr_image=None,
    iv_flux_snr_image=None,
):
    # trim the negative SNR sources. The reason these show up is
    # because the likelihood formalism sucks
    idx, idy = np.meshgrid(np.arange(B), np.arange(A))
    idx = idx.reshape(A * B)
    idy = idy.reshape(A * B)
    for n in range(n_keep):
        s = sort_inds[0, 0, n, :, :].reshape(A * B)
        SNR = snr_image[0, 0, s, idy, idx]
        alpha = alpha_image[0, 0, s, idy, idx]
        if flux_snr_image is not None:
            flux_snr_vals = flux_snr_image[0, 0, s, idy, idx]
        else:
            flux_snr_vals = 0.0
        if iv_flux_snr_image is not None:
            iv_flux_snr_vals = iv_flux_snr_image[0, 0, s, idy, idx]
        else:
            iv_flux_snr_vals = 0.0

        where = SNR > 0
        inds = where.nonzero()[:, 0]
        logger.debug(f"keep index length: {inds.shape}")
        logger.debug(f"length of indexs: {(s[inds]).shape}")
        if n == 0:
            keeps = np.zeros((len(inds), 7), dtype=dtype)
            keeps[:, 0] = idx[inds]
            keeps[:, 1] = idy[inds]
            keeps[:, 2] = s.reshape(A * B)[inds]
            keeps[:, 3] = np.asarray(flux_snr_vals).reshape(A * B)[inds]
            keeps[:, 4] = alpha.reshape(A * B)[inds]
            keeps[:, 5] = SNR.reshape(A * B)[inds]
            keeps[:, 6] = np.asarray(iv_flux_snr_vals).reshape(A * B)[inds]
        else:
            nkeeps = np.zeros((len(inds), 7), dtype=dtype)
            logger.debug(f"Keeps size: {nkeeps.shape}")
            nkeeps[:, 0] = idx[inds]
            nkeeps[:, 1] = idy[inds]
            nkeeps[:, 2] = s[inds]
            nkeeps[:, 3] = np.asarray(flux_snr_vals).reshape(A * B)[inds]
            nkeeps[:, 4] = alpha.reshape(A * B)[inds]
            nkeeps[:, 5] = SNR.reshape(A * B)[inds]
            nkeeps[:, 6] = np.asarray(iv_flux_snr_vals).reshape(A * B)[inds]
            keeps = np.concatenate([keeps, nkeeps])

    logger.debug(f"Keeping {len(keeps)} candidates")

    detections = np.array(keeps)
    del keeps, idx, idy
    return detections


def trim_negative_flux(detections):
    pos = np.where(detections[:, 4] > 0)
    detections = detections[pos]
    logger.debug(f"Keeping {len(detections)} positive flux candidates")
    return detections


def brightness_filter(
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
):

    device = get_device()
    nb_ref = torch.as_tensor(
        10.0 ** np.linspace(np.log10(test_low), np.log10(test_high), n_bright_test),
        dtype=im_datas.dtype,
        device=device,
    )

    for ir in range(len(rates)):
        t1 = time.time()
        w = np.where(
            (detections[:, 2] == rates[ir][0]) & (detections[:, 3] == rates[ir][1])
        )

        for idx in range(1, n_im):
            shifts = (
                -round(dmjds[idx] * rates[ir][1]),
                -round(dmjds[idx] * rates[ir][0]),
            )
            c[0, 0, idx] = torch.roll(im_datas[0, 0, idx], shifts=shifts, dims=[0, 1])
            cv[0, 0, idx] = torch.roll(inv_vars[0, 0, idx], shifts=shifts, dims=[0, 1])

        arg_mins = torch.zeros(len(detections), dtype=torch.uint32)
        for idx in w[0]:
            (x, y) = detections[idx, :2]
            x = int(x) + khw
            # array of scaled brightnesses in steps of brightness*test_low
            # to brightness*test_high
            y = int(y) + khw
            nb = nb_ref * detections[idx, 4]
            k = kernel.repeat((1, n_bright_test, 1, 1, 1))
            for ib in range(nb.size()[0]):
                k[:, ib, :, :, :] *= nb[ib]

            diff = c[:, :, :, y - khw : y + khw, x - khw : x + khw].repeat(
                (1, n_bright_test, 1, 1, 1)
            )
            diff -= k
            diff = diff * diff
            diff *= cv[:, :, :, y - khw : y + khw, x - khw : x + khw].repeat(
                (1, n_bright_test, 1, 1, 1)
            )

            tmp = torch.sum(diff, (0, 2, 3, 4))
            arg_mins[idx] = torch.argmin(tmp)

        arg_mins_cpu = arg_mins.cpu()

        W = np.where((arg_mins_cpu != 0) & (arg_mins_cpu != (n_bright_test - 1)))
        logger.debug(
            (
                f"{ir + 1}/{len(rates)}, pre: {len(w[0])}, "
                f"post: {len(W[0])},  in time {time.time() - t1}"
            )
        )
        if ir == 0:
            keeps = W[0]
        else:
            keeps = np.concatenate([keeps, W[0]])
    logger.debug(
        (
            f"Number kept after brightness filter {len(keeps)}"
            f" of {len(detections)} total detections."
        )
    )

    return keeps


def create_stamps(im_datas, im_masks, c, cv, dmjds, rates, filt_detections, khw):
    # create stamps that are centred on the location of sources in filt_detections
    # do all the sources wit the same shift rates in batches. 
    stamp_width = 2*khw+1
    mean_stamps = mean_stamps = np.full(
            (len(filt_detections), stamp_width, stamp_width), np.nan, dtype=np.float32
            )
    for ir in range(len(rates)):
        # these are required to reset from the nans below
        c[0, 0, 0] = im_datas[0, 0, 0]
        cv[0, 0, 0] = im_masks[0, 0, 0]

        # t1 = time.time()
        w = np.where(np.round(filt_detections[:, 2]).astype("int") == ir)

        for idx in range(1, len(dmjds)):
            shifts = (
                -round(dmjds[idx] * rates[ir][1]),
                -round(dmjds[idx] * rates[ir][0]),
            )
            c[0, 0, idx] = torch.roll(im_datas[0, 0, idx], shifts=shifts, dims=[0, 1])
            # mask values with 1 are GOOD pixels
            cv[0, 0, idx] = torch.roll(im_masks[0, 0, idx], shifts=shifts, dims=[0, 1])
        mean_stamp_frame = torch.sum(c, 2)
        mask_frame = torch.sum(cv, 2)

        for iw in w[0]:
            x, y = filt_detections[iw, :2].astype("int")
            # im_datas is padded by khw; detection (x, y) is in the original frame.
            y0, y1 = y, y + stamp_width
            x0, x1 = x, x + stamp_width
            mean_stamp = mean_stamp_frame[0, 0, y0:y1, x0:x1]
            mask = np.copy(mask_frame[0, 0, y0:y1, x0:x1].cpu())
            mask[np.where((np.isnan(mask)) | (np.isinf(mask)))] = 0.0
            np.clip(mask, 1.0, 1.0*len(dmjds), out=mask)
            stamp = np.copy(mean_stamp.cpu()) / mask
            if stamp.shape != (stamp_width, stamp_width):
                  logger.warning(
                      "Skipping detection %d: stamp shape %s (expected %dx%d)",
                      iw, stamp.shape, stamp_width, stamp_width,
                  )
                  continue
            mean_stamps[iw] = stamp

    del mask, mean_stamp, mean_stamp_frame, mask_frame
    gc.collect()
    torch.cuda.empty_cache()

    return mean_stamps


# trim the ones with peak offset more than peak_offset_max pixels
def peak_offset_filter(stamps, peak_offset_max):
   """Drop stamps whose brightest pixel is far from the stamp center.
 
      Assumes stamps were built with detection (x, y) centered at (khw, khw),
      i.e. via centered extraction in :func:`create_stamps`.
   """
   (N, a, b) = stamps.shape
   (gx, gy) = np.meshgrid(np.arange(b), np.arange(a))
   gx = gx.reshape(a * b)
   gy = gy.reshape(a * b)
   rs_stamps = stamps.reshape(N, a * b)
   args = np.argmax(rs_stamps, axis=1)
   X = gx[args]
   Y = gy[args]
   # For size 2*khw+1, (b/2, a/2) is the nominal center pixel.
   radial_d = np.hypot(X - (b - 1) / 2.0, Y - (a - 1) / 2.0)
   keep = np.where(radial_d < peak_offset_max)[0]
   return keep


# do predictive line clustering
def predictive_line_cluster_indices(
    filt_detections, dmjds, dist_lim, rates, min_samp=2, init_select_proc_distance=60
):
    """Indices into ``filt_detections`` (and parallel ``stamps``) to keep.

    Returns:
        numpy.ndarray: 1D int indices, in the order clusters are accepted.
    """
    proc_filt_detections = np.copy(filt_detections)
    rates = np.asarray(rates)

    proc_inds = np.arange(len(proc_filt_detections), dtype=int)
    clust_inds = []

    while len(proc_filt_detections) > 0:
        arg_max = np.argmax(proc_filt_detections[:, 5])  # 5 - max on SNR
        x_o = proc_filt_detections[arg_max, 0]
        y_o = proc_filt_detections[arg_max, 1]
        rate_idx_o = int(round(proc_filt_detections[arg_max, 2]))
        rx_o = rates[rate_idx_o, 0]
        ry_o = rates[rate_idx_o, 1]

        # this secondary where command is necessary because of memory
        # overflows in large detection lists
        w = np.where(
            (proc_filt_detections[:, 0] > proc_filt_detections[arg_max, 0] - init_select_proc_distance)
            & (
                proc_filt_detections[:, 0]
                < proc_filt_detections[arg_max, 0] + init_select_proc_distance
            )
            & (proc_filt_detections[:, 1] > proc_filt_detections[arg_max, 1] - init_select_proc_distance)
            & (
                proc_filt_detections[:, 1]
                < proc_filt_detections[arg_max, 1] + init_select_proc_distance
            )
        )

        W = np.where(
            (
                (proc_filt_detections[w[0], 0] - proc_filt_detections[arg_max, 0]) ** 2
                + (proc_filt_detections[w[0], 1] - proc_filt_detections[arg_max, 1])
                ** 2
            )
            < init_select_proc_distance**2
        )
        w = w[0][W[0]]

        fd_subset = proc_filt_detections[w]
        fd_rate_idx = np.round(fd_subset[:, 2]).astype(np.intp)
        fd_rx = rates[fd_rate_idx, 0]
        fd_ry = rates[fd_rate_idx, 1]

        drx = fd_rx - rx_o
        dry = fd_ry - ry_o
        dt = dmjds  # just for clarity

        # predicted centroid  position of secondary detection shifted at the
        # differential wrong rate.
        x_n, y_n = x_o - drx * dt[-1], y_o - dry * dt[-1]
        # predicted centroid shifted such that best detection is now at origin
        dx, dy = (x_n - x_o), (y_n - y_o)

        dxp = dx * fd_subset[:, 1]
        dyp = dy * fd_subset[:, 0]
        xm = x_n * y_o
        ym = y_n * x_o
        dx2 = dx**2
        dy2 = dy**2
        top = np.abs(dyp - dxp + xm - ym)
        bottom = np.sqrt(dx2 + dy2)
        dist = top / bottom

        clust = np.where(
            (dist < dist_lim)
            | (np.isnan(dist))
            | ((dist < dist_lim) & (drx == 0) & (dry == 0))
        )
        if len(clust[0]) >= min_samp:
            clust_inds.append(proc_inds[arg_max])

        mask = np.ones(len(proc_filt_detections), dtype="bool")
        mask[w[clust]] = False
        proc_filt_detections = proc_filt_detections[mask]
        proc_inds = proc_inds[mask]

    return np.array(clust_inds, dtype=int)


def predictive_line_cluster(
    filt_detections, stamps, dmjds, dist_lim, rates, min_samp=2,
    init_select_proc_distance=60,
):
    """Backward-compatible wrapper: subset of ``filt_detections`` and ``stamps``."""
    idx = predictive_line_cluster_indices(
        filt_detections, dmjds, dist_lim, rates, min_samp, init_select_proc_distance
    )
    return filt_detections[idx], stamps[idx]


def position_filter(
    detections,
    stamps,
    im_datas,
    inv_vars,
    c,
    cv,
    kernel,
    dmjds,
    rates,
    khw,
    n_offsets=5,
):
    clust_detections = detections
    clust_stamps = stamps
    # now apply a positional filter on the clust_detections to see
    # if the likelihood minimimum is near the centre
    # n_offsets = 5 # +- n_offsets in x and y
    n_o = n_offsets * 2 + 1
    # copy the tensor data in kernel to k with extra space for offsets
    k = kernel.repeat((1, n_o * n_o, 1, 1, 1))

    danger_edges = []
    # these are the indices of the maximal offsets in x and y
    for iy in range(n_o):
        for ix in range(n_o):
            i = iy * n_o + ix
            shifts = (0, iy - n_offsets, ix - n_offsets)
            # roll the kernel data by offset amounts and place
            # into offset kernel tensor.  This moves centre of the
            # kernel over a grid of size n_o x n_o positions
            k[0, i, :, :, :] = torch.roll(kernel[0, 0], shifts=shifts, dims=[0, 1, 2])
            if iy == 0 or iy == n_o - 1 or ix == 0 or ix == n_o - 1:
                # if the best offset pushes to the edget then
                # this likely indicates the sources is not a real
                # moving source but noise that is coherent at the shift rate
                danger_edges.append(i)

    # cv will hold the variances which will be rolled along with
    # the image data which is in 'c'
    cv[0, 0, 0] = inv_vars[0, 0, 0]
    c[0, 0, 0] = im_datas[0, 0, 0]

    keeps = []
    for ir in range(len(rates)):
        w = np.where(np.round(clust_detections[:, 2]).astype("int") == ir)
        if len(w[0]) == 0:
            continue

        for idx in range(1, len(dmjds)):
            shifts = (
                -round(dmjds[idx] * rates[ir][1]),
                -round(dmjds[idx] * rates[ir][0]),
            )
            c[0, 0, idx] = torch.roll(im_datas[0, 0, idx], shifts=shifts, dims=[0, 1])
            cv[0, 0, idx] = torch.roll(inv_vars[0, 0, idx], shifts=shifts, dims=[0, 1])

        for idx in w[0]:
            (x, y) = clust_detections[idx, :2]
            x = int(x)  # +khw
            y = int(y)  # +khw

            K = k * clust_detections[idx, 4]
            c_patch = c[:, :, :, y : y + khw * 2, x : x + khw * 2]
            cv_patch = cv[:, :, :, y : y + khw * 2, x : x + khw * 2]

            diff = c_patch.repeat((1, n_o * n_o, 1, 1, 1))
            diff -= K
            diff = diff**2
            diff *= cv_patch.repeat((1, n_o * n_o, 1, 1, 1))
            arg_min = torch.argmin(torch.sum(diff, (0, 2, 3, 4)))
            min_ix = arg_min % n_o
            min_iy = int((arg_min - min_ix) / n_o)

            min_ix -= n_offsets
            min_iy -= n_offsets
            logger.debug(
                (
                    f"position 'Xi^2' match for sources at {x},{y} at rate:"
                    f"{rates[ir]} is offset {min_ix},{min_iy}"
                )
            )
            kept = arg_min not in danger_edges
            if kept:
                keeps.append(idx)

    return np.array(keeps)


def brightness_filter_fast(
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
    n_det_iter=200,
    word_dtype=torch.float16,
):

    device = get_device()

    nb_ref = torch.as_tensor(
        10.0 ** np.linspace(np.log10(test_low), np.log10(test_high), n_bright_test),
        dtype=word_dtype,
        device=device,
    )

    ks = 2 * khw  # kernel spatial size
    # Precompute the unit-scaled kernel: shape (n_bright_test, n_im, ks, ks)
    # k_unit[ib] = kernel * nb_ref[ib]
    kern = kernel[0, 0]  # (n_im, ks, ks)
    # Wes added None and None to the front of each index selection
    # (n_bright_test, n_im, ks, ks)
    k_unit = kern[None, None, :, :, :] * nb_ref[None, :, None, None, None]

    # Precompute patch offset indices (reused every rate iteration)
    dy = torch.arange(ks, device=device)
    dx = torch.arange(ks, device=device)
    im_idx = torch.arange(n_im, device=device)

    keeps_list = []
    log_info_enabled = logger.isEnabledFor(logging.INFO)

    for ir in range(len(rates)):
        t1 = time.time()
        W = np.where(np.round(detections[:, 2]).astype("int") == ir)

        if len(W[0]) == 0:
            continue

        # Roll images for this rate
        for idx in range(1, n_im):
            shifts = (
                -round(dmjds[idx] * rates[ir][1]),
                -round(dmjds[idx] * rates[ir][0]),
            )
            c[0, 0, idx] = torch.roll(im_datas[0, 0, idx], shifts=shifts, dims=[0, 1])
            cv[0, 0, idx] = torch.roll(inv_vars[0, 0, idx], shifts=shifts, dims=[0, 1])

        kept_chunks = []
        n_done_iter = 0
        while n_done_iter < len(W[0]):
            w = W[0][n_done_iter : min(n_done_iter + n_det_iter, len(W[0]))]

            det_idx = w
            n_det = len(det_idx)
            if n_det == 0:
                continue

            # Extract coordinates for all detections at this rate
            xs = detections[det_idx, 0].astype(int) + khw
            ys = detections[det_idx, 1].astype(int) + khw
            fluxes = torch.as_tensor(
                detections[det_idx, 4], dtype=word_dtype, device=device
            )

            # Batch-extract all patches via advanced indexing
            c_3d = c[0, 0]  # (n_im, H, W)
            cv_3d = cv[0, 0]  # (n_im, H, W)

            ys_t = torch.tensor(ys, device=device, dtype=torch.long)
            xs_t = torch.tensor(xs, device=device, dtype=torch.long)

            y_idx = ys_t[:, None] - khw + dy[None, :]  # (n_det, ks)
            x_idx = xs_t[:, None] - khw + dx[None, :]  # (n_det, ks)

            # Expand indices for (n_det, n_im, ks, ks) gather
            y_exp = y_idx[:, None, :, None].expand(n_det, n_im, ks, ks)
            x_exp = x_idx[:, None, None, :].expand(n_det, n_im, ks, ks)
            im_exp = im_idx[None, :, None, None].expand(n_det, n_im, ks, ks)

            patches_c = c_3d[im_exp, y_exp, x_exp]  # (n_det, n_im, ks, ks)
            patches_cv = cv_3d[im_exp, y_exp, x_exp]  # (n_det, n_im, ks, ks)

            # Exploit argmin scale-invariance:
            # argmin((p - k*f)^2 * w) == argmin((p/f - k)^2 * w)
            # Divide patches by flux (scalar per detection) instead
            # of scaling k_unit by flux.
            # k_unit is (n_bright_test, n_im, ks, ks) — shared
            # across detections, NOT (n_det, n_bright_test, ...).
            # Guard against non-positive flux (should not happen after
            # trim_negative_flux).
            safe_fluxes = fluxes.clone()
            # fallback; won't pass filter anyway
            safe_fluxes[safe_fluxes <= 0] = 1.0

            # Wilson's version
            # (n_det, n_im, ks, ks)
            patches_c_norm = patches_c / safe_fluxes[:, None, None, None]
            # Broadcasting:
            # (n_det,1,n_im,ks,ks) - (1,n_bright_test,n_im,ks,ks) ->
            #          (n_det,n_bright_test,n_im,ks,ks)
            # instead of [None, :, :, :, :]
            diff = patches_c_norm[:, None, :, :, :] - k_unit[None, :, :, :, :]
            diff = diff * diff
            # (n_det, n_bright_test, n_im, ks, ks)
            diff = diff * patches_cv[:, None, :, :, :]
            """
            # using kernel*f like below results the same
            # as wilson's version above.
            # patches_c = patches_c  # (n_det, n_im, ks, ks)
            # Broadcasting:
            # (n_det,1,n_im,ks,ks) - (1,n_bright_test,n_im,ks,ks)
            #                    -> (n_det,n_bright_test,n_im,ks,ks)
            # instead of [None, :, :, :, :]
            diff = patches_c[:, None, :, :, :] - k_unit[None, :, :, :, :] *
                        safe_fluxes[:, None, None,None, None]
            diff = diff * diff
             # (n_det, n_bright_test, n_im, ks, ks)
            diff = diff * patches_cv[:, None, :, :, :]
            """
            tmp_l = diff.sum(dim=(3, 4, 5))[0]  # (n_det, n_bright_test)
            arg_mins = torch.argmin(tmp_l, dim=1).cpu().numpy()  # (n_det,)

            # Filter: keep detections if best brightness not at the boundary
            valid = (arg_mins != 0) & (arg_mins != (n_bright_test - 1))

            kept_batch = det_idx[np.where(valid)]
            if len(kept_batch) > 0:
                kept_chunks.append(kept_batch)

            n_done_iter += len(w)
            del diff

        if kept_chunks:
            kept_idx = np.concatenate(kept_chunks)
            keeps_list.append(kept_idx)
        else:
            kept_idx = np.array([], dtype=np.intp)

        if log_info_enabled:
            logger.debug(
                ("%d/%d, vx: %.5f, vy: %.5f, pre: %d, post: %d, in time %.3f"),
                ir + 1,
                len(rates),
                rates[ir][0],
                rates[ir][1],
                len(W[0]),
                len(kept_idx),
                time.time() - t1,
            )
    if keeps_list:
        keeps = np.concatenate(keeps_list)
    else:
        keeps = np.array([], dtype=np.intp)

    logger.debug(
        (
            f"Number kept after brightness filter {len(keeps)} "
            f"of {len(detections)} total detections."
        )
    )

    return keeps


def run_shifts_topk(
    datas,
    inv_variances,
    rates,
    dmjds,
    min_snr,
    n_keep,
    tile_w=256,
    work_dtype=torch.float16,
    output_dtype=torch.float16,
):
    """Run shift-and-stack in low-memory mode and keep online per-pixel top-k.

    Do PHI/PSI sums in loop instead, keeps GPU limited to one image foot print

    Returns CPU tensors with shapes:
      top_snr: (k, A, B) float16 — matched-filter detection statistic (nu)
      top_flux_snr: (k, A, B) float16 — empirical flux-coherence SNR
      top_iv_flux_snr: (k, A, B) float16 — inv-variance propagated flux SNR
      top_alpha: (k, A, B) float16
      top_rate_idx: (k, A, B) int32
    """
    n_im = int(datas.shape[2])
    A = int(datas.shape[3])
    B = int(datas.shape[4])
    k = int(min(n_keep, len(rates)))
    device = get_device()

    if k <= 0:
        raise ValueError(f"n_keep: {n_keep} and N rates: {len(rates)}")

    top_snr_cpu = torch.full((k, A, B), -float("inf"), dtype=output_dtype, device="cpu")
    top_flux_snr_cpu = torch.zeros((k, A, B), dtype=output_dtype, device="cpu")
    top_iv_flux_snr_cpu = torch.zeros((k, A, B), dtype=output_dtype, device="cpu")
    top_alpha_cpu = torch.zeros((k, A, B), dtype=output_dtype, device="cpu")
    top_rate_idx_cpu = torch.full((k, A, B), -1, dtype=torch.int32, device="cpu")

    for ir, rate in enumerate(rates):
        sum_psi = torch.zeros((A, B), dtype=work_dtype, device=device)
        sum_phi = torch.zeros((A, B), dtype=work_dtype, device=device)
        sum_flux = torch.zeros((A, B), dtype=work_dtype, device=device)
        sum_flux_sq = torch.zeros((A, B), dtype=work_dtype, device=device)
        sum_inv_phi = torch.zeros((A, B), dtype=work_dtype, device=device)

        for idx in range(n_im):
            if idx == 0:
                psi = datas[0, 0, idx]
                phi = inv_variances[0, 0, idx]
            else:
                shifts = (-round(dmjds[idx] * rate[1]), -round(dmjds[idx] * rate[0]))
                psi = torch.roll(datas[0, 0, idx], shifts=shifts, dims=[0, 1])
                phi = torch.roll(inv_variances[0, 0, idx], shifts=shifts, dims=[0, 1])

            psi = psi.to(work_dtype)
            phi = phi.to(work_dtype)
            flux_i = torch.nan_to_num(psi / phi, nan=0.0, posinf=0.0, neginf=0.0)
            sum_psi += psi
            sum_phi += phi
            sum_flux += flux_i
            sum_flux_sq += flux_i * flux_i
            sum_inv_phi += torch.where(phi > 0, 1.0 / phi, torch.zeros_like(phi))

        nu = torch.nan_to_num(
            sum_psi / torch.sqrt(sum_phi), nan=-1.0, posinf=0.0, neginf=-1.0
        )
        alpha, flux_snr = _flux_snr_from_moments(
            sum_flux, sum_flux_sq, n_im, work_dtype
        )
        iv_flux_snr = _flux_snr_from_inv_var(
            alpha, sum_inv_phi, n_im, work_dtype
        )

        nu = torch.where(nu > min_snr, nu, torch.full_like(nu, -float("inf")))
        alpha = torch.where(torch.isfinite(nu), alpha, torch.zeros_like(alpha))
        flux_snr = torch.where(torch.isfinite(nu), flux_snr, torch.zeros_like(flux_snr))
        iv_flux_snr = torch.where(
            torch.isfinite(nu), iv_flux_snr, torch.zeros_like(iv_flux_snr)
        )

        x0 = 0
        while x0 < B:
            x1 = min(x0 + tile_w, B)

            prev_snr = top_snr_cpu[:, :, x0:x1].to(device=device, dtype=work_dtype)
            prev_flux_snr = top_flux_snr_cpu[:, :, x0:x1].to(
                device=device, dtype=work_dtype
            )
            prev_iv_flux_snr = top_iv_flux_snr_cpu[:, :, x0:x1].to(
                device=device, dtype=work_dtype
            )
            prev_alpha = top_alpha_cpu[:, :, x0:x1].to(device=device, dtype=work_dtype)
            prev_rate = top_rate_idx_cpu[:, :, x0:x1].to(
                device=device, dtype=torch.int32
            )

            cand_snr = nu[:, x0:x1].unsqueeze(0)
            cand_flux_snr = flux_snr[:, x0:x1].unsqueeze(0)
            cand_iv_flux_snr = iv_flux_snr[:, x0:x1].unsqueeze(0)
            cand_alpha = alpha[:, x0:x1].unsqueeze(0)
            cand_rate = torch.full(
                (1, A, x1 - x0), ir, dtype=torch.int32, device=device
            )

            all_snr = torch.cat([prev_snr, cand_snr], dim=0)
            _, idx = torch.topk(all_snr, k=k, dim=0, largest=True, sorted=True)

            all_flux_snr = torch.cat([prev_flux_snr, cand_flux_snr], dim=0)
            new_flux_snr = torch.gather(all_flux_snr, 0, idx)
            all_iv_flux_snr = torch.cat([prev_iv_flux_snr, cand_iv_flux_snr], dim=0)
            new_iv_flux_snr = torch.gather(all_iv_flux_snr, 0, idx)
            all_alpha = torch.cat([prev_alpha, cand_alpha], dim=0)
            new_alpha = torch.gather(all_alpha, 0, idx)
            all_rate = torch.cat([prev_rate, cand_rate], dim=0)
            new_rate = torch.gather(all_rate, 0, idx)
            new_snr = torch.gather(all_snr, 0, idx)

            top_snr_cpu[:, :, x0:x1] = new_snr.to(output_dtype).cpu()
            top_flux_snr_cpu[:, :, x0:x1] = new_flux_snr.to(output_dtype).cpu()
            top_iv_flux_snr_cpu[:, :, x0:x1] = new_iv_flux_snr.to(output_dtype).cpu()
            top_alpha_cpu[:, :, x0:x1] = new_alpha.to(output_dtype).cpu()
            top_rate_idx_cpu[:, :, x0:x1] = new_rate.cpu()
            x0 = x1

        logger.debug(f"Low-mem shift {ir + 1}/{len(rates)} complete")

    return top_snr_cpu, top_flux_snr_cpu, top_iv_flux_snr_cpu, top_alpha_cpu, top_rate_idx_cpu


def topk_to_detections(
    top_snr, top_flux_snr, top_iv_flux_snr, top_alpha, top_rate_idx, rates, dtype=np.float16
):
    """Convert top-k cubes into detection table compatible with sns_utils.

    Columns: x, y, rate_idx, flux_snr, flux, det_snr, iv_flux_snr
    """
    if top_snr.shape[0] == 0:
        return np.zeros((0, 7), dtype=dtype)

    snr_np = np.array(top_snr, dtype=dtype)
    flux_snr_np = np.array(top_flux_snr, dtype=dtype)
    iv_flux_snr_np = np.array(top_iv_flux_snr, dtype=dtype)
    alpha_np = np.array(top_alpha, dtype=dtype)
    rate_idx_np = np.array(top_rate_idx, dtype=np.int32)

    k, A, B = snr_np.shape
    idx, idy = np.meshgrid(np.arange(B), np.arange(A))
    idx = idx.reshape(A * B)
    idy = idy.reshape(A * B)

    chunks = []
    for n in range(k):
        s = rate_idx_np[n].reshape(A * B)
        snr = snr_np[n].reshape(A * B)
        flux_snr = flux_snr_np[n].reshape(A * B)
        iv_flux_snr = iv_flux_snr_np[n].reshape(A * B)
        alpha = alpha_np[n].reshape(A * B)

        keep = (snr > 0) & (s >= 0)
        if not np.any(keep):
            continue

        nkeeps = np.zeros((keep.sum(), 7), dtype=dtype)
        nkeeps[:, 0] = idx[keep]
        nkeeps[:, 1] = idy[keep]
        nkeeps[:, 2] = s[keep]
        nkeeps[:, 3] = flux_snr[keep]
        nkeeps[:, 4] = alpha[keep]
        nkeeps[:, 5] = snr[keep]
        nkeeps[:, 6] = iv_flux_snr[keep]
        chunks.append(nkeeps)

    if not chunks:
        return np.zeros((0, 7), dtype=dtype)
    return np.concatenate(chunks)
