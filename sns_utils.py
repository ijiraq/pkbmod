from astropy.io import fits
import gc
import logging
import numpy as np
import time
import torch
from sns_data_nh import get_device


def run_shifts(datas, inv_variances, rates, dmjds, min_snr,
               writeTestImages=False):
    n_im = len(datas[0, 0, :])
    logging.debug(f'NUM IM {n_im}')
    c = torch.zeros_like(datas)
    c[0, 0, 0] = datas[0, 0, 0]
    cv = torch.zeros_like(datas)
    cv[0, 0, 0] = inv_variances[0, 0, 0]

    snr_image = torch.zeros((1, 1,
                             len(rates), datas.size()[3], datas.size()[4]),
                            dtype=torch.float16)
    alpha_image = torch.zeros((1, 1,
                               len(rates), datas.size()[3], datas.size()[4]),
                              dtype=torch.float32)

    # rates=[[-200.6777606841237, 78.88276756451387]]
    for ir in range(len(rates)):
        for id in range(1, n_im):
            shifts = (-round(dmjds[id]*rates[ir][1]),
                      -round(dmjds[id]*rates[ir][0]))
            c[0, 0, id,] = torch.roll(datas[0, 0, id],
                                      shifts=shifts, dims=[0, 1])
            cv[0, 0, id] = torch.roll(inv_variances[0, 0, id],
                                      shifts=shifts, dims=[0, 1])
        # C = functional.conv3d(c, kernel)
        # sums = torch.sum(functional.conv3d(c, ones,padding='same'), 2)

        # these are set abov
        PSI = c
        PHI = cv

        # median_alpha = torch.median(PSI/PHI, dim=2)[0]
        # doesn't seem to be used for anything

        alpha = torch.nansum(PSI/PHI, 2)  # flux estimate
        alpha = torch.nan_to_num(alpha, 0.0)

        # SNR estimate
        nu = torch.sum(PSI, 2)/torch.pow(torch.sum(PHI, 2), 0.5)
        nu = torch.nan_to_num(nu, -1.0)

        # check to see if making all snr==inf go to zero fixes the
        # infinities error you are getting
        nu[nu == float("Inf")] = 0
        #

        where = nu[0, 0] > min_snr
        inds = where.nonzero()

        snr_image[0, 0, ir, inds[:, 0], inds[:, 1]] = (
            nu[0, 0, inds[:, 0], inds[:, 1]].cpu().type(torch.float16))
        # flux per image not per stack
        alpha_image[0, 0, ir, inds[:, 0], inds[:, 1]] = (
            alpha[0, 0, inds[:, 0], inds[:, 1]].cpu()/n_im)

        gc.collect()
        torch.cuda.empty_cache()

        # wes testing
        if ir == 19 and writeTestImages:
            fits.PrimaryHDU(np.array(nu.cpu())
                            ).writeto('snr_test.fits', overwrite=True)

    logging.info(f'Max per image flux of candidates: {torch.max(alpha_image)}')
    logging.info(f'Max per image snr of candidates: {torch.max(snr_image)}')

    del c, cv, datas, inv_variances, PSI, PHI
    gc.collect()
    torch.cuda.empty_cache()

    return snr_image, alpha_image


def trim_negative_snr(snr_image, alpha_image, sort_inds,
                      n_keep, rates, A, B, use_index=False):
    # trim the negative SNR sources. The reason these show up is
    # because the likelihood formalism sucks
    idx, idy = np.meshgrid(np.arange(B), np.arange(A))
    idx = idx.reshape(A*B)
    idy = idy.reshape(A*B)
    for n in range(n_keep):
        s = sort_inds[0, 0, n, :, :].reshape(A*B)
        SNR = snr_image[0, 0, s, idy, idx]
        alpha = alpha_image[0, 0, s, idy, idx]

        where = SNR > 0
        inds = where.nonzero()[:, 0]
        logging.debug(f"keep index length: {inds.shape}")
        logging.debug(f"length of indexs: {(s[inds]).shape}")
        if n == 0:
            keeps = np.zeros((len(inds), 7), dtype='float32')
            keeps[:, 0] = idx[inds]
            keeps[:, 1] = idy[inds]
            if use_index:
                keeps[:, 2] = s.reshape(A*B)[inds]
                keeps[:, 3] = 0.0
            else:
                keeps[:, 2] = rates[s.reshape(A*B)[inds], 0]
                keeps[:, 3] = rates[s[inds], 1]
            keeps[:, 4] = alpha.reshape(A*B)[inds]
            keeps[:, 5] = SNR.reshape(A*B)[inds]
        else:
            nkeeps = np.zeros((len(inds), 7), dtype='float32')
            logging.debug(f"Keeps size: {nkeeps.shape}")
            nkeeps[:, 0] = idx[inds]
            nkeeps[:, 1] = idy[inds]
            if use_index:
                nkeeps[:, 2] = s[inds]
                nkeeps[:, 3] = 0.0
            else:
                nkeeps[:, 2] = rates[s.reshape(A*B)[inds], 0]
                nkeeps[:, 3] = rates[s[inds], 1]
            nkeeps[:, 4] = alpha.reshape(A*B)[inds]
            nkeeps[:, 5] = SNR.reshape(A*B)[inds]
            keeps = np.concatenate([keeps, nkeeps])

    logging.info(f'Keeping {len(keeps)} candidates')

    detections = np.array(keeps)
    del keeps, idx, idy
    return detections


def trim_negative_flux(detections):
    pos = np.where(detections[:, 4] > 0)
    detections = detections[pos]
    logging.info(f'Keeping {len(detections)} positive flux candidates')
    return detections


def brightness_filter(im_datas, inv_vars, c, cv, kernel,
                      dmjds, rates, detections, khw, n_im,
                      n_bright_test=10, test_high=1.15, test_low=0.85):

    device = get_device()
    nb_ref = torch.tensor(10.0**np.linspace(np.log10(test_low),
                                            np.log10(test_high),
                                            n_bright_test)).to(device)

    for ir in range(len(rates)):
        t1 = time.time()
        w = np.where((detections[:, 2] == rates[ir][0]) &
                     (detections[:, 3] == rates[ir][1]))

        for id in range(1, n_im):
            shifts = (-round(dmjds[id]*rates[ir][1]),
                      -round(dmjds[id]*rates[ir][0]))
            c[0, 0, id] = torch.roll(im_datas[0, 0, id],
                                     shifts=shifts, dims=[0, 1])
            cv[0, 0, id] = torch.roll(inv_vars[0, 0, id],
                                      shifts=shifts, dims=[0, 1])

        arg_mins = torch.zeros(len(detections), dtype=torch.uint32)
        for id in w[0]:
            (x, y) = detections[id, :2]
            x = int(x) + khw
            # array of scaled brightnesses in steps of brightness*test_low
            # to brightness*test_high
            y = int(y) + khw
            nb = nb_ref*detections[id, 4]
            k = kernel.repeat((1, n_bright_test, 1, 1, 1))
            for ib in range(nb.size()[0]):
                k[:, ib, :, :, :] *= nb[ib]

            diff = c[:, :, :, y-khw:y+khw, x-khw:x+khw].repeat(
                (1, n_bright_test, 1, 1, 1))
            diff -= k
            diff = diff*diff
            diff *= cv[:, :, :, y-khw:y+khw, x-khw:x+khw].repeat(
                (1, n_bright_test, 1, 1, 1))

            tmp = torch.sum(diff, (0, 2, 3, 4))
            arg_mins[id] = torch.argmin(tmp)

        arg_mins_cpu = arg_mins.cpu()

        W = np.where((arg_mins_cpu != 0) & (arg_mins_cpu != (n_bright_test-1)))
        logging.debug((f'{ir+1}/{len(rates)}, pre: {len(w[0])}, '
                       f'post: {len(W[0])},  in time {time.time()-t1}'))
        if ir == 0:
            keeps = W[0]
        else:
            keeps = np.concatenate([keeps, W[0]])
    logging.info((f'Number kept after brightness filter {len(keeps)}'
                  f' of {len(detections)} total detections.'))

    return keeps


def create_stamps(im_datas, im_masks, c, cv, dmjds, rates,
                  filt_detections, khw,
                  exact_check=False, inexact_rtol=1.e-7, use_index=False):
    mean_stamps = []
    indices = []
    # saved = False
    for ir in range(len(rates)):

        # these are required to reset from the nans below
        c[0, 0, 0] = im_datas[0, 0, 0]
        cv[0, 0, 0] = im_masks[0, 0, 0]

        # t1 = time.time()
        if use_index:
            w = np.where(np.round(filt_detections[:, 2]).astype("int") == ir)
        elif exact_check:
            w = np.where((filt_detections[:, 2] == rates[ir][0]) &
                         (filt_detections[:, 3] == rates[ir][1]))
        else:
            w = np.where((np.isclose(filt_detections[:, 2],
                                     rates[ir][0],
                                     rtol=inexact_rtol)) &
                         (np.isclose(filt_detections[:, 3],
                                     rates[ir][1],
                                     rtol=inexact_rtol)))

        for id in range(1, len(dmjds)):
            shifts = (-round(dmjds[id]*rates[ir][1]),
                      -round(dmjds[id]*rates[ir][0]))
            c[0, 0, id] = torch.roll(im_datas[0, 0, id],
                                     shifts=shifts, dims=[0, 1])
            # mask values with 1 are GOOD pixels
            cv[0, 0, id] = torch.roll(im_masks[0, 0, id],
                                      shifts=shifts, dims=[0, 1])
        mean_stamp_frame = torch.sum(c, 2)

        mask_frame = torch.sum(cv, 2)

        for iw in w[0]:
            x, y = filt_detections[iw, :2].astype('int')
            mean_stamp = mean_stamp_frame[0, 0, y:y+khw*2+1, x:x+khw*2+1]

            """
            med_sections = c[0,0,:,y:y+khw*2+1, x:x+khw*2+1].cpu()
            mask_sections = cv[0,0,:,y:y+khw*2+1, x:x+khw*2+1].cpu()
            mask_sections[np.where((np.isnan(mask_sections)) |
                                   (np.isinf(mask_sections)))] = 0

            # mask_sections==0 are bad pixels
            # numpy masks value == 1 are bad pixels
            masked_med_sections = ma.array(med_sections, mask= 1-mask_sections)
            med_stamps.append(ma.median(masked_med_sections,axis=0).filled(0.0))
            """

            mask = np.copy(mask_frame[0, 0, y:y+khw*2+1, x:x+khw*2+1].cpu())
            mask[np.where((np.isnan(mask)) | (np.isinf(mask)))] = 0.0
            np.clip(mask, 1., len(dmjds), out=mask)
            mean_stamps.append(np.copy(mean_stamp.cpu())/mask)

            indices.append(iw)

    indices = np.array(indices)
    mean_stamps = np.array(mean_stamps)[indices]

    del mask, mean_stamp, mean_stamp_frame, mask_frame
    gc.collect()
    torch.cuda.empty_cache()

    return mean_stamps


# trim the ones with peak offset more than peak_offset_max pixels
def peak_offset_filter(stamps, filt_detections, peak_offset_max):
    (N, a, b) = stamps.shape
    (gx, gy) = np.meshgrid(np.arange(b), np.arange(a))
    gx = gx.reshape(a*b)
    gy = gy.reshape(a*b)
    rs_stamps = stamps.reshape(N, a*b)
    args = np.argmax(rs_stamps, axis=1)
    X = gx[args]
    Y = gy[args]
    radial_d = ((X-b/2)**2+(Y-a/2)**2)**0.5
    w = np.where(radial_d < peak_offset_max)
    filt_detections = filt_detections[w]
    stamps = stamps[w]

    return stamps, filt_detections


# do predictive line clustering
def predictive_line_cluster(filt_detections, stamps, dmjds, dist_lim,
                            min_samp=2, init_select_proc_distance=60):

    proc_filt_detections = np.copy(filt_detections)

    proc_inds = np.arange(len(proc_filt_detections))
    clust_detections, clust_inds = [], []

    while len(proc_filt_detections) > 0:

        arg_max = np.argmax(proc_filt_detections[:, 5])  # 5 - max on SNR
        x_o, y_o, rx_o, ry_o, f_o, snr_o = proc_filt_detections[arg_max, :6]

        # this secondary where command is necessary because of memory
        # overflows in large detection lists
        w = np.where((proc_filt_detections[:, 0] >
                      proc_filt_detections[arg_max, 0]-55)
                     & (proc_filt_detections[:, 0] <
                        proc_filt_detections[arg_max, 0] +
                        init_select_proc_distance)
                     & (proc_filt_detections[:, 1] >
                        proc_filt_detections[arg_max, 1]-55)
                     & (proc_filt_detections[:, 1] <
                        proc_filt_detections[arg_max, 1] +
                        init_select_proc_distance))

        W = np.where(((proc_filt_detections[w[0], 0] -
                       proc_filt_detections[arg_max, 0])**2 + (
                           proc_filt_detections[w[0], 1] -
                           proc_filt_detections[arg_max, 1])**2) <
                     init_select_proc_distance**2)
        w = w[0][W[0]]

        fd_subset = proc_filt_detections[w]

        drx = fd_subset[:, 2] - rx_o
        dry = fd_subset[:, 3] - ry_o
        dt = dmjds  # just for clarity

        # predicted centroid  position of secondary detection shifted at the
        # differential wrong rate.
        x_n, y_n = x_o - drx*dt[-1], y_o - dry*dt[-1]
        # predicted centroid shifted such that best detection is now at origin
        dx, dy = (x_n - x_o), (y_n - y_o)

        dxp = dx*fd_subset[:, 1]
        dyp = dy*fd_subset[:, 0]
        xm = x_n*y_o
        ym = y_n*x_o
        dx2 = dx**2
        dy2 = dy**2
        top = np.abs(dyp - dxp + xm - ym)
        bottom = np.sqrt(dx2 + dy2)
        dist = top/bottom

        clust = np.where((dist < dist_lim) |
                         (np.isnan(dist)) |
                         ((dist < dist_lim) &
                          (drx == 0) & (dry == 0)))
        if len(clust[0]) >= min_samp:
            clust_detections.append(proc_filt_detections[arg_max])
            clust_inds.append(proc_inds[arg_max])

        mask = np.ones(len(proc_filt_detections), dtype='bool')
        mask[w[clust]] = False
        proc_filt_detections = proc_filt_detections[mask]
        proc_inds = proc_inds[mask]

    clust_detections = np.array(clust_detections)
    clust_stamps = stamps[np.array(clust_inds)]

    return clust_detections, clust_stamps


def position_filter(clust_detections, clust_stamps, im_datas,
                    inv_vars, c, cv, kernel,
                    dmjds, rates, khw, n_offsets=5,
                    exact_check=True, inexact_rtol=1.e-7, use_index=False):

    # now apply a positional filter on the clust_detections to see
    # if the likelihood minimimum is near the centre
    # n_offsets = 5 # +- n_offsets in x and y
    n_o = n_offsets*2+1

    k = kernel.repeat((1, n_o*n_o, 1, 1, 1))

    danger_edges = []
    # these are the indices of the maximal offsets in x and y
    for iy in range(n_o):
        for ix in range(n_o):
            i = iy*n_o+ix
            shifts = (0, iy-n_offsets, ix-n_offsets)
            k[0, i, :, :, :] = torch.roll(kernel[0, 0],
                                          shifts=shifts,
                                          dims=[0, 1, 2])
            if iy == 0 or iy == n_o-1 or ix == 0 or ix == n_o-1:
                danger_edges.append(i)

    cv[0, 0, 0] = inv_vars[0, 0, 0]

    keeps = []
    for ir in range(len(rates)):
        if use_index:
            w = np.where(clust_detections[:, 2] == ir)
        elif exact_check:
            w = np.where((clust_detections[:, 2] == rates[ir][0]) &
                         (clust_detections[:, 3] == rates[ir][1]))
        else:
            w = np.where((np.isclose(clust_detections[:, 2],
                                     rates[ir][0],
                                     rtol=inexact_rtol)) &
                         (np.isclose(clust_detections[:, 3],
                                     rates[ir][1],
                                     rtol=inexact_rtol)))
        if len(w[0]) == 0:
            continue

        for id in range(1, len(dmjds)):
            shifts = (-round(dmjds[id]*rates[ir][1]),
                      -round(dmjds[id]*rates[ir][0]))
            c[0, 0, id] = torch.roll(im_datas[0, 0, id],
                                     shifts=shifts,
                                     dims=[0, 1])
            cv[0, 0, id] = torch.roll(inv_vars[0, 0, id],
                                      shifts=shifts,
                                      dims=[0, 1])

        for id in w[0]:

            (x, y) = clust_detections[id, :2]
            x = int(x)  # +khw
            y = int(y)  # +khw

            K = k*clust_detections[id, 4]

            diff = c[:, :, :, y:y+khw*2, x:x+khw*2].repeat(
                (1, n_o*n_o, 1, 1, 1))
            diff -= K
            diff = diff**2
            diff *= cv[:, :, :, y:y+khw*2, x:x+khw*2].repeat(
                (1, n_o*n_o, 1, 1, 1))
            arg_min = torch.argmin(torch.sum(diff, (0, 2, 3, 4)))
            min_ix = arg_min % n_o
            min_iy = int((arg_min-min_ix)/n_o)

            min_ix -= n_offsets
            min_iy -= n_offsets
            if arg_min not in danger_edges:
                keeps.append(id)

    keeps = np.array(keeps)
    grid_detections = clust_detections[keeps]
    grid_stamps = clust_stamps[keeps]
    logging.debug(len(grid_detections), len(clust_detections))
    logging.debug(grid_detections[0])
    logging.debug(clust_detections[0])
    logging.info((f'Number of sources kept after the positional'
                  f' grid minimum search: {len(grid_detections)}.'))

    return grid_detections, grid_stamps


def brightness_filter_fast(im_datas, inv_vars, c, cv, kernel,
                           dmjds, rates, detections, khw, n_im,
                           n_bright_test=10, test_high=1.15,
                           test_low=0.85,
                           exact_check=True, inexact_rtol=1.e-7,
                           use_index=False,
                           n_det_iter=200):

    device = get_device()

    nb_ref = torch.tensor(10.0**np.linspace(np.log10(test_low),
                                            np.log10(test_high),
                                            n_bright_test)).to(device)

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

    for ir in range(len(rates)):

        t1 = time.time()
        if use_index:
            W = np.where(detections[:, 2] == ir)
        elif exact_check:
            W = np.where((detections[:, 2] == rates[ir][0]) &
                         (detections[:, 3] == rates[ir][1]))
        else:
            W = np.where((np.isclose(detections[:, 2],
                                     rates[ir][0],
                                     rtol=inexact_rtol)) &
                         (np.isclose(detections[:, 3],
                                     rates[ir][1],
                                     rtol=inexact_rtol)))

        if len(W[0]) == 0:
            continue

        # Roll images for this rate
        for id in range(1, n_im):
            shifts = (-round(dmjds[id]*rates[ir][1]),
                      -round(dmjds[id]*rates[ir][0]))
            c[0, 0, id] = torch.roll(im_datas[0, 0, id],
                                     shifts=shifts,
                                     dims=[0, 1])
            cv[0, 0, id] = torch.roll(inv_vars[0, 0, id],
                                      shifts=shifts,
                                      dims=[0, 1])

        n_done_iter = 0
        while n_done_iter < len(W[0])-1:

            w = W[0][n_done_iter:min(n_done_iter+n_det_iter, len(W[0]))]

            det_idx = w
            n_det = len(det_idx)
            if n_det == 0:
                continue

            # Extract coordinates for all detections at this rate
            xs = detections[det_idx, 0].astype(int) + khw
            ys = detections[det_idx, 1].astype(int) + khw
            fluxes = torch.tensor(detections[det_idx, 4],
                                  dtype=torch.float64,
                                  device=device)

            # Batch-extract all patches via advanced indexing
            c_3d = c[0, 0]    # (n_im, H, W)
            cv_3d = cv[0, 0]  # (n_im, H, W)

            ys_t = torch.tensor(ys, device=device, dtype=torch.long)
            xs_t = torch.tensor(xs, device=device, dtype=torch.long)

            y_idx = ys_t[:, None] - khw + dy[None, :]  # (n_det, ks)
            x_idx = xs_t[:, None] - khw + dx[None, :]  # (n_det, ks)

            # Expand indices for (n_det, n_im, ks, ks) gather
            y_exp = y_idx[:, None, :, None].expand(n_det, n_im, ks, ks)
            x_exp = x_idx[:, None, None, :].expand(n_det, n_im, ks, ks)
            im_exp = im_idx[None, :, None, None].expand(n_det, n_im, ks, ks)

            patches_c = c_3d[im_exp, y_exp, x_exp]   # (n_det, n_im, ks, ks)
            patches_cv = cv_3d[im_exp, y_exp, x_exp]   # (n_det, n_im, ks, ks)

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

            if n_done_iter == 0:
                kept_idx = det_idx[np.where(valid)]
            else:
                kept_idx = np.concatenate([kept_idx, det_idx[np.where(valid)]])

            n_done_iter += len(w)
            del diff
            gc.collect()

            logging.info((f'{ir+1}/{len(rates)}, '
                          f'vx: {str(rates[ir][0])[:7]}, '
                          f'vy: {str(rates[ir][1])[:7]}, '
                          f'pre: {len(W[0])}, '
                          f'post: {len(kept_idx)},  '
                          f'in time {time.time()-t1}'))

        if len(kept_idx) > 0:
            keeps_list.append(kept_idx)
    if keeps_list:
        keeps = np.concatenate(keeps_list)
    else:
        keeps = np.array([], dtype=np.intp)

    logging.info((f'Number kept after brightness filter {len(keeps)} '
                  f'of {len(detections)} total detections.'))
    print((f'Number kept after brightness filter {len(keeps)} '
           f'of {len(detections)} total detections.'))

    return keeps
