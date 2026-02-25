from astropy.table import Table
import logging
import numpy as np
import scipy as sci
import torch

NOMINAL_PIXEL_SCALE = 0.17  # arcsec/pixel
MAX_SHIFT_RATE = 4.5  # arcsec/hour
MAX_RATE_PIX_PER_DAY = 24*MAX_SHIFT_RATE/NOMINAL_PIXEL_SCALE


def get_device() -> torch.device:
    """Determine if a GPU available and select as torch device.

    Returns:
        torch.device: The device that torch will use for processing.
    """
    gpu_available = torch.cuda.is_available()
    message = "Using GPU" if gpu_available else "No GPU. Using CPU"
    logging.info(message)
    return torch.device("cuda:0" if gpu_available else "cpu")


def get_shift_rates(plants: Table,
                    fwhms: np.array,
                    dmjds: np.array,
                    rate_fwhm_grid_step: float):
    """
    get a grid of shift rates from the plant list
    """
    # rotation hack because the chip 24 rates are
    # all positive in x not negative
    swap_signs = False
    if np.sum(plants['rate_x'] > 0) == len(plants):
        plants['rate_x'] *= -1.
        plants['rate_y'] *= -1.
        swap_signs = True

    logging.debug(f'Number of planted sources: {len(plants)}')
    w = (plants['rate_x']**2 + plants['rate_y']**2) < MAX_RATE_PIX_PER_DAY**2
    angs = np.arctan2(plants['rate_y'][w],
                      plants['rate_x'][w]) % (2*np.pi)

    min_ang = np.min(angs)
    max_ang = np.max(angs)
    med_ang = np.median(angs)
    # bodge angle hack
    if min_ang < 0:
        while min_ang < 0:
            min_ang += 2*np.pi
            med_ang += 2*np.pi

    d_ang = max(max_ang-med_ang, med_ang-min_ang)
    max_ang = med_ang + d_ang
    min_ang = med_ang - d_ang
    logging.debug((f"Angles (min:{min_ang}, max: {max_ang}, "
                   f"med: {med_ang}, delta: {d_ang})"))

    W = np.abs(plants['rate_x']) < 200
    line = sci.stats.linregress(plants['rate_x'][W],
                                plants['rate_y'][W])

    max_x = np.max(plants['rate_x'] + 5)
    max_y = max_x*line.slope + line.intercept

    logging.debug(f"Max x/y: {max_x}, {max_y}")

    seeing = np.mean(fwhms)*NOMINAL_PIXEL_SCALE  # 0.7
    logging.debug(f'Mean FWHM {seeing}" ')
    seeing /= NOMINAL_PIXEL_SCALE  # pixels

    # dh = (mjds[-1]-mjds[0]) # days
    # days, need to take the np.max and np.min because
    # images aren't necessarily in order of increase time.
    dh = np.max(dmjds)
    drate = rate_fwhm_grid_step*seeing/dh  # 0.75 seems to be a good sweet spot
    ang_steps_h = np.linspace(med_ang, max_ang+0.0, 80)
    ang_steps_l = np.linspace(min_ang-0.0, med_ang, 80)

    rates = [[max_x, max_y]]
    current_rate = (max_x**2+max_y**2)**0.5
    while current_rate < MAX_RATE_PIX_PER_DAY:
        n_x = np.cos(ang_steps_h)*current_rate  # + max_x
        n_y = np.sin(ang_steps_h)*current_rate  # + max_y

        dist_rates = (((n_x - n_x[0])**2 + (n_y - n_y[0])**2)**0.5) / drate
        dist_rates = dist_rates.astype('int')
        unique_dist_rates = np.unique(dist_rates)
        for ind in unique_dist_rates:
            w = np.where(dist_rates == ind)
            rates.append([n_x[w[0][0]], n_y[w[0][0]]])

        n_x = np.cos(ang_steps_l[::-1]) * current_rate  # + max_x
        n_y = np.sin(ang_steps_l[::-1]) * current_rate  # + max_y
        dist_rates = (((n_x - n_x[0])**2 + (n_y - n_y[0])**2)**0.5) / drate
        dist_rates = dist_rates.astype('int')
        unique_dist_rates = np.unique(dist_rates)
        for ind in unique_dist_rates.astype('int'):
            if ind == 0:
                continue
            w = np.where(dist_rates == ind)
            rates.append([n_x[w[0][0]], n_y[w[0][0]]])

        current_rate += drate
    # the first rate is duplicated in the above algorithm
    rates = np.array(rates)[1:]
    logging.debug(f"Number of rates: {len(rates)}")

    if swap_signs:
        rates *= -1.
        plants['rate_x'] *= -1.
        plants['rate_y'] *= -1.

    return rates


def create_kernel(psfs, dmjds,
                  rates=None,
                  useNegativeWell=True,
                  useGaussianKernel=False,
                  kernel_width=14, im_nums=None):
    if psfs is None and not useGaussianKernel:
        logging.error("Set useGaussianKernal when no psfs provided")
        raise ValueError("Set useGaussianKernal when no psfs provided")
    mean_rate = np.mean(rates, axis=0)
    logging.debug(f"Creating kernel for rates: {mean_rate}")
    device = get_device()
    if useGaussianKernel:
        logging.debug("Using a Gaussian Kernel")
        # kernel_width = 10
        std = 1.5
        khw = kernel_width//2
        (x, y) = np.meshgrid(np.arange(kernel_width),
                             np.arange(kernel_width))
        gauss = np.exp(-((x-khw-0.5)**2 + (y-khw-0.5)**2)/(2*std*std))
        gauss /= np.sum(gauss)

        kernel = torch.tensor(
            np.zeros((1, 1, len(dmjds), kernel_width, kernel_width),
                     dtype='float32')
            ).to(device)  # .cuda()
        for ir in range(len(dmjds)):
            kernel[0, 0, ir, :, :] = torch.tensor(np.copy(gauss))

    else:
        logging.debug('Using PSF kernel')
        # kernel_width = 1000
        # for i in range(len(psfs)):
        #    kernel_width = min(kernel_width, psfs[i].shape[0])
        # khw = kernel_width//2
        # using kernel widths between 10 and 30 doesn't
        # produce much different outputs in terms of depth
        # kernel_width = 14
        khw = kernel_width//2

        kernel = torch.tensor(
            np.zeros((1, 1, len(psfs), kernel_width, kernel_width),
                     dtype='float32')
            ).to(device)
        for ir in range(len(psfs)):
            psf = psfs[ir]
            (a, b) = psf.shape
            delt = (a-kernel_width)//2

            psf_section = psf[delt:delt+kernel_width, delt:delt+kernel_width]
            psf_section /= np.sum(psf_section)

            kernel[0, 0, ir, :, :] = torch.tensor(np.copy(psf_section))

    if useNegativeWell:
        mean_kernel = torch.sum(kernel[0, 0], 0)
        mean_kernel /= torch.sum(mean_kernel)

        c = torch.zeros_like(kernel)
        mid_im = len(psfs)//2
        DMJDS = dmjds-dmjds[mid_im]

        for id in range(0, len(psfs)):
            shifts = (int(-np.round(DMJDS[id]*mean_rate[1])),
                      int(-np.round(DMJDS[id]*mean_rate[0])))
            logging.debug(f"shifts for negative wells: {shifts}")
            if (abs(shifts[0]) < khw) & (abs(shifts[1]) < khw):
                c[0, 0, id,] = torch.roll(mean_kernel,
                                          shifts=shifts,
                                          dims=[0, 1])
        trail = torch.sum(c[0, 0], 0)
        trail /= torch.sum(trail)*3.

        for id in range(len(psfs)):
            kernel[0, 0, id] -= trail

    return kernel
