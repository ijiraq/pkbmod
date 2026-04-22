import torch
from astropy.io import fits
import glob
from astropy.wcs import WCS
import numpy as np, scipy as sci, pylab as pyl
from sns_utils import rots

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_data(patch_id, image_path, variance_trim, bit_mask, verbose=False, var_trim_keyword='SAT'):
    """
    Read in all the requesite image data
    """

    datas, masks, variances, mjds, psfs, fwhms, im_nums = [], [], [], [], [], [], []
    #fits_files = glob.glob(f'{image_path}/*_{patch_id}_*special*.repro.fits')
    fits_files = glob.glob(f'{image_path}/*_{patch_id}_*.repro')
    
    indices = []
    for i in range(len(fits_files)):
        s = fits_files[i].split('/')
        ind = s[-1].split('_')[0]
        indices.append(int(float(ind)))
    indices = np.array(indices)
    args = np.argsort(indices)
    fits_files = np.array(fits_files)[args]

    if len(fits_files)==0:
        print(f'Cannot find any warps at {visit}.')
        exit(1)
    else:
        print(f'Reading {len(fits_files)} files from {image_path}/')

    for i in range(20):#len(fits_files)):
        with fits.open(fits_files[i]) as han:

            datas.append(han[0].data)
            masks.append(han[2].data)
            variances.append(han[1].data)

            if i ==0:
                wcs = WCS(han[1].header)
                
        im_num = i
        mjd = han[0].header['MJD'] 
        mjds.append(mjd + 30./(24.*3600./2.))

        fwhms.append(1.0)
        im_nums.append(im_num)

        nan_med_variance = np.nanmedian(variances[-1])
        print(mjds[i], fits_files[i], nan_med_variance)
        w = np.where(variances[-1]>variance_trim*nan_med_variance)
        masks[-1][w] += 2**bit_mask[var_trim_keyword]
    print(f'Read in {len(datas)} images.\n')

    
    return (datas, masks, variances, mjds, psfs, fwhms, im_nums, wcs)


def get_shift_rates(ecl_ang, mjds, rate_lims, ang_lims, fwhm, pix_scale, rate_fwhm_grid_step, save_rates_figure=False):
    """                                                                                                                                   
    get a grid of shift rates from the planted classy imagery                                                                             
    """

    d2r = np.pi/180.

    min_ang = (ecl_ang + np.min(np.array(ang_lims)))*d2r
    max_ang = (ecl_ang + np.max(np.array(ang_lims)))*d2r
    med_ang = (min_ang+max_ang)/2.

    # bodge angle hack                                                                                                                    
    if min_ang<0:
        while min_ang<0:
            min_ang+=2*np.pi
            med_ang+=2*np.pi

    d_ang = max(max_ang-med_ang, med_ang-min_ang)
    print('Angles (min, max, med, delta):', min_ang, max_ang, med_ang, d_ang)


    seeing = fwhm
    print(f'Mean FWHM {seeing}" ', end='')
    seeing /= pix_scale # pixels                                                                                                          
    print(f'{seeing} pix.')


    dh = (np.max(mjds)-np.min(mjds)) # days, need to take the np.max and np.min because images aren't necessarily in order of increase time.                                                                                                                                       
    drate = rate_fwhm_grid_step*seeing/dh  # 0.75 seems to be a good sweet spot                                                           

    ang_steps_h = np.linspace(med_ang, max_ang+0.0, 150)
    ang_steps_l = np.linspace(min_ang-0.0, med_ang, 150)

    rx,ry = np.cos(ecl_ang*d2r)*np.min(np.array(rate_lims)), np.sin(ecl_ang*d2r)*np.min(np.array(rate_lims))
    rates = [[rx, ry]]

    current_rate = np.min(np.array(rate_lims))
    while current_rate < 24*np.max(np.array(rate_lims))/pix_scale:
        n_x = np.cos(ang_steps_h)*current_rate# + max_x                                                                                   
        n_y = np.sin(ang_steps_h)*current_rate# + max_y                                                                                   

        dist_rates = ( ((n_x - n_x[0])**2 + (n_y - n_y[0])**2)**0.5 / drate).astype('int')
        unique_dist_rates = np.unique(dist_rates)
        for ind in unique_dist_rates:
            w = np.where(dist_rates == ind)
            rates.append([n_x[w[0][0]], n_y[w[0][0]]])


        n_x = np.cos(ang_steps_l[::-1])*current_rate# + max_x                                                                             
        n_y = np.sin(ang_steps_l[::-1])*current_rate# + max_y                                                                             
        dist_rates = (((n_x - n_x[0])**2 + (n_y - n_y[0])**2)**0.5 / drate).astype('int')
        unique_dist_rates = np.unique(dist_rates)
        for ind in unique_dist_rates:
            if ind == 0: continue
            w = np.where(dist_rates == ind)
            rates.append([n_x[w[0][0]], n_y[w[0][0]]])

        current_rate += drate

    rates = np.array(rates)[1:] ## the first entry is duplicated


    print('Number of rates:', len(rates))

    if save_rates_figure:
        fig = pyl.figure(1)
        sp = fig.add_subplot(111)
        pyl.scatter(rates[:,0], rates[:,1], alpha = 0.5, marker='s', s=70)

        pyl.grid(linestyle=':')

        print('Saving Rates Figure.')
        pyl.savefig('rates_figure.png')
        exit()


    return np.array(rates)



def create_kernel(psfs, dmjds, rates, useNegativeWell=True, useGaussianKernel=False, kernel_width=14, im_nums=None, visit=None):
    if useGaussianKernel:
        print("Using a Gaussian Kernel")
        #kernel_width = 10
        std = 1.5
        khw = kernel_width//2
        (x,y) = np.meshgrid(np.arange(kernel_width), np.arange(kernel_width))
        gauss = np.exp(-((x-khw-0.5)**2 + (y-khw-0.5)**2)/(2*std*std))
        gauss/=np.sum(gauss)
        #print(gauss)


        kernel = torch.tensor(np.zeros((1, 1, len(dmjds), kernel_width, kernel_width),dtype='float32')).to(device)#.cuda()
        for ir in range(len(dmjds)):
            kernel[0,0,ir,:,:] = torch.tensor(np.copy(gauss))

    else:
        print('Using PSF kernel')
        #kernel_width = 1000
        #for i in range(len(psfs)):
        #    kernel_width = min(kernel_width, psfs[i].shape[0])
        #khw = kernel_width//2
        kernel_width = 14 # using kernel widths between 10 and 30 doesn't produce much different outputs in terms of depth
        khw = kernel_width//2

        kernel = torch.tensor(np.zeros((1, 1, len(psfs), kernel_width, kernel_width),dtype='float32')).to(device) #.cuda()
        for ir in range(len(psfs)):
            psf = psfs[ir]
            (a,b) = psf.shape
            delt = (a-kernel_width)//2

            psf_section = psf[delt:delt+kernel_width, delt:delt+kernel_width]
            psf_section /= np.sum(psf_section)

            kernel[0,0,ir,:,:] = torch.tensor(np.copy(psf_section))

            #pyl.imshow(kernel[0,0,ir].cpu())
            #break

    if useNegativeWell:
        """
        with open(f'/home/fraserw/arc/projects/classy/visitLists/{visit}/{visit}_template_visit_list.txt') as han:
            data = han.readlines()
        template_inds = []
        for i in range(len(data)):
            imn = int(float(data[i].split()[0]))
            for j in range(len(im_nums)):
                if im_nums[j]==imn:
                    template_inds.append(j)
                    break

        c = torch.zeros_like(kernel[0,0,0])
        experimenting with a variable kernel
        """
        mean_kernel = torch.sum(kernel[0,0], 0)
        mean_kernel /= torch.sum(mean_kernel)

        mean_rate = np.mean(rates, axis=0)

        c = torch.zeros_like(kernel)
        mid_im = len(psfs)//2
        DMJDS = dmjds-dmjds[mid_im]

        for id in range(0, len(psfs)):
            shifts = (-round(DMJDS[id]*mean_rate[1]), -round(DMJDS[id]*mean_rate[0]))
            if abs(shifts[0])<khw and abs(shifts[1])<khw:
                c[0,0,id,] = torch.roll(mean_kernel, shifts=shifts, dims=[0,1])
        trail = torch.sum(c[0,0], 0)
        trail /=torch.sum(trail)*3.

        for id in range(len(psfs)):
            kernel[0,0,id] -= trail

    return kernel
