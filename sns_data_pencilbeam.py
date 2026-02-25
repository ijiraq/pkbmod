from astropy.io import fits
from astropy.wcs import WCS
import glob
import numpy as np, pylab as pyl
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_data(visit, warps_dir, variance_trim, bit_mask, var_trim_keyword='BAD', verbose=False, filelist = None):
    """
    Read in all the requesite image data
    """

    datas, masks, variances, mjds, psfs,  = [], [], [], [], []
    if verbose:
        if filelist is not None:
            print(f'Reading files from list {filelist}.')
        else:
            print(f'Reading images from {warps_dir}/{visit}/*fits')

    if filelist is None:
        fits_files = glob.glob(f'{warps_dir}/{visit}/*fits')
        fits_files.sort()
        
        if len(fits_files)==0:
            print(f'Cannot find any warps at {visit}.')
            exit(1)

    else:
        fits_files = []
        with open(filelist) as han:
            data = han.readlines()
        
        for i in range(len(data)):
            fits_files.append(data[i].split()[0])
            

    for i in range(len(fits_files)):
        with fits.open(fits_files[i]) as han:

            datas.append(han[1].data)
            masks.append(han[2].data)
            #variances.append(han[3].data**2) ## the JWST err extension is supposed to be STD but it seems the vartiance planes haven't been updated
            (l,h) = np.percentile(han[1].data[np.where(~np.isnan(han[1].data))], np.array([16,84]))
            variances.append(datas[-1]*0.0+((h-l)/2.)**2)
            header = han[0].header
            if i ==0:
                wcs = WCS(han[1].header)

        mjds.append(header['EXPMID'])
        #w = np.where(variances[-1]>variance_trim*np.nanmedian(variances[-1]))
        #masks[-1][w] += 2**bit_mask[var_trim_keyword]
        w = np.where(np.isnan(datas[-1]))
        masks[-1][w] += 2**bit_mask[var_trim_keyword]
        datas[-1][w] = 0.

        #im = fits.PrimaryHDU(datas[-1])
        #im.writeto(f'{i}.fits', overwrite=True)
    im_nums = np.arange(len(mjds))
    return (datas, masks, variances, mjds, im_nums, wcs)



def get_shift_rates(ref_wcs, mjds, visit, spacing = 70., rate_lims_custom = None):
    """
    get a grid of shift rates from the planted classy imagery
    """
    # if epoch==1, rates should be circular, -1000 to 1000, -2000 to 2000 pix per day
    # if epoch==2, rates should be circular, -1000 to 1000 pix per day
    # if epoch==3, rates should be circular, -1000 to 1000 pix per day

    # spacing sould be ~125 pix per day, which corresponds to roughly 1 pixel in the single epoch start to finish

    if rate_lims_custom is not None:
        rate_lims = np.array(rate_lims_custom)
        mx = np.max(np.abs(rate_lims[:,0]))
        my = np.max(np.abs(rate_lims[:,1]))
        max_rate = (mx**2 + my**2)**0.5
    elif visit[1]=='1':
        rate_lims = [[-1000., 1000.], [-2000., 2000.]]
        max_rate = 2000
    elif visit[1]=='2':
        rate_lims = [[-1000., 1000.], [-1000., 1000.]]
        max_rate = 1000
    elif visit[1]=='3':
        rate_lims = [[-1000., 1000.], [-1000., 1000.]]
        max_rate = 1000

    plants_dir = '/arc/projects/jwst-tnos/planted/implants_v4'
    with open(f'{plants_dir}/implant_converts_epoch{visit[1]}_dither1.csv') as han:
        data = han.readlines()
    plant_rates = []
    for i in range(1, len(data)):
        s = data[i].split()
        plant_rates.append([float(s[8]), float(s[9])]) # already in pix/day

    plant_rates = np.array(plant_rates)

    mean_rate = np.mean(plant_rates, axis=0) if rate_lims_custom is None else [0.0,0.0]

    dx,dy = np.meshgrid(np.arange(rate_lims[0][0], rate_lims[0][1]+spacing, spacing), np.arange(rate_lims[1][0], rate_lims[1][1]+spacing, spacing))
    (a,b) = dx.shape
    dx,dy = dx.reshape(a*b), dy.reshape(a*b)

    r2 = dx**2+dy**2
    w = np.where(r2<max_rate**2)
    dx,dy = dx[w],dy[w]

    rates = np.zeros((len(dx), 2), dtype='float')
    rates[:,0] = dx+mean_rate[0]
    rates[:,1] = dy+mean_rate[1]


    return (rates, plant_rates)


    pyl.scatter(plant_rates[:,0], plant_rates[:,1], zorder = 10)
    pyl.scatter(rates[:,0], rates[:,1], zorder=1, marker='s', alpha=0.5)
    pyl.grid(linestyle=':')
    pyl.savefig('junk.png')
    exit()



    return (rates, plant_rates)


def create_kernel(n_im, useNegativeWell=False, useGaussianKernel=False, kernel_width=6, visit=None):
    if useGaussianKernel:
        print("Using a Gaussian Kernel")
        #kernel_width = 10
        std = 1.5
        khw = kernel_width//2
        (x,y) = np.meshgrid(np.arange(kernel_width), np.arange(kernel_width))
        gauss = np.exp(-((x-khw-0.5)**2 + (y-khw-0.5)**2)/(2*std*std))
        gauss/=np.sum(gauss)
        #print(gauss)


        #kernel = torch.tensor(np.zeros((1, 1, n_im, kernel_width, kernel_width),dtype='float32')).cuda()
        kernel = torch.tensor(np.zeros((1, 1, n_im, kernel_width, kernel_width),dtype='float32')).to(device)
        for ir in range(datas.size()[2]):
            kernel[0,0,ir,:,:] = torch.tensor(np.copy(gauss))

    else:
        print('Using PSF kernel')
        ## the JWST kernel provided by Bryan is 4x over sampled
        khw = kernel_width//2
        #kernel = torch.tensor(np.zeros((1, 1, n_im, kernel_width, kernel_width),dtype='float32')).cuda()
        kernel = torch.tensor(np.zeros((1, 1, n_im, kernel_width, kernel_width),dtype='float32')).to(device)

        with fits.open('/arc/projects/jwst-tnos/scripts/NIRCam_A4_PSF.fits') as han:
            over_samp_psf = han[0].data
        (A, B) = over_samp_psf.shape

        over_samp_subsec = over_samp_psf[A//2-4*khw:A//2+4*khw, B//2-4*khw:B//2+4*khw]
        psf = np.zeros((kernel_width, kernel_width), dtype='float')
        for i in range(psf.shape[0]):
            for j in range(psf.shape[1]):
                psf[i,j] = np.sum(over_samp_psf[i*4+A//2-4*khw:(i+1)*4+A//2-4*khw, j*4+B//2-4*khw:(j+1)*4+B//2-4*khw])


        for ir in range(n_im):
            kernel[0,0,ir,:,:] = torch.tensor(np.copy(psf))

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

        print('This is not yet setup for JWST data')
        exit()
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
