import trippy, torch
from astropy.io import fits
import glob
from astropy.wcs import WCS
import numpy as np, scipy as sci, pylab as pyl
from sns_utils import rots

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_data(visit, chip, warps_dir, dbimages, variance_trim, bit_mask, verbose=False, var_trim_keyword='SAT'):
    """
    Read in all the requesite image data
    """

    if 'rtwarp' in warps_dir:
        with open(f'/arc/projects/classy/visitLists/{visit}/{visit}_rtvisit_list.txt') as han:
            data = han.readlines()
        new_mjds = {}
        for d in data:
            s = d.split()
            new_mjds[s[0]]=float(s[2])
            
    datas, masks, variances, mjds, psfs, fwhms, im_nums = [], [], [], [], [], [], []
    fits_files = glob.glob(f'{warps_dir}/{visit}/{chip}/*fits')
    fits_files.sort()

    if len(fits_files)==0:
        print(f'Cannot find any warps at {visit}.')
        exit(1)
    else:
        print(f'Reading {len(fits_files)} files from {warps_dir}/{visit}/{chip}')

    for i in range(len(fits_files)):
        with fits.open(fits_files[i]) as han:

            datas.append(han[1].data)
            masks.append(han[2].data)
            variances.append(han[3].data)

            ## testing adding a 100 pixel pad
            #d = han[1].data
            #m = han[2].data
            #v = han[3].data
            #D = np.zeros(d.shape+np.array([200,200]), dtype=d.dtype)
            #M = np.zeros(d.shape+np.array([200,200]), dtype=m.dtype)
            #V = np.zeros(d.shape+np.array([200,200]), dtype=v.dtype)+np.nanmedian(v)
            #D[100:-100,100:-100] = d
            #M[100:-100,100:-100] = m
            #V[100:-100,100:-100] = v
            #datas.append(D)
            #masks.append(M)
            #variances.append(V)
            
            if i ==0:
                wcs = WCS(han[1].header)
        ## force a non-RT mask onto the RT data because the RT masks are fucked up
        if 'rtwarp' in warps_dir:
            with fits.open(fits_files[i].replace('rtwarp', 'warp')) as han:
                masks[-1] = han[2].data
                
        im_num = fits_files[i].split('DIFFEXP-')[1][:7]
        mjd = han[0].header['MJD-OBS'] if 'rtwarp' not in warps_dir else new_mjds[im_num]
        mjds.append(mjd + han[0].header['EXPTIME']/(3600.*24.*2.))

        model = trippy.psf.modelPSF(restore=f'{dbimages}/{im_num}/ccd{chip}/{im_num}p{chip}.psf.fits', verbose=verbose)
        psfs.append(model.psf)
        fwhms.append(model.FWHM())
        im_nums.append(im_num)

        nan_med_variance = np.nanmedian(variances[-1])
        print(mjds[i], fits_files[i], nan_med_variance)
        w = np.where(variances[-1]>variance_trim*nan_med_variance)
        masks[-1][w] += 2**bit_mask[var_trim_keyword]
    print(f'Read in {len(datas)} images.\n')

    return (datas, masks, variances, mjds, psfs, fwhms, im_nums, wcs)



def get_shift_rates(ref_wcs, mjds, visit, chip, ref_im, ref_im_ind, warps_dir, fwhms, rate_fwhm_grid_step, A, B, save_rates_figure=False):
    """
    get a grid of shift rates from the planted classy imagery
    """


    mid_ra, mid_dec = ref_wcs.all_pix2world(B/2., A/2., 0)

    plant_rates = []
    for i in range(40):
        c = str(i).zfill(2)

        ## hacks to skip missing chips
        if visit=='2023-08-19-AS3Y2_Aug19UTC' and c=='29': continue
        ##
        
        print(f'{warps_dir}/{visit}/{c}/{ref_im}?{c}-*plantList')
        plant_files = glob.glob(f'{warps_dir}/{visit}/{c}/{ref_im}?{c}-*plantList')
        plant_files.sort()

        with open(plant_files[0]) as han:
            data = han.readlines()
        for i in range(1,len(data)):
            s = data[i].split()
            ra,dec,rate_ra,rate_dec = float(s[1]), float(s[2]), float(s[7]), float(s[8])
            x0,y0 = ref_wcs.all_world2pix(mid_ra, mid_dec,0)
            x1,y1 = ref_wcs.all_world2pix(mid_ra+rate_ra/3600.0, mid_dec+rate_dec/3600.0,0)

            rate_x = (x1-x0)*24.0
            rate_y = (y1-y0)*24.0


            plant_rates.append([rate_x, rate_y])
    plant_rates = np.array(plant_rates)
    if rots[chip] == 0:
        plant_rates*=-1
    print(f'Number of planted sources: {len(plant_rates)}')
    
    w = np.where(np.less(plant_rates[:,0]**2+plant_rates[:,1]**2, (24*4.5/0.187)**2))
    angs = np.arctan2(plant_rates[:,1][w], plant_rates[:,0][w])%(2*np.pi)


    min_ang = np.min(angs)
    max_ang = np.max(angs)
    med_ang = np.median(angs)
    # bodge angle hack
    if min_ang<0:
        while min_ang<0:
            min_ang+=2*np.pi
            med_ang+=2*np.pi

    d_ang = max(max_ang-med_ang, med_ang-min_ang)
    max_ang = med_ang + d_ang
    min_ang = med_ang - d_ang
    print('Angles (min, max, med, delta):', min_ang, max_ang, med_ang, d_ang)


    W = np.where(np.abs(plant_rates[:,0])<200)
    l = sci.stats.linregress(plant_rates[:,0][W], plant_rates[:,1][W])

    max_x = np.max(plant_rates[:,0] + 5)
    max_y = max_x*l.slope+l.intercept

    print('Max x/y:', max_x, max_y)

    slopes = (plant_rates[:,1]-max_y)/(plant_rates[:,0]-max_x)
    m_low = np.min(slopes[w])
    b_low = max_y - m_low*max_x
    m_high = np.max(slopes[w])
    b_high = max_y - m_high*max_x

    x = np.linspace(max_x, -(24*4.5/0.187), 5)
    y_low = x*m_low+b_low
    y_high = x*m_high+b_high

    if save_rates_figure:
        fig = pyl.figure(1)
        fig.add_subplot(211)
        pyl.scatter(max_x, max_y)
        pyl.plot(x,y_low)
        pyl.plot(x,y_high)
        pyl.scatter(plant_rates[:,0], plant_rates[:,1], zorder=-1, marker='.')

    # In[5]:


    seeing = np.mean(fwhms)*0.187 #0.7
    print(f'Mean FWHM {seeing}"')
    seeing /= 0.187 # pixels

    #dh = (mjds[-1]-mjds[0]) # days
    dh = (np.max(mjds)-np.min(mjds)) # days, need to take the np.max and np.min because images aren't necessarily in order of increase time.
    drate = rate_fwhm_grid_step*seeing/dh  # 0.75 seems to be a good sweet spot


    dmjds = mjds-mjds[ref_im_ind]

    ang_steps_h = np.linspace(med_ang, max_ang+0.0, 80)
    ang_steps_l = np.linspace(min_ang-0.0, med_ang, 80)


    rates = [[max_x, max_y]]
    rx, ry = max_x, max_y
    current_rate = (max_x**2+max_y**2)**0.5
    while current_rate < (24*4.5/0.187):
        n_x = np.cos(ang_steps_h)*current_rate# + max_x
        n_y = np.sin(ang_steps_h)*current_rate# + max_y

        dist_rates = ( ((n_x - n_x[0])**2 + (n_y - n_y[0])**2)**0.5 / drate).astype('int')
        unique_dist_rates = np.unique(dist_rates)
        for ind in unique_dist_rates:
            w = np.where(dist_rates == ind)
            rates.append([n_x[w[0][0]], n_y[w[0][0]]])
            #pyl.scatter(rates[-1][0], rates[-1][1], alpha = 0.5, marker='s', s=70, c='r')


        n_x = np.cos(ang_steps_l[::-1])*current_rate# + max_x
        n_y = np.sin(ang_steps_l[::-1])*current_rate# + max_y
        dist_rates = (((n_x - n_x[0])**2 + (n_y - n_y[0])**2)**0.5 / drate).astype('int')
        unique_dist_rates = np.unique(dist_rates)
        for ind in unique_dist_rates:
            if ind == 0: continue
            w = np.where(dist_rates == ind)
            rates.append([n_x[w[0][0]], n_y[w[0][0]]])
            #pyl.scatter(rates[-1][0], rates[-1][1], alpha = 0.5, marker='s', s=70, c='r')

        current_rate += drate
    rates = np.array(rates)[1:]  ### the first rate is duplicated in the above algorithm
    print('Number of rates:', len(rates))
    
    if save_rates_figure:
        sp = fig.add_subplot(212)
        pyl.scatter(plant_rates[:, 0], plant_rates[:, 1],alpha=0.25, marker='.')#,  zorder = -1)
        pyl.scatter(rates[:,0], rates[:,1], alpha = 0.5, marker='s', s=70)

        sp.grid(linestyle=':')
        sp.set_xlim(sp.get_xlim()[0],0.0)

        print('Saving Rates Figure.')
        pyl.savefig('Rates_figure.png')
        exit()
        
    if rots[chip]==0:
        rates *= -1

    return (rates, plant_rates)


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


        kernel = torch.tensor(np.zeros((1, 1, len(mjds), kernel_width, kernel_width),dtype='float32')).to(device)#.cuda()
        for ir in range(datas.size()[2]):
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
