#!/usr/bin/env python

import sys
sys.path.append('/arc/home/fraserw/git/trippy')

import torch
import numpy as np, pylab as pyl, scipy as sci
from numpy import ma
import glob, os, gc
from torch.nn import functional
from sklearn.cluster import DBSCAN
import time

from astropy.visualization import ManualInterval, ZScaleInterval

gpu_available = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if gpu_available:
    print('Using GPU.')
else:
    print('GPU not found. Attempting to use CPU cores.')



from argparse import ArgumentParser
import logging

from sns_utils import *
from sns_data_pencilbeam import *

r2d = 180./np.pi



parser = ArgumentParser()
parser.add_argument('visit', default = '01001')
parser.add_argument('chip', default = 'nrca1')
parser.add_argument('--useNegativeWell', default = False, action='store_true')
parser.add_argument('--saves_path', default = '/arc/projects/jwst-tnos/wesmod_results')
parser.add_argument('--min_snr', default=4.5, type=float)
parser.add_argument('--trim_snr', default=5.5, type=float)
parser.add_argument('--n-keep', default=4, type=int)
parser.add_argument('--clust_dist_lim', default=1.0, type=float)
parser.add_argument('--clust_min_samp', default=2, type=int)
parser.add_argument('--peak-offset-max', default=4.0, type=float)
parser.add_argument('--rate_fwhm_grid_step', default=0.75, type=float)
parser.add_argument('--variance-trim', default=1.4, type=float, help='Not curently used.')
parser.add_argument('--use-gaussian-kernel', action='store_true', default=False)
parser.add_argument('--kernel-width', default=6, type=int) ## comparing 6,8,and 14 it doesn't eem to get deeper, but wider (eg.14) results in more sources to vet
parser.add_argument('--log-level', default=logging.INFO, type=lambda x: getattr(logging, x),
                    help="Configure the logging level.", choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'])
parser.add_argument('--log-dir', default='/arc/projects/jwst-tnos/logs/wesmod', type=str)
parser.add_argument('--save-rates-figure', action='store_true', default=False)
parser.add_argument('--bitmask', default='bitmask_pencilbeam.dat', help='The bitmask to use with these data. Not yet reading from image headers. DEFAULT=%(default)s')
parser.add_argument('--flagkeys', default='flagkeys_pencilbeam.dat', help='The file containing the keys to mask. DEFAULT=%(default)s')
parser.add_argument('--ultrafine', default=False, action='store_true')
parser.add_argument('--custom-rate-limits', default=None, help='Search over custom rate limits dxlow,dxhigh,dylow,dyhigh.')
args = parser.parse_args()

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)
logging.basicConfig(level=args.log_level, filename=f'{args.log_dir}/wesmod_{args.visit}_{args.chip}.log', encoding='utf-8',)



useNegativeWell = args.useNegativeWell

saves_path = args.saves_path
warps_dir = '/arc/projects/jwst-tnos/tiles_v4/subtracted/kbmod_prep'
visit = args.visit+'_'+args.chip

plants_dir = '/arc/projects/jwst-tnos/planted/implants_v4'

# snr=4.5 and grid_step=0.75 seem to be sweet spots
min_snr = args.min_snr ## original SNR during initial kernel search

rate_fwhm_grid_step = args.rate_fwhm_grid_step ## the shift rate grid spacing in units of mean FWHM of sequence

n_keep = args.n_keep ## number of sources to keep after initial kernel search

dist_lim = args.clust_dist_lim ## distance frmo candidate to line in predictive clustering routine
min_samp = args.clust_min_samp ## number of clustered detections required to keep a source
trim_snr = args.trim_snr ## min SNR of sources to keep after predictive clustering default 5.5

dist_lim_x = 4
dist_lim_y = 6

peak_offset_max = args.peak_offset_max ## the max allowable distance between stamp peak and centre of stamp

variance_trim = args.variance_trim # the factor above the median variance at which we mask all pixels

spacing = 70. if not args.ultrafine else 30.


(bit_mask, flag_keys) = read_bitmask(args.bitmask, args.flagkeys)


flags = 0
for bit in flag_keys:
    flags += 2**bit_mask[bit]

badflags = flags

## setup the data arrays
(datas, masks, variances, mjds, im_nums, wcs) = read_data(visit, warps_dir, variance_trim, bit_mask, var_trim_keyword='BAD', verbose=True)
(A,B) = datas[0].shape


np_datas = np.expand_dims(np.expand_dims(np.array(datas, dtype='float32'),0),0)
np_inv_variances = np.expand_dims(np.expand_dims(1.0/np.array(variances, dtype='float32'),0),0)
np_masks = np.expand_dims(np.expand_dims(np.array(masks, dtype='int'),0),0)



# (np_masks & badflags) == 0 is FALSE when a pixel matches a badflag value
# so ~((np_masks & badflags) == 0) is TRUE when a pixel matches a badflag value
# so ~((np_masks & badflags) == 0) | np.isnan(datas) is TRUE when a pixel matches a badflag or is nans
w = np.where(~((np_masks & badflags) == 0) | np.isnan(datas)) ## where pixels are bad
np_datas[w]=0.0
np_inv_variances[w] = 0.0

np_masks[w] = 0
np_masks = np.clip(np_masks,0,1) ## masks with 1 are GOOD pixels, 0 are BAD pixels


#datas = torch.tensor(np_datas).cuda()
#inv_variances = torch.tensor(np_inv_variances).cuda()
datas = torch.tensor(np_datas).to(device)
inv_variances = torch.tensor(np_inv_variances).to(device)

mjds = np.array(mjds)

mid_time = (mjds[-1]+mjds[0])/2.
diff_times = mjds-mid_time
ref_im_ind = 0
ref_im = im_nums[0]
print('Reference image:', ref_im)
logging.info('Using reference image '+str(ref_im))

#n_im = int(torch.tensor(float(datas.size()[2])).cuda().item())
n_im = int(torch.tensor(float(datas.size()[2])).to(device).item())

dmjds = mjds-mjds[ref_im_ind]


## setup the shift rates
if args.custom_rate_limits is not None:
    s = args.custom_rate_limits.split(',')
    custom_rate_limits = [[float(s[0]), float(s[1])], [float(s[2]), float(s[3])]]
else:
    custom_rate_limits = None

(rates, plant_rates) = get_shift_rates(wcs, mjds, visit, spacing=spacing, rate_lims_custom = custom_rate_limits)
print(len(rates))

logging.info(f'\nUsing {len(rates)} rates.')
for r in rates:
    logging.info(r)



## setup the kernel
useGaussianKernel = args.use_gaussian_kernel
useNegativeWell = True if not args.useNegativeWell else False


khw = args.kernel_width//2
kernel = create_kernel(n_im, useNegativeWell=False, useGaussianKernel=False, kernel_width=args.kernel_width, visit=None)
rot_kernel = torch.rot90(kernel, k=2, dims=(3,4))


for ir in range(n_im):
    datas[0,0,ir,:,:] = torch.conv2d(datas[:,:,ir,:,:]*inv_variances[:,:,ir,:,:], kernel[:,:,ir,:,:], padding='same')
    inv_variances[0,0,ir,:,:] = torch.conv2d(inv_variances[:,:,ir,:,:], kernel[:,:,ir,:,:]*kernel[:,:,ir,:,:], padding='same')


## do the shift-stacking
snr_image, alpha_image = run_shifts(datas, inv_variances, rates, dmjds, min_snr, writeTestImages=False)

#### sort inds hack
# sort_inds = torch.zeros((1,1,len(rates),A,B), dtype=torch.int64, device='cpu')
# sort_step = 100
# a = 0
# b=sort_step
# while b<B:
#     b = min(a+sort_step, B)
#     print('sorting', a, b)
#     sort_inds_wedge = torch.sort(snr_image[:,:,:,:,a:b], 2, descending=True)[1]
#     sort_inds[:,:,:,:,a:b] = sort_inds_wedge
#     a+=sort_step

## sort and keep the top n_keep detections,
## this step approximately doubles the memory footprint to 60 GB. Could do this in stages to reduce memory footprint at the cost of processing speed
sort_inds = torch.sort(snr_image, 2, descending=True)[1]

## trim the negative SNR sources. The reason these show up is because the likelihood formalism sucks
detections = trim_negative_snr(snr_image, alpha_image, sort_inds, n_keep, rates, A, B)
del snr_image, alpha_image, sort_inds
gc.collect()
torch.cuda.empty_cache()


## trim the flux negative sources
detections = trim_negative_flux(detections)


## now apply the brightness filter. Check n_bright_test values between test_low and test_high fraction of the estimated value
apply_brightness_filter = True
#im_datas = functional.pad(torch.tensor(np_datas).cuda(), (khw, khw, khw, khw))
#inv_vars = functional.pad(torch.tensor(0.5*np_inv_variances).cuda(), (khw, khw, khw, khw))
im_datas = functional.pad(torch.tensor(np_datas).to(device), (khw, khw, khw, khw))
inv_vars = functional.pad(torch.tensor(0.5*np_inv_variances).to(device), (khw, khw, khw, khw))



del np_datas # I don't think this is used again.
gc.collect()

c = torch.zeros_like(im_datas)
c[0,0,0] = im_datas[0,0,0]
cv = torch.zeros_like(im_datas)
cv[0,0,0] = inv_vars[0,0,0]

if apply_brightness_filter:
    keeps = brightness_filter(im_datas, inv_vars, c, cv, kernel, dmjds, rates, detections, khw, n_im, n_bright_test = 10, test_high = 1.15, test_low = 0.85)
else:
    keeps = np.arange(len(detections))



print(len(keeps), len(detections))
filt_detections = np.copy(detections[keeps])
print(filt_detections.shape)

del keeps, inv_vars
gc.collect()
torch.cuda.empty_cache()


# now create the stamps
#im_masks = functional.pad(torch.tensor(np_masks), (khw, khw, khw, khw)).cuda()
im_masks = functional.pad(torch.tensor(np_masks), (khw, khw, khw, khw)).to(device)
del np_masks


mean_stamps = create_stamps(im_datas, im_masks, c, cv, dmjds, rates, filt_detections, khw)
stamps = mean_stamps

del im_masks
gc.collect()
torch.cuda.empty_cache()


show_test_stamps = False
if show_test_stamps:
    (z1,z2) = ZScaleInterval().get_limits(mean_stamps)
    normer = ManualInterval(z1,z2)

    args = np.argsort(filt_detections[:,5] )[::-1]
    for i in range(0):
        fig = pyl.figure()
        sp1 = fig.add_subplot(141)
        pyl.imshow(normer(mean_stamps[args[i]]))
        sp2 = fig.add_subplot(142)
        pyl.imshow(normer(med_stamps[args[i]]))

        sp3 = fig.add_subplot(143)
        d = mean_stamps[args[i]]-med_stamps[args[i]]
        print(np.std(d)/np.max(d), (np.max(d)-np.min(d))/np.max(d))
        print(np.sum((d/mean_stamps[args[i]])**2)**0.5)
        pyl.imshow(d)

        sp4 = fig.add_subplot(144)
        d = mean_stamps[args[i]]/med_stamps[args[i]]
        print(np.std(d))
        pyl.imshow(d)

        #print(stamps[args[i]])
        (x,y,f,snr) = filt_detections[args[i]][np.array([0,1,4,5])]
        pyl.title('{} {} {:.2f} {:.2f}'.format(x,y,f,snr))
        pyl.show()




## trim the ones with peak offset more than peak_offset_max pixels
apply_peak_offset_filter = True
if apply_peak_offset_filter:
    stamps, filt_detections = peak_offset_filter(stamps, filt_detections, peak_offset_max)

save_filt_detections = False
if save_filt_detections:
    with open('filt_detections.npy', 'wb') as han:
        np.save(han, filt_detections)



## do predictive line clustering
apply_predictive_line_cluster = True
if apply_predictive_line_cluster:
    clust_detections, clust_stamps = predictive_line_cluster(filt_detections, stamps, dmjds, dist_lim, min_samp, init_select_proc_distance=15, show_plot=False)
    print(len(clust_detections))
    logging.info(f'Number of sources kept after brightness and peak location filtering: {len(clust_detections)}.')
else:
    clust_detections = filt_detections
    clust_stamps = stamps
    logging.info(f'Number of sources kept: {len(clust_detections)}.')
del stamps
gc.collect()



# trim on snr
w = np.where(clust_detections[:,5]>=trim_snr)
clust_detections = clust_detections[w]
clust_stamps = clust_stamps[w]
print(len(clust_detections))
logging.info(f'Number of sources kept after final SNR trim: {len(clust_detections)}.')



#inv_vars = functional.pad(torch.tensor(0.5*np_inv_variances).cuda(), (khw, khw, khw, khw))
inv_vars = functional.pad(torch.tensor(0.5*np_inv_variances).to(device), (khw, khw, khw, khw))
cv[0,0,0] = inv_vars[0,0,0]


## now apply a positional filter on the clust_detections to see if the likelihood minimimum is near the centre
apply_positional_filter = False
if apply_positional_filter:
    grid_detections, grid_stamps = position_filter(clust_detections, clust_stamps, im_datas, inv_vars, c, cv, kernel, dmjds, rates, khw)
else:
    grid_detections = clust_detections
    grid_stamps = clust_stamps



## trim on snr
## I don't think this extra step is necessary.
print(len(grid_detections))
w = np.where(grid_detections[:,5]>=trim_snr)
final_detections = grid_detections[w]
final_stamps = grid_stamps[w]
print(len(final_detections))


## sort by SNR and save
snr_args = np.argsort(final_detections[:,5])[::-1]
final_detections = final_detections[snr_args]
final_stamps = final_stamps[snr_args]

try:
    os.makedirs(f'{saves_path}/results_{visit}/')
except:
    pass

output_file = 'results_.txt' if not args.ultrafine else 'results_uf.txt'

logging.info(f'Saving to {saves_path}/results_{visit}/{output_file}')
with open(f'{saves_path}/results_{visit}/{output_file}', 'w+') as han:
    for i in range(len(final_detections)):
        (x,y,rx,ry,f,snr) = final_detections[i,:6]
        print(f'snr: {snr} flux: {f} x: {x} y: {y} x_v: {rx} y_v: {ry}', file=han)

with open(f'{saves_path}/results_{visit}/input.pars', 'w+') as han:
    print('useNegativeWell:', useNegativeWell, file=han)
    print('saves_path:',  saves_path, file=han)
    print('warps_dir:', warps_dir, file=han)
    print('min_snr:', min_snr, file=han)
    print('rate_fwhm_grid_step:', rate_fwhm_grid_step, file=han)
    print('n_keep:', n_keep, file=han)
    print('dist_lim:', dist_lim, file=han)
    print('min_samp:', min_samp, file=han)
    print('trim_snr:', trim_snr, file=han)
    print('peak_offset_max:', peak_offset_max, file=han)
    print('variance_trim:', variance_trim, file=han)
    print('bitmask:', bit_mask, file=han)
    print('flag_keys:', flag_keys, file=han)


save_stamps_figs = True
if save_stamps_figs:
    (z1,z2) = ZScaleInterval().get_limits(mean_stamps)
    normer = ManualInterval(z1,z2)

    fig = pyl.figure('', (13,13))
    n_p = 0
    while n_p<9*9:
        if n_p == len(final_detections):
            break
        sp = fig.add_subplot(9, 9, n_p+1, xticklabels='', yticklabels='')
        pyl.imshow(normer(final_stamps[n_p]))
        pyl.title('{} {} {:.1f}'.format(int(final_detections[n_p,0]), int(final_detections[n_p,1]), final_detections[n_p,5]), fontsize=7)
        n_p+=1
    pyl.savefig(f'{saves_path}/results_{visit}/final_stamps.png')
    

    
## now compare the outputs to the plants
plants = []
logging.info(f'{plants_dir}/implant_converts_epoch{visit[1]}_dither1.csv')
with open(f'{plants_dir}/implant_converts_epoch{visit[1]}_dither1.csv') as han:
    data = han.readlines()

for i in range(1,len(data)):
    s = data[i].split()
    if s[1]==visit:
        x, y, rate_x, rate_y = float(s[6]), float(s[7]), float(s[8]), float(s[9])
        plants.append([x, y, rate_x, rate_y, float(s[10]), 0,0,0,0])

plants = np.array(plants)
plants = plants[np.argsort(plants[:,4])]

for i in range(len(plants)):
    for j,det in enumerate([detections, filt_detections, clust_detections, final_detections]):

        dist_sq = np.sum((plants[i][:2] - det[:,:2])**2, axis=1)
        dist_rate_sq = np.sum((plants[i][2:4] - det[:,2:4])**2, axis=1)
        w = np.where((dist_sq<3.5**2) & (dist_rate_sq<2000.**2))
        if len(w[0])>0:
            plants[i,5+j]=1
    print("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.2f} {} {} {} {} {}".format(np.min(dist_sq**0.5), np.min(dist_rate_sq**0.5), plants[i][0], plants[i][1], plants[i,2], plants[i,3], plants[i][4], plants[i,5], plants[i,6], plants[i,7], plants[i,8], len(w[0])))
    logging.info("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.2f} {} {} {} {} {}".format(np.min(dist_sq**0.5), np.min(dist_rate_sq**0.5), plants[i][0], plants[i][1], plants[i,2], plants[i,3], plants[i][4], plants[i,5], plants[i,6], plants[i,7], plants[i,8], len(w[0])))

print('Number of plants found:', len(np.where(plants[:,-1])[0]))
logging.info('Number of plants found: '+str(len(np.where(plants[:,-1])[0])))

show_eff_plot = False
if show_eff_plot:
    eff_bin_width = 0.25
    mags = np.arange(20,np.max(plants[:,4])+eff_bin_width, eff_bin_width)
    n = mags*0.0
    f = [mags*0.0,mags*0.0,mags*0.0,mags*0.0]
    k = ((plants[:, 4]-mags[0])/(mags[1]-mags[0])).astype('int')
    for i in range(len(plants)):
        n[k[i]]+=1.
        for j in [5,6,7,8]:
            if plants[i,j]:
                f[j-5][k[i]]+=1.
    labels = ['det', 'filt', 'clust', 'final']
    for j in range(len(labels)):
        pyl.scatter(mags+(mags[1]-mags[0])/2.+j*0.02,f[j]/n, label=labels[j])
    pyl.legend()
    pyl.show()

