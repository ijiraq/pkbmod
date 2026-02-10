#!/usr/bin/env python

import sys
sys.path.append('/arc/home/fraserw/git/trippy')

#from astropy.io import fits
#from astropy.wcs import WCS
import torch
import numpy as np, pylab as pyl, scipy as sci
from numpy import ma
import glob, os, gc
from torch.nn import functional
from sklearn.cluster import DBSCAN
#import trippy
import time
#from tensorflow import keras

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
from sns_data_nh import *

parser = ArgumentParser()
parser.add_argument('date', default = '20210609')
parser.add_argument('chip', default = '0_04')
parser.add_argument('--dontUseNegativeWell', default = False, action='store_true')
parser.add_argument('--saves_path', default = '/arc/projects/NewHorizons/wesmod_results_v27', help='Path to save the results.txt and input.pars files to. Default=%(default)s. if --rt is used, wesmod will be replaced with rtwesmod')
parser.add_argument('--min_snr', default=4.5, type=float)
parser.add_argument('--trim_snr', default=5.5, type=float)
parser.add_argument('--n-keep', default=4, type=int)
parser.add_argument('--clust_dist_lim', default=4.0, type=float)
parser.add_argument('--clust_min_samp', default=2, type=int)
parser.add_argument('--peak-offset-max', default=4.0, type=float)
parser.add_argument('--rate_fwhm_grid_step', default=0.75, type=float)
parser.add_argument('--variance-trim', default=1.3, type=float)
parser.add_argument('--use-gaussian-kernel', action='store_true', default=False)
parser.add_argument('--kernel-width', default=14, type=int)
parser.add_argument('--log-level', default=logging.INFO, type=lambda x: getattr(logging, x),
                    help="Configure the logging level.", choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'])
parser.add_argument('--log-dir', default='/arc/projects/NewHorizons/logs/wesmod', type=str)
parser.add_argument('--bitmask', default='bitmask_v27.dat', help='The bitmask to use with these data. Not yet reading from image headers. DEFAULT=%(default)s')
parser.add_argument('--flagkeys', default='flagkeys_nh.dat', help='The file containing the keys to mask. DEFAULT=%(default)s')
parser.add_argument('--read-from-params', action = 'store_true', default=False, help='Read from NewHorizons/params/wesmod.params and ignore command line inputs')
parser.add_argument('--save-rates-figure', action='store_true', default=False)
args = parser.parse_args()


if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)
logging.basicConfig(level=args.log_level, filename=f'{args.log_dir}/wesmod_{args.date}_{args.chip}.log', encoding='utf-8',)

# In[2]:

valid_region = [2900, 7150 , 3990, 6040]

r2d = 180./np.pi

useNegativeWell = True if not args.dontUseNegativeWell else False

saves_path = args.saves_path
warps_dir = f'/arc/projects/NewHorizons/HSC_2024/DIFFS'
chip = args.chip
date = args.date

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


(bit_mask, flag_keys) = read_bitmask(args.bitmask, args.flagkeys)

flags = 0
for bit in flag_keys:
    flags += 2**bit_mask[bit]

badflags = flags


if args.read_from_params:
    print('Reading from NewHorizons/params/wesmod.params')
    logging.info('Reading from NewHorizons/params/wesmod.params')
    with open('/arc/projects/NewHorizons/params/wesmod.params') as han:
        data = han.readlines()
    found=False
    for d in data:
        s = d.split()
        if s[0] == args.visit:
            found=True
            break
    if not found:
        print('cant find the visit!')
        exit()
    min_snr = float(s[1])
    trim_snr = float(s[2])
    n_keep = int(float(s[3]))
    dist_lim = float(s[4])
    min_samp = int(float(s[5]))
    peak_offset_max = float(s[6])
    rate_fwhm_grid_step = float(s[7])
    variance_trim = float(s[8])


# In[3]:


(datas, masks, variances, mjds, psfs, fwhms, im_nums, wcs) = read_data(args.date, chip, warps_dir,  variance_trim, bit_mask, verbose=False, valid_region=valid_region)
(A,B) = datas[0].shape
print(f'Image shape: {B}x{A} pix')


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


datas = torch.tensor(np_datas).to(device)
inv_variances = torch.tensor(np_inv_variances).to(device)

mjds = np.array(mjds)
im_nums = np.array(im_nums, dtype='int')

mid_time = (mjds[-1]+mjds[0])/2.
diff_times = mjds-mid_time
ref_im_ind = np.argmin(np.abs(diff_times-mid_time))
ref_im = im_nums[ref_im_ind]
ref_im_ind = 0
ref_im = im_nums[0]
print('Reference image:', ref_im)
logging.info('Using reference image '+str(ref_im))

#n_im = int(torch.tensor(float(datas.size()[2])).cuda().item())
n_im = int(torch.tensor(float(datas.size()[2])).to(device).item())

fwhms = np.array(fwhms)

dmjds = mjds-mjds[ref_im_ind]

# In[4]:


(rates, plant_rates) = get_shift_rates(wcs, mjds, args.date, args.chip, ref_im, ref_im_ind, warps_dir, fwhms, rate_fwhm_grid_step, A, B, save_rates_figure=args.save_rates_figure)

logging.info(f'\nUsing {len(rates)} rates.')
for r in rates:
    logging.info(r)



# In[6]:




useGaussianKernel = args.use_gaussian_kernel


khw = args.kernel_width//2
kernel = create_kernel(psfs, dmjds, rates, useNegativeWell, useGaussianKernel, kernel_width=args.kernel_width, im_nums=im_nums)
rot_kernel = torch.rot90(kernel, k=2, dims=(3,4))


for ir in range(n_im):
    datas[0,0,ir,:,:] = torch.conv2d(datas[:,:,ir,:,:]*inv_variances[:,:,ir,:,:], kernel[:,:,ir,:,:], padding='same')
    inv_variances[0,0,ir,:,:] = torch.conv2d(inv_variances[:,:,ir,:,:], kernel[:,:,ir,:,:]*kernel[:,:,ir,:,:], padding='same')


# do the shift-stacking
snr_image, alpha_image = run_shifts(datas, inv_variances, rates, dmjds, min_snr, writeTestImages=False)
print('Done shifting')

# In[8]:

## sort and keep the top n_keep detections,
## this step approximately doubles the memory footprint to 60 GB. Could do this in stages to reduce memory footprint at the cost of processing speed

#sort_inds = torch.sort(snr_image, 2, descending=True)[1]
#### sort inds hack
sort_inds = torch.zeros((1, 1, n_keep, A, B), dtype=torch.int64, device='cpu')

sort_step = 100
a = 0
b=sort_step
while b<B:
    b = min(a+sort_step, B)
    print(f' Sorting {a} to {b} of {B}...', end=' ')
    sort_inds_wedge = torch.sort(snr_image[:,:,:,:,a:b].to(device), 2, descending=True)[1]
    sort_inds[:,:,:,:,a:b] = sort_inds_wedge[:,:,:4,:,:]
    a+=sort_step
    print('Done')


# # trim the negative SNR sources. The reason these show up is because the likelihood formalism sucks
detections = trim_negative_snr(snr_image, alpha_image, sort_inds, n_keep, rates, A, B)
fluxes = np.sort(detections[:,4])

del snr_image, alpha_image, sort_inds
gc.collect()
torch.cuda.empty_cache()


# # trim the flux negative sources
detections = trim_negative_flux(detections)

# In[10]:


# # now apply the brightness filter. Check n_bright_test values between test_low and test_high fraction of the estimated value
im_datas = functional.pad(torch.tensor(np_datas).to(device), (khw, khw, khw, khw))
inv_vars = functional.pad(torch.tensor(0.5*np_inv_variances).to(device), (khw, khw, khw, khw))
#
del np_datas # I don't think this is used again.
gc.collect()

c = torch.zeros_like(im_datas)
c[0,0,0] = im_datas[0,0,0]
cv = torch.zeros_like(im_datas)
cv[0,0,0] = inv_vars[0,0,0]

keeps = brightness_filter(im_datas, inv_vars, c, cv, kernel, dmjds, rates, detections, khw, n_im, n_bright_test = 10, test_high = 1.15, test_low = 0.85)
# In[12]:


print(len(keeps), len(detections))
filt_detections = np.copy(detections[keeps])
print(filt_detections.shape)
del keeps


# In[13]:


del inv_vars
gc.collect()
torch.cuda.empty_cache()

#im_masks = functional.pad(torch.tensor(np_masks), (khw, khw, khw, khw)).cuda()
im_masks = functional.pad(torch.tensor(np_masks), (khw, khw, khw, khw)).to(device)
del np_masks

# create the stamps
# mean_stamps = []
# #med_stamps = []
# indices = []
# saved= False
# for ir in range(len(rates)):
#
#     # these are required to reset from the nans below
#     c[0,0,0] = im_datas[0,0,0]
#     cv[0,0,0] = im_masks[0,0,0]
#
#     t1 = time.time()
#     w = np.where((filt_detections[:,2]==rates[ir][0]) & (filt_detections[:,3] == rates[ir][1]))
#
#     for id in range(1, n_im):
#         shifts = (-round(dmjds[id]*rates[ir][1]), -round(dmjds[id]*rates[ir][0]))
#         c[0,0,id]  = torch.roll(im_datas[0,0,id], shifts=shifts, dims=[0,1])
#         cv[0,0,id] = torch.roll(im_masks[0,0,id], shifts=shifts, dims=[0,1]) # mask values with 1 are GOOD pixels
#     mean_stamp_frame = torch.sum(c, 2)
#
#     #c[cv == 0] = float('nan')
#     #med_stamp_frame = torch.nanmedian(c, 2)[0]
#
#     mask_frame = torch.sum(cv, 2)
#
#     for iw in w[0]:
#         x,y = filt_detections[iw,:2].astype('int')
#         #print(x,y, iw, khw)
#         mean_stamp = mean_stamp_frame[0,0,y:y+khw*2+1, x:x+khw*2+1]
#
#         """
#         med_sections = c[0,0,:,y:y+khw*2+1, x:x+khw*2+1].cpu()
#         mask_sections = cv[0,0,:,y:y+khw*2+1, x:x+khw*2+1].cpu()
#         mask_sections[np.where((np.isnan(mask_sections)) | (np.isinf(mask_sections)))] = 0
#
#         # mask_sections==0 are bad pixels
#         # numpy masks value == 1 are bad pixels
#         masked_med_sections = ma.array(med_sections, mask= 1-mask_sections)
#         med_stamps.append(ma.median(masked_med_sections,axis=0).filled(0.0))
#         """
#
#         mask = np.copy(mask_frame[0,0,y:y+khw*2+1, x:x+khw*2+1].cpu())
#         mask[np.where((np.isnan(mask)) | (np.isinf(mask)))] = 0.0
#         np.clip(mask, 1., n_im, out=mask)
#         mean_stamps.append(np.copy(mean_stamp.cpu())/mask )
#
#         indices.append(iw)



# indices = np.array(indices)
# mean_stamps = np.array(mean_stamps)[indices]
# #med_stamps = np.array(med_stamps)[indices]
mean_stamps = create_stamps(im_datas, im_masks, c, cv, dmjds, rates, filt_detections, khw)

del im_masks
gc.collect()
torch.cuda.empty_cache()

(z1,z2) = ZScaleInterval().get_limits(mean_stamps)
normer = ManualInterval(z1,z2)

stamps = mean_stamps

show_test_stamps = False
if show_test_stamps:
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



# In[14]:


# # trim the ones with peak offset more than peak_offset_max pixels
# (N, a, b) = stamps.shape
# (gx,gy) = np.meshgrid(np.arange(b), np.arange(a))
# gx = gx.reshape(a*b)
# gy = gy.reshape(a*b)
# rs_stamps = stamps.reshape(N,a*b)
# args = np.argmax(rs_stamps,axis=1)
# X = gx[args]
# Y = gy[args]
# radial_d = ((X-b/2)**2+(Y-a/2)**2)**0.5
# w = np.where(radial_d<peak_offset_max)
# filt_detections = filt_detections[w]
# stamps = stamps[w]
stamps, filt_detections = peak_offset_filter(stamps, filt_detections, peak_offset_max)

save_filt_detections = False
if save_filt_detections:
    with open('filt_detections.npy', 'wb') as han:
        np.save(han, filt_detections)

# # do predictive line clustering
# show_plot = False
#
# proc_filt_detections = np.copy(filt_detections)
#
# proc_inds = np.arange(len(proc_filt_detections))
# clust_detections, clust_inds = [], []
#
# #for i in range(0,10):
# while len(proc_filt_detections)>0:
#
#     arg_max = np.argmax(proc_filt_detections[:,5]) # 5 - max on SNR, 4 is flux
#     #pyl.imshow(normer(stamps[arg_max]))
#     #pyl.show()
#     #exit()
#     x_o, y_o, rx_o, ry_o, f_o, snr_o = proc_filt_detections[arg_max, :6]
#
#     #this secondary where command is necessary because of memory overflows in large detection lists
#     w = np.where( (proc_filt_detections[:,0] > proc_filt_detections[arg_max,0]-55) & (proc_filt_detections[:,0] < proc_filt_detections[arg_max,0] +60)
#                  & (proc_filt_detections[:,1] > proc_filt_detections[arg_max,1]-55) & (proc_filt_detections[:,1] < proc_filt_detections[arg_max,1] +60))
#
#     W = np.where( ((proc_filt_detections[w[0],0]-proc_filt_detections[arg_max,0])**2 + (proc_filt_detections[w[0],1]-proc_filt_detections[arg_max,1])**2) < 60**2)
#     w = w[0][W[0]]
#
#     fd_subset = proc_filt_detections[w]
#
#     drx = fd_subset[:,2] - rx_o
#     dry = fd_subset[:,3] - ry_o
#     dt = dmjds # just for clarity
#
#     x_n, y_n = x_o - drx*dt[-1], y_o - dry*dt[-1] # predicted centroid  position of secondary detection shifted at the differential wrong rate.
#
#     dx, dy = (x_n - x_o), (y_n - y_o) # predicted centroid shifted such that best detection is now at origin
#     dxp = dx*fd_subset[:,1]
#     dyp = dy*fd_subset[:,0]
#     xm = x_n*y_o
#     ym = y_n*x_o
#     dx2 = dx**2
#     dy2 = dy**2
#     top = np.abs(dyp - dxp + xm - ym)
#     bottom = np.sqrt( dx2 + dy2 )
#     dist = top/bottom
#     #dist = np.abs( (y_n-y_o)*fd_subset[:, 0] - (x_n-x_o)*fd_subset[:,1] + x_n*y_o - y_n*x_o    ) / np.sqrt( (x_n-x_o)**2 + (y_n-y_o)**2)
#
#     vert_distance = np.abs(y_n - fd_subset[:,1])
#     hor_distance = np.abs(x_n - fd_subset[:,0])
#
#
#     clust = np.where( (dist<dist_lim) | (np.isnan(dist)) | ((dist<dist_lim) & (drx==0) & (dry==0)))
#     not_clust = np.where(~( (dist<dist_lim) | (np.isnan(dist)) | ((dist<dist_lim) & (drx==0) & (dry==0))) )
#
#     #clust = np.where( ( (hor_distance < dist_lim_x)&(vert_distance<dist_lim_y) ) | (np.isnan(dist)) | ((dist<dist_lim)&(dx==0)&(dy==0) ))
#     #not_clust = np.where( ~( ( (hor_distance < dist_lim_x)&(vert_distance<dist_lim_y) ) | (np.isnan(dist)) | ((dist<dist_lim)&(dx==0)&(dy==0) ) ))
#
#     if len(clust[0])>=min_samp:
#         clust_detections.append(proc_filt_detections[arg_max])
#         clust_inds.append(proc_inds[arg_max])
#
#     if show_plot:
#         fig = pyl.figure(1)
#         sp = fig.add_subplot(111, projection='3d')
#         sp.scatter3D(fd_subset[clust,0], fd_subset[clust,1], fd_subset[clust,3], marker='o', c='b')
#         sp.scatter3D(fd_subset[not_clust,0], fd_subset[not_clust,1], fd_subset[not_clust,3], marker='^', c='r')
#         sp.scatter3D([proc_filt_detections[arg_max,0]],[proc_filt_detections[arg_max,1]],[proc_filt_detections[arg_max,3]], marker='s', c='k', s=200)
#         pyl.title(dist_lim)
#         pyl.xlabel('X')
#         pyl.ylabel('Y')
#         sp.set_zlabel('rX')
#         pyl.show()
#
#
#     mask = np.ones(len(proc_filt_detections), dtype='bool')
#     mask[w[clust]] = False
#     proc_filt_detections = proc_filt_detections[mask]
#     proc_inds = proc_inds[mask]
#
#
# clust_detections = np.array(clust_detections)
# clust_stamps = stamps[np.array(clust_inds)]
clust_detections, clust_stamps = predictive_line_cluster(filt_detections, stamps, dmjds, dist_lim, min_samp, init_select_proc_distance=60, show_plot=False)
del stamps
gc.collect()

# trim on snr

print(len(clust_detections))
logging.info(f'Number of sources kept after brightness and peak location filtering: {len(clust_detections)}.')

w = np.where(clust_detections[:,5]>=trim_snr)
clust_detections = clust_detections[w]
clust_stamps = clust_stamps[w]
print(len(clust_detections))
logging.info(f'Number of sources kept after final SNR trim: {len(clust_detections)}.')


# In[15]:

#inv_vars = functional.pad(torch.tensor(0.5*np_inv_variances).cuda(), (khw, khw, khw, khw))
inv_vars = functional.pad(torch.tensor(0.5*np_inv_variances).to(device), (khw, khw, khw, khw))
cv[0,0,0] = inv_vars[0,0,0]

# # now apply a positional filter on the clust_detections to see if the likelihood minimimum is near the centre
# n_offsets = 5 # +- n_offsets in x and y
# n_o = n_offsets*2+1
#
# k = kernel.repeat((1, n_o*n_o, 1, 1, 1))
#
# danger_edges = [] # these are the indices of the maximal offsets in x and y
# for iy in range(n_o):
#     for ix in range(n_o):
#         i = iy*n_o+ix
#         shifts = (0, iy-n_offsets, ix-n_offsets)
#         k[0, i , :, :, :] = torch.roll(kernel[0,0], shifts=shifts, dims=[0,1,2])
#         if iy==0 or iy == n_o-1 or ix==0 or ix == n_o-1:
#             danger_edges.append(i)
#
#
#
# keeps = []
# for ir in range(len(rates)):
#
#     t1 = time.time()
#     w = np.where((clust_detections[:,2]==rates[ir][0]) & (clust_detections[:,3] == rates[ir][1]))
#     if len(w[0])==0: continue
#
#     for id in range(1, n_im):
#         shifts = (-round(dmjds[id]*rates[ir][1]), -round(dmjds[id]*rates[ir][0]))
#         c[0,0,id]  = torch.roll(im_datas[0,0,id], shifts=shifts, dims=[0,1])
#         cv[0,0,id] = torch.roll(inv_vars[0,0,id], shifts=shifts, dims=[0,1])
#
#     for id in w[0]:
#
#
#         (x,y) = clust_detections[id, :2]
#         x = int(x)#+khw
#         y = int(y)#+khw
#
#         K = k*clust_detections[id, 4]
#
#         diff = c[:,:,:,y:y+khw*2, x:x+khw*2].repeat((1, n_o*n_o, 1, 1, 1))
#         diff -= K
#         diff = diff**2
#         diff *= cv[:,:,:,y:y+khw*2, x:x+khw*2].repeat((1, n_o*n_o, 1, 1, 1))
#         l = torch.sum(diff, (0,2,3,4))
#         arg_min = torch.argmin(l)
#         min_ix = arg_min%n_o
#         min_iy = int((arg_min-min_ix)/n_o)
#
#         min_ix -= n_offsets
#         min_iy -= n_offsets
#         if arg_min not in danger_edges:
#             #print(arg_min,min_ix, min_iy,x,y,clust_detections[id, 4],clust_detections[id, 5])
#             #pyl.imshow(torch.sum(c[0,0,:,y:y+khw*2, x:x+khw*2],0).cpu())
#             #pyl.show()
#             #pyl.imshow(torch.sum(diff[0,arg_min,:,:,:],0).cpu())
#             #pyl.show()
#             keeps.append(id)
#         #pyl.plot(nb.cpu(), l.cpu()-torch.max(l.cpu()), marker='o')
#
#
# keeps = np.array(keeps)
# grid_detections = clust_detections[keeps]
# grid_stamps = clust_stamps[keeps]
# print(len(grid_detections), len(clust_detections))
# print(grid_detections[0])
# print(clust_detections[0])
# logging.info(f'Number of sources kept after the positional grid minimum search: {len(grid_detections)}.')

grid_detections, grid_stamps = position_filter(clust_detections, clust_stamps, im_datas, inv_vars, c, cv, kernel, dmjds, rates, khw)


# In[16]:


# trim on snr

print(len(grid_detections))
w = np.where(grid_detections[:,5]>=trim_snr)
final_detections = grid_detections[w]
final_stamps = grid_stamps[w]
#clust_stamps = clust_stamps[w]
print(len(final_detections))


# In[17]:


plants = []

pf_glob = f'{warps_dir}/{args.date}/{args.chip}/*{ref_im}*.plantList'
print(pf_glob)
logging.info(f'\n Using plantList {pf_glob}')
plant_files = glob.glob(pf_glob)
plant_files.sort()
with open(plant_files[0]) as han:
    data = han.readlines()
for i in range(1,len(data)):
    s = data[i].split()

    ra,dec,rate_ra,rate_dec = float(s[1]), float(s[2]), float(s[7]), float(s[8])
    x0,y0 = wcs.all_world2pix(ra,dec,0)
    x1,y1 = wcs.all_world2pix(ra+rate_ra/3600.0,dec+rate_dec/3600.0,0)

    rate_x = (x1-x0)*24.0
    rate_y = (y1-y0)*24.0

    plants.append([x0, y0, rate_x, rate_y, float(s[9]), 0,0,0,0])

plants = np.array(plants)
plants = plants[np.argsort(plants[:,4])]

for i in range(len(plants)):
    for j,det in enumerate([detections, filt_detections, clust_detections, final_detections]):

        dist_sq = np.sum((plants[i][:2] - det[:,:2])**2, axis=1)
        dist_rate_sq = np.sum((plants[i][2:4] - det[:,2:4])**2, axis=1)
        w = np.where((dist_sq<10**2) & (dist_rate_sq<160.**2))  # 4 and 30 seems like good values
        if len(w[0])>0:
            #print(dist[w])
            #print(dist_rate[w])
            #print()
            plants[i,5+j]=1
    print("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.2f} {} {} {} {} {}".format(np.min(dist_sq**0.5), np.min(dist_rate_sq**0.5), plants[i][0], plants[i][1], plants[i,2], plants[i,3], plants[i][4], plants[i,5], plants[i,6], plants[i,7], plants[i,8], len(w[0])))
    logging.info("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.2f} {} {} {} {} {}".format(np.min(dist_sq**0.5), np.min(dist_rate_sq**0.5), plants[i][0], plants[i][1], plants[i,2], plants[i,3], plants[i][4], plants[i,5], plants[i,6], plants[i,7], plants[i,8], len(w[0])))



# In[18]:


print('Number of plants found:', len(np.where(plants[:,-1])[0]))
logging.info('Number of plants found: '+str(len(np.where(plants[:,-1])[0])))

show_plot = False
if show_plot:
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


# In[21]:


fd_args = np.argsort(final_detections[:,5])[::-1]
final_detections = final_detections[fd_args]
final_stamps = final_stamps[fd_args]

try:
    os.makedirs(f'{saves_path}/{args.date}/results_{args.chip}/')
except:
    pass

logging.info(f'Saving to {saves_path}/{args.date}/results_{args.chip}/results_.txt')
with open(f'{saves_path}/{args.date}/results_{args.chip}/results_.txt', 'w+') as han:
    for i in range(len(final_detections)):
        (x,y,rx,ry,f,snr) = final_detections[i,:6]
        print(f'snr: {snr} flux: {f} x: {x} y: {y} x_v: {rx} y_v: {ry}', file=han)

with open(f'{saves_path}/{args.date}/results_{args.chip}/input.pars', 'w+') as han:
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
