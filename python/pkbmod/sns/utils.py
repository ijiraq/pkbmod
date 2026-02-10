#cell 3
#from astropy.wcs import WCS
import trippy, glob, logging, time
from astropy.io import fits
import numpy as np, scipy as sci
import torch
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rots = {"00": 2 , "01": 2 , "02": 2 , "03": 2 , "04": 2 , "05": 2 , "06": 2 , "07": 2 , "08": 2 , "09": 2 , "10": 2 , "11": 2 , "12": 2 , "13": 2 , "14": 2 , "15": 2 , "16": 2 , "17": 2 , "18": 0 , "19": 0 , "20": 0 , "21": 0 , "22": 0 , "23": 0 , "24": 0 , "25": 0 , "26": 0 , "27": 0 , "28": 0 , "29": 0 , "30": 0 , "31": 0 , "32": 0 , "33": 0 , "34": 0 , "35": 0 , "36": 2 , "37": 2 , "38": 0 , "39": 0}




def run_shifts(datas, inv_variances, rates, dmjds, min_snr, writeTestImages=False):
    n_im = len(datas[0,0,:])
    print('NUM IM', n_im)
    c = torch.zeros_like(datas)
    c[0,0,0] = datas[0,0,0]
    cv = torch.zeros_like(datas)
    cv[0,0,0] = inv_variances[0,0,0]

    # In[7]:


    snr_image = torch.zeros((1,1,len(rates), datas.size()[3], datas.size()[4]), dtype=torch.float16)
    alpha_image = torch.zeros((1,1,len(rates), datas.size()[3], datas.size()[4]), dtype=torch.float32)

    #rates=[[-200.6777606841237, 78.88276756451387]]
    for ir in range(len(rates)):
        for id in range(1, n_im):
            shifts = (-round(dmjds[id]*rates[ir][1]), -round(dmjds[id]*rates[ir][0]))
            c[0,0,id,] = torch.roll(datas[0,0,id], shifts=shifts, dims=[0,1])
            cv[0,0,id] = torch.roll(inv_variances[0,0,id], shifts=shifts, dims=[0,1])
        #C = functional.conv3d(c, kernel)
        #sums = torch.sum(functional.conv3d(c, ones,padding='same'), 2)

        ### these are set above
        PSI = c
        PHI = cv

        
        #median_alpha = torch.median(PSI/PHI, dim=2)[0] # doesn't seem to be used for anything

        alpha = torch.nansum(PSI/PHI, 2) # flux estimate
        alpha = torch.nan_to_num(alpha, 0.0)

        nu = torch.sum(PSI, 2)/torch.pow(torch.sum(PHI, 2) ,0.5) # SNR estimate
        nu = torch.nan_to_num(nu, -1.0)

        ## check to see if making all snr==inf go to zero fixes the infinities error you are getting
        nu[nu == float("Inf")] = 0
        ##
        
        where = nu[0,0] > min_snr
        inds = where.nonzero()


        snr_image[0,0,ir, inds[:,0], inds[:,1]] = nu[0,0,inds[:,0], inds[:,1]].cpu().type(torch.float16)
        alpha_image[0,0,ir, inds[:,0], inds[:,1]] = alpha[0,0,inds[:,0], inds[:,1]].cpu()/n_im ## flux per image not per stack

        ### don't actually need to calculate the likelihood, just use the SNR
        #PSI = torch.pow(PSI, 2)
        #PHI = torch.pow(PHI, 2)
        #PHI = torch.div(PHI, 2.)
        #PSI = torch.div(PSI, PHI) ## now likelihoods
        #likelihood = torch.sum(PSI, 2)


        gc.collect()
        torch.cuda.empty_cache()

        #print(ir,len(rates))

        if ir==19 and writeTestImages: ## wes testing
            #im = fits.PrimaryHDU(np.array(likelihood.cpu()))
            #im.writeto('like_test.fits', overwrite=True)
            l = np.array(nu.cpu())
            #w = np.where((l<=min_snr)|np.isnan(l))
            #l[w]=0.0
            im = fits.PrimaryHDU(l)
            im.writeto('snr_test.fits', overwrite=True)
            #

    print('Max per image flux of candidates:' , torch.max(alpha_image))
    print('Max per image snr of candidates:' , torch.max(snr_image))
    logging.info('Max per image flux of candidates:'+str(torch.max(alpha_image)))
    logging.info('Max per image snr of candidates:'+str(torch.max(snr_image)))

    del c, cv, datas, inv_variances, PSI, PHI
    gc.collect()
    torch.cuda.empty_cache()

    return snr_image, alpha_image


def trim_negative_snr(snr_image, alpha_image, sort_inds, n_keep, rates, A, B):
    # trim the negative SNR sources. The reason these show up is because the likelihood formalism sucks
    idx,idy = np.meshgrid(np.arange(B), np.arange(A))
    idx = idx.reshape(A*B)
    idy = idy.reshape(A*B)
    for n in range(n_keep):
        s = sort_inds[0,0,n,:,:].reshape(A*B)
        SNR = snr_image[0,0,s,idy,idx]
        alpha = alpha_image[0,0,s,idy,idx]

        where = SNR>0
        inds = where.nonzero()[:,0]

        if n == 0:
            keeps = np.zeros((len(inds),7),dtype='float32')
            keeps[:,0] = idx[inds]
            keeps[:,1] = idy[inds]
            keeps[:,2] = rates[s.reshape(A*B)[inds], 0]
            keeps[:,3] = rates[s.reshape(A*B)[inds], 1]
            keeps[:,4] = alpha.reshape(A*B)[inds]
            keeps[:,5] = SNR.reshape(A*B)[inds]
        else:
            nkeeps = np.zeros((len(inds),7),dtype='float32')
            nkeeps[:,0] = idx[inds]
            nkeeps[:,1] = idy[inds]
            nkeeps[:,2] = rates[s.reshape(A*B)[inds], 0]
            nkeeps[:,3] = rates[s.reshape(A*B)[inds], 1]
            nkeeps[:,4] = alpha.reshape(A*B)[inds]
            nkeeps[:,5] = SNR.reshape(A*B)[inds]
            keeps = np.concatenate([keeps, nkeeps])


    print(f'Keeping {len(keeps)} candidates')
    logging.info(f'Keeping {len(keeps)} candidates')

    detections= np.array(keeps)
    del keeps, idx, idy
    return detections

def trim_negative_flux(detections):
    #import pickle
    #with open('junk.pickle','bw') as han:
    #    pickle.dump(detections, han)
    #exit()
    # trim the flux negative sources

    pos = np.where(detections[:,4]>0)
    detections = detections[pos]
    print(f'Keeping {len(detections)} positive flux candidates')
    logging.info(f'Keeping {len(detections)} positive flux candidates')
    return detections

def brightness_filter(im_datas, inv_vars, c, cv, kernel, dmjds, rates, detections, khw, n_im, n_bright_test = 10, test_high = 1.15, test_low = 0.85):
    nb_ref = torch.tensor(10.0**np.linspace(np.log10(test_low), np.log10(test_high), n_bright_test)).to(device) # .cuda()

    for ir in range(len(rates)):

        t1 = time.time()
        w = np.where((detections[:,2]==rates[ir][0]) & (detections[:,3] == rates[ir][1]))

        for id in range(1, n_im):
            shifts = (-round(dmjds[id]*rates[ir][1]), -round(dmjds[id]*rates[ir][0]))
            c[0,0,id]  = torch.roll(im_datas[0,0,id], shifts=shifts, dims=[0,1])
            cv[0,0,id] = torch.roll(inv_vars[0,0,id], shifts=shifts, dims=[0,1])

        arg_mins = torch.zeros(len(detections), dtype=torch.uint32)
        for id in w[0]:
            (x,y) = detections[id, :2]
            x = int(x) + khw
            y = int(y) + khw
            nb = nb_ref*detections[id, 4] # array of scaled brightnesses in steps of brightness*test_low to brightness*test_high

            k = kernel.repeat((1, n_bright_test, 1, 1, 1))
            for ib in range(nb.size()[0]):
                k[:, ib, :, :, :]*=nb[ib]

            diff = c[:,:,:,y-khw:y+khw, x-khw:x+khw].repeat((1, n_bright_test, 1, 1, 1))
            diff -= k
            diff = diff*diff
            diff *= cv[:,:,:,y-khw:y+khw, x-khw:x+khw].repeat((1, n_bright_test, 1, 1, 1))
            l = torch.sum(diff, (0,2,3,4))

            arg_mins[id] = torch.argmin(l)

        arg_mins_cpu = arg_mins.cpu()

        W = np.where((arg_mins_cpu!=0) & (arg_mins_cpu!=(n_bright_test-1)))
        print(f'{ir+1}/{len(rates)}, pre: {len(w[0])}, post: {len(W[0])},  in time {time.time()-t1}')
        if ir == 0:
            keeps = W[0]
        else:
            keeps = np.concatenate([keeps, W[0]])
    print(len(keeps))
    print(np.max(keeps))
    logging.info(f'Number kept after brightness filter {len(keeps)} of {len(detections)} total detections.')

    return keeps

def create_stamps(im_datas, im_masks, c, cv, dmjds, rates, filt_detections, khw):
    mean_stamps = []
    #med_stamps = []
    indices = []
    saved= False
    for ir in range(len(rates)):

        # these are required to reset from the nans below
        c[0,0,0] = im_datas[0,0,0]
        cv[0,0,0] = im_masks[0,0,0]

        t1 = time.time()
        w = np.where((filt_detections[:,2]==rates[ir][0]) & (filt_detections[:,3] == rates[ir][1]))

        for id in range(1, len(dmjds)):
            shifts = (-round(dmjds[id]*rates[ir][1]), -round(dmjds[id]*rates[ir][0]))
            c[0,0,id]  = torch.roll(im_datas[0,0,id], shifts=shifts, dims=[0,1])
            cv[0,0,id] = torch.roll(im_masks[0,0,id], shifts=shifts, dims=[0,1]) # mask values with 1 are GOOD pixels
        mean_stamp_frame = torch.sum(c, 2)

        #c[cv == 0] = float('nan')
        #med_stamp_frame = torch.nanmedian(c, 2)[0]

        mask_frame = torch.sum(cv, 2)

        for iw in w[0]:
            x,y = filt_detections[iw,:2].astype('int')
            #print(x,y, iw, khw)
            mean_stamp = mean_stamp_frame[0,0,y:y+khw*2+1, x:x+khw*2+1]

            """
            med_sections = c[0,0,:,y:y+khw*2+1, x:x+khw*2+1].cpu()
            mask_sections = cv[0,0,:,y:y+khw*2+1, x:x+khw*2+1].cpu()
            mask_sections[np.where((np.isnan(mask_sections)) | (np.isinf(mask_sections)))] = 0

            # mask_sections==0 are bad pixels
            # numpy masks value == 1 are bad pixels
            masked_med_sections = ma.array(med_sections, mask= 1-mask_sections)
            med_stamps.append(ma.median(masked_med_sections,axis=0).filled(0.0))
            """

            mask = np.copy(mask_frame[0,0,y:y+khw*2+1, x:x+khw*2+1].cpu())
            mask[np.where((np.isnan(mask)) | (np.isinf(mask)))] = 0.0
            np.clip(mask, 1., len(dmjds), out=mask)
            mean_stamps.append(np.copy(mean_stamp.cpu())/mask )

            indices.append(iw)

    indices = np.array(indices)
    mean_stamps = np.array(mean_stamps)[indices]
    #med_stamps = np.array(med_stamps)[indices]

    del mask, mean_stamp, mean_stamp_frame, mask_frame
    gc.collect()
    torch.cuda.empty_cache()

    return mean_stamps


# trim the ones with peak offset more than peak_offset_max pixels
def peak_offset_filter(stamps, filt_detections, peak_offset_max):
    (N, a, b) = stamps.shape
    (gx,gy) = np.meshgrid(np.arange(b), np.arange(a))
    gx = gx.reshape(a*b)
    gy = gy.reshape(a*b)
    rs_stamps = stamps.reshape(N,a*b)
    args = np.argmax(rs_stamps,axis=1)
    X = gx[args]
    Y = gy[args]
    radial_d = ((X-b/2)**2+(Y-a/2)**2)**0.5
    w = np.where(radial_d<peak_offset_max)
    filt_detections = filt_detections[w]
    stamps = stamps[w]

    return stamps, filt_detections

# do predictive line clustering
def predictive_line_cluster(filt_detections, stamps, dmjds, dist_lim, min_samp=2, init_select_proc_distance=60, show_plot=False):

    proc_filt_detections = np.copy(filt_detections)

    proc_inds = np.arange(len(proc_filt_detections))
    clust_detections, clust_inds = [], []

    #for i in range(0,10):
    while len(proc_filt_detections)>0:

        arg_max = np.argmax(proc_filt_detections[:,5]) # 5 - max on SNR, 4 is flux
        #pyl.imshow(normer(stamps[arg_max]))
        #pyl.show()
        #exit()
        x_o, y_o, rx_o, ry_o, f_o, snr_o = proc_filt_detections[arg_max, :6]

        #this secondary where command is necessary because of memory overflows in large detection lists
        w = np.where( (proc_filt_detections[:,0] > proc_filt_detections[arg_max,0]-55) & (proc_filt_detections[:,0] < proc_filt_detections[arg_max,0] +init_select_proc_distance)
                     & (proc_filt_detections[:,1] > proc_filt_detections[arg_max,1]-55) & (proc_filt_detections[:,1] < proc_filt_detections[arg_max,1] +init_select_proc_distance))

        W = np.where( ((proc_filt_detections[w[0],0]-proc_filt_detections[arg_max,0])**2 + (proc_filt_detections[w[0],1]-proc_filt_detections[arg_max,1])**2) < init_select_proc_distance**2)
        w = w[0][W[0]]

        fd_subset = proc_filt_detections[w]

        drx = fd_subset[:,2] - rx_o
        dry = fd_subset[:,3] - ry_o
        dt = dmjds # just for clarity

        x_n, y_n = x_o - drx*dt[-1], y_o - dry*dt[-1] # predicted centroid  position of secondary detection shifted at the differential wrong rate.

        dx, dy = (x_n - x_o), (y_n - y_o) # predicted centroid shifted such that best detection is now at origin
        dxp = dx*fd_subset[:,1]
        dyp = dy*fd_subset[:,0]
        xm = x_n*y_o
        ym = y_n*x_o
        dx2 = dx**2
        dy2 = dy**2
        top = np.abs(dyp - dxp + xm - ym)
        bottom = np.sqrt( dx2 + dy2 )
        dist = top/bottom
        #dist = np.abs( (y_n-y_o)*fd_subset[:, 0] - (x_n-x_o)*fd_subset[:,1] + x_n*y_o - y_n*x_o    ) / np.sqrt( (x_n-x_o)**2 + (y_n-y_o)**2)

        vert_distance = np.abs(y_n - fd_subset[:,1])
        hor_distance = np.abs(x_n - fd_subset[:,0])


        clust = np.where( (dist<dist_lim) | (np.isnan(dist)) | ((dist<dist_lim) & (drx==0) & (dry==0)))
        not_clust = np.where(~( (dist<dist_lim) | (np.isnan(dist)) | ((dist<dist_lim) & (drx==0) & (dry==0))) )

        #clust = np.where( ( (hor_distance < dist_lim_x)&(vert_distance<dist_lim_y) ) | (np.isnan(dist)) | ((dist<dist_lim)&(dx==0)&(dy==0) ))
        #not_clust = np.where( ~( ( (hor_distance < dist_lim_x)&(vert_distance<dist_lim_y) ) | (np.isnan(dist)) | ((dist<dist_lim)&(dx==0)&(dy==0) ) ))

        if len(clust[0])>=min_samp:
            clust_detections.append(proc_filt_detections[arg_max])
            clust_inds.append(proc_inds[arg_max])

        if show_plot:
            fig = pyl.figure(1)
            sp = fig.add_subplot(111, projection='3d')
            sp.scatter3D(fd_subset[clust,0], fd_subset[clust,1], fd_subset[clust,3], marker='o', c='b')
            sp.scatter3D(fd_subset[not_clust,0], fd_subset[not_clust,1], fd_subset[not_clust,3], marker='^', c='r')
            sp.scatter3D([proc_filt_detections[arg_max,0]],[proc_filt_detections[arg_max,1]],[proc_filt_detections[arg_max,3]], marker='s', c='k', s=200)
            pyl.title(dist_lim)
            pyl.xlabel('X')
            pyl.ylabel('Y')
            sp.set_zlabel('rX')
            pyl.show()


        mask = np.ones(len(proc_filt_detections), dtype='bool')
        mask[w[clust]] = False
        proc_filt_detections = proc_filt_detections[mask]
        proc_inds = proc_inds[mask]


    clust_detections = np.array(clust_detections)
    clust_stamps = stamps[np.array(clust_inds)]

    return clust_detections, clust_stamps

def position_filter(clust_detections, clust_stamps, im_datas, inv_vars, c, cv, kernel, dmjds, rates, khw, n_offsets=5):

    # now apply a positional filter on the clust_detections to see if the likelihood minimimum is near the centre
    #n_offsets = 5 # +- n_offsets in x and y
    n_o = n_offsets*2+1

    k = kernel.repeat((1, n_o*n_o, 1, 1, 1))

    danger_edges = [] # these are the indices of the maximal offsets in x and y
    for iy in range(n_o):
        for ix in range(n_o):
            i = iy*n_o+ix
            shifts = (0, iy-n_offsets, ix-n_offsets)
            k[0, i , :, :, :] = torch.roll(kernel[0,0], shifts=shifts, dims=[0,1,2])
            if iy==0 or iy == n_o-1 or ix==0 or ix == n_o-1:
                danger_edges.append(i)


    cv[0,0,0] = inv_vars[0,0,0]

    keeps = []
    for ir in range(len(rates)):

        t1 = time.time()
        w = np.where((clust_detections[:,2]==rates[ir][0]) & (clust_detections[:,3] == rates[ir][1]))
        if len(w[0])==0: continue

        for id in range(1, len(dmjds)):
            shifts = (-round(dmjds[id]*rates[ir][1]), -round(dmjds[id]*rates[ir][0]))
            c[0,0,id]  = torch.roll(im_datas[0,0,id], shifts=shifts, dims=[0,1])
            cv[0,0,id] = torch.roll(inv_vars[0,0,id], shifts=shifts, dims=[0,1])

        for id in w[0]:


            (x,y) = clust_detections[id, :2]
            x = int(x)#+khw
            y = int(y)#+khw

            K = k*clust_detections[id, 4]

            diff = c[:,:,:,y:y+khw*2, x:x+khw*2].repeat((1, n_o*n_o, 1, 1, 1))
            diff -= K
            diff = diff**2
            diff *= cv[:,:,:,y:y+khw*2, x:x+khw*2].repeat((1, n_o*n_o, 1, 1, 1))
            l = torch.sum(diff, (0,2,3,4))
            arg_min = torch.argmin(l)
            min_ix = arg_min%n_o
            min_iy = int((arg_min-min_ix)/n_o)

            min_ix -= n_offsets
            min_iy -= n_offsets
            if arg_min not in danger_edges:
                #print(arg_min,min_ix, min_iy,x,y,clust_detections[id, 4],clust_detections[id, 5])
                #pyl.imshow(torch.sum(c[0,0,:,y:y+khw*2, x:x+khw*2],0).cpu())
                #pyl.show()
                #pyl.imshow(torch.sum(diff[0,arg_min,:,:,:],0).cpu())
                #pyl.show()
                keeps.append(id)
            #pyl.plot(nb.cpu(), l.cpu()-torch.max(l.cpu()), marker='o')


    keeps = np.array(keeps)
    grid_detections = clust_detections[keeps]
    grid_stamps = clust_stamps[keeps]
    print(len(grid_detections), len(clust_detections))
    print(grid_detections[0])
    print(clust_detections[0])
    logging.info(f'Number of sources kept after the positional grid minimum search: {len(grid_detections)}.')

    return grid_detections, grid_stamps


def read_bitmask(bitmask_fn, flags_fn):
    with open(bitmask_fn) as han:
        data = han.readlines()

    bitmask = {}
    for i in range(len(data)):
        if '#' in data[i]: continue

        s = data[i].split(': ')
        key, val = s[0], int(float(s[1]))
        bitmask[key] = val


    with open(flags_fn) as han:
        data = han.readlines()
        
    flag_keys = []
    for i in range(len(data)):
        if '#' in data[i]: continue
        key = data[i].split()[0]
        flag_keys.append(key)

    return (bitmask, flag_keys)
        
