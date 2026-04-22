import numpy as np
import logging

def get_shift_rates_from_angle(ecl_ang, mjds, rate_lims, ang_lims, fwhm, rate_fwhm_grid_step):
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
    logging.info("Angles (min, max, mid, delta): {min_ang}, {max_ang}, {med_ang}, {d_ang}")


    seeing = fwhm
    logging.info(f"FWHM {seeing} pixels")


    dh = (np.max(mjds) - np.min(mjds)) # days, need to take the np.max and np.min because images aren't necessarily in order of increase time.                                                                                                                                       
    drate = rate_fwhm_grid_step*seeing/dh  # 0.75 seems to be a good sweet spot                                                           

    ang_steps_h = np.linspace(med_ang, max_ang+0.0, 150)
    ang_steps_l = np.linspace(min_ang-0.0, med_ang, 150)

    # rx,ry = np.cos(ecl_ang*d2r)*np.min(np.array(rate_lims)), np.sin(ecl_ang*d2r)*np.min(np.array(rate_lims))
    rates = []

    current_rate = 24*np.min(np.array(rate_lims))
    while current_rate < 24*np.max(np.array(rate_lims)):
        n_x = np.cos(ang_steps_h)*current_rate  # + max_x                                                                                   
        n_y = np.sin(ang_steps_h)*current_rate  # + max_y                                                                                   

        dist_rates = ( ((n_x - n_x[0])**2 + (n_y - n_y[0])**2)**0.5 / drate).astype('int')
        unique_dist_rates = np.unique(dist_rates)
        for ind in unique_dist_rates:
            w = np.where(dist_rates == ind)
            rates.append([n_x[w[0][0]], n_y[w[0][0]]])


        n_x = np.cos(ang_steps_l[::-1])*current_rate  # + max_x                                                                             
        n_y = np.sin(ang_steps_l[::-1])*current_rate  # + max_y                                                                             
        dist_rates = (((n_x - n_x[0])**2 + (n_y - n_y[0])**2)**0.5 / drate).astype('int')
        unique_dist_rates = np.unique(dist_rates)
        for ind in unique_dist_rates:
            if ind == 0: continue
            w = np.where(dist_rates == ind)
            rates.append([n_x[w[0][0]], n_y[w[0][0]]])

        current_rate += drate

    return np.array(rates)
