from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np


def calc_ecliptic_angle(wcs, A, B, retrograde=True):
    """
    Returns the retrograde or prograde direction angle of the ecliptic in the provided image.
    """

    d2r = np.pi/180.0
    r2d = 180.0/np.pi
    
    y,x = A/2, B/2

    (ra, dec) = wcs.all_pix2world(x, y, 0)

    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    e = c.geocentricmeanecliptic

    E = SkyCoord(lon=e.lon+0.01*u.deg, lat=e.lat, frame='geocentricmeanecliptic')

    C = E.icrs

    RA, DEC = C.ra.deg, C.dec.deg

    X, Y = wcs.all_world2pix(RA, DEC, 0)
    
    pix_ang = np.arctan2( (Y-y)*d2r, (X-x)*d2r)*r2d

    if retrograde:
        if RA>ra: return pix_ang+180.
        else: return pix_ang
    else:
        if RA<ra: return pix_ang+180.
        else: return pix_ang
    
