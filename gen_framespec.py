#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as units

import json


def get_wcs_dict(lon0, lat0, pixscale, shape, galactic=True):
    pixscale_deg = pixscale.to('deg').value
    wcs = {
        'NAXIS': 2,
        'NAXIS1': shape[0],
        'NAXIS2': shape[1],
        'CTYPE1': 'GLON-TAN' if galactic else 'RA---TAN',
        'CTYPE2': 'GLAT-TAN' if galactic else 'DEC--TAN',
        'CRVAL1': lon0.to('deg').value,
        'CRVAL2': lat0.to('deg').value,
        'CRPIX1': shape[0]/2 + 0.5,
        'CRPIX2': shape[1]/2 + 0.5,
        'CD1_1': -pixscale_deg,
        'CD1_2': 0.,
        'CD2_1': 0.,
        'CD2_2': pixscale_deg,
        'IMAGEW': shape[0],
        'IMAGEH': shape[1]
    }
    return wcs


def main():
    fname = 'zoom_l317_b-4.json'
    lon,lat = [317.15, -4.15] * units.deg
    galactic = True
    img_scale_0 = 10. * units.deg
    img_scale_1 = 1000 * units.arcsec
    img_shape = (1920, 1080)
    n_frames = 15

    pixscale0 = img_scale_0 / img_shape[0]
    pixscale1 = img_scale_1 / img_shape[0]

    pixscale = np.exp(np.linspace(
        np.log(pixscale0.to('deg').value),
        np.log(pixscale1.to('deg').value),
        n_frames
    )) * units.deg

    wcs = []
    for s in pixscale:
        w = get_wcs_dict(lon, lat, s, img_shape, galactic=galactic)
        w['layer'] = ['decaps2[2]','decaps2[1]','decaps2[0]']
        wcs.append(w)

    with open(fname, 'w') as f:
        json.dump(wcs, f, indent=2)
    
    return 0

if __name__ == '__main__':
    main()

