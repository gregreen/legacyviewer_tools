#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as units

import json


def sph2cart(lon, lat):
    lon = lon.to('rad').value
    lat = lat.to('rad').value
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)
    return np.stack([x,y,z], axis=-1)


def cart2sph(x):
    assert x.shape[-1] == 3
    x,y,z = [x[...,k] for k in range(3)]
    r = np.sqrt(x**2 + y**2 + z**2)
    lon = np.arctan2(y, x)
    lat = np.arcsin(z/r)
    return r, lon, lat


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
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Generate WCS frame specifications for video.',
        add_help=True
    )
    parser.add_argument(
        '--coords-start', '-c0',
        metavar='DEG',
        type=float,
        nargs=3,
        required=True,
        help='Longitude, latitude, image width (all in deg) of first frame.'
    )
    parser.add_argument(
        '--coords-end', '-c1',
        metavar='DEG',
        type=float,
        nargs=3,
        required=True,
        help='Longitude, latitude, image width (all in deg) of last frame.'
    )
    parser.add_argument(
        '--n-frames', '-n',
        metavar='N',
        type=int,
        required=True,
        help='# of frames.'
    )
    parser.add_argument(
        '--layers', '-l',
        metavar='LAYER[BAND]',
        type=str,
        nargs=3,
        required=True,
        help='Layers to use for R,G,B channels. Eg., decaps2[0] for DECaPS2 g.'
    )
    parser.add_argument(
        '--coordsys',
        metavar='galactic/equatorial',
        type=str,
        choices=('galactic','equatorial'),
        default='galactic',
        help='Coordinate system to use.'
    )
    parser.add_argument(
        '--resolution', '-r',
        type=str,
        nargs='+',
        default='480p',
        help='Resolution, in pixels (width, height), or (480p, 720p, 1080p).'
    )
    args = parser.parse_args()

    # Parse image resolution
    img_shape_opts = {
        '480p':(848,480),
        '720p':(1280,720),
        '1080p':(1920,1080)
    }

    if len(args.resolution) == 1:
        img_shape = img_shape_opts.get(args.resolution[0])
    elif len(args.resolution) == 2:
        img_shape = [int(r) for r in args.resolution]
    else:
        img_shape = None

    if img_shape is None:
        print(f'--resolution must be a pair of integers '
              'or one of {img_shape_opts.keys()}.')

    # Image scale, as a function of frame
    img_scale = np.exp(np.linspace(
        np.log(args.coords_start[2]),
        np.log(args.coords_end[2]),
        args.n_frames
    )) * units.deg

    # Pixel scale
    pix_scale = img_scale / img_shape[0]

    # Fractional path distance, as a function of frame
    t = np.linspace(0., 1., args.n_frames)
    inv_zoom = img_scale[-1] / img_scale[0] + 1e-10
    s = 1/(1-inv_zoom) * (1 - np.exp(np.log(inv_zoom)*t))

    # Cartesian coordinates of start and ending points
    x0 = sph2cart(*(args.coords_start[:2]*units.deg))
    x1 = sph2cart(*(args.coords_end[:2]*units.deg))
    xt = x0[None,:] + s[:,None] * (x1 - x0)[None,:]
    _, lon_t, lat_t = cart2sph(xt)

    # Generate WCS header for each frame
    wcs = []
    for t in range(args.n_frames):
        w = get_wcs_dict(
            lon_t[t], lat_t[t],
            pix_scale[t],
            img_shape,
            galactic=(args.coordsys=='galactic')
        )
        w['layer'] = ['decaps2[2]','decaps2[1]','decaps2[0]']
        wcs.append(w)

    txt = json.dumps(wcs, indent=2)
    print(txt)
    #with open(args.fname, 'w') as f:
    #    json.dump(wcs, f, indent=2)

    return 0

if __name__ == '__main__':
    main()

