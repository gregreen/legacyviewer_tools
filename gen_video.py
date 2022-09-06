#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
import requests
import json

import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord, ICRS, Galactic
from astropy_healpix import HEALPix
import astropy.units as units

import reproject
from reproject import reproject_interp, reproject_exact, reproject_adaptive
from reproject.mosaicking import reproject_and_coadd

from scipy.optimize import minimize_scalar

from PIL import Image

from tqdm import tqdm


cutout_dir = os.environ.get('CUTOUT_DIR', 'cutouts/')
#n_cores = os.environ.get('REPROJ_NCPU', 1)


def fetch_cutout_healpix(layer, nside, pix_idx,
                         spacing=0.45, overwrite=False,
                         verbose=False):
    fname = os.path.join(
        cutout_dir,
        f'{layer:s}_nside{nside:d}_pix{pix_idx:d}_spacing{spacing:.3f}.fits'
    )

    hpix = HEALPix(nside=nside)
    ra,dec = hpix.healpix_to_lonlat(pix_idx)
    ra = ra.to('deg').value
    dec = dec.to('deg').value

    cutout_npix = 256
    pixscale = 3600*np.degrees((1+spacing)/cutout_npix * np.sqrt(4*np.pi/12)/nside)
    if verbose:
        print(f'pixscale = {pixscale} arcsec = {pixscale/3600} deg')

    if (not os.path.exists(fname)) or overwrite:
        query_params = f'layer={layer:s}&ra={ra:f}&dec={dec:f}&pixscale={pixscale:f}'
        url = f'https://www.legacysurvey.org/viewer/fits-cutout/?{query_params}'
        if verbose:
            print(f'Fetching URL: {url} -> {fname}')
        response = requests.get(url, stream=True)
        with open(fname, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
    else:
        if verbose:
            print(f'{fname} already exists.')

    with fits.open(fname, mode='readonly') as f:
        hdu = f[0].copy()

    return hdu


def fix_legacyviewer_header(header):
    if header['NAXIS'] == 3:
        header['NAXIS'] = 2
        n_bands = header.pop('NAXIS3')
    else:
        n_bands = 1
    return n_bands


def get_hdulist_for_band(hdulist, band):
    hdulist_b = []

    for hdu in hdulist:
        if hdu.header['NAXIS'] == 3:
            hdu.header['NAXIS'] = 2
            hdu.header.pop('NAXIS3')
            d = hdu.data[band,:,:].copy()
            hdulist_b.append(fits.ImageHDU(header=hdu.header, data=d))
        else:
            hdulist_b.append(hdu)

    return hdulist_b


def find_gamma_stretch(img, target, reduce=np.nanmedian, gamma_bounds=(0.25,1.75)):
    if reduce == np.nanmedian:
        gamma_opt = np.log(target) / np.log(np.nanmedian(img))
        return np.clip(gamma_opt, *gamma_bounds)

    def f(gamma):
        return (reduce(img**gamma) - target)**2

    res = minimize_scalar(f, bounds=gamma_bounds, method='bounded')
    gamma_opt = res.x

    return gamma_opt


def healpix_mosaic(layer, wcs, band=0,
                   spacing=0.45, nside_buffer=0.2,
                   reproj_method='interp', verbose=False):
    # Determine nside from WCS pixel scale
    cutout_npix = 256
    wcs_pixscale = np.min(proj_plane_pixel_scales(wcs))
    if verbose:
        print(f'Desired pixel scale: {wcs_pixscale} deg')
    nside = (
        np.sqrt(4*np.pi/12)
      * (1+spacing) / (cutout_npix*np.radians(wcs_pixscale))
      * (1+nside_buffer)
    )
    nside = 2**(int(np.ceil(np.log2(nside))))
    if verbose:
        print(f' -> Using nside={nside} cutouts.')

    # Map each pixel in output to an equatorial SkyCoord
    shape_out = wcs.pixel_shape
    out_x,out_y = np.meshgrid(np.arange(shape_out[0]), np.arange(shape_out[1]))
    out_coords = SkyCoord.from_pixel(out_x.flat, out_y.flat, wcs)
    hpix = HEALPix(nside=nside, frame=ICRS())
    hpix_idx = np.unique(hpix.skycoord_to_healpix(out_coords))
    if verbose:
        print(f'{len(hpix_idx)} HEALPix pixels covered: {hpix_idx}')
    #g = hpix.healpix_to_skycoord(hpix_idx).transform_to('galactic')
    #plt.scatter(g.l.deg, g.b.deg)
    #plt.show()

    hpix_idx_iter = hpix_idx
    if verbose:
        hpix_idx_iter = tqdm(hpix_idx)

    hdulist = [
        fetch_cutout_healpix(layer, nside, i, spacing=spacing, verbose=verbose)
        for i in hpix_idx_iter
    ]
    hdulist = get_hdulist_for_band(hdulist, band)

    shape_out = wcs.pixel_shape[::-1]

    if reproj_method == 'exact':
        reproj = reproject_exact
        kwargs = dict()
    elif reproj_method == 'adaptive':
        reproj = reproject_adaptive
        kwargs = dict()
    elif reproj_method == 'interp':
        reproj = reproject_interp
        kwargs = dict(order=1)

    if verbose:
        print(f'Reprojecting images using {reproj.__name__} ...')

    img,_ = reproject_and_coadd(
        fits.HDUList(hdulist),
        wcs,
        shape_out=shape_out,
        reproject_function=reproj,
        **kwargs
    )

    if verbose:
        print(f'Image shape = {img.shape}')

    return img


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
    wcs = WCS(wcs)
    return wcs


def save_image(img, fname,
               subtract_min=True,
               vmax_pct=99.5, vmax_abs=None,
               gamma=0.5, norm_last=None, norm_momentum=0.9,
               vmax_last=None, vmax_momentum=0.9,
               verbose=False):
    # Ensure that there are no negative values
    if subtract_min:
        img = img - np.nanmin(np.nanmin(img, axis=0), axis=0)[None,None,:]
    else:
        img[img < 0] = 0.

    # Determine vmax
    vmax = 1.

    if vmax_pct is not None:
        img_flat = np.reshape(img, (-1,img.shape[2]))
        vmax = np.nanpercentile(img_flat, vmax_pct, axis=0)[None,None,:]
    elif vmax_abs is not None:
        vmax = np.array(vmax_abs)[None,None,:]

    if vmax_last is not None:
        vmax = vmax_momentum*vmax_last + (1-vmax_momentum)*vmax

    # Apply vmax
    #print('vmax', vmax)
    img = img / vmax

    # Apply gamma stretch to norm of each pixel value
    img_norm = np.linalg.norm(img, axis=2)
    img_norm[img_norm==0] = 1.e-5

    if norm_last is None:
        gamma_opt = gamma
    else:
        # Search for optimal gamma stretch, balancing continuity in image
        # appearance and matching target gamma stretch.
        target_norm = np.nanmedian(img_norm)**gamma
        target_norm = norm_momentum*norm_last + (1-norm_momentum)*target_norm
        gamma_opt = find_gamma_stretch(img_norm, target_norm)
        #if not np.isfinite(gamma_opt):
        #    print(np.nanmedian(img_norm))
        #    print(target_norm)
        #    print(np.nanpercentile(img_norm, [1., 10., 25., 50., 75., 90., 99.]))
        #if verbose:
        print(f'Applying gamma={gamma_opt} stretch to norm of each pixel.')

    img_norm_reduced = np.nanmedian(img_norm)**gamma_opt

    img = img * (img_norm**(gamma_opt-1))[:,:,None]
    #img = img**gamma

    # Remove NaNs from image
    img[~np.isfinite(img)] = 0.

    # Convert image to uint8
    img *= 255.
    img = np.clip(img, 0, 255).astype('uint8')

    if img.shape[0] == 1:
        im = Image.fromarray(img[::-1,:,0], mode='L')
    else:
        im = Image.fromarray(img[::-1,:,:], mode='RGB')

    im.save(fname)

    # Return vmax and average pixel norm in image
    return vmax, img_norm_reduced


def load_wcs_list(fname):
    with open(fname, 'r') as f:
        d = json.load(f)
    layer_list = [wcs.pop('layer') for wcs in d]
    wcs_list = [WCS(wcs) for wcs in d]
    return layer_list, wcs_list


def save_fits(fname, img, wcs):
    header = wcs.to_header()
    hdu = fits.ImageHDU(data=img, header=header)
    with fits.open(fname, mode='append') as f:
        f.append(hdu)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Generate video frames of astronomical surveys.',
        add_help=True
    )
    parser.add_argument(
        'framespec',
        type=str,
        help='JSON containing a list of WCS dictionaries.'
    )
    parser.add_argument(
        '--img-outpattern',
        type=str,
        help='Image output filename, in fstring format, taking frame index.'
    )
    parser.add_argument(
        '--fits-out',
        type=str,
        help='FITS output filename.'
    )
    parser.add_argument(
        '--reproject-method',
        type=str,
        default='interp',
        choices=('interp','adaptive','exact'),
        help='Reprojection method: interp, adaptive or exact.'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Smoothly change vmax and gamma (1=no change, 0=instant change).'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.5,
        help='Gamma stretch to apply to the images.'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output.'
    )
    args = parser.parse_args()

    vmax_last, norm_last = None, None
    layers_last = None

    layer_list, wcs_list = load_wcs_list(args.framespec)

    for i,(layers,wcs) in enumerate(zip(tqdm(layer_list),wcs_list)):
        if layers != layers_last:
            layers_last = layers
            vmax_last, norm_last = None, None
            print('Reset vmax_last and norm_last.')

        img = []
        for l in layers:
            # Determine layer name, and which band to extract
            l,b = l.split('[')
            b = int(b.rstrip(']'))
            # Generate image of selected (layer,band)
            img.append(healpix_mosaic(
                l, wcs, band=b,
                reproj_method=args.reproject_method,
                verbose=args.verbose
            ))

        # Combine bands
        img = np.stack(img, axis=2)

        if args.verbose:
            print(f'Combined image shape: {img.shape}')

        if args.fits_out is not None:
            save_fits(args.fits_out, img, wcs)

        if args.img_outpattern is not None:
            fname = args.img_outpattern.format(i)
            vmax_last, norm_last = save_image(
                img, fname,
                subtract_min=False,
                vmax_pct=99.5,
                vmax_last=vmax_last,
                norm_last=norm_last,
                gamma=args.gamma,
                vmax_momentum=args.momentum,
                verbose=args.verbose
            )

    return 0

if __name__ == '__main__':
    main()

