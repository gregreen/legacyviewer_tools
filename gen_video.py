#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
import requests

import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord, ICRS, Galactic
from astropy_healpix import HEALPix
import astropy.units as units

import reproject
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd

from PIL import Image

from tqdm import tqdm


def fetch_cutout_healpix(layer, nside, pix_idx,
                         spacing=0.45, overwrite=False,
                         verbose=False):
    fname = os.path.join(
        'cutouts',
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


def get_hdulist_per_band(hdulist):
    band_hdulist = None

    for hdu in hdulist:
        if band_hdulist is None:
            n_bands = hdu.header['NAXIS3']
            band_hdulist = [[] for b in range(n_bands)]

        if hdu.header['NAXIS'] == 3:
            hdu.header['NAXIS'] = 2
            hdu.header.pop('NAXIS3')
            for b in range(n_bands):
                d = hdu.data[b,:,:].copy()
                band_hdulist[b].append(fits.ImageHDU(header=hdu.header, data=d))
        else:
            band_hdulist[0].append(hdu)

    return band_hdulist


def healpix_mosaic(layer, wcs, spacing=0.45, verbose=False):
    # Determine nside from WCS pixel scale
    cutout_npix = 256
    wcs_pixscale = np.min(proj_plane_pixel_scales(wcs))
    if verbose:
        print(f'Desired pixel scale: {wcs_pixscale} deg')
    nside = (
        np.sqrt(4*np.pi/12)
      * (1+spacing) / (cutout_npix*np.radians(wcs_pixscale))
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
    hdulist = get_hdulist_per_band(hdulist)

    img = []
    shape_out = wcs.pixel_shape[::-1]

    if verbose:
        hdulist = tqdm(hdulist)
        print('Reprojecting images ...')

    for hdulist_b in hdulist:
        im,_ = reproject_and_coadd(
            fits.HDUList(hdulist_b),
            wcs,
            shape_out=shape_out,
            reproject_function=reproject_interp
        )
        img.append(im)

    img = np.stack(img, axis=0)
    if verbose:
        print(f'Image shape = {img.shape}')

    return img


def get_galactic_wcs(l0, b0, pixscale, shape):
    pixscale_deg = pixscale.to('deg').value
    wcs = {
        'NAXIS': 2,
        'NAXIS1': shape[0],
        'NAXIS2': shape[1],
        'CTYPE1': 'GLON-TAN',
        'CTYPE2': 'GLAT-TAN',
        'CRVAL1': l0.to('deg').value,
        'CRVAL2': b0.to('deg').value,
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


def save_image(img, fname, subtract_min=True, vmax_pct=99.5, vmax_abs=None):
    if subtract_min:
        img -= np.nanmin(img)
    if vmax_pct is not None:
        img = img / np.percentile(img, vmax_pct)
    elif vmax_abs is not None:
        img /= vmax_abs
    img = np.sqrt(img)
    img *= 255.
    img = np.clip(img, 0, 255).astype('uint8')
    im = Image.fromarray(img.T, mode='L')
    im.save(fname)


def main():
    wcs = get_galactic_wcs(-10.*units.deg, 5.0*units.deg, 2.0*units.arcsec, (1024,512))
    img = healpix_mosaic('decaps2', wcs, verbose=True)

    print(np.nanmin(img), np.nanmax(img))
    img -= np.nanmin(np.nanmin(img, axis=1), axis=1)[:,None,None]
    img = np.sqrt(img)
    vmax = np.nanpercentile(np.reshape(img,(2,-1)), 99.8, axis=1)

    print(wcs)
    plt.subplot(projection=wcs)
    plt.imshow(img[0], origin='lower', vmin=0, vmax=vmax[0])
    plt.grid(color='gray', ls='solid', alpha=0.5)
    plt.savefig('decaps2.png', dpi=200)
    plt.show()

    return 0

if __name__ == '__main__':
    main()

