#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as units

import json


def great_circle_path(lon0, lat0, lon1, lat1, n_points=100):
    """Generate a great circle path between two points on the sphere.

    Parameters
    ----------
    lon0, lat0 : `astropy.units.Quantity`
        Longitude and latitude of the starting point.
    lon1, lat1 : `astropy.units.Quantity`
        Longitude and latitude of the ending point.
    n_points : int
        Number of points along the path.

    Returns
    -------
    Interpolation function that takes a scalar s in [0, 1], and
    returns (lon, lat) at that fraction (by distance) along the
    path.
    """
    lon0_rad = lon0.to('rad').value
    lat0_rad = lat0.to('rad').value
    lon1_rad = lon1.to('rad').value
    lat1_rad = lat1.to('rad').value

    # Convert to Cartesian coordinates
    x0 = np.cos(lat0_rad) * np.cos(lon0_rad)
    y0 = np.cos(lat0_rad) * np.sin(lon0_rad)
    z0 = np.sin(lat0_rad)

    x1 = np.cos(lat1_rad) * np.cos(lon1_rad)
    y1 = np.cos(lat1_rad) * np.sin(lon1_rad)
    z1 = np.sin(lat1_rad)

    # Compute the angle between the two points
    dot_product = x0 * x1 + y0 * y1 + z0 * z1
    angle = np.arccos(dot_product)

    # Generate points along the great circle
    t = np.linspace(0, 1, n_points)
    A = np.sin((1-t)*angle) / np.sin(angle)
    B = np.sin(t*angle) / np.sin(angle)
    x = A * x0 + B * x1
    y = A * y0 + B * y1
    z = A * z0 + B * z1
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon = np.arctan2(y, x)

    # Return interpolator that takes scalar s in [0, 1], and
    # returns (lon, lat) at that fraction (by distance) along
    # the path.
    def lon_lat_interpolator(s):
        lon_interp = np.interp(s, t, lon) * units.rad
        lat_interp = np.interp(s, t, lat) * units.rad
        return lon_interp, lat_interp

    return lon_lat_interpolator


def tangent_wcs_dict(center, fov, w, frame, rotation=0):
    lon,lat = center.to('deg').value
    fov = fov.to('deg').value

    # Determine height (in pixels) of image, based on width
    # and fov ratio
    tan_th0 = np.tan(np.radians(fov[0]/2))
    tan_th1 = np.tan(np.radians(fov[1]/2))
    h = int(np.round(w * tan_th1 / tan_th0))

    # Coordinate frame
    coordsys,ctype0,ctype1 = {
        'galactic': ('GAL', 'GLON-TAN', 'GLAT-TAN'),
        'icrs':     ('EQU', 'RA---TAN', 'DEC--TAN')
    }[frame]

    # Rotation matrix
    th = np.radians(rotation)
    pc = [[np.cos(th), np.sin(th)],
          [-np.sin(th), np.cos(th)]]
    
    wcs = dict(
        naxis=2,
        naxis1=w,
        naxis2=h,
        ctype1=ctype0,
        crpix1=w//2+0.5,
        crval1=lon,
        cdelt1=-np.degrees(tan_th0)/(0.5*w),
        cunit1='deg',
        ctype2=ctype1,
        crpix2=h//2+0.5,
        crval2=lat,
        cdelt2=np.degrees(tan_th1)/(0.5*h),
        cunit2='deg',
        coordsys=coordsys,
        pc1_1=pc[0][0],
        pc1_2=pc[0][1],
        pc2_1=pc[1][0],
        pc2_2=pc[1][1]
    )
    return wcs


def interpolate_lon_lat_fov(lon0, lat0, fov0, lon1, lat1, fov1, t):
    # Convert t to fov and path length s.
    # We assume
    #   d(fov)/dt = -a*fov,
    #   ds/dt = b*fov.
    # With s(0) = 0, s(1) = 1, fov(0) = fov0, fov(1) = fov1.
    # The solution is
    #   fov(t) = fov0 * exp(-a*t),
    #   s(t) = c * (1 - exp(-a*t)),
    # with
    #   a = ln(fov0/fov1),
    #   c = 1 / (1 - exp(-a))
    a = np.log((fov0 / fov1).to(''))
    c = 1 / (1 - np.exp(-a))

    # Calculate s(t) and fov(t)
    s = c * (1 - np.exp(-a * t))
    fov = fov0 * np.exp(-a * t)

    # Get lon, lat at s(t) using interpolation
    lon_lat_interp = great_circle_path(lon0, lat0, lon1, lat1)
    lon, lat = lon_lat_interp(s)

    return lon, lat, fov


def plot_path(lon0, lat0, lon1, lat1, fov0, fov1):
    # lon, lat = great_circle_path(
    #     lon0, lat0, lon1, lat1
    # )(np.linspace(0, 1, 16))
    # fov = np.linspace(
    #     fov0.to('deg').value,
    #     fov1.to('deg').value,
    #     16
    # )
    lon, lat, fov = interpolate_lon_lat_fov(
        lon0, lat0, fov0,
        lon1, lat1, fov1,
        np.linspace(0, 1, 16)
    )

    fig,ax = plt.subplots(
        1, 1,
        subplot_kw={'projection':'mollweide'}
    )
    ax.scatter(
        lon.to('rad').value,
        lat.to('rad').value,
        s=fov.to('deg').value**2 * 4,
        color='blue',
        alpha=0.1
    )
    
    return fig, ax


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
    # Start / end coordinates
    lon0, lat0 = [244.6, -41.5] * units.deg
    lon1, lat1 = [242.34, -39.10] * units.deg
    input_frame = 'icrs'
    output_frame = 'galactic'

    # Field of view at start / end
    fov0 = 12.8 * units.deg
    fov1 = 1.2 * units.deg

    # Rotation angle of images (w.r.t. output frame)
    rot = 0.0 * units.deg

    # Image shape and number of frames
    img_shape = (1920//4, 1080//4) # in pixels
    n_frames = 60

    # Which layers to use for RGB
    layer = ['decaps2[2]','decaps2[1]','decaps2[0]']

    # Output filename
    fname = 'path_framespec.json'

    # Transform start / end coordinates to output frame
    if input_frame != output_frame:
        from astropy.coordinates import SkyCoord
        c0 = SkyCoord(lon0, lat0, frame=input_frame)
        c1 = SkyCoord(lon1, lat1, frame=input_frame)
        c0_out = c0.transform_to(output_frame).represent_as('spherical')
        c1_out = c1.transform_to(output_frame).represent_as('spherical')
        lon0 = c0_out.lon
        lat0 = c0_out.lat
        lon1 = c1_out.lon
        lat1 = c1_out.lat

    # fig,ax = plot_path(lon0, lat0, lon1, lat1, fov0, fov1)
    # plt.show()

    # Get the path
    lon,lat,fov_x = interpolate_lon_lat_fov(
        lon0, lat0, fov0,
        lon1, lat1, fov1,
        np.linspace(0, 1, n_frames)
    )
    lonlat = np.stack([lon, lat], axis=1)

    # Calculate inferred fov in the other dimension, based on the
    # image shape, the fov in the first dimension, assuming square
    # pixels and a tangent projection.
    # fov_ratio = 2 * np.arctan(
    #     img_shape[1]/img_shape[0]*np.tan(fov_x.to('rad').value/2)
    # )
    # print(f'fov_y/fov_x = {fov_ratio}')

    # Generate WCS for each frame
    wcs = []
    for center,fx in zip(lonlat, fov_x):
        # Calculate inferred fov in the other dimension, based on the
        # image shape, the fov in the first dimension, assuming square
        # pixels and a tangent projection.
        fy = 2 * np.arctan(
            img_shape[1]/img_shape[0]
          * np.tan(fx.to('rad').value/2)
        ) * units.rad
        print(f'fov_y/fov_x = {(fy/fx).to("")}')
        print()
        fov = np.array([
            fx.to('rad').value,
            fy.to('rad').value
        ]) * units.rad

        w = tangent_wcs_dict(
            center,
            fov,
            img_shape[0],
            frame=output_frame,
            rotation=rot.to('deg').value
        )
        w['layer'] = layer
        wcs.append(w)
    
    with open(fname, 'w') as f:
        json.dump(wcs, f, indent=2)

    # fname = 'dust254_res2048.json'
    # lon,lat = [254.56, -46.045] * units.deg
    # galactic = False
    # img_scale_0, img_scale_1 = [0.60, 0.60] * units.deg
    # img_shape = (2048, 2048)
    # #img_shape = (512, 512)
    # n_frames = 1

    # #fname = 'zoom_l317_b-4.json'
    # #lon,lat = [317.15, -4.15] * units.deg
    # #galactic = True
    # #img_scale_0 = 15. * units.deg
    # #img_scale_1 = 1.0 * units.deg
    # ##img_scale_1 = 1000 * units.arcsec
    # #img_shape = (1920, 1080)
    # #n_frames = 150

    # pixscale0 = img_scale_0 / img_shape[0]
    # pixscale1 = img_scale_1 / img_shape[0]

    # pixscale = np.exp(np.linspace(
    #     np.log(pixscale0.to('deg').value),
    #     np.log(pixscale1.to('deg').value),
    #     n_frames
    # )) * units.deg

    # wcs = []
    # for s in pixscale:
    #     w = get_wcs_dict(lon, lat, s, img_shape, galactic=galactic)
    #     w['layer'] = ['decaps2[2]','decaps2[1]','decaps2[0]']
    #     wcs.append(w)

    #     w = get_wcs_dict(lon, lat, s, img_shape, galactic=galactic)
    #     #w['layer'] = ['unwise-neo7[1]','unwise-neo7[0]','decaps2-riy[2]']
    #     w['layer'] = ['unwise-neo7[1]','unwise-neo7[0]','unwise-neo7[0]']
    #     wcs.append(w)

    # with open(fname, 'w') as f:
    #     json.dump(wcs, f, indent=2)
    
    return 0

if __name__ == '__main__':
    main()

