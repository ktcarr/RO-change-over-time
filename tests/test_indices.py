"""
Known-answer tests for src/indices.py.
"""

import numpy as np
import xarray as xr

import src.indices


def make_linear_field():
    """
    Synthetic lat/lon field equal to its longitude (degrees east).

    Returns:
        - field (xr.DataArray): field on a 1-degree grid, 120E-290E, 20S-20N
    """

    ## 1-degree grid covering the tropical Pacific
    lon = np.arange(120, 291, 1.0)
    lat = np.arange(-20, 21, 1.0)

    ## field value = longitude, the same at every latitude
    values = np.tile(lon, (len(lat), 1))
    field = xr.DataArray(
        values, coords=dict(latitude=lat, longitude=lon), dims=["latitude", "longitude"]
    )

    return field


def test_box_averages():
    """For a field equal to longitude, box averages are the box's mid-longitude."""

    ## build the field
    field = make_linear_field()

    ## western box (150-190E) averages to 170, eastern box (240-280E) to 260
    np.testing.assert_allclose(src.indices.get_w(field), 170)
    np.testing.assert_allclose(src.indices.get_e(field), 260)

    ## east minus west
    np.testing.assert_allclose(src.indices.get_dx(field), 90)


def test_eq_avg_uses_equatorial_band():
    """Values outside 5S-5N don't affect the equatorial average."""

    ## field equal to 1 inside the band and 100 outside it
    field = xr.ones_like(make_linear_field())
    field = field.where(np.abs(field.latitude) <= 5, 100)

    ## the average over any longitude range is 1
    np.testing.assert_allclose(src.indices.get_eq_avg(field, [150, 250]), 1)


def test_eq_profile():
    """The equatorial profile keeps longitude and ignores values outside 5S-5N."""

    ## field equal to longitude inside the band and -999 outside it
    field = make_linear_field()
    field = field.where(np.abs(field.latitude) <= 5, -999)

    ## profile over 130-280E equals the longitude at each point
    profile = src.indices.get_eq_profile(field, lons=(130, 280))
    np.testing.assert_allclose(profile, profile.longitude)

    ## and covers exactly that range
    np.testing.assert_allclose(profile.longitude[[0, -1]], [130, 280])
