"""
Indices: averages of a field over the regions used in the paper.
"""

import src.constants


def get_eq_avg(data, lons):
    """
    Average over the equatorial band (LATS_EQ) and a range of longitudes.

    Args:
        - data (xr.DataArray or xr.Dataset): data with "longitude" and "latitude"
        - lons (list of float): [west, east] longitude bounds (degrees east)

    Returns:
        - avg (xr.DataArray or xr.Dataset): data averaged over the box
    """

    ## select the box, then average over both horizontal dimensions
    box = data.sel(longitude=slice(*lons), latitude=slice(*src.constants.LATS_EQ))
    avg = box.mean(["longitude", "latitude"])

    return avg


def get_w(data):
    """
    Average over the western equatorial box (LONS_W).

    Args:
        - data (xr.DataArray or xr.Dataset): data with "longitude" and "latitude"

    Returns:
        - avg (xr.DataArray or xr.Dataset): western-box average
    """
    return get_eq_avg(data, src.constants.LONS_W)


def get_e(data):
    """
    Average over the eastern equatorial box (LONS_E).

    Args:
        - data (xr.DataArray or xr.Dataset): data with "longitude" and "latitude"

    Returns:
        - avg (xr.DataArray or xr.Dataset): eastern-box average
    """
    return get_eq_avg(data, src.constants.LONS_E)


def get_dx(data):
    """
    East-minus-west difference of equatorial box averages.

    Args:
        - data (xr.DataArray or xr.Dataset): data with "longitude" and "latitude"

    Returns:
        - dx (xr.DataArray or xr.Dataset): eastern-box minus western-box average
    """
    return get_e(data) - get_w(data)
