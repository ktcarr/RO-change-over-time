"""
Known-answer tests for src/data.py.
"""

import numpy as np
import pytest
import xarray as xr

import src.data
import src.utils


def test_load_T34_anom_rejects_unknown_index():
    """An unknown index name raises a ValueError before any data are read."""

    ## "ml_80" is a typo for "ml80"
    with pytest.raises(ValueError):
        src.data.load_T34_anom(index="ml_80")


def test_load_T34_total_rejects_unknown_index():
    """An unknown index name raises a ValueError before any data are read."""

    ## "SST" (capitalized) is not a valid name
    with pytest.raises(ValueError):
        src.data.load_T34_total(index="SST")


def test_get_window_matches_get_windowed():
    """get_window cuts out the same months as get_windowed's window."""

    ## monthly series 1850-2100 equal to the month's index
    time = xr.date_range("1850-01-01", "2100-12-01", freq="MS")
    x = xr.DataArray(np.arange(len(time), dtype=float), coords=dict(time=time))

    ## all windows (40-year length, 5-year stride)
    windowed = src.utils.get_windowed(x, window_size=480, stride=60)

    ## compare full windows, including the first (2080 is the last full one)
    for year in [1870, 2010, 2080]:
        window = src.data.get_window(x, year, window_size=480)
        np.testing.assert_array_equal(window.values, windowed.sel(year=year).values)


def test_get_window_month_keeps_one_month():
    """get_window_month keeps one value per year: the chosen month in the window."""

    ## monthly data, 1850-1949, whose value is the month number
    time = xr.date_range("1850-01-01", periods=1200, freq="MS", use_cftime=True)
    x = xr.DataArray(
        np.tile(time.month, (2, 1)).astype(float),
        coords=dict(member=[0, 1], time=time),
    )

    ## December in the window centered on 1900 (1880-1919)
    x_dec = src.data.get_window_month(x, 1900, month=12, window_size=480)

    ## dims (member, time), with 40 Decembers per member
    assert x_dec.dims == ("member", "time")
    assert x_dec.sizes["time"] == 40

    ## the first is in 1880, and all values are December's
    assert x_dec.time.dt.year.values[0] == 1880
    assert (x_dec == 12).all()


def test_flux_to_tendency():
    """100 W/m^2 into a 50 m layer warms it by about 15 K/yr."""

    ## 100 W/m^2 * (360 days * 86400 s) / (1020 kg/m^3 * 4200 J/kg/K * 50 m)
    expected = 100 * 360 * 8.64e4 / (1.02e3 * 4.2e3 * 50)
    assert np.isclose(src.data.flux_to_tendency(100.0, H=50), expected)

    ## halving the depth doubles the tendency
    assert np.isclose(src.data.flux_to_tendency(100.0, H=25), 2 * expected)


def test_get_djf_keeps_full_seasons():
    """
    A 40-year window has 39 full DJF seasons; each DJF mean is the mean of its
    December, January, and February.
    """

    ## monthly values equal to the month's index in the window (0, 1, 2, ...)
    time = xr.date_range("1850-01-01", periods=480, freq="MS")
    x = xr.DataArray(np.arange(480.0), coords=dict(time=time))

    ## DJF means
    x_djf = src.data.get_djf(x)

    ## the first full DJF is Dec 1850 (index 11) to Feb 1851 (index 13)
    assert x_djf.sizes["time"] == 39
    np.testing.assert_allclose(x_djf.values[0], 12)

    ## the rest follow every 12 months; the last is Dec 1888 to Feb 1889
    np.testing.assert_allclose(np.diff(x_djf.values), 12)
    assert x_djf.time.dt.year.values[-1] == 1888


def test_get_season_keeps_full_seasons():
    """
    Seasonal means for the chosen season only, dropping partial seasons at the
    ends of the data.
    """

    ## three years of monthly data whose value is the month number
    time = xr.date_range("1850-01-01", periods=36, freq="MS")
    x = xr.DataArray(time.month.astype(float), coords=dict(time=time))

    ## Sep-Oct-Nov: every year is complete; mean of 9, 10, 11 is 10
    son = src.data.get_season(x, start_month=9)
    np.testing.assert_array_equal(son.time.dt.year, [1850, 1851, 1852])
    np.testing.assert_allclose(son.values, 10.0)

    ## Nov-Dec-Jan: the first (Jan only) and last (Nov-Dec) are partial
    ndj = src.data.get_season(x, start_month=11)
    np.testing.assert_array_equal(ndj.time.dt.year, [1850, 1851])
    np.testing.assert_allclose(ndj.values, (11 + 12 + 1) / 3)


def test_get_3month_means():
    """Seasons start in November; the time mean is removed first."""

    ## months Nov 2000 - Oct 2001 with values 0, ..., 11 (mean 5.5)
    time = xr.date_range("2000-11-01", periods=12, freq="MS")
    x = xr.DataArray(np.arange(12.0), coords=dict(time=time))
    x_3mo = src.data.get_3month_means(x)

    ## NDJ, FMA, MJJ, ASO means: 1, 4, 7, 10, minus 5.5
    np.testing.assert_allclose(x_3mo, [-4.5, -1.5, 1.5, 4.5])
    assert list(x_3mo.time.dt.month.values) == [11, 2, 5, 8]
