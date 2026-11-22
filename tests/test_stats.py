"""
Known-answer tests for src/stats.py.
"""

import numpy as np
import pytest
import scipy.stats
import xarray as xr

import src.stats


def test_frac_change():
    """Fractional change relative to the first year, or to a given baseline."""

    ## series that doubles from the first to the last year
    x = xr.DataArray([2.0, 3.0, 4.0], coords=dict(year=[1870, 1875, 1880]))

    ## default baseline is the first year
    np.testing.assert_allclose(src.stats.frac_change(x), [0, 0.5, 1])

    ## explicit baseline
    np.testing.assert_allclose(src.stats.frac_change(x, x0=4.0), [-0.5, -0.25, 0])


def test_get_composite_hov():
    """
    A field equal to 2 * T34 composites to 2 * (the T34 composite); El Niño
    composites peak above zero and La Niña composites below.
    """

    ## random monthly T34 for 3 members, 1850-1909
    rng = np.random.default_rng(0)
    time = xr.date_range("1850-01-01", periods=720, freq="MS", use_cftime=True)
    T34 = xr.DataArray(
        rng.standard_normal((3, 720)), coords=dict(member=[0, 1, 2], time=time)
    )

    ## one EOF mode whose pattern is 1 everywhere once the coslat weights that
    ## reconstruct_fn divides by are applied
    lat = np.arange(-10, 11, 2.0)
    lon = np.arange(140, 281, 10.0)

    ## pattern = the coslat weights, constant in longitude
    coslat = np.sqrt(np.cos(np.deg2rad(lat)))
    comp = xr.DataArray(
        np.tile(coslat[:, None], (1, len(lon)))[None],
        coords=dict(mode=[0], latitude=lat, longitude=lon),
    )

    ## scores of the field "x": 2 * T34
    anom = xr.Dataset({"x": (2 * T34).expand_dims(mode=[0], axis=-1), "x_comp": comp})

    ## composites in two 20-year windows
    kwargs = dict(
        anom=anom, T34=T34, varnames=["x"], years=[1870, 1890], window_size=240
    )
    warm = src.stats.get_composite_hov(is_warm=True, **kwargs)
    cold = src.stats.get_composite_hov(is_warm=False, **kwargs)

    ## dims, and the field is twice the index at every longitude
    assert warm["x"].dims == ("year", "lag", "longitude")
    xr.testing.assert_allclose(warm["x"], 2 * warm["T_34"].broadcast_like(warm["x"]))

    ## at the peak, El Niño composites are positive and La Niña ones negative
    assert (warm["T_34"].sel(lag=0) > 1).all()
    assert (cold["T_34"].sel(lag=0) < -1).all()


def make_windowed(values):
    """
    Wrap an array as windowed data (dims member, year, time) with monthly times.

    Args:
        - values (np.ndarray): data, shape (member, year, time)

    Returns:
        - x (xr.DataArray): data with monthly times starting in January 1850
    """

    ## coordinates like those from src.utils.get_windowed
    n_member, n_year, n_time = values.shape
    time = xr.date_range("1850-01-01", periods=n_time, freq="MS")
    coords = dict(member=np.arange(n_member), year=1870 + np.arange(n_year), time=time)

    return xr.DataArray(values, coords=coords, dims=["member", "year", "time"])


def test_regress_by_sign():
    """For y = 2x when x > 0 and y = x/2 when x < 0, the slopes are 2, 1/2."""

    ## random predictor
    rng = np.random.default_rng(0)
    x = make_windowed(rng.standard_normal((2, 3, 480)))

    ## piecewise-linear target
    y = xr.where(x > 0, 2 * x, 0.5 * x)

    ## slopes for each subset
    slopes = src.stats.regress_by_sign(x, y)
    np.testing.assert_allclose(slopes["pos"], 2)
    np.testing.assert_allclose(slopes["neg"], 0.5)

    ## all x together: between the two
    assert ((slopes["all"] > 0.5) & (slopes["all"] < 2)).all()

    ## fitting by month gives the same slopes, for each month
    slopes_month = src.stats.regress_by_sign(x, y, by_month=True)
    assert slopes_month.sizes["month"] == 12
    np.testing.assert_allclose(slopes_month["pos"], 2)


def test_get_lag_growth_rate():
    """An AR(1) process x(t+1) = a x(t) + noise grows at 12 log(a) per year."""

    ## AR(1) with a = 0.9, long enough for a precise estimate
    rng = np.random.default_rng(1)
    a = 0.9
    n_time = 48000

    ## white-noise forcing, and the series (starting from zero)
    noise = rng.standard_normal((1, 1, n_time))
    values = np.zeros((1, 1, n_time))

    ## step forward in time
    for t in range(1, n_time):
        values[..., t] = a * values[..., t - 1] + noise[..., t]

    ## monthly growth rate, all months together
    R = src.stats.get_lag_growth_rate(make_windowed(values), lag=1)
    np.testing.assert_allclose(R["all"], 12 * np.log(a), atol=0.05)

    ## over 3 months the slope is a^3, so the rate is the same
    R3 = src.stats.get_lag_growth_rate(make_windowed(values), lag=3, month=11)
    np.testing.assert_allclose(R3["all"], 12 * np.log(a), atol=0.1)


def test_get_extremes():
    """The n largest or smallest values of the index pick the samples."""

    ## index and data along "time"; data = 10 * index
    index = xr.DataArray([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0], dims="time")
    data = 10 * index

    ## two largest and two smallest samples
    warm = src.stats.get_extremes(data, index, n=2, largest=True)
    cold = src.stats.get_extremes(data, index, n=2, largest=False)

    ## largest: 6 and 9; smallest: 1 and 1.5 (in increasing order of the index)
    np.testing.assert_allclose(warm, [60, 90])
    np.testing.assert_allclose(cold, [10, 15])


def test_get_extremes_rejects_nan():
    """NaN in the index raises (argsort would rank it as the largest value)."""

    ## index with a missing value
    index = xr.DataArray([3.0, np.nan, 4.0], dims="time")

    ## asking for extremes fails
    with pytest.raises(ValueError):
        src.stats.get_extremes(index, index, n=1)


def make_eof_field(scores):
    """
    One-mode EOF field equal to (score * longitude / 100) at every grid point.

    Args:
        - scores (xr.DataArray): the mode's scores, dims (member, time)

    Returns:
        - comp (xr.DataArray): pattern (mode, latitude, longitude), including the
          coslat weights that reconstruction divides by
        - scores (xr.DataArray): scores with a mode dim, dims (member, time, mode)
    """

    ## grid covering the equatorial Pacific
    lat = np.arange(-10, 11, 2.0)
    lon = np.arange(140, 281, 10.0)

    ## pattern = coslat weights times longitude / 100
    coslat = np.sqrt(np.cos(np.deg2rad(lat)))
    values = coslat[:, None] * lon[None, :] / 100
    comp = xr.DataArray(
        values[None], coords=dict(mode=[0], latitude=lat, longitude=lon)
    )

    return comp, scores.expand_dims(mode=[0], axis=-1)


def test_get_eq_sigma_and_skew():
    """
    For a one-mode field, each season's standard deviation is the scores' standard
    deviation times the pattern, and its skewness is the scores' skewness.
    """

    ## skewed (exponential) monthly scores for 3 members, 1850-1859
    rng = np.random.default_rng(1)
    time = xr.date_range("1850-01-01", periods=120, freq="MS", use_cftime=True)
    raw = xr.DataArray(
        rng.exponential(size=(3, 120)), coords=dict(member=[0, 1, 2], time=time)
    )

    ## the one-mode field built from them
    comp, scores = make_eof_field(raw)

    ## statistics by season over all longitudes of the grid
    sigma = src.stats.get_eq_sigma(comp, scores, "season", lons=(130, 290))
    skew = src.stats.get_eq_skew(comp, scores, "season", lons=(130, 290))

    ## known answers for DJF, pooling members and times
    djf = raw.where(raw.time.dt.season == "DJF", drop=True).values.ravel()
    sigma_djf = djf.std() * sigma.longitude / 100
    np.testing.assert_allclose(sigma.sel(season="DJF"), sigma_djf)

    ## skewness doesn't depend on the (positive) pattern
    np.testing.assert_allclose(skew.sel(season="DJF"), scipy.stats.skew(djf))
