"""
Known-answer tests for src/figure_utils.py (helpers shared by figure scripts).
"""

import matplotlib

## non-interactive backend, so the tests run without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import src.figure_utils as fu
import src.utils


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
    np.testing.assert_allclose(fu.get_w(field), 170)
    np.testing.assert_allclose(fu.get_e(field), 260)

    ## east minus west
    np.testing.assert_allclose(fu.get_dx(field), 90)


def test_eq_avg_uses_equatorial_band():
    """Values outside 5S-5N don't affect the equatorial average."""

    ## field equal to 1 inside the band and 100 outside it
    field = xr.ones_like(make_linear_field())
    field = field.where(np.abs(field.latitude) <= 5, 100)

    ## the average over any longitude range is 1
    np.testing.assert_allclose(fu.get_eq_avg(field, [150, 250]), 1)


def test_make_centered_levels():
    """Levels are symmetric, evenly spaced, with a band centered on zero."""

    ## dx=1, two levels per side
    levels = fu.make_centered_levels(dx=1, nlevs=2)
    np.testing.assert_allclose(levels, [-1.5, -0.5, 0.5, 1.5])

    ## a non-integer spacing gives the same pattern
    levels = fu.make_centered_levels(dx=0.4, nlevs=5)

    ## 10 levels, the first positive one at dx/2, evenly spaced
    assert len(levels) == 10
    np.testing.assert_allclose(levels[5], 0.2)
    np.testing.assert_allclose(np.diff(levels), 0.4)


def test_frac_change():
    """Fractional change relative to the first year, or to a given baseline."""

    ## series that doubles from the first to the last year
    x = xr.DataArray([2.0, 3.0, 4.0], coords=dict(year=[1870, 1875, 1880]))

    ## default baseline is the first year
    np.testing.assert_allclose(fu.frac_change(x), [0, 0.5, 1])

    ## explicit baseline
    np.testing.assert_allclose(fu.frac_change(x, x0=4.0), [-0.5, -0.25, 0])


def test_save_fig(tmp_path, monkeypatch):
    """save_fig writes SAVE_FP/<subdir>/<name>.pdf, creating the folder."""

    ## point SAVE_FP at a temporary folder
    monkeypatch.setenv("SAVE_FP", str(tmp_path))

    ## save a small figure
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    ## save it to a subfolder that doesn't exist yet
    fp = fu.save_fig(fig, "test-fig", subdir="figs")

    ## free the figure's memory
    plt.close(fig)

    ## the file exists where expected and is a PDF
    assert fp == tmp_path / "figs" / "test-fig.pdf"
    assert fp.read_bytes()[:4] == b"%PDF"


def test_load_T34_anom_rejects_unknown_index():
    """An unknown index name raises a ValueError before any data are read."""

    ## pytest checks that the error is raised
    import pytest

    ## "ml_80" is a typo for "ml80"
    with pytest.raises(ValueError):
        fu.load_T34_anom(index="ml_80")


def test_get_window_matches_get_windowed():
    """get_window cuts out the same months as get_windowed's window."""

    ## monthly series 1850-2100 equal to the month's index
    time = xr.date_range("1850-01-01", "2100-12-01", freq="MS")
    x = xr.DataArray(np.arange(len(time), dtype=float), coords=dict(time=time))

    ## all windows (40-year length, 5-year stride)
    windowed = src.utils.get_windowed(x, window_size=480, stride=60)

    ## compare full windows, including the first (2080 is the last full one)
    for year in [1870, 2010, 2080]:
        window = fu.get_window(x, year, window_size=480)
        np.testing.assert_array_equal(window.values, windowed.sel(year=year).values)


def test_load_T34_total_rejects_unknown_index():
    """An unknown index name raises a ValueError before any data are read."""

    ## pytest checks that the error is raised
    import pytest

    ## "SST" (capitalized) is not a valid name
    with pytest.raises(ValueError):
        fu.load_T34_total(index="SST")


def test_get_window_month_keeps_one_month():
    """get_window_month keeps one value per year: the chosen month in the window."""

    ## monthly data, 1850-1949, whose value is the month number
    time = xr.date_range("1850-01-01", periods=1200, freq="MS", use_cftime=True)
    x = xr.DataArray(
        np.tile(time.month, (2, 1)).astype(float),
        coords=dict(member=[0, 1], time=time),
    )

    ## December in the window centered on 1900 (1880-1919)
    x_dec = fu.get_window_month(x, 1900, month=12, window_size=480)

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
    assert np.isclose(fu.flux_to_tendency(100.0, H=50), expected)

    ## halving the depth doubles the tendency
    assert np.isclose(fu.flux_to_tendency(100.0, H=25), 2 * expected)


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
    warm = fu.get_composite_hov(is_warm=True, **kwargs)
    cold = fu.get_composite_hov(is_warm=False, **kwargs)

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
    slopes = fu.regress_by_sign(x, y)
    np.testing.assert_allclose(slopes["pos"], 2)
    np.testing.assert_allclose(slopes["neg"], 0.5)

    ## all x together: between the two
    assert ((slopes["all"] > 0.5) & (slopes["all"] < 2)).all()

    ## fitting by month gives the same slopes, for each month
    slopes_month = fu.regress_by_sign(x, y, by_month=True)
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
    R = fu.get_lag_growth_rate(make_windowed(values), lag=1)
    np.testing.assert_allclose(R["all"], 12 * np.log(a), atol=0.05)

    ## over 3 months the slope is a^3, so the rate is the same
    R3 = fu.get_lag_growth_rate(make_windowed(values), lag=3, month=11)
    np.testing.assert_allclose(R3["all"], 12 * np.log(a), atol=0.1)
