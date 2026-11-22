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
