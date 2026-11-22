"""
Known-answer tests for src/ro_figure_utils.py (shared steps of the RO figures).
"""

import numpy as np
import xarray as xr

import src.ro_figure_utils as rfu


def test_get_djf_keeps_full_seasons():
    """
    A 40-year window has 39 full DJF seasons; each DJF mean is the mean of its
    December, January, and February.
    """

    ## monthly values equal to the month's index in the window (0, 1, 2, ...)
    time = xr.date_range("1850-01-01", periods=480, freq="MS")
    x = xr.DataArray(np.arange(480.0), coords=dict(time=time))

    ## DJF means
    x_djf = rfu.get_djf(x)

    ## the first full DJF is Dec 1850 (index 11) to Feb 1851 (index 13)
    assert x_djf.sizes["time"] == 39
    np.testing.assert_allclose(x_djf.values[0], 12)

    ## the rest follow every 12 months; the last is Dec 1888 to Feb 1889
    np.testing.assert_allclose(np.diff(x_djf.values), 12)
    assert x_djf.time.dt.year.values[-1] == 1888
