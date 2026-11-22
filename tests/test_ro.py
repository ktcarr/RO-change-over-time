"""
Known-answer tests for src/ro.py (shared steps of the RO figures).
"""

import numpy as np
import xarray as xr

import src.ro


def make_fits_and_data():
    """
    Synthetic fits and data for testing get_R_at and get_eps.

    T is +a(month) in member 0 and -a(month) in member 1, so its standard
    deviation for each calendar month (pooling members and times) is a(month).
    The fitted coefficients are constants: R_1 = -0.5, R_2 = 0.2, R_3 = -0.1, and
    the h-equation's h coefficient is -0.3.

    Returns:
        - fits (xr.Dataset): "Lac" and "NROT_Lac", dims (year, cycle, ...)
        - data (xr.Dataset): "T_3", dims (member, year, time)
        - a (np.ndarray): standard deviation of T for each calendar month
    """

    ## standard deviation for each calendar month: 1, 1.1, ..., 2.1
    a = 1 + 0.1 * np.arange(12)

    ## two years of monthly times, two windows, two members
    time = xr.date_range("1850-01-01", periods=24, freq="MS")

    ## T = +a in member 0 and -a in member 1, the same in both windows
    sign = xr.DataArray([1.0, -1.0], dims="member")
    T = sign * xr.DataArray(np.tile(a, 2), coords=dict(time=time))
    data = xr.Dataset(dict(T_3=T.expand_dims(year=[1870, 1880])))

    ## coordinates of the fits: windows, seasonal cycle, and matrix indices
    coords = dict(year=[1870, 1880], cycle=np.arange(12) / 12)
    ones = xr.DataArray(np.ones((2, 12)), coords=coords)

    ## linear operator: R_1 = -0.5 (T on T), -0.3 (h on h)
    Lac = xr.concat(
        [
            xr.concat([-0.5 * ones, 0 * ones], dim="rankx"),
            xr.concat([0 * ones, -0.3 * ones], dim="rankx"),
        ],
        dim="ranky",
    )

    ## nonlinear T terms: R_2 = 0.2 (T^2), R_3 = -0.1 (T^3)
    NROT_Lac = xr.concat([0.2 * ones, -0.1 * ones], dim="nro_form")
    NROT_Lac = NROT_Lac.assign_coords(nro_form=["T2", "T3"])

    ## gather
    fits = xr.Dataset(dict(Lac=Lac, NROT_Lac=NROT_Lac))

    return fits, data, a


def test_get_R_at():
    """
    R(T) = R_1 + R_2 T + R_3 T^2 at T = n_sigma * a(month).
    """

    ## synthetic fits and data
    fits, data, a = make_fits_and_data()

    ## growth rate of a 1.5-sigma anomaly
    R = src.ro.get_R_at(fits, data, n_sigma=1.5)

    ## expected: the same in both windows
    T = 1.5 * a
    expected = -0.5 + 0.2 * T - 0.1 * T**2

    ## check dims and values in each window
    assert set(R.dims) == {"year", "cycle"}
    np.testing.assert_allclose(R.sel(year=1870).values, expected)
    np.testing.assert_allclose(R.sel(year=1880).values, expected)

    ## at T = 0, R is the linear coefficient
    np.testing.assert_allclose(src.ro.get_R_at(fits, data, n_sigma=0).values, -0.5)


def test_get_eps():
    """
    epsilon is minus the h-equation's h coefficient.
    """

    ## synthetic fits
    fits, _, _ = make_fits_and_data()

    ## h coefficient is -0.3
    np.testing.assert_allclose(src.ro.get_eps(fits).values, 0.3)
