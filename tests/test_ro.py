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


def make_labeled_fits():
    """
    Synthetic fits whose values change over time, with the coordinate labels of
    real fits (ranky/rankx = 1, 2; nro_form = "T2", "TH", ...).

    Returns:
        - fits (xr.Dataset): "Lac", "NROT_Lac", "xi_stdac", dims (year, ...)
    """

    ## three windows; each value grows by 1 per window
    year = xr.DataArray([1870, 1880, 1890], dims="year")
    trend = xr.DataArray([0.0, 1.0, 2.0], coords=dict(year=year))

    ## linear operator: distinct starting value for each entry
    ranks = dict(ranky=[1, 2], rankx=[1, 2], cycle=np.arange(3))
    Lac0 = xr.DataArray(np.arange(12.0).reshape(2, 2, 3), coords=ranks)

    ## nonlinear terms and noise
    forms = dict(nro_form=["T2", "TH", "T3"], cycle=np.arange(3))
    NROT0 = xr.DataArray(np.ones((3, 3)), coords=forms)
    xi0 = xr.DataArray(np.ones((2, 3)), coords=dict(ranky=[1, 2], cycle=np.arange(3)))

    ## gather, with "year" first as in real fits
    fits = xr.Dataset(
        dict(
            Lac=(Lac0 + trend).transpose("year", ...),
            NROT_Lac=(NROT0 + trend).transpose("year", ...),
            xi_stdac=(xi0 + trend).transpose("year", ...),
        )
    )

    return fits


def test_fix_terms():
    """
    Fixed terms keep their 1870 values; all other entries are unchanged.
    """

    ## synthetic fits; fix R (Lac T-on-T entry, T^2 and T^3 terms)
    fits = make_labeled_fits()
    fixed = src.ro.fix_terms(fits, src.ro.RO_TERMS["R"], ref_year=1870)

    ## fixed entries: equal to their 1870 values in every window
    R_1 = fixed["Lac"].sel(ranky=1, rankx=1)
    np.testing.assert_allclose(R_1.values, R_1.sel(year=1870).values * np.ones((3, 1)))

    ## T^2 and T^3 terms start at 1
    NRO = fixed["NROT_Lac"].sel(nro_form=["T2", "T3"])
    np.testing.assert_allclose(NRO.values, 1.0)

    ## other entries unchanged: the T*h term and the other Lac entries
    xr.testing.assert_equal(
        fixed["NROT_Lac"].sel(nro_form="TH"), fits["NROT_Lac"].sel(nro_form="TH")
    )
    xr.testing.assert_equal(fixed["Lac"].sel(ranky=2), fits["Lac"].sel(ranky=2))
    xr.testing.assert_equal(fixed["xi_stdac"], fits["xi_stdac"])

    ## dims keep their order, and the input isn't modified
    assert fixed["Lac"].dims == fits["Lac"].dims
    assert fits["Lac"].sel(ranky=1, rankx=1, year=1890).values[0] == 2.0


def test_fix_terms_all_entries():
    """
    An empty selection (e.g. the noise) fixes every entry of the variable.
    """

    ## synthetic fits; fix the noise
    fits = make_labeled_fits()
    fixed = src.ro.fix_terms(fits, [("xi_stdac", dict())], ref_year=1880)

    ## every window has the 1880 values
    np.testing.assert_allclose(fixed["xi_stdac"].values, 2.0)


def test_get_amplitudes():
    """
    DJF tails of a known distribution: T uniform on -1..1 in DJF (other months
    far outside, so including them would change the answer).
    """

    ## two years of monthly times, one window
    time = xr.date_range("1850-01-01", periods=24, freq="MS")
    is_djf = time.month.isin([12, 1, 2])

    ## 6 DJF months x 101 members: values -1, -0.98, ..., 1 in each month
    T = np.where(is_djf[:, None], np.linspace(-1, 1, 101)[None, :], 10.0)
    sims = xr.Dataset(dict(T_3=(("time", "member"), T)), coords=dict(time=time))
    sims = sims.expand_dims(year=[1870])

    ## amplitudes in K, with T_scale = 2
    amp = src.ro.get_amplitudes(sims, T_scale=2.0)

    ## 95th percentile of uniform(-1, 1) is 0.9; times T_scale
    np.testing.assert_allclose(amp["warm"].values, 1.8)
    np.testing.assert_allclose(amp["cold"].values, 1.8)
    np.testing.assert_allclose(amp["mean"].values, 1.8)
