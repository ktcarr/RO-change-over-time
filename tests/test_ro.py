"""
Known-answer tests for src/ro.py (shared steps of the RO figures).
"""

import numpy as np
import xarray as xr

## the RO model class that get_model should return
import ro_tools.xro

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


def test_fit_alpha():
    """
    Recovers known cubic coefficients, which differ by calendar month and
    window, from noise-free fluxes.
    """

    ## random T in two windows: 10 members x 4 years of monthly times
    rng = np.random.default_rng(0)
    time = xr.date_range("1850-01-01", periods=48, freq="MS")

    ## dims (member, year, time)
    coords = dict(member=np.arange(10), year=[1870, 1880], time=time)
    T = xr.DataArray(rng.normal(size=(10, 2, 48)), coords=coords)

    ## true coefficients: alpha_1 = month / 12, alpha_2 = 0.1, alpha_3 = -0.05,
    ## plus 1 in the second window
    month = T.time.dt.month
    shift = xr.DataArray([0.0, 1.0], coords=dict(year=[1870, 1880]))
    a1 = month / 12 + shift

    ## constant quadratic and cubic coefficients
    a2 = 0.1
    a3 = -0.05

    ## flux that damps T at rate alpha(T): Q = -(alpha_1 T + alpha_2 T^2 + alpha_3 T^3)
    Q = -(a1 * T + a2 * T**2 + a3 * T**3)

    ## fit; label the months with a seasonal-cycle coordinate
    cycle = xr.DataArray(np.arange(12) / 12, dims="cycle")
    alpha = src.ro.fit_alpha(T, Q, cycle)

    ## check dims and coefficients
    assert alpha.dims == ("year", "cycle", "param")

    ## alpha_1 = month / 12 (+1 in the second window); alpha_2, alpha_3 constant
    expected_a1 = np.arange(1, 13) / 12 + np.array([[0.0], [1.0]])
    np.testing.assert_allclose(alpha.sel(param="T1").values, expected_a1)

    ## quadratic and cubic coefficients
    np.testing.assert_allclose(alpha.sel(param="T2").values, 0.1)
    np.testing.assert_allclose(alpha.sel(param="T3").values, -0.05, atol=1e-12)


def test_get_R_coefs():
    """
    R_1, R_2, R_3 come from the T-on-T entry of Lac and the T^2, T^3 terms.
    """

    ## synthetic fits
    fits = make_labeled_fits()
    R = src.ro.get_R_coefs(fits)

    ## check dims and values
    assert R.dims == ("year", "cycle", "param")
    xr.testing.assert_equal(
        R.sel(param="T1", drop=True), fits["Lac"].sel(ranky=1, rankx=1, drop=True)
    )
    xr.testing.assert_equal(
        R.sel(param="T3", drop=True),
        fits["NROT_Lac"].sel(nro_form="T3", drop=True),
    )


def test_remove_change_in_R():
    """
    Fixing all of R matches fix_terms; fixing a part that doesn't change
    leaves the fits unchanged.
    """

    ## synthetic fits and their R coefficients
    fits = make_labeled_fits()
    R = src.ro.get_R_coefs(fits)

    ## part = all of R: the same as fixing R's terms
    fixed = src.ro.remove_change_in_R(fits, R, ref_year=1870)
    expected = src.ro.fix_terms(fits, src.ro.RO_TERMS["R"], ref_year=1870)
    for name in ["Lac", "NROT_Lac", "xi_stdac"]:
        xr.testing.assert_allclose(fixed[name], expected[name])

    ## part that is the same in every window: nothing changes
    fixed = src.ro.remove_change_in_R(fits, xr.ones_like(R), ref_year=1870)
    xr.testing.assert_identical(fixed, fits)


def test_get_model_settings():
    """
    get_model returns ro_tools' XRO with the paper's settings: 12 steps per year,
    3 harmonics, centered differences and lag 1 for each harmonic's covariance.
    """

    ## build the model used by all RO figures
    model = src.ro.get_model()

    ## it's the ro_tools model, not the old src/XRO.py copy
    assert isinstance(model, ro_tools.xro.XRO)

    ## settings: monthly cycle, 3 harmonics, centered differences
    assert model.ncycle == 12
    assert model.ac_order == 3
    assert not model.is_forward

    ## XRO's default lag for centered differences: 1 for each of the 4 terms
    assert list(model.taus) == [1, 1, 1, 1]


def test_fit_alpha_harmonic():
    """
    Recovers known coefficients from noise-free fluxes: a linear coefficient
    with an annual and a semi-annual cycle (exactly representable by the
    harmonics), and either flat or seasonal nonlinear coefficients.
    """

    ## random T in two windows: 10 members x 4 years of monthly times
    rng = np.random.default_rng(0)
    time = xr.date_range("1850-01-01", periods=48, freq="MS")

    ## dims (member, year, time)
    coords = dict(member=np.arange(10), year=[1870, 1880], time=time)
    T = xr.DataArray(rng.normal(size=(10, 2, 48)), coords=coords)

    ## the fits' seasonal-cycle coordinate, the middle of each month
    cycle = xr.DataArray((np.arange(12) + 0.5) / 12, dims="cycle")

    ## annual and semi-annual cycles at each sample and at each month of the cycle
    phase = 2 * np.pi * (T.time.dt.month - 0.5) / 12
    phase_cycle = 2 * np.pi * cycle.values

    ## true linear coefficient, +1 in the second window
    shift = xr.DataArray([0.0, 1.0], coords=dict(year=[1870, 1880]))
    a1 = 1 + 0.5 * np.cos(phase) + 0.2 * np.sin(2 * phase) + shift

    ## its value at each month of the cycle, in each window
    expected_a1 = 1 + 0.5 * np.cos(phase_cycle) + 0.2 * np.sin(2 * phase_cycle)
    expected_a1 = expected_a1 + np.array([[0.0], [1.0]])

    ## flat nonlinear coefficients: recovered by both forms
    Q = -(a1 * T + 0.1 * T**2 - 0.05 * T**3)
    for flat_nonlinear in [True, False]:
        alpha = src.ro.fit_alpha_harmonic(T, Q, cycle, flat_nonlinear=flat_nonlinear)

        ## dims, then each coefficient
        assert alpha.dims == ("year", "cycle", "param")
        np.testing.assert_allclose(alpha.sel(param="T1").values, expected_a1)

        ## the nonlinear coefficients are the same in every month
        np.testing.assert_allclose(alpha.sel(param="T2").values, 0.1, atol=1e-12)
        np.testing.assert_allclose(alpha.sel(param="T3").values, -0.05, atol=1e-12)

    ## seasonal quadratic coefficient: recovered by the seasonal form
    a2 = 0.1 + 0.05 * np.sin(phase)
    Q = -(a1 * T + a2 * T**2)
    alpha = src.ro.fit_alpha_harmonic(T, Q, cycle, flat_nonlinear=False)

    ## its value at each month of the cycle (the same in both windows)
    expected_a2 = 0.1 + 0.05 * np.sin(phase_cycle)
    np.testing.assert_allclose(
        alpha.sel(param="T2").values, np.tile(expected_a2, (2, 1)), atol=1e-12
    )


def make_windowed_record(n_members=3, n_years=20, seed=0):
    """
    Synthetic windowed RO data: a damped, nonlinear T and h with noise, in two
    windows of several members, dims (member, year, time) as from load_ro_data.

    Args:
        - n_members (int): number of members
        - n_years (int): length of each window (years)
        - seed (int): seed of the random numbers

    Returns:
        - data (xr.Dataset): "T_3" and "h_hat"
    """

    ## generator
    rng = np.random.default_rng(seed)
    n_months = 12 * n_years

    ## the series of each member and window
    T = np.zeros((n_members, 2, n_months))
    h = np.zeros((n_members, 2, n_months))

    ## monthly forward steps of a damped oscillator with T^2 and T^3 terms
    dt = 1 / 12
    for i in range(n_months - 1):
        tendency_T = -1.5 * T[..., i] + h[..., i] + 0.3 * T[..., i] ** 2
        tendency_T = tendency_T - 0.5 * T[..., i] ** 3
        T[..., i + 1] = (
            T[..., i] + dt * tendency_T + 0.3 * rng.normal(size=(n_members, 2))
        )

        ## h, recharged by T
        tendency_h = -1.0 * T[..., i] - 0.5 * h[..., i]
        h[..., i + 1] = (
            h[..., i] + dt * tendency_h + 0.2 * rng.normal(size=(n_members, 2))
        )

    ## with coordinates
    time = xr.date_range("0001-01", periods=n_months, freq="MS", use_cftime=True)
    coords = dict(year=[1870, 1880], time=time)

    ## as a Dataset
    dims = ["member", "year", "time"]
    data = xr.Dataset(dict(T_3=(dims, T), h_hat=(dims, h)), coords=coords)

    return data


def test_fit_flux_feedback_adds_up():
    """
    A "flux" whose warming rate is dT/dt itself is the whole T equation: alpha
    is minus R_1, R_2, R_3 and F_1 of the RO's fit, in every window and month.
    """

    ## synthetic data and the RO's settings
    data = make_windowed_record()
    model = src.ro.get_model(taus=2, ac_order=2)
    fit_kwargs = src.ro.get_fit_kwargs(fit_constant=True, flat_constant=True)

    ## the RO fit
    fits = src.ro.fit_over_time(data, model, fit_kwargs)

    ## dT/dt as XRO computes it (centered differences, per year)
    dTdt = data["T_3"].copy(
        data=ro_tools.xro.gradient(data["T_3"].values, axis=-1, is_forward=False)
    )

    ## the flux part of the T equation, and alpha from it
    flux_fits = src.ro.fit_flux_feedback(data, dTdt, model, fit_kwargs)
    alpha = src.ro.get_alpha(flux_fits)

    ## minus the RO's coefficients
    R = src.ro.get_R_coefs(fits)
    np.testing.assert_allclose(alpha.sel(param=src.ro.R_PARAMS).values, -R.values)

    ## the flux's part of F_1
    F1 = src.ro.get_params(fits)["F1"].transpose("year", "cycle")
    np.testing.assert_allclose(alpha.sel(param="h").values, -F1.values)


def test_get_flux_tendency():
    """
    For forward differences, the warming rate is averaged to the half step:
    a ramp t becomes t + 1/2 (and t - 1/2 in the last month); for centered
    ones it's unchanged.
    """

    ## a ramp in time, in one member and window
    time = xr.date_range("0001-01", periods=6, freq="MS", use_cftime=True)
    Q = xr.DataArray(np.arange(6.0), coords=dict(time=time))

    ## forward: the half step
    Q_fwd = src.ro.get_flux_tendency(Q, src.ro.get_model(taus=1, is_forward=True))
    np.testing.assert_allclose(Q_fwd.values, [0.5, 1.5, 2.5, 3.5, 4.5, 4.5])

    ## centered: unchanged
    Q_ctr = src.ro.get_flux_tendency(Q, src.ro.get_model(taus=2))
    np.testing.assert_allclose(Q_ctr.values, Q.values)


def test_fit_window_bound():
    """
    With the T^3 bound, the fitted T^3 coefficient is at most zero in every month,
    and the number of months at the bound is saved with the fit.
    """

    ## synthetic data and the settings, with seasonal nonlinear terms
    data = make_windowed_record(seed=1)
    model = src.ro.get_model(taus=2, ac_order=1)
    fit_kwargs = src.ro.get_fit_kwargs(fit_constant=True, flat_constant=True)

    ## the bounded fit
    fits = src.ro.fit_over_time(data, model, fit_kwargs, nonpositive_T3=True)

    ## within the bound, and the months counted
    R3 = fits["NROT_Lac"].sel(nro_form="T3")
    assert float(R3.max()) <= 1e-8
    assert "T3_bound_months" in fits
