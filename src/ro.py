"""
The recharge oscillator (RO) steps shared by the RO figures.

The RO figures (fig:RO-validation, fig:gr-over-time, fig:RO-experiments*) all fit
the same nonlinear RO in 40-year windows of CESM2 output and simulate it with the
fitted parameters. These functions do that once, the same way for every figure.
They follow the "Load data" to "validate" cells of explore/RO-change-over-time_NL.py.
The RO model is ro_tools.xro.XRO from ~/ro_tools (installed editable in envs/);
with these settings it gives the same fits and simulations as the old copy in
src/XRO.py (to roundoff), so the figures match the paper drafts.

The RO's variables are non-dimensional: T is divided by the standard deviation
of the Niño 3 index (all members and times) and h by that of the western
Pacific thermocline index h_w.
"""

import warnings

import numpy as np
import xarray as xr

## the RO model (Sen Zhao's XRO, modified), and its noise estimate, from ~/ro_tools
import ro_tools.fit
import ro_tools.validate
import ro_tools.xro

import src.data
import src.stats
import src.utils


## windows: 40 years long, centered every 10 years; the window centered on 2090
## runs past the end of the simulation, so 2080 is the last one kept
WINDOW_SIZE = 480


STRIDE = 120


LAST_YEAR = 2080


## RO variables: Niño 3 SST index and thermocline depth with its linear
## dependence on T removed ("h_hat")
VARNAMES = ["T_3", "h_hat"]


## which nonlinear terms to fit: T^2 and T^3 in the T equation, T^2 in the h
## equation; ac_mask_idx=None lets every term vary with the seasonal cycle
FIT_KWARGS = dict(ac_mask_idx=None, maskNT=["T2", "T3"], maskNH=["T2"])


## entries of ac_mask_idx (fit_matrix): (equation, predictor) pairs whose
## coefficient has no seasonal cycle. Equations: 0 = T, 1 = h. With the
## nonlinear terms of FIT_KWARGS and fit_constant=True, the predictors are
## T, h, T^2, T^3, constant (T equation) and T, h, T^2, constant (h equation).
## The constant in both equations:
AC_MASK_CONSTANT = [(0, 4), (1, 3)]


## the nonlinear terms: T^2 and T^3 in the T equation, T^2 in the h equation
AC_MASK_NONLINEAR = [(0, 2), (0, 3), (1, 2)]


def get_fit_kwargs(fit_constant=False, flat_constant=False, flat_nonlinear=False):
    """
    Arguments for fit_matrix: FIT_KWARGS's nonlinear terms, optionally with a
    constant and with some coefficients' seasonal cycle removed. The defaults
    give FIT_KWARGS.

    Args:
        - fit_constant (bool): fit a constant in each equation
        - flat_constant (bool): no seasonal cycle on the constant (needs
          fit_constant)
        - flat_nonlinear (bool): no seasonal cycle on the nonlinear terms

    Returns:
        - fit_kwargs (dict): passed to model.fit_matrix (via fit_over_time)
    """

    ## the (equation, predictor) pairs whose coefficient has no seasonal cycle
    ac_mask_idx = []
    if flat_constant:
        assert fit_constant, "flat_constant needs fit_constant"
        ac_mask_idx += AC_MASK_CONSTANT
    else:
        pass
    if flat_nonlinear:
        ac_mask_idx += AC_MASK_NONLINEAR
    else:
        pass

    ## fit_matrix takes None for "every coefficient has a seasonal cycle"
    if len(ac_mask_idx) == 0:
        ac_mask_idx = None
    else:
        pass

    ## FIT_KWARGS with these settings
    fit_kwargs = dict(FIT_KWARGS, fit_constant=fit_constant, ac_mask_idx=ac_mask_idx)

    return fit_kwargs


## simulations: 500 copies of 40 years for each window, with white noise whose
## covariance and seasonal amplitude come from the fit residuals. The noise is
## scaled up by 1.45, a value tuned by hand so the simulated variance matches
## CESM2's (the fit's residuals underestimate the noise needed). The seed makes
## the simulations repeatable.
SIM_KWARGS = dict(
    nyear=40,
    ncopy=500,
    seed=1000,
    noise_type="white",
    use_noise_cov=True,
    is_xi_stdac=True,
    noise_scale=1.45,
)


def get_model(taus=1, ac_order=3, is_forward=False):
    """
    The RO model used for all fits: seasonal cycle with 12 steps per year and
    ac_order harmonics (3 by default), fitted with centered time differences (or
    forward ones). The covariances in the fit use lag taus for each harmonic.
    XRO's default for centered differences is 1; with 2, the lagged state no
    longer overlaps the centered difference, so it shares none of its noise. For
    forward differences, lag 1 is enough.

    Args:
        - taus (int): lag (time steps) of the fit's covariances, for every harmonic
        - ac_order (int): number of harmonics of every seasonal coefficient
        - is_forward (bool): forward time differences (else centered)

    Returns:
        - model (ro_tools.xro.XRO): model object (fits and simulates)
    """

    ## monthly steps; the same lag for the mean and each harmonic
    model = ro_tools.xro.XRO(
        ncycle=12,
        ac_order=ac_order,
        is_forward=is_forward,
        taus=[taus] * (ac_order + 1),
    )

    return model


def load_ro_data(t_var="T_3", remove_median=True, h_by_month=True):
    """
    Windowed, non-dimensional T and h_hat for fitting the RO.

    Steps (as in explore/RO-change-over-time_NL.py): load the indices (ensemble mean
    removed); fill the one missing h value (January 1850, one member) with 0;
    divide T by the standard deviation of T_3 and h by that of h_w; cut into
    windows; remove h's linear dependence on T (pooling all windows, and
    separately for each calendar month, or not); and remove each window's median
    (or mean) for each calendar month.

    Args:
        - t_var (str): SST index for T, from Th_anom.nc ("T_3", "T_34", ...)
        - remove_median (bool): if True, remove each window's median for each
          calendar month (ENSO is skewed, so the median is a better "neutral"
          state); if False, remove the mean
        - h_by_month (bool): if True, regress h on T separately for each calendar
          month (12 coefficients); if False, use one coefficient for all months

    Returns:
        - data (xr.Dataset): t_var and "h_hat", dims (member, year, time),
          non-dimensional
        - T_scale (float): standard deviation of T_3 (K), to convert T back to K
    """

    ## monthly indices (ensemble mean removed), dims (member, time)
    Th = src.utils.load_cesm_indices()[list({t_var, "T_3", "h", "h_w"})]

    ## h is missing for one member in January 1850
    Th = Th.fillna(0)

    ## scales for non-dimensionalizing: overall standard deviations
    T_scale = Th["T_3"].std().values.item()
    h_scale = Th["h_w"].std().values.item()

    ## non-dimensional T and h
    data = xr.Dataset()
    data[t_var] = Th[t_var] / T_scale
    data["h"] = Th["h"] / h_scale

    ## cut into windows, dims (member, year, time); keep full windows
    data = src.utils.get_windowed(data, window_size=WINDOW_SIZE, stride=STRIDE)
    data = data.sel(year=slice(None, LAST_YEAR))

    ## h without its linear dependence on T: one regression per calendar month,
    ## or one for all months, pooling members, times and windows
    all_dims = ["time", "member", "year"]
    if h_by_month:
        data["h_hat"] = src.utils.remove_sst_dependence_v2(
            Th=data, h_var="h", T_var=t_var, dims=all_dims
        )
    else:
        data["h_hat"] = src.utils.remove_sst_dependence_core(
            h=data["h"], T=data[t_var], dims=all_dims
        )

    ## each window's median or mean for each calendar month (over members, times)
    if remove_median:
        center = data.groupby("time.month").median(["time", "member"])
    else:
        center = data.groupby("time.month").mean(["time", "member"])

    ## remove it
    data = data.groupby("time.month") - center

    return data[[t_var, "h_hat"]], T_scale


def fit_window(data, model, fit_kwargs=FIT_KWARGS, nonpositive_T3=False):
    """
    Fit the RO to one window (or any record), pooling all ensemble members.

    Args:
        - data (xr.Dataset): RO variables, dims (member, time) or (time,)
        - model (ro_tools.xro.XRO): model to fit (from get_model)
        - fit_kwargs (dict): passed to model.fit_matrix
        - nonpositive_T3 (bool): if True, bound the T^3 coefficient of the T
          equation at zero in every month, so the cubic term always damps
          (ro_tools.validate.fit_with_T3_bound)

    Returns:
        - fit (xr.Dataset): fitted parameters and residual statistics
    """

    ## silencing the fit's warnings, as the old script did
    with warnings.catch_warnings(action="ignore"):
        if nonpositive_T3:
            fit = ro_tools.validate.fit_with_T3_bound(model, data, **fit_kwargs)

            ## the months held at the bound, as a variable (attrs don't survive
            ## stacking the windows)
            fit["T3_bound_months"] = xr.DataArray(fit.attrs["T3_bound_months"])
        else:
            fit = model.fit_matrix(data, **fit_kwargs)

    return fit


def fit_over_time(
    data, model, fit_kwargs=FIT_KWARGS, fit_noise=False, nonpositive_T3=False
):
    """
    Fit the RO separately in each window, pooling all ensemble members.

    Args:
        - data (xr.Dataset): RO variables, dims (member, year, time), as from
          load_ro_data
        - model (ro_tools.xro.XRO): model to fit (from get_model)
        - fit_kwargs (dict): passed to model.fit_matrix
        - fit_noise (bool): if True, replace the noise from the fit's residuals
          with ro_tools.fit.set_fit_noise's estimate, which corrects for the
          residuals' memory (so the simulations need no noise_scale)
        - nonpositive_T3 (bool): bound the T^3 coefficient at zero in every
          month (fit_window)

    Returns:
        - fits (xr.Dataset): fitted parameters and residual statistics, with a
          "year" dimension
    """

    ## fit each window
    fits = []
    for year in data.year:
        fit = fit_window(data.sel(year=year), model, fit_kwargs, nonpositive_T3)
        with warnings.catch_warnings(action="ignore"):

            ## memory-corrected noise, on the same harmonics as the operator
            if fit_noise:
                fit = ro_tools.fit.set_fit_noise(
                    fit, ac_order=model.ac_order, use_covariance=True
                )
            else:
                pass
        fits.append(fit)

    ## stack the windows along "year"
    fits = xr.concat(fits, dim=data.year)

    return fits


def get_R_at(fits, data, n_sigma, t_var="T_3"):
    """
    The SST growth rate R for an anomaly of a given size, in each window and
    calendar month.

    The fitted T equation has dT/dt = R_1 T + R_2 T^2 + R_3 T^3 + (h and noise
    terms), so the growth rate of an anomaly of size T is
    R(T) = R_1 + R_2 T + R_3 T^2. T is set to n_sigma standard deviations of T
    for that calendar month (pooling members and times in each window, then
    averaging over windows), as in the "Get coefficients" to "Do the
    reconstruction" cells of explore/RO-change-over-time_NL.py.

    Args:
        - fits (xr.Dataset): fitted parameters, with a "year" dimension (from
          fit_over_time)
        - data (xr.Dataset): the data the RO was fitted to, dims (member, year,
          time) (from load_ro_data)
        - n_sigma (float): anomaly size, in standard deviations (e.g. 1.5 for
          warm, -1.5 for cold, 0 for the linear growth rate)
        - t_var (str): name of the T variable in data

    Returns:
        - R (xr.DataArray): growth rate (yr^-1), dims (year, cycle)
    """

    ## standard deviation of T for each calendar month and window, then the
    ## average over windows (so the anomaly size is the same in every window)
    sigma_T = data[t_var].groupby("time.month").std(["time", "member"]).mean("year")

    ## match the fits' coordinate for the seasonal cycle
    sigma_T = sigma_T.rename({"month": "cycle"}).assign_coords(cycle=fits["cycle"])

    ## anomaly size for each calendar month
    T = n_sigma * sigma_T

    ## linear, quadratic, and cubic coefficients of the T equation
    R_1 = fits["Lac"].isel(ranky=0, rankx=0)
    R_2 = fits["NROT_Lac"].sel(nro_form="T2")
    R_3 = fits["NROT_Lac"].sel(nro_form="T3")

    ## growth rate of an anomaly of size T
    R = R_1 + R_2 * T + R_3 * T**2

    return R


def get_eps(fits):
    """
    The thermocline damping rate epsilon (dh/dt = -epsilon h + ...), in each
    window and calendar month.

    Args:
        - fits (xr.Dataset): fitted parameters (from fit_over_time)

    Returns:
        - eps (xr.DataArray): damping rate (yr^-1), dims (year, cycle)
    """

    ## minus the h coefficient of the h equation
    eps = -fits["Lac"].isel(ranky=1, rankx=1)

    return eps


def get_params(fits):
    """
    The RO's coefficients in one Dataset, with the usual RO signs:
    dT/dt = R_1 T + F_1 h + R_2 T^2 + R_3 T^3 + c_T and
    dh/dt = -F_2 T - eps h + b_h T^2 + c_h (+ noise).

    Args:
        - fits (xr.Dataset): fitted parameters (from fit_over_time or fit_matrix)

    Returns:
        - params (xr.Dataset): "R1", "F1", "F2", "eps", "R2", "R3", "bh", and
          "cT", "ch" if a constant was fitted; dims (cycle,) plus any of the
          fits' other dims (e.g. year)
    """

    ## the linear operator: rows are equations (T, h), columns predictors (T, h)
    L = fits["Lac"]
    terms = dict(
        R1=L.isel(ranky=0, rankx=0),
        F1=L.isel(ranky=0, rankx=1),
        F2=-L.isel(ranky=1, rankx=0),
        eps=-L.isel(ranky=1, rankx=1),
    )

    ## nonlinear terms of the T and h equations
    terms["R2"] = fits["NROT_Lac"].sel(nro_form="T2")
    terms["R3"] = fits["NROT_Lac"].sel(nro_form="T3")
    terms["bh"] = fits["NROH_Lac"].sel(nro_form="T2")

    ## the constant, if fitted
    if "const_Lac" in fits:
        terms["cT"] = fits["const_Lac"].isel(ranky=0)
        terms["ch"] = fits["const_Lac"].isel(ranky=1)
    else:
        pass

    ## drop the rank/form labels, which differ between terms
    params = xr.Dataset()
    for name, x in terms.items():
        params[name] = x.reset_coords(drop=True)

    return params


## the RO's terms, as lists of (fit variable, entries) pairs, for the
## perturbation experiments (fix_terms). The simulation reads the linear
## operator "Lac" (ranky/rankx: 1 = T, 2 = h; ranky is the equation), the
## nonlinear terms of the T and h equations ("NROT_Lac", "NROH_Lac"), and the
## noise's seasonal standard deviation and covariance ("xi_stdac", "xi_covac").
## The T*h and h^2 terms of F1 are zero in these fits (not in FIT_KWARGS) but
## are listed so that F1 stays complete if they're fitted.
RO_TERMS = dict(
    ## SST growth rate: R_1 T + R_2 T^2 + R_3 T^3 in the T equation
    R=[
        ("Lac", dict(ranky=1, rankx=1)),
        ("NROT_Lac", dict(nro_form="T2")),
        ("NROT_Lac", dict(nro_form="T3")),
    ],
    ## thermocline feedback: h-dependent terms of the T equation
    F1=[
        ("Lac", dict(ranky=1, rankx=2)),
        ("NROT_Lac", dict(nro_form="TH")),
        ("NROT_Lac", dict(nro_form="H2")),
    ],
    ## recharge by T (Sverdrup transport): T-dependent terms of the h equation
    F2=[("Lac", dict(ranky=2, rankx=1)), ("NROH_Lac", dict(nro_form="T2"))],
    ## thermocline damping: h term of the h equation
    eps=[("Lac", dict(ranky=2, rankx=2))],
    ## stochastic forcing of both equations
    noise=[("xi_stdac", dict()), ("xi_covac", dict())],
)


def fix_terms(fits, terms, ref_year):
    """
    Copy of the fits with some RO terms fixed to their values in a reference
    window (the other terms still change over time). Simulating with these
    parameters shows what happens without the changes in those terms.

    Args:
        - fits (xr.Dataset): fitted parameters with a "year" dimension (from
          fit_over_time)
        - terms (list of tuple): (fit variable, entries) pairs, e.g. from
          RO_TERMS; entries is a dict of coordinate labels (empty for all)
        - ref_year (int): window whose values the terms are fixed to

    Returns:
        - fixed (xr.Dataset): same as fits, with the terms fixed
    """

    ## copy, so the original fits are unchanged
    fixed = fits.copy(deep=True)

    ## fix each term in turn (so several terms can share a fit variable)
    for name, entries in terms:
        x = fixed[name]

        ## True at the term's entries, False elsewhere
        is_term = xr.zeros_like(x, dtype=bool)
        is_term.loc[entries] = True

        ## the reference window's values (broadcast over windows)
        x_ref = x.sel(year=ref_year, drop=True)

        ## replace the term's entries, keeping the original order of dims
        fixed[name] = xr.where(is_term, x_ref, x).transpose(*x.dims)

    return fixed


def get_amplitudes(sims, T_scale, t_var="T_3"):
    """
    Amplitude of warm and cold events in each window: the tails of the
    distribution of monthly T in DJF (pooling all simulation copies and times).

    Args:
        - sims (xr.Dataset): simulated RO variables (non-dimensional), dims
          (year, time, member) (from simulate_over_time)
        - T_scale (float): standard deviation of T_3 (K) (from load_ro_data)
        - t_var (str): name of the T variable in sims

    Returns:
        - amp (xr.Dataset): "warm" (95th percentile), "cold" (minus the 5th
          percentile), and "mean" (their average), in K, dims (year,)
    """

    ## monthly T in K, Dec-Jan-Feb only
    T = T_scale * sims[t_var]
    T_djf = T.sel(time=T.time.dt.month.isin([12, 1, 2]))

    ## 5th and 95th percentiles, pooling members and times
    q = T_djf.quantile([0.05, 0.95], dim=["time", "member"])

    ## amplitudes of warm and cold events (both positive)
    amp = xr.Dataset()
    amp["warm"] = q.sel(quantile=0.95, drop=True)
    amp["cold"] = -q.sel(quantile=0.05, drop=True)

    ## average amplitude
    amp["mean"] = 0.5 * (amp["warm"] + amp["cold"])

    return amp


def get_contributions(sims, T_scale, t_var="T_3"):
    """
    Contributions of warm and cold DJF anomalies to the variance of T in each
    window (src.stats.get_variance_contributions), for CESM2 data or RO
    simulations. Unlike get_amplitudes, this doesn't depend on how T was
    centered, since each window's own median is the reference.

    Args:
        - sims (xr.Dataset): RO variables (non-dimensional), dims (year, time,
          member) (from simulate_over_time or load_ro_data)
        - T_scale (float): standard deviation of T_3 (K) (from load_ro_data)
        - t_var (str): name of the T variable in sims

    Returns:
        - contrib (xr.Dataset): "total", "warm" and "cold" (K^2), dims
          (member, year)
    """

    ## DJF means in K (full seasons only)
    T_djf = T_scale * src.data.get_djf(sims[t_var])

    ## warm and cold contributions about each window's median
    contrib = src.stats.get_variance_contributions(T_djf)

    return contrib


def simulate_over_time(model, fits, X0, sim_kwargs=SIM_KWARGS):
    """
    Simulate the RO with each window's fitted parameters.

    Args:
        - model (ro_tools.xro.XRO): model used for the fits
        - fits (xr.Dataset): fitted parameters with a "year" dimension (from
          fit_over_time)
        - X0 (xr.Dataset): initial condition (one value of each RO variable)
        - sim_kwargs (dict): passed to model.simulate

    Returns:
        - sims (xr.Dataset): simulated RO variables, dims (year, time, member),
          where "member" counts the simulation copies
    """

    ## simulate each window (the same seed for each)
    sims = []
    for year in fits.year:
        sims.append(model.simulate(fit_ds=fits.sel(year=year), X0_ds=X0, **sim_kwargs))

    ## stack the windows along "year"
    sims = xr.concat(sims, dim=fits.year)

    return sims


## surface heat fluxes for the feedback alpha: net, shortwave, latent, longwave
FLUX_TYPES = ["nhf", "fsns", "lhflx", "flns"]

## powers of T in the SST growth rate R(T) T = R_1 T + R_2 T^2 + R_3 T^3, in the
## same order as RO_TERMS["R"]
R_PARAMS = ["T1", "T2", "T3"]


def load_ro_flux_data(T_scale, flux_types=FLUX_TYPES, H=50, remove_median=True):
    """
    Windowed, non-dimensional surface heat fluxes over Niño 3, prepared the same
    way as T in load_ro_data, for estimating the heat flux feedback alpha.

    Each flux is converted to the warming rate it gives a mixed layer of depth H
    (K/yr; positive into the ocean), divided by T_scale (so it has the units of
    the RO's dT/dt), cut into windows, and has each window's median (or mean)
    removed for each calendar month.

    Args:
        - T_scale (float): standard deviation of T_3 (K) (from load_ro_data)
        - flux_types (list of str): fluxes to load (see src.data.load_Q)
        - H (float): mixed-layer depth (m)
        - remove_median (bool): remove each window's median (True) or mean
          (False) for each calendar month, as in load_ro_data

    Returns:
        - Q (xr.Dataset): one variable per flux type (yr^-1, non-dimensional),
          dims (member, year, time)
    """

    ## non-dimensional warming rate from each flux, dims (member, time)
    Q = xr.Dataset()
    for flux_type in flux_types:
        Q[flux_type] = src.data.load_Q(H=H, varname=flux_type, region="3") / T_scale

    ## cut into windows, dims (member, year, time); keep full windows
    Q = src.utils.get_windowed(Q, window_size=WINDOW_SIZE, stride=STRIDE)
    Q = Q.sel(year=slice(None, LAST_YEAR))

    ## each window's median or mean for each calendar month (over members, times)
    if remove_median:
        center = Q.groupby("time.month").median(["time", "member"])
    else:
        center = Q.groupby("time.month").mean(["time", "member"])

    ## remove it
    Q = Q.groupby("time.month") - center

    return Q


def fit_alpha(T, Q, cycle):
    """
    Heat flux feedback alpha in each window and calendar month: minus the
    coefficients of the cubic fit Q = a_1 T + a_2 T^2 + a_3 T^3 (least squares,
    pooling members and times), so that the flux damps T at the rate
    alpha(T) = alpha_1 + alpha_2 T + alpha_3 T^2. As in the "Compute alpha" cells
    of explore/RO-change-over-time_NL.py (regress_pinv).

    Args:
        - T (xr.DataArray): non-dimensional T, dims (member, year, time) (from
          load_ro_data)
        - Q (xr.DataArray): non-dimensional heat flux, same dims (from
          load_ro_flux_data)
        - cycle (xr.DataArray): the fits' seasonal-cycle coordinate (one value
          per calendar month), to label the result like the RO's parameters

    Returns:
        - alpha (xr.DataArray): coefficients alpha_1, alpha_2, alpha_3, dims
          (year, cycle, param), with param = R_PARAMS
    """

    ## predictors T, T^2, T^3, dims (member, year, time, param)
    powers = []
    for i in range(1, 4):
        powers.append(T**i)
    X = xr.concat(powers, dim=xr.DataArray(R_PARAMS, dims="param"))

    ## one regression per calendar month
    alpha = []
    for month in range(1, 13):
        is_month = T.time.dt.month == month

        ## pool members and times into one sample dim "s", window first
        X_m = X.sel(time=is_month).stack(s=["time", "member"])
        X_m = X_m.transpose("year", "s", "param")

        ## same for the flux (the samples line up with X_m's)
        Q_m = Q.sel(time=is_month).stack(s=["time", "member"])
        Q_m = Q_m.transpose("year", "s")

        ## least-squares coefficients in each window, dims (year, param)
        coef = np.einsum("yps,ys->yp", np.linalg.pinv(X_m.values), Q_m.values)
        coef = xr.DataArray(coef, coords=dict(year=T.year, param=R_PARAMS))

        ## alpha is minus the coefficient (positive alpha damps T)
        alpha.append(-coef)

    ## stack the calendar months along the fits' seasonal-cycle coordinate
    alpha = xr.concat(alpha, dim=cycle).transpose("year", "cycle", "param")

    return alpha


def get_harmonics(month, ac_order=3):
    """
    Seasonal-cycle basis functions like the RO's: 1, cos(k w t), sin(k w t) for
    k = 1..ac_order, with t the middle of the calendar month in years
    ((month - 0.5) / 12, the fits' "cycle" coordinate) and w = 2 pi / yr.

    Args:
        - month (xr.DataArray): calendar month (1-12) of each sample
        - ac_order (int): number of harmonics

    Returns:
        - basis (xr.DataArray): month's dims plus "harmonic" (2 ac_order + 1)
    """

    ## phase of the middle of each month
    phase = 2 * np.pi * (month - 0.5) / 12

    ## the mean, then a cosine and a sine for each harmonic
    basis = [xr.ones_like(phase)]
    for k in range(1, ac_order + 1):
        basis.append(np.cos(k * phase))
        basis.append(np.sin(k * phase))
    basis = xr.concat(basis, dim="harmonic")

    return basis


def fit_alpha_harmonic(T, Q, cycle, ac_order=3, flat_nonlinear=False):
    """
    Heat flux feedback alpha with the same functional form as the RO's R: the
    linear coefficient's seasonal cycle has ac_order harmonics, and the
    nonlinear coefficients either have the same seasonal cycle or none
    (flat_nonlinear, matching get_fit_kwargs(flat_nonlinear=True)). One least-
    squares fit per window, pooling members, times and all calendar months:
        Q = -[alpha_1(t) T + alpha_2(t) T^2 + alpha_3(t) T^3]
    Then alpha(T) = alpha_1 + alpha_2 T + alpha_3 T^2 damps T, as in fit_alpha,
    and R_o = R + alpha has the same form as R.

    Args:
        - T (xr.DataArray): non-dimensional T, dims (member, year, time) (from
          load_ro_data)
        - Q (xr.DataArray): non-dimensional heat flux, same dims (from
          load_ro_flux_data)
        - cycle (xr.DataArray): the fits' seasonal-cycle coordinate (one value
          per calendar month), where alpha is evaluated
        - ac_order (int): number of harmonics (as in get_model)
        - flat_nonlinear (bool): no seasonal cycle on alpha_2 and alpha_3

    Returns:
        - alpha (xr.DataArray): coefficients alpha_1, alpha_2, alpha_3, dims
          (year, cycle, param), with param = R_PARAMS
    """

    ## seasonal basis for each sample, and evaluated at each month of the cycle
    basis = get_harmonics(T.time.dt.month, ac_order=ac_order)
    basis_cycle = get_harmonics(xr.DataArray(np.arange(1, 13), dims="cycle"), ac_order)
    basis_cycle = basis_cycle.assign_coords(cycle=cycle.values)

    ## predictors: T^i times each basis function, or (flat) times 1 only
    X = []
    labels = []
    for i, param in enumerate(R_PARAMS, start=1):

        ## how many basis functions this power of T gets
        if flat_nonlinear and (i > 1):
            n_basis = 1
        else:
            n_basis = 2 * ac_order + 1

        ## one predictor per basis function, labeled (param, basis index)
        for j in range(n_basis):
            X.append(T**i * basis.isel(harmonic=j))
            labels.append((param, j))
    X = xr.concat(X, dim="predictor")

    ## pool members and times into one sample dim "s", window first
    X = X.stack(s=["time", "member"]).transpose("year", "s", "predictor")
    Q_s = Q.stack(s=["time", "member"]).transpose("year", "s")

    ## least-squares coefficients in each window, dims (year, predictor)
    coef = np.einsum("yps,ys->yp", np.linalg.pinv(X.values), Q_s.values)

    ## each alpha coefficient's seasonal cycle: its basis coefficients times
    ## the basis at each month (minus, since positive alpha damps T)
    alpha = []
    for param in R_PARAMS:

        ## start from zero in every window and month
        alpha_p = xr.zeros_like(basis_cycle.isel(harmonic=0), dtype=float)
        alpha_p = alpha_p.expand_dims(year=T.year.values).copy()

        ## add this coefficient's basis functions
        for k, (param_k, j) in enumerate(labels):
            if param_k == param:
                c = xr.DataArray(coef[:, k], coords=dict(year=T.year.values))
                alpha_p = alpha_p - c * basis_cycle.isel(harmonic=j, drop=True)
            else:
                pass
        alpha.append(alpha_p)

    ## stack along "param", in the same layout as fit_alpha
    alpha = xr.concat(alpha, dim=xr.DataArray(R_PARAMS, dims="param"))
    alpha = alpha.transpose("year", "cycle", "param")

    return alpha


## the T equation's coefficients a heat flux contributes to: the powers of T in
## R (R_PARAMS) and h (the flux's part of F_1)
FLUX_PARAMS = R_PARAMS + ["h"]


## each coefficient's (fit variable, entries), as in RO_TERMS
PARAM_TERMS = dict(
    T1=("Lac", dict(ranky=1, rankx=1)),
    T2=("NROT_Lac", dict(nro_form="T2")),
    T3=("NROT_Lac", dict(nro_form="T3")),
    h=("Lac", dict(ranky=1, rankx=2)),
)


def get_flux_tendency(Q, model):
    """
    A flux's warming rate aligned with the fit's time difference: as it is for
    centered differences, and averaged to the half step, (Q_t + Q_t+1) / 2, for
    forward ones (the last month, like XRO's backward difference there, with
    the month before).

    Args:
        - Q (xr.DataArray): warming rate, dims (member, year, time) (from
          load_ro_flux_data)
        - model (ro_tools.xro.XRO): the model (from get_model)

    Returns:
        - Q_fit (xr.DataArray): the same dims
    """

    ## forward differences: the mean of each month and the next
    if model.is_forward:
        Q_next = (Q + Q.shift(time=-1)) / 2

        ## the last month has no next one: the mean with the month before
        Q_before = (Q + Q.shift(time=1)) / 2
        Q_fit = Q_next.combine_first(Q_before)

    ## centered differences: as it is
    else:
        Q_fit = Q

    return Q_fit


def fit_flux_feedback(data, Q, model, fit_kwargs=FIT_KWARGS, t_var="T_3"):
    """
    The part of the RO's T equation due to a heat flux: XRO's fit of the flux's
    warming rate Q in place of dT/dt, with the same predictors, seasonal
    harmonics, masks and lagged covariances as the RO (fit_matrix's tendency).
    The fit is linear in the tendency, so the fits of Q and of dT/dt - Q add up
    to the RO's T equation exactly; R_o = R + alpha is then the rest. (The T^3
    bound isn't applied here: alpha's terms may have either sign.) Only the T
    equation's coefficients are the flux's; the h equation is the RO's own.

    Args:
        - data (xr.Dataset): RO variables, dims (member, year, time), as from
          load_ro_data
        - Q (xr.DataArray): the flux's warming rate, dims (member, year, time)
          (from load_ro_flux_data)
        - model (ro_tools.xro.XRO): model (from get_model)
        - fit_kwargs (dict): the RO's fit settings (passed to model.fit_matrix)
        - t_var (str): name of T in data

    Returns:
        - flux_fits (xr.Dataset): the coefficients ("Lac", "NROT_Lac", and
          "const_Lac" if fitted), with a "year" dimension
    """

    ## the tendencies: Q for T (aligned with the time difference); h's own time
    ## difference for h (a zero tendency would leave XRO's solve empty), so the
    ## h equation is the RO's and isn't used
    tendency = xr.Dataset()
    for name in data.data_vars:
        if name == t_var:
            tendency[name] = get_flux_tendency(Q, model).transpose(*data[name].dims)
        else:
            difference = ro_tools.xro.gradient(
                data[name].values,
                axis=data[name].get_axis_num("time"),
                is_forward=model.is_forward,
                ncycle=model.ncycle,
            )
            tendency[name] = data[name].copy(data=difference)

    ## the coefficients the fit returns
    names = ["Lac", "NROT_Lac"]
    if fit_kwargs.get("fit_constant", False):
        names.append("const_Lac")
    else:
        pass

    ## fit each window
    flux_fits = []
    for year in data.year:
        with warnings.catch_warnings(action="ignore"):
            fit = model.fit_matrix(
                data.sel(year=year), tendency=tendency.sel(year=year), **fit_kwargs
            )
        flux_fits.append(fit[names])

    ## stack the windows along "year"
    flux_fits = xr.concat(flux_fits, dim=data.year)

    return flux_fits


def get_alpha(flux_fits, params=FLUX_PARAMS):
    """
    The heat flux feedback alpha from a flux's fit (fit_flux_feedback): minus
    its coefficients, so that a positive alpha damps T (as in fit_alpha).

    Args:
        - flux_fits (xr.Dataset): from fit_flux_feedback
        - params (list of str): coefficients, keys of PARAM_TERMS

    Returns:
        - alpha (xr.DataArray): dims (year, cycle, param)
    """

    ## each coefficient's entry, with the sign of a damping
    coefs = []
    for param in params:
        name, entries = PARAM_TERMS[param]
        coefs.append(-flux_fits[name].sel(entries, drop=True))

    ## stack along "param"
    alpha = xr.concat(coefs, dim=xr.DataArray(params, dims="param"))
    alpha = alpha.transpose("year", "cycle", "param")

    return alpha


def get_R_coefs(fits):
    """
    Coefficients of the SST growth rate, R_1, R_2, R_3 (R(T) = R_1 + R_2 T +
    R_3 T^2), in one array.

    Args:
        - fits (xr.Dataset): fitted parameters (from fit_over_time)

    Returns:
        - R (xr.DataArray): dims (year, cycle, param), with param = R_PARAMS
    """

    ## each coefficient's entry in the fits
    coefs = []
    for name, entries in RO_TERMS["R"]:
        coefs.append(fits[name].sel(entries, drop=True))

    ## stack along "param"
    R = xr.concat(coefs, dim=xr.DataArray(R_PARAMS, dims="param"))
    R = R.transpose("year", "cycle", "param")

    return R


def remove_change_in_R(fits, part, ref_year):
    """
    Copy of the fits in which one part of the SST growth rate R stays at its
    value in a reference window, while the rest of R (and all other terms)
    still change over time: R is replaced by R - (part - part(ref_year)).

    For example, with R = R_o - alpha, part = -alpha fixes the heat flux
    feedback and part = R_o fixes the ocean part.

    Args:
        - fits (xr.Dataset): fitted parameters with a "year" dimension (from
          fit_over_time)
        - part (xr.DataArray): the part's contribution to R_1, R_2, R_3 (and
          optionally F_1, param "h"), dims (year, cycle, param) (like
          get_R_coefs, get_alpha)
        - ref_year (int): window whose value the part is fixed to

    Returns:
        - fixed (xr.Dataset): same as fits, with R changed
    """

    ## change in the part since the reference window
    change = part - part.sel(year=ref_year)

    ## copy, so the original fits are unchanged
    fixed = fits.copy(deep=True)

    ## subtract the change from each coefficient in the part
    for param in part.param.values:
        name, entries = PARAM_TERMS[param]
        x = fixed[name]

        ## True at the coefficient's entry, False elsewhere
        is_term = xr.zeros_like(x, dtype=bool)
        is_term.loc[entries] = True

        ## subtract the change there, keeping the original order of dims
        x_new = x - change.sel(param=param, drop=True)
        fixed[name] = xr.where(is_term, x_new, x).transpose(*x.dims)

    return fixed
