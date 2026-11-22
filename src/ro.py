"""
The recharge oscillator (RO) steps shared by the RO figures.

The RO figures (fig:RO-validation, fig:gr-over-time, fig:RO-experiments*) all fit
the same nonlinear RO in 40-year windows of CESM2 output and simulate it with the
fitted parameters. These functions do that once, the same way for every figure.
They follow the "Load data" to "validate" cells of RO-change-over-time_NL.py,
and use the old RO implementation in src/XRO.py so that the figures match the
paper drafts; the plan is to switch to ~/ro_tools later.

The RO's variables are non-dimensional: T is divided by the standard deviation
of the Niño 3 index (all members and times) and h by that of the western
Pacific thermocline index h_w.
"""

import warnings

import xarray as xr

import src.utils
import src.XRO


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


def get_model():
    """
    The RO model used for all fits: seasonal cycle with 12 steps per year and 3
    harmonics, fitted with a backward time difference.

    Returns:
        - model (src.XRO.XRO): model object (fits and simulates)
    """

    ## monthly steps, annual + semi-annual + 4-month harmonics
    model = src.XRO.XRO(ncycle=12, ac_order=3, is_forward=False)

    return model


def load_ro_data(t_var="T_3", remove_median=True):
    """
    Windowed, non-dimensional T and h_hat for fitting the RO.

    Steps (as in RO-change-over-time_NL.py): load the indices (ensemble mean
    removed); fill the one missing h value (January 1850, one member) with 0;
    divide T by the standard deviation of T_3 and h by that of h_w; cut into
    windows; remove h's linear dependence on T (separately for each calendar
    month, pooling all windows); and remove each window's median (or mean) for
    each calendar month.

    Args:
        - t_var (str): SST index for T, from Th_anom.nc ("T_3", "T_34", ...)
        - remove_median (bool): if True, remove each window's median for each
          calendar month (ENSO is skewed, so the median is a better "neutral"
          state); if False, remove the mean

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

    ## h without its linear dependence on T (one regression per calendar month)
    data["h_hat"] = src.utils.remove_sst_dependence_v2(
        Th=data, h_var="h", T_var=t_var, dims=["time", "member", "year"]
    )

    ## each window's median or mean for each calendar month (over members, times)
    if remove_median:
        center = data.groupby("time.month").median(["time", "member"])
    else:
        center = data.groupby("time.month").mean(["time", "member"])

    ## remove it
    data = data.groupby("time.month") - center

    return data[[t_var, "h_hat"]], T_scale


def fit_over_time(data, model, fit_kwargs=FIT_KWARGS):
    """
    Fit the RO separately in each window, pooling all ensemble members.

    Args:
        - data (xr.Dataset): RO variables, dims (member, year, time), as from
          load_ro_data
        - model (src.XRO.XRO): model to fit (from get_model)
        - fit_kwargs (dict): passed to model.fit_matrix

    Returns:
        - fits (xr.Dataset): fitted parameters and residual statistics, with a
          "year" dimension
    """

    ## fit each window (silencing the fit's warnings, as the old script did)
    fits = []
    for year in data.year:
        with warnings.catch_warnings(action="ignore"):
            fits.append(model.fit_matrix(data.sel(year=year), **fit_kwargs))

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
    reconstruction" cells of RO-change-over-time_NL.py.

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


def simulate_over_time(model, fits, X0, sim_kwargs=SIM_KWARGS):
    """
    Simulate the RO with each window's fitted parameters.

    Args:
        - model (src.XRO.XRO): model used for the fits
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
