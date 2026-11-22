"""
Statistics: changes over time, composites, and regressions.
"""

import numpy as np
import xarray as xr

import src.constants
import src.data
import src.utils


def frac_change(x, x0=None):
    """
    Fractional change relative to a baseline, (x - x0) / x0.

    Args:
        - x (xr.DataArray): data with a "year" dimension
        - x0 (xr.DataArray or float): baseline; if None, use the first year of x

    Returns:
        - frac (xr.DataArray): fractional change (0.1 means +10%)
    """

    ## default baseline is the first window
    if x0 is None:
        x0 = x.isel(year=0)
    else:
        pass

    ## compute the fractional change
    frac = (x - x0) / x0

    return frac


def get_composite_hov(
    anom,
    T34,
    varnames,
    years,
    window_size=480,
    peak_month=12,
    q=0.95,
    is_warm=True,
    event_type=None,
):
    """
    Equatorial (LATS_EQ) composites of strong ENSO events in each window.

    In each window, events are the months `peak_month` when T34 is beyond its q
    (El Niño) or 1 - q (La Niña) quantile over that calendar month; each event's
    sample runs from lag -12 to +12 months. The composite is the mean over events.

    Args:
        - anom (xr.Dataset): EOF-compressed anomalies (ensemble mean removed): for
          each name in varnames, the scores (member, time, mode) and the patterns
          in "<name>_comp"
        - T34 (xr.DataArray): Niño 3.4 anomalies (member, time), used to pick events
        - varnames (list of str): fields to composite
        - years (list of int): window centers (see get_window)
        - window_size (int): window length (months)
        - peak_month (int): calendar month of the event peak (lag 0)
        - q (float): quantile defining a strong event
        - is_warm (bool): True for El Niño, False for La Niña
        - event_type (int or None): None keeps all events; 1, 2, or 3 keeps only
          single-, double-, or triple-year events (consecutive peaks beyond the
          cutoff; see src.utils.sort_peak_idx). For a multi-year event, the
          composite uses its first peak

    Returns:
        - comp (xr.Dataset): dims (year, lag, longitude) for each field in varnames
          (meridional mean over LATS_EQ), plus the composite "T_34" (year, lag)
    """

    ## warm events are above the q quantile; cold events below the 1 - q quantile
    if is_warm:
        q_cut = q
        check_cutoff = lambda x, cut: x > cut  # noqa: E731
    else:
        q_cut = 1 - q
        check_cutoff = lambda x, cut: x < cut  # noqa: E731

    ## scores of the fields, plus the index used to pick the events
    scores = xr.merge([anom[varnames], T34.rename("T_34")])

    ## meridional mean, applied to the EOF patterns before reconstructing
    def eq_mean(x):
        """
        Mean over the equatorial band.

        Args:
            - x (xr.DataArray): field with a "latitude" dimension

        Returns:
            - x_eq (xr.DataArray): mean over LATS_EQ
        """
        return x.sel(latitude=slice(*src.constants.LATS_EQ)).mean("latitude")

    ## composite for each window
    comps = []
    for year in years:

        ## this window's months
        window = src.data.get_window(scores, year, window_size)

        ## mean over the events in this window, dims (lag, ...)
        comp_scores = src.utils.make_composite(
            idx=window["T_34"],
            data=window,
            peak_month=peak_month,
            q=q_cut,
            check_cutoff=check_cutoff,
            event_type=event_type,
        )

        ## equatorial fields from the composite scores
        comp = xr.Dataset({"T_34": comp_scores["T_34"]})
        for varname in varnames:
            comp[varname] = src.utils.reconstruct_fn(
                components=anom[f"{varname}_comp"],
                scores=comp_scores[varname],
                fn=eq_mean,
            )
        comps.append(comp)

    ## stack the windows along a "year" dimension
    comp = xr.concat(comps, dim=xr.DataArray(years, dims="year", name="year"))

    return comp.transpose("year", "lag", ...)


def regress_by_sign(x, y, by_month=False):
    """
    Regression slope of y on x, separately for x > 0, x < 0, and all x.

    Each member and window gets its own fit: the regression is over "time" only.
    Fitting x > 0 and x < 0 separately gives a piecewise-linear relation, which
    captures El Niño/La Niña asymmetry.

    Args:
        - x (xr.DataArray): predictor, e.g. windowed Niño 3.4 anomalies with dims
          (member, year, time)
        - y (xr.DataArray): target, same dims as x
        - by_month (bool): if True, fit each calendar month separately (adds a
          "month" dimension)

    Returns:
        - slopes (xr.Dataset): slope of y on x for x > 0 ("pos"), x < 0 ("neg"),
          and all x ("all"), with x's dims other than "time" (plus "month")
    """

    ## predictor and target in one dataset, so masking applies to both
    data = xr.merge([x.rename("x"), y.rename("y")])

    ## subsets: all samples, warm (x > 0), and cold (x < 0)
    subsets = dict(
        all=data, pos=data.where(data["x"] > 0), neg=data.where(data["x"] < 0)
    )

    ## regression settings: one fit per member and window
    kwargs = dict(x_vars=["x"], y_vars=["y"], dims=["time"])

    ## fit each subset
    slopes = xr.Dataset()
    for name, subset in subsets.items():

        ## one fit for each calendar month, or one for all months together
        if by_month:
            coefs = subset.groupby("time.month").map(
                src.utils.regress_xr_proj, **kwargs
            )
        else:
            coefs = src.utils.regress_xr_proj(subset, **kwargs)

        ## keep the slope (dropping the predictor-index coordinate)
        slopes[name] = coefs["y"].drop_vars("j")

    return slopes


def get_lag_growth_rate(T, lag=1, month=None, by_month=False):
    """
    Growth rate of T from lag regression, separately for warm and cold starts.

    Regresses T(t + lag) on T(t), giving a slope S per member and window. If
    T(t + lag) = S T(t), then T grows at the rate R = log(S) / lag. Warm and cold
    subsets are chosen by the sign of T(t).

    Args:
        - T (xr.DataArray): windowed monthly anomalies, dims (member, year, time),
          as from src.utils.get_windowed
        - lag (int): months between the two times
        - month (int or None): calendar month of the starting time t; None uses
          all months
        - by_month (bool): if True, fit each starting month separately (adds a
          "month" dimension)

    Returns:
        - R (xr.Dataset): growth rate (yr^-1) for T(t) > 0 ("pos"), T(t) < 0
          ("neg"), and all T(t) ("all")
    """

    ## starting values, and the values `lag` months later on the same time axis;
    ## pairs stay within a window, since the shift is along "time"
    T0 = T.isel(time=slice(None, -lag))
    Tf = T.isel(time=slice(lag, None)).assign_coords(time=T0.time)

    ## (optional) keep only one starting month
    if month is None:
        T0_sel = T0
        Tf_sel = Tf
    else:
        T0_sel = src.utils.sel_month(T0, months=month)
        Tf_sel = src.utils.sel_month(Tf, months=month)

    ## lag-regression slope S
    S = regress_by_sign(T0_sel, Tf_sel, by_month=by_month)

    ## growth rate: log(S) per `lag` months, times 12 months per year
    R = 12 / lag * np.log(S)

    return R
