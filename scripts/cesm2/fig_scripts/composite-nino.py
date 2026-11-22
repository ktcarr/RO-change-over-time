import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Strong El Niño composites, early and late

    Paper figure `fig:composite-nino` (`paper-outline.tex` in the ch3 Overleaf project).
    Composites of strong El Niños (December Niño 3.4 above the 95th percentile) in the
    40-year windows centered on 2010 and 2080.

    - **(a)** Equatorial (5°S-5°N) SST anomaly composite as a function of longitude and lag,
      for the 2010 window. The black line is the ENSO longitude index (ELI) of the composite.
    - **(b)** Same as (a), for the 2080 window (dashed line: 2080 ELI).
    - **(c)** Niño 3.4 index ($T_{34}$) for the events in (a, b): median over events
      (thick) and 10th/90th percentiles (2010 shaded, 2080 thin lines).
    - **(d)** Change in $T_{34}$ tendency between a 2080 event and a 2010 event, each drawn
      at random from its window: median (thick) and 10th/90th percentiles (thin) of the
      differences over all pairs of events. Gray shading marks a negative median.
    - **(e)** Same as (d), for the net surface heat flux contribution to the $T_{34}$
      tendency ($Q_{34}$, assuming a 50 m mixed layer).

    Anomalies have the ensemble mean removed. In (c-e), they are also measured from the
    median of each calendar month in the window. (a, b) are means over events; (c-e) show
    percentiles of the distribution over events, at each lag.

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX`
    (`Th_anom.nc` by default), SST and net heat flux EOF scores
    (`consolidated/{forced,anom}.nc`), and the ELI (`eli.nc`).

    **Output:** `SAVE_FP/figures/composite-nino.pdf`

    Before this script, (a, b) came from the `spatial_comp` cell of `composite-over-time.py`
    (via a composite cached in `SAVE_FP/composites/`), and (c-e) from the `scalar_comp`
    cell of `spaghetti_updated.py`. Both panels now use the same windows (the scalar panels
    used windows one year later, 2011 and 2081). The old (d, e) took percentiles of the
    differences for one pairing of events (the i-th event in each window); they now use
    all pairs, so the result doesn't depend on the order of the events. The figure has
    two rows, as
    in the outline: (a, b) on top and (c-e) below.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Imports
    """
    )
    return


@app.cell
def _():
    ## colormaps ("cmo.*"), registered with matplotlib on import
    import cmocean  # noqa: F401

    ## plotting and arrays
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    ## labeled arrays (and file paths, for the ELI)
    import os
    import pathlib
    import xarray as xr

    ## this project's modules (shared figure helpers, general utilities)
    import src.figure_utils as fu
    import src.utils

    ## plotting style used for all paper figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return fu, np, os, pathlib, plt, sns, src, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Settings
    """
    )
    return


@app.cell
def _():
    ## Niño 3.4 index used to select events: "sst" (SST-based, used by the RO fits),
    ## or "ml50"/"ml80" (mixed-layer temperature over the top 50/80 m)
    T34_INDEX = "sst"

    ## window length (months) and the windows compared (center years)
    WINDOW_SIZE = 480
    YEARS = [2010, 2080]

    ## events: December Niño 3.4 above this percentile
    PEAK_MONTH = 12
    Q_EVENT = 0.95

    ## mixed-layer depth (m) for converting heat flux to a temperature tendency
    H_ML = 50

    ## percentiles of the distribution over events shown in (c-e): lower, median, upper
    QUANTILES = [0.1, 0.5, 0.9]
    return H_ML, PEAK_MONTH, QUANTILES, Q_EVENT, T34_INDEX, WINDOW_SIZE, YEARS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data
    """
    )
    return


@app.cell
def _(T34_INDEX, fu, os, pathlib, src, xr):
    ## Niño 3.4 anomalies (ensemble mean removed), dims (member, time)
    T34 = fu.load_T34_anom(index=T34_INDEX)

    ## EOF-compressed anomalies (ensemble mean removed): SST and net heat flux
    _, anom = src.utils.load_consolidated()

    ## ENSO longitude index: forced part (ensemble mean) and anomalies
    eli = xr.open_dataset(pathlib.Path(os.environ["DATA_FP"], "cesm", "eli.nc"))[
        "eli_05"
    ]
    eli_forced, eli_anom = src.utils.separate_forced(eli)
    return T34, anom, eli_anom, eli_forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Heat flux contribution to the $T_{34}$ tendency

    $Q_{34} = \frac{\text{NHF}_{34}}{\rho C_p H}$, in K yr$^{-1}$, where $\text{NHF}_{34}$ is
    the net surface heat flux averaged over the Niño 3.4 region and $H$ is `H_ML`.
    """
    )
    return


@app.cell
def _(H_ML, anom, src):
    ## seawater density (kg m^-3), specific heat (J kg^-1 K^-1), seconds per year
    RHO = 1.02e3
    CP = 4.2e3
    SEC_PER_YR = 12 * 30 * 8.64e4

    ## net heat flux (W m^-2) averaged over Niño 3.4, from the EOF scores
    nhf_34 = src.utils.reconstruct_wrapper(
        anom[["nhf", "nhf_comp"]], fn=src.utils.get_nino34
    )["nhf"]

    ## convert to a temperature tendency (K/yr)
    Q34 = nhf_34 * SEC_PER_YR / (RHO * CP * H_ML)
    return (Q34,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Select events

    In each window, events are Decembers when $T_{34}$ exceeds the 95th percentile of
    December values (over all members). Each event's sample runs from lag -12 to +12
    months around that December. The window's first and last years can't hold an event.
    """
    )
    return


@app.cell
def _(PEAK_MONTH, Q_EVENT, src):
    def get_events(window):
        """
        Samples around each strong El Niño in one window.

        Args:
            - window (xr.Dataset): monthly anomalies in one window, dims (member, time),
              including "T_34" (used to select the events)

        Returns:
            - events (xr.Dataset): dims (sample, lag); one sample per event, lags
              -12 to 12 months from the event's December
        """

        ## Decembers above the Q_EVENT percentile (make_composite_helper's default
        ## test is "greater than the cutoff", i.e., warm events)
        events = src.utils.make_composite_helper(
            idx=window["T_34"], data=window, peak_month=PEAK_MONTH, q=Q_EVENT
        )

        return events

    return (get_events,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (a, b) SST composites
    """
    )
    return


@app.cell
def _(
    PEAK_MONTH,
    T34,
    WINDOW_SIZE,
    YEARS,
    anom,
    eli_anom,
    eli_forced,
    fu,
    get_events,
    np,
    src,
    xr,
):
    ## data for the spatial composite: SST scores, Niño 3.4, ELI anomalies
    spatial_data = xr.merge([anom["sst"], T34, eli_anom])

    ## equatorial average (5S-5N), applied after reconstructing the SST field
    def eq_avg(x):
        """
        Meridional mean over the equatorial band.

        Args:
            - x (xr.DataArray): field with a "latitude" dimension

        Returns:
            - avg (xr.DataArray): mean over fu.LATS_EQ
        """
        return x.sel(latitude=slice(*fu.LATS_EQ)).mean("latitude")

    ## composite SST and ELI for each window
    sst_comp = []
    eli_comp = []
    for _year in YEARS:

        ## events in this window, then the mean over events
        _comp = get_events(fu.get_window(spatial_data, _year, WINDOW_SIZE))
        _comp = _comp.mean("sample")

        ## SST field from the composite scores, then the equatorial average
        _sst = src.utils.reconstruct_fn(
            components=anom["sst_comp"], scores=_comp["sst"], fn=lambda x: x
        )
        sst_comp.append(eq_avg(_sst))

        ## forced ELI for the month at each lag (lag 0 is PEAK_MONTH)
        _eli_clim = fu.get_window(eli_forced, _year, WINDOW_SIZE)
        _eli_clim = _eli_clim.groupby("time.month").mean()

        ## calendar month of each lag, then the climatology at those months
        _months = 1 + np.mod(_comp.lag + PEAK_MONTH - 1, 12)
        _eli_clim = _eli_clim.sel(month=_months).drop_vars("month")

        ## total ELI = forced + composite anomaly
        eli_comp.append(_eli_clim + _comp["eli_05"])

    ## stack the windows along a "year" dimension
    sst_comp = xr.concat(sst_comp, dim=xr.DataArray(YEARS, dims="year"))
    eli_comp = xr.concat(eli_comp, dim=xr.DataArray(YEARS, dims="year"))
    return eli_comp, sst_comp


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (c-e) $T_{34}$, its tendency, and $Q_{34}$ for each event

    Anomalies are measured from each calendar month's median in the window. Each event's
    series is smoothed with a 3-month running mean.
    """
    )
    return


@app.cell
def _(Q34, T34, WINDOW_SIZE, YEARS, fu, get_events, xr):
    ## data for the scalar panels
    scalar_data = xr.merge([T34, Q34.rename("Q")])

    ## events in each window
    events = {}
    for _year in YEARS:

        ## measure from the median of each calendar month in the window
        _window = fu.get_window(scalar_data, _year, WINDOW_SIZE)
        _window = _window.groupby("time.month") - _window.groupby("time.month").median()

        ## samples around each event
        _events = get_events(_window)

        ## T34 tendency (K/yr; lags are in months)
        _events["ddt_T"] = 12 * _events["T_34"].differentiate("lag")

        ## 3-month running mean
        events[_year] = _events.rolling({"lag": 3}, center=True).mean()
    return (events,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Percentiles over events, and the distribution of the change

    For (c), take the 10th, 50th, and 90th percentiles of $T_{34}$ over the events in each
    window, at each lag.

    For (d, e), the change is the difference between one event drawn from the late window
    and one drawn independently from the early window. Its distribution is computed
    exactly, from the differences of all pairs of events (one from each window), at each
    lag. This is the limit of repeatedly pairing events at random.
    """
    )
    return


@app.cell
def _(QUANTILES, YEARS, events):
    ## percentiles over events at each lag, for each window
    quantiles = {}
    for _year in YEARS:
        quantiles[_year] = events[_year].quantile(q=QUANTILES, dim="sample")

    ## events in each window, with separate sample dimensions so they broadcast
    early = events[YEARS[0]][["ddt_T", "Q"]].rename(sample="sample_early")
    late = events[YEARS[1]][["ddt_T", "Q"]].rename(sample="sample_late")

    ## differences for all pairs (late minus early), dims (lag, sample_late, sample_early)
    pair_diff = late - early

    ## percentiles of the pairwise differences at each lag
    diff_quantiles = pair_diff.quantile(
        q=QUANTILES, dim=["sample_late", "sample_early"]
    )
    return diff_quantiles, quantiles


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting functions
    """
    )
    return


@app.cell
def _(fu, src):
    def plot_hov(ax, sst, eli, eli_ls, title):
        """
        Panel (a) or (b): SST composite as a function of longitude and lag.

        Args:
            - ax (matplotlib Axes): where to draw
            - sst (xr.DataArray): equatorial SST composite, dims (lag, longitude)
            - eli (xr.DataArray): ELI of the composite, dims (lag)
            - eli_ls (str): line style for the ELI
            - title (str): panel title

        Returns:
            - cp (QuadContourSet): filled contours
        """

        ## filled contours, interval 0.7 K
        cp = ax.contourf(
            sst.longitude,
            sst.lag,
            sst.transpose("lag", "longitude"),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(3.5, 0.7),
            extend="both",
        )

        ## ELI of the composite
        ax.plot(eli, eli.lag, c="k", lw=1, ls=eli_ls)

        ## reference lines: peak (lag 0) and the edges of the Niño 3.4 region
        ref_kwargs = dict(ls="--", c="w", lw=0.75, alpha=0.75)
        ax.axhline(0, **ref_kwargs)
        for lon in fu.LONS_W[1:] + fu.LONS_E[:1]:
            ax.axvline(lon, **ref_kwargs)

        ## axes: longitude ticks at the Niño 3.4 edges (190E = 170W, 240E = 120W)
        ax.set_ylim([-6.5, 6.5])
        ax.set_xlim([140, 285])

        ## longitude ticks and title
        ax.set_xticks([190, 240], labels=[r"170$^{\circ}$W", r"120$^{\circ}$W"])
        ax.set_title(title, loc="left")

        return cp

    def shade_decrease(ax, x):
        """
        Shade between zero and a curve where the curve is negative.

        Args:
            - ax (matplotlib Axes): where to draw
            - x (xr.DataArray): curve, dims (lag)

        Returns:
            - None
        """

        ## curve where negative, zero elsewhere
        x_neg = x.where(x < 0, other=0)
        ax.fill_betweenx(y=x.lag, x1=0, x2=x_neg, color="k", alpha=0.1, lw=0)

        return

    def format_scalar_ax(ax):
        """
        Shared formatting for panels (c-e).

        Args:
            - ax (matplotlib Axes): axes to format

        Returns:
            - None
        """

        ## zero line, and dashed lines at Oct(-1) and Jan(+1)
        ax.axvline(0, c="k", lw=1)
        for lag in [-2, 1]:
            ax.axhline(lag, lw=0.75, c="k", ls="--")

        ## lag range
        ax.set_ylim([-6, 6])

        return

    return format_scalar_ax, plot_hov, shade_decrease


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(
    YEARS,
    eli_comp,
    format_scalar_ax,
    fu,
    plot_hov,
    plt,
    diff_quantiles,
    quantiles,
    shade_decrease,
    sns,
    sst_comp,
):
    ## two rows: SST composites (a, b) on top, scalar panels (c-e) below
    fig = plt.figure(figsize=(5.5, 5.2), layout="constrained")
    fig_top, fig_bottom = fig.subfigures(2, 1)

    ## two panels on top, three below
    axs_l = fig_top.subplots(1, 2)
    axs_r = fig_bottom.subplots(1, 3)

    ## palette colors: early, late, and the difference
    colors = sns.color_palette()
    c_early, c_late, c_diff = colors[0], colors[1], colors[4]

    ## (a, b) SST composites; the 2010 ELI is drawn in both for reference
    eli_early = eli_comp.sel(year=YEARS[0])
    eli_late = eli_comp.sel(year=YEARS[1])

    ## draw the composites
    plot_hov(axs_l[0], sst_comp.sel(year=YEARS[0]), eli_early, "-", f"(a) {YEARS[0]}")
    plot_hov(axs_l[1], sst_comp.sel(year=YEARS[1]), eli_late, "--", f"(b) {YEARS[1]}")
    axs_l[1].plot(eli_early, eli_early.lag, c="k", lw=1)

    ## ELI legend in (a); the late line is drawn behind the contours, for the legend only
    axs_l[0].plot(eli_early, eli_early.lag, c="k", lw=1, label=f"{YEARS[0]} ELI")
    axs_l[0].plot(
        eli_late, eli_late.lag, c="k", lw=1, ls="--", zorder=0, label=f"{YEARS[1]} ELI"
    )

    ## legend on a white background
    legend = axs_l[0].legend(prop=dict(size=10), loc=(-0.35, 0.62))
    legend.get_frame().set_color("white")

    ## lag labels on (a) only
    axs_l[0].set_yticks([-6, 0, 6], labels=["Jun(-1)", "Dec(0)", "Jun(+1)"])
    axs_l[1].set_yticks([])

    ## contour interval, in place of a colorbar
    axs_l[0].text(
        x=0.43,
        y=0.05,
        s="Interval: 0.7 K",
        transform=axs_l[0].transAxes,
        fontdict=dict(size=10, color="w"),
    )

    ## (c) T34: median over events (thick); lower/upper percentiles (first/last quantile)
    for year, c in zip(YEARS, [c_early, c_late]):
        T = quantiles[year]["T_34"]
        axs_r[0].plot(T.sel(quantile=0.5), T.lag, lw=4, c=c)
        lower, upper = T.isel(quantile=0), T.isel(quantile=-1)

        ## early window shaded; late window as thin lines
        if year == YEARS[0]:
            axs_r[0].fill_betweenx(y=T.lag, x1=lower, x2=upper, color=c, alpha=0.1)
        else:
            axs_r[0].plot(lower, T.lag, lw=1, c=c)
            axs_r[0].plot(upper, T.lag, lw=1, c=c)

    ## (c) labels for the two windows, and axes
    axs_r[0].text(s=YEARS[0], color=c_early, x=3, y=-5)
    axs_r[0].text(s=YEARS[1], color=c_late, x=0.5, y=-1)

    ## (c) axes and title
    axs_r[0].set_xlabel("K")
    axs_r[0].set_xticks([0, 3])
    axs_r[0].set_xlim([-0.5, None])

    ## lag labels on (c), marking the onset (Oct) and decay (Jan) months
    axs_r[0].set_yticks(
        [-6, -2, 1, 6], labels=["Jun(-1)", "Oct(-1)", "Jan(+1)", "Jun(+1)"]
    )
    axs_r[0].set_title("(c) $T_{34}$", loc="left")

    ## (d, e) percentiles of the pairwise differences: median thick, lower/upper thin
    for ax, v in zip(axs_r[1:], ["ddt_T", "Q"]):
        diff = diff_quantiles[v]
        for q in diff["quantile"].values:
            lw = 4 if q == 0.5 else 1
            ax.plot(diff.sel(quantile=q), diff.lag, lw=lw, c=c_diff)

        ## shade where the median difference is negative (faster decay / more damping)
        shade_decrease(ax, diff.sel(quantile=0.5))

        ## axes
        ax.set_yticks([])
        ax.set_xlabel("K yr$^{-1}$")

        ## shared x range for (d, e)
        ax.set_xticks([-4, 0])
        ax.set_xlim([-7.2, 3])

    ## (d, e) titles
    axs_r[1].set_title(r"(d) $\Delta\left(\frac{d T_{34}}{dt}\right)$", loc="left")
    axs_r[2].set_title(r"(e) $\Delta ~Q_{34}$", loc="left")

    ## shared formatting for (c-e)
    for ax in axs_r:
        format_scalar_ax(ax)

    ## save to SAVE_FP/figures
    fu.save_fig(fig, "composite-nino")

    fig
    return


if __name__ == "__main__":
    app.run()
