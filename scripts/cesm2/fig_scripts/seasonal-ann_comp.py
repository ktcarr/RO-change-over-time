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
    # Growth rate and heat flux damping of warm and cold anomalies over time

    Paper figure `fig:seasonal-ann_comp` (`paper-outline.tex` in the ch3 Overleaf
    project). Statistics of the Niño 3.4 index ($T_{34}$) in 40-year windows every 10
    years, computed separately for each ensemble member; lines show the median (thick)
    and 10th and 90th percentiles (thin) across members, for warm ($T_{34}>0$, red),
    cold ($T_{34}<0$, blue), and all anomalies (black, median only).

    - **(a, d) Growth rate** $R$ of $T_{34}$, from the lag regression of $T_{34}$ on its
      earlier value: if $T_{34}(t+\tau) = S\,T_{34}(t)$, then $R = \log(S)/\tau$
    - **(b, e) Heat flux contribution:** slope of the regression of $Q_{34}$ (net surface
      heat flux over Niño 3.4 as a temperature tendency, assuming a 50 m mixed layer)
      on $T_{34}$
    - **(c, f) Residual:** (a) minus (b), or (d) minus (e), the growth rate from everything but the surface
      heat flux
    - **Top row (a-c):** boreal winter. $R$ from November to the following February
      ($\tau$ = 3 months); (b) from November-January.
    - **Bottom row (d-f):** all months. $R$ from each month to the next ($\tau$ = 1 month)
      and (b) for each calendar month, then averaged over the 12 months.

    Warm and cold are chosen by the sign of $T_{34}$ at the starting time (a, d) or at the
    same time (b, e).

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX`
    (`Th_anom.nc` by default) and net heat flux EOF scores (`consolidated/anom.nc`).

    **Output:** `SAVE_FP/figures/seasonal-ann_comp.pdf`

    Before this script, the two rows came from the "Get $R$ over time" cells of
    `explore/spaghetti_updated.py`, saved as separate figures and joined by hand. Changes:

    - The windows are centered on 1870, 1880, ..., 2080. The old script dropped 1850
      (for NaNs in heat content indices this figure doesn't use), so its windows were
      centered on 1871, ..., 2081.
    - The y-axis limits are the same in both rows, and the legend appears once.
    - The bottom row's panels are labeled (d)-(f); both rows used to be (a)-(c).
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
    ## arrays, plotting, and plotting style
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    ## this project's modules (shared figure helpers, general utilities)
    import src.figure_utils as fu
    import src.utils

    ## plotting style used for all paper figures; color_codes makes "r" and "b"
    ## the colorblind palette's red and blue
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return fu, np, plt, src


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Settings
    """
    )
    return


@app.cell
def _(np):
    ## Niño 3.4 index: "sst" (SST-based, used by the RO fits), or "ml50"/"ml80"
    ## (mixed-layer temperature over the top 50/80 m)
    T34_INDEX = "sst"

    ## window length and spacing (months), and the centers kept: every 10 years,
    ## while the window fits in the simulation (1850-2100)
    WINDOW_SIZE = 480
    STRIDE = 120
    YEARS = np.arange(1870, 2081, 10)

    ## mixed-layer depth (m) for converting heat flux to a temperature tendency
    H_ML = 50

    ## winter: growth rate from November to February (3 months later), and heat
    ## flux regression over November-January
    MONTH_R_WINTER = 11
    LAG_R_WINTER = 3
    MONTHS_Q_WINTER = [11, 12, 1]
    return (
        H_ML,
        LAG_R_WINTER,
        MONTHS_Q_WINTER,
        MONTH_R_WINTER,
        STRIDE,
        T34_INDEX,
        WINDOW_SIZE,
        YEARS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data
    """
    )
    return


@app.cell
def _(H_ML, STRIDE, T34_INDEX, WINDOW_SIZE, YEARS, fu, src):
    ## Niño 3.4 anomalies (ensemble mean removed), K, dims (member, time)
    T34 = fu.load_T34_anom(index=T34_INDEX)

    ## net heat flux over Niño 3.4 as a temperature tendency (K/yr), on T34's months
    Q34 = fu.load_Q34(H=H_ML).sel(time=T34.time)

    ## cut both into windows, dims (member, year, time); keep full windows only
    _kwargs = dict(window_size=WINDOW_SIZE, stride=STRIDE)
    T34_win = src.utils.get_windowed(T34, **_kwargs).sel(year=YEARS)
    Q34_win = src.utils.get_windowed(Q34, **_kwargs).sel(year=YEARS)
    return Q34_win, T34_win


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Growth rate and heat flux contribution

    Each quantity has variables "pos" ($T_{34}>0$), "neg" ($T_{34}<0$), and "all",
    with dims (member, year), in yr$^{-1}$.
    """
    )
    return


@app.cell
def _(
    LAG_R_WINTER,
    MONTHS_Q_WINTER,
    MONTH_R_WINTER,
    Q34_win,
    T34_win,
    fu,
    src,
):
    ## winter growth rate: November to February
    R_winter = fu.get_lag_growth_rate(T34_win, lag=LAG_R_WINTER, month=MONTH_R_WINTER)

    ## winter heat flux contribution: one regression over November-January
    Q_winter = fu.regress_by_sign(
        src.utils.sel_month(T34_win, months=MONTHS_Q_WINTER),
        src.utils.sel_month(Q34_win, months=MONTHS_Q_WINTER),
    )
    return Q_winter, R_winter


@app.cell
def _(Q34_win, T34_win, fu):
    ## all months: month-to-month growth rate for each starting month, averaged
    R_annual = fu.get_lag_growth_rate(T34_win, lag=1, by_month=True).mean("month")

    ## heat flux contribution for each calendar month, averaged
    Q_annual = fu.regress_by_sign(T34_win, Q34_win, by_month=True).mean("month")
    return Q_annual, R_annual


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _():
    def plot_by_sign(ax, coefs):
        """
        Plot the member median and 10th-90th percentiles of a quantity over time,
        for warm (red), cold (blue), and all (black, median only) anomalies.

        Args:
            - ax (matplotlib.axes.Axes): axis to plot on
            - coefs (xr.Dataset): "pos", "neg", "all", dims (member, year)

        Returns:
            - None
        """

        ## warm and cold: median (thick) and 10th/90th percentiles (thin)
        for name, color, label in [
            ("pos", "r", r"$T_{34}>0$"),
            ("neg", "b", r"$T_{34}<0$"),
        ]:
            quantiles = coefs[name].quantile([0.1, 0.5, 0.9], dim="member")

            ## median (labeled for the legend), then the percentiles
            ax.plot(coefs.year, quantiles.sel(quantile=0.5), c=color, lw=2, label=label)
            ax.plot(coefs.year, quantiles.sel(quantile=0.1), c=color, lw=0.6)
            ax.plot(coefs.year, quantiles.sel(quantile=0.9), c=color, lw=0.6)

        ## all anomalies: median only
        ax.plot(coefs.year, coefs["all"].median("member"), c="k", lw=2, label="all")

        ## zero line and the 2010 window
        line_kwargs = dict(ls="--", c="gray", lw=0.8)
        ax.axvline(2010, **line_kwargs)
        ax.axhline(0, **line_kwargs)

        ## label the first window and 2010
        ax.set_xticks([1870, 2010])

        return

    return (plot_by_sign,)


@app.cell
def _(Q_annual, Q_winter, R_annual, R_winter, fu, np, plot_by_sign, plt):
    ## one row per season: growth rate, heat flux contribution, residual
    fig, axs = plt.subplots(2, 3, figsize=(8.5, 5.5), layout="constrained")

    ## rows: (label, growth rate, heat flux contribution, panel letters)
    _rows = [
        ("Nov-Jan", R_winter, Q_winter, "abc"),
        ("All months", R_annual, Q_annual, "def"),
    ]

    ## y-axis limits: (a) and (b) share them; (c) is shifted up
    _ylim = np.array([-4, 1.5])
    _yticks = np.array([-3, 0, 1])

    ## draw each row
    for _axs, (_label, _R, _Q, _p) in zip(axs, _rows):

        ## (a) growth rate, (b) heat flux contribution, (c) residual
        plot_by_sign(_axs[0], _R)
        plot_by_sign(_axs[1], _Q)
        plot_by_sign(_axs[2], _R - _Q)

        ## titles
        _axs[0].set_title(f"{_p[0]}) $T_{{34}}$ growth rate", loc="left")
        _axs[1].set_title(f"{_p[1]}) $Q_{{34}}$ contribution", loc="left")
        _axs[2].set_title(f"{_p[2]}) Residual ({_p[0]}-{_p[1]})", loc="left")

        ## (a) and (b) share limits; (b) has no tick labels
        for _ax in _axs[:2]:
            _ax.set_ylim(_ylim)
        _axs[0].set_yticks(_yticks)
        _axs[1].set_yticks([])

        ## (c): mirror image of (a)'s limits and ticks
        _axs[2].set_ylim(-_ylim[::-1])
        _axs[2].set_yticks(-_yticks[::-1])

        ## units, and the row label to their left
        _axs[0].set_ylabel(r"yr$^{-1}$")
        _axs[0].text(-0.4, 0.5, _label, transform=_axs[0].transAxes, ha="right")

    ## legend in the top-left panel
    axs[0, 0].legend(frameon=False, prop=dict(size=10), loc=(-0.03, 0.12))

    ## save to SAVE_FP/figures
    fu.save_fig(fig, "seasonal-ann_comp")

    fig
    return


if __name__ == "__main__":
    app.run()
