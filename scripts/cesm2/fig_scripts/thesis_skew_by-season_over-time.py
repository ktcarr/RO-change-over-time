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
    # Seasonality of SST skewness over time (thesis)

    Thesis appendix figure `fig:skew_by-season_over-time` (`appendix_enso-change.tex`).
    Skewness of SST anomalies (ensemble mean removed) for each calendar month (x-axis)
    and year (y-axis), averaged over **(a)** Niño 4, **(b)** Niño 3.4, and **(c)** Niño
    3. The skewness for a given year and month pools all 100 members and that month in
    the 41 years centered on the year (e.g., 2010 uses 1990-2030), so the years run from
    1870 to 2080. Dashed white lines mark May and 2010.

    **Inputs:** `DATA_FP/cesm/Th_anom.nc` (SST-based Niño indices).

    **Output:** `SAVE_FP/figures/thesis_skew_by-season_over-time.pdf`

    Before this script, the figure was `skew_by-season_over-time`, made by
    `explore/CESM_climate-change.py` (cells under "Nicer plot, for thesis"). Changes:

    - The month ticks sit on January and December (before, at 1.6 and 11.4, so the
      labels wouldn't run past the panel edges; here the edge labels are aligned
      inward instead).
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
    ## plotting and colormaps
    import cmocean  # noqa: F401 (registers the "cmo.*" colormaps)
    import matplotlib.pyplot as plt

    ## statistics (skewness), plot style
    import scipy.stats
    import seaborn as sns

    ## this project's modules: loading data, plotting
    import src.data
    import src.plot

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return plt, scipy, src


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
    ## Niño indices (panels) and their titles
    INDICES = {"T_4": "(a) Niño 4", "T_34": "(b) Niño 3.4", "T_3": "(c) Niño 3"}

    ## half-width of the window (years): each year uses the 2n+1 = 41 years around it
    HALF_WINDOW = 20

    ## largest skewness shown in the colors
    AMP = 1.5

    ## year and month marked by dashed lines; ticks on the time axis
    YEAR_LINE = 2010
    MONTH_LINE = 5
    YTICKS = [1870, 2010, 2080]
    return AMP, HALF_WINDOW, INDICES, MONTH_LINE, YEAR_LINE, YTICKS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data
    """
    )
    return


@app.cell
def _(src):
    ## Niño 4, 3.4, and 3 anomalies (ensemble mean removed); dims (member, time)
    nino = src.data.load_nino_anom(index="sst")
    return (nino,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Skewness by month and year

    For each calendar month separately: the skewness over a 41-year window centered on
    each year, pooling members. Years whose window runs past the simulation (the first
    and last 20) are dropped.
    """
    )
    return


@app.cell
def _(HALF_WINDOW, nino, scipy, src):
    ## rolling skewness for each calendar month; dims (time,)
    _skew = src.utils.get_rolling_fn_bymonth(nino, fn=scipy.stats.skew, n=HALF_WINDOW)

    ## split time into year and month; dims (year, month)
    skew = src.utils.unstack_month_and_year(_skew).transpose("year", "month")
    return (skew,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(AMP, INDICES, MONTH_LINE, YEAR_LINE, YTICKS, plt, skew, src):
    ## one panel per Niño region
    fig, axs = plt.subplots(1, 3, figsize=(7, 3), layout="constrained")

    ## color levels and dashed reference lines
    _levels = src.utils.make_cb_range(AMP, AMP / 10)
    _line_kwargs = dict(ls="--", c="w", lw=1, alpha=0.5)

    for _ax, (_name, _title) in zip(axs, INDICES.items()):

        ## skewness, by month and year
        _cp = _ax.contourf(
            skew.month,
            skew.year,
            skew[_name],
            cmap="cmo.balance",
            levels=_levels,
            extend="both",
        )

        ## reference lines and title
        _ax.axhline(YEAR_LINE, **_line_kwargs)
        _ax.axvline(MONTH_LINE, **_line_kwargs)
        _ax.set_title(_title, loc="left")

        ## month axis; the edge labels are aligned inward so they stay in the panel
        _ax.set_xticks([1, 5, 12], labels=["Jan", "May", "Dec"])
        _ax.get_xticklabels()[0].set_horizontalalignment("left")
        _ax.get_xticklabels()[-1].set_horizontalalignment("right")

        ## axis label
        _ax.set_xlabel("Month")

        ## time ticks are added to the first panel below
        _ax.set_yticks([])

    ## colorbar, and time ticks only on the first panel
    fig.colorbar(_cp, ax=axs[-1], ticks=[-AMP, 0, AMP], label="Skew (non-dim.)")
    axs[0].set_yticks(YTICKS)

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "thesis_skew_by-season_over-time")

    fig
    return


if __name__ == "__main__":
    app.run()
