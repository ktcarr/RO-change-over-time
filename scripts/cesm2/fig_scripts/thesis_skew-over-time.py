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
    # Niño 3.4 statistics over time (thesis)

    Thesis appendix figure `fig:skew-over-time` (`appendix_enso-change.tex`). Statistics
    of DJF-mean Niño 3.4 anomalies (ensemble mean removed), computed for each member in
    each 40-year window (39 full DJF seasons):

    - **(a)** median
    - **(b)** standard deviation $\sigma$
    - **(c)** skewness, $m_3 / \sigma^3$
    - **(d)** third central moment $m_3$

    Thick lines: ensemble median; thin lines: 10th and 90th percentiles over members.
    Windows are 40 years (480 months), every 5 years, labeled by their centers
    (1870-2080); the dashed gray vertical line marks 2010.

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX`:
    `Th_anom.nc` (SST, the default), or `Th_50.nc` / `Th_80.nc` (mixed-layer
    temperature).

    **Output:** `SAVE_FP/figures/thesis_skew-over-time.pdf`

    Before this script, the figure was `nino34_djf_stats-over-time`, made by
    `explore/stats-changes.py` (cells under "Variance and skewness over time").
    Changes:

    - The index is set by `T34_INDEX`, the SST-based index by default. The thesis
      figure was made with the SST-based index (the old code reproduces it with that
      index), but the notebook's committed code had since been switched to the top-80 m
      mixed-layer temperature, as for `thesis_pdf-over-time.py`.
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
    ## plotting
    import matplotlib.pyplot as plt

    ## statistics (skewness), plot style, labeled arrays
    import scipy.stats
    import seaborn as sns
    import xarray as xr

    ## this project's modules: loading data, plotting
    import src.data
    import src.plot

    ## plotting style used for all figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return plt, scipy, src, xr


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
    ## Niño 3.4 index: "sst" (SST-based, used by the RO fits), or "ml50"/"ml80"
    ## (mixed-layer temperature over the top 50/80 m)
    T34_INDEX = "sst"

    ## window length and stride (months)
    WINDOW_SIZE = 480
    STRIDE = 60

    ## percentiles over members: median (thick), and bounds (thin)
    MEMBER_QUANTILES = [0.5, 0.1, 0.9]

    ## year marked with a line; ticks on the time axis
    YEAR_LINE = 2010
    XTICKS = [1870, 2010, 2080]
    return MEMBER_QUANTILES, STRIDE, T34_INDEX, WINDOW_SIZE, XTICKS, YEAR_LINE


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data
    """
    )
    return


@app.cell
def _(STRIDE, T34_INDEX, WINDOW_SIZE, src):
    ## Niño 3.4 anomalies (ensemble mean removed), cut into windows
    _T34_windowed = src.data.load_T34_windowed(
        index=T34_INDEX, window_size=WINDOW_SIZE, stride=STRIDE
    )

    ## keep only full windows (the last one runs past the end of the simulation)
    _T34_windowed = _T34_windowed.dropna("year", how="any")

    ## DJF means (full seasons only); dims (member, year, time)
    T = src.data.get_djf(_T34_windowed)
    return (T,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Statistics in each window

    For each member and window, over the 39 DJF values; then percentiles over members.
    """
    )
    return


@app.cell
def _(MEMBER_QUANTILES, T, scipy, xr):
    ## median and standard deviation (ddof=0, as np.std); dims (member, year)
    stats = xr.Dataset()
    stats["median"] = T.median("time")
    stats["std"] = T.std("time")

    ## skewness (biased estimator, scipy's default)
    stats["skew"] = xr.apply_ufunc(
        scipy.stats.skew, T, input_core_dims=[["time"]], kwargs={"axis": -1}
    )

    ## third central moment, from skewness and standard deviation
    stats["m3"] = stats["skew"] * stats["std"] ** 3

    ## percentiles over members; dims (quantile, year)
    stats_q = stats.quantile(q=MEMBER_QUANTILES, dim="member")
    return (stats_q,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(MEMBER_QUANTILES, XTICKS, YEAR_LINE, plt, src, stats_q):
    ## each panel: statistic, title, y-label, whether to draw a zero line
    _panels = [
        ("median", r"(a) Median", r"$^{\circ}$C", True),
        ("std", r"(b) Standard deviation ($\sigma$)", r"$^{\circ}$C", False),
        ("skew", r"(c) Skewness ($m_3 / \sigma^3$)", "non-dimensional", True),
        ("m3", r"(d) $m_3$", r"$^{\circ}$C$^3$", True),
    ]

    ## 2x2 panels
    fig, axs = plt.subplots(2, 2, figsize=(6, 5), layout="constrained")

    for _ax, (_name, _title, _ylabel, _zero) in zip(axs.flatten(), _panels):

        ## median over members (thick) and bounds (thin)
        for _q in MEMBER_QUANTILES:
            _lw = 2.5 if _q == 0.5 else 1
            _line = stats_q[_name].sel(quantile=_q)
            _ax.plot(_line.year, _line, lw=_lw, c="k")

        ## reference lines
        _ax.axvline(YEAR_LINE, ls="--", lw=0.8, c="gray")
        if _zero:
            _ax.axhline(0, ls="--", lw=0.8, c="gray")

        ## labels; time ticks are added to the bottom row below
        _ax.set_title(_title, loc="left")
        _ax.set_ylabel(_ylabel)
        _ax.set_xticks([])

    ## time axis only on the bottom row
    for _ax in axs[1]:
        _ax.set_xticks(XTICKS)
        _ax.set_xlabel("Year")

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "thesis_skew-over-time")

    fig
    return


if __name__ == "__main__":
    app.run()
