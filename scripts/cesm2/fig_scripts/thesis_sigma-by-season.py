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
    # Seasonal changes in equatorial SST variability (thesis)

    Thesis figure `fig:sigma-by-season` (`chapter_enso-change.tex`). Standard deviation
    of SST along the equator by month and longitude, for two 40-year windows:

    - **(a)** the first window (1850-1889, labeled 1870); the dashed black line is the
      -0.5 contour of SST skewness
    - **(b)** a late window (2060-2099, labeled 2080); the solid black line is the -0.5
      skewness contour
    - **(c)** the difference, (b) minus (a), with both skewness contours

    The standard deviation and skewness pool all 100 members and the window's years, for
    each month. The standard deviation is computed at each grid point from the anomalies
    (from the ensemble mean), then averaged over 5°S-5°N; the skewness is that of the
    5°S-5°N average. Dashed white lines mark the Niño 3 longitudes (150°W-90°W).

    `thesis_sigma-over-time.py` (`fig:sigma-over-time`) shows the change over time, by
    season.

    **Inputs:** `DATA_FP/cesm/consolidated/anom.nc` (SST).

    **Output:** `SAVE_FP/figures/thesis_sigma-by-season.pdf`

    Before this script, the figure was `sigma-cycle-change`, made by
    `explore/mean-state-changes.py` (cell under "Variance over time (v2)"). Changes:

    - The windows are selected by their years in `WINDOWS` (before, by their centers);
      the thesis caption says 41-year windows, but they are 40 years (480 months).
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

    ## numerics, plot style, labeled arrays
    import numpy as np
    import seaborn as sns
    import xarray as xr

    ## this project's modules: loading data, plotting, statistics
    import src.data
    import src.plot
    import src.stats

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return np, plt, src, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Settings
    """
    )
    return


@app.cell
def _(np, src):
    ## the two 40-year windows (labels are their centers)
    WINDOWS = {1870: slice("1850", "1889"), 2080: slice("2060", "2099")}

    ## skewness contour drawn on each panel
    SKEW_LEVEL = -0.5

    ## color levels (°C): standard deviation, and the difference
    LEVELS = np.arange(0.4, 2.2, 0.2)
    LEVELS_DIFF = src.utils.make_cb_range(0.5, 0.05)

    ## longitudes shown (degrees east), and the Niño 3 longitudes marked by lines
    XLIM = [140, 280]
    XTICKS = [140, 210, 270]
    LON_LINES = [210, 270]
    return LEVELS, LEVELS_DIFF, LON_LINES, SKEW_LEVEL, WINDOWS, XLIM, XTICKS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data

    SST anomalies (from the ensemble mean), EOF-compressed: patterns and scores.
    """
    )
    return


@app.cell
def _(src):
    ## anomalies from the ensemble mean (the ensemble mean itself isn't needed)
    _, _anom = src.data.load_eofs(["sst"])

    ## patterns (mode, latitude, longitude) and scores (member, time, mode)
    _components, _scores = src.utils.split_components(_anom)
    components = _components["sst"]
    scores = _scores["sst"]
    return components, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Standard deviation and skewness by month

    For each window and month: the standard deviation at each grid point near the
    equator (from the scores' covariance), averaged over 5°S-5°N; and the skewness of
    the 5°S-5°N average.
    """
    )
    return


@app.cell
def _(WINDOWS, components, scores, src, xr):
    ## statistics in each window; dims (month, longitude)
    stats = dict()
    for _label, _years in WINDOWS.items():
        _window = scores.sel(time=_years).compute()
        stats[_label] = xr.Dataset(
            dict(
                sigma=src.stats.get_eq_sigma(components, _window, "month"),
                skew=src.stats.get_eq_skew(components, _window, "month"),
            )
        )
    return (stats,)


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
    LEVELS,
    LEVELS_DIFF,
    LON_LINES,
    SKEW_LEVEL,
    XLIM,
    XTICKS,
    plt,
    src,
    stats,
):
    ## early and late windows, and the difference
    early, late = stats[1870], stats[2080]
    _panels = [
        ("(a) 1870", early["sigma"], LEVELS, "cmo.amp", "min"),
        ("(b) 2080", late["sigma"], LEVELS, "cmo.amp", "min"),
        (
            "(c) Difference",
            late["sigma"] - early["sigma"],
            LEVELS_DIFF,
            "cmo.balance",
            "both",
        ),
    ]

    ## one panel each
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

    ## standard deviation (colors), by longitude and month (Jan repeated at the top)
    _cps = []
    for _ax, (_title, _data, _levels, _cmap, _extend) in zip(axs, _panels):
        _values, _months = src.plot.add_cyclic_month(_data)
        _cps.append(
            _ax.contourf(
                _data.longitude,
                _months,
                _values,
                levels=_levels,
                cmap=_cmap,
                extend=_extend,
            )
        )
        _ax.set_title(_title, loc="left")

    ## skewness contours: early (dashed) on (a) and (c), late (solid) on (b) and (c)
    _contour_kwargs = dict(levels=[SKEW_LEVEL], colors="k", linewidths=0.8, alpha=0.8)
    for _stats, _ax, _ls in [(early, axs[0], "--"), (late, axs[1], "-")]:
        _values, _months = src.plot.add_cyclic_month(_stats["skew"])
        for _ax_skew in [_ax, axs[2]]:
            _ax_skew.contour(
                _stats.longitude, _months, _values, linestyles=_ls, **_contour_kwargs
            )

    ## Niño 3 longitudes, and the longitude axis
    for _ax in axs:
        for _lon in LON_LINES:
            _ax.axvline(_lon, ls="--", c="w", lw=0.85)
        _ax.set_xlim(XLIM)
        _ax.set_xticks(XTICKS)

    ## month axis only on the first panel
    axs[0].set_yticks([1, 5, 9, 13], labels=["Jan", "May", "Sep", "Jan"])
    axs[0].set_ylabel("Month")
    for _ax in axs[1:]:
        _ax.set_yticks([])

    ## longitude label under the middle panel
    axs[1].set_xlabel("Longitude")

    ## colorbars: (a) and (b) share theirs
    _cb_label = r"$^{\circ}\text{C}$"
    fig.colorbar(_cps[1], ax=axs[1], ticks=[0.4, 1.2, 2], label=_cb_label)
    fig.colorbar(_cps[2], ax=axs[2], ticks=[-0.5, 0, 0.5], label=_cb_label)

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "thesis_sigma-by-season")

    fig
    return


if __name__ == "__main__":
    app.run()
