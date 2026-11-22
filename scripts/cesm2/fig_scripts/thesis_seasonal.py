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
    # Surface mean state changes by season (thesis)

    Thesis figure `fig:seasonal` (`chapter_enso-change.tex`). Ensemble-mean climatology
    along the equator (averaged over 5°S-5°N), by month and longitude, for two 41-year
    windows:

    - **(a)** relative SST (SST minus the tropical-mean SST over 15°S-15°N, for the same
      month) in the first window (1850-1890, labeled 1870)
    - **(b)** same, in a late window (2060-2100, labeled 2080)
    - **(c)** the difference, (b) minus (a)
    - **(d-f)** same as (a-c), for precipitation

    Dashed white lines mark the Niño 3 longitudes (150°W-90°W).

    **Inputs** (under `DATA_FP/cesm`): `consolidated/forced.nc` (SST, precipitation) and
    `trop_sst.nc` (tropical-mean SST).

    **Output:** `SAVE_FP/figures/thesis_seasonal.pdf`

    Before this script, the figure was `seasonal-clim`, made by
    `explore/CESM_climate-change.py` (cell under "updated, combined fig"). Changes:

    - Precipitation is converted to mm/day with 86400 s/day (before, 8.6e4, so values
      were 0.5% low).
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
    ## standard library: paths, environment, partial functions
    import functools
    import os
    import pathlib

    ## plotting and colormaps
    import cmocean  # noqa: F401 (registers the "cmo.*" colormaps)
    import matplotlib.pyplot as plt

    ## numerics, plot style, labeled arrays
    import numpy as np
    import seaborn as sns
    import xarray as xr

    ## this project's modules: loading data, indices, plotting
    import src.data
    import src.indices
    import src.plot

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return functools, np, os, pathlib, plt, src, xr


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
    ## the two 41-year windows (labels are their centers)
    WINDOWS = {1870: slice("1850", "1890"), 2080: slice("2060", "2100")}

    ## tropical-mean SST subtracted to get relative SST (15S-15N)
    TROP_SST = "trop_sst_15"

    ## longitudes averaged along the equator (degrees east)
    LONS = (140, 285)

    ## seconds per day, to convert precipitation from kg/m^2/s to mm/day
    SEC_PER_DAY = 8.64e4

    ## color levels for relative SST (°C), and its change
    LEVELS_SST = src.utils.make_cb_range(4, 0.4)
    LEVELS_SST_DIFF = src.utils.make_cb_range(2, 0.2)

    ## color levels for precipitation (mm/day), and its change
    LEVELS_PR = np.arange(0, 13, 1)
    LEVELS_PR_DIFF = src.utils.make_cb_range(6, 0.6)

    ## longitudes shown, their ticks, and the Niño 3 longitudes marked by lines
    XLIM = [140, 280]
    XTICKS = [140, 210, 270]
    LON_LINES = [210, 270]
    return (
        LEVELS_PR,
        LEVELS_PR_DIFF,
        LEVELS_SST,
        LEVELS_SST_DIFF,
        LONS,
        LON_LINES,
        SEC_PER_DAY,
        TROP_SST,
        WINDOWS,
        XLIM,
        XTICKS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data

    The ensemble mean of SST and precipitation, EOF-compressed (scores and patterns);
    and the tropical-mean SST for each member.
    """
    )
    return


@app.cell
def _(TROP_SST, os, pathlib, src, xr):
    ## ensemble mean (the anomalies from it aren't needed)
    _forced, _ = src.data.load_eofs(["sst", "pr"])

    ## patterns (mode, latitude, longitude) and scores (time, mode)
    components, scores = src.utils.split_components(_forced)

    ## tropical-mean SST, ensemble mean; dims (time,)
    _fp = pathlib.Path(os.environ["DATA_FP"], "cesm", "trop_sst.nc")
    trop_sst = xr.open_dataset(_fp)[TROP_SST].mean("member")
    return components, scores, trop_sst


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Climatology in each window

    Averaging the scores by month and then reconstructing gives the monthly
    climatology of the fields, here averaged over 5°S-5°N. Relative SST subtracts the
    tropical-mean SST's climatology over the same window.
    """
    )
    return


@app.cell
def _(LONS, SEC_PER_DAY, WINDOWS, components, functools, scores, src, trop_sst):
    ## equatorial profile (average over latitude) at each longitude
    _fn = functools.partial(src.indices.get_eq_profile, lons=LONS)

    ## monthly climatology in each window; dims (month, longitude)
    clim = dict()
    for _label, _years in WINDOWS.items():
        _scores_clim = scores.sel(time=_years).groupby("time.month").mean()
        _clim = src.utils.reconstruct_fn(
            components=components, scores=_scores_clim, fn=_fn
        )

        ## relative SST: minus the tropical mean for the same month
        _trop_clim = trop_sst.sel(time=_years).groupby("time.month").mean()
        _clim["sst_rel"] = _clim["sst"] - _trop_clim

        ## precipitation in mm/day
        _clim["pr"] = SEC_PER_DAY * _clim["pr"]
        clim[_label] = _clim
    return (clim,)


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
    LEVELS_PR,
    LEVELS_PR_DIFF,
    LEVELS_SST,
    LEVELS_SST_DIFF,
    LON_LINES,
    XLIM,
    XTICKS,
    clim,
    np,
    plt,
    src,
):
    ## early and late windows, and their difference
    early, late = clim[1870], clim[2080]
    diff = late - early

    ## each panel: data, levels, colormap, color extension (rows: SST, precipitation)
    _panels = np.array(
        [
            [
                (early["sst_rel"], LEVELS_SST, "cmo.balance", "both"),
                (late["sst_rel"], LEVELS_SST, "cmo.balance", "both"),
                (diff["sst_rel"], LEVELS_SST_DIFF, "cmo.balance", "both"),
            ],
            [
                (early["pr"], LEVELS_PR, "cmo.rain", "max"),
                (late["pr"], LEVELS_PR, "cmo.rain", "max"),
                (diff["pr"], LEVELS_PR_DIFF, "cmo.balance_r", "both"),
            ],
        ],
        dtype=object,
    )

    ## two rows (relative SST, precipitation), three columns (early, late, difference)
    fig, axs = plt.subplots(2, 3, figsize=(9, 5), layout="constrained")

    ## filled contours, by longitude and month (January repeated at the top)
    cps = np.empty(axs.shape, dtype=object)
    for _idx in np.ndindex(axs.shape):
        _data, _levels, _cmap, _extend = _panels[_idx]
        _values, _months = src.plot.add_cyclic_month(_data)
        cps[_idx] = axs[_idx].contourf(
            _data.longitude,
            _months,
            _values,
            levels=_levels,
            cmap=_cmap,
            extend=_extend,
        )

    ## Niño 3 longitudes, and the longitude range, on every panel
    for _ax in axs.flatten():
        for _lon in LON_LINES:
            _ax.axvline(_lon, ls="--", c="w", lw=0.85)
        _ax.set_xlim(XLIM)

    ## panel letters in the lower-left corner (black where the background is light)
    _colors = ["w", "w", "w", "w", "w", "k"]
    for _ax, _letter, _c in zip(axs.flatten(), "abcdef", _colors):
        _ax.text(0.02, 0.05, f"({_letter})", c=_c, transform=_ax.transAxes)

    ## column titles
    for _ax, _title in zip(axs[0], ["1870", "2080", "Difference"]):
        _ax.set_title(_title)

    ## month axis only on the left column
    for _ax in axs[:, 0]:
        _ax.set_yticks([1, 5, 9, 13], labels=["Jan", "May", "Sep", "Jan"])
        _ax.set_ylabel("Month")
    for _ax in axs[:, 1:].flatten():
        _ax.set_yticks([])

    ## longitude axis only on the bottom row
    for _ax in axs[0]:
        _ax.set_xticks([])
    for _ax in axs[1]:
        _ax.set_xticks(XTICKS)
        _ax.set_xlabel("Longitude")

    ## colorbars: the early and late panels of each row share theirs
    _sst_label = r"$^{\circ}\text{C}$"
    fig.colorbar(cps[0, 1], ax=axs[0, 1], ticks=[-4, 0, 4])
    fig.colorbar(cps[0, 2], ax=axs[0, 2], ticks=[-2, 0, 2], label=_sst_label)

    ## precipitation colorbars
    _pr_label = r"mm day$^{-1}$"
    fig.colorbar(cps[1, 1], ax=axs[1, 1], ticks=[0, 6, 12])
    fig.colorbar(cps[1, 2], ax=axs[1, 2], ticks=[-6, 0, 6], label=_pr_label)

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "thesis_seasonal")

    fig
    return


if __name__ == "__main__":
    app.run()
