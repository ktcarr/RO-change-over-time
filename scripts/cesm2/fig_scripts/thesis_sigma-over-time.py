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
    # Change over time in equatorial SST variability (thesis)

    Thesis figure `fig:sigma-over-time` (`chapter_enso-change.tex`). Hovmöller plots of
    the fractional change in the standard deviation of SST along the equator, relative to
    the first window, for each season: **(a)** DJF, **(b)** MAM, **(c)** JJA, **(d)**
    SON. Each row is a 40-year window (labeled by its center, every 5 years, 1870-2080);
    the standard deviation pools all 100 members and the window's months of that season.
    It is computed at each grid point from the anomalies (from the ensemble mean), then
    averaged over 5°S-5°N. Dashed gray lines mark 2030 and 150°W.

    `thesis_sigma-by-season.py` (`fig:sigma-by-season`) shows the standard deviation
    itself, by month, for the first (1870) and a late (2080) window.

    **Inputs:** `DATA_FP/cesm/consolidated/anom.nc` (SST).

    **Output:** `SAVE_FP/figures/thesis_sigma-over-time.pdf`

    Before this script, the figure was `sigma-over-time_hov_simple`, made by
    `explore/mean-state-changes.py` (cell under "Variance over time"). Changes:

    - The colors are the fractional change, as the colorbar label and the thesis caption
      say. Before, they were the change in °C (the standard deviation is 0.4-1.5°C, so
      the two look alike east of the dateline but differ in the west).
    - The colors extend past ±0.4 (before, values beyond it were left blank).
    - Only full 40-year windows are computed (before, the 2085 window, cut short by the
      end of the simulation, was computed but came out empty).
    - The thesis caption says 41-year windows; they are 40 years (480 months).
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

    ## plot style, labeled arrays
    import pandas as pd
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
    return pd, plt, src, xr


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
    ## window length and stride (months)
    WINDOW_SIZE = 480
    STRIDE = 60

    ## seasons (panels), in order
    SEASONS = ["DJF", "MAM", "JJA", "SON"]

    ## longitudes shown (degrees east), and their ticks
    XLIM = [140, 280]
    XTICKS = [150, 210, 270]

    ## window (year) and longitude marked by dashed lines; ticks on the time axis
    YEAR_LINE = 2030
    LON_LINE = 210
    YTICKS = [1870, 2030, 2080]
    return LON_LINE, SEASONS, STRIDE, WINDOW_SIZE, XLIM, XTICKS, YEAR_LINE, YTICKS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data

    SST anomalies (from the ensemble mean), EOF-compressed: patterns and scores. The
    scores are split into 40-year windows.
    """
    )
    return


@app.cell
def _(STRIDE, WINDOW_SIZE, src):
    ## anomalies from the ensemble mean (the ensemble mean itself isn't needed)
    _, _anom = src.data.load_eofs(["sst"])

    ## patterns (mode, latitude, longitude) and scores (member, time, mode)
    _components, _scores = src.utils.split_components(_anom)
    components = _components["sst"]
    _scores = _scores["sst"].compute()

    ## 40-year windows; dims (member, year, time, mode)
    _windows = src.utils.get_windowed(_scores, window_size=WINDOW_SIZE, stride=STRIDE)

    ## keep only full windows (the last one runs past the end of the simulation)
    windows = _windows.dropna("year", how="any")
    return components, windows


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Standard deviation by season

    For each window and season, the standard deviation at each grid point near the
    equator (from the scores' covariance), averaged over 5°S-5°N; then the fractional
    change from the first window.
    """
    )
    return


@app.cell
def _(components, pd, src, windows, xr):
    ## standard deviation in each window; dims (year, season, longitude)
    _sigma_list = []
    for _year in windows.year.values:
        _window = windows.sel(year=_year)
        _sigma_list.append(src.stats.get_eq_sigma(components, _window, "season"))
    sigma = xr.concat(_sigma_list, dim=pd.Index(windows.year.values, name="year"))

    ## fractional change from the first window
    frac = src.stats.frac_change(sigma)
    return (frac,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(LON_LINE, SEASONS, XLIM, XTICKS, YEAR_LINE, YTICKS, frac, plt, src):
    ## one panel per season
    fig, axs = plt.subplots(1, 4, figsize=(8, 3), layout="constrained")

    ## color levels and dashed reference lines
    _levels = src.utils.make_cb_range(0.4, 0.04)
    _line_kwargs = dict(ls="--", c="gray", lw=0.8)

    for _ax, _season, _letter in zip(axs, SEASONS, "abcd"):

        ## fractional change, by longitude and window
        _data = frac.sel(season=_season).transpose("year", "longitude")
        _cp = _ax.contourf(
            _data.longitude,
            _data.year,
            _data,
            cmap="cmo.balance",
            levels=_levels,
            extend="both",
        )

        ## reference lines, labels, and longitude ticks
        _ax.axhline(YEAR_LINE, **_line_kwargs)
        _ax.axvline(LON_LINE, **_line_kwargs)
        _ax.set_title(f"({_letter}) {_season}", loc="left")

        ## longitude axis
        _ax.set_xlim(XLIM)
        _ax.set_xticks(XTICKS)
        _ax.set_xlabel("Lon.")

    ## colorbar, and time ticks only on the first panel
    fig.colorbar(_cp, ax=axs[-1], ticks=[-0.4, 0, 0.4], label="Frac. change")
    axs[0].set_yticks(YTICKS)
    for _ax in axs[1:]:
        _ax.set_yticks([])

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "thesis_sigma-over-time")

    fig
    return


if __name__ == "__main__":
    app.run()
