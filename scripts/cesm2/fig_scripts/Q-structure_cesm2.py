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
    # Zonal structure of surface heat flux during El Niño and La Niña (CESM2)

    Paper figure `fig:Q-structure_cesm2` (`paper-outline.tex` in the ch3 Overleaf
    project). Same as `fig:Q-structure_era5` (`Q-structure_era5.py`), but for CESM2:
    composites of Nov-Dec-Jan (NDJ) means along the equator (5°S-5°N), for the 10% of
    seasons (pooling all 100 members, 1950-1980) with the warmest (El Niño, red) and
    coldest (La Niña, blue) Niño 3 SST:

    - **Top row:** totals: SST, precipitation, and the net, shortwave, and latent
      surface heat fluxes (positive into the ocean). Black is the climatology: the
      ensemble mean, averaged over all seasons.
    - **Bottom row:** anomalies (from the ensemble mean), with La Niña's sign reversed
      for comparison. The flux panels share their y-limits, with the y-axis pointing down
      (heat loss by the ocean is up).

    **Inputs** (under `DATA_FP/cesm/consolidated`): `forced.nc` and `anom.nc` (SST,
    precipitation, net flux), `flux_forced_with_clearsky.nc` and
    `flux_anom_with_clearsky.nc` (shortwave and latent fluxes).

    **Output:** `SAVE_FP/figures/Q-structure_cesm2.pdf`

    Before this script, the figure came from the "composites" section of
    `explore/nhf_validation.py` (CESM2 cells). Changes:

    - Events are picked by Niño 3 SST, as for the observations (before, by Niño 3.4).
    - The number of events is set as a fraction of the seasons (`FRAC_EVENTS`): 300
      (before, 301).
    - The equatorial average is taken after undoing the EOFs' latitude weighting
      (before, the weighted patterns were averaged; the weights differ by less than 0.2%
      within 5° of the equator).
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
    ## plotting style
    import seaborn as sns

    ## this project's modules: loading data, regional averages
    import src.data
    import src.indices

    ## this project's modules: plotting, statistics (composites)
    import src.plot
    import src.stats

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all paper figures; color_codes (on by default) makes
    ## "r" and "b" the colorblind palette's red and blue
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return (src,)


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
    ## first month of the season: 11 = Nov-Dec-Jan (NDJ)
    SEASON_START = 11

    ## years (a period overlapping the observations)
    YEARS = slice("1950", "1980")

    ## fraction of seasons in each composite (the warmest and the coldest)
    FRAC_EVENTS = 0.1

    ## variables (columns) and their titles
    PANELS = dict(
        sst="SST",
        pr="Precip. (mm day$^{-1}$)",
        nhf=r"NHF (W m$^{-2}$)",
        sw="Shortwave",
        lhf="Latent heat flux",
    )

    ## surface fluxes (positive into the ocean)
    FLUX_VARS = ["nhf", "sw", "lhf"]
    return FLUX_VARS, FRAC_EVENTS, PANELS, SEASON_START, YEARS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data

    The fields are EOF-compressed: scores (time series) and patterns. Unit and sign
    changes are applied to the scores, and seasonal means are taken of the scores
    (full seasons only); both commute with reconstructing the fields.
    """
    )
    return


@app.cell
def _(SEASON_START, YEARS, src):
    ## ensemble mean and anomalies from it (scores and patterns)
    _forced, _anom = src.data.load_eofs(["sst", "pr", "nhf", "fsns", "lhflx"])

    ## patterns (no time dimension) and scores, for each
    patterns = dict()
    scores_season = dict()
    for _name, _ds in [("forced", _forced), ("anom", _anom)]:
        patterns[_name], _scores = src.utils.split_components(_ds)

        ## precipitation in mm/day (from m/s)
        _scores["pr"] = _scores["pr"] * 8.64e4

        ## latent heat flux positive into the ocean (CESM's LHFLX is positive up)
        _scores["lhflx"] = -_scores["lhflx"]

        ## seasonal means over the chosen years
        _scores = _scores.sel(time=YEARS)
        scores_season[_name] = src.data.get_season(_scores, SEASON_START).compute()

    ## short names for the shortwave (FSNS, positive down) and latent fluxes
    _names = dict(fsns="sw", lhflx="lhf")
    for _name in patterns:
        patterns[_name] = patterns[_name].rename(_names)
        scores_season[_name] = scores_season[_name].rename(_names)
    return patterns, scores_season


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Composites

    Climatology: the ensemble mean, averaged over all seasons. Composite anomalies: the
    mean of the anomalies over the events (pooling members and seasons).
    """
    )
    return


@app.cell
def _(FRAC_EVENTS, patterns, scores_season, src):
    ## climatological profiles; dims (longitude,)
    clim = src.utils.reconstruct_fn(
        components=patterns["forced"],
        scores=scores_season["forced"].mean("time"),
        fn=src.indices.get_eq_profile,
    )

    ## pool members and seasons into one sample dimension
    _scores = scores_season["anom"].stack(sample=["time", "member"])

    ## Niño 3 SST anomaly of each sample, which picks the events; dims (sample,)
    _nino3 = src.utils.reconstruct_fn(
        components=patterns["anom"]["sst"],
        scores=_scores["sst"],
        fn=src.utils.get_nino3,
    )

    ## number of events of each kind
    n_events = round(FRAC_EVENTS * _nino3.size)

    ## scores of the warmest (El Niño) and coldest (La Niña) samples
    _kwargs = dict(index=_nino3, n=n_events, dim="sample")
    _scores_warm = src.stats.get_extremes(_scores, largest=True, **_kwargs)
    _scores_cold = src.stats.get_extremes(_scores, largest=False, **_kwargs)

    ## composite anomalies: mean scores over the events, then profiles; dims (longitude,)
    warm = src.utils.reconstruct_fn(
        components=patterns["anom"],
        scores=_scores_warm.mean("sample"),
        fn=src.indices.get_eq_profile,
    )
    cold = src.utils.reconstruct_fn(
        components=patterns["anom"],
        scores=_scores_cold.mean("sample"),
        fn=src.indices.get_eq_profile,
    )
    return clim, cold, n_events, warm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(FLUX_VARS, PANELS, clim, cold, src, warm):
    ## totals (top) and anomalies (bottom) of each variable
    fig, axs = src.plot.plot_zonal_composites(clim, warm, cold, PANELS, FLUX_VARS)

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "Q-structure_cesm2")

    fig
    return


if __name__ == "__main__":
    app.run()
