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
    # Zonal structure of surface heat flux during El Niño and La Niña (observations)

    Paper figure `fig:Q-structure_era5` (`paper-outline.tex` in the ch3 Overleaf
    project). Composites of Nov-Dec-Jan (NDJ) means along the equator (5°S-5°N), for the
    10% of seasons with the warmest (El Niño, red) and coldest (La Niña, blue) Niño 3
    SST, 1950-2022:

    - **Top row:** totals: SST (ERSSTv5), precipitation (GPCP), and the net, shortwave,
      and latent surface heat fluxes (ERA5; positive into the ocean). Black is the
      climatology (mean over all seasons).
    - **Bottom row:** anomalies (composite minus climatology), with La Niña's sign
      reversed for comparison. The flux panels share their y-limits, with the y-axis
      pointing down (heat loss by the ocean is up).

    GPCP starts in 1979, so the precipitation composites and climatology use only the
    seasons since then (of the 7 strongest events of each kind, 5 El Niños and 3 La
    Niñas).

    `fig:Q-structure_cesm2` (`Q-structure_cesm2.py`) is the same figure for CESM2.

    **Inputs:** `DATA_FP/era5_nhf_glade.nc` (ERA5 fluxes), and in CVDP-OBS
    (`src.data.CVDP_OBS_FP`): `ersstv5.185401-202412.nc` (SST),
    `gpcp.mon.mean.197901-202403.nc` (precipitation).

    **Output:** `SAVE_FP/figures/Q-structure_era5.pdf`

    Before this script, the figure came from the "composites" section of
    `explore/nhf_validation.py` (ERA5 cells). Changes:

    - The composites use the 7 strongest events of each kind. Before, the record ended
      with a partial Nov-Dec 2022 season whose Niño 3 index was missing (NaN), and the
      ranking counted it as the strongest El Niño; so the El Niño composite left out
      2009 and averaged only 6 events (7 for precipitation, with 2022's partial season).
    - All variables end with the ERA5 record (the last season is NDJ 2021-22); before,
      the precipitation climatology also included NDJ 2022-23.
    - The SST anomalies are the composite minus the climatology, like the other
      variables (before, they were ERSST's anomalies from CVDP's climatology, which
      shifts the curves by a constant).
    - The number of events is set as a fraction of the seasons (`FRAC_EVENTS`).
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

    ## labeled arrays
    import xarray as xr

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
    return src, xr


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
    return FLUX_VARS, FRAC_EVENTS, PANELS, SEASON_START


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data

    Monthly equatorial profiles on ERSST's grid, over the ERA5 period, then seasonal
    means (full seasons only).
    """
    )
    return


@app.cell
def _(FLUX_VARS, SEASON_START, src, xr):
    ## ERA5 surface fluxes (positive into the ocean), 1950-2022
    _flux = src.data.load_era5_flux()[FLUX_VARS]

    ## ERSST total SST (°C) over the ERA5 period
    _sst = src.data.load_ersst("sst").sel(time=_flux.time)

    ## GPCP precipitation (mm/day) on ERSST's grid, over the ERA5 period (from 1979)
    _pr = src.data.load_gpcp().sel(time=slice(None, "2022"))
    _pr = _pr.interp(latitude=_sst.latitude, longitude=_sst.longitude)

    ## equatorial profiles, with the fluxes on ERSST's longitudes; dims (time, longitude)
    _profile_flux = src.indices.get_eq_profile(_flux)
    _profile_flux = _profile_flux.interp(longitude=_sst.longitude)
    profiles = xr.merge(
        [
            src.indices.get_eq_profile(_sst).rename("sst"),
            src.indices.get_eq_profile(_pr),
            _profile_flux,
        ]
    )

    ## Niño 3 SST, which picks the events; dims (time,)
    nino3 = src.utils.get_nino3(_sst)

    ## seasonal means (full seasons only)
    profiles_season = src.data.get_season(profiles, SEASON_START).compute()
    nino3_season = src.data.get_season(nino3, SEASON_START).compute()
    return nino3_season, profiles_season


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Composites

    Climatology: the mean over all seasons. Composite anomalies: the mean over the
    events, minus the climatology.
    """
    )
    return


@app.cell
def _(FRAC_EVENTS, nino3_season, profiles_season, src):
    ## climatological profiles; dims (longitude,)
    clim = profiles_season.mean("time")

    ## number of events of each kind
    n_events = round(FRAC_EVENTS * nino3_season.size)

    ## the warmest (El Niño) and coldest (La Niña) seasons
    _kwargs = dict(index=nino3_season, n=n_events)
    events_warm = src.stats.get_extremes(profiles_season, largest=True, **_kwargs)
    events_cold = src.stats.get_extremes(profiles_season, largest=False, **_kwargs)

    ## composite anomalies; dims (longitude,)
    warm = events_warm.mean("time") - clim
    cold = events_cold.mean("time") - clim
    return clim, cold, events_cold, events_warm, warm


@app.cell
def _(events_cold, events_warm, mo):
    ## the events, listed by the year their season starts
    mo.md(
        f"""
    El Niños: {list(events_warm.time.dt.year.values)}

    La Niñas: {list(events_cold.time.dt.year.values)}
    """
    )
    return


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

    ## fixed y-limits for the total fluxes (W m^-2)
    axs[0, 2].set_ylim([0, 130])
    axs[0, 3].set_ylim([175, 275])
    axs[0, 4].set_ylim([-150, -50])

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "Q-structure_era5")

    fig
    return


if __name__ == "__main__":
    app.run()
