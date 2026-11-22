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
    # Composite El Niño and La Niña evolution (thesis)

    Thesis figures `fig:nino-hov` and `fig:nina-hov` (`chapter_enso-change.tex`):
    composites of strong El Niños (December Niño 3.4 above its 95th percentile) and La
    Niñas (below its 5th percentile) in two 40-year windows, centered on 2010 and 2080,
    averaged over the equator (5°S-5°N). Each panel shows longitude against lag (lag 0 =
    December).

    - **Columns:** the 2010 window; the 2080 window; their difference (2080 minus 2010)
    - **Rows:** SST; SST tendency; the net surface heat flux contribution to the SST
      tendency ($Q$, assuming a 50 m mixed layer); precipitation

    Black lines show the total ENSO longitude index (ELI; the average longitude of
    equatorial grid points warmer than the 5°S-5°N mean SST), solid for 2010 and
    dashed for 2080. The box marks the Niño 3.4 longitudes, from November to January.

    Only single-year events are used (peaks beyond the cutoff that don't follow or
    precede another), as in the old code; the thesis text doesn't say so.

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX`
    (`Th_anom.nc` by default), the ELI (`eli.nc`), and SST, net heat flux, and
    precipitation EOF scores (`consolidated/anom.nc`).

    **Outputs:** `SAVE_FP/figures/thesis_nino-hov.pdf` (El Niño) and
    `thesis_nina-hov.pdf` (La Niña)

    Before this script, both figures came from the cell under "SST" in
    `explore/composite-over-time.py` (`combined_hov`, `combined_hov_nina`), switched between El
    Niño and La Niña by hand. Changes:

    - The committed code had since moved on (1870 and 2000 windows, La Niña
      normalized by Niño 3); the settings here are those that made the thesis
      figures (git history up to commit 81a2bc1).
    - The composites are computed here from the EOF scores, averaged over 5°S-5°N
      (with `src.stats.get_composite_hov`), instead of read from the full 3-D
      composites saved under `SAVE_FP/composites`.
    - All boxes are black. In the thesis figures, the boxes on the SST tendency and
      heat flux rows are magenta (the thesis text refers to "magenta boxes" in
      `fig:nino-hov` f and i) and the others black, edited by hand after export; the
      code drew them all in magenta.
    - Precipitation is converted to mm day$^{-1}$ with 86400 s per day (the old code used
      86000, so its contour levels were 0.5% apart from these).
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
    ## labeled arrays, plot style
    import seaborn as sns
    import xarray as xr

    ## this project's modules: loading data, plotting, statistics
    import src.data
    import src.plot
    import src.stats

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})
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
    ## Niño 3.4 index used to select events: "sst" (SST-based, used by the RO fits),
    ## or "ml50"/"ml80" (mixed-layer temperature over the top 50/80 m)
    T34_INDEX = "sst"

    ## window length (months) and the two window centers compared
    WINDOW_SIZE = 480
    YEARS = [2010, 2080]

    ## events: December Niño 3.4 beyond this percentile (or 1 minus it, for La Niña)
    PEAK_MONTH = 12
    Q_EVENT = 0.95

    ## which events: 1 = single-year only (as in the thesis figures); None = all events
    EVENT_TYPE = 1

    ## mixed-layer depth (m) for converting heat flux to a temperature tendency
    H_ML = 50

    ## ELI variable: threshold is the 5S-5N mean SST
    ELI_VAR = "eli_05"
    return ELI_VAR, EVENT_TYPE, H_ML, PEAK_MONTH, Q_EVENT, T34_INDEX, WINDOW_SIZE, YEARS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data
    """
    )
    return


@app.cell
def _(ELI_VAR, T34_INDEX, src):
    ## Niño 3.4 anomalies (ensemble mean removed), K, dims (member, time)
    T34 = src.data.load_T34_anom(index=T34_INDEX)

    ## ELI anomalies (member, time) and forced ELI (time), degrees east
    eli_anom = src.data.load_eli(ELI_VAR)
    eli_forced = src.data.load_eli_forced(ELI_VAR)

    ## EOF scores and patterns of the fields shown (ensemble mean removed); loaded into
    ## memory, since both windows read them
    _, _anom = src.utils.load_consolidated()
    anom = _anom[["sst", "sst_comp", "nhf", "nhf_comp", "pr", "pr_comp"]].load()
    return T34, anom, eli_anom, eli_forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Composites

    For each window, the mean over strong events of each field (lags -12 to 12 months),
    averaged over 5°S-5°N, and of the ELI anomaly.
    """
    )
    return


@app.cell
def _(EVENT_TYPE, PEAK_MONTH, Q_EVENT, T34, WINDOW_SIZE, YEARS, anom, eli_anom, src):
    ## settings shared by the El Niño and La Niña composites
    _kwargs = dict(
        anom=anom,
        T34=T34,
        varnames=["sst", "nhf", "pr"],
        years=YEARS,
        window_size=WINDOW_SIZE,
        peak_month=PEAK_MONTH,
        q=Q_EVENT,
        event_type=EVENT_TYPE,
        indices=eli_anom.rename("eli").to_dataset(),
    )

    ## composites, dims (year, lag, longitude), plus "T_34" and "eli" (year, lag)
    comp_warm = src.stats.get_composite_hov(is_warm=True, **_kwargs)
    comp_cold = src.stats.get_composite_hov(is_warm=False, **_kwargs)
    return comp_cold, comp_warm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Total ELI

    The composite ELI anomaly plus the forced ELI's monthly climatology in each window,
    arranged along the lags.
    """
    )
    return


@app.cell
def _(PEAK_MONTH, WINDOW_SIZE, YEARS, comp_warm, eli_forced, src, xr):
    ## monthly climatology of the forced ELI in each window, dims (year, month)
    _clims = []
    for _year in YEARS:
        _window = src.data.get_window(eli_forced, _year, WINDOW_SIZE)
        _clims.append(_window.groupby("time.month").mean())
    _eli_clim = xr.concat(_clims, dim=xr.DataArray(YEARS, dims="year", name="year"))

    ## the climatology at each lag's calendar month, dims (year, lag)
    eli_clim = src.stats.clim_at_lags(_eli_clim, comp_warm.lag, PEAK_MONTH)
    return (eli_clim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Quantities shown

    SST tendency is a centered difference in lag; $Q$ is the net heat flux divided by
    $\rho C_p H$.
    """
    )
    return


@app.cell
def _(H_ML, comp_cold, comp_warm, eli_clim, src):
    def get_rows(comp):
        """
        Quantities in each row of the figure, plus the total ELI.

        Args:
            - comp (xr.Dataset): composite with "sst" (K), "nhf" (W m^-2), "pr"
              (kg m^-2 s^-1), and "eli" (ELI anomaly, degrees east), dims (year,
              lag, longitude)

        Returns:
            - rows (xr.Dataset): "sst" (K), "ddt_sst" and "Q" (K/month), "pr"
              (mm/day), and "eli" (total ELI, degrees east)
        """

        ## SST is shown as is
        rows = comp[["sst"]].copy()

        ## SST tendency (K/month, since lags are in months)
        rows["ddt_sst"] = comp["sst"].differentiate("lag")

        ## heat flux as a temperature tendency (K/yr -> K/month)
        rows["Q"] = src.data.flux_to_tendency(comp["nhf"], H=H_ML) / 12

        ## precipitation in mm/day (1 kg m^-2 = 1 mm of water; 86400 s per day)
        rows["pr"] = 8.64e4 * comp["pr"]

        ## total ELI: anomaly plus the forced climatology
        rows["eli"] = comp["eli"] + eli_clim

        return rows

    ## quantities for each composite
    rows_warm = get_rows(comp_warm)
    rows_cold = get_rows(comp_cold)
    return rows_cold, rows_warm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure settings
    """
    )
    return


@app.cell
def _(YEARS):
    ## variable, colorbar label, and colorbar limit of each row (differences use half);
    ## precipitation is reversed so wet anomalies are blue
    ROWS = [
        dict(var="sst", label="SST (K)", amp=4),
        dict(var="ddt_sst", label=r"$\frac{\partial}{\partial t}$SST (K/mo)", amp=1),
        dict(var="Q", label=r"$Q$ (K/mo)", amp=1),
        dict(var="pr", label="pr (mm/day)", amp=12, cmap="cmo.balance_r"),
    ]

    ## column titles
    TITLES = [f"{YEARS[0]}", f"{YEARS[1]}", "Difference"]
    return ROWS, TITLES


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## El Niño (`fig:nino-hov`)
    """
    )
    return


@app.cell
def _(ROWS, TITLES, YEARS, rows_warm, src):
    ## 2010 and 2080 windows, and their difference
    fig_nino = src.plot.plot_lag_hov_grid(
        rows_warm.sel(year=YEARS[0]),
        rows_warm.sel(year=YEARS[1]),
        ROWS,
        TITLES,
    )

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig_nino, "thesis_nino-hov")

    fig_nino
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## La Niña (`fig:nina-hov`)
    """
    )
    return


@app.cell
def _(ROWS, TITLES, YEARS, rows_cold, src):
    ## 2010 and 2080 windows, and their difference
    fig_nina = src.plot.plot_lag_hov_grid(
        rows_cold.sel(year=YEARS[0]),
        rows_cold.sel(year=YEARS[1]),
        ROWS,
        TITLES,
    )

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig_nina, "thesis_nina-hov")

    fig_nina
    return


if __name__ == "__main__":
    app.run()
