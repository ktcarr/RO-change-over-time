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
    # Surface heat flux components in ENSO composites over time

    Paper figure `fig:nhf-hov` (`paper-outline.tex` in the ch3 Overleaf project). The
    same as `fig:ddt-hov` (`ddt-hov.py`), for two components of the net surface heat
    flux $Q$ shown there.
    Composites of strong single-year El Niños and La Niñas (December Niño 3.4 beyond its
    95th or 5th percentile, but not in the year before or after) in 40-year windows
    every 10 years, averaged over October-December and over the equator (5°S-5°N). Each
    panel shows longitude against the window's center year.

    - **Left:** composite anomalies in each window
    - **Right:** difference from the first window (1870)
    - **Rows:** the contributions to the SST tendency (assuming a 50 m mixed layer) of the
      net shortwave radiation (SW) and the latent heat flux (LHF), positive into the
      ocean. Both are divided by the composite Niño 3.4 index ($T_{34}$, in K) at each
      lag before averaging over October-December.

    Dashed lines mark the Niño 3.4 region (190°E-240°E) and the 2010 window.

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX`
    (`Th_anom.nc` by default) and shortwave and latent heat flux EOF scores
    (`consolidated/flux_anom_with_clearsky.nc`).

    **Output:** `SAVE_FP/figures/nhf-hov.pdf`

    Before this script, the figure came from the cell under "Heat flux" in
    `explore/composite-over-time.py`, which drew the two halves as separate figures (with extra
    rows for cloud shortwave and longwave) that were joined by hand. Changes:

    - $T_{34}$ is in K. It was the standardized index (divided by its standard deviation,
      1.26 K), so the old normalized rows were in units "per standard deviation" and are
      larger than here by that factor.
    - The window centered on 2090, which ran past the end of the simulation, is dropped.
    - The old figure was titled "Difference from 1880", but its first window was 1870.
    - The dashed lines mark Niño 3.4, as in the paper figure (the code had moved them to
      180°E and 230°E).
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
    ## arrays, labeled arrays, and plotting style
    import numpy as np
    import seaborn as sns
    import xarray as xr

    ## this project's modules: loading data, plotting, statistics
    import src.data
    import src.plot
    import src.stats

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all paper figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return np, src, xr


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
    ## Niño 3.4 index used to select events and normalize: "sst" (SST-based, used by
    ## the RO fits), or "ml50"/"ml80" (mixed-layer temperature over the top 50/80 m)
    T34_INDEX = "sst"

    ## window length (months) and centers: every 10 years, while the window fits in
    ## the simulation (1850-2100)
    WINDOW_SIZE = 480
    YEARS = list(np.arange(1870, 2081, 10))

    ## events: December Niño 3.4 beyond this percentile (or 1 minus it, for La Niña)
    PEAK_MONTH = 12
    Q_EVENT = 0.95

    ## which events: 1 = single-year only, as in the paper figure (drops events whose
    ## peak follows or precedes another beyond the cutoff, which matters most for La
    ## Niña); None = all events
    EVENT_TYPE = 1

    ## lags averaged (October-December, with lag 0 = December)
    LAGS = slice(-2, 0)

    ## mixed-layer depth (m) for converting heat flux to a temperature tendency
    H_ML = 50
    return EVENT_TYPE, H_ML, LAGS, PEAK_MONTH, Q_EVENT, T34_INDEX, WINDOW_SIZE, YEARS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data
    """
    )
    return


@app.cell
def _(T34_INDEX, src):
    ## Niño 3.4 anomalies (ensemble mean removed), K, dims (member, time)
    T34 = src.data.load_T34_anom(index=T34_INDEX)

    ## EOF scores and patterns of the fluxes shown (ensemble mean removed); loaded into
    ## memory, since every window reads them
    _, _anom = src.utils.load_flux_data()
    anom = _anom[["fsns", "fsns_comp", "lhflx", "lhflx_comp"]].load()
    return T34, anom


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Composites

    For each window, the mean over strong events of each field (lags -12 to 12 months),
    averaged over 5°S-5°N.
    """
    )
    return


@app.cell
def _(EVENT_TYPE, PEAK_MONTH, Q_EVENT, T34, WINDOW_SIZE, YEARS, anom, src):
    ## settings shared by the El Niño and La Niña composites
    _kwargs = dict(
        anom=anom,
        T34=T34,
        varnames=["fsns", "lhflx"],
        years=YEARS,
        window_size=WINDOW_SIZE,
        peak_month=PEAK_MONTH,
        q=Q_EVENT,
        event_type=EVENT_TYPE,
    )

    ## composites, dims (year, lag, longitude)
    comp_warm = src.stats.get_composite_hov(is_warm=True, **_kwargs)
    comp_cold = src.stats.get_composite_hov(is_warm=False, **_kwargs)
    return comp_cold, comp_warm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Quantities shown

    Each flux divided by $\rho C_p H$, then by the composite $T_{34}$ at each lag, then
    averaged over October-December.
    """
    )
    return


@app.cell
def _(H_ML, LAGS, comp_cold, comp_warm, src, xr):
    def get_rows(comp):
        """
        October-December means of the quantities in each row of the figure.

        Args:
            - comp (xr.Dataset): composite with "fsns" and "lhflx" (W m^-2, as in
              CESM: shortwave positive down, latent heat positive up) and "T_34" (K),
              dims (year, lag, longitude)

        Returns:
            - rows (xr.Dataset): "Q_sw" and "Q_lhf" (yr^-1), dims (year, longitude)
        """

        ## fluxes as temperature tendencies (K/yr), positive into the ocean
        Q_sw = src.data.flux_to_tendency(comp["fsns"], H=H_ML)
        Q_lhf = src.data.flux_to_tendency(-comp["lhflx"], H=H_ML)

        ## normalize by T34 at each lag
        rows = xr.Dataset({"Q_sw": Q_sw / comp["T_34"], "Q_lhf": Q_lhf / comp["T_34"]})

        ## October-December mean
        rows = rows.sel(lag=LAGS).mean("lag")

        return rows

    ## quantities for each composite
    rows_warm = get_rows(comp_warm)
    rows_cold = get_rows(comp_cold)
    return rows_cold, rows_warm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(rows_cold, rows_warm, src):
    ## variable, label, units, and colorbar limit of each row (differences use half)
    rows = [
        dict(var="Q_sw", label=r"SW / $T_{34}$", units=r"yr$^{-1}$", amp=3),
        dict(var="Q_lhf", label=r"LHF / $T_{34}$", units=r"yr$^{-1}$", amp=3),
    ]

    ## values over time (left) and differences from the first window (right)
    fig = src.plot.plot_hov_grid(rows_warm, rows_cold, rows, figsize=(11, 4.5))

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "nhf-hov")

    fig
    return


if __name__ == "__main__":
    app.run()
