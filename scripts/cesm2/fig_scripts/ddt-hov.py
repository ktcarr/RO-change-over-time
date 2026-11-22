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
    # Zonal structure of ENSO composites over time

    Paper figure `fig:ddt-hov` (`paper-outline.tex` in the ch3 Overleaf project).
    Composites of strong single-year El Niños and La Niñas (December Niño 3.4 beyond its
    95th or 5th percentile, but not in the year before or after) in 40-year windows
    every 10 years, averaged over October-December and over the equator (5°S-5°N). Each
    panel shows longitude against the window's center year.

    - **Left:** composite anomalies in each window
    - **Right:** difference from the first window (1870)
    - **Rows:** SST; SST tendency; the net surface heat flux contribution to the SST
      tendency ($Q$, assuming a 50 m mixed layer); precipitation. All but SST are
      divided by the composite Niño 3.4 index ($T_{34}$, in K) at each lag before
      averaging over October-December.

    Dashed lines mark the Niño 3.4 region (190°E-240°E) and the 2010 window.

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX`
    (`Th_anom.nc` by default) and SST, net heat flux, and precipitation EOF scores
    (`consolidated/anom.nc`).

    **Output:** `SAVE_FP/figures/ddt-hov.pdf`

    Before this script, the figure came from the cell under "Clean up fig." in
    `composite-over-time.py`, which drew the two halves as separate figures (with an
    extra $\tau_x$ row) that were joined by hand. Changes:

    - $T_{34}$ is in K. It was the standardized index (divided by its standard deviation,
      1.26 K), so the old normalized rows were in units "per standard deviation" and are
      larger than here by that factor.
    - The precipitation colorbar is labeled ±4 (±2 for differences), the range its
      levels had; the old labels said ±10.
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
    ## arrays and plotting style
    import numpy as np
    import seaborn as sns

    ## this project's modules: loading data, plotting, statistics
    import src.data
    import src.plot
    import src.stats

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all paper figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return np, src


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

    ## EOF scores and patterns of the fields shown (ensemble mean removed); loaded into
    ## memory, since every window reads them
    _, _anom = src.utils.load_consolidated()
    anom = _anom[["sst", "sst_comp", "nhf", "nhf_comp", "pr", "pr_comp"]].load()
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
        varnames=["sst", "nhf", "pr"],
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

    SST tendency is a centered difference in lag; $Q$ is the net heat flux divided by
    $\rho C_p H$. Except SST, each is divided by the composite $T_{34}$ at each lag,
    then averaged over October-December.
    """
    )
    return


@app.cell
def _(H_ML, LAGS, comp_cold, comp_warm, src):
    def get_rows(comp):
        """
        October-December means of the quantities in each row of the figure.

        Args:
            - comp (xr.Dataset): composite with "sst" (K), "nhf" (W m^-2), "pr"
              (kg m^-2 s^-1), and "T_34" (K), dims (year, lag, longitude)

        Returns:
            - rows (xr.Dataset): "sst" (K), "ddt_sst" and "Q" (yr^-1), "pr"
              (mm day^-1 K^-1), dims (year, longitude)
        """

        ## SST tendency (K/month -> K/yr)
        ddt_sst = 12 * comp["sst"].differentiate("lag")

        ## heat flux as a temperature tendency (K/yr)
        Q = src.data.flux_to_tendency(comp["nhf"], H=H_ML)

        ## precipitation in mm/day (1 kg m^-2 = 1 mm of water; 86400 s per day)
        pr = 8.64e4 * comp["pr"]

        ## SST is shown as is
        rows = comp[["sst"]].copy()

        ## the others are normalized by T34 at each lag
        rows["ddt_sst"] = ddt_sst / comp["T_34"]
        rows["Q"] = Q / comp["T_34"]
        rows["pr"] = pr / comp["T_34"]

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
        dict(var="sst", label="SST", units="K", amp=4),
        dict(
            var="ddt_sst",
            label=r"$\partial_t$SST / $T_{34}$",
            units=r"yr$^{-1}$",
            amp=3,
        ),
        dict(var="Q", label=r"$Q$ / $T_{34}$", units=r"yr$^{-1}$", amp=3),
        dict(
            var="pr",
            label=r"Precip. / $T_{34}$",
            units=r"mm day$^{-1}$ K$^{-1}$",
            amp=4,
            cmap="cmo.balance_r",
        ),
    ]

    ## values over time (left) and differences from the first window (right)
    fig = src.plot.plot_hov_grid(rows_warm, rows_cold, rows, figsize=(11, 8))

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "ddt-hov")

    fig
    return


if __name__ == "__main__":
    app.run()
