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
    # Asymmetric effect of the weaker zonal SST gradient (thesis)

    Thesis figure `fig:eli` (`chapter_enso-change.tex`). DJF means of the ENSO
    longitude index (ELI; the mean longitude of equatorial Pacific grid points warmer
    than the 5°S-5°N mean) and the Niño indices, in 40-year windows labeled by their
    centers:

    - **(a)** PDFs of the scaled ELI anomaly ($\Delta$ELI) in three windows. The scaled
      ELI is the ELI anomaly (from the ensemble mean) times the window's mean zonal SST
      difference $T_w - T_e$ (west Pacific 120°E-160°E minus Niño 3, 5°S-5°N; ensemble
      mean, same calendar month) divided by its 1870 value (annual mean), and is then
      measured from the window's median.
    - **(b)** Niño 4 anomaly (from the ensemble mean) against $\Delta$ELI, for the 2010
      (blue) and 2080 (orange) windows, with cubic least-squares fits (dashed: 2010,
      solid: 2080), drawn over the 1st-99th percentiles of each window's $\Delta$ELI.
      The 2080 points are shifted 40° to the right for clarity, with the 2010 fit
      repeated there (over the 2080 range) for comparison.
    - **(c, d)** same as (b), for Niño 3.4 and Niño 3

    **Inputs** (under `DATA_FP/cesm`): `eli.nc` (ELI, from
    `preprocess/preprocess_cesm.py`), `consolidated/forced.nc` (ensemble-mean SST, for
    $T_w - T_e$), and the Niño indices chosen by `INDEX`: `Th_anom.nc` (SST, the default),
    or `Th_50.nc` / `Th_80.nc` (mixed-layer temperature).

    **Output:** `SAVE_FP/figures/thesis_eli.pdf`

    Before this script, the figure was `eli-comparison`, made by `explore/stats-changes.py`
    (cells under "Plot temperature as a function of ELI", using the PDFs from the cell
    under "ELI"). Changes:

    - The fits are cubic polynomials. Before, the fitted function was
      $e x^3 + c x^2 + b x + c$: a typo tied the constant to the $x^2$ coefficient,
      which pinned the intercept near 0. That held down the cubic term of the 2080 Niño 4
      fit (whose free intercept is +0.3 K). A cubic fits better than a quadratic for
      every panel (also with 10-fold cross-validation); the thesis says only "best fit
      polynomial".
    - The fit curves are drawn over the 1st-99th percentiles of each window's
      $\Delta$ELI (before, from -10° to a fixed 35°, or 25° for Niño 3), so they don't
      extrapolate past the data.
    - The x-axis ticks are labeled with the $\Delta$ELI values of each cluster. Before,
      the labels (-55, -15, 30) didn't match the data at those ticks.
    - DJF means use full seasons only, 39 per window (before, each window also had a
      Jan-Feb-only and a Dec-only value).
    - The Niño indices are set by `INDEX`, SST-based by default (before, the top-80 m
      mixed-layer temperature).
    - $T_w - T_e$ is computed here from the ensemble-mean SST (before, read from
      `SAVE_FP/cesm_dTdx_Tw_v3.nc`, written by `scale.py` with the same definition).
      The thesis (Eqn. `eqn:eli-scaled`) writes the scaling as $(T_w - T_e)$ times ELI;
      in the code it has always been the ratio to the 1870 value, which keeps degrees
      longitude as the unit.
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

    ## numerics, plot style
    import numpy as np
    import seaborn as sns

    ## this project's modules: loading data, regional averages, plotting
    import src.data
    import src.indices
    import src.plot

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all figures (colorblind palette: blue, orange, ...)
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return np, plt, src


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
    ## Niño indices: "sst" (SST-based, used by the RO fits), or "ml50"/"ml80"
    ## (mixed-layer temperature over the top 50/80 m)
    INDEX = "sst"

    ## window length and stride (months)
    WINDOW_SIZE = 480
    STRIDE = 60

    ## windows (center years) shown in the PDFs (a), and in the scatter plots (b-d)
    PDF_YEARS = [1870, 2010, 2080]
    YEAR_EARLY = 2010
    YEAR_LATE = 2080

    ## histogram bin edges (degrees longitude) for the PDFs
    PDF_EDGES = np.linspace(-20, 40, 20)

    ## regions for the zonal SST difference T_w - T_e (degrees east)
    LONS_TW = [120, 160]
    LONS_TE = [210, 270]

    ## Niño indices (panels b-d) and their titles
    PANELS = dict(T_4="Niño 4", T_34="Niño 3.4", T_3="Niño 3")

    ## percentiles of each window's ΔELI over which the fit curves are drawn (the
    ## fits themselves use all samples)
    FIT_DISPLAY_RANGE = [1, 99]

    ## shift of the late window's points (degrees longitude), and the ticks of each
    ## cluster (ΔELI values)
    SHIFT = 40
    ELI_TICKS = [-10, 10]
    return (
        ELI_TICKS,
        FIT_DISPLAY_RANGE,
        INDEX,
        LONS_TE,
        LONS_TW,
        PANELS,
        PDF_EDGES,
        PDF_YEARS,
        SHIFT,
        STRIDE,
        WINDOW_SIZE,
        YEAR_EARLY,
        YEAR_LATE,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Zonal SST difference

    $T_w - T_e$ of the ensemble mean, averaged over each window and calendar month
    (the average commutes with reconstructing the field from its EOFs), divided by the
    first window's annual mean.
    """
    )
    return


@app.cell
def _(LONS_TE, LONS_TW, STRIDE, WINDOW_SIZE, src):
    ## ensemble-mean SST: patterns and scores
    _forced, _ = src.data.load_eofs(["sst"])
    _components, _scores = src.utils.split_components(_forced)

    ## window-mean scores for each calendar month; dims (year, month, mode)
    _windows = src.utils.get_windowed(
        _scores["sst"], window_size=WINDOW_SIZE, stride=STRIDE
    )
    _scores_mean = _windows.groupby("time.month").mean()

    def _get_dx(x):
        """
        West-minus-east equatorial SST difference, T_w - T_e.

        Args:
            - x (xr.DataArray): field with "longitude" and "latitude"

        Returns:
            - dx (xr.DataArray): T_w minus T_e
        """
        return src.indices.get_eq_avg(x, LONS_TW) - src.indices.get_eq_avg(x, LONS_TE)

    ## T_w - T_e in each window and month; dims (year, month)
    dTdx = src.utils.reconstruct_fn(_components["sst"], _scores_mean, fn=_get_dx)

    ## ratio to the first window's annual mean (about 0.7 by 2080)
    dTdx_ratio = dTdx / dTdx.isel(year=0).mean("month")
    return (dTdx_ratio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load ELI and Niño indices

    Monthly anomalies (from the ensemble mean), cut into windows; the ELI is scaled,
    then both are averaged over DJF (full seasons only).
    """
    )
    return


@app.cell
def _(INDEX, STRIDE, WINDOW_SIZE, dTdx_ratio, src):
    ## shared window arguments
    _kwargs = dict(window_size=WINDOW_SIZE, stride=STRIDE)

    ## ELI anomalies in windows, scaled by each window's (monthly) gradient ratio
    _eli = src.utils.get_windowed(src.data.load_eli("eli_05"), **_kwargs)
    _eli_scaled = _eli.groupby("time.month") * dTdx_ratio

    ## DJF means, measured from each window's median; dims (member, year, time)
    eli = src.data.get_djf_anom(_eli_scaled)

    ## Niño anomalies in windows, and their DJF means (from the ensemble mean)
    _nino = src.utils.get_windowed(src.data.load_nino_anom(INDEX), **_kwargs)
    nino = src.data.get_djf(_nino)
    return eli, nino


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## PDFs and fits
    """
    )
    return


@app.cell
def _(
    FIT_DISPLAY_RANGE,
    PANELS,
    PDF_EDGES,
    PDF_YEARS,
    YEAR_EARLY,
    YEAR_LATE,
    eli,
    nino,
    np,
    src,
):
    ## (a) normalized histogram for each window, pooling members and years
    pdfs = []
    for _year in PDF_YEARS:
        _pdf, _ = src.utils.get_empirical_pdf(
            eli.sel(year=_year).values.flatten(), PDF_EDGES
        )
        pdfs.append(_pdf)

    ## (b-d) samples (pooling members and years) and cubic fits, for each window
    samples = dict()
    fits = dict()
    for _year in [YEAR_EARLY, YEAR_LATE]:

        ## same (member, time) order for ΔELI and the Niño indices, so samples pair up
        _x = eli.sel(year=_year).transpose("member", "time").values.flatten()
        samples[_year] = dict(eli=_x)
        fits[_year] = dict()

        ## ΔELI values for drawing the fits: the window's central range
        _lo, _hi = np.percentile(_x, FIT_DISPLAY_RANGE)
        samples[_year]["eli_fit"] = np.linspace(_lo, _hi)

        ## each Niño index against ΔELI
        for _name in PANELS:
            _y = nino[_name].sel(year=_year).transpose("member", "time")
            _y = _y.values.flatten()
            samples[_year][_name] = _y

            ## cubic least-squares fit
            fits[_year][_name] = np.polynomial.Polynomial.fit(_x, _y, deg=3)
    return fits, pdfs, samples


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
    ELI_TICKS,
    PANELS,
    PDF_EDGES,
    PDF_YEARS,
    SHIFT,
    YEAR_EARLY,
    YEAR_LATE,
    fits,
    np,
    pdfs,
    plt,
    samples,
    src,
):
    ## PDFs, then one scatter plot per Niño index
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.5), layout="constrained")

    ## (a) the first window shaded, the others as lines
    axs[0].stairs(
        pdfs[0], PDF_EDGES, fill=True, alpha=0.1, color="k", label=PDF_YEARS[0]
    )
    axs[0].stairs(pdfs[1], PDF_EDGES, lw=2, label=PDF_YEARS[1])
    axs[0].stairs(pdfs[2], PDF_EDGES, lw=2, label=PDF_YEARS[2])

    ## (a) zero line and legend
    axs[0].axvline(0, ls="--", c="gray", lw=0.8)
    axs[0].legend()

    ## (a) ticks
    axs[0].set_xticks([-15, 0, 30])
    axs[0].set_yticks([0, 0.05])

    ## (a) labels
    axs[0].set_xlabel(r"$\Delta$ELI ($^{\circ}$lon)")
    axs[0].set_ylabel("Density")
    axs[0].set_title("(a) ELI PDF", loc="left")

    ## (b-d) scatter plots; the late window is shifted right by SHIFT
    _scatter_kwargs = dict(s=3, alpha=0.3)
    for _ax, (_name, _title), _letter in zip(axs[1:], PANELS.items(), "bcd"):
        _s0, _s1 = samples[YEAR_EARLY], samples[YEAR_LATE]

        ## points: early window, then late window (shifted)
        _ax.scatter(_s0["eli"], _s0[_name], **_scatter_kwargs)
        _ax.scatter(_s1["eli"] + SHIFT, _s1[_name], **_scatter_kwargs)

        ## ΔELI ranges of the fit curves, for each window
        _z0, _z1 = _s0["eli_fit"], _s1["eli_fit"]

        ## fits: early (dashed) on both clusters, late (solid) on its own
        _ax.plot(_z0, fits[YEAR_EARLY][_name](_z0), c="k", ls="--")
        _ax.plot(_z1 + SHIFT, fits[YEAR_EARLY][_name](_z1), c="k", ls="--")
        _ax.plot(_z1 + SHIFT, fits[YEAR_LATE][_name](_z1), c="k")

        ## reference lines at 0 and ±3.5 K
        for _t in [-3.5, 0, 3.5]:
            _ax.axhline(_t, ls="--", c="gray", lw=0.8)

        ## ticks at the same ΔELI values in each cluster
        _ticks = ELI_TICKS + [_t + SHIFT for _t in ELI_TICKS]
        _ax.set_xticks(_ticks, labels=ELI_TICKS + ELI_TICKS)
        _ax.set_xlabel(r"$\Delta$ELI ($^{\circ}$lon)")

        ## title; no y-ticks except on the last panel
        _ax.set_title(f"({_letter}) {_title}", loc="left")
        _ax.set_yticks([])

    ## (b-d) share limits; y-axis on the right of the last panel
    src.utils.set_lims(axs[1:])
    axs[-1].set_yticks([-3.5, 0, 3.5])

    ## y-label on the right
    axs[-1].yaxis.set_label_position("right")
    axs[-1].yaxis.tick_right()
    axs[-1].set_ylabel("K")

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "thesis_eli")

    fig
    return


if __name__ == "__main__":
    app.run()
