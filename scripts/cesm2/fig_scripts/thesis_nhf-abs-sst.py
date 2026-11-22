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
    # Heat flux and absolute SST (thesis)

    Thesis appendix figure `fig:nhf-abs-sst` (`appendix_enso-change.tex`); the same as
    `fig:nhf-rel-sst` (made by `nhf_scatter.py`), but for absolute instead of relative
    SST, and with the heat flux in W m$^{-2}$.

    - **(a)** PDFs of December Niño 3.4 SST in three 40-year windows (labeled by their
      center year)
    - **(b)** December Niño 3.4 SST against the net surface heat flux anomaly averaged
      over Niño 3.4 (positive into the ocean), for the 2010 window
    - **(c)** Same as (b), for the 2080 window

    The SST is the total index (forced change included). The heat flux is an anomaly
    (ensemble mean removed); the thesis caption doesn't say so.

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX` (`Th.nc`,
    SST-based, the default; or `Th_50.nc` / `Th_80.nc`), and the net heat flux EOF scores
    (`consolidated/anom.nc`).

    **Output:** `SAVE_FP/figures/thesis_nhf-abs-sst.pdf`

    Before this script, the figure was `nhf_abs-sst_triple`, made by
    `explore/stats-changes.py` (cells under "same, but with total SST"). Changes:

    - The index is set by `T34_INDEX`, the SST-based index by default. The thesis
      figure was made with the SST-based index (the old code reproduces it with that
      index), but the notebook's committed code had since been switched to the top-80 m
      mixed-layer temperature, as for `thesis_skew-over-time.py`.
    - The heat flux unit is written upright, "W m$^{-2}$" (before, in math italics).
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
    ## plotting, arrays, plot style
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    ## this project's modules: loading data, plotting
    import src.data
    import src.plot

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return np, plt, sns, src


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
    ## Niño 3.4 index: "sst" (SST-based, used by the RO fits), or "ml50"/"ml80"
    ## (mixed-layer temperature over the top 50/80 m)
    T34_INDEX = "sst"

    ## window length (months) and the calendar month shown
    WINDOW_SIZE = 480
    MONTH = 12

    ## windows in the PDFs (a) and in the scatter plots (b, c)
    PDF_YEARS = [1870, 2010, 2080]
    SCATTER_YEARS = [2010, 2080]

    ## histogram bin edges (degrees C) for the PDFs
    PDF_EDGES = np.linspace(22, 35, 20)

    ## SST range and ticks for each panel (a, b, c)
    XLIMS = [[None, 34.5], [22, 32], [25, 35]]
    XTICKS = [[23, 28, 33], [24, 31], [27, 34]]
    return (
        MONTH,
        PDF_EDGES,
        PDF_YEARS,
        SCATTER_YEARS,
        T34_INDEX,
        WINDOW_SIZE,
        XLIMS,
        XTICKS,
    )


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
    ## total Niño 3.4 index (degrees C); dims (member, time)
    T34 = src.data.load_T34_total(index=T34_INDEX)

    ## net heat flux anomaly over Niño 3.4 (W/m^2, positive into the ocean)
    F34 = src.data.load_F(varname="nhf", region="34")
    return F34, T34


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## December values in each window
    """
    )
    return


@app.cell
def _(F34, MONTH, PDF_YEARS, T34, WINDOW_SIZE, src):
    ## December values for each window shown (PDF_YEARS includes SCATTER_YEARS),
    ## dims (member, time)
    T_dec = {}
    F_dec = {}
    for _year in PDF_YEARS:
        T_dec[_year] = src.data.get_window_month(T34, _year, MONTH, WINDOW_SIZE)
        F_dec[_year] = src.data.get_window_month(F34, _year, MONTH, WINDOW_SIZE)
    return F_dec, T_dec


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (a) PDFs
    """
    )
    return


@app.cell
def _(PDF_EDGES, PDF_YEARS, T_dec, src):
    ## normalized histogram for each window, pooling members and years
    pdfs = []
    for _year in PDF_YEARS:
        _pdf, _ = src.utils.get_empirical_pdf(T_dec[_year].values.flatten(), PDF_EDGES)
        pdfs.append(_pdf)
    return (pdfs,)


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
    F_dec,
    PDF_EDGES,
    PDF_YEARS,
    SCATTER_YEARS,
    T_dec,
    XLIMS,
    XTICKS,
    pdfs,
    plt,
    sns,
    src,
):
    ## three panels in a row
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 3), layout="constrained")

    ## colors for the 2010 and 2080 windows (first two palette colors)
    _colors = sns.color_palette()[:2]

    ## (a) PDFs: the first window shaded gray, the later ones as colored outlines
    axs[0].stairs(
        pdfs[0], PDF_EDGES, fill=True, alpha=0.1, color="k", label=PDF_YEARS[0]
    )
    axs[0].stairs(pdfs[1], PDF_EDGES, lw=2, label=PDF_YEARS[1])
    axs[0].stairs(pdfs[2], PDF_EDGES, lw=2, label=PDF_YEARS[2])

    ## (a) y axis and legend
    axs[0].set_yticks([0, 0.1, 0.2, 0.3])
    axs[0].set_ylabel("Prob. density")
    axs[0].legend()

    ## (b, c) scatter of the samples in each window, with a zero line
    for _ax, _year, _c in zip(axs[1:], SCATTER_YEARS, _colors):
        _ax.scatter(T_dec[_year], F_dec[_year], s=1, color=_c)
        _ax.axhline(0, ls="--", c="gray", lw=0.8)

    ## same heat flux range for (b) and (c) (the SST ranges are set below)
    src.utils.set_lims(axs[1:])

    ## titles
    axs[0].set_title("(a) PDFs", loc="left")
    axs[1].set_title(f"(b) {SCATTER_YEARS[0]}", loc="left")
    axs[2].set_title(f"(c) {SCATTER_YEARS[1]}", loc="left")

    ## SST axes: range, ticks, label
    for _ax, _xlim, _xticks in zip(axs, XLIMS, XTICKS):
        _ax.set_xlim(_xlim)
        _ax.set_xticks(_xticks)
        _ax.set_xlabel(r"$^{\circ}$C")

    ## no y ticks on (b)
    axs[1].set_yticks([])

    ## y axis of (c) on the right
    axs[2].set_ylabel(r"W m$^{-2}$")
    axs[2].yaxis.set_label_position("right")
    axs[2].yaxis.tick_right()

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "thesis_nhf-abs-sst")

    fig
    return


if __name__ == "__main__":
    app.run()
