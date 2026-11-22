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
    # Heat flux damping and relative SST

    Paper figure `fig:nhf_scatter` (`paper-outline.tex` in the ch3 Overleaf project); also
    the thesis figure `nhf_rel-sst_triple` (`chapter_enso-change.tex`).

    - **(a)** PDFs of December relative SST in the Niño 3.4 region ($T_{34}^*$) in three
      40-year windows (labeled by their center year). Triangles mark the medians for the
      2010 and 2080 windows.
    - **(b)** December $T_{34}^*$ against the net surface heat flux contribution to the
      $T_{34}$ tendency ($Q_{34}$), for the 2010 window, with an exponential fit (solid)
      and the medians (triangle).
    - **(c)** Same as (b), for the 2080 window; the 2080 fit is dashed and the 2010 fit and
      median are repeated for reference.

    The slope of a fit is the heat flux contribution to the $T_{34}$ growth rate (yr$^{-1}$).

    $T_{34}^*$ is the total Niño 3.4 index minus the tropical-mean SST (15°S-15°N), so it
    includes the forced change. $Q_{34}$ is an anomaly (ensemble mean removed), assuming a
    50 m mixed layer.

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX` (`Th.nc`,
    SST-based, by default; or `Th_50.nc`/`Th_80.nc`), the tropical-mean SST
    (`trop_sst.nc`), and net heat flux EOF scores (`consolidated/anom.nc`).

    **Output:** `SAVE_FP/figures/nhf_scatter.pdf`

    Before this script, the figure came from the `nhf_rel-sst_triple` cell of
    `stats-changes.py`, which used the mixed-layer index (`"ml80"`) for $T_{34}$. The fit
    function there had a fourth parameter that it didn't use; it's dropped here.
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
    ## plotting and arrays
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    ## file paths
    import os
    import pathlib

    ## curve fitting and labeled arrays
    import scipy.optimize
    import xarray as xr

    ## this project's modules (shared figure helpers, general utilities)
    import src.figure_utils as fu
    import src.utils

    ## plotting style used for all paper figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return fu, np, os, pathlib, plt, scipy, sns, src, xr


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

    ## tropical-mean SST subtracted to get relative SST (15S-15N)
    TROP_SST = "trop_sst_15"

    ## mixed-layer depth (m) for converting heat flux to a temperature tendency
    H_ML = 50

    ## window length (months) and the calendar month shown
    WINDOW_SIZE = 480
    MONTH = 12

    ## windows in the PDFs (a) and in the scatter plots (b, c)
    PDF_YEARS = [1870, 2010, 2080]
    SCATTER_YEARS = [2010, 2080]

    ## histogram bin edges (K) for the PDFs
    PDF_EDGES = np.linspace(-5, 3, 20)
    return (
        H_ML,
        MONTH,
        PDF_EDGES,
        PDF_YEARS,
        SCATTER_YEARS,
        T34_INDEX,
        TROP_SST,
        WINDOW_SIZE,
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
def _(H_ML, T34_INDEX, TROP_SST, fu, os, pathlib, xr):
    ## total Niño 3.4 index and tropical-mean SST, dims (member, time)
    T34 = fu.load_T34_total(index=T34_INDEX)
    trop_sst_fp = pathlib.Path(os.environ["DATA_FP"], "cesm", "trop_sst.nc")
    trop_sst = xr.open_dataset(trop_sst_fp)[TROP_SST]

    ## relative SST in Niño 3.4
    T34_rel = T34 - trop_sst

    ## heat flux contribution to the T34 tendency (anomaly, K/yr)
    Q34 = fu.load_Q34(H=H_ML)
    return Q34, T34_rel


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## December values in each window
    """
    )
    return


@app.cell
def _(MONTH, PDF_YEARS, Q34, T34_rel, WINDOW_SIZE, fu):
    def get_month(x, year):
        """
        Values in one calendar month (MONTH) within one window.

        Args:
            - x (xr.DataArray): monthly data, dims (member, time)
            - year (int): window center

        Returns:
            - x_month (xr.DataArray): dims (member, time), one value per year
        """

        ## cut out the window, then keep one calendar month
        window = fu.get_window(x, year, WINDOW_SIZE)
        x_month = window.sel(time=window.time.dt.month == MONTH)

        return x_month.transpose("member", "time")

    ## December values for each window shown (PDF_YEARS includes SCATTER_YEARS)
    T_dec = {}
    Q_dec = {}
    for _year in PDF_YEARS:
        T_dec[_year] = get_month(T34_rel, _year)
        Q_dec[_year] = get_month(Q34, _year)
    return Q_dec, T_dec


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
    ## (b, c) Exponential fits

    Least-squares fit of $Q_{34} = c\,e^{a T_{34}^*} + b$ over all members and years in a
    window. Its slope, $ac\,e^{a T_{34}^*}$, is the heat flux contribution to the growth
    rate of $T_{34}$, and it changes with $T_{34}^*$.
    """
    )
    return


@app.cell
def _(SCATTER_YEARS, Q_dec, T_dec, np, scipy):
    def exp_curve(x, a, b, c):
        """
        Exponential curve, c * exp(a * x) + b.

        Args:
            - x (np.ndarray): relative SST (K)
            - a (float): growth rate of the exponential (1/K)
            - b (float): offset (K/yr)
            - c (float): amplitude (K/yr)

        Returns:
            - y (np.ndarray): heat flux contribution (K/yr)
        """
        return c * np.exp(a * x) + b

    ## fitted parameters (a, b, c) for each window shown in the scatter plots
    fit_params = {}
    for _year in SCATTER_YEARS:
        _x = T_dec[_year].values.flatten()
        _y = Q_dec[_year].values.flatten()
        fit_params[_year], _ = scipy.optimize.curve_fit(exp_curve, _x, _y)
    return exp_curve, fit_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting functions
    """
    )
    return


@app.cell
def _(PDF_EDGES, PDF_YEARS, SCATTER_YEARS, T_dec):
    def plot_pdfs(ax, pdfs, colors):
        """
        Panel (a): PDFs of December relative SST, with medians.

        Args:
            - ax (matplotlib Axes): where to draw
            - pdfs (list of np.ndarray): one normalized histogram per year in PDF_YEARS
            - colors (list): colors for the windows in SCATTER_YEARS

        Returns:
            - None
        """

        ## first window shaded gray
        ax.stairs(
            pdfs[0], PDF_EDGES, fill=True, alpha=0.1, color="k", label=PDF_YEARS[0]
        )

        ## later windows as colored outlines
        ax.stairs(pdfs[1], PDF_EDGES, lw=2, label=PDF_YEARS[1])
        ax.stairs(pdfs[2], PDF_EDGES, lw=2, label=PDF_YEARS[2])

        ## triangles at the medians, near the bottom of the axes
        for year, c in zip(SCATTER_YEARS, colors):
            median = T_dec[year].median()
            ax.scatter(median, 0.015, color=c, s=100, marker="v", zorder=11)

        ## y axis
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_ylabel("Prob. density")

        ## legend and title
        ax.legend(prop=dict(size=11), frameon=False, loc=(0.65, 0.65))
        ax.set_title("(a) PDFs", loc="left")

        return

    return (plot_pdfs,)


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
    Q_dec,
    SCATTER_YEARS,
    T_dec,
    exp_curve,
    fit_params,
    fu,
    np,
    pdfs,
    plot_pdfs,
    plt,
    sns,
    src,
):
    ## three panels in a row
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.75), layout="constrained")

    ## colors for the 2010 and 2080 windows (first two palette colors)
    colors = sns.color_palette()[:2]
    year0, year1 = SCATTER_YEARS

    ## (a) PDFs
    plot_pdfs(axs[0], pdfs, colors)

    ## (b, c) scatter of the samples in each window
    for ax, year, c in zip(axs[1:], SCATTER_YEARS, colors):
        ax.scatter(T_dec[year], Q_dec[year], s=1, color=c)

    ## titles for (b, c)
    axs[1].set_title(f"(b) {year0}", loc="left")
    axs[2].set_title(f"(c) {year1}", loc="left")

    ## fits: 2010 in both panels (solid), 2080 in (c) (dashed)
    fit_kwargs = dict(c="k", lw=1.5, zorder=3)

    ## evaluate the fits over the range of T34*
    x_fit = np.linspace(-4, 2)
    y0_fit = exp_curve(x_fit, *fit_params[year0])
    y1_fit = exp_curve(x_fit, *fit_params[year1])

    ## draw the fits
    axs[1].plot(x_fit, y0_fit, label=f"{year0} fit", **fit_kwargs)
    axs[2].plot(x_fit, y0_fit, **fit_kwargs)
    axs[2].plot(x_fit, y1_fit, ls="--", label=f"{year1} fit", **fit_kwargs)

    ## medians (triangles): 2010 in both panels, 2080 in (c)
    median_kwargs = dict(s=100, marker="v", zorder=11, edgecolor="w")
    for ax, year, c, label in [
        (axs[1], year0, colors[0], f"{year0} median"),
        (axs[2], year0, colors[0], None),
        (axs[2], year1, colors[1], f"{year1} median"),
    ]:
        ax.scatter(
            T_dec[year].median(),
            Q_dec[year].median(),
            color=c,
            label=label,
            **median_kwargs,
        )

    ## zero lines (horizontal in the scatter plots only)
    ref_kwargs = dict(ls="--", c="gray", lw=0.8)
    for ax in axs[1:]:
        ax.axhline(0, **ref_kwargs)

    ## same limits for (b) and (c)
    src.utils.set_lims(axs[1:])

    ## x axes: zero line, ticks, label
    for ax in axs:
        ax.axvline(0, **ref_kwargs)
        ax.set_xticks([-4, -2, 0, 2])
        ax.set_xlabel(r"$T_{34}^*$ (K)")

    ## no y ticks on (b)
    axs[1].set_yticks([])

    ## y axis of (c) on the right
    axs[2].set_ylabel(r"$Q_{34}$ (K yr$^{-1})$")
    axs[2].yaxis.set_label_position("right")
    axs[2].yaxis.tick_right()

    ## no tick marks on (c), only labels
    axs[2].tick_params(axis="y", which="both", length=0)

    ## legends
    for ax in axs[1:]:
        ax.legend(prop=dict(size=11), frameon=False)

    ## save to SAVE_FP/figures
    fu.save_fig(fig, "nhf_scatter")

    fig
    return


if __name__ == "__main__":
    app.run()
