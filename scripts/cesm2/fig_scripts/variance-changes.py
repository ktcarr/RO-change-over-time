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
    # Asymmetric changes in ENSO variance

    Paper figure `fig:variance-changes` (`paper-outline.tex` in the ch3 Overleaf project).

    - **(a)** PDFs of the DJF Niño 3.4 index ($T_{34}$) in three 40-year windows (labeled by
      their center year), measured from the median of each window
    - **(b)** Contribution of warm anomalies (above the 75th percentile) to the variance of
      $T_{34}$ over time: ensemble median (thick) and 10th/90th percentiles (dashed).
      The black line is half of the total, for reference.
    - **(c)** Same as (b), for cold anomalies (below the 25th percentile)

    In (b, c), "variance" is the mean square about the median rather than the mean, so the
    warm and cold contributions add up to the total.

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX`: `Th_anom.nc`
    (SST-based, the default; from `preprocess/preprocess_cesm.py`) or `Th_50.nc`/`Th_80.nc`
    (from `preprocess/merge_Th_3D.py`).

    **Output:** `SAVE_FP/figures/variance-changes.pdf`

    Before this script, (a) came from the `pdf-over-time_bounds` cell of `stats-changes.py`,
    and (b, c) from the `variance-contrib` cell of `timeseries.py` (now in `explore/`).
    All panels now use the same DJF values. The old (a) also kept each window's partial first
    and last seasons (Jan-Feb only and Dec only), so its PDFs differ very slightly.
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
    ## plotting and data libraries
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    ## this project's modules (shared figure helpers, general utilities)
    import src.figure_utils as fu
    import src.utils

    ## plotting style used for all paper figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return fu, np, plt, src


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
    ## Niño 3.4 index for ENSO statistics: "sst" (SST-based, used by the RO fits),
    ## or "ml50"/"ml80" (mixed-layer temperature over the top 50/80 m)
    T34_INDEX = "sst"

    ## window length and stride (months) for changes over time
    WINDOW_SIZE = 480
    STRIDE = 60

    ## windows (center years) shown in the PDFs (a)
    PDF_YEARS = [1870, 2010, 2080]

    ## histogram bin edges (K) for the PDFs
    PDF_EDGES = np.linspace(-4.5, 4.5, 20)

    ## percentiles separating warm and cold anomalies in (b, c)
    Q_COLD = 0.25
    Q_WARM = 0.75
    return PDF_EDGES, PDF_YEARS, Q_COLD, Q_WARM, STRIDE, T34_INDEX, WINDOW_SIZE


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data
    """
    )
    return


@app.cell
def _(STRIDE, T34_INDEX, WINDOW_SIZE, fu):
    ## Niño 3.4 anomalies (ensemble mean removed), cut into windows
    T34_windowed = fu.load_T34_windowed(
        index=T34_INDEX, window_size=WINDOW_SIZE, stride=STRIDE
    )

    ## DJF means, measured from the median of each window (used in all panels)
    T = fu.get_djf_anom(T34_windowed)
    return (T,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (a) PDFs
    """
    )
    return


@app.cell
def _(PDF_EDGES, PDF_YEARS, T, src):
    ## normalized histogram for each window shown, pooling members and years
    pdfs = []
    for year in PDF_YEARS:
        T_year = T.sel(year=year).values.flatten()
        pdf, _ = src.utils.get_empirical_pdf(T_year, PDF_EDGES)
        pdfs.append(pdf)
    return (pdfs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (b, c) Warm and cold contributions to variance

    For DJF anomalies $T$ measured from the median, the mean square
    $\frac{1}{N}\sum T^2$ splits into a warm part (samples above the 75th percentile),
    a cold part (below the 25th), and a neutral part. Percentiles are computed over all
    members in each window; the sums are over the years in each window, for each member.
    """
    )
    return


@app.cell
def _(Q_COLD, Q_WARM, T, np):
    ## warm/cold thresholds: percentiles over members and time in each window
    q_cold = T.quantile(q=Q_COLD, dim=["member", "time"]).drop_vars("quantile")
    q_warm = T.quantile(q=Q_WARM, dim=["member", "time"]).drop_vars("quantile")

    ## number of DJF samples per member in each window
    N = (~np.isnan(T)).sum("time")

    ## mean square about the median: total, and from warm and cold samples only
    var_total = (T**2).sum("time") / N
    var_warm = (T.where(T > q_warm) ** 2).sum("time") / N
    var_cold = (T.where(T < q_cold) ** 2).sum("time") / N
    return var_cold, var_total, var_warm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting functions
    """
    )
    return


@app.cell
def _(PDF_EDGES, PDF_YEARS):
    def plot_pdfs(ax, pdfs):
        """
        Panel (a): PDFs of DJF Niño 3.4 in three windows.

        Args:
            - ax (matplotlib Axes): where to draw
            - pdfs (list of np.ndarray): one normalized histogram per year in PDF_YEARS

        Returns:
            - None
        """

        ## first window shaded gray; later windows as colored outlines
        ax.stairs(
            pdfs[0], PDF_EDGES, fill=True, alpha=0.1, color="k", label=PDF_YEARS[0]
        )
        ax.stairs(pdfs[1], PDF_EDGES, lw=2, label=PDF_YEARS[1])
        ax.stairs(pdfs[2], PDF_EDGES, lw=2, label=PDF_YEARS[2])

        ## dashed line at zero (the median)
        ax.axvline(ls="--", c="gray", lw=0.8)

        ## axis range and ticks
        ax.set_xlim([-4.5, 4.5])
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([0, 0.1, 0.2, 0.3])

        ## axis labels
        ax.set_xlabel(r"$T_{34}$ (K)")
        ax.set_ylabel("Prob. density")

        ## legend and title
        ax.legend(prop=dict(size=10), frameon=False, loc=(-0.03, 0.65))
        ax.set_title("(a) PDFs", loc="left")

        return

    def plot_contribution(ax, var_part, var_total, color, title, year_tick):
        """
        Panel (b) or (c): warm or cold contribution to variance over time.

        Args:
            - ax (matplotlib Axes): where to draw
            - var_part (xr.DataArray): warm or cold contribution, dims (member, year)
            - var_total (xr.DataArray): total mean square, dims (member, year)
            - color (str): color for the contribution
            - title (str): panel title
            - year_tick (int): year to mark with a vertical line and tick

        Returns:
            - None
        """

        ## reference: half of the total (ensemble median)
        half_total = 0.5 * var_total.median("member")
        ax.plot(half_total.year, half_total, c="k", label="1/2 total")

        ## ensemble median (thick) and 10th/90th percentiles (dashed)
        for q, lw, ls, label in [
            (0.5, 2, "-", "Median"),
            (0.1, 1, "--", "10-90%"),
            (0.9, 1, "--", None),
        ]:
            var_q = var_part.quantile(q=q, dim="member")
            ax.plot(var_q.year, var_q, c=color, lw=lw, ls=ls, label=label)

        ## legend on a white background, so lines don't show through it
        legend = ax.legend(prop=dict(size=10), loc=(-0.1, 0.68), frameon=True)
        legend.get_frame().set_color("white")

        ## reference lines at 1 K^2 and at year_tick
        ax.axhline(1, lw=0.5, c="gray")
        ax.axvline(year_tick, lw=0.5, c="gray")

        ## axes and title
        ax.set_xticks([1870, year_tick])
        ax.set_xlabel("Year")
        ax.set_title(title, loc="left")

        return

    return plot_contribution, plot_pdfs


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
    fu,
    pdfs,
    plot_contribution,
    plot_pdfs,
    plt,
    src,
    var_cold,
    var_total,
    var_warm,
):
    ## three panels in a row
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.9), layout="constrained")

    ## (a) PDFs
    plot_pdfs(axs[0], pdfs)

    ## (b) warm and (c) cold contributions ("r" and "b" are the palette's red and blue)
    plot_contribution(
        axs[1], var_warm, var_total, "r", "(b) El Niño contribution", year_tick=2010
    )
    plot_contribution(
        axs[2], var_cold, var_total, "b", "(c) La Niña contribution", year_tick=2060
    )

    ## same axis limits for (b) and (c)
    src.utils.set_lims(axs[1:])

    ## y ticks on the right of (c) only
    axs[1].set_yticks([])
    axs[2].set_yticks([0.5, 1, 1.5, 2])
    axs[2].yaxis.tick_right()

    ## y label on the right of (c)
    axs[2].yaxis.set_label_position("right")
    axs[2].set_ylabel(r"Variance (K$^2$)")

    ## no tick marks on (c), only labels
    axs[2].tick_params(axis="y", which="both", length=0)

    ## save to SAVE_FP/figures
    fu.save_fig(fig, "variance-changes")

    fig
    return


if __name__ == "__main__":
    app.run()
