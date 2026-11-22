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
    # Niño 3.4 distribution and strong-event intensity over time (thesis)

    Thesis figure `fig:pdf-over-time` (`chapter_enso-change.tex`). DJF-mean Niño 3.4
    anomalies (ensemble mean removed, then measured from the median of each 40-year
    window, pooling members and years):

    - **(a)** PDFs (normalized histograms) for three windows, labeled by their centers
    - **(b)** intensity of strong El Niños in each window: for each member, the mean of
      the DJF values above that member's 95th percentile (the 2 warmest of the 39 DJF
      seasons). Thick red: ensemble median; thin red: 10th and 90th percentiles over
      members. Dashed black: the ensemble median of each member's mean intensity of
      strong El Niños and strong La Niñas (the same line in (b) and (c)).
    - **(c)** same as (b), for strong La Niñas (below the 5th percentile; sign reversed)

    Windows are 40 years (480 months), every 5 years, labeled by their centers
    (1870-2080); the gray line marks 2010.

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX`:
    `Th_anom.nc` (SST, the default), or `Th_50.nc` / `Th_80.nc` (mixed-layer
    temperature).

    **Output:** `SAVE_FP/figures/thesis_pdf-over-time.pdf`

    Before this script, the figure was `pdf-over-time_bounds`, made by
    `stats-changes.py` (cells under "similar, but with bounds"). Changes:

    - The index is set by `T34_INDEX`, the SST-based index by default, as in the paper
      figures. The thesis figure was made with the SST-based index too (the old code
      reproduces it with that index), but the notebook's committed code had since been
      switched to the top-80 m mixed-layer temperature (`T34_INDEX = "ml80"`).
    - DJF means use full seasons only, as in `fig:variance-changes` (a) (before, each
      window also had a Jan-Feb-only and a Dec-only "season", so 41 values per member,
      which is why the caption says 41-year periods). In (b, c), anomalies are measured
      from the median of the DJF means (before, from the median of the DJF months).
    - The dashed line is the ensemble median of each member's average intensity, as the
      caption says (before, the average of the El Niño and La Niña medians).
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

    ## numerics, plot style, labeled arrays
    import numpy as np
    import seaborn as sns
    import xarray as xr

    ## this project's modules: loading data, plotting
    import src.data
    import src.plot

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all figures; color_codes (on by default) makes "r" and
    ## "b" the colorblind palette's red and blue
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return np, plt, src, xr


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

    ## window length and stride (months)
    WINDOW_SIZE = 480
    STRIDE = 60

    ## windows (center years) shown in the PDFs (a), also the time ticks of (b, c)
    PDF_YEARS = [1870, 2010, 2080]

    ## histogram bin edges (K) for the PDFs
    PDF_EDGES = np.linspace(-4.5, 4.5, 20)

    ## percentile defining strong El Niños (and 1 - Q for strong La Niñas)
    Q = 0.95

    ## percentiles over members shown in (b, c): median (thick), and bounds (thin)
    MEMBER_QUANTILES = [0.5, 0.1, 0.9]

    ## year marked with a line in (b, c)
    YEAR_LINE = 2010
    return (
        MEMBER_QUANTILES,
        PDF_EDGES,
        PDF_YEARS,
        Q,
        STRIDE,
        T34_INDEX,
        WINDOW_SIZE,
        YEAR_LINE,
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
def _(STRIDE, T34_INDEX, WINDOW_SIZE, src):
    ## Niño 3.4 anomalies (ensemble mean removed), cut into windows
    _T34_windowed = src.data.load_T34_windowed(
        index=T34_INDEX, window_size=WINDOW_SIZE, stride=STRIDE
    )

    ## DJF means (full seasons), measured from the median of each window;
    ## dims (member, year, time)
    T = src.data.get_djf_anom(_T34_windowed)
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
    for _year in PDF_YEARS:
        _T_year = T.sel(year=_year).values.flatten()
        _pdf, _ = src.utils.get_empirical_pdf(_T_year, PDF_EDGES)
        pdfs.append(_pdf)
    return (pdfs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (b, c) Strong-event intensity

    For each member and window: the mean of the DJF values beyond the member's 95th
    (El Niño) or 5th (La Niña) percentile; then percentiles over members.
    """
    )
    return


@app.cell
def _(MEMBER_QUANTILES, Q, T, xr):
    ## each member's percentiles in each window; dims (member, year)
    _q_warm = T.quantile(q=Q, dim="time").drop_vars("quantile")
    _q_cold = T.quantile(q=1 - Q, dim="time").drop_vars("quantile")

    ## mean intensity of strong events (La Niña's sign reversed)
    intensity = xr.Dataset()
    intensity["warm"] = T.where(T > _q_warm).mean("time")
    intensity["cold"] = -T.where(T < _q_cold).mean("time")

    ## average of the two, for each member
    intensity["both"] = 0.5 * (intensity["warm"] + intensity["cold"])

    ## percentiles over members; dims (quantile, year)
    intensity_stats = intensity.quantile(q=MEMBER_QUANTILES, dim="member")
    return (intensity_stats,)


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
    MEMBER_QUANTILES,
    PDF_EDGES,
    PDF_YEARS,
    YEAR_LINE,
    intensity_stats,
    pdfs,
    plt,
    src,
):
    ## PDFs, then strong El Niño and La Niña intensity
    fig, axs = plt.subplots(1, 3, figsize=(8.5, 2.75), layout="constrained")

    ## (a) the first window shaded, the others as lines
    axs[0].stairs(
        pdfs[0], PDF_EDGES, fill=True, alpha=0.1, color="k", label=PDF_YEARS[0]
    )
    axs[0].stairs(pdfs[1], PDF_EDGES, lw=2, label=PDF_YEARS[1])
    axs[0].stairs(pdfs[2], PDF_EDGES, lw=2, label=PDF_YEARS[2])

    ## (a) zero line and legend
    axs[0].axvline(0, ls="--", c="gray", lw=0.8)
    axs[0].legend(prop=dict(size=10), frameon=False, loc=(-0.03, 0.65))

    ## (a) axes
    axs[0].set_xlim(PDF_EDGES[[0, -1]])
    axs[0].set_xticks([-3, 0, 3])
    axs[0].set_yticks([0, 0.1, 0.2, 0.3])

    ## (a) labels
    axs[0].set_xlabel(r"$T_{34}$ (K)")
    axs[0].set_ylabel("Prob. density")
    axs[0].set_title("(a) Niño 3.4 PDF", loc="left")

    ## (b, c) median over members (thick) and bounds (thin), and the shared reference
    _rows = [
        (axs[1], "warm", "r", "(b) Strong El Niño"),
        (axs[2], "cold", "b", "(c) Strong La Niña"),
    ]
    _both = intensity_stats["both"].sel(quantile=0.5)
    for _ax, _name, _color, _title in _rows:
        _ax.plot(_both.year, _both, c="k", ls="--")
        for _q in MEMBER_QUANTILES:

            ## thick line for the median
            _lw = 2.5 if _q == 0.5 else 1
            _line = intensity_stats[_name].sel(quantile=_q)
            _ax.plot(_line.year, _line, lw=_lw, c=_color)

        ## reference year, ticks, title
        _ax.axvline(YEAR_LINE, ls="--", lw=0.8, c="gray")
        _ax.set_xticks(PDF_YEARS)
        _ax.set_title(_title, loc="left")

    ## (b, c) share limits; no y-ticks on (b)
    src.utils.set_lims(axs[1:])
    axs[1].set_yticks([])
    axs[1].set_xlabel("Year")

    ## y-axis of (c) on the right
    axs[2].yaxis.tick_right()
    axs[2].yaxis.set_label_position("right")
    axs[2].set_ylabel("K")

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "thesis_pdf-over-time")

    fig
    return


if __name__ == "__main__":
    app.run()
