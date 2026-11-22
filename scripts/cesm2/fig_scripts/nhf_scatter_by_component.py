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
    # Heat flux components and relative SST

    Paper figure `fig:nhf_scatter_by_component` (`paper-outline.tex` in the ch3 Overleaf
    project; formerly `fig:nhf-scatter`). It splits the net heat flux in panels (b, c) of
    `fig:nhf_scatter` (`nhf_scatter.py`) into components.

    - **Columns:** December relative SST in the Niño 3.4 region ($T_{34}^*$) against the
      contribution to the $T_{34}$ tendency of the net surface heat flux, shortwave plus
      latent heat flux, shortwave only, and latent heat flux only ($Q_{34}$)
    - **Top row:** the 40-year window centered on 2010, with a cubic fit (solid) and the
      medians (triangle)
    - **Bottom row:** the window centered on 2080; the 2080 fit is dashed, and the 2010 fit
      and median are repeated for reference

    $T_{34}^*$ is the total Niño 3.4 index minus the tropical-mean SST (15°S-15°N), so it
    includes the forced change. $Q_{34}$ is an anomaly (ensemble mean removed), positive
    into the ocean, assuming a 50 m mixed layer.

    **Inputs** (under `DATA_FP/cesm`): the Niño 3.4 index chosen by `T34_INDEX` (`Th.nc`,
    SST-based, by default; or `Th_50.nc`/`Th_80.nc`), the tropical-mean SST
    (`trop_sst.nc`), net heat flux EOF scores (`consolidated/anom.nc`), and shortwave and
    latent heat flux EOF scores (`consolidated/flux_anom_with_clearsky.nc`).

    **Output:** `SAVE_FP/figures/nhf_scatter_by_component.pdf`

    Before this script, the figure came from the cell "Similar but for NHF components" in
    `stats-changes.py`, which used the mixed-layer index (`"ml80"`) for $T_{34}$ and didn't
    save the figure.
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

    ## labeled arrays
    import xarray as xr

    ## this project's modules (shared figure helpers, general utilities)
    import src.figure_utils as fu
    import src.utils

    ## plotting style used for all paper figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return fu, np, os, pathlib, plt, sns, src, xr


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

    ## windows shown (top and bottom rows)
    YEARS = [2010, 2080]

    ## heat flux shown in each column, and its title
    FLUX_TITLES = {"nhf": "Net", "sw_lhf": "SW+LHF", "fsns": "SW", "lhflx": "LHF"}

    ## degree of the polynomial fits
    FIT_DEGREE = 3
    return (
        FIT_DEGREE,
        FLUX_TITLES,
        H_ML,
        MONTH,
        T34_INDEX,
        TROP_SST,
        WINDOW_SIZE,
        YEARS,
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

    ## net heat flux and its shortwave and latent heat components, as contributions
    ## to the T34 tendency (anomalies, K/yr, positive into the ocean)
    Q34 = xr.Dataset()
    for _varname in ["nhf", "fsns", "lhflx"]:
        Q34[_varname] = fu.load_Q34(H=H_ML, varname=_varname)

    ## shortwave plus latent heat flux (the two largest feedbacks)
    Q34["sw_lhf"] = Q34["fsns"] + Q34["lhflx"]
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
def _(MONTH, Q34, T34_rel, WINDOW_SIZE, YEARS, fu):
    ## December values for each window, dims (member, time)
    T_dec = {}
    Q_dec = {}
    for _year in YEARS:
        T_dec[_year] = fu.get_window_month(T34_rel, _year, MONTH, WINDOW_SIZE)
        Q_dec[_year] = fu.get_window_month(Q34, _year, MONTH, WINDOW_SIZE)
    return Q_dec, T_dec


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Cubic fits

    Least-squares fit of $Q_{34} = e\,T^{*3}_{34} + c\,T^{*2}_{34} + b\,T^*_{34} + a$ over
    all members and years in a window, for each heat flux.
    """
    )
    return


@app.cell
def _(FIT_DEGREE, FLUX_TITLES, Q_dec, T_dec, YEARS, np):
    ## polynomial coefficients (highest power first) for each window and heat flux
    fit_coefs = {}
    for _year in YEARS:
        fit_coefs[_year] = {}
        _x = T_dec[_year].values.flatten()
        for _varname in FLUX_TITLES:
            _y = Q_dec[_year][_varname].values.flatten()
            fit_coefs[_year][_varname] = np.polyfit(_x, _y, deg=FIT_DEGREE)
    return (fit_coefs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(FLUX_TITLES, Q_dec, T_dec, YEARS, fit_coefs, fu, np, plt, sns, src):
    ## one row per window, one column per heat flux
    fig, axs = plt.subplots(2, 4, figsize=(9, 4.5), layout="constrained")

    ## colors for the 2010 and 2080 windows (first two palette colors)
    colors = sns.color_palette()[:2]
    year0, year1 = YEARS

    ## style of the fits, the medians, and the zero lines
    fit_kwargs = dict(c="k", lw=1.5, zorder=3)
    median_kwargs = dict(s=100, marker="v", zorder=11, edgecolor="w")
    ref_kwargs = dict(ls="--", c="gray", lw=0.8)

    ## values of T34* where the fits are drawn
    x_fit = np.linspace(-4, 2)

    for j, varname in enumerate(FLUX_TITLES):

        ## column title on the top row
        axs[0, j].set_title(FLUX_TITLES[varname])

        ## scatter of the samples in each window (one row each)
        for i, year in enumerate(YEARS):
            axs[i, j].scatter(T_dec[year], Q_dec[year][varname], s=1, color=colors[i])

        ## fits: 2010 in both rows (solid), 2080 in the bottom row (dashed)
        y0_fit = np.polyval(fit_coefs[year0][varname], x_fit)
        y1_fit = np.polyval(fit_coefs[year1][varname], x_fit)

        ## draw the fits
        axs[0, j].plot(x_fit, y0_fit, **fit_kwargs)
        axs[1, j].plot(x_fit, y0_fit, **fit_kwargs)
        axs[1, j].plot(x_fit, y1_fit, ls="--", **fit_kwargs)

        ## medians (triangles): 2010 in both rows, 2080 in the bottom row
        for ax, year, c in [
            (axs[0, j], year0, colors[0]),
            (axs[1, j], year0, colors[0]),
            (axs[1, j], year1, colors[1]),
        ]:
            ax.scatter(
                T_dec[year].median(),
                Q_dec[year][varname].median(),
                color=c,
                **median_kwargs,
            )

    ## same limits in all panels, so the fluxes can be compared
    src.utils.set_lims(axs.flatten())

    ## zero lines; no ticks except on the outer panels (set below)
    for ax in axs.flatten():
        ax.axvline(0, **ref_kwargs)
        ax.axhline(0, **ref_kwargs)

        ## clear the ticks
        ax.set_xticks([])
        ax.set_yticks([])

    ## x ticks and labels on the bottom row
    for ax in axs[1]:
        ax.set_xticks([-4, -2, 0, 2])
        ax.set_xlabel(r"$T_{34}^*$ (K)")

    ## y ticks and labels on the left column, naming the window of each row
    for ax, year in zip(axs[:, 0], YEARS):
        ax.set_yticks([-10, -5, 0, 5])
        ax.set_ylabel(f"{year}\n" + r"$Q_{34}$ (K yr$^{-1}$)")

    ## save to SAVE_FP/figures
    fu.save_fig(fig, "nhf_scatter_by_component")

    fig
    return


if __name__ == "__main__":
    app.run()
