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
    # Seasonality of changes in the RO's parameters

    Paper figure `fig:RO-experiments_param_hov` (`paper-outline.tex` in the ch3 Overleaf
    project). The nonlinear recharge oscillator (RO) is fitted to CESM2's Niño 3 index
    ($T$) and thermocline depth ($h$) in each 40-year window (centered every 10 years),
    separately for each calendar month. Each panel shows a parameter's change from the
    first window (1870), by calendar month (x) and window (y):

    - **(a)** the SST growth rate $R$, **(b)** its ocean part $R_o$, and **(c)** the
      surface heat flux feedback's part $-\alpha$ ($R = R_o - \alpha$), all linearized
      around $T = 0$ (the coefficients of $T$; the black curve in `fig:gr-over-time`)
    - **(d)** the standard deviation of the stochastic forcing ("noise") of the $T$
      equation

    $\alpha$ comes from a cubic fit of the net surface heat flux over Niño 3 (as a warming
    rate of a 50 m mixed layer) to $T$, as in `fig:RO-experiments_Ro`.

    **Inputs** (under `DATA_FP/cesm`): `Th_anom.nc` (Niño 3 index and thermocline
    depth), `consolidated/anom.nc` (net surface heat flux).

    **Output:** `SAVE_FP/figures/RO-experiments_param_hov.pdf`

    Before this script, the figure came from the "Cleaned up plot" cell of
    `explore/RO-change-over-time_NL.py` (saved as `param-hov-...`). Changes:

    - The noise is in K yr$^{-1}$ (the RO's non-dimensional noise times the standard
      deviation of the Niño 3 index); before, the non-dimensional values were plotted
      with a K$^{-1}$ yr$^{-1}$ label.
    - Each colorbar belongs to its own panels (before, both colorbars were drawn from
      the noise panel's contours), and panels are labeled (a)-(d).
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
    ## plotting and plotting style
    import matplotlib.pyplot as plt
    import seaborn as sns

    ## registers the "cmo.balance" colormap with matplotlib
    import cmocean  # noqa: F401

    ## this project's modules: plotting, RO fits
    import src.plot
    import src.ro

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all paper figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return plt, src


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
    ## window that the changes are measured from (the first window)
    REF_YEAR = 1870

    ## mixed-layer depth (m) for converting the heat flux to a warming rate
    H_ML = 50

    ## years marked on the y-axis, and the year marked with a line
    YTICKS = [1870, 1990, 2080]
    YLINE = 1990

    ## color scale limits for the growth rates (yr^-1) and the noise (K yr^-1)
    AMP_R = 1.0
    AMP_NOISE = 0.6
    return AMP_NOISE, AMP_R, H_ML, REF_YEAR, YLINE, YTICKS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fit the RO and the heat flux feedback

    `src.ro.FIT_KWARGS` holds the fit settings shared by all RO figures.
    """
    )
    return


@app.cell
def _(H_ML, src):
    ## windowed, non-dimensional T (Niño 3) and h_hat, dims (member, year, time)
    data, T_scale = src.ro.load_ro_data(t_var="T_3")

    ## fit the RO in each window
    model = src.ro.get_model()
    fits = src.ro.fit_over_time(data[src.ro.VARNAMES], model)

    ## net surface heat flux over Niño 3 in the same units and windows as dT/dt
    Q = src.ro.load_ro_flux_data(T_scale, flux_types=["nhf"], H=H_ML)

    ## heat flux feedback alpha_1, alpha_2, alpha_3, dims (year, cycle, param)
    alpha = src.ro.fit_alpha(data["T_3"], Q["nhf"], cycle=fits["cycle"])
    return T_scale, alpha, fits


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Parameters

    Each parameter has dims (year, cycle). The linear growth rates are in yr$^{-1}$; the
    noise is in K yr$^{-1}$.
    """
    )
    return


@app.cell
def _(REF_YEAR, T_scale, alpha, fits, src):
    ## linear coefficients of R and of the heat flux feedback
    R_1 = src.ro.get_R_coefs(fits).sel(param="T1", drop=True)
    alpha_1 = alpha.sel(param="T1", drop=True)

    ## noise of the T equation (std. dev. of its stochastic forcing), in K yr^-1
    noise_T = T_scale * fits["xi_stdac"].sel(ranky=1, drop=True)

    ## R, its two parts (R = R_o - alpha), and the noise
    params = dict(R=R_1, Ro=R_1 + alpha_1, minus_alpha=-alpha_1, noise=noise_T)

    ## change from the reference window
    delta = dict()
    for _name, _p in params.items():
        delta[_name] = _p - _p.sel(year=REF_YEAR)
    return (delta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(AMP_NOISE, AMP_R, YLINE, YTICKS, delta, plt, src):
    ## one panel per parameter: (change, title, color scale limit)
    _panels = [
        (delta["R"], "a) $R$", AMP_R),
        (delta["Ro"], "b) $R_o$", AMP_R),
        (delta["minus_alpha"], r"c) $-\alpha$", AMP_R),
        (delta["noise"], "d) Noise", AMP_NOISE),
    ]

    ## one row of panels
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.5), layout="constrained")

    ## contour each change by calendar month (x) and window (y)
    _cps = []
    for _ax, (_d, _title, _amp) in zip(axs, _panels):
        _cp = _ax.contourf(
            _d.cycle,
            _d.year,
            _d.transpose("year", "cycle"),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(_amp, _amp / 10),
            extend="both",
        )

        ## keep the contours for the colorbars; title
        _cps.append(_cp)
        _ax.set_title(_title, loc="left")

        ## January and June ticks, with a line at June and at the marked year
        _ax.set_xticks([_d.cycle[0], _d.cycle[5]], labels=["Jan", "Jun"])
        _ax.axvline(_d.cycle[5], ls="--", c="gray", lw=0.8)
        _ax.axhline(YLINE, ls="--", c="gray", lw=1)

        ## year labels only on the first panel (added below)
        _ax.set_yticks([])

    ## colorbars: growth rates (a-c) and noise (d)
    fig.colorbar(_cps[0], ax=axs[:3], ticks=[-AMP_R, 0, AMP_R], label=r"yr$^{-1}$")
    fig.colorbar(
        _cps[3], ax=axs[3], ticks=[-AMP_NOISE, 0, AMP_NOISE], label=r"K yr$^{-1}$"
    )

    ## year labels on the first panel only
    axs[0].set_yticks(YTICKS)

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "RO-experiments_param_hov")

    fig
    return


if __name__ == "__main__":
    app.run()
