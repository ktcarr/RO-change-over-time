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
    # RO growth rate over time

    Paper figure `fig:gr-over-time` (`paper-outline.tex` in the ch3 Overleaf project).
    The nonlinear recharge oscillator (RO) is fitted to CESM2's Niño 3 index ($T$) and
    thermocline depth ($h$) in each 40-year window (centered every 10 years); each panel
    shows a fitted parameter over time, averaged over the seasonal cycle.

    - **(a) SST growth rate** $R$ for warm ($T = 1.5\sigma$, red), cold
      ($T = -1.5\sigma$, blue), and neutral ($T = 0$, black) anomalies. The T equation is
      nonlinear ($dT/dt = R_1 T + R_2 T^2 + R_3 T^3 + \dots$), so the growth rate depends on
      the anomaly: $R(T) = R_1 + R_2 T + R_3 T^2$. $\sigma$ is the standard deviation of
      $T$ for each calendar month (averaged over windows).
    - **(b) Growth rate of the RO oscillation**, $\frac{1}{2}(R - \varepsilon)$, where
      $\varepsilon$ is the damping rate of $h$
    - **(c) Stochastic forcing amplitude**: fractional change in the standard deviation
      of the noise in the $T$ and $h$ equations, relative to the 1880 window

    The fitting step is shared with the other RO figures (`src/ro.py`) and uses the old
    RO code in `src/XRO.py`.

    **Inputs** (under `DATA_FP/cesm`): `Th_anom.nc` (Niño 3 index and thermocline
    depth).

    **Output:** `SAVE_FP/figures/gr-over-time.pdf`

    Before this script, the figure came from the "Plot params over time" cells of
    `RO-change-over-time_NL.py` (saved as `gr-over-time_ann-...`). Changes:

    - Panels are labeled (a)-(c), and the legend labels give the anomaly size in
      standard deviations ($\pm 1.5\sigma$, and $T=0$ instead of "All").
    - $R(T)$ is evaluated at exactly $\pm 1.5\sigma$ and 0, not at the nearest point of a
      grid from $-3\sigma$ to $3\sigma$ (the difference is rounding error).
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

    ## this project's modules: plotting, RO fits
    import src.plot
    import src.ro

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all paper figures; color_codes (on by default) makes
    ## "r" and "b" the colorblind palette's red and blue
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
    ## anomaly size for warm and cold events, in standard deviations
    N_SIGMA = 1.5

    ## window that the change in stochastic forcing is measured from
    REF_YEAR = 1880

    ## years marked on the time axis
    XTICKS = [1880, 2010, 2050]
    return N_SIGMA, REF_YEAR, XTICKS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fit the RO

    `src.ro.FIT_KWARGS` holds the fit settings shared by all RO figures.
    """
    )
    return


@app.cell
def _(src):
    ## windowed, non-dimensional T (Niño 3) and h_hat, dims (member, year, time)
    data, _ = src.ro.load_ro_data(t_var="T_3")

    ## fit the RO in each window
    model = src.ro.get_model()
    fits = src.ro.fit_over_time(data[src.ro.VARNAMES], model)
    return data, fits, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Parameters

    Each has dims (year, cycle); the figure averages over the seasonal cycle.
    """
    )
    return


@app.cell
def _(N_SIGMA, data, fits, src):
    ## SST growth rate for warm, cold, and neutral anomalies (yr^-1)
    R = dict(
        warm=src.ro.get_R_at(fits, data, n_sigma=N_SIGMA),
        cold=src.ro.get_R_at(fits, data, n_sigma=-N_SIGMA),
        neutral=src.ro.get_R_at(fits, data, n_sigma=0),
    )

    ## damping rate of h (yr^-1)
    eps = src.ro.get_eps(fits)
    return R, eps


@app.cell
def _(REF_YEAR, fits, model):
    ## standard deviation of the noise in the T and h equations (seasonal mean)
    _params = model.get_RO_parameters(fits)
    xi = _params[["xi_T", "xi_h"]].mean("cycle")

    ## fractional change from the reference window
    dxi = xi / xi.sel(year=REF_YEAR) - 1
    return (dxi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(N_SIGMA, R, XTICKS, dxi, eps, plt, src):
    ## (a) R, (b) growth rate of the oscillation, (c) stochastic forcing
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")

    ## (a, b): warm, cold, and neutral anomalies
    _lines = [
        ("warm", "r", rf"$T={N_SIGMA}\sigma$"),
        ("cold", "b", rf"$T=-{N_SIGMA}\sigma$"),
        ("neutral", "k", r"$T=0$"),
    ]
    for _name, _c, _label in _lines:
        _R = R[_name]

        ## seasonal means of R and of the oscillation's growth rate
        axs[0].plot(_R.year, _R.mean("cycle"), c=_c, label=_label)
        axs[1].plot(_R.year, 0.5 * (_R - eps).mean("cycle"), c=_c, label=_label)

    ## (c): noise in the T and h equations
    axs[2].plot(dxi.year, dxi["xi_T"], label=r"$T$")
    axs[2].plot(dxi.year, dxi["xi_h"], label=r"$h$")

    ## titles
    axs[0].set_title(r"a) $R$", loc="left")
    axs[1].set_title(r"b) $\frac{1}{2}\left(R-\varepsilon\right)$", loc="left")
    axs[2].set_title(r"c) $\Delta$ (stochastic forcing amp.)", loc="left")

    ## units
    axs[0].set_ylabel(r"Growth rate (yr$^{-1}$)")
    axs[2].set_ylabel("Fraction")

    ## legends: anomaly size, and T vs. h noise
    axs[0].legend()
    axs[2].legend()

    ## zero line for R
    axs[0].axhline(0, ls="--", c="k", lw=0.8)

    ## y-ticks
    axs[0].set_yticks([-1, 0])
    axs[1].set_yticks([-1, -0.5])
    axs[2].set_yticks([0, 0.1])

    ## time axis: ticks and lines at the marked years
    src.utils.add_vticks(axs, xticks=XTICKS, xlines=XTICKS)

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "gr-over-time")

    fig
    return


if __name__ == "__main__":
    app.run()
