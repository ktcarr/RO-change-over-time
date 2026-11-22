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
    # RO parameter perturbation experiments

    Paper figure `fig:RO-experiments` (`paper-outline.tex` in the ch3 Overleaf project).
    The nonlinear recharge oscillator (RO) is fitted to CESM2's Niño 3 index ($T$) and
    thermocline depth ($h$) in each 40-year window (centered every 10 years) and simulated
    with each window's parameters (the "control"). Each experiment re-runs the simulations
    with some terms fixed to their values in the first window, so it lacks the changes in
    those terms; the difference from the control is the effect of those changes.

    Each panel shows the change in the amplitude of El Niño (red; 95th percentile of
    monthly DJF $T$), La Niña (blue; minus the 5th percentile), and their average (black):

    - **(a)-(d) Effect of** the stochastic forcing ("noise"), the SST growth rate $R$, the
      thermocline feedback $F_1$, and the recharge terms $F_2$ and $\varepsilon$:
      control minus the experiment
    - **(e) Sum** of (a)-(d)
    - **(f) Control**: change in the control simulations from the first window

    The fitting and simulation steps are shared with the other RO figures (`src/ro.py`)
    and use the RO model from `~/ro_tools` (`ro_tools.xro.XRO`).

    **Inputs** (under `DATA_FP/cesm`): `Th_anom.nc` (Niño 3 index and thermocline
    depth).

    **Output:** `SAVE_FP/figures/RO-experiments.pdf`

    Before this script, the figure came from the "Check linearity" cell of
    `explore/RO-change-over-time_NL.py` (saved as `exp-comparison_general-...`). Changes:

    - Amplitudes are in K (the RO's non-dimensional $T$ times the standard deviation of
      the Niño 3 index); before, the non-dimensional values were plotted with a
      $^{\circ}$C label.
    - The noise is fixed to its value in the first window (1870), like the other
      parameters; before, it was fixed to the second window (1880), because the noise in
      the first window was NaN before the missing $h$ value was filled.
    - Panels are labeled (a)-(f), and the $F_1$ title is typeset like the others.
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

    ## this project's modules: plotting, RO fits and simulations
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
    ## window that the experiments fix parameters to (the first window)
    REF_YEAR = 1870

    ## years marked on the time axis, and the year marked with a line
    XTICKS = [1870, 2010, 2080]
    XLINES = [2010]
    return REF_YEAR, XLINES, XTICKS


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
    data, T_scale = src.ro.load_ro_data(t_var="T_3")

    ## fit the RO in each window
    model = src.ro.get_model()
    fits = src.ro.fit_over_time(data[src.ro.VARNAMES], model)
    return T_scale, data, fits, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Perturbed parameters

    Each experiment fixes some of the RO's terms (`src.ro.RO_TERMS`) to their values in
    the reference window; all other terms still change over time.
    """
    )
    return


@app.cell
def _(REF_YEAR, fits, src):
    ## the terms fixed in each experiment
    _terms = src.ro.RO_TERMS

    ## parameters for each experiment
    fits_fixed = dict(
        noise=src.ro.fix_terms(fits, _terms["noise"], ref_year=REF_YEAR),
        R=src.ro.fix_terms(fits, _terms["R"], ref_year=REF_YEAR),
        F1=src.ro.fix_terms(fits, _terms["F1"], ref_year=REF_YEAR),
        F2_eps=src.ro.fix_terms(fits, _terms["F2"] + _terms["eps"], ref_year=REF_YEAR),
    )
    return (fits_fixed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Simulate

    `src.ro.SIM_KWARGS` holds the simulation settings shared by all RO figures (500
    copies of 40 years for each window, with the same seed in every experiment).
    """
    )
    return


@app.cell
def _(T_scale, data, fits, fits_fixed, model, src):
    ## initial condition: the first month of the first window in one member
    X0 = data[src.ro.VARNAMES].isel(year=0, member=0, time=0)

    ## control: simulate with each window's fitted parameters
    _sims = src.ro.simulate_over_time(model, fits, X0)
    amp_ctrl = src.ro.get_amplitudes(_sims, T_scale)

    ## experiments: simulate with the perturbed parameters
    amp_fixed = dict()
    for _name, _fits in fits_fixed.items():
        _sims = src.ro.simulate_over_time(model, _fits, X0)
        amp_fixed[_name] = src.ro.get_amplitudes(_sims, T_scale)
    return amp_ctrl, amp_fixed


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Effects

    Amplitudes (K) of El Niño and La Niña, dims (year,).
    """
    )
    return


@app.cell
def _(REF_YEAR, amp_ctrl, amp_fixed):
    ## effect of each set of terms: control minus the experiment without its changes
    effect = dict()
    for _name, _amp in amp_fixed.items():
        effect[_name] = amp_ctrl - _amp

    ## sum of the effects
    effect_sum = sum(effect.values())

    ## change in the control from the reference window
    change_ctrl = amp_ctrl - amp_ctrl.sel(year=REF_YEAR)
    return change_ctrl, effect, effect_sum


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(XLINES, XTICKS, change_ctrl, effect, effect_sum, plt, src):
    ## one panel for each experiment, then the sum and the control
    _panels = [
        (effect["noise"], "a) Effect of noise"),
        (effect["R"], r"b) Effect of $R$"),
        (effect["F1"], r"c) Effect of $F_1$"),
        (effect["F2_eps"], r"d) Effect of $F_2$, $\varepsilon$"),
        (effect_sum, "e) Sum"),
        (change_ctrl, "f) Control"),
    ]

    ## same y-axis for all panels
    fig, axs = plt.subplots(1, 6, figsize=(12, 2), layout="constrained", sharey=True)

    ## amplitude changes of El Niño, La Niña, and their average
    for _ax, (_amp, _title) in zip(axs, _panels):
        src.plot.plot_amplitudes(_ax, _amp)
        _ax.set_title(_title, loc="left")

    ## units, y-ticks, and legend
    axs[0].set_ylabel(r"$\Delta$ amplitude ($^{\circ}$C)")
    axs[0].set_yticks([-0.4, 0, 0.4])
    axs[0].legend(prop=dict(size=8), loc="lower left")

    ## time axis: ticks at the marked years, line at 2010
    src.utils.add_vticks(axs, xticks=XTICKS, xlines=XLINES)

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "RO-experiments")

    fig
    return


if __name__ == "__main__":
    app.run()
