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
    # RO perturbation experiments: ocean vs. heat flux parts of $R$

    Paper figure `fig:RO-experiments_Ro` (`paper-outline.tex` in the ch3 Overleaf project).
    Same as `fig:RO-experiments` (`RO-experiments.py`), but splitting the SST growth rate
    into an ocean part and the surface heat flux feedback, $R = R_o - \alpha$. The
    feedback $\alpha(T) = \alpha_1 + \alpha_2 T + \alpha_3 T^2$ comes from a cubic fit of
    the net surface heat flux over Niño 3 (as a warming rate of a 50 m mixed layer) to
    $T$, in each window and calendar month; $R_o = R + \alpha$ is the rest of $R$
    (advection and mixing).

    Each experiment fixes one part to its value in the first window (1870) while the other
    part still changes. Each panel shows the change in the amplitude of El Niño (red; 95th
    percentile of monthly DJF $T$), La Niña (blue; minus the 5th percentile), and their
    average (black):

    - **(a) Effect of $R_o$** and **(b) effect of $\alpha$**: control minus the experiment
    - **(c) Sum** of (a) and (b)
    - **(d) Effect of $R$**: control minus the experiment with all of $R$ fixed (panel (b)
      of `fig:RO-experiments`)

    **Inputs** (under `DATA_FP/cesm`): `Th_anom.nc` (Niño 3 index and thermocline
    depth), `consolidated/anom.nc` (net surface heat flux).

    **Output:** `SAVE_FP/figures/RO-experiments_Ro.pdf`

    Before this script, the figure came from the "Check linearity of subset" cell of
    `explore/RO-change-over-time_NL.py` (saved as `exp-comparison_Ro-...`). Changes:

    - Amplitudes are in K (the RO's non-dimensional $T$ times the standard deviation of
      the Niño 3 index); before, the non-dimensional values were plotted with a
      $^{\circ}$C label.
    - Panels are labeled (a)-(d), and (d) is titled "Effect of $R$" (it was "Control
      (effect of $R$)").
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

    ## mixed-layer depth (m) for converting the heat flux to a warming rate
    H_ML = 50

    ## years marked on the time axis, and the year marked with a line
    XTICKS = [1870, 2010, 2080]
    XLINES = [2010]
    return H_ML, REF_YEAR, XLINES, XTICKS


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
    return T_scale, alpha, data, fits, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Perturbed parameters

    $R = R_o - \alpha$, so the heat flux feedback's part of $R$ is $-\alpha$ and the ocean
    part is $R_o = R + \alpha$. Each experiment keeps one part at its value in the
    reference window.
    """
    )
    return


@app.cell
def _(REF_YEAR, alpha, fits, src):
    ## the two parts of R's coefficients (R_1, R_2, R_3), dims (year, cycle, param)
    _part_alpha = -alpha
    _part_Ro = src.ro.get_R_coefs(fits) - _part_alpha

    ## parameters for each experiment
    fits_fixed = dict(
        Ro=src.ro.remove_change_in_R(fits, _part_Ro, ref_year=REF_YEAR),
        alpha=src.ro.remove_change_in_R(fits, _part_alpha, ref_year=REF_YEAR),
        R=src.ro.fix_terms(fits, src.ro.RO_TERMS["R"], ref_year=REF_YEAR),
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
def _(amp_ctrl, amp_fixed):
    ## effect of each part's changes: control minus the experiment without them
    effect = dict()
    for _name, _amp in amp_fixed.items():
        effect[_name] = amp_ctrl - _amp

    ## sum of the effects of the two parts of R
    effect_sum = effect["Ro"] + effect["alpha"]
    return effect, effect_sum


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(XLINES, XTICKS, effect, effect_sum, plt, src):
    ## one panel for each part of R, then their sum and the effect of all of R
    _panels = [
        (effect["Ro"], r"a) Effect of $R_o$"),
        (effect["alpha"], r"b) Effect of $\alpha$"),
        (effect_sum, "c) Sum"),
        (effect["R"], r"d) Effect of $R$"),
    ]

    ## same y-axis for all panels
    fig, axs = plt.subplots(1, 4, figsize=(8, 2), layout="constrained", sharey=True)

    ## amplitude changes of El Niño, La Niña, and their average
    for _ax, (_amp, _title) in zip(axs, _panels):
        src.plot.plot_amplitudes(_ax, _amp)
        _ax.set_title(_title, loc="left")

    ## units, y-ticks, and legend
    axs[0].set_ylabel(r"$\Delta$ amplitude ($^{\circ}$C)")
    axs[0].set_yticks([-0.8, -0.4, 0, 0.4])
    axs[0].legend(prop=dict(size=8), loc="lower left")

    ## time axis: ticks at the marked years, line at 2010
    src.utils.add_vticks(axs, xticks=XTICKS, xlines=XLINES)

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "RO-experiments_Ro")

    fig
    return


if __name__ == "__main__":
    app.run()
