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
    # RO validation: ENSO statistics over time in CESM2 and the RO

    Paper figure `fig:RO-validation` (`paper-outline.tex` in the ch3 Overleaf project).
    The nonlinear recharge oscillator (RO) is fitted to CESM2's Niño 3 index ($T$) and
    thermocline depth in each 40-year window (centered every 10 years), then simulated
    with each window's parameters (500 copies of 40 years). Statistics use DJF means.

    - **(a) Standard deviation** of $T$ and **(b) skewness**: computed for each CESM2
      member or RO copy; thick lines show the median and thin lines the 10th and 90th
      percentiles, for CESM2 (solid) and the RO (dashed)
    - **(c) CESM2** and **(d) RO**: size of strong events, pooling all members (or
      copies) and years in the window. Red: 95th percentile (strong El Niño); blue:
      minus the 5th percentile (strong La Niña); black: their average (the width of the
      distribution)

    The fitting and simulation steps are shared with the other RO figures
    (`src/ro.py`) and use the RO model from `~/ro_tools` (`ro_tools.xro.XRO`).

    **Inputs** (under `DATA_FP/cesm`): `Th_anom.nc` (Niño 3 index and thermocline
    depth).

    **Output:** `SAVE_FP/figures/RO-validation.pdf`

    Before this script, the figure was two PDFs placed side by side
    (`RO-valid2_T_3-...` and `RO-valid_T_3-...`, from the "validate" cells of
    `explore/RO-change-over-time_NL.py`). Changes:

    - One figure, with panels labeled (a)-(d).
    - $T$ is shown in K. The RO works with $T$ divided by the standard deviation of the
      Niño 3 index (1.23 K), and the old panels showed those non-dimensional values
      with "K" on the axis; here they are multiplied back by 1.23 K. Skewness is
      unchanged.
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

    ## statistics and labeled arrays
    import scipy.stats
    import xarray as xr

    ## this project's modules: loading data, plotting, RO fits and simulations
    import src.data
    import src.plot
    import src.ro

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all paper figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return plt, scipy, sns, src, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fit and simulate the RO

    `src.ro.FIT_KWARGS` and `src.ro.SIM_KWARGS` hold the fit and simulation settings
    shared by all RO figures.
    """
    )
    return


@app.cell
def _(src):
    ## windowed, non-dimensional T (Niño 3) and h_hat, dims (member, year, time);
    ## T_scale (K) converts T back to K
    data, T_scale = src.ro.load_ro_data(t_var="T_3")

    ## fit the RO in each window
    model = src.ro.get_model()
    fits = src.ro.fit_over_time(data[src.ro.VARNAMES], model)

    ## simulate each window, starting from the first CESM2 value
    X0 = data[src.ro.VARNAMES].isel(year=0, member=0, time=0)
    sims = src.ro.simulate_over_time(model, fits, X0)
    return T_scale, data, sims


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Statistics

    DJF means of $T$ (in K) for each window, then the statistics in each panel.
    """
    )
    return


@app.cell
def _(T_scale, data, sims, src):
    ## DJF means of T in K: CESM2 (dims member, year, time) and RO (member = copy)
    T_djf = dict(
        CESM2=T_scale * src.data.get_djf(data["T_3"]),
        RO=T_scale * src.data.get_djf(sims["T_3"]),
    )
    return (T_djf,)


@app.cell
def _(T_djf, scipy, xr):
    def get_stats(T):
        """
        Statistics of DJF T in each window.

        Args:
            - T (xr.DataArray): DJF means (K), dims (member, year, time)

        Returns:
            - stats (xr.Dataset), dims (year, quantile):
                - "sigma", "skew": standard deviation (K) and skewness over time
                  for each member, then the 10th, 50th, and 90th percentiles over
                  members
                - "q": 5th, 50th, and 95th percentiles (K), pooling members and
                  times
        """

        ## quantiles across members of the per-member statistics
        q_member = [0.1, 0.5, 0.9]

        ## standard deviation over time, for each member
        stats = xr.Dataset()
        stats["sigma"] = T.std("time").quantile(q_member, dim="member")

        ## skewness over time, for each member
        skew = xr.apply_ufunc(
            scipy.stats.skew, T, input_core_dims=[["time"]], kwargs={"axis": -1}
        )
        stats["skew"] = skew.quantile(q_member, dim="member")

        ## tails of the distribution, pooling members and times
        stats["q"] = T.quantile([0.05, 0.5, 0.95], dim=["time", "member"]).rename(
            quantile="q_pooled"
        )

        return stats

    ## statistics for CESM2 and the RO
    stats = dict(CESM2=get_stats(T_djf["CESM2"]), RO=get_stats(T_djf["RO"]))
    return (stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(plt, sns, src, stats):
    ## left: standard deviation and skewness; right: tails in CESM2 and the RO
    fig = plt.figure(figsize=(12.5, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2, wspace=0.05)

    ## two panels in each half; (c) and (d) share the y-axis
    axs_l = subfigs[0].subplots(1, 2)
    axs_r = subfigs[1].subplots(1, 2, sharey=True)

    ## (a, b): median (thick) and 10th/90th percentiles (thin); CESM2 solid, RO dashed
    _colors = sns.color_palette()
    for _ax, _v in zip(axs_l, ["sigma", "skew"]):
        for _name, _c, _ls in [("CESM2", _colors[0], "-"), ("RO", _colors[1], "--")]:
            _s = stats[_name][_v]

            ## median, labeled for the legend
            _ax.plot(_s.year, _s.sel(quantile=0.5), c=_c, ls=_ls, lw=2, label=_name)

            ## percentiles
            _ax.plot(_s.year, _s.sel(quantile=0.1), c=_c, ls=_ls, lw=1)
            _ax.plot(_s.year, _s.sel(quantile=0.9), c=_c, ls=_ls, lw=1)

    ## (a, b) titles and units
    axs_l[0].set_title(r"a) $\sigma(T)$ (DJF)", loc="left")
    axs_l[1].set_title(r"b) skew$(T)$ (DJF)", loc="left")
    axs_l[0].set_ylabel("K")

    ## zero line for skewness
    axs_l[1].axhline(0, ls="--", c="k", lw=0.8)

    ## legend for CESM2 vs. RO
    axs_l[0].legend()

    ## (c, d): strong El Niño, strong La Niña, and their average
    for _ax, _name in zip(axs_r, ["CESM2", "RO"]):
        _q = stats[_name]["q"]
        _q95 = _q.sel(q_pooled=0.95)
        _q05 = _q.sel(q_pooled=0.05)

        ## 95th percentile and minus the 5th (both positive)
        _ax.plot(_q.year, _q95, c="r", label="El Niño")
        _ax.plot(_q.year, -_q05, c="b", label="La Niña")

        ## their average: half the 5th-95th percentile range
        _ax.plot(_q.year, 0.5 * (_q95 - _q05), c="k", label="Average")

    ## (c, d) labels
    axs_r[0].set_title("c) CESM2", loc="left")
    axs_r[1].set_title("d) RO", loc="left")
    axs_r[0].set_ylabel("K")

    ## legend for the three curves
    axs_r[1].legend()

    ## time axis: first and last windows, and a line at 2010
    for _axs in [axs_l, axs_r]:
        src.utils.add_vticks(_axs, xticks=[1870, 2010, 2080], xlines=[2010])

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "RO-validation")

    fig
    return


if __name__ == "__main__":
    app.run()
