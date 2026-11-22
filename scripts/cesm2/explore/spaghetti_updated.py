import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spaghetti
    Make spaghetti plots for ENSO events
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import matplotlib as mpl
    import numpy as np
    import seaborn as sns
    import xarray as xr
    import tqdm
    import pathlib
    import cmocean
    import os
    import copy
    import pandas as pd

    # Import custom modules
    import src.utils

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return (
        DATA_FP,
        SAVE_FP,
        copy,
        mpl,
        np,
        os,
        pathlib,
        plt,
        sns,
        src,
        tqdm,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funcs
    """)
    return


@app.cell
def _(os, pathlib, src, xr):
    def get_spaghetti(idx, data, peak_month, event_type=None, q=0.95, is_warm=True):
        """
        Get hovmoller composite based on specified:
        - data: used to compute index/make composite
        - peak_month: month to center composite on
        - q: quantile threshold for composite
        """

        ## handle warm/cold case
        if is_warm:
            kwargs = dict(q=q, check_cutoff=lambda x, cut: x > cut)
        else:
            kwargs = dict(q=1 - q, check_cutoff=lambda x, cut: x < cut)

        ## kwargs for composite
        kwargs = dict(
            kwargs,
            avg=False,
            peak_month=peak_month,
            idx=idx,
            data=data,
            event_type=event_type,
        )

        ## composite of projected data
        spag = src.utils.make_composite(**kwargs)

        return spag


    def save(fig, fname, dpi=800, do_save=False):
        """save figure to file"""

        ## get save directory
        # save_dir = pathlib.Path(os.environ["SAVE_FP"], "ch3-outline")
        save_dir = pathlib.Path(os.environ["SAVE_FP"], "egu_poster_figs")

        ## get fname
        fname = save_dir / f"{fname}.pdf"

        if do_save:
            fig.savefig(fname, dpi=dpi, format="pdf")

        return


    def load_consolidated_wide():
        """utility function to load consolidated data"""

        ## directory with data
        CONS_DIR = pathlib.Path(os.environ["DATA_FP"], "cesm", "consolidated_05")

        ## function to align and open
        kwargs = dict(pop_vars=["u", "u_comp", "T", "T_comp", "w", "w_comp"])
        align_and_open = lambda fp: src.utils.align_pop_times(xr.open_dataset(fp), **kwargs)

        ## open data and align pop times
        kwargs = dict(pop_vars=["u", "u_comp", "T", "T_comp", "w", "w_comp"])
        forced = align_and_open(CONS_DIR / "forced.nc")
        anom = align_and_open(CONS_DIR / "anom.nc")

        return forced, anom


    def load_h_ohc_idxs():
        """load OHC-based h indices"""

        ## specify subsetting funcs
        LONS_E = dict(longitude=slice(210, 270))
        LONS_W = dict(longitude=slice(120, 180))

        ## averaging funcs
        lon_avg = lambda x, lons: x.sel(lons).mean("longitude")
        lat_avg = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")

        ## load spatial data
        _, anom_wide = load_consolidated_wide()

        ## empty array to hold result
        h_idxs = xr.Dataset()

        ## compute ohc
        h_idxs["h_w_ohc"] = src.utils.reconstruct_wrapper(
            anom_wide[["T", "T_comp"]],
            lambda x: lon_avg(x.integrate("z_t"), LONS_W) / 300,
        )["T"]
        h_idxs["h_e_ohc"] = src.utils.reconstruct_wrapper(
            anom_wide[["T", "T_comp"]],
            lambda x: lon_avg(x.integrate("z_t"), LONS_E) / 300,
        )["T"]

        h_idxs["h_ohc"] = 0.5 * (h_idxs["h_w_ohc"] + h_idxs["h_e_ohc"])
        h_idxs["dh_ohc"] = h_idxs["h_e_ohc"] - h_idxs["h_w_ohc"]

        return h_idxs

    return get_spaghetti, load_h_ohc_idxs, save


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.function
## function to get merged nino3/34 regions
def get_nino_merged(x):
    idx = dict(latitude=slice(-5, 5), longitude=slice(190, 270))
    return x.sel(idx).mean(["longitude", "latitude"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### spatial data
    """)
    return


@app.cell
def _(src):
    _, anom = src.utils.load_consolidated()
    return (anom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### T, h
    """)
    return


@app.cell
def _(load_h_ohc_idxs, src, xr):
    ## fits

    ## Th data
    Th = src.utils.load_cesm_indices(load_z20=True)

    ## load ohc data
    Th = xr.merge([Th, load_h_ohc_idxs()])

    ## omit first year (bc of NaN in h,hw vars)
    Th = Th.sel(time=slice("1851", None))
    return (Th,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load ELI
    """)
    return


@app.cell
def _(DATA_FP, Th, pathlib, src, xr):
    ## Load ELI
    eli_all = xr.open_dataset(pathlib.Path(DATA_FP, 'cesm/eli.nc'))
    eli_forced, eli = src.utils.separate_forced(eli_all)
    ## add to data
    Th_1 = xr.merge([Th, eli.sel(time=slice('1851', None))])
    return (Th_1,)


@app.cell
def _(Th_1, anom, src):
    ## get NHF in Niño regions
    nhf_3 = src.utils.reconstruct_wrapper(anom[['nhf', 'nhf_comp']], fn=src.utils.get_nino3)['nhf']
    nhf_34 = src.utils.reconstruct_wrapper(anom[['nhf', 'nhf_comp']], fn=src.utils.get_nino34)['nhf']
    nhf_m = src.utils.reconstruct_wrapper(anom[['nhf', 'nhf_comp']], fn=get_nino_merged)['nhf']
    T_m = src.utils.reconstruct_wrapper(anom[['sst', 'sst_comp']], fn=get_nino_merged)['sst']
    sec_per_mo = 86400.0 * 30
    sec_per_yr = 12 * sec_per_mo
    rho = 1020.0
    Cp = 4200.0
    H = 50
    Q = nhf_34 * sec_per_yr / (rho * Cp * H)
    Q_3 = nhf_3 * sec_per_yr / (rho * Cp * H)
    Q_m = nhf_m * sec_per_yr / (rho * Cp * H)
    Th_1['Q'] = Q.sel(time=Th_1.time)
    ## convert to units of K/mo
    Th_1['Q_3'] = Q_3.sel(time=Th_1.time)
    Th_1['Q_m'] = Q_m.sel(time=Th_1.time)
    ## merge with Th data
    Th_1['T_m'] = T_m.sel(time=Th_1.time)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### get windowed data
    """)
    return


@app.cell
def _(Th_1, src):
    ## get windowed data (used to estimate change in parameters over time)
    Th_rolling = src.utils.get_windowed(Th_1, window_size=480, stride=120)
    return (Th_rolling,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scale ELI by gradient
    """)
    return


@app.cell
def _(SAVE_FP, Th_rolling, pathlib, xr):
    ## load dT/dx data
    dTdx = xr.open_dataarray(pathlib.Path(SAVE_FP, 'cesm_dTdx_Tw.nc'))
    dTdx_scale = dTdx / dTdx.isel(year=0).mean(['month'])
    ## scale by initial value
    dTdx_scale = dTdx_scale.assign_coords({'year': Th_rolling.year})
    for _v in list(Th_rolling):
    ## update year
        if 'eli' in _v:
            Th_rolling[f'{_v}_scaled'] = Th_rolling[_v].groupby('time.month') * dTdx_scale
            grouped = Th_rolling[f'{_v}_scaled'].groupby('time.month')
            Th_rolling[f'{_v}_scaled_median'] = grouped - grouped.median(['member', 'time'])  ## subtrack median
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Early and late
    """)
    return


@app.cell
def _(Th_rolling):
    ## specify years
    years = [2011, 2081]

    ## get early and late
    Th_early = Th_rolling.sel(year=years[0])
    Th_late = Th_rolling.sel(year=years[1])

    ## remove median
    Th_early = Th_early.groupby("time.month") - Th_early.groupby("time.month").median()
    Th_late = Th_late.groupby("time.month") - Th_late.groupby("time.month").median()
    return Th_early, Th_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## compute spaghetti/composite
    """)
    return


@app.cell
def _(Th_early, Th_late, get_spaghetti):
    ## specify composite specs
    comp_kwargs = dict(
        is_warm=True,
        event_type=None,
        peak_month=12,
        q=0.95,
    )

    ## specify variable to use for composite
    VARNAME = "T_34"

    ## do the compute
    spag_early = get_spaghetti(
        idx=Th_early[VARNAME], data=Th_early, **comp_kwargs
    ).drop_vars("year")
    spag_late = get_spaghetti(idx=Th_late[VARNAME], data=Th_late, **comp_kwargs).drop_vars(
        "year"
    )
    return spag_early, spag_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mean
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Clean up; add bounds
    """)
    return


@app.cell
def _(spag_early, spag_late):
    def prep(spag):
        """prep spaghetti"""

        ## differentiate / normalize
        spag["ddt_T"] = 12 * spag["T_34"].differentiate("lag")
        spag["Q_gr"] = spag["Q"] / spag["T_34"]
        spag["gr"] = spag["ddt_T"] / spag["T_34"]

        ## rolling
        return spag.rolling({"lag": 3}, center=True).mean()
        # return spag


    def relu_neg(x):
        """get (negative) relu of dataarray"""
        return x.where(x < 0, other=0)


    spag_diff = prep(spag_late) - prep(spag_early)
    spag_diff_q = spag_diff.quantile(q=[0.1, 0.5, 0.9], dim="sample").rename(
        {"quantile": "q"}
    )
    return prep, relu_neg, spag_diff_q


@app.cell
def _(np, plt, prep, relu_neg, save, sns, spag_diff_q, spag_early, spag_late):
    ## specify y-lim
    xlim = [-7.2, 3]
    _ylim = [-6, 6]
    colors = sns.color_palette()
    ## specify colors
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 2.5), layout='constrained')
    for _i, (sp, _c, yr) in enumerate(zip([spag_early, spag_late], colors, [2010, 2080])):
        sp_ = prep(sp)['T_34'].quantile(q=[0.1, 0.5, 0.9], dim='sample').rename({'quantile': 'q'})
        axs[0].plot(sp_.sel(q=0.5), sp.lag, lw=4, c=_c, label=yr)
    ## plot difference
        lb, ub = [sp_.sel(q=q) for q in [0.1, 0.9]]
        if _i:
            for q in sp_.q:
                axs[0].plot(sp_.sel(q=q), sp.lag, lw=1, c=_c)
        else:
            axs[0].fill_betweenx(y=sp_.lag, x1=lb, x2=ub, color=_c, alpha=0.1)
    for _ax, _v in zip(axs[1:], ['ddt_T', 'Q']):  # for q, lw in zip(sp_.q, [.7,4,.7]):
        for q, lw in zip([0.5, 0.1, 0.9], [4, 1, 1]):
            _ax.plot(spag_diff_q[_v].sel(q=q), spag_diff_q.lag, lw=lw, c=colors[4])
        med = spag_diff_q[_v].sel(q=0.5)  ## plot bounds
        _ax.fill_betweenx(y=med.lag, x2=relu_neg(med), x1=np.zeros_like(med.lag), color='k', alpha=0.1, lw=0)
        _ax.set_yticks([])
        _ax.set_xlabel('K yr$^{-1}$')
        _ax.set_xticks([-4, 0])
        _ax.set_xlim(xlim)
        _ax.set_ylim(_ylim)
    axs[0].set_xlabel('K')
    axs[0].set_xticks([0, 3])
    axs[0].set_yticks([-6, -2, 1, 6], labels=['Jun(-1)', 'Oct(-1)', 'Jan(+1)', 'Jun(+1)'])
    axs[0].set_ylim(_ylim)
    axs[0].set_xlim([-0.5, None])
    f_kwargs = dict(lw=0.75, c='k', ls='--')
    for _ax in axs:
        _ax.axvline(0, c='k', lw=1)
        _ax.axhline(1, **f_kwargs)  ## highlight damping
        _ax.axhline(-2, **f_kwargs)
    axs[0].text(s=2010, color=colors[0], x=3, y=-5)
    axs[0].text(s=2080, color=colors[1], x=0.5, y=-1)
    axs[0].set_title('(c) $T_{34}$', loc='left')
    axs[1].set_title('(d) $\\Delta\\left(\\frac{d T_{34}}{dt}\\right)$', loc='left')
    axs[2].set_title('(e) $\\Delta ~Q_{34}$', loc='left')
    save(fig, fname='scalar_comp', do_save=False)
    ## formatting
    # axs[0].legend(prop=dict(size=10))
    # axs[0].set_ylabel("Lag (months)")
    ## add titles
    plt.gca()
    return axs, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get $R$ over time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Nov-Jan
    """)
    return


@app.cell
def _():
    show_all = False
    name = "pos"
    (not (show_all)) & (name == "all")
    return


@app.cell
def _(axs, fig, src):
    def regress_over_time(
        data_windowed,
        x_vars,
        y_vars,
        dims=["time"],
        by_month=False,
    ):
        """regression over time"""

        ## shared args
        kwargs = dict(x_vars=x_vars, y_vars=y_vars, dims=dims)

        ## do regression
        if by_month:
            coefs = data_windowed.groupby("time.month").map(
                src.utils.regress_xr_proj, **kwargs
            )
        else:
            coefs = src.utils.regress_xr_proj(data_windowed, **kwargs)

        return coefs.drop_vars("j")


    def plot_timeseries_v2(ax, coefs, center=False, show_all=False):
        """plot timeseries comparison"""

        ## normalize coefficients
        coefs_center = coefs - coefs.isel(year=0).mean("member")

        ## select coefficients to plot
        if center:
            plot_coefs = coefs_center
        else:
            plot_coefs = coefs

        ## loop thru pos and negative
        for i, (name, color, label) in enumerate(
            zip(
                ["pos", "neg", "all"],
                ["r", "b", "k"],
                [r"$T_{34}>0$ (warm)", "$T_{34}<0$ (cold)", "all"],
            )
        ):
            # if False:
            if (not (show_all)) & (name == "all"):
                pass

            else:

                ## plot median and bounds
                if name == "all":
                    q_vals = [0.5]
                else:
                    q_vals = [0.5, 0.1, 0.9]

                for q, lw in zip(q_vals, [2, 0.6, 0.6]):
                    label_ = label if (q == 0.5) else None

                    ## finally, plot data
                    ax.plot(
                        plot_coefs.year,
                        plot_coefs[name].quantile(q=q, dim="member"),
                        c=color,
                        lw=lw,
                        label=label_,
                    )

        ## formatting
        ax.set_xticks([1870, 2010])
        ax_kwargs = dict(ls="--", c="gray", lw=0.8)
        ax.axvline(2010, **ax_kwargs)
        ax.axhline(0, **ax_kwargs)
        # ax.set_yticks([])

        return fig, axs


    def plot_R_comparison(axs, R, alpha, **kwargs):
        """plot comparison of R stuff on axs"""

        ## Bjerknes
        plot_timeseries_v2(axs[0], coefs=R, **kwargs)

        ## damping
        plot_timeseries_v2(axs[1], coefs=alpha, **kwargs)

        ## difference
        plot_timeseries_v2(axs[2], coefs=R - alpha, **kwargs)

        ##formatting
        for ax in axs[1:]:
            ax.set_yticks([])
        axs[0].set_ylabel(r"yr$^{-1}$")

        return axs

    return plot_R_comparison, regress_over_time


@app.cell
def _(Th_rolling):
    # T_VAR, Q_VAR = "T_m", "Q_m"
    T_VAR, Q_VAR = "T_34", "Q"
    # T_VAR, Q_VAR = "T_3", "Q_3"
    # T_VAR, Q_VAR = "eli_05_scaled", "Q_3"
    data_ = Th_rolling[[T_VAR, Q_VAR]]
    T_idx = data_[T_VAR]
    return Q_VAR, T_VAR, T_idx, data_


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute $\alpha$
    """)
    return


@app.cell
def _(Q_VAR, T_VAR, T_idx, data_, regress_over_time, src, xr):
    MONTHS_ALPHA = [11, 12, 1]
    MONTHS_R = [11, 2]
    data_ndj = src.utils.sel_month(data_, months=MONTHS_ALPHA)
    _data_pos = data_ndj.where(T_idx > 0)
    _data_neg = data_ndj.where(T_idx < 0)
    _kwargs = dict(x_vars=[T_VAR], y_vars=[Q_VAR])
    _alpha_all = regress_over_time(data_ndj, **_kwargs)[Q_VAR]
    _alpha_pos = regress_over_time(_data_pos, **_kwargs)[Q_VAR]
    _alpha_neg = regress_over_time(_data_neg, **_kwargs)[Q_VAR]
    alpha = xr.merge([_alpha_all.rename('all'), _alpha_pos.rename('pos'), _alpha_neg.rename('neg')])
    alpha = alpha * -1
    return MONTHS_R, alpha


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute $R$
    """)
    return


@app.cell
def _(MONTHS_R, T_VAR, data_, np, regress_over_time, src, xr):
    ## get data to compute alpha
    _data0 = src.utils.sel_month(data_[T_VAR], months=MONTHS_R[0]).isel(time=slice(None, -1))
    _data1 = src.utils.sel_month(data_[T_VAR], months=MONTHS_R[1]).isel(time=slice(1, None))
    _data1 = _data1.assign_coords({'time': _data0.time})
    _data_all = xr.merge([_data0.rename('T0'), _data1.rename('Tf')])
    _data_pos = _data_all.where(_data_all['T0'] > 0)
    _data_neg = _data_all.where(_data_all['T0'] < 0)
    _kwargs = dict(x_vars=['T0'], y_vars=['Tf'])
    ## compute
    _S_all = regress_over_time(_data_all, **_kwargs)['Tf']
    _S_pos = regress_over_time(_data_pos, **_kwargs)['Tf']
    _S_neg = regress_over_time(_data_neg, **_kwargs)['Tf']
    _S = xr.merge([_S_all.rename('all'), _S_pos.rename('pos'), _S_neg.rename('neg')])
    _tau = 3
    _mpy = 12
    ## divide by 3 to convert to 1/month, then mult. by 12 to convert to 1/yr
    R = 1 / _tau * _mpy * np.log(_S)  # months  # months per year
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot
    """)
    return


@app.cell
def _(R, alpha, np, plot_R_comparison, plt, save, src):
    fig_1, axs_1 = plt.subplots(1, 3, figsize=(8, 2.75), layout='constrained')
    axs_1 = plot_R_comparison(axs_1, R=R, alpha=-alpha)
    _labels = ['$R$', '$\\alpha$', '$R_o=R+\\alpha$']
    _labels = ['a) $T_{34}$ growth rate', 'b) $Q_{34}$ contribution', 'c) Residual (a-b)']
    for _ax, _label in zip(axs_1, _labels):
        _ax.set_title(_label, loc='left')
    _yticks = np.array([-3, 0, 1])
    _ylim = np.array([-3.5, 1.3])
    axs_1[0].set_yticks(_yticks)
    axs_1[2].set_yticks((-_yticks)[::-1])
    axs_1[0].set_ylim(_ylim)
    for _ax in axs_1[2:]:
        _ax.set_ylim((-_ylim)[::-1])
    src.utils.set_lims(axs_1[:2])
    axs_1[0].legend(frameon=False, prop=dict(size=10), loc=(-0.03, 0.12))
    save(fig_1, fname='gr-alpha_comparison', do_save=False)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### All months
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute $\alpha$
    """)
    return


@app.cell
def _(Q_VAR, T_VAR, T_idx, data_, regress_over_time, xr):
    _data_pos = data_.where(T_idx > 0)
    _data_neg = data_.where(T_idx < 0)
    _kwargs = dict(x_vars=[T_VAR], y_vars=[Q_VAR], by_month=True)
    _alpha_all = regress_over_time(data_, **_kwargs)[Q_VAR]
    _alpha_pos = regress_over_time(_data_pos, **_kwargs)[Q_VAR]
    _alpha_neg = regress_over_time(_data_neg, **_kwargs)[Q_VAR]
    alpha_1 = xr.merge([_alpha_all.rename('all'), _alpha_pos.rename('pos'), _alpha_neg.rename('neg')])
    alpha_1 = alpha_1 * -1
    return (alpha_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute $R$
    """)
    return


@app.cell
def _(T_VAR, data_, np, regress_over_time, xr):
    _data0 = data_[T_VAR].isel(time=slice(None, -1))
    _data1 = data_[T_VAR].isel(time=slice(1, None))
    _data1 = _data1.assign_coords({'time': _data0.time})
    _data_all = xr.merge([_data0.rename('T0'), _data1.rename('Tf')])
    _data_pos = _data_all.where(_data_all['T0'] > 0)
    _data_neg = _data_all.where(_data_all['T0'] < 0)
    _kwargs = dict(x_vars=['T0'], y_vars=['Tf'], by_month=True)
    _S_all = regress_over_time(_data_all, **_kwargs)['Tf']
    _S_pos = regress_over_time(_data_pos, **_kwargs)['Tf']
    _S_neg = regress_over_time(_data_neg, **_kwargs)['Tf']
    _S = xr.merge([_S_all.rename('all'), _S_pos.rename('pos'), _S_neg.rename('neg')])
    _tau = 1
    _mpy = 12
    R_1 = 1 / _tau * _mpy * np.log(_S)
    return (R_1,)


@app.cell
def _(R_1, alpha_1, np, plot_R_comparison, plt, src):
    fig_2, axs_2 = plt.subplots(1, 3, figsize=(8, 2.75), layout='constrained')
    sel_ndj = lambda x: x.mean('month')
    CENTER = False
    axs_2 = plot_R_comparison(axs_2, R=sel_ndj(R_1), alpha=-sel_ndj(alpha_1), show_all=True, center=CENTER)
    if CENTER:
        for _ax in axs_2:
            _ax.set_ylim([-1, 1])
        axs_2[0].set_yticks([-1, 0, 1])
        _labels = ['a) $\\Delta T_{34}$ growth rate', 'b) $\\Delta Q_{34}$ contribution', 'c) Residual (a-b)']
    else:
        _yticks = np.array([-3, 0, 1])
        _ylim = np.array([-3.5, 1.5])
        axs_2[0].set_yticks(_yticks)
        axs_2[2].set_yticks((-_yticks)[::-1])
        axs_2[0].set_ylim(_ylim)
        for _ax in axs_2[2:]:
            _ax.set_ylim((-_ylim)[::-1])
        src.utils.set_lims(axs_2[:2])
        _labels = ['a) $T_{34}$ growth rate', 'b) $Q_{34}$ contribution', 'c) Residual (a-b)']
    axs_2[0].legend()
    for _ax, _label in zip(axs_2, _labels):
        _ax.set_title(_label, loc='left')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot seasonal cycle
    """)
    return


@app.cell
def _(R_1, np, plt, src):
    _months = np.arange(1, 13)
    fig_3, axs_3 = plt.subplots(1, 3, figsize=(8, 2.5))
    for _ax, n in zip(axs_3, ['all', 'pos', 'neg']):
        for _y in [1871, 2011, 2051, 2081]:
            _ax.plot(_months, R_1[n].mean('member').sel(year=_y))
        _ax.axhline(0, ls='--', c='k', lw=1)
        _ax.set_yticks([])
        _ax.set_xticks([2, 7, 12], labels=['Feb', 'Jul', 'Dec'])
        _ax.set_title(n)
    src.utils.set_lims(axs_3)
    axs_3[0].set_yticks([-4, 0, 4])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nonlinear $R$ over time
    """)
    return


@app.cell(hide_code=True)
def _(copy, np, tqdm, xr):
    def get_ddt(data):
        """differentiate with respect to time"""
        data_ = copy.deepcopy(data)
        data_ = data_.assign_coords({"t_idx": ("time", np.arange(len(data_.time)))})
        data_ = data_.swap_dims({"time": "t_idx"})

        ## differentiate
        ddt_data = data_.differentiate("t_idx").swap_dims({"t_idx": "time"})

        ## mult. by 12 to convert from 1/mo to 1/yr
        return 12 * ddt_data


    def regress_pinv(data, x_var, y_var, order=4):
        """do nonlinear regression"""

        ## get feature matrix
        X_nl = xr.merge([(data[x_var] ** i).rename(i) for i in range(0, order)])

        ## prep data
        prep = lambda x: x.transpose("member", ...)
        X_ = prep(X_nl.to_dataarray(dim="order"))
        Y_ = prep(data[y_var])

        ## empty array to hold results
        m = xr.DataArray(
            coords={"order": X_.order.values, "member": X_.member.values},
            dims=["member", "order"],
        )

        ## do regression
        X_pinv = np.linalg.pinv(X_.values)
        m.values = np.einsum("mi,mij->mj", Y_.values, X_pinv)

        return m


    def regress_pinv_overtime(data, x_var, y_var, order=4):
        """do regression over time"""

        ## subset and fit
        kwargs = dict(x_var=x_var, y_var=y_var, order=order)
        coef = []
        for y in tqdm.tqdm(data.year):
            coef.append(data.sel(year=y).groupby("time.month").map(regress_pinv, **kwargs))
        coef = xr.concat(coef, dim=data.year)

        return coef

    return get_ddt, regress_pinv_overtime


@app.cell
def _(T_VAR, data_, get_ddt, regress_pinv_overtime, xr):
    ## target data
    ddt_T = get_ddt(data_[T_VAR]).rename('ddt_T')
    XY = xr.merge([data_[T_VAR], ddt_T])
    ## Feature matrix
    _kwargs = dict(x_var=T_VAR, y_var='ddt_T')
    ## compute
    R_nl = regress_pinv_overtime(XY, **_kwargs)
    return R_nl, XY


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot NDJ
    """)
    return


@app.cell
def _(R_nl, plt, src):
    MONTHS = [10, 11, 12]
    R_nl_dec = R_nl.sel(month=MONTHS).mean(['month', 'member'])
    fig_4, _ax = plt.subplots(figsize=(3, 2.5))
    for _o in range(4):
        _ax.plot(R_nl.year, R_nl_dec.sel(order=_o), label=_o)
    src.utils.add_vticks([_ax], xticks=[1870, 2010, 2080], xlines=[2010])
    _ax.axhline(0, ls='--', c='k', lw=0.8)
    _ax.legend(prop=dict(size=8))
    plt.gca()
    return MONTHS, R_nl_dec


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot cycle
    """)
    return


@app.cell
def _(R_nl, np, plt, src):
    _months = np.arange(1, 13)
    fig_5, axs_4 = plt.subplots(1, 4, figsize=(10, 2.5))
    for _ax, _o in zip(axs_4, np.arange(0, 4)):
        for _y in [1871, 2011, 2051, 2081]:
            _ax.plot(_months, R_nl.sel(order=_o).mean('member').sel(year=_y))
        _ax.axhline(0, ls='--', c='k', lw=1)
        _ax.set_yticks([])
        _ax.set_xticks([2, 7, 12], labels=['Feb', 'Jul', 'Dec'])
        _ax.set_title(f'Order {_o}')
    src.utils.set_lims(axs_4)
    axs_4[0].set_yticks([-4, 0, 4])
    plt.gca()
    return


@app.cell
def _(mpl):
    mpl.rcParams["figure.dpi"] = 100
    return


@app.cell
def _(MONTHS, R_1, R_nl, R_nl_dec, T_VAR, XY, np, plt, src, xr):
    T = np.linspace(-3, 3)
    zz = xr.DataArray(T, coords=dict(T=T))
    zz = zz ** R_nl_dec.order
    YEARS = [1871, 2011, 2081]
    fig_6, axs_5 = plt.subplots(3, 3, figsize=(7.5, 5.5), layout='constrained')
    for _i, month in enumerate(MONTHS):
        for j, _y in enumerate(YEARS):
            axs_5[j, _i].scatter(src.utils.sel_month(XY.sel(year=_y)[T_VAR], [month]), src.utils.sel_month(XY.sel(year=_y)['ddt_T'], [month]), s=3, c='gray', alpha=0.2)
            axs_5[j, _i].axhline(0, ls='--', c='k', lw=1)
            axs_5[j, _i].axvline(0, ls='--', c='k', lw=1)
            R_nl_month = R_nl.sel(year=_y, month=month).mean('member')
            R_ = (R_nl_month * zz).sum('order')
            axs_5[j, _i].plot(R_['T'], R_, c='k', lw=3)
            zz_pos = np.linspace(0, T[-1])
            zz_neg = np.linspace(T[0], 0)
            R_ = R_1.sel(month=month, year=_y).mean('member')
            axs_5[j, _i].plot(zz_pos, R_['pos'].values.item() * zz_pos, c='r', lw=3)
            axs_5[j, _i].plot(zz_neg, R_['neg'].values.item() * zz_neg, c='b', lw=3)
            axs_5[j, _i].set_yticks([])
            axs_5[j, _i].set_xticks([])
    src.utils.set_lims(axs_5.flatten())
    for _ax, m in zip(axs_5[0, :], MONTHS):
        _ax.set_title(f'Month {m}')
    for _ax, _y in zip(axs_5[:, -1], YEARS):
        _ax.text(x=1.2, y=0.5, s=_y, transform=_ax.transAxes)
    for _ax in axs_5[-1, :]:
        _ax.set_xticks([-3, 0, 3])
        _ax.set_xlabel('$T_{34}$ (K)')
    for _ax in axs_5[:, 0]:
        _ax.set_yticks([-6, 0, 6])
        _ax.set_ylabel('$\\frac{d T_{34}}{dt}$ (K/yr)')
    plt.gca()
    return (zz,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try to plot nonlinear R over time
    """)
    return


@app.cell
def _(MONTHS, R_nl, zz):
    zz2 = zz.isel(order=slice(None, -1))
    zz2 = zz2.assign_coords({'order': zz.order.values[1:]})
    R__1 = R_nl.sel(order=slice(1, None)) * zz2
    R__1 = R__1.sel(order=slice(1, 3)).sum('order')
    ## select which to sum over
    R_ndj = R__1.sel(month=MONTHS).mean(['month', 'member'])
    # R_ = R_.mean(["month","member"])
    R_ann = R__1.mean(['month', 'member'])
    return R__1, R_ann, R_ndj


@app.cell
def _(R__1, R_ann, R_ndj, plt, src):
    fig_7, axs_6 = plt.subplots(1, 2, figsize=(6, 2.5), layout='constrained')
    for _ax, R__ in zip(axs_6, [R_ndj, R_ann]):
        for T_, _c, l in zip([0, -1, 1], ['k', 'b', 'r'], ['$T=0$', '$T=-1$', '$T=1$']):
            _ax.plot(R__1.year, R__.sel(T=T_, method='nearest'), c=_c, label=l)
        _ax.axhline(0, ls='--', c='k', lw=1)
    src.utils.add_vticks(axs_6, xticks=[1870, 2010, 2080], xlines=[2010])
    src.utils.set_lims(axs_6)
    axs_6[1].set_yticks([])
    axs_6[0].set_title('NDJ')
    axs_6[1].set_title('All months')
    axs_6[1].legend()
    axs_6[0].set_ylabel('$R$ (K yr$^{-1}$)')
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
