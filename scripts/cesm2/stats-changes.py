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
    # Stats changes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at how stats change over time
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
    import datetime
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import seaborn as sns
    import xarray as xr
    import tqdm
    import pathlib
    import cmocean
    import os
    import src.XRO
    import copy
    import scipy.stats
    import warnings
    import calendar
    import pandas as pd
    import cartopy.util

    # import gsw

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])

    ## set style specs
    ax_kwargs = dict(ls="--", c="gray", lw=0.8)
    return (
        DATA_FP,
        SAVE_FP,
        ax_kwargs,
        cmocean,
        np,
        os,
        pathlib,
        pd,
        plt,
        scipy,
        sns,
        src,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Function to get windowed data
    """)
    return


@app.cell
def _(os, pathlib, src, xr):
    def get_windowed(x):
        return src.utils.get_windowed(x, stride=60, window_size=480)


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

    return get_windowed, load_consolidated_wide, save


@app.cell
def _(DATA_FP, pathlib, src, xr):
    ## load spatial data
    forced, anom = src.utils.load_consolidated()
    Th_ = src.utils.load_cesm_indices_3d(mld=80)
    ## load Th data
    # Th = src.utils.load_cesm_indices()
    Th = Th_ - Th_.mean('member')
    _eli = xr.open_dataset(pathlib.Path(DATA_FP, 'cesm/eli.nc'))
    _eli_forced, _eli_anom = src.utils.separate_forced(_eli)
    ## load ELI
    ## merge
    Th = xr.merge([Th, _eli_anom])
    return Th, Th_


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute relative SST
    """)
    return


@app.cell
def _(DATA_FP, Th, Th_, pathlib, xr):
    ## load tropical SST data and total data
    # Th_total = xr.open_dataset(DATA_FP / "cesm" / "Th.nc")
    Th_total = Th_
    trop_sst = xr.open_dataset(pathlib.Path(DATA_FP, 'cesm/trop_sst.nc'))
    for _v in ['T_3', 'T_34', 'T_4']:
    # ## subtract median
    # Th = src.utils.separate_forced(Th, use_mean=False, n = 18)[1]
        Th[f'{_v}_rel'] = Th_total[_v] - trop_sst['trop_sst_15']
    return (Th_total,)


@app.cell
def _(DATA_FP, xr):
    Th_total2 = xr.open_dataset(DATA_FP / "cesm" / "Th.nc")
    return (Th_total2,)


@app.cell
def _(Th_total, Th_total2, plt):
    plt.plot(Th_total2["T_34"].isel(time=slice(None,240),member=0))
    plt.plot(Th_total["T_34"].isel(time=slice(None,240),member=0))
    return


@app.cell
def _(Th_total, Th_total2, np, plt):
    _fig, _ax = plt.subplots(figsize=(4, 4))
    _ax.scatter(Th_total2['T_34'].isel(time=slice(None, 240, 12)), Th_total['T_34'].isel(time=slice(None, 240, 12)), s=1)
    _zz = np.linspace(22, 30)
    _ax.plot(_zz, _zz, c='k')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load OHC
    """)
    return


@app.cell
def _(Th, load_consolidated_wide, src):
    ## load
    forced_wide, anom_wide = load_consolidated_wide()

    ## funcs
    lon_avg = lambda x, lons: x.sel(lons).mean("longitude")
    lat_avg = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")
    LONS_E = dict(longitude=slice(210, 270))
    LONS_W = dict(longitude=slice(120, 180))
    LONS_WE = dict(longitude=slice(120, 280))

    ## compute ohc
    Th["h_w_ohc"] = src.utils.reconstruct_wrapper(
        anom_wide[["T", "T_comp"]],
        lambda x: lon_avg(x.integrate("z_t"), LONS_W) / x.z_t.max(),
    )["T"]

    ## compute ohc
    Th["h_ohc"] = src.utils.reconstruct_wrapper(
        anom_wide[["T", "T_comp"]],
        lambda x: lon_avg(x.integrate("z_t"), LONS_WE) / x.z_t.max(),
    )["T"]
    return (anom_wide,)


@app.cell
def _(Th, anom_wide, src):
    get_ml_avg = lambda x: x.sel(z_t=slice(0, 50)).mean("z_t")
    Th["T_3_ml"] = src.utils.reconstruct_wrapper(
        anom_wide[["T", "T_comp"]],
        fn=lambda x: get_ml_avg(x).sel(longitude=slice(210, 270)).mean("longitude"),
    )["T"]

    Th["T_34_ml"] = src.utils.reconstruct_wrapper(
        anom_wide[["T", "T_comp"]],
        fn=lambda x: get_ml_avg(x).sel(longitude=slice(190, 240)).mean("longitude"),
    )["T"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get windowed data
    """)
    return


@app.cell
def _(Th, Th_total, get_windowed):
    for _v in ['T_3', 'T_34', 'T_4']:
        Th[f'{_v}_abs'] = Th_total[_v].sel(time=Th.time)
    Th_1 = get_windowed(Th)
    return (Th_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scale ELI data
    """)
    return


@app.cell
def _(SAVE_FP, Th_1, pathlib, xr):
    dTdx = xr.open_dataarray(pathlib.Path(SAVE_FP, 'cesm_dTdx_Tw_v3.nc'))
    dTdx_scale = dTdx / dTdx.isel(year=0).mean(['month'])
    for _v in list(Th_1):
        if 'eli' in _v:
            Th_1[f'{_v}_scaled'] = Th_1[_v].groupby('time.month') * dTdx_scale
            grouped = Th_1[f'{_v}_scaled'].groupby('time.month')
            Th_1[f'{_v}_scaled_median'] = grouped - grouped.median(['member', 'time'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute gradient
    """)
    return


@app.function
def get_ddx(x):
    """compute horizontal gradient"""

    ## ranges
    lat_range = dict(latitude=slice(-5, 5))
    lon_range_w = dict(longitude=slice(120, 160))
    lon_range_e = dict(longitude=slice(210, 270))

    ## spatial avg
    avg = lambda x: x.mean(["latitude", "longitude"])

    ## get east and west
    Tw = avg(x.sel(**lon_range_w, **lat_range))
    Te = avg(x.sel(**lon_range_e, **lat_range))

    return Te - Tw


@app.cell
def _(Th_1, get_windowed, src, xr):
    ## load data
    forced_1, anom_1 = src.utils.load_consolidated()
    _, flux_anom = src.utils.load_flux_data()
    ## load flux data
    flux_anom['lhflx'] = -flux_anom['lhflx']
    anom_1 = xr.merge([anom_1, flux_anom])
    ## update sign of LHFLX
    ssta = anom_1[['sst', 'sst_comp']]
    sst = xr.merge([forced_1[['sst_comp']], forced_1['sst'] + anom_1['sst']])
    ## merge
    Th_1['dTdx'] = get_windowed(src.utils.reconstruct_wrapper(ssta, get_ddx)['sst'])
    ## subset for temperature
    ## compute grad
    Th_1['dTdx_total'] = get_windowed(src.utils.reconstruct_wrapper(sst, get_ddx)['sst'])
    return anom_1, forced_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Heat flux
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter plots
    """)
    return


@app.cell
def _(Th_1, anom_1, ax_kwargs, forced_1, get_windowed, plt, src):
    _fn = src.utils.get_nino34
    varname = 'T_34_rel'
    _prQ_n34_anom = get_windowed(src.utils.reconstruct_wrapper(anom_1[['pr', 'pr_comp', 'nhf', 'nhf_comp']], fn=_fn))
    _prQ_n34_forced = get_windowed(src.utils.reconstruct_wrapper(forced_1[['pr', 'pr_comp', 'nhf', 'nhf_comp']], fn=_fn))
    prQ_n34 = 86000.0 * (_prQ_n34_forced + _prQ_n34_anom)
    _PLOT_VAR = 'pr'
    sel = lambda x: src.utils.sel_month(x, 12).transpose('time', 'member', ...)
    x0 = sel(Th_1[varname])
    x1 = sel(prQ_n34[_PLOT_VAR])
    _fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.5), layout='constrained')
    for _ax, _y_ in zip(axs, [2010, 2085]):
        _ax.scatter(x0.sel(year=_y_), x1.sel(year=_y_), s=3, alpha=0.5)
        _ax.axhline(0, **ax_kwargs)
        _ax.axvline(0, **ax_kwargs)
    src.utils.set_lims(axs)
    axs[1].set_yticks([])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### PDFs and scatter plot
    """)
    return


@app.cell
def _(Th_1, anom_1, get_windowed, src):
    prQ_n34_1 = get_windowed(src.utils.reconstruct_wrapper(anom_1[['pr', 'pr_comp', 'nhf', 'nhf_comp', 'fsns', 'fsns_comp', 'lhflx', 'lhflx_comp']], fn=src.utils.get_nino34))
    prQ_n34_1['Q_'] = prQ_n34_1['fsns'] + prQ_n34_1['lhflx']
    _MONTH = 12
    _PLOT_VAR = 'nhf'
    sel_1 = lambda x: src.utils.sel_month(x, _MONTH).transpose('time', 'member', ...)
    x0_1 = sel_1(Th_1['T_34_rel'])
    x1_1 = sel_1(prQ_n34_1[['nhf', 'fsns', 'lhflx', 'Q_']])
    sec_per_mo = 86400.0 * 30
    sec_per_yr = 12 * sec_per_mo
    rho = 1020.0
    Cp = 4200.0
    H = 50
    x1_scaled = x1_1 * sec_per_yr / (rho * Cp * H)
    return x0_1, x1_1, x1_scaled


@app.cell
def _(np, src, x0_1):
    _n_bins = 20
    _nhf_bins = np.linspace(-75, 40, _n_bins)
    sst_bins = np.linspace(-5, 3, _n_bins)
    _get_pdf = lambda x, bins: src.utils.get_empirical_pdf(x.stack(samp=['time', 'member']).values, bins)[0]
    pdfs_sst = []
    _pdfs_nhf = []
    for _y_ in [1870, 2010, 2080]:
        pdfs_sst.append(_get_pdf(x0_1.sel(year=_y_), sst_bins))
        _pdfs_nhf.append(_get_pdf(x0_1.sel(year=_y_), sst_bins))
    return pdfs_sst, sst_bins


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Original version
    """)
    return


@app.cell
def _(ax_kwargs, pdfs_sst, plt, sst_bins, x0_1, x1_1):
    _fig, axs_1 = plt.subplots(1, 2, figsize=(6.5, 3), layout='constrained')
    _x0_plot = x0_1.sel(member=slice(10, 15))
    x1_plot = x1_1['nhf'].sel(member=slice(10, 15))
    axs_1[0].scatter(_x0_plot, x1_plot, s=3, alpha=1, c='k')
    axs_1[0].axhline(0, **ax_kwargs)
    for _ax in axs_1:
        _ax.axvline(0, **ax_kwargs)
        _ax.set_xticks([-4, -2, 0, 2])
    axs_1[1].stairs(pdfs_sst[0], sst_bins, fill=True, alpha=0.1, color='k', label=1870)
    axs_1[1].stairs(pdfs_sst[1], sst_bins, lw=2, label=2010)
    axs_1[1].stairs(pdfs_sst[2], sst_bins, lw=2, label=2080)
    axs_1[1].set_yticks([])
    axs_1[0].set_ylabel('$W~m^{-2}$')
    axs_1[1].set_ylabel('Prob. density')
    axs_1[0].set_xlabel('$^{\\circ}C$')
    axs_1[1].set_xlabel('$^{\\circ}C$')
    axs_1[1].yaxis.set_label_position('right')
    axs_1[1].legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Thesis version
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Helper funcs to get curve fits
    """)
    return


@app.cell
def _(np, scipy):
    def _psi_(x, a, b, c, e):
        """base transfer function"""
        return c * np.exp(a * x) + b
      # return e * x**3 + c * x**2 + b * x + a
    def get_psi(x, y):  # return c * x**2 + b * x + a
        """get transfer function from data"""
        stack = lambda x: x.stack(s=['time', 'member']).values
        x_ = stack(x)
        y_ = stack(y)
        p, _ = scipy.optimize.curve_fit(f=_psi_, xdata=x_, ydata=y_)
        psi = lambda x: _psi_(x, a=p[0], b=p[1], c=p[2], e=p[3])  ## prep data
        return psi  ## get parameter fit  # f=psi_, xdata=x_, ydata=y_, p0=[-1, -1, 1, 1]  ## define transfer func

    return (get_psi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot data
    """)
    return


@app.cell
def _(
    ax_kwargs,
    get_psi,
    np,
    pdfs_sst,
    plt,
    save,
    sns,
    src,
    sst_bins,
    x0_1,
    x1_scaled,
):
    _Q_VAR = 'nhf'
    _colors = sns.color_palette()
    _fig, axs_2 = plt.subplots(1, 3, figsize=(9, 2.75), layout='constrained')
    axs_2[1].scatter(x0_1.sel(year=2010), x1_scaled[_Q_VAR].sel(year=2010), s=1, color=_colors[0])
    axs_2[2].scatter(x0_1.sel(year=2080), x1_scaled[_Q_VAR].sel(year=2080), s=1, color=_colors[1])
    psi_2010 = get_psi(x=x0_1.sel(year=2010), y=x1_scaled[_Q_VAR].sel(year=2010))
    psi_2080 = get_psi(x=x0_1.sel(year=2080), y=x1_scaled[_Q_VAR].sel(year=2080))
    _fit_kwargs = dict(c='k', lw=1.5, zorder=3)
    _zz = np.linspace(-4, 2)
    axs_2[1].plot(_zz, psi_2010(_zz), label='2010 fit', **_fit_kwargs)
    axs_2[2].plot(_zz, psi_2010(_zz), **_fit_kwargs)
    axs_2[2].plot(_zz, psi_2080(_zz), ls='--', label='2080 fit', **_fit_kwargs)
    axs_2[1].axhline(0, **ax_kwargs)
    axs_2[2].axhline(0, **ax_kwargs)
    src.utils.set_lims(axs_2[1:])
    for _ax in axs_2:
        _ax.axvline(0, **ax_kwargs)
        _ax.set_xticks([-4, -2, 0, 2])
        _ax.set_xlabel('$T_{34}^*$ (K)')
    axs_2[0].stairs(pdfs_sst[0], sst_bins, fill=True, alpha=0.1, color='k', label=1870)
    axs_2[0].stairs(pdfs_sst[1], sst_bins, lw=2, label=2010)
    axs_2[0].stairs(pdfs_sst[2], sst_bins, lw=2, label=2080)
    axs_2[0].set_yticks([0, 0.1, 0.2, 0.3])
    axs_2[1].set_yticks([])
    axs_2[0].set_ylabel('Prob. density')
    axs_2[-1].set_ylabel('$Q_{34}$ (K yr$^{-1})$')
    axs_2[-1].yaxis.set_label_position('right')
    axs_2[-1].yaxis.tick_right()
    axs_2[-1].tick_params(axis='y', which='both', length=0)
    _fn = lambda x: x.median()
    sc_kwargs0 = dict(s=100, marker='v', zorder=11)
    for _y, _c in zip([2010, 2080], _colors[:2]):
        axs_2[0].scatter(_fn(x0_1.sel(year=_y)), y=0.015, color=_c, **sc_kwargs0)
    _sc_kwargs12 = dict(sc_kwargs0, edgecolor='w')
    axs_2[1].scatter(_fn(x0_1.sel(year=2010)), _fn(x1_scaled[_Q_VAR].sel(year=2010)), color=_colors[0], label='2010 median', **_sc_kwargs12)
    axs_2[2].scatter(_fn(x0_1.sel(year=2010)), _fn(x1_scaled[_Q_VAR].sel(year=2010)), color=_colors[0], **_sc_kwargs12)
    axs_2[2].scatter(_fn(x0_1.sel(year=2080)), _fn(x1_scaled[_Q_VAR].sel(year=2080)), color=_colors[1], label='2080 median', **_sc_kwargs12)
    axs_2[0].set_title('(a) PDFs', loc='left')
    axs_2[1].set_title('(b) 2010', loc='left')
    axs_2[2].set_title('(c) 2080', loc='left')
    legend_kwargs = dict(prop=dict(size=11), frameon=False)
    axs_2[0].legend(**legend_kwargs, loc=(0.65, 0.65))
    axs_2[1].legend(**legend_kwargs)
    axs_2[2].legend(**legend_kwargs)
    save(_fig, fname='nhf_rel-sst_triple', do_save=False)
    plt.gca()
    return (sc_kwargs0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Similar but for NHF components
    """)
    return


@app.cell
def _(scipy):
    def _psi_(x, a, b, c, e):
        """base transfer function"""
        return e * x ** 3 + c * x ** 2 + b * x + a

    def get_psi_1(x, y):
        """get transfer function from data"""
        stack = lambda x: x.stack(s=['time', 'member']).values
        x_ = stack(x)
        y_ = stack(y)
        p, _ = scipy.optimize.curve_fit(f=_psi_, xdata=x_, ydata=y_)
        psi = lambda x: _psi_(x, a=p[0], b=p[1], c=p[2], e=p[3])
        return psi

    return (get_psi_1,)


@app.cell
def _(ax_kwargs, get_psi_1, np, plt, sc_kwargs0, sns, src, x0_1, x1_scaled):
    _Y0 = 2010
    _Y1 = 2080
    _Q_VAR = 'nhf'
    _colors = sns.color_palette()
    _fig, axs_3 = plt.subplots(2, 4, figsize=(9, 4.5), layout='constrained')
    for _j, (q_var, label) in enumerate(zip(['nhf', 'Q_', 'fsns', 'lhflx'], ['Net', 'SW+LHF', 'SW', 'LHF'])):
        axs_3[0, _j].set_title(label)
        axs_3[0, _j].scatter(x0_1.sel(year=_Y0), x1_scaled[q_var].sel(year=_Y0), s=1, color=_colors[0])
        axs_3[1, _j].scatter(x0_1.sel(year=_Y1), x1_scaled[q_var].sel(year=_Y1), s=1, color=_colors[1])
        psi_Y0 = get_psi_1(x=x0_1.sel(year=_Y0), y=x1_scaled[q_var].sel(year=_Y0))
        psi_Y1 = get_psi_1(x=x0_1.sel(year=_Y1), y=x1_scaled[q_var].sel(year=_Y1))
        _fit_kwargs = dict(c='k', lw=1.5, zorder=3)
        _zz = np.linspace(-4, 2)
        axs_3[0, _j].plot(_zz, psi_Y0(_zz), label=f'{_Y0} fit', **_fit_kwargs)
        axs_3[1, _j].plot(_zz, psi_Y0(_zz), **_fit_kwargs)
        axs_3[1, _j].plot(_zz, psi_Y1(_zz), ls='--', label=f'{_Y1} fit', **_fit_kwargs)
        _fn = lambda x: x.median()
        _sc_kwargs12 = dict(sc_kwargs0, edgecolor='w')
        axs_3[0, _j].scatter(_fn(x0_1.sel(year=_Y0)), _fn(x1_scaled[q_var].sel(year=_Y0)), color=_colors[0], label=f'{_Y0} median', **_sc_kwargs12)
        axs_3[1, _j].scatter(_fn(x0_1.sel(year=_Y0)), _fn(x1_scaled[q_var].sel(year=_Y0)), color=_colors[0], **_sc_kwargs12)
        axs_3[1, _j].scatter(_fn(x0_1.sel(year=_Y1)), _fn(x1_scaled[q_var].sel(year=_Y1)), color=_colors[1], label=f'{_Y1} median', **_sc_kwargs12)
    src.utils.set_lims(axs_3[1:])
    for _ax in axs_3.flatten():
        _ax.axvline(0, **ax_kwargs)
        _ax.axhline(0, **ax_kwargs)
        _ax.set_yticks([])
        _ax.set_xticks([])
    for _ax in axs_3[1]:
        _ax.set_xticks([-4, -2, 0, 2])
        _ax.set_xlabel('$T_{34}^*$ (K)')
    for _ax in axs_3[:, 0]:
        _ax.set_yticks([-10, -5, 0, 5])
        _ax.set_ylabel('K yr$^{-1}$')
    src.utils.set_lims(axs_3.flatten())
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### same, but with total SST
    """)
    return


@app.cell
def _(Th_1, anom_1, get_windowed, src):
    prQ_n34_2 = get_windowed(src.utils.reconstruct_wrapper(anom_1[['pr', 'pr_comp', 'nhf', 'nhf_comp']], fn=src.utils.get_nino34))
    _MONTH = 12
    _PLOT_VAR = 'nhf'
    sel_2 = lambda x: src.utils.sel_month(x, _MONTH).transpose('time', 'member', ...)
    x0_2 = sel_2(Th_1['T_34_abs'])
    x1_2 = sel_2(prQ_n34_2['nhf'])
    return x0_2, x1_2


@app.cell
def _(np, src, x0_2):
    _n_bins = 20
    _nhf_bins = np.linspace(-75, 40, _n_bins)
    sst_bins_1 = np.linspace(22, 35, _n_bins)
    _get_pdf = lambda x, bins: src.utils.get_empirical_pdf(x.stack(samp=['time', 'member']).values, bins)[0]
    pdfs_sst_1 = []
    _pdfs_nhf = []
    for _y_ in [1870, 2010, 2080]:
        pdfs_sst_1.append(_get_pdf(x0_2.sel(year=_y_), sst_bins_1))
        _pdfs_nhf.append(_get_pdf(x0_2.sel(year=_y_), sst_bins_1))
    return pdfs_sst_1, sst_bins_1


@app.cell
def _(ax_kwargs, pdfs_sst_1, plt, save, sns, src, sst_bins_1, x0_2, x1_2):
    _fig, axs_4 = plt.subplots(1, 3, figsize=(9.5, 3), layout='constrained')
    _x0_plot = x0_2.sel(member=slice(10, 15))
    x1_plot_1 = x1_2.sel(member=slice(10, 15))
    axs_4[1].scatter(x0_2.sel(year=2010), x1_2.sel(year=2010), s=1, color=sns.color_palette()[0])
    axs_4[2].scatter(x0_2.sel(year=2080), x1_2.sel(year=2080), s=1, color=sns.color_palette()[1])
    axs_4[1].axhline(0, **ax_kwargs)
    axs_4[2].axhline(0, **ax_kwargs)
    src.utils.set_lims(axs_4[1:])
    for _ax in axs_4:
        _ax.set_xlabel('$^{\\circ}$C')
    axs_4[0].stairs(pdfs_sst_1[0], sst_bins_1, fill=True, alpha=0.1, color='k', label=1870)
    axs_4[0].stairs(pdfs_sst_1[1], sst_bins_1, lw=2, label=2010)
    axs_4[0].stairs(pdfs_sst_1[2], sst_bins_1, lw=2, label=2080)
    axs_4[0].set_yticks([0, 0.1, 0.2, 0.3])
    axs_4[1].set_yticks([])
    axs_4[0].set_ylabel('Prob. density')
    axs_4[-1].set_ylabel('$W~m^{-2}$')
    axs_4[-1].yaxis.set_label_position('right')
    axs_4[-1].yaxis.tick_right()
    axs_4[0].legend()
    axs_4[0].set_title('(a) PDFs', loc='left')
    axs_4[1].set_title('(b) 2010', loc='left')
    axs_4[2].set_title('(c) 2080', loc='left')
    axs_4[0].set_xlim([None, 34.5])
    axs_4[0].set_xticks([23, 28, 33])
    axs_4[1].set_xlim([22, 32])
    axs_4[2].set_xlim([25, 35])
    axs_4[1].set_xticks([24, 31])
    axs_4[2].set_xticks([27, 34])
    save(_fig, fname='nhf_abs-sst_triple', dpi=300)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SST
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute quantiles
    """)
    return


@app.cell
def _(Th_1):
    ## specify width bound
    wb = 0.05
    ENS_AVG = False
    ## specify whether to ensemble average before/after
    if ENS_AVG:
        quantile_dims = ['time']
        reduce_fn = lambda x: x.mean('member')
        reduce_fn = lambda x: x.quantile(q=[0.1, 0.5, 0.9], dim='member')
    else:
        quantile_dims = ['time', 'member']
        reduce_fn = lambda x: x
    quant_ = Th_1.groupby('time.season').quantile(q=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], dim=quantile_dims).rename({'quantile': 'q'})
    quant = reduce_fn(quant_)
    width_lower = quant.sel(q=0.5) - quant.sel(q=wb)
    ## compute quantiles
    width_upper = quant.sel(q=1 - wb) - quant.sel(q=0.5)
    ## compute upper and lower width
    width_mean = 0.5 * (width_lower + width_upper)
    return width_lower, width_mean, width_upper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute PDFs
    """)
    return


@app.cell
def _(Th_1, np, plt, src, width_lower, width_mean, width_upper):
    VARNAME = 'T_34'
    SEASON, _MONTH = ('DJF', 12)
    YEARS = [1870, 2010, 2080]
    _x = src.utils.sel_month(Th_1[VARNAME].resample({'time': 'QS-DEC'}).mean(), _MONTH)
    _x = _x.stack(s=['member', 'time'])
    _x = _x - _x.median('s')
    edges = np.linspace(-4.5, 4.5, 20)
    _get_pdf = lambda y: src.utils.get_empirical_pdf(y.values, edges=edges)[0]
    pdfs = [_get_pdf(_x.sel(year=year)) for year in YEARS]
    _sel_ = lambda x: x[VARNAME].sel(season=SEASON)
    _fig, axs_5 = plt.subplots(1, 2, figsize=(6.5, 3), layout='constrained')
    axs_5[0].stairs(pdfs[0], edges, fill=True, alpha=0.1, color='k', label=YEARS[0])
    axs_5[0].stairs(pdfs[1], edges, lw=2, label=YEARS[1])
    axs_5[0].stairs(pdfs[2], edges, lw=2, label=YEARS[2])
    axs_5[0].axvline(ls='--', c='gray', lw=0.8)
    axs_5[0].legend()
    axs_5[0].set_xlabel('${\\circ}$C')
    axs_5[0].set_ylabel('Prob. density')
    axs_5[0].set_yticks([])
    axs_5[1].plot(width_lower.year, _sel_(width_lower), c='blue', label='Cold')
    axs_5[1].plot(width_lower.year, _sel_(width_upper), c='r', label='Warm')
    axs_5[1].plot(width_lower.year, _sel_(width_mean), c='k', label='Avg.')
    axs_5[1].set_ylabel('$^{\\circ}$C')
    axs_5[1].set_xlabel('Year')
    axs_5[1].set_yticks([2, 2.5, 3])
    axs_5[0].set_yticks([0, 0.1, 0.2, 0.3])
    axs_5[1].set_xticks(YEARS)
    axs_5[1].axvline(2010, ls='--', lw=0.8, c='gray')
    axs_5[1].legend()
    axs_5[1].yaxis.tick_right()
    axs_5[1].yaxis.set_label_position('right')
    plt.gca()
    return (axs_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### similar, but with bounds
    """)
    return


@app.cell
def _(Th_1, xr):
    ## specify thresh
    Q = 0.95
    median = Th_1['T_34'].groupby('time.season').median(['member', 'time'])
    ## remove median
    Th_median = Th_1['T_34'].groupby('time.season') - median
    Th_median_djf = Th_median.resample({'time': 'QS-DEC'}).mean()
    Th_median_djf = Th_median_djf.sel(time=Th_median_djf.time.dt.month == 12)
    ## resample to DJF
    nino_mag = Th_median_djf.where(Th_median_djf > Th_median_djf.quantile(q=Q, dim='time').drop_vars('quantile')).mean('time')
    nina_mag = -Th_median_djf.where(Th_median_djf < Th_median_djf.quantile(q=1 - Q, dim='time').drop_vars('quantile')).mean('time')
    mag = xr.merge([nino_mag.rename('warm'), nina_mag.rename('cold')])
    ## get magnitude for extremes
    mag['sigma'] = Th_median_djf.std('time')
    mag_mean = mag.mean('member')
    ## merge
    ## add median
    ## compute quantiles
    mag_stats = mag.quantile(q=[0.1, 0.5, 0.9], dim='member')
    return mag, mag_stats


@app.cell
def _(axs_5):
    axs_5[0].get_xlim()
    return


@app.cell
def _(Th_1, mag, mag_stats, np, plt, save, src):
    VARNAME_1 = 'T_34'
    SEASON_1, _MONTH = ('DJF', 12)
    YEARS_1 = [1870, 2010, 2080]
    _x = src.utils.sel_month(Th_1[VARNAME_1].resample({'time': 'QS-DEC'}).mean(), _MONTH)
    _x = _x.stack(s=['member', 'time'])
    _x = _x - _x.median('s')
    edges_1 = np.linspace(-4.5, 4.5, 20)
    _get_pdf = lambda y: src.utils.get_empirical_pdf(y.values, edges=edges_1)[0]
    pdfs_1 = [_get_pdf(_x.sel(year=year)) for year in YEARS_1]
    _sel_ = lambda x: x[VARNAME_1].sel(season=SEASON_1)
    _fig, axs_6 = plt.subplots(1, 3, figsize=(8.5, 2.75), layout='constrained')
    axs_6[0].stairs(pdfs_1[0], edges_1, fill=True, alpha=0.1, color='k', label=YEARS_1[0])
    axs_6[0].stairs(pdfs_1[1], edges_1, lw=2, label=YEARS_1[1])
    axs_6[0].stairs(pdfs_1[2], edges_1, lw=2, label=YEARS_1[2])
    axs_6[0].set_xlim([-4.5, 4.5])
    axs_6[0].set_xticks([-3, 0, 3])
    axs_6[0].axvline(ls='--', c='gray', lw=0.8)
    axs_6[0].legend(prop=dict(size=10), frameon=False, loc=(-0.03, 0.65))
    axs_6[0].set_xlabel('$T_{34}$ (K)')
    axs_6[0].set_ylabel('Prob. density')
    axs_6[0].set_yticks([])
    for _ax, n, _c in zip(axs_6[1:], ['warm', 'cold'], ['r', 'b']):
        avg = 0.5 * (mag_stats['warm'] + mag_stats['cold']).sel(quantile=0.5)
        _ax.plot(mag.year, avg, c='k', ls='--')
        for _q, _lw in zip([0.5, 0.1, 0.9], [2.5, 1, 1]):
            _ax.plot(mag.year, mag_stats[n].sel(quantile=_q), lw=_lw, c=_c)
        _ax.axvline(2010, ls='--', lw=0.8, c='gray')
        _ax.set_xticks(YEARS_1)
    axs_6[2].set_ylabel('$^{\\circ}$C')
    axs_6[1].set_xlabel('Year')
    axs_6[1].set_yticks([2, 2.5, 3])
    axs_6[0].set_yticks([0, 0.1, 0.2, 0.3])
    axs_6[1].set_yticks([])
    axs_6[2].yaxis.tick_right()
    axs_6[2].yaxis.set_label_position('right')
    src.utils.set_lims(axs_6[1:])
    axs_6[0].set_title('(a) PDFs', loc='left')
    axs_6[1].set_title('(b) Strong El Niño', loc='left')
    axs_6[2].set_title('(c) Strong La Niña', loc='left')
    save(_fig, 'pdf-over-time_bounds', do_save=False)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Variance and skewness over time
    """)
    return


@app.cell
def _(Th_1, np, scipy, xr):
    #### Get DJF
    nino_vars = ['T_3', 'T_34', 'T_4']
    resample = lambda x: x.resample({'time': 'QS-DEC'}).mean().isel(time=slice(1, -1), year=slice(None, -1))
    Th_seasonal = resample(Th_1[nino_vars])
    Th_djf = Th_seasonal.sel(time=Th_seasonal.time.dt.month == 12)
    get_stat = lambda x, fn: xr.apply_ufunc(fn, x, input_core_dims=[['time']], kwargs={'axis': -1})
    median_ot = get_stat(Th_djf, np.median)
    skew_ot = get_stat(Th_djf, scipy.stats.skew)
    std_ot = get_stat(Th_djf, np.std)
    ## compute stats
    m3_ot = skew_ot * std_ot ** 3
    return m3_ot, median_ot, skew_ot, std_ot


@app.cell
def _(m3_ot, median_ot, plt, save, skew_ot, std_ot):
    VAR = 'T_34'
    YEARS_2 = [1870, 2010, 2080]
    _fig, axs_7 = plt.subplots(2, 2, figsize=(6, 5), layout='constrained')
    for _ax, stat in zip(axs_7.flatten(), [median_ot, std_ot, skew_ot, m3_ot]):
        for _q, _lw in zip([0.5, 0.1, 0.9], [2.5, 1, 1]):
            _ax.plot(stat.year, stat[VAR].quantile(q=_q, dim='member'), lw=_lw, c='k')
        _ax.axvline(2010, ls='--', lw=0.8, c='gray')
        _ax.set_xticks([])
    axs_7[0, 0].set_ylabel('$^{\\circ}$C')
    axs_7[0, 1].set_ylabel('$^{\\circ}$C')
    axs_7[1, 0].set_ylabel('non-dimensional')
    axs_7[1, 1].set_ylabel('$^{\\circ}$C$^3$')
    axs_7[0, 0].set_title('(a) Median', loc='left')
    axs_7[0, 1].set_title('(b) Standard deviation ($\\sigma$)', loc='left')
    axs_7[1, 0].set_title('(c) Skewness ($m_3 / \\sigma^3$)', loc='left')
    axs_7[1, 1].set_title('(d) $m_3$', loc='left')
    for _ax in axs_7[1, :]:
        _ax.set_xticks(YEARS_2)
        _ax.set_xlabel('Year')
    for _ax in [axs_7[0, 0], axs_7[1, 0], axs_7[1, 1]]:
        _ax.axhline(0, ls='--', lw=0.8, c='gray')
    save(_fig, 'nino34_djf_stats-over-time', dpi=300)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OHC
    """)
    return


@app.cell
def _(Th_1, np, src):
    VARNAME_2 = 'h_w_ohc'
    SEASON_2, _MONTH = ('DJF', 12)
    YEARS_3 = [1870, 2010, 2085]
    _x = src.utils.sel_month(Th_1[VARNAME_2].resample({'time': 'QS-DEC'}).mean(), _MONTH)
    _x = _x.stack(s=['member', 'time'])
    _x = _x - _x.median('s')
    edges_2 = np.linspace(-3, 3, 20)
    _get_pdf = lambda y: src.utils.get_empirical_pdf(y.values, edges=edges_2)[0]
    pdfs_2 = [_get_pdf(_x.sel(year=year)) for year in YEARS_3]
    return SEASON_2, VARNAME_2, YEARS_3, edges_2, pdfs_2


@app.cell
def _(
    SEASON_2,
    VARNAME_2,
    YEARS_3,
    edges_2,
    pdfs_2,
    plt,
    width_lower,
    width_mean,
    width_upper,
):
    _sel_ = lambda x: x[VARNAME_2].sel(season=SEASON_2)
    _fig, axs_8 = plt.subplots(1, 2, figsize=(6.5, 3), layout='constrained')
    axs_8[0].stairs(pdfs_2[0], edges_2, fill=True, alpha=0.1, color='k', label=YEARS_3[0])
    axs_8[0].stairs(pdfs_2[1], edges_2, lw=2, label=YEARS_3[1])
    axs_8[0].stairs(pdfs_2[2], edges_2, lw=2, label=YEARS_3[2])
    axs_8[0].axvline(ls='--', c='gray', lw=0.8)
    axs_8[0].legend()
    axs_8[0].set_xlabel('${\\circ}$C')
    axs_8[0].set_ylabel('Density')
    axs_8[0].set_yticks([])
    axs_8[1].plot(width_lower.year, _sel_(width_lower), c='blue', label='Cold')
    axs_8[1].plot(width_lower.year, _sel_(width_upper), c='r', label='Warm')
    axs_8[1].plot(width_lower.year, _sel_(width_mean), c='k', label='Avg.')
    axs_8[1].set_ylabel('$^{\\circ}$C')
    axs_8[1].set_xlabel('Year')
    axs_8[1].set_ylim([0, None])
    axs_8[1].set_xticks(YEARS_3)
    axs_8[1].axvline(2010, ls='--', lw=0.8, c='gray')
    axs_8[1].legend()
    axs_8[1].yaxis.tick_right()
    axs_8[1].yaxis.set_label_position('right')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ELI
    """)
    return


@app.cell
def _(Th_1, np, src):
    VARNAME_3 = 'eli_05_scaled_median'
    SEASON_3, _MONTH = ('DJF', 12)
    YEARS_4 = [1870, 2010, 2080]
    _x = src.utils.sel_month(Th_1[VARNAME_3].resample({'time': 'QS-DEC'}).mean(), _MONTH)
    _x = _x.stack(s=['member', 'time'])
    _x = _x - _x.median('s')
    edges_3 = np.linspace(-20, 40, 20)
    _get_pdf = lambda y: src.utils.get_empirical_pdf(y.values, edges=edges_3)[0]
    pdfs_3 = [_get_pdf(_x.sel(year=year)) for year in YEARS_4]
    return SEASON_3, VARNAME_3, YEARS_4, edges_3, pdfs_3


@app.cell
def _(
    SEASON_3,
    VARNAME_3,
    YEARS_4,
    edges_3,
    pdfs_3,
    plt,
    width_lower,
    width_mean,
    width_upper,
):
    _sel_ = lambda x: x[VARNAME_3].sel(season=SEASON_3)
    _fig, axs_9 = plt.subplots(1, 2, figsize=(6.5, 3), layout='constrained')
    axs_9[0].stairs(pdfs_3[0], edges_3, fill=True, alpha=0.1, color='k', label=YEARS_4[0])
    axs_9[0].stairs(pdfs_3[1], edges_3, lw=2, label=YEARS_4[1])
    axs_9[0].stairs(pdfs_3[2], edges_3, lw=2, label=YEARS_4[2])
    axs_9[0].axvline(ls='--', c='gray', lw=0.8)
    axs_9[0].legend()
    axs_9[0].set_xlabel('${\\circ}$lon')
    axs_9[0].set_ylabel('Density')
    axs_9[0].set_yticks([])
    axs_9[1].plot(width_lower.year, _sel_(width_lower), c='blue', label='Cold')
    axs_9[1].plot(width_lower.year, _sel_(width_upper), c='r', label='Warm')
    axs_9[1].plot(width_lower.year, _sel_(width_mean), c='k', label='Avg.')
    axs_9[1].set_ylabel('$^{\\circ}$lon')
    axs_9[1].set_xlabel('Year')
    axs_9[1].set_xticks(YEARS_4)
    axs_9[1].axvline(2010, ls='--', lw=0.8, c='gray')
    axs_9[1].legend()
    axs_9[1].yaxis.tick_right()
    axs_9[1].yaxis.set_label_position('right')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot temperature as a function of ELI
    """)
    return


@app.cell
def _(scipy):
    def _psi_(x, a, b, c, e):
        """base transfer function"""
        return e * x ** 3 + c * x ** 2 + b * x + c

    def get_psi_2(x, y):
        """get transfer function from data"""
        p, _ = scipy.optimize.curve_fit(f=_psi_, xdata=x.values, ydata=y.values, p0=[-1, -1, 1, 1])
        psi = lambda x: _psi_(x, a=p[0], b=p[1], c=p[2], e=p[3])
        return psi

    return (get_psi_2,)


@app.cell
def _(Th_1, YEARS_4, edges_3, get_psi_2, np, pdfs_3, plt, save, src):
    _dx = 40
    _MONTH = 12
    _Y0 = 2010
    _Y1 = 2080
    _V0 = 'eli_05_scaled_median'
    sel_3 = lambda x: src.utils.sel_month(x.resample({'time': 'QS-DEC'}).mean(), _MONTH).stack(s=['member', 'time'])
    _plot_kwargs = dict(s=3, alpha=0.3)
    _fig, axs_10 = plt.subplots(1, 4, figsize=(10.5, 2.5), layout='constrained')
    axs_10[0].stairs(pdfs_3[0], edges_3, fill=True, alpha=0.1, color='k', label=YEARS_4[0])
    axs_10[0].stairs(pdfs_3[1], edges_3, lw=2, label=YEARS_4[1])
    axs_10[0].stairs(pdfs_3[2], edges_3, lw=2, label=YEARS_4[2])
    axs_10[0].axvline(ls='--', c='gray', lw=0.8)
    axs_10[0].legend()
    _x = sel_3(Th_1[_V0])
    for _ax, _V1, _eli_max, _title, let in zip(axs_10[1:], ['T_4', 'T_34', 'T_3'], [35, 35, 25], ['Niño 4', 'Niño 3.4', 'Niño 3'], ['(b)', '(c)', '(d)']):
        _y = sel_3(Th_1[_V1])
        _psi0 = get_psi_2(_x.sel(year=_Y0), _y.sel(year=_Y0))
        _psi1 = get_psi_2(_x.sel(year=_Y1), _y.sel(year=_Y1))
        _ax.scatter(_x.sel(year=_Y0), _y.sel(year=_Y0), **_plot_kwargs)
        _ax.scatter(_x.sel(year=_Y1) + _dx, _y.sel(year=_Y1), **_plot_kwargs)
        _zz = np.linspace(-10, _eli_max)
        _ax.plot(_zz, _psi0(_zz), c='k', ls='--')
        _ax.plot(_zz + _dx, _psi0(_zz), c='k', ls='--')
        _ax.plot(_zz + _dx, _psi1(_zz), c='k')
        _ax.set_title(f'{let} {_title}', loc='left')
        _ax.set_xticks([-10, 30, 75], labels=[-55, -15, 30])
        _ax.set_xlabel('$\\Delta$ ELI (${\\circ}$lon)')
        _ax.set_yticks([])
        for _t in [-3.5, 0, 3.5]:
            _ax.axhline(_t, ls='--', c='gray', lw=0.8)
    axs_10[0].set_yticks([0, 0.05])
    axs_10[0].set_xlabel('${\\circ}$lon')
    axs_10[0].set_ylabel('Density')
    axs_10[0].set_xticks([-15, 0, 30])
    axs_10[-1].set_yticks([-3.5, 0, 3.5])
    axs_10[-1].yaxis.set_label_position('right')
    axs_10[-1].yaxis.tick_right()
    axs_10[-1].set_ylabel('$^{\\circ}$C')
    axs_10[0].set_title('(a) ELI PDF', loc='left')
    src.utils.set_lims(axs_10[1:])
    save(_fig, 'eli-comparison', dpi=300)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same, but make the plots separate
    """)
    return


@app.cell
def _(Th_1, get_psi_2, np, plt, src):
    _dx = 40
    _MONTH = 12
    _Y0 = 2010
    _Y1 = 2080
    _V0 = 'eli_05_scaled_median'
    _V1 = 'T_4'
    sel_4 = lambda x: src.utils.sel_month(x.resample({'time': 'QS-DEC'}).mean(), _MONTH).stack(s=['member', 'time'])
    _plot_kwargs = dict(s=3, alpha=0.3)
    _fig, axs_11 = plt.subplots(1, 3, figsize=(9.5, 3), layout='constrained')
    _x = sel_4(Th_1[_V0])
    for _ax, _V1, _eli_max, _title in zip(axs_11, ['T_4', 'T_34', 'T_3'], [35, 35, 25], ['Niño 4', 'Niño 3.4', 'Niño 3']):
        _y = sel_4(Th_1[_V1])
        _psi0 = get_psi_2(_x.sel(year=_Y0), _y.sel(year=_Y0))
        _psi1 = get_psi_2(_x.sel(year=_Y1), _y.sel(year=_Y1))
        _ax.scatter(_x.sel(year=_Y0), _y.sel(year=_Y0), **_plot_kwargs)
        _ax.scatter(_x.sel(year=_Y1) + _dx, _y.sel(year=_Y1), **_plot_kwargs)
        _zz = np.linspace(-10, _eli_max)
        _ax.plot(_zz, _psi0(_zz), c='k', ls='--')
        _ax.plot(_zz + _dx, _psi0(_zz), c='k', ls='--')
        _ax.plot(_zz + _dx, _psi1(_zz), c='k')
        _ax.set_title(_title)
        _ax.set_xticks([-10, 30, 75], labels=[-55, -15, 30])
        _ax.set_xlabel('$\\Delta$ ELI (${\\circ}$lon)')
        _ax.set_yticks([])
        for _t in [-3.5, 0, 3.5]:
            _ax.axhline(_t, ls='--', c='gray', lw=0.8)
    axs_11[-1].set_yticks([-3.5, 0, 3.5])
    axs_11[-1].yaxis.set_label_position('right')
    axs_11[-1].yaxis.tick_right()
    axs_11[-1].set_ylabel('$^{\\circ}$C')
    src.utils.set_lims(axs_11[1:])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To-do...
    """)
    return


@app.cell
def _(Th_1, np, src):
    VARNAME_4 = 'eli_05'
    SEASON_4, _MONTH = ('DJF', 12)
    YEARS_5 = [1870, 2010, 2080]
    _x = src.utils.sel_month(Th_1[VARNAME_4].resample({'time': 'QS-DEC'}).mean(), _MONTH)
    _x = _x.stack(s=['member', 'time'])
    _x = _x - _x.median('s')
    edges_4 = np.linspace(-20, 40, 20)
    _get_pdf = lambda y: src.utils.get_empirical_pdf(y.values, edges=edges_4)[0]
    pdfs_4 = [_get_pdf(_x.sel(year=year)) for year in YEARS_5]
    return YEARS_5, edges_4, pdfs_4


@app.cell
def _(YEARS_5, edges_4, pdfs_4, plt, src):
    _dx = 40
    _MONTH = 12
    _Y0 = 2010
    _Y1 = 2080
    sel_5 = lambda x: src.utils.sel_month(x.resample({'time': 'QS-DEC'}).mean(), _MONTH).stack(s=['member', 'time'])
    _plot_kwargs = dict(s=3, alpha=0.3)
    _fig, axs_12 = plt.subplots(1, 2, figsize=(6.5, 3), layout='constrained')
    axs_12[0].stairs(pdfs_4[0], edges_4, fill=True, alpha=0.1, color='k', label=YEARS_5[0])
    axs_12[0].stairs(pdfs_4[1], edges_4, lw=2, label=YEARS_5[1])
    axs_12[0].stairs(pdfs_4[2], edges_4, lw=2, label=YEARS_5[2])
    axs_12[0].axvline(ls='--', c='gray', lw=0.8)
    axs_12[0].legend()
    axs_12[0].set_yticks([0, 0.05])
    axs_12[0].set_xlabel('${\\circ}$lon')
    axs_12[0].set_ylabel('Density')
    axs_12[0].set_xticks([-15, 0, 30])
    axs_12[0].set_title('DJF')
    src.utils.set_lims(axs_12[1:])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Gradient
    """)
    return


@app.cell
def _(Th_1, np, src):
    VARNAME_5 = 'dTdx_total'
    SEASON_5, _MONTH = ('DJF', 12)
    YEARS_6 = [1870, 2010, 2085]
    _x = src.utils.sel_month(Th_1[VARNAME_5].resample({'time': 'QS-DEC'}).mean(), _MONTH)
    _x = _x.stack(s=['member', 'time'])
    edges_5 = np.linspace(-7.5, 1.5, 20)
    _get_pdf = lambda y: src.utils.get_empirical_pdf(y.values, edges=edges_5)[0]
    pdfs_5 = [_get_pdf(_x.sel(year=year)) for year in YEARS_6]
    return SEASON_5, VARNAME_5, YEARS_6, edges_5, pdfs_5


@app.cell
def _(
    SEASON_5,
    VARNAME_5,
    YEARS_6,
    edges_5,
    pdfs_5,
    plt,
    width_lower,
    width_mean,
    width_upper,
):
    _sel_ = lambda x: x[VARNAME_5].sel(season=SEASON_5)
    _fig, axs_13 = plt.subplots(1, 2, figsize=(6.5, 3), layout='constrained')
    axs_13[0].stairs(pdfs_5[0], edges_5, fill=True, alpha=0.1, color='k', label=YEARS_6[0])
    axs_13[0].stairs(pdfs_5[1], edges_5, lw=2, label=YEARS_6[1])
    axs_13[0].stairs(pdfs_5[2], edges_5, lw=2, label=YEARS_6[2])
    axs_13[0].axvline(ls='--', c='gray', lw=0.8)
    axs_13[0].legend()
    axs_13[0].set_xlabel('${\\circ}$lon')
    axs_13[0].set_ylabel('Density')
    axs_13[0].set_yticks([])
    axs_13[1].plot(width_lower.year, _sel_(width_lower), c='blue', label='Cold')
    axs_13[1].plot(width_lower.year, _sel_(width_upper), c='r', label='Warm')
    axs_13[1].plot(width_lower.year, _sel_(width_mean), c='k', label='Avg.')
    axs_13[1].set_ylabel('$^{\\circ}$C')
    axs_13[1].set_xlabel('Year')
    axs_13[1].set_yticks([2, 3])
    axs_13[1].set_xticks(YEARS_6)
    axs_13[1].axvline(2010, ls='--', lw=0.8, c='gray')
    axs_13[1].legend()
    axs_13[1].yaxis.tick_right()
    axs_13[1].yaxis.set_label_position('right')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ELI and precip
    """)
    return


@app.function
def average_interval(data, idx="eli_05", q_interval=[0.45, 0.55], avg_dim="s"):
    """average over interval"""

    ## boolean
    above_lb = data[idx] > data[idx].quantile(q=q_interval[0], dim="s")
    below_ub = data[idx] < data[idx].quantile(q=q_interval[1], dim="s")

    return data.where(above_lb & below_ub).mean(avg_dim)


@app.cell
def _(DATA_FP, anom_1, forced_1, get_windowed, np, pathlib, pd, src, xr):
    YEARS_7 = [1870, 2010, 2080]
    _eli = xr.open_dataset(pathlib.Path(DATA_FP, 'cesm/eli.nc'))
    _eli_forced, _eli_anom = src.utils.separate_forced(_eli)
    eli_total = _eli_forced + _eli_anom
    data = xr.merge([eli_total, forced_1['pr'] + anom_1['pr'], forced_1['sst'] + anom_1['sst']])
    data = get_windowed(data).sel(year=YEARS_7)
    data_seasonal = data.resample({'time': 'QS-DEC'}).mean()
    x_dec = src.utils.sel_month(data_seasonal, 12)
    x_dec = x_dec.stack(s=['member', 'time'])
    x_mar = src.utils.sel_month(data_seasonal, 6)
    x_mar = x_mar.stack(s=['member', 'time'])
    edges_6 = np.linspace(140, 240, 20)
    _get_pdf = lambda y: src.utils.get_empirical_pdf(y.values, edges=edges_6)[0]
    pdfs_dec = [_get_pdf(x_dec['eli_05'].sel(year=year)) for year in YEARS_7]
    pdfs_mar = [_get_pdf(x_mar['eli_05'].sel(year=year)) for year in YEARS_7]
    comps = []
    for x_ in [x_dec, x_mar]:
        comps_ = []
        for interval in [[0.025, 0.075], [0.225, 0.275], [0.475, 0.525], [0.725, 0.775], [0.925, 0.975]]:
            comps_.append(average_interval(x_, q_interval=interval))
        comps.append(xr.concat(comps_, dim=pd.Index(np.array([0.05, 0.25, 0.5, 0.75, 0.95]), name='q')))
    comps = xr.concat(comps, dim=pd.Index(['DJF', 'MAM'], name='season'))
    comps = src.utils.reconstruct_wrapper(xr.merge([comps, forced_1[['sst_comp', 'pr_comp']]]))
    return YEARS_7, comps, edges_6, pdfs_dec, pdfs_mar


@app.cell
def _(YEARS_7, cmocean, comps, edges_6, pdfs_dec, pdfs_mar, plt, sns):
    merimean = lambda x: x.sel(latitude=slice(-5, 5)).mean('latitude')
    _fig, axs_14 = plt.subplots(2, 3, figsize=(9, 6), layout='constrained')
    for _j, pdfs_ in enumerate([pdfs_dec, pdfs_mar]):
        axs_14[_j, 0].stairs(pdfs_[0], edges_6, fill=True, alpha=0.1, color='k', label=YEARS_7[0])
        axs_14[_j, 0].stairs(pdfs_[1], edges_6, lw=2, label=YEARS_7[1])
        axs_14[_j, 0].stairs(pdfs_[2], edges_6, lw=2, label=YEARS_7[2])
        axs_14[_j, 0].set_xlim([145, 245])
        axs_14[_j, 0].legend()
        axs_14[_j, 0].set_xlabel('${\\circ}$lon')
        axs_14[_j, 0].set_ylabel('Density')
        axs_14[_j, 0].set_yticks([])
    _colors = [sns.color_palette()[i] for i in [0, 3, 7]]
    CMAP = cmocean.cm.balance
    CMAP_NORM = plt.Normalize(vmin=-1, vmax=5)
    for _j, season in enumerate(comps.season):
        for _ax, year in zip(axs_14[_j, 1:], [2010, 2080]):
            _sel_ = lambda x: 86000.0 * merimean(x['pr'].sel(year=year, season=season))
            for i, _q in enumerate(comps.q):
                _ax.plot(comps.longitude, _sel_(comps).sel(q=_q), c=CMAP(CMAP_NORM(i)))
            _ax.plot(comps.longitude, _sel_(comps).sel(q=0.5), c='k')
    for _ax in axs_14[:, 1:].flatten():
        _ax.set_xlim([145, 275])
    for _ax in axs_14[:, 1:].flatten():
        _ax.set_ylim([-0.5, 16])
        _ax.axhline(0, ls='--', c='k', lw=0.8)
        _ax.set_xticks([145, 210, 275])
    axs_14[0, -1].set_yticks([])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute
    """)
    return


@app.cell
def _(anom_1, forced_1, get_windowed, src):
    _fn = src.utils.get_nino4
    XVAR = 'T_4_rel'
    _prQ_n34_anom = get_windowed(src.utils.reconstruct_wrapper(anom_1[['pr', 'pr_comp', 'nhf', 'nhf_comp']], fn=_fn))
    _prQ_n34_forced = get_windowed(src.utils.reconstruct_wrapper(forced_1[['pr', 'pr_comp', 'nhf', 'nhf_comp']], fn=_fn))
    prQ_n34_3 = _prQ_n34_forced + _prQ_n34_anom
    return (prQ_n34_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Stats
    """)
    return


@app.cell
def _(ax_kwargs, plt, prQ_n34_3, src):
    _PLOT_VAR = 'pr'
    sel_6 = lambda x, m: src.utils.sel_month(x, m)
    prep = lambda x: x.resample({'time': 'QS-NOV'}).mean().transpose('time', 'member', ...)
    x0_3 = 86000.0 * prep(prQ_n34_3[_PLOT_VAR])
    x1_3 = prep(prQ_n34_3['nhf'])
    M0 = 11
    M1 = 2
    _fig, axs_15 = plt.subplots(1, 2, figsize=(6.5, 3), layout='constrained')
    _x0_plot = x0_3.sel(year=2010)
    x1_plot_2 = x1_3.sel(year=2010)
    x2_plot = x0_3.sel(year=2080)
    x3_plot = x1_3.sel(year=2080)
    axs_15[0].scatter(sel_6(_x0_plot, M0), sel_6(x1_plot_2, M0), s=3, alpha=0.5)
    axs_15[0].scatter(sel_6(x2_plot, M0), sel_6(x3_plot, M0), s=3, alpha=0.5)
    axs_15[1].scatter(sel_6(_x0_plot, M1), sel_6(x1_plot_2, M1), s=3, alpha=0.5)
    axs_15[1].scatter(sel_6(x2_plot, M1), sel_6(x3_plot, M1), s=3, alpha=0.5)
    axs_15[0].axhline(0, **ax_kwargs)
    for _ax in axs_15:
        _ax.axvline(0, **ax_kwargs)
    axs_15[0].set_ylabel('$W~m^{-2}$')
    axs_15[0].set_xlabel('$^{\\circ}C$')
    src.utils.set_lims(axs_15)
    plt.gca()
    return M0, M1, sel_6, x1_plot_2, x3_plot


@app.cell
def _(M0, M1, np, plt, sel_6, src, x1_plot_2, x3_plot):
    edges_7 = np.linspace(0, 17, 20)
    _get_pdf = lambda y: src.utils.get_empirical_pdf(86000.0 * y.values, edges=edges_7)[0]
    pdfs_6 = [_get_pdf(sel_6(x1_plot_2, M0)), _get_pdf(sel_6(x3_plot, M0)), _get_pdf(sel_6(x1_plot_2, M1)), _get_pdf(sel_6(x3_plot, M1))]
    _fig, axs_16 = plt.subplots(1, 2, figsize=(6.5, 3), layout='constrained')
    axs_16[0].stairs(pdfs_6[0], edges_7, lw=2)
    axs_16[0].stairs(pdfs_6[1], edges_7, lw=2)
    axs_16[1].stairs(pdfs_6[2], edges_7, lw=2)
    axs_16[1].stairs(pdfs_6[3], edges_7, lw=2)
    for _ax in axs_16:
        _ax.axvline(ls='--', c='gray', lw=0.8)
        _ax.set_xlabel('${\\circ}$C')
    axs_16[0].set_ylabel('Density')
    axs_16[1].set_yticks([])
    src.utils.set_lims(axs_16)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
