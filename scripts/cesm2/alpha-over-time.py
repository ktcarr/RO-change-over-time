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

    # import gsw

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return np, plt, src, tqdm, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funcs
    """)
    return


@app.cell
def _():
    def get_Q(data):
        """get heat flux over region using specified fn"""

        # ### reconstruct flux over region
        # data_ = src.utils.reconstruct_wrapper(data, fn=fn)

        ## convert units to K/yr
        sec_per_mo = 8.64e4 * 30
        sec_per_yr = 12 * sec_per_mo
        rho = 1.02e3
        Cp = 4.2e3
        H = 50
        # H = 80

        ## apply scaling fac.
        Q = data * sec_per_yr / (rho * Cp * H)

        return Q

    def sel_eq(data):
        """select equatorial region"""
        idx = dict(latitude=slice(-5,5), longitude=slice(120,280))
        return data.sel(idx).mean("latitude")

    def sel_epac(data):
        """select equatorial region"""
        idx = dict(longitude=slice(210,270))
        return data.sel(idx).mean("longitude")

    return get_Q, sel_epac, sel_eq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### surface heat fluxes
    """)
    return


@app.cell
def _(src):
    flux_forced, flux_anom = src.utils.load_flux_data()
    for _v in ['flns', 'lhflx']:
    ## switch sign convection to be positive into the ocean
        flux_anom[_v] = -1 * flux_anom[_v]
        flux_forced[_v] = -1 * flux_forced[_v]
    ## load into memory
    flux_anom.load()
    return flux_anom, flux_forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### cloud fraction
    """)
    return


@app.cell
def _(src, xr):
    ## load
    cld_forced, cld_anom = src.utils.load_cloud_data()

    ## Get total cloud fraction
    cld_comp, cld_forced_scores = src.utils.split_components(cld_forced)
    cld_comp = cld_comp.rename({v:f"{v}_comp" for v in list(cld_comp)})

    _, cld_anom_scores = src.utils.split_components(cld_anom)
    cld_total_scores = cld_forced_scores + cld_anom_scores
    cld_total = xr.merge([cld_total_scores, cld_comp])
    return cld_anom, cld_anom_scores, cld_comp, cld_forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### $T$, $h$
    """)
    return


@app.cell
def _(src):
    Th = src.utils.load_cesm_indices().compute()
    return (Th,)


@app.cell
def _(Th, xr):
    T = xr.merge(
        [(Th["T_3"]**i).rename(f"T{i}") for i in range(4)]
    )
    return (T,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### zonal grad.
    """)
    return


@app.cell
def _(src):
    forced, _ = src.utils.load_consolidated()
    forced = forced[["sst","sst_comp"]]

    ## compute diff. def of Tw
    sel_lat = lambda x: x.sel(latitude=slice(-5, 5)).mean(["latitude", "longitude"])
    get_Tw = lambda x: sel_lat(x.sel(longitude=slice(140, 190)))
    get_Te = lambda x: sel_lat(x.sel(longitude=slice(240, 280)))
    get_dTdx = lambda x: get_Tw(x) - get_Te(x)

    ## compute dTdx
    dTdx = src.utils.reconstruct_wrapper(forced, get_dTdx)
    dTdx = dTdx.rename({"sst":"dTdx"})
    return (dTdx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### merge
    """)
    return


@app.cell
def _(T, cld_anom_scores, cld_forced, dTdx, flux_anom, sel_eq, src, xr):
    data = xr.merge(
        [
            T,
            flux_anom[[v for v in list(flux_anom) if "comp" not in v]],
            cld_anom_scores,
            dTdx,
        ]
    )

    ## window in time
    get_windowed = lambda x : src.utils.get_windowed(x, stride=120, window_size=480)
    data = get_windowed(data)
    data = data.sel(year=slice(None,2080))

    ## compute clim.
    cld_clim = get_windowed(cld_forced).groupby("time.month").mean()
    cld_clim_eq = src.utils.reconstruct_wrapper(cld_clim, fn=sel_eq)
    return cld_clim_eq, data, get_windowed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## fit
    """)
    return


@app.cell(hide_code=True)
def _(np, xr):
    def regress_pinv(X, x_vars, y_var):
        """do nonlinear regression"""

        ## prep data
        # prep = lambda x: x.transpose("member", ...)
        prep = lambda x: x.stack(s=["time", "member"]).transpose("year",..., "s")

        X_ = prep(X[x_vars].to_dataarray(dim="v"))
        Y_ = prep(X[y_var]).transpose("year","mode",...)

        ## empty array to hold results
        m = xr.DataArray(
            coords={"year": X.year, "mode": Y_.mode, "v": x_vars},
            dims=["year", "mode", "v"],
        )

        ## do regression
        X_pinv = np.linalg.pinv(X_.values)
        m.values = np.einsum("...ki,...ij->...kj", Y_.values, X_pinv)

        return m.rename({"v": "param"})


    def regress_pinv_wrapper(X, x_vars, y_var):
        return X.groupby("time.month").map(regress_pinv, x_vars=x_vars, y_var=y_var)

    return (regress_pinv_wrapper,)


@app.cell
def _(cld_anom_scores, data, regress_pinv_wrapper, tqdm, xr):
    flux_types=["fsns", "lhflx", "flns"]
    cld_types = list(cld_anom_scores)
    all_types = flux_types + cld_types

    ## empty array to hold results
    alpha = []

    ## loop thru flux types
    for ftype in tqdm.tqdm(all_types):

        kwargs = dict(x_vars=["T0", "T1", "T2", "T3"], y_var=ftype)
        alpha.append(-regress_pinv_wrapper(data, **kwargs))

    ## merge
    alpha = xr.merge([a.rename(f) for a, f in zip(alpha, all_types)])
    return alpha, flux_types


@app.cell
def _(
    alpha,
    cld_comp,
    flux_anom,
    flux_types,
    get_Q,
    sel_epac,
    sel_eq,
    src,
    xr,
):
    ## add components
    alpha_1 = xr.merge([alpha, flux_anom[[f'{f}_comp' for f in flux_types]], cld_comp])
    alpha_eq = src.utils.reconstruct_wrapper(alpha_1, fn=sel_eq)
    alpha_epac = src.utils.reconstruct_wrapper(alpha_1, fn=sel_epac)
    Q = get_Q(alpha_eq[flux_types])
    ## reconstruct
    ## compute to heat flux
    Q_epac = get_Q(alpha_epac[flux_types])
    return Q, Q_epac, alpha_eq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## plot results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### timeseries
    """)
    return


@app.cell
def _(data, plt):
    plt.plot(data["dTdx"].mean("time"))
    return


@app.cell
def _(Q, data, flux_types, plt, src):
    sel_ = lambda x: x.sel(longitude=slice(210, 270), month=slice(6, 12)).mean(['longitude', 'month'])
    delta = lambda x: sel_(x) - sel_(x).isel(year=0)
    delta2 = lambda x: x - x.isel(year=0)
    _fig, _axs = plt.subplots(1, 4, figsize=(12, 3), layout='constrained')
    for _ax, _p in zip(_axs, list(Q.param.values)):
        for f in flux_types:
            _ax.plot(Q.year, delta(Q[f].sel(param=_p)), label=f)
        _ax.axhline(0, ls='--', c='k', lw=1)
        _ax.plot(data.year, 0.2 * delta2(data['dTdx'].mean('time')), c='k', ls='--', label='$T_w-T_e$ (scaled)')
        _ax.legend()
        _ax.set_title(f'Feedback change ({_p})')
    src.utils.set_lims(_axs)  # ax2 = ax.twinx()
    plt.gca()  # ax.set_ylabel("
    return (delta2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot spatial structure
    """)
    return


@app.cell
def _(Q, plt, src):
    sel__1 = lambda x: x.sel(month=slice(6, 12)).mean(['month'])
    _fig, _axs = plt.subplots(1, 4, figsize=(10, 2.5), layout='constrained')
    for _ax, _p in zip(_axs, Q.param.values):
        for _y in [1870, 2000, 2080]:
            _ax.plot(Q.longitude, sel__1(Q['fsns']).sel(year=_y, param=_p))
        _ax.set_yticks([])
        _ax.axhline(0, ls='--', c='k', lw=0.8)
        _ax.set_title(_p)
    src.utils.set_lims(_axs)
    src.utils.add_vticks(_axs, xticks=[210, 270], xlines=[210, 270])
    plt.gca()
    return


@app.cell
def _(Q, plt, src):
    sel__2 = lambda x: x.sel(month=slice(6, 12)).mean(['month'])
    _fig, _axs = plt.subplots(1, 3, figsize=(7.5, 2.5), layout='constrained')
    for _ax, _y in zip(_axs, [1870, 2000, 2080]):
        _ax.plot(Q.longitude, sel__2(Q['fsns']).sel(year=_y, param='T0'))
        _ax.plot(Q.longitude, -sel__2(Q['fsns']).sel(year=_y, param='T2'))
        _ax.set_yticks([])
        _ax.axhline(0, ls='--', c='k', lw=0.8)
        _ax.set_title(_y)
    src.utils.set_lims(_axs)
    src.utils.add_vticks(_axs, xticks=[210, 270], xlines=[210, 270])
    plt.gca()
    return


@app.cell
def _(Q, Q_epac, plt, src):
    sel__3 = lambda x: x.sel(month=slice(6, 12)).mean(['month'])
    _fig, _axs = plt.subplots(1, 4, figsize=(10, 2.5), layout='constrained')
    for _ax, _p in zip(_axs, Q.param.values):
        for _y in [1870, 2000]:
            _ax.plot(Q_epac.latitude, sel__3(Q_epac['fsns']).sel(year=_y, param=_p))
        _ax.set_yticks([])
        _ax.axhline(0, ls='--', c='k', lw=0.8)
        _ax.set_title(_p)
    src.utils.set_lims(_axs)
    src.utils.add_vticks(_axs, xticks=[-5, 5], xlines=[-5, 5])
    plt.gca()
    return (sel__3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot climatology
    Could also plot rainfall
    """)
    return


@app.cell
def _(flux_forced, get_Q, get_windowed, sel_eq, src):
    fsns_forced = get_Q(
        src.utils.reconstruct_wrapper(
            get_windowed(flux_forced[["fsns","fsns_comp"]]), 
            fn=sel_eq
        )["fsns"]
    )
    fsns_clim = fsns_forced.groupby("time.month").mean()
    return (fsns_clim,)


@app.cell
def _(Q, fsns_clim, plt, sel__3, src):
    _fig, _ax = plt.subplots(figsize=(4, 3), layout='constrained')
    for _y in [1870, 2000]:
        _ax.plot(Q.longitude, sel__3(fsns_clim).sel(year=_y))
    src.utils.add_vticks([_ax], xticks=[210, 270], xlines=[210, 270])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load cloud fraction
    """)
    return


@app.cell
def _(cld_clim_eq, plt, src):
    sel__4 = lambda x: x.sel(month=slice(6, 12)).mean(['month'])
    _fig, _axs = plt.subplots(1, 3, figsize=(7.5, 2.5), layout='constrained')
    for _ax, _v in zip(_axs, list(cld_clim_eq)):
        for _y in [1870, 2010, 2080]:
            _ax.plot(cld_clim_eq.longitude, sel__4(cld_clim_eq[_v]).sel(year=_y), label=_y)
        _ax.set_yticks([])
        _ax.axhline(0, ls='--', c='k', lw=0.8)
        _ax.set_title(_v[-3:])
    _axs[0].set_yticks([0, 0.5, 1])
    _axs[0].set_ylabel('Cloud fraction')
    _axs[-1].legend()
    src.utils.set_lims(_axs)
    src.utils.add_vticks(_axs, xticks=[210, 270], xlines=[210, 270])
    plt.gca()
    return (sel__4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Timeseries
    """)
    return


@app.cell
def _(cld_clim_eq, sel__4):
    Q_cld = sel__4(cld_clim_eq.sel(longitude=slice(210, 270)).mean('longitude'))
    return (Q_cld,)


@app.cell
def _(Q_cld, data, delta2, plt):
    delt = lambda x: x - x.isel(year=0)
    _fig, _ax = plt.subplots(figsize=(4, 3))
    for _v in list(Q_cld):
        _ax.plot(Q_cld.year, delt(Q_cld[_v]), label=_v)
    _ax.axhline(0, ls='--', c='k', lw=0.8)
    _ax.plot(data.year, -0.05 * delta2(data['dTdx'].mean('time')), c='k', ls='--', label='$T_w-T_e$ (scaled)')
    _ax.set_title('Change in climatology')
    ## label/formatting
    _ax.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter
    """)
    return


@app.cell
def _(Th, cld_anom, get_windowed, src, xr):
    # Q_cld_tot = src.utils.reconstruct_wrapper(
    #     cld_total, fn=src.utils.get_nino3,
    # )

    Q_cld_tot = src.utils.reconstruct_wrapper(
        cld_anom, fn=src.utils.get_nino3,
    )

    ## add Q data and window
    Th_ = get_windowed(xr.merge([Th, Q_cld_tot]))
    return Q_cld_tot, Th_


@app.cell
def _(alpha_eq):
    alpha_eq
    return


@app.cell
def _(Q_cld_tot, Th_, plt, src):
    sel__5 = lambda x: x.sel(time=x.time.dt.month == 12)
    _fig, _axs = plt.subplots(3, 3, figsize=(8, 8))
    for j, _y in enumerate([1870, 2000, 2080]):
        axs_ = _axs[j]
        for _ax, _v in zip(axs_, list(Q_cld_tot)):
            _ax.scatter(sel__5(Th_['T_3'].sel(year=_y)), sel__5(Th_[_v].sel(year=_y)), s=7)
            if j == 0:
                _ax.set_title(_v[-3:])
            ax_kwargs = dict(ls='--', c='k', lw=1)
            _ax.axhline(**ax_kwargs)
            _ax.axvline(**ax_kwargs)
            if j < 2:
                _ax.set_xticks([])
            else:
                _ax.set_xticks([-3, 0, 3])
                _ax.set_xlabel('$T_3$')
        axs_[0].set_ylabel('Fraction')
    src.utils.set_lims(_axs)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get fits over time
    """)
    return


@app.cell
def _(Q, alpha_eq, plt, src):
    sel__6 = lambda x: x.sel(month=slice(6, 12)).mean(['month'])
    _fig, _axs = plt.subplots(1, 4, figsize=(10, 2.5), layout='constrained')
    for _ax, _p in zip(_axs, alpha_eq.param.values):
        for _y in [1870, 2000, 2080]:
            _ax.plot(Q.longitude, -sel__6(alpha_eq['cldmed']).sel(year=_y, param=_p))
        _ax.set_yticks([])
        _ax.axhline(0, ls='--', c='k', lw=0.8)
        _ax.set_title(_p)
    src.utils.set_lims(_axs)
    src.utils.add_vticks(_axs, xticks=[210, 270], xlines=[210, 270])
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
