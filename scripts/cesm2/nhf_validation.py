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
    # NHF validation
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

    # import gsw

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return DATA_FP, np, os, pathlib, plt, scipy, src, tqdm, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funcs
    """)
    return


@app.cell
def _(os, pathlib):
    def get_Te(x):
        idx = dict(latitude=slice(-5, 5), longitude=slice(230, 280))
        return x.sel(idx).mean(["longitude", "latitude"])


    def get_Tw(x):
        idx = dict(latitude=slice(-5, 5), longitude=slice(180, 230))
        return x.sel(idx).mean(["longitude", "latitude"])


    def save(fig, fname, dpi=800, do_save=False):
        """save figure to file"""

        ## get save directory
        save_dir = pathlib.Path(os.environ["SAVE_FP"], "june-updates")

        ## get fname
        fname = f"{fname}.pdf"

        if do_save:
            fig.savefig(save_dir / fname, dpi=dpi, format="pdf")

        return

    return get_Te, get_Tw, save


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### funcs to get best fit
    """)
    return


@app.cell
def _(np, scipy, xr):
    def psi_poly(T, c, b, a, e):
        """base transfer function"""
        # return e * x**3 + c * x**2 + b * x + a
        return (c / 3) * T**3 + (b / 2) * T**2 + a * T + e


    def get_psi_poly(x, y):
        """get transfer function from data"""

        ## prep data
        if "member" in x.dims:
            stack = lambda x: x.stack(s=["time", "member"]).values
            x = stack(x)
            y = stack(y)

        ## get parameter fit
        p, _ = scipy.optimize.curve_fit(
            f=psi_poly,
            xdata=x,
            ydata=y,
        )

        ## define transfer func
        psi = lambda T: psi_poly(T, c=p[0], b=p[1], a=p[2], e=p[3])

        ## put params in dataarray
        params = xr.DataArray(p, coords=dict(param=["c", "b", "a", "e"])).rename("params")

        return psi, params


    def psi_exp(x, a, b, c):
        """base transfer function"""
        return c * np.exp(a * x) + b


    def get_psi_exp(x, y):
        """get transfer function from data"""

        ## prep data
        if "member" in x.dims:
            stack = lambda x: x.stack(s=["time", "member"]).values
            x = stack(x)
            y = stack(y)

        ## get parameter fit
        p, _ = scipy.optimize.curve_fit(
            f=psi_exp,
            xdata=x,
            ydata=y,
            p0=np.array([0.7, 1.7, 1]),
        )

        ## define transfer func
        psi = lambda x: psi_exp(x, a=p[0], b=p[1], c=p[2])

        ## put parameters in array
        params = xr.DataArray(p, coords=dict(param=["a", "b", "c"])).rename("params")

        return psi, params


    def get_psi_wrapper(xy, x_var, y_var, get_psi_func=get_psi_poly):
        """get transfer function from data"""

        ## get transfer func
        psi_func, p = get_psi_func(x=xy[x_var], y=xy[y_var])

        ## get coefficient array (standardized units)
        z = np.arange(-3.5, 3.6, 0.1)
        z = xr.DataArray(z, coords=dict(sigma=z))

        ## evaluate function
        psi_eval = xr.zeros_like(z)
        psi_eval.values = psi_func(z.values)

        ## merge eval and params
        return xr.merge([psi_eval.rename("psi_eval"), p])

    return get_psi_exp, get_psi_wrapper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## load data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ERA5
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### heat/radiation fluxes
    """)
    return


@app.cell
def _(DATA_FP, xr):
    ## nhf
    nhf_era5 = xr.open_dataset(DATA_FP / "era5_nhf_glade.nc").drop_vars("utc_date")

    ## make sure latitude is ascending
    nhf_era5 = nhf_era5.reindex({"latitude": nhf_era5.latitude[::-1]})

    ## rename flux variables
    varname_dict = dict(
        MSLHF="lhf",
        MSSHF="shf",
        MSNLWRF="lw",
        MSNLWRFCS="lw_cs",
        MSNSWRF="sw",
        MSNSWRFCS="sw_cs",
    )
    nhf_era5 = nhf_era5.rename(varname_dict)

    ## net heat flux
    ## Note: "The ECMWF convention for vertical fluxes is positive downwards."
    nhf_era5["nhf"] = nhf_era5["sw"] + nhf_era5["lw"] + nhf_era5["lhf"] + nhf_era5["shf"]
    return (nhf_era5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SST
    """)
    return


@app.cell
def _(nhf_era5, pathlib, xr):
    ## load ERSST
    REF_FP = pathlib.Path("/glade/campaign/cgd/cas/asphilli/CVDP-OBS/")

    sst_ersst = xr.merge(
        [
            xr.open_dataset(REF_FP / "ersstv5.185401-202412.nc")["ssta"],
            xr.open_dataset(REF_FP / "ersstv5.185401-202412.nc")["sst"],
            # xr.open_dataset(REF_FP / "gpcp.mon.mean.197901-202403.nc")["precip"],
        ]
    )

    sst_ersst = sst_ersst.rename({"lon": "longitude", "lat": "latitude"})
    sst_ersst["time"] = xr.date_range(start="1854-01", end="2024-12", freq="MS")
    sst_ersst = sst_ersst.sel(time=nhf_era5.time)

    ## compute Ttrop
    sst_ersst["Ttrop"] = (
        sst_ersst["sst"].sel(latitude=slice(-15, 15)).mean(["latitude", "longitude"])
    )
    sst_ersst["sst_rel"] = sst_ersst["sst"] - sst_ersst["Ttrop"]

    ## rename
    sst_ersst = sst_ersst.rename({"sst": "sst_total", "ssta": "sst"})
    return REF_FP, sst_ersst


@app.cell
def _(REF_FP, sst_ersst, xr):
    ## load precip
    pr_ref = xr.open_dataset(REF_FP / "gpcp.mon.mean.197901-202403.nc")["precip"]
    pr_ref = (
        pr_ref.rename({"lon": "longitude", "lat": "latitude"})
        .interp({"latitude": sst_ersst.latitude, "longitude": sst_ersst.longitude})
        .rename("pr")
    )
    return (pr_ref,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CESM2
    """)
    return


@app.cell
def _(DATA_FP, pathlib, src, xr):
    ## load data
    forced, anom = src.utils.load_consolidated()
    flux_forced, flux_anom = src.utils.load_flux_data()
    ## load flux data
    flux_anom['lhflx'] = -flux_anom['lhflx']
    flux_forced['lhflx'] = -flux_forced['lhflx']
    ## update sign of LHFLX
    anom = xr.merge([anom, flux_anom])
    forced = xr.merge([forced, flux_forced])
    anom = anom.sel(time=slice('1950', '1980')).compute()
    ## merge
    forced = forced.sel(time=slice('1950', '1980')).compute()
    varname_dict_cesm = dict(lhflx='lhf', flns='lw', fsns='sw', fsnsc='sw_cs')
    for v0 in list(varname_dict_cesm.keys()):
    ## subset in time
        varname_dict_cesm[f'{v0}_comp'] = f'{varname_dict_cesm[v0]}_comp'
    anom = anom.rename(varname_dict_cesm)
    # anom = anom.sel(time=slice("2070", "2100")).compute()
    # forced = forced.sel(time=slice("2070", "2100")).compute()
    forced = forced.rename(varname_dict_cesm)
    ## rename flux variables
    varnames = ['sst', 'pr', 'nhf', 'sw', 'lw', 'lhf', 'sw_cs']
    varnames = varnames + [f'{v}_comp' for v in varnames]
    anom = anom[varnames]
    forced = forced[varnames]
    Ttrop_mod = xr.open_dataset(pathlib.Path(DATA_FP, 'cesm/trop_sst.nc'))
    anom['Ttrop'] = Ttrop_mod['trop_sst_15'].sel(time=anom.time)
    anom['sst_total'] = anom['sst'] + forced['sst']
    ## add component information
    anom['sst_total_comp'] = anom['sst_comp']
    anom['pr'] = anom['pr'] * 86000.0
    ## update names
    ## subset for given variables
    ## load tropical SST for CESM2
    ## add total SST to anom for convenience
    ## convert precip to mm/day
    forced['pr'] = forced['pr'] * 86000.0
    return anom, forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load mixed layer temp
    """)
    return


@app.cell
def _(anom, forced, src, xr):
    ## load thermocline data
    Th_3d = src.utils.load_cesm_3d().sel(time=anom.time)
    Th_3d = Th_3d.interp({'longitude': anom.longitude})
    Th_3d_forced = Th_3d.mean('member')
    Th_3d_anom = Th_3d - Th_3d_forced
    anom_1 = xr.merge([anom, Th_3d_anom])
    ## merge
    forced_1 = xr.merge([forced, Th_3d_forced])
    return anom_1, forced_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute indices
    """)
    return


@app.cell
def _(get_Te, get_Tw, src):
    fns = dict(
        _3=src.utils.get_nino3,
        _34=src.utils.get_nino34,
        _4=src.utils.get_nino4,
        _e=get_Te,
        _w=get_Tw,
    )
    return (fns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ERA5
    """)
    return


@app.cell
def _(fns, nhf_era5, pr_ref, sst_ersst, tqdm, xr):
    idx_obs = xr.Dataset()
    # idx_mod = xr.Dataset()
    for ds in [nhf_era5, sst_ersst[['sst', 'sst_total']], pr_ref.to_dataset()]:
        for fn_name, _fn in tqdm.tqdm(fns.items()):
            for _v in list(ds):
                idx_obs[f'{_v}{fn_name}'] = _fn(ds[_v])
    for _v in list(idx_obs):
    ## add relative SST
        if 'sst_total' in _v:
            idx_obs[f'sst_rel_{_v[10:]}'] = idx_obs[_v] - sst_ersst['Ttrop']
    return (idx_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### CESM2
    """)
    return


@app.cell
def _(anom_1, fns, forced_1, src, tqdm, xr):
    idx_mod = xr.Dataset()
    idx_mod_forced = xr.Dataset()
    for n, _fn in tqdm.tqdm(fns.items()):
        idx_mod_ = src.utils.reconstruct_wrapper(anom_1, _fn)
        idx_mod_forced_ = src.utils.reconstruct_wrapper(forced_1, _fn)
        name_dict = dict(zip(list(idx_mod_), [f'{i}{n}' for i in list(idx_mod_)]))
        idx_mod = xr.merge([idx_mod, idx_mod_.rename(name_dict)])
        name_dict = dict(zip(list(idx_mod_forced_), [f'{i}{n}' for i in list(idx_mod_forced_)]))
        idx_mod_forced = xr.merge([idx_mod_forced, idx_mod_forced_.rename(name_dict)])
    for _v in list(idx_mod):
        if 'sst_total' in _v:
            idx_mod[f'sst_rel_{_v[10:]}'] = idx_mod[_v] - anom_1['Ttrop']
    idx_mod_total = idx_mod + idx_mod_forced
    return idx_mod, idx_mod_total


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### resample to NDJ
    """)
    return


@app.cell
def _(idx_mod, idx_mod_total, idx_obs):
    def resample_ndj(x):
        """resample data to NDJ avg"""
        x = x.resample({'time': 'QS-NOV'}).mean().isel(time=slice(4, -4, 4))
        return x  ## resample and trim

    def resample(x):
        """resample data to NDJ avg"""  ## remove mean
        x = x.resample({'time': 'QS-NOV'}).mean().isel(time=slice(1, -4, None))
        return x
    idx_obs_1 = resample_ndj(idx_obs)
    idx_mod_1 = resample_ndj(idx_mod)
    # ## subset for two seasons
    # SEL_NDJ = lambda x : x.sel(time=(x.time.dt.month==11))
    # SEL_ASO = lambda x : x.sel(time=(x.time.dt.month==8))
    # idx_obs = SEL_NDJ(idx_obs)
    # idx_mod = SEL_NDJ(idx_mod)
    # idx_mod_total = SEL_NDJ(idx_mod_total)
    idx_mod_total_1 = resample_ndj(idx_mod_total)  ## resample and trim  ## remove mean
    return idx_mod_1, idx_mod_total_1, idx_obs_1, resample_ndj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    normalized version
    """)
    return


@app.cell
def _(idx_mod_1, idx_obs_1):
    for _v in list(idx_obs_1):
        idx_obs_1[f'{_v}_norm'] = idx_obs_1[_v] / idx_obs_1[_v].std()
    for _v in list(idx_mod_1):
        idx_mod_1[f'{_v}_norm'] = idx_mod_1[_v] / idx_mod_1[_v].std()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ERA5
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter and timeseries
    """)
    return


@app.cell
def _(idx_obs_1, plt, src):
    yr = idx_obs_1.time.dt.year
    ax_kwargs = dict(c='k', ls='--', lw=0.8, zorder=0.5)
    prep = lambda x: x - x.mean()
    for _suf in ['e', 'w']:
        print(f'\nLoc: {_suf}')
        _fig, _axs = plt.subplots(2, 5, figsize=(10, 4.5), layout='constrained')
        for _j, _v in enumerate(['nhf', 'sw', 'lhf', 'lw', 'shf']):
            _axs[0, _j].scatter(prep(idx_obs_1[f'sst_{_suf}']), prep(idx_obs_1[f'{_v}_{_suf}']), s=20)
            _axs[0, _j].set_title(_v)
            _axs[0, _j].axvline(0, **ax_kwargs)
            _axs[0, _j].axhline(0, **ax_kwargs)
            _axs[0, _j].set_xticks([-2, 0, 2])
            _axs[0, _j].set_xlabel('ssta')
            _axs[1, _j].plot(yr, prep(idx_obs_1[f'{_v}_{_suf}']))
            _axs[1, _j].axhline(0, **ax_kwargs)
            _axs[1, _j].axvline(1979, **ax_kwargs)
            _axs[1, _j].set_xticks([1979, 2022])
        src.utils.set_lims(_axs[0])
        src.utils.set_lims(_axs[1])
        for _ax in _axs[:, 0]:
            _ax.set_ylabel('W m$^{-2}$')
        for _ax in _axs[:, 1:].flatten():
            _ax.set_yticks([])
        plt.show()
    return ax_kwargs, prep, yr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SST over time
    """)
    return


@app.cell
def _(ax_kwargs, idx_obs_1, np, plt, src, yr):
    _fig, _axs = plt.subplots(1, 3, figsize=(9, 2.25), layout='constrained')
    for _ax, _suf in zip(_axs[:2], ['w', 'e']):
        _ax.plot(yr, idx_obs_1[f'sst_{_suf}'])
        _ax.axhline(0, **ax_kwargs)
        _ax.axvline(1979, **ax_kwargs)
        _ax.set_xticks([1979, 2022])
    _axs[-1].scatter(idx_obs_1[f'sst_e'], idx_obs_1[f'sst_w'], s=15)
    _axs[-1].axhline(0, **ax_kwargs)
    _axs[-1].axvline(0, **ax_kwargs)
    _axs[-1].set_aspect('equal')
    zz = np.linspace(-2, 2.5)
    src.utils.set_lims(_axs[:2])
    _axs[1].set_yticks([])
    _axs[0].set_title('central SST')
    _axs[1].set_title('east SST')
    _axs[2].set_ylabel('central')
    _axs[2].set_xlabel('east')
    plt.gca()
    return


@app.cell
def _(ax_kwargs, get_psi_wrapper, idx_mod_1, idx_obs_1, np, plt, src):
    _psi_evals = np.empty(shape=2, dtype=object)
    _fig, _axs = plt.subplots(1, 2, figsize=(5.5, 2.75), layout='constrained')
    for _j, (_ax, _idx_, _s, ls) in enumerate(zip(_axs, [idx_obs_1, idx_mod_1], [20, 1], ['-', '--'])):
        _ax.scatter(_idx_[f'sst_e_norm'], _idx_[f'sst_w_norm'], s=_s)
        _psi_eval = get_psi_wrapper(_idx_, x_var='sst_e_norm', y_var='sst_w_norm')['psi_eval']
        _min_ = _idx_['sst_e_norm'].values.min()
        _max_ = _idx_['sst_e_norm'].values.max()
        _psi_evals[_j] = _psi_eval.sel(sigma=slice(_min_, _max_))
        _ax.plot(_psi_evals[_j].sigma, _psi_evals[_j], c='k', ls=ls)
        _ax.axhline(0, **ax_kwargs)
        _ax.axvline(0, **ax_kwargs)
        _ax.set_aspect('equal')
    src.utils.set_lims(_axs)
    _axs[1].set_yticks([])
    _axs[1].plot(_psi_evals[0].sigma, _psi_evals[0], c='k')
    _axs[0].set_title('ERSST')
    _axs[1].set_title('CESM2')
    plt.gca()
    return (ls,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Precip
    """)
    return


@app.cell
def _(
    ax_kwargs,
    get_psi_exp,
    get_psi_wrapper,
    idx_mod_1,
    idx_mod_total_1,
    idx_obs_1,
    ls,
    np,
    plt,
    src,
    xr,
):
    _fig, _axs = plt.subplots(2, 2, figsize=(5, 3.5), layout='constrained')
    for _j, _suf in enumerate(['w', 'e']):
        _xvar = f'sst_rel_{_suf}'
        _yvar = f'pr_{_suf}'
        _axs[0, _j].scatter(idx_obs_1[_xvar], idx_obs_1[_yvar], s=20)
        _axs[1, _j].scatter(idx_mod_1[_xvar], idx_mod_total_1[_yvar], s=1, alpha=0.5)
        psi_kwargs = dict(x_var=_xvar, y_var=_yvar, get_psi_func=get_psi_exp)
        psi_eval_obs = get_psi_wrapper(idx_obs_1.isel(time=~np.isnan(idx_obs_1[_yvar])), **psi_kwargs)['psi_eval']
        _max_ = idx_obs_1[_xvar].values.max() + 0.2
        _min_ = idx_obs_1[_xvar].values.min() + 0.2
        psi_eval_obs = psi_eval_obs.sel(sigma=slice(_min_, _max_))
        psi_data_mod = xr.merge([idx_mod_1[_xvar], idx_mod_total_1[_yvar]])
        psi_eval_mod = get_psi_wrapper(psi_data_mod, **psi_kwargs)['psi_eval']
        _max_ = psi_data_mod[_xvar].values.max()
        psi_eval_mod = psi_eval_mod.sel(sigma=slice(None, _max_))
        for _ax in _axs[:, _j]:
            _ax.plot(psi_eval_obs.sigma, psi_eval_obs, c='k', ls=ls, lw=2)
        _axs[1, _j].plot(psi_eval_mod.sigma, psi_eval_mod, c='k', lw=2)
        for _ax in _axs[:, _j]:
            _ax.axhline(0, **ax_kwargs)
            _ax.axhline(6, **ax_kwargs)
            _ax.axvline(0, **ax_kwargs)
        src.utils.set_lims(_axs)
    for _ax in _axs[:, 1]:
        _ax.set_yticks([])
    for _ax in _axs[:, 0]:
        _ax.set_yticks([0, 6, 12])
        _ax.set_ylabel('mm / day')
    for _ax in _axs[0, :]:
        _ax.set_xticks([])
    src.utils.set_lims(_axs)
    for _ax, _s in zip(_axs[:, -1], ['Obs', 'CESM2']):
        _ax.text(s=_s, x=1, y=0.5, transform=_ax.transAxes)
    _axs[0, 0].set_title('C. Pac')
    _axs[0, 1].set_title('E. Pac')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison between ERA5 and CESM2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Make plot
    """)
    return


@app.cell
def _(get_psi_wrapper, idx_mod_1, idx_obs_1, np, plt, prep, save, src):
    yr_1 = idx_obs_1.time.dt.year
    ax_kwargs_1 = dict(c='k', ls='--', lw=0.8, zorder=0.5)
    for _suf in ['3']:
        print(f'\nLoc: {_suf}')
        _psi_evals = np.empty(shape=(2, 4), dtype=object)
        _fig, _axs = plt.subplots(2, 4, figsize=(9, 4.5), layout='constrained')
        for _j, (_v, _label) in enumerate(zip(['nhf', 'sw', 'lhf', 'lw'], ['Net', 'Shortwave', 'Latent', 'Longwave'])):
            _axs[0, _j].set_title(_label)
            _axs[1, _j].set_xlabel('sst')
            _xvar, _yvar = (f'sst_{_suf}_norm', f'{_v}_{_suf}')
            for i, (_idx_, _s, ls_1, a, row_label) in enumerate(zip([idx_obs_1, idx_mod_1], [20, 1], ['-', '--'], [1, 0.5], ['ERA5', 'CESM2'])):
                _axs[i, _j].scatter(prep(_idx_[_xvar]), prep(_idx_[_yvar]), s=_s, alpha=a)
                _psi_eval = get_psi_wrapper(prep(_idx_), x_var=_xvar, y_var=_yvar)['psi_eval']
                _min_ = prep(_idx_[_xvar]).values.min()
                _max_ = prep(_idx_[_xvar]).values.max()
                _psi_evals[i, _j] = _psi_eval.sel(sigma=slice(_min_, _max_))
                _axs[i, _j].plot(_psi_evals[i, _j].sigma, _psi_evals[i, _j], c='k', ls=ls_1)
                _axs[i, _j].axvline(0, **ax_kwargs_1)
                _axs[i, _j].axhline(0, **ax_kwargs_1)
                _axs[i, _j].set_xticks([-2, 0, 2])
                if _j == 3:
                    _axs[i, _j].text(s=row_label, x=1.1, y=0.5, transform=_axs[i, _j].transAxes)
        for _j, _ax in enumerate(_axs[1]):
            _ax.plot(_psi_evals[0, _j].sigma, _psi_evals[0, _j], c='k', ls='-')
        src.utils.set_lims(_axs)
        for _ax in _axs[:, 0]:
            _ax.set_ylabel('W m$^{-2}$')
        for _ax in _axs[:, 1:].flatten():
            _ax.set_yticks([])
        for _ax in _axs[0, :]:
            _ax.set_xticks([])
        save(_fig, fname='Q-validation', do_save=False)
        plt.show()
    return (ax_kwargs_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PDFs
    """)
    return


@app.cell
def _(ax_kwargs_1, idx_mod_1, idx_obs_1, np, plt, src):
    edges = np.linspace(-5, 2.5, 17)
    _fig, _axs = plt.subplots(1, 2, figsize=(6.5, 2.5), layout='constrained')
    for _ax, _suf in zip(_axs, ['w', 'e']):
        _ax.set_title(_suf)
        _ax.axvline(0, **ax_kwargs_1)
        _ax.set_xlabel('rel. SST (K)')
        for _idx_, _label in zip([idx_obs_1, idx_mod_1], ['ERSST', 'CESM2']):
            pdf, _ = src.utils.get_empirical_pdf(_idx_[f'sst_rel_{_suf}'].values.flatten(), edges=edges)
            _ax.stairs(pdf, edges, label=_label)
    src.utils.set_lims(_axs)
    _axs[1].set_yticks([])
    _axs[0].set_ylabel('Density (K$^{-1}$)')
    _axs[0].legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### composites
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Merge data
    """)
    return


@app.cell
def _(resample_ndj):
    def trim(x):
        return x.sel(latitude=slice(-5, 5), longitude=slice(130, 280)).mean('latitude')

    def prep_1(x):
        return resample_ndj(trim(x))

    return prep_1, trim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ERA5
    """)
    return


@app.cell
def _(idx_obs_1, nhf_era5, pr_ref, prep_1, sst_ersst, xr):
    data_obs = prep_1(sst_ersst)
    data_obs = xr.merge([data_obs, prep_1(nhf_era5).interp({'longitude': data_obs.longitude}), prep_1(pr_ref), idx_obs_1])
    for _v in list(nhf_era5) + list(sst_ersst) + ['pr']:
        data_obs[f'{_v}_anom'] = data_obs[_v] - data_obs[_v].mean('time')
    return (data_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### CESM2
    """)
    return


@app.cell
def _(resample_ndj, src):
    def resample_ndj_cesm(x):
        """resample CESM eofs to djf"""
        comps, scores = src.utils.split_components(x)
        scores_ndj = resample_ndj(scores)
        x_ndj = src.utils.combine_components(comps, scores_ndj)

        return x_ndj

    return (resample_ndj_cesm,)


@app.cell
def _(anom_1, forced_1, idx_mod_1, resample_ndj_cesm, trim, xr):
    anom_ndj = resample_ndj_cesm(trim(anom_1))
    data_mod_forced = resample_ndj_cesm(trim(forced_1))
    ## create new dataarray
    data_mod = xr.merge([anom_ndj, idx_mod_1])
    return data_mod, data_mod_forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ERA5
    """)
    return


@app.cell
def _(np):
    def get_composite(data, idx_name, is_warm=True, n=7):
        """helper func to make composite"""

        ## get sorted indiex
        idx_sorted = data[idx_name].argsort().values
        if is_warm:
            idx = idx_sorted[-n:]
        else:
            idx = idx_sorted[:n]

        ## select events
        spag = data.isel(time=idx)

        ## rename coordinate
        spag = spag.rename({"time": "sample"}).assign_coords({"sample": np.arange(n)})

        ## rename coordinate
        return spag

    return (get_composite,)


@app.cell
def _(data_obs, get_composite):
    obs_nino_spag = get_composite(data_obs, idx_name="sst_3")
    obs_nina_spag = get_composite(data_obs, idx_name="sst_3", is_warm=False)
    return obs_nina_spag, obs_nino_spag


@app.cell
def _(ax_kwargs_1, data_obs, obs_nina_spag, obs_nino_spag, plt, save, src):
    _fig, _axs = plt.subplots(2, 5, figsize=(12, 4.5), layout='constrained')
    for _ax, _varname in zip(_axs[0], ['sst_total', 'pr', 'nhf', 'sw', 'lhf']):
        _ax.plot(data_obs.longitude, data_obs[_varname].mean('time'), c='k')
        for _spags, _color in zip([obs_nino_spag, obs_nina_spag], ['r', 'b']):
            _ax.plot(_spags.longitude, _spags[_varname].mean('sample'), c=_color)
    for _ax, _varname in zip(_axs[1], ['sst', 'pr_anom', 'nhf_anom', 'sw_anom', 'lhf_anom']):
        for _spags, _color in zip([obs_nino_spag, -obs_nina_spag], ['r', 'b']):
            _ax.plot(_spags.longitude, _spags[_varname].mean('sample'), c=_color)
    _axs[0, 0].set_title('SST')
    _axs[0, 1].set_title('Precip. (mm day$^{-1}$)')
    _axs[0, -3].set_title('NHF (W m$^{-2})$')
    _axs[0, -2].set_title('Shortwave')
    _axs[0, -1].set_title('Latent heat flux')
    for _ax in _axs[1]:
        _ax.axhline(0, **ax_kwargs_1)
    for _ax, _s in zip(_axs[:, -1], ['Total', 'Anom.']):
        _ax.text(s=_s, x=1.1, y=0.5, transform=_ax.transAxes)
    src.utils.add_vticks(_axs.flatten(), xticks=[130, 210, 270], xlines=[210, 270])
    src.utils.set_lims(_axs[1, -3:])
    _axs[0, -3].set_ylim([0, 130])
    _axs[0, -2].set_ylim([175, 275])
    _axs[0, -1].set_ylim([-150, -50])
    for _ax in _axs[-1]:
        _ax.set_xlabel('Longitude')
    for _ax in _axs[0]:
        _ax.set_xticks([])
    for _ax in _axs[1, -3:]:
        _ax.set_ylim(_ax.get_ylim()[::-1])
    save(_fig, fname='Q-structure_era5', do_save=False)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### CESM2
    """)
    return


@app.function
def get_composite_cesm(data, idx_name, is_warm=True, n=301):
    """helper func to make composite"""

    ## create sample dim
    data_ = data.stack(sample=["time", "member"])

    ## get sorted inde
    idx_sorted = data_[idx_name].argsort().values
    if is_warm:
        idx = idx_sorted[-n:]
    else:
        idx = idx_sorted[:n]

    ## select events
    spag = data_.isel(sample=idx)

    # ## rename coordinate
    # spag = spag.rename({"time":"sample"}).assign_coords({"sample":np.arange(n)})

    ## rename coordinate
    return spag


@app.cell
def _(data_mod, data_mod_forced, src, xr):
    ## get climatology
    mod_clim = src.utils.reconstruct_wrapper(
        data_mod_forced.mean(["time"]),
        fn=lambda x: x,
    )

    ## add clim. for spatial data
    mod_clim = xr.merge([mod_clim, data_mod_forced[["T_50", "T_80", "thc"]].mean("time")])

    # get_composite_cesm(data_mod, idx_name="ssta_34")
    data_mod_components, data_mod_scores = src.utils.split_components(data_mod)
    return data_mod_components, data_mod_scores, mod_clim


@app.cell
def _(data_mod_components, data_mod_scores, src, xr):
    mod_proj_nino_spag = get_composite_cesm(data_mod_scores, "sst_34")
    mod_proj_nina_spag = get_composite_cesm(data_mod_scores, "sst_34", is_warm=False)

    mod_nino_spag = src.utils.reconstruct_fn(
        scores=mod_proj_nino_spag,
        components=data_mod_components,
        fn=lambda x: x,
    )

    mod_nina_spag = src.utils.reconstruct_fn(
        scores=mod_proj_nina_spag,
        components=data_mod_components,
        fn=lambda x: x,
    )

    # Add back in some spatial information
    mod_nino_spag = xr.merge(
        [
            mod_proj_nino_spag[["T_50", "T_80", "thc"]],
            mod_nino_spag,
        ]
    )

    mod_nina_spag = xr.merge(
        [
            mod_proj_nina_spag[["T_50", "T_80", "thc"]],
            mod_nina_spag,
        ]
    )
    return mod_nina_spag, mod_nino_spag


@app.cell
def _(ax_kwargs_1, mod_clim, mod_nina_spag, mod_nino_spag, plt, save, src):
    _fig, _axs = plt.subplots(2, 5, figsize=(12, 4.5), layout='constrained')
    for _ax, _varname in zip(_axs[0], ['sst', 'pr', 'nhf', 'sw', 'lhf']):
        _ax.plot(mod_clim.longitude, mod_clim[_varname], c='k')
        for _spags, _color in zip([mod_nino_spag, mod_nina_spag], ['r', 'b']):
            _ax.plot(_spags.longitude, mod_clim[_varname] + _spags[_varname].mean('sample'), c=_color)
    for _ax, _varname in zip(_axs[1], ['sst', 'pr', 'nhf', 'sw', 'lhf']):
        for _spags, _color in zip([mod_nino_spag, -mod_nina_spag], ['r', 'b']):
            _ax.plot(_spags.longitude, _spags[_varname].mean('sample'), c=_color)
    _axs[0, 0].set_title('SST')
    _axs[0, 1].set_title('Precip. (mm day$^{-1}$)')
    _axs[0, -3].set_title('NHF (W m$^{-2})$')
    _axs[0, 3].set_title('Shortwave')
    _axs[0, 4].set_title('Latent heat flux')
    for _ax in _axs[1]:
        _ax.axhline(0, **ax_kwargs_1)
    for _ax, _s in zip(_axs[:, -1], ['Total', 'Anom.']):
        _ax.text(s=_s, x=1.1, y=0.5, transform=_ax.transAxes)
    src.utils.add_vticks(_axs.flatten(), xticks=[130, 210, 270], xlines=[210, 270])
    for _ax in _axs[-1]:
        _ax.set_xlabel('Longitude')
    src.utils.set_lims(_axs[1, 2:])
    for _ax in _axs[0]:
        _ax.set_xticks([])
    for _ax in _axs[1, 2:]:
        _ax.set_ylim(_ax.get_ylim()[::-1])
    save(_fig, fname='Q-structure_cesm2', do_save=False)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add precip to this!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## to-do
    - same plots, but for CESM2
    - relative SST comparison between ERSST and CESM2 (spatial plot?)
    - rainfall comparison?
    """)
    return


if __name__ == "__main__":
    app.run()
