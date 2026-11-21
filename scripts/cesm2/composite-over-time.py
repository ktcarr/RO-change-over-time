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
    # ENSO composite over time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## imports
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import seaborn as sns
    import xarray as xr
    import tqdm
    import pathlib
    import cmocean
    import os
    import copy
    import time
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs

    # Import custom modules
    import src.utils

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return DATA_FP, ccrs, copy, mpatches, np, os, pathlib, plt, src, tqdm, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funcs
    """)
    return


@app.cell
def _(DATA_FP, copy, np, pathlib, src, xr):
    def get_dT_sub(Tsub, Hm, delta=25):
        """
        Get temperature difference b/n entrainment zone and mixed layer.
        (positive if entrainment zone is warmer than ML)
        """
        in_ml = Tsub.z_t <= Hm
        in_ez = (Tsub.z_t > Hm) & (Tsub.z_t < delta + Hm)  ## find indices in ML and entrainment zone (ez)
        Tbar = Tsub.where(in_ml).mean('z_t')
        Tplus = Tsub.where(in_ez).mean('z_t')
        dT = Tplus - Tbar
        return dT  ## get Tbar and Tplus (following Frankignoul et al paper)

    def _prep(data):
        """remove sst dependence and compute tendencies"""
        for h_idx in ['h_w', 'h', 'h_w_z20']:  ## get gradient
            data[f'{h_idx}_hat'] = src.utils.remove_sst_dependence_v2(data, h_var=h_idx, T_var='T_34')
        return data

    def load_budget_data(t_early, t_late, target_grid):
        """load ground truth heat budget data"""

        def load_var(varname):
            """load variable from prepped folder"""
            data = xr.open_mfdataset(sorted(list(pathlib.Path(DATA_FP, 'cesm', f'{varname}_temp').glob('*.nc'))), concat_dim='member', combine='nested', parallel=True)  ## remove from h indices
            return data.assign_coords({'member': np.arange(100)})
        budget_data = xr.merge([load_var(v) for v in ['adv', 'ddt_T']])
        M_PER_CM = 0.01
        SEC_PER_MO = 86400.0 * 30
        budget_data = budget_data.assign_coords({'z_t': budget_data.z_t * M_PER_CM})
        budget_data = budget_data * SEC_PER_MO
        budget_data = budget_data.assign_coords({'nlon': budget_data.lon.values})
        budget_data = budget_data.drop_vars('lon').rename({'nlon': 'longitude'})
        budget_data = xr.concat([budget_data.sel(t_early), budget_data.sel(t_late)], dim='time')
        budget_data = budget_data.interp_like(target_grid)
        forced_bud, anom_bud = src.utils.separate_forced(budget_data)
        return (forced_bud, anom_bud)

    def plot_mlds(axs, bar_early, bar_late, month=None):
        """plot mixed layer depth on pair of axs objects"""  ## open data
        lon = src.utils.merimean(bar_early).longitude
        if month is None:
            sel = lambda x: x.mean('month')
        else:
            sel = lambda x: x.sel(month=month)
        axs[0].plot(lon, sel(src.utils.merimean(bar_early['mld'])), c='k')
        axs[1].plot(lon, sel(src.utils.merimean(bar_late['mld'])), c='k', ls='--')
        axs[2].plot(lon, sel(src.utils.merimean(bar_early['mld'])), c='k')
        axs[2].plot(lon, sel(src.utils.merimean(bar_late['mld'])), c='k', ls='--')
        return  ## load data

    def get_nino3_da(x):
        """get nino3 on dataarray"""  ## get difference
        if 'latitude' not in x.dims:  # budget_data["diff"] = budget_data["TEND_TEMP"] - budget_data["ADV_3D_TEMP"]
            x = x.expand_dims('latitude')
        return src.utils.get_nino3(x)  ## convert z coord from cm to m, and unit from K/s to K/mo

    def get_nino34_da(x):
        """get nino3.4 on dataarray"""
        if 'latitude' not in x.dims:  ## convert from (i) cm to m and (ii) K/s to K/mo
            x = x.expand_dims('latitude')
        return src.utils.get_nino34(x)

    def get_composite(idx, data, peak_month, q=0.95, is_warm=True, event_type=None):  ## fix longitude coordinate
        """
        Get hovmoller composite based on specified:
        - data: used to compute index/make composite
        - peak_month: month to center composite on  ## trim in time
        - q: quantile threshold for composite
        """
        if is_warm:
            kwargs = dict(q=q, check_cutoff=lambda x, cut: x > cut)
        else:  ## interpolate
            kwargs = dict(q=1 - q, check_cutoff=lambda x, cut: x < cut)
        kwargs = dict(kwargs, peak_month=peak_month, idx=idx, data=data, event_type=event_type)
        comp_proj = src.utils.make_composite(**kwargs)  # ## separate forced/anomalies
        return comp_proj

    def get_spatial_composite(data, **composite_kwargs):
        """
        Get spatial composite
        """
        components, scores = src.utils.split_components(data)
        comp_proj = get_composite(data=scores, **composite_kwargs)
        comp = reconstruct_helper(comp_proj, components, func=lambda x: x).drop_vars('mode')  ## get longitude
        return comp

    def reconstruct_helper(composite, components, func):  ## helper function to get month
        """reconstruction helper function for composite"""
        composite_recon = copy.deepcopy(composite)
        for c in list(components):
            composite_recon[c] = src.utils.reconstruct_fn(components=components[c], scores=composite[c], fn=func)
        for c in list(composite):
            if '_total' in c:  ## plot
                n = c[:-6]
                if n in list(components):
                    composite_recon[c] = src.utils.reconstruct_fn(components=components[n], scores=composite[c], fn=func)
        return composite_recon

    def get_spatial_clim(forced, lags, peak_month, fn=lambda x: x):
        """get climatologies of spatial variables"""
        clim = src.utils.reconstruct_clim(data=forced, fn=fn)
        months = 1 + np.mod(lags + peak_month - 1, 12)
        clim_comp = xr.concat([clim.sel(month=m).drop_vars('month') for m in months], dim=lags)
        return clim_comp

    def get_T_ml_tendency(T, mld, delta=5, H0=None):
        """compute mixed-layer temperature tendency"""
        T_ml = src.utils.get_ml_avg(T, Hm=mld, delta=delta, H0=H0)
        return T_ml.differentiate('lag')

    def get_spatial_composite_wrapper(data, forced_scores, peak_month, H0=None, **composite_kwargs):
        """
        Get spatial composite
        """
        composite = get_spatial_composite(data=data, peak_month=peak_month, **composite_kwargs)
        comp_clim = get_spatial_clim(forced=forced_scores, lags=composite.lag, peak_month=peak_month)
        H_clim = src.utils.get_H_int(comp_clim['T'], thresh=0.04)
        H_full = src.utils.get_H_int(comp_clim['T'] + composite['T'], thresh=0.04)
        comp_clim['H'] = H_clim
        composite['H'] = H_full - H_clim
        feedbacks = src.utils.get_feedbacks(bar=comp_clim, prime=composite, use_bulk=True)
        composite = xr.merge([composite, feedbacks])
        sec_per_mo = 86400.0 * 30
        rho = 1020.0
        Cp = 4200.0
        if H0 is None:
            H = comp_clim['mld']
        else:  ## handle warm/cold case
            H = H0
        composite['Q'] = composite['nhf'] * sec_per_mo / (rho * Cp * H)
        return (composite, comp_clim)  ## kwargs for composite  ## composite of projected data  ## pull out components  ## get projected composite  ## reconstruct spatial fields  # ## reconstruct relative SST  # comp["sst_rel"] = comp["sst_total"] - comp["trop_sst_05"]  ## copy to hold reconstructed results  ## reconstruct anomalies  ## check for "total" fields  ## reconstruct monthly climatology for period  ## convert to lag coordinates  ## integrate over mixed layer  ## compute tendency  ## get spatial composite of anomalies  ## get background state  ## get thermocline depth  ## compute ocean feedbacks  ## get differences  # composite["diff"] = composite["TEND_TEMP"] - composite["ADV_3D_TEMP"]  ## add SST tendency  # composite["ddt_sst"] = composite["sst"].differentiate("lag")  # composite["ddt_ssh"] = composite["ssh"].differentiate("lag")  # composite["ddt_z20"] = composite["z20"].differentiate("lag")  # composite["ddt_ssh_hat"] = composite["ssh_hat"].differentiate("lag")  ## get NHF in units of K/mo  ## get H

    return get_nino3_da, get_spatial_composite_wrapper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### $T$, $h$
    """)
    return


@app.cell
def _(DATA_FP, pathlib, src, xr):
    ## open data
    Th = src.utils.load_cesm_indices(load_z20=True)
    Th = Th.rename({'north_tropical_atlantic': 'natl', 'atlantic_nino': 'nino_atl', 'tropical_indian_ocean': 'iobm', 'indian_ocean_dipole': 'iod', 'north_pacific_meridional_mode': 'npmm', 'south_pacific_meridional_mode': 'spmm'})
    ## rename indices for convenience
    trop_sst = xr.open_dataset(pathlib.Path(DATA_FP, 'cesm/trop_sst.nc'))
    Th_total = xr.open_dataset(DATA_FP / 'cesm' / 'Th.nc')
    for _n in ['T_3', 'T_34', 'T_4']:
        Th[f'{_n}_rel'] = Th_total[_n] - trop_sst['trop_sst_10']
    eli = xr.open_dataset(pathlib.Path(DATA_FP, 'cesm/eli.nc'))
    ## load tropical SST avg
    ## Load T,h (total)
    ## compute relative sst
    ## Load ELI
    eli_forced, eli_anom = src.utils.separate_forced(eli)
    return Th, eli_anom, eli_forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute composite
    """)
    return


@app.cell
def _(
    Th,
    eli_anom,
    eli_forced,
    get_spatial_composite_wrapper,
    os,
    pathlib,
    src,
    tqdm,
    xr,
):
    ## should we use wide data?
    USE_WIDE = True
    _VARNAME = 'T_34'
    ## specify what variable to use
    kwargs = dict(peak_month=12, q=0.95, event_type=1, H0=50)
    SAVE_DIR = pathlib.Path(os.environ['SAVE_FP'], 'composites', f"{_VARNAME}_{kwargs['q']}_bulk_Hm50_with_clearksky")
    ## specify shared args
    if (SAVE_DIR / 'cold.nc').is_file():
        try:
            comps_cold = xr.open_dataset(SAVE_DIR / 'cold.nc')
            comps_warm = xr.open_dataset(SAVE_DIR / 'warm.nc')
            clim = xr.open_dataset(SAVE_DIR / 'clim.nc')
        except:
            print('error...')
    ## save filepath
    else:
        forced_wide, anom_wide = src.utils.load_consolidated_wide()  # os.environ["SAVE_FP"], "composites", f"{VARNAME}_{kwargs['q']}"
        forced, anom = src.utils.load_consolidated()
        if USE_WIDE:
            for _v in ['T', 'u', 'w']:  # f"{VARNAME}_{kwargs['q']}_Hm50_narrow",
                for suf in ['', '_comp']:  # f"{VARNAME}_{kwargs['q']}_bulk_Hm80",
                    forced[f'{_v}{suf}'] = forced_wide[f'{_v}{suf}']
                    anom[f'{_v}{suf}'] = anom_wide[f'{_v}{suf}']  # os.environ["SAVE_FP"], "composites", f"{VARNAME}_{kwargs['q']}_with-eli"
        anom = xr.merge([anom, Th / Th.std(), eli_anom])
        forced = xr.merge([forced, eli_forced])
        flux_forced, flux_anom = src.utils.load_flux_data()
        anom = xr.merge([anom, flux_anom])
        forced = xr.merge([forced, flux_forced])
        anom = src.utils.get_windowed(anom, stride=120).compute()
        forced = src.utils.get_windowed(forced, stride=120).compute()
        comps_cold = []
        comps_warm = []
        clims = []
        for _y in tqdm.tqdm(anom.year):
            comp_kwargs = dict(idx=anom[_VARNAME].sel(year=_y), forced_scores=forced.sel(year=_y), data=anom.sel(year=_y))
            comp_warm, clim = get_spatial_composite_wrapper(is_warm=True, **comp_kwargs, **kwargs)
            comp_cold, _ = get_spatial_composite_wrapper(is_warm=False, **comp_kwargs, **kwargs)  ## load data
            clims.append(clim)
            comps_warm.append(comp_warm)
            comps_cold.append(comp_cold)
        comps_warm = xr.concat(comps_warm, dim=anom.year)  ## replace with wide data
        comps_cold = xr.concat(comps_cold, dim=anom.year)
        clims = xr.concat(clims, dim=anom.year)
        comps_cold.to_netcdf(SAVE_DIR / 'cold.nc')
        comps_warm.to_netcdf(SAVE_DIR / 'warm.nc')
        clims.to_netcdf(SAVE_DIR / 'clim.nc')  ## add flux data  ## "window" it  ## empty array to hold coefficients  ## get args for composite  ## warm  ## cold  ## append to list  ## convert to xarray  ## save to fild
    return clim, comps_cold, comps_warm, kwargs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Mixed layer averages
    """)
    return


@app.cell
def _(comps_cold, comps_warm, src):
    ml_kwargs = dict(H0=50, Hm=None)
    comps_cold_1 = src.utils.get_ml_avg_ds(comps_cold, **ml_kwargs)
    comps_warm_1 = src.utils.get_ml_avg_ds(comps_warm, **ml_kwargs)
    for _comp in [comps_cold_1, comps_warm_1]:
        for _v in ['T_ml', 'sst', 'T']:
            _comp[f'ddt_{_v}'] = _comp[_v].differentiate('lag')
    return comps_cold_1, comps_warm_1, ml_kwargs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Convert surface flux to temperature units
    """)
    return


@app.cell
def _(comps_cold_1, comps_warm_1, ml_kwargs, src):
    for _comps in [comps_warm_1, comps_cold_1]:
        _comps['ndh_z'] = src.utils.get_THF_bulk(prime=_comps, bar=_comps)
        _comps['ndh_ml'] = src.utils.get_ZAF(prime=_comps, bar=_comps, u_var='u', T_var='T').sel(z_t=slice(0, ml_kwargs['H0'])).mean('z_t')
        _comps['zonal'] = _comps['ndh_ml'] + _comps['ZAF_ml'] + _comps['DD_ml']
        _comps['ddt_sst'] = _comps['sst'].differentiate('lag')
        _comps['ohc'] = _comps['T'].integrate('z_t') / _comps.z_t.max().item()
        _comps['ddt_ohc'] = _comps['ohc'].differentiate('lag')
        sec_per_mo = 86400.0 * 30
        rho = 1020.0
        Cp = 4200.0
        H = ml_kwargs['H0']
        _comps['Q_sw'] = _comps['fsns'] * sec_per_mo / (rho * Cp * H)
        _comps['Q_lw'] = -_comps['flns'] * sec_per_mo / (rho * Cp * H)
        _comps['Q_lhf'] = -_comps['lhflx'] * sec_per_mo / (rho * Cp * H)
        _comps['Q_swc'] = _comps['fsnsc'] * sec_per_mo / (rho * Cp * H)
        _comps['Q_cloud'] = _comps['Q_sw'] - _comps['Q_swc']
        _comps['Q_shf'] = _comps['Q'] - _comps['Q_sw'] - _comps['Q_lw'] - _comps['Q_lhf']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Get surface flux clim
    """)
    return


@app.cell
def _():
    # forced, anom = src.utils.load_consolidated()

    # ## add flux data
    # flux_forced, flux_anom = src.utils.load_flux_data()
    # anom = xr.merge([anom, flux_anom])
    # forced = xr.merge([forced, flux_forced])

    # ## "window" it
    # anom = src.utils.get_windowed(anom, stride=120).compute()
    # forced = src.utils.get_windowed(forced, stride=120).compute()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### get total ELI
    """)
    return


@app.cell
def _(comps_cold_1, comps_warm_1, eli_forced, kwargs, np, src, xr):
    eli_clim = src.utils.get_windowed(eli_forced).groupby('time.month').mean()
    months = 1 + np.mod(comps_warm_1.lag + kwargs['peak_month'] - 1, 12)
    eli_clim = xr.concat([eli_clim.sel(month=m).drop_vars('month') for m in months], dim=comps_warm_1.lag)
    for _comps in [comps_warm_1, comps_cold_1]:
        _comps['eli_total'] = _comps['eli_05'] + eli_clim['eli_05']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Get relative SST
    """)
    return


@app.cell
def _(clim, comps_cold_1, comps_warm_1, src):
    for _comp in [comps_cold_1, comps_warm_1]:
        _comp['trop_sst'] = src.utils.get_nino34(clim['sst']) + _comp['T_34'] - _comp['T_34_rel']
        _comp['sst_rel'] = clim['sst'] + _comp['sst'] - _comp['trop_sst']
        _comp['T_3_ml'] = _comp['T_ml'].sel(longitude=slice(210, 270)).mean('longitude')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot
    """)
    return


@app.cell
def _(os, pathlib, src):
    def plot_cycle_hov_lagged(ax, data, amp, is_filled=True, xticks=[210, 270], lat_bound=5, nlev=5, cmap='cmo.balance', levels=None):
        """plot data on ax object"""
        if levels is None:
            levels = src.utils.make_cb_range(amp, amp / nlev)
        shared_kwargs = dict(levels=levels, extend='both')
        if is_filled:
            plot_fn = ax.contourf
            kwargs = dict(cmap=cmap)
        else:
            plot_fn = ax.contour
            kwargs = dict(colors='k', linewidths=0.8)
        if 'latitude' in data.coords:
            plot_data = src.utils.merimean(data, lat_bound=lat_bound)
        else:  ## set levels if not specified
            plot_data = data
        cp = plot_fn(plot_data.longitude, plot_data.lag, plot_data.transpose('lag', 'longitude'), **kwargs, **shared_kwargs)
        kwargs = dict(c='w', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlim([145, 280])  ## specify shared kwargs
        ax.set_xlabel('Lon')
        ax.set_xticks(xticks)
        for tick in xticks:  ## specify kwargs
            ax.axvline(tick, **kwargs)
        ax.axhline(0, **kwargs)
        return cp

    def format_hov_axs(axs, peak_mon):
        """format hov axs"""
        font_kwargs = dict(size=10)
        axs[0].set_title('Early', **font_kwargs)
        axs[1].set_title('Late', **font_kwargs)  ## average over latitudes (if necessary)
        axs[2].set_title('Difference (scaled)', **font_kwargs)
        for ax in axs:
            ax.set_yticks([])
        src.utils.label_hov_yaxis(axs[0], peak_mon=peak_mon)
        return axs
      ## do the plotting
    def plot_eli_on_axs(axs, x0, x1, name='eli_05'):
        """plot ELI boundary on hovmoller axs"""
        axs[0].plot(x0[name], x0.lag, c='magenta')
        axs[1].plot(x1[name], x1.lag, c='magenta', ls='--')
        axs[2].plot(x0[name], x0.lag, c='magenta', ls='-')
        axs[2].plot(x1[name], x1.lag, c='magenta', ls='--')
        return

    def _plot_level(ax, comp, level, ls='-', c='magenta'):  ## format ax object
        """plot single level on hovmoller"""
        ax.contour(comp.longitude, comp.lag, comp.transpose('lag', ...), levels=[level], colors=c, linestyles=ls)
        return

    def plot_sst_rel_on_axs(axs, x0, x1, lev=0):
        """plot ELI boundary on hovmoller axs"""
        _plot_level(axs[0], x0['sst_rel'], level=lev)
        _plot_level(axs[1], x1['sst_rel'], level=lev, ls='--')
        _plot_level(axs[2], x0['sst_rel'], level=lev)
        _plot_level(axs[2], x1['sst_rel'], level=lev, ls='--')
        return

    def plot_comp_on_axs(axs, x0, x1, name, amp, peak_month, amp_diff=None, nlev=5, return_cfs=False, normalize=True, **kwargs):
        """plot composite on axs objects"""
        if amp_diff is None:  ## label
            amp_diff = amp / 2
        cfs = []
        if normalize:
            for ax, x in zip(axs, [x0, x1, amp / amp_diff * (x1 - x0)]):
                cfs.append(plot_cycle_hov_lagged(ax=ax, data=x[name], amp=amp, nlev=nlev, **kwargs))
        else:
            for ax, x, amp_ in zip(axs, [x0, x1, x1 - x0], [amp, amp, amp_diff]):
                cfs.append(plot_cycle_hov_lagged(ax=ax, data=x[name], amp=amp_, nlev=nlev, **kwargs))
        format_hov_axs(axs, peak_mon=peak_month)
        if return_cfs:
            cf_ = cfs
        else:
            cf_ = cfs[-1]
        return cf_

    def save(fig, fname, dpi=800, do_save=False):  ## plot ELI
        """save figure to file"""
        save_dir = pathlib.Path(os.environ['SAVE_FP'], 'egu_poster_figs')
        fname = save_dir / f'{fname}.pdf'
        if do_save:
            fig.savefig(fname, dpi=dpi, format='pdf')
        return  ## plot ELI  ## handle amp_diff=None  ## empty array to hold results  ## format/label  ## get save directory  ## get fname

    return plot_comp_on_axs, save


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### show plots
    """)
    return


@app.cell
def _(kwargs):
    ## shared args
    hov_kwargs = dict(
        peak_month=kwargs["peak_month"],
    )
    merimean = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")

    norm = lambda x: x / x["T_34"]

    ## specify years for plotting
    Y0 = 1870
    Y1 = 2000
    return Y0, Y1, hov_kwargs, merimean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SST
    """)
    return


@app.cell
def _(Y0, Y1, comps_cold_1, hov_kwargs, mpatches, np, plot_comp_on_axs, plt):
    _COMPS = comps_cold_1 / comps_cold_1['T_3']
    _fig, _axs = plt.subplots(4, 3, figsize=(8, 8), layout='constrained')
    for _j, (_name, _amp, _l) in enumerate(zip(['sst', 'ddt_sst', 'Q', 'pr'], [4, 1, 1, 12 / 86000.0], ['SST (K)', '$\\frac{\\partial}{\\partial t}$SST (K/mo)', '$Q$ (K/mo)', 'pr (mm/day)'])):
        if _name == 'pr':
            _cmap = 'cmo.balance_r'
            _amp_ = np.round(86000.0 * _amp)
        else:
            _cmap = 'cmo.balance'
            _amp_ = _amp
        _cfs = plot_comp_on_axs(_axs[_j], name=_name, amp=_amp, amp_diff=_amp / 3, x0=_COMPS.sel(year=Y0), x1=_COMPS.sel(year=Y1), nlev=10, cmap=_cmap, return_cfs=True, normalize=False, **hov_kwargs)
        _ELI = _COMPS['eli_total']
        _LAG = _COMPS.lag
        _axs[_j, 0].plot(_ELI.sel(year=Y0), _LAG, c='k', ls='-')
        _axs[_j, 1].plot(_ELI.sel(year=Y1), _LAG, c='k', ls='--')
        for _y_, _ls in zip([Y0, Y1], ['-', '--']):
            _axs[_j, -1].plot(_ELI.sel(year=_y_), _LAG, c='k', ls=_ls)
        _fig.colorbar(_cfs[1], ax=_axs[_j, 1], ticks=[-_amp_, 0, _amp_])
        _fig.colorbar(_cfs[-1], ax=_axs[_j, 2], ticks=[-_amp_ / 2, 0, _amp_ / 2], label=_l)
        _axs[_j, -1].add_patch(mpatches.Rectangle(xy=(190, -1), width=50, height=2, edgecolor='magenta', facecolor='none'))
    _axs[0, 0].set_title(f'{Y0}', fontsize=12)
    _axs[0, 1].set_title(f'{Y1}', fontsize=12)
    _axs[0, -1].set_title('Difference', fontsize=12)
    for _ax in _axs[1:].flatten():
        _ax.set_title(None)
    for _ax in _axs[:-1].flatten():
        _ax.set_xticks([])
        _ax.set_xlabel(None)
    for _ax in _axs.flatten():
        _ax.set_ylim([-6.5, 6.5])
    for _ax in _axs[:, 0]:
        _ax.set_yticks([-6, 0, 6], labels=['Jun(-1)', 'Dec(0)', 'Jun(+1)'])
    _bbox = dict(boxstyle='round', facecolor='white', alpha=0)
    posnx, posny = (0.02, 0.05)
    _labels = np.array([['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l']])
    for _ax, label in zip(_axs.flatten(), _labels.flatten()):
        _ax.text(posnx, posny, s=f'({label})', bbox=_bbox, transform=_ax.transAxes, c='k')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Smaller plot for poster
    """)
    return


@app.cell
def _():
    def _plot_level(ax, comp, level, ls='-', c='k'):
        """plot single level on hovmoller"""
        ax.contour(comp.longitude, comp.lag, comp.transpose('lag', ...), levels=[level], colors=c, linestyles=ls)
        return

    return


@app.cell
def _(comps_warm_1, merimean, plt, save, src):
    _fig, _axs = plt.subplots(1, 2, figsize=(5.5, 2.5), layout='constrained')
    for _ax, yr, _ls in zip(_axs, [2010, 2080], ['-', '--']):
        _cp = _ax.contourf(comps_warm_1.longitude, comps_warm_1.lag, merimean(comps_warm_1['sst']).sel(year=yr).transpose('lag', ...), cmap='cmo.balance', levels=src.utils.make_cb_range(3.5, 0.7), extend='both')
    _ELI = comps_warm_1['eli_total']
    eli_kwargs = dict(c='k', lw=1)
    for _ax in _axs:
        _ax.plot(_ELI.sel(year=2010), comps_warm_1.lag, ls='-', **eli_kwargs, label='2010 ELI')
    _axs[1].plot(_ELI.sel(year=2080), comps_warm_1.lag, ls='--', **eli_kwargs, label='2080 ELI')
    _axs[0].plot(_ELI.sel(year=2080), comps_warm_1.lag, ls='--', **eli_kwargs, label='2080 ELI', zorder=0)
    for _ax in _axs:
        _ax.set_ylim([-6.5, 6.5])
        _ax.set_xlim([140, 285])
        ax_kwargs = dict(ls='--', c='w', lw=0.75, alpha=0.75)
        _ax.axhline(0, **ax_kwargs)
        for _t in [190, 240]:
            _ax.axvline(_t, **ax_kwargs)
        _ax.set_xticks([185, 245], labels=['170$^{\\circ}$W', '120$^{\\circ}$W'])
    _axs[0].set_yticks([-6, 0, 6], labels=['Jun(-1)', 'Dec(0)', 'Jun(+1)'])
    _axs[1].set_yticks([])
    _axs[0].set_title('(a) 2010', loc='left')
    _axs[1].set_title('(b) 2080', loc='left')
    _bbox = dict(edgecolor='none', facecolor='none')
    legend_kwargs = dict(x=0.43, y=0.05, fontdict=dict(size=10, color='w'), bbox=_bbox)
    _axs[0].text(s=f'Interval: 0.7 K', transform=_axs[0].transAxes, **legend_kwargs)
    leg = _axs[0].legend(prop=dict(size=10), loc=(-0.35, 0.62))
    frame = leg.get_frame()
    frame.set_color('white')
    save(_fig, fname='spatial_comp', do_save=False)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Heat fluxes
    """)
    return


@app.cell
def _(comps_cold_1, hov_kwargs, mpatches, np, plot_comp_on_axs, plt):
    Y0_1 = 2050
    Y1_1 = 2080
    _COMPS = comps_cold_1
    _AMP_ = 1.0
    _fig, _axs = plt.subplots(5, 3, figsize=(8, 10), layout='constrained')
    for _j, (_name, _amp, _l) in enumerate(zip(['pr', 'Q_sw', 'Q_lhf', 'Q_lw', 'Q_shf'], [12 / 86000.0, _AMP_, _AMP_, _AMP_, _AMP_], ['pr (mm/day)', 'shortwave (K/mo)', 'latent (K/mo)', 'longwave (K/mo)', 'sensible (K/mo)'])):
        if _name == 'pr':
            _cmap = 'cmo.balance_r'
            _amp_ = np.round(86000.0 * _amp)
        else:
            _cmap = 'cmo.balance'
            _amp_ = _amp
        _cfs = plot_comp_on_axs(_axs[_j], name=_name, amp=_amp, amp_diff=_amp / 2, x0=_COMPS.sel(year=Y0_1), x1=_COMPS.sel(year=Y1_1), nlev=10, cmap=_cmap, return_cfs=True, normalize=False, **hov_kwargs)
        _ELI = _COMPS['eli_total']
        _LAG = _COMPS.lag
        _axs[_j, 0].plot(_ELI.sel(year=Y0_1), _LAG, c='k', ls='-')
        _axs[_j, 1].plot(_ELI.sel(year=Y1_1), _LAG, c='k', ls='--')
        for _y_, _ls in zip([Y0_1, Y1_1], ['-', '--']):
            _axs[_j, -1].plot(_ELI.sel(year=_y_), _LAG, c='k', ls=_ls)
        _fig.colorbar(_cfs[1], ax=_axs[_j, 1], ticks=[-_amp_, 0, _amp_])
        _fig.colorbar(_cfs[-1], ax=_axs[_j, 2], ticks=[-_amp_ / 2, 0, _amp_ / 2], label=_l)
        _axs[_j, -1].add_patch(mpatches.Rectangle(xy=(190, -1), width=50, height=2, edgecolor='magenta', facecolor='none'))
    _axs[0, 0].set_title(f'{Y0_1}', fontsize=12)
    _axs[0, 1].set_title(f'{Y1_1}', fontsize=12)
    _axs[0, -1].set_title('Difference', fontsize=12)
    for _ax in _axs[1:].flatten():
        _ax.set_title(None)
    for _ax in _axs[:-1].flatten():
        _ax.set_xticks([])
        _ax.set_xlabel(None)
    for _ax in _axs.flatten():
        _ax.set_ylim([-5, 5])
    for _ax in _axs[:, 0]:
        _ax.set_yticks([-6, 0, 6], labels=['Jun(-1)', 'Dec(0)', 'Jun(+1)'])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### dynamical terms
    """)
    return


@app.cell
def _(comps_cold_1, hov_kwargs, mpatches, np, plot_comp_on_axs, plt):
    Y0_2 = 2050
    Y1_2 = 2080
    _COMPS = comps_cold_1 / comps_cold_1['T_3']
    _AMP_ = 0.5
    _fig, _axs = plt.subplots(4, 3, figsize=(8, 8), layout='constrained')
    for _j, (_name, _amp, _l) in enumerate(zip(['ddt_T_ml', 'THF', 'EKM', 'zonal'], [_AMP_, _AMP_, _AMP_, _AMP_], ['$\\frac{\\partial}{\\partial t}\\left<T\\right>$', "$\\left<\\overline{w}~\\frac{\\partial T'}{\\partial z}\\right>$", "$\\left<w'~\\frac{\\partial \\overline{T}}{\\partial z}\\right>$", 'zonal advection'])):
        if _name == 'pr':
            _cmap = 'cmo.balance_r'
            _amp_ = np.round(86000.0 * _amp)
        else:
            _cmap = 'cmo.balance'
            _amp_ = _amp
        _cfs = plot_comp_on_axs(_axs[_j], name=_name, amp=_amp, amp_diff=_amp / 2, x0=_COMPS.sel(year=Y0_2), x1=_COMPS.sel(year=Y1_2), nlev=10, cmap=_cmap, return_cfs=True, normalize=False, **hov_kwargs)
        _ELI = _COMPS['eli_total']
        _LAG = _COMPS.lag
        _axs[_j, 0].plot(_ELI.sel(year=Y0_2), _LAG, c='k', ls='-')
        _axs[_j, 1].plot(_ELI.sel(year=Y1_2), _LAG, c='k', ls='--')
        for _y_, _ls in zip([Y0_2, Y1_2], ['-', '--']):
            _axs[_j, -1].plot(_ELI.sel(year=_y_), _LAG, c='k', ls=_ls)
        _fig.colorbar(_cfs[1], ax=_axs[_j, 1], ticks=[-_amp_, 0, _amp_])
        _fig.colorbar(_cfs[-1], ax=_axs[_j, 2], ticks=[-_amp_ / 2, 0, _amp_ / 2], label=_l)
        _axs[_j, -1].add_patch(mpatches.Rectangle(xy=(190, -1), width=50, height=2, edgecolor='magenta', facecolor='none'))
    _axs[0, 0].set_title('2010', fontsize=12)
    _axs[0, 1].set_title('2080', fontsize=12)
    _axs[0, -1].set_title('Difference', fontsize=12)
    for _ax in _axs[1:].flatten():
        _ax.set_title(None)
    for _ax in _axs[:-1].flatten():
        _ax.set_xticks([])
        _ax.set_xlabel(None)
    for _ax in _axs.flatten():
        _ax.set_ylim([-5, 5])
    for _ax in _axs[:, 0]:
        _ax.set_yticks([-6, 0, 6], labels=['Jun(-1)', 'Dec(0)', 'Jun(+1)'])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### dynamical (scaled by temperature anom.)
    """)
    return


@app.cell
def _(comps_cold_1, hov_kwargs, mpatches, np, plot_comp_on_axs, plt):
    Y0_3 = 2010
    Y1_3 = 2080
    _COMPS = comps_cold_1 / comps_cold_1['T_3']
    _AMP_ = 0.5
    _fig, _axs = plt.subplots(4, 3, figsize=(8, 8), layout='constrained')
    for _j, (_name, _amp, _l) in enumerate(zip(['ddt_T_ml', 'Q', 'THF', 'zonal'], [_AMP_, _AMP_, _AMP_, _AMP_], ["$\\left(T'\\right)^{-1}\\cdot \\frac{\\partial}{\\partial t}\\left<T\\right>$", "$\\left(T'\\right)^{-1}\\cdot Q$", "$\\left(T'\\right)^{-1}\\cdot\\left<\\overline{w}~\\frac{\\partial T'}{\\partial z}\\right>$", "$\\left(T'\\right)^{-1}\\cdot$ zonal adv"])):
        if _name == 'pr':
            _cmap = 'cmo.balance_r'
            _amp_ = np.round(86000.0 * _amp)
        else:
            _cmap = 'cmo.balance'
            _amp_ = _amp
        _cfs = plot_comp_on_axs(_axs[_j], name=_name, amp=_amp, amp_diff=_amp / 2, x0=_COMPS.sel(year=Y0_3), x1=_COMPS.sel(year=Y1_3), nlev=10, cmap=_cmap, return_cfs=True, normalize=False, **hov_kwargs)
        _ELI = _COMPS['eli_total']
        _LAG = _COMPS.lag
        _axs[_j, 0].plot(_ELI.sel(year=Y0_3), _LAG, c='k', ls='-')
        _axs[_j, 1].plot(_ELI.sel(year=Y1_3), _LAG, c='k', ls='--')
        for _y_, _ls in zip([Y0_3, Y1_3], ['-', '--']):
            _axs[_j, -1].plot(_ELI.sel(year=_y_), _LAG, c='k', ls=_ls)
        _fig.colorbar(_cfs[1], ax=_axs[_j, 1], ticks=[-_amp_, 0, _amp_])
        _fig.colorbar(_cfs[-1], ax=_axs[_j, 2], ticks=[-_amp_ / 2, 0, _amp_ / 2], label=_l)
        _axs[_j, -1].add_patch(mpatches.Rectangle(xy=(190, -1), width=50, height=2, edgecolor='magenta', facecolor='none'))
    _axs[0, 0].set_title('2010', fontsize=12)
    _axs[0, 1].set_title('2080', fontsize=12)
    _axs[0, -1].set_title('Difference', fontsize=12)
    for _ax in _axs[1:].flatten():
        _ax.set_title(None)
    for _ax in _axs[:-1].flatten():
        _ax.set_xticks([])
        _ax.set_xlabel(None)
    for _ax in _axs.flatten():
        _ax.set_ylim([-5, 5])
    for _ax in _axs[:, 0]:
        _ax.set_yticks([-5, 0, 5], labels=['Jul(-1)', 'Dec(0)', 'May(+1)'])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### zonal terms
    """)
    return


@app.cell
def _(comps_warm_1, hov_kwargs, mpatches, np, plot_comp_on_axs, plt):
    Y0_4 = 2010
    Y1_4 = 2080
    _COMPS = comps_warm_1
    _AMP_ = 1.0
    _fig, _axs = plt.subplots(4, 3, figsize=(8, 8), layout='constrained')
    for _j, (_name, _amp, _l) in enumerate(zip(['zonal', 'ZAF_ml', 'DD_ml', 'ndh_ml'], [_AMP_, _AMP_, _AMP_, _AMP_], ['zonal (sum)', "$\\left<u'~\\frac{\\partial \\overline{T}}{\\partial x}\\right>$", "$\\left<\\overline{u}~\\frac{\\partial T'}{\\partial x}\\right>$", "$\\left<u'~\\frac{\\partial T'}{\\partial x}\\right>$"])):
        if _name == 'pr':
            _cmap = 'cmo.balance_r'
            _amp_ = np.round(86000.0 * _amp)
        else:
            _cmap = 'cmo.balance'
            _amp_ = _amp
        _cfs = plot_comp_on_axs(_axs[_j], name=_name, amp=_amp, amp_diff=_amp / 2, x0=_COMPS.sel(year=Y0_4), x1=_COMPS.sel(year=Y1_4), nlev=10, cmap=_cmap, return_cfs=True, normalize=False, **hov_kwargs)
        _ELI = _COMPS['eli_total']
        _LAG = _COMPS.lag
        _axs[_j, 0].plot(_ELI.sel(year=Y0_4), _LAG, c='k', ls='-')
        _axs[_j, 1].plot(_ELI.sel(year=Y1_4), _LAG, c='k', ls='--')
        for _y_, _ls in zip([Y0_4, Y1_4], ['-', '--']):
            _axs[_j, -1].plot(_ELI.sel(year=_y_), _LAG, c='k', ls=_ls)
        _fig.colorbar(_cfs[1], ax=_axs[_j, 1], ticks=[-_amp_, 0, _amp_])
        _fig.colorbar(_cfs[-1], ax=_axs[_j, 2], ticks=[-_amp_ / 2, 0, _amp_ / 2], label=_l)
        _axs[_j, -1].add_patch(mpatches.Rectangle(xy=(190, -1), width=50, height=2, edgecolor='magenta', facecolor='none'))
    _axs[0, 0].set_title('2010', fontsize=12)
    _axs[0, 1].set_title('2080', fontsize=12)
    _axs[0, -1].set_title('Difference', fontsize=12)
    for _ax in _axs[1:].flatten():
        _ax.set_title(None)
    for _ax in _axs[:-1].flatten():
        _ax.set_xticks([])
        _ax.set_xlabel(None)
    for _ax in _axs.flatten():
        _ax.set_ylim([-5, 5])
    for _ax in _axs[:, 0]:
        _ax.set_yticks([-6, 0, 6], labels=['Jun(-1)', 'Dec(0)', 'Jun(+1)'])
    plt.gca()
    return Y0_4, Y1_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spatial plot for wind
    """)
    return


@app.cell
def _(ccrs, src):
    def plot_2d(
        ax,
        data,
        amp=None,
        is_filled=True,
        xticks=[190, 240],
        lat_bound=5,
        nlev=10,
        cmap="cmo.balance",
        levels=None,
    ):
        """plot data on ax object"""

        ## set levels if not specified
        if levels is None:
            levels = src.utils.make_cb_range(amp, amp / nlev)

        ## specify shared kwargs
        shared_kwargs = dict(levels=levels, extend="both")

        ## specify kwargs
        if is_filled:
            plot_fn = ax.contourf
            kwargs = dict(cmap=cmap)

        else:
            plot_fn = ax.contour
            kwargs = dict(colors="k", linewidths=0.8)

        ## do the plotting
        cp = plot_fn(
            data.longitude,
            data.latitude,
            data.transpose("latitude", "longitude"),
            transform=ccrs.PlateCarree(),
            **kwargs,
            **shared_kwargs,
        )
        return cp

    return (plot_2d,)


@app.cell
def _(Y0_4, Y1_4, clim, plot_2d, plt, src):
    _LAG = 2
    _AMP = 20
    _X = 86000.0 * clim['pr'].sel(lag=_LAG)
    _X0 = _X.sel(year=Y0_4)
    _X1 = _X.sel(year=Y1_4)
    _dX = _X1 - _X0
    _fig = plt.figure(figsize=(10, 3), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=1, format_func=_format_func)
    ddt_sst_kwargs = dict(cmap='cmo.balance_r')
    _cp_pr = plot_2d(_axs[0, 0], _X0, amp=_AMP, **ddt_sst_kwargs)
    _cp_pr = plot_2d(_axs[1, 0], _X1, amp=_AMP, **ddt_sst_kwargs)
    _cp_pr = plot_2d(_axs[2, 0], _dX, amp=_AMP / 2, **ddt_sst_kwargs)
    for _ax in _axs.flatten():
        src.utils.plot_nino4_box(_ax, c='k', lw=0.8)
        src.utils.plot_nino3_box(_ax, c='k', lw=0.8)
    plt.gca()
    return


@app.cell
def _(Y0_4, Y1_4, comps_cold_1, np, plot_2d, plt, src):
    _LAG = 2
    _AMP = 0.06
    _X = comps_cold_1['taux']
    _X0 = _X.sel(year=Y0_4)
    _X1 = _X.sel(year=Y1_4)
    _dX = _X1 - _X0
    _fig = plt.figure(figsize=(10, 10), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    _axs = src.utils.subplots_with_proj(_fig, nrows=8, ncols=1, format_func=_format_func)
    for _j, _lag in enumerate(np.arange(-2, 6)):
        plot_2d(_axs[_j, 0], _X0.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
    for _ax in _axs.flatten():
        src.utils.plot_nino4_box(_ax, c='k', lw=0.8)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### spatial asymmetry for wind
    """)
    return


@app.cell
def _(comps_cold_1, comps_warm_1, plot_2d, plt, src):
    _LAG = slice(1, 3)
    _sel = lambda x: x.sel(year=2010, lag=_LAG).mean('lag')
    _AMP = 0.03
    p_warm = comps_warm_1['taux'] / comps_warm_1['T_34']
    p_cold = comps_cold_1['taux'] / comps_cold_1['T_34']
    _fig = plt.figure(figsize=(10, 4.5), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=1, format_func=_format_func)
    plot_2d(_axs[0, 0], _sel(p_warm), amp=_AMP, cmap='cmo.balance')
    plot_2d(_axs[1, 0], _sel(p_cold), amp=_AMP, cmap='cmo.balance')
    plot_2d(_axs[2, 0], _sel(p_warm) - _sel(p_cold), amp=_AMP, cmap='cmo.balance')
    for _ax in _axs.flatten():
        src.utils.plot_nino34_box(_ax, c='k', lw=0.8)
    plt.gca()
    return


@app.cell
def _(comps_cold_1, comps_warm_1, plt):
    get_ohc = lambda x: x['T'].sel(longitude=slice(190, 240)).mean(['longitude', 'z_t'])
    get_ohc_wrapper = lambda x: get_ohc(x).sel(year=1870)
    ohc_warm = get_ohc_wrapper(comps_warm_1)
    ohc_cold = get_ohc_wrapper(comps_cold_1)
    _fig, _ax = plt.subplots(figsize=(3, 2.5))
    _ax.plot(ohc_cold.lag, -ohc_cold.differentiate('lag'))
    _ax.plot(ohc_cold.lag, ohc_warm.differentiate('lag'))
    _ax.axvline(0, ls='--', c='k', lw=1)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### more...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Spatial plot for heat flux
    """)
    return


@app.cell
def _(ccrs, src):
    def plot_2d_1(ax, data, amp=None, is_filled=True, xticks=[190, 240], lat_bound=5, nlev=10, cmap='cmo.balance', levels=None):
        """plot data on ax object"""
        if levels is None:
            levels = src.utils.make_cb_range(amp, amp / nlev)
        shared_kwargs = dict(levels=levels, extend='both')
        if is_filled:
            plot_fn = ax.contourf
            kwargs = dict(cmap=cmap)
        else:
            plot_fn = ax.contour
            kwargs = dict(colors='k', linewidths=0.8)
        cp = plot_fn(data.longitude, data.latitude, data.transpose('latitude', 'longitude'), transform=ccrs.PlateCarree(), **kwargs, **shared_kwargs)
        return cp  ## set levels if not specified  ## specify shared kwargs  ## specify kwargs  ## do the plotting

    return (plot_2d_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $sst$
    """)
    return


@app.cell
def _(comps_warm_1, np, plot_2d_1, plt, src):
    Y0_5 = 1870
    Y1_5 = 1990
    _AMP = 2
    _X = comps_warm_1['sst_rel'] / comps_warm_1['T_34']
    _X0 = _X.sel(year=Y0_5)
    _X1 = _X.sel(year=Y1_5)
    _dX = _X1 - _X0
    _fig = plt.figure(figsize=(12, 9), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    _axs = src.utils.subplots_with_proj(_fig, nrows=8, ncols=3, format_func=_format_func)
    for _j, _lag in enumerate(np.arange(0, 8)):
        plot_2d_1(_axs[_j, 0], _X0.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
        plot_2d_1(_axs[_j, 1], _X1.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
        plot_2d_1(_axs[_j, 2], (_X1 - _X0).sel(lag=_lag), amp=_AMP * 0.3, cmap='cmo.balance')
        _axs[_j, 0].set_title(_lag)
    for _ax in _axs.flatten():
        src.utils.plot_box(_ax, lons=[210, 270], lats=[-5, 5], c='k', lw=0.8)
    plt.gca()
    return


@app.cell
def _(comps_warm_1):
    lat_hov = (comps_warm_1['ssh'] / comps_warm_1['T_3']).isel(year=0).sel(lag=slice(-3, 5), longitude=slice(130, 240), latitude=slice(-15, 15)).mean('longitude')
    lon_hov = (comps_warm_1['ssh'] / comps_warm_1['T_3']).isel(year=0).sel(lag=slice(-3, 5), longitude=slice(130, 280), latitude=slice(-5, 5)).mean('latitude')
    return lat_hov, lon_hov


@app.cell
def _(lat_hov, lon_hov, plt, src):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.contourf(lat_hov.lag, lat_hov.latitude, lat_hov, cmap='cmo.balance', levels=src.utils.make_cb_range(10, 1))
    for _t in [-5, 5]:
        _ax.axhline(_t, ls='--', c='k', lw=0.8)
    plt.show()
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.contourf(lon_hov.longitude, lon_hov.lag, lon_hov.T, cmap='cmo.balance', levels=src.utils.make_cb_range(10, 1))
    for _t in [190, 240]:
        _ax.axvline(_t, ls='--', c='k', lw=0.8)
    plt.gca()
    return


@app.cell
def _(comps_warm_1, np, plot_2d_1, plt, src):
    Y0_6 = 1870
    Y1_6 = 1990
    _AMP = 1
    _X = comps_warm_1['ddt_sst'] / comps_warm_1['T_3']
    _X0 = _X.sel(year=Y0_6)
    _X1 = _X.sel(year=Y1_6)
    _dX = _X1 - _X0
    _fig = plt.figure(figsize=(12, 9), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    _axs = src.utils.subplots_with_proj(_fig, nrows=8, ncols=3, format_func=_format_func)
    for _j, _lag in enumerate(np.arange(0, 8)):
        plot_2d_1(_axs[_j, 0], _X0.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
        plot_2d_1(_axs[_j, 1], _X1.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
        plot_2d_1(_axs[_j, 2], (_X1 - _X0).sel(lag=_lag), amp=_AMP * 0.3, cmap='cmo.balance')
        _axs[_j, 0].set_title(_lag)
    for _ax in _axs.flatten():
        src.utils.plot_box(_ax, lons=[180, 230], lats=[-5, 5], c='k', lw=0.8)
    plt.gca()
    return Y0_6, Y1_6


@app.cell
def _(Y0_6, Y1_6, comps_warm_1, np, plot_2d_1, plt, src):
    _AMP = 4
    _X = comps_warm_1['sst_rel']
    _X0 = _X.sel(year=Y0_6)
    _X1 = _X.sel(year=Y1_6)
    _dX = _X1 - _X0
    _fig = plt.figure(figsize=(12, 9), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    _axs = src.utils.subplots_with_proj(_fig, nrows=8, ncols=3, format_func=_format_func)
    for _j, _lag in enumerate(np.arange(-2, 6)):
        plot_2d_1(_axs[_j, 0], _X0.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
        plot_2d_1(_axs[_j, 1], _X1.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
        plot_2d_1(_axs[_j, 2], (_X1 - _X0).sel(lag=_lag), amp=_AMP * 0.5, cmap='cmo.balance')
        _axs[_j, 0].set_title(_lag - 1)
    for _ax in _axs.flatten():
        src.utils.plot_box(_ax, lons=[180, 230], lats=[-5, 5], c='k', lw=0.8)
    plt.gca()
    return


@app.cell
def _(Y0_6, Y1_6, clim, comps_warm_1, np, plot_2d_1, plt, src):
    _AMP = 20
    _X = (clim['pr'] + comps_warm_1['pr']) * 86000.0
    _X0 = _X.sel(year=Y0_6)
    _X1 = _X.sel(year=Y1_6)
    _dX = _X1 - _X0
    _fig = plt.figure(figsize=(12, 9), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    _axs = src.utils.subplots_with_proj(_fig, nrows=8, ncols=3, format_func=_format_func)
    for _j, _lag in enumerate(np.arange(-2, 6)):
        plot_2d_1(_axs[_j, 0], _X0.sel(lag=_lag), amp=_AMP, cmap='cmo.balance_r')
        plot_2d_1(_axs[_j, 1], _X1.sel(lag=_lag), amp=_AMP, cmap='cmo.balance_r')
        plot_2d_1(_axs[_j, 2], (_X1 - _X0).sel(lag=_lag), amp=_AMP * 0.5, cmap='cmo.balance_r')
        _axs[_j, 0].set_title(_lag - 1)
    for _ax in _axs.flatten():
        src.utils.plot_box(_ax, lons=[180, 230], lats=[-5, 5], c='k', lw=0.8)
    plt.gca()
    return


@app.cell
def _(src):
    help(src.utils.plot_box)
    return


@app.cell
def _(Y0_6, Y1_6, comps_warm_1, np, plot_2d_1, plt, src):
    _AMP = 0.5
    _X = comps_warm_1['ddt_sst'] / comps_warm_1['T_3']
    _X0 = _X.sel(year=Y0_6)
    _X1 = _X.sel(year=Y1_6)
    _dX = _X1 - _X0
    _fig = plt.figure(figsize=(12, 9), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    _axs = src.utils.subplots_with_proj(_fig, nrows=8, ncols=3, format_func=_format_func)
    for _j, _lag in enumerate(np.arange(-2, 6)):
        plot_2d_1(_axs[_j, 0], _X0.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
        plot_2d_1(_axs[_j, 1], _X1.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
        plot_2d_1(_axs[_j, 2], (_X1 - _X0).sel(lag=_lag), amp=_AMP * 0.5, cmap='cmo.balance')
        _axs[_j, 0].set_title(_lag - 1)
    for _ax in _axs.flatten():
        src.utils.plot_box(_ax, lons=[180, 230], lats=[-5, 5], c='k', lw=0.8)
    plt.gca()
    return


@app.cell
def _(Y0_6, Y1_6, comps_warm_1, np, plot_2d_1, plt, src):
    _AMP = 20
    _X = comps_warm_1['taux'] / comps_warm_1['T_3'] * 1000.0
    _X0 = _X.sel(year=Y0_6)
    _X1 = _X.sel(year=Y1_6)
    _dX = _X1 - _X0
    _fig = plt.figure(figsize=(12, 9), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    _axs = src.utils.subplots_with_proj(_fig, nrows=8, ncols=3, format_func=_format_func)
    for _j, _lag in enumerate(np.arange(-2, 6)):
        plot_2d_1(_axs[_j, 0], _X0.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
        plot_2d_1(_axs[_j, 1], _X1.sel(lag=_lag), amp=_AMP, cmap='cmo.balance')
        plot_2d_1(_axs[_j, 2], (_X1 - _X0).sel(lag=_lag), amp=_AMP * 0.5, cmap='cmo.balance')
        _axs[_j, 0].set_title(_lag - 1)
    for _ax in _axs.flatten():
        src.utils.plot_box(_ax, lons=[180, 230], lats=[-5, 5], c='k', lw=0.8)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### sst
    """)
    return


@app.cell
def _(Y0_6, Y1_6, comps_warm_1, plot_2d_1, plt, src):
    _sel = lambda x: x.sel(lag=slice(-2, 0)).mean('lag')
    _COMPS = _sel(comps_warm_1)
    _diff = _COMPS.sel(year=Y1_6) - _COMPS.sel(year=Y0_6)
    _AMP = 1.5
    _fig = plt.figure(figsize=(15, 4.375), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=2, format_func=_format_func)
    cp_nhf = plot_2d_1(_axs[0, 0], _COMPS.sel(year=Y0_6)['ddt_sst'], amp=0.5)
    cp_nhf = plot_2d_1(_axs[0, 1], _diff['ddt_sst'], amp=0.5)
    cp_nhf = plot_2d_1(_axs[1, 0], _COMPS.sel(year=Y0_6)['sst'], amp=4)
    cp_nhf = plot_2d_1(_axs[1, 1], _diff['sst'], amp=_AMP)
    _cp_pr = plot_2d_1(_axs[2, 0], 86000.0 * _COMPS.sel(year=Y0_6)['pr'], amp=12, cmap='cmo.balance_r')
    _cp_pr = plot_2d_1(_axs[2, 1], 86000.0 * _diff['pr'], amp=6, cmap='cmo.balance_r')
    for _ax in _axs.flatten():
        src.utils.plot_nino34_box(_ax, c='k', lw=1)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### same, but subsurface
    """)
    return


@app.cell
def _(Y0_6, Y1_6, comps_cold_1, comps_warm_1, plt, src):
    _AMP = 5.0
    AMP_DIFF = 2.5
    _sel = lambda x: x.sel(lag=slice(4, 6)).mean('lag')
    _fig, _axs = plt.subplots(3, 3, figsize=(10, 6), layout='constrained')
    for _j, comps_ in enumerate([comps_warm_1, comps_cold_1]):
        _COMPS = _sel(comps_)
        _diff = _COMPS.sel(year=Y1_6) - _COMPS.sel(year=Y0_6)
        for _i, _y in enumerate([Y0_6, Y1_6]):
            _cp = _axs[_j, _i].contourf(_COMPS.longitude, _COMPS.z_t, _COMPS['T'].sel(year=_y), cmap='cmo.balance', levels=src.utils.make_cb_range(_AMP, _AMP / 10), extend='both')
        cp_diff = _axs[_j, 2].contourf(_diff.longitude, _diff.z_t, _diff['T'], cmap='cmo.balance', levels=src.utils.make_cb_range(AMP_DIFF, AMP_DIFF / 10), extend='both')
    _cb = _fig.colorbar(_cp, ax=_axs[0, 1], ticks=[-_AMP, _AMP], label='$K$')
    cb_diff = _fig.colorbar(cp_diff, ax=_axs[0, 2], ticks=[-AMP_DIFF, AMP_DIFF], label='$K$')
    for _ax in _axs.flatten():
        _ax.set_xlim([140, 280])
        _ax.set_ylim([200, 5])
        _ax.axvline(190, ls='--', c='w', lw=0.8)
        _ax.axvline(240, ls='--', c='w', lw=0.8)
    for _ax in _axs[:, 0]:
        _ax.set_yticks([200, 100, 0])
    for _ax in _axs[:, 1:].flatten():
        _ax.set_yticks([])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### relative SST
    """)
    return


@app.cell
def _(clim):
    list(clim)
    return


@app.cell
def _():
    Y1_7 = 2000
    return (Y1_7,)


@app.cell
def _(Y0_6, Y1_7, comps_warm_1, hov_kwargs, plot_comp_on_axs, plt):
    _COMPS = comps_warm_1
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='sst_rel', amp=4, x0=_COMPS.sel(year=Y0_6), x1=_COMPS.sel(year=Y1_7), nlev=10, **hov_kwargs)
    _ELI = _COMPS['eli_total']
    _LAG = _COMPS.lag
    _axs[0].plot(_ELI.sel(year=Y0_6), _LAG, c='k', ls='-')
    _axs[1].plot(_ELI.sel(year=Y1_7), _LAG, c='k', ls='--')
    for _y_, _ls in zip([Y0_6, Y1_7], ['-', '--']):
        _axs[-1].plot(_ELI.sel(year=_y_), _LAG, c='k', ls=_ls)
    for _ax in _axs:
        _ax.axhline(4, ls='--', c='gray', lw=0.8)
        _ax.axhline(6, ls='--', c='gray', lw=0.8)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-3.5, 0, 3.5], label='K')
    _axs[-1].set_title('Difference (x2)', fontsize=10)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### precip
    """)
    return


@app.cell
def _(Y0_6, Y1_7, comps_cold_1, hov_kwargs, plot_comp_on_axs, plt):
    _COMPS = comps_cold_1
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='pr', amp=12, x0=86000.0 * _COMPS.sel(year=Y0_6), x1=86000.0 * _COMPS.sel(year=Y1_7), nlev=10, cmap='cmo.balance_r', **hov_kwargs)
    _ELI = _COMPS['eli_total']
    _LAG = _COMPS.lag
    _axs[0].plot(_ELI.sel(year=Y0_6), _LAG, c='k', ls='-')
    _axs[1].plot(_ELI.sel(year=Y1_7), _LAG, c='k', ls='--')
    for _y_, _ls in zip([Y0_6, Y1_7], ['-', '--']):
        _axs[-1].plot(_ELI.sel(year=_y_), _LAG, c='k', ls=_ls)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-12, 0, 12], label='mm day$^{-1}$')
    _axs[-1].set_title('Difference (x2)', fontsize=10)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Wind
    """)
    return


@app.cell
def _(Y0_6, Y1_7, comps_cold_1, hov_kwargs, plot_comp_on_axs, plt):
    _COMPS = comps_cold_1
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='taux', amp=3, x0=100.0 * _COMPS.sel(year=Y0_6), x1=100.0 * _COMPS.sel(year=Y1_7), nlev=10, cmap='cmo.balance', **hov_kwargs)
    _ELI = _COMPS['eli_total']
    _LAG = _COMPS.lag
    _axs[0].plot(_ELI.sel(year=Y0_6), _LAG, c='k', ls='-')
    _axs[1].plot(_ELI.sel(year=Y1_7), _LAG, c='k', ls='--')
    for _y_, _ls in zip([Y0_6, Y1_7], ['-', '--']):
        _axs[-1].plot(_ELI.sel(year=_y_), _LAG, c='k', ls=_ls)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-12, 0, 12], label='mm day$^{-1}$')
    _axs[-1].set_title('Difference (x2)', fontsize=10)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Heat flux
    """)
    return


@app.cell
def _(Y0_6, Y1_7, comps_cold_1, hov_kwargs, plot_comp_on_axs, plt):
    _COMPS = comps_cold_1
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='nhf', amp=100, x0=_COMPS.sel(year=Y0_6), x1=_COMPS.sel(year=Y1_7), nlev=10, cmap='cmo.balance', **hov_kwargs)
    _ELI = _COMPS['eli_total']
    _LAG = _COMPS.lag
    _axs[0].plot(_ELI.sel(year=Y0_6), _LAG, c='magenta', ls='-')
    _axs[1].plot(_ELI.sel(year=Y1_7), _LAG, c='magenta', ls='--')
    for _y_, _ls in zip([Y0_6, Y1_7], ['-', '--']):
        _axs[-1].plot(_ELI.sel(year=_y_), _LAG, c='magenta', ls=_ls)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-100, 0, 100], label='W m$^{-2}$')
    _axs[-1].set_title('Difference (x2)', fontsize=10)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### shortwave
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TO-DO...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### warm pool stuff
    """)
    return


@app.cell
def _(comps_warm_1):
    # eq_peak = -comps_cold.sel(latitude=slice(-5,5), lag=0).mean("latitude")
    eq_peak = comps_warm_1.sel(latitude=slice(-5, 5), lag=0).mean('latitude')
    return (eq_peak,)


@app.cell
def _(eq_peak, np, plt):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    for _y in np.arange(1870, 2120, 50):
        _ax.plot(eq_peak.longitude, eq_peak['sst'].sel(year=_y), label=_y)
        _ax.set_xlim([140, 280])
    _ax.legend()
    plt.gca()
    return


@app.cell
def _(ccrs, comps_warm_1, plt, src):
    COMP = comps_warm_1['sst'].sel(lag=1)
    _fig = plt.figure(figsize=(7.5, 3), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=15)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=1, format_func=_format_func)
    kwargs_1 = dict(cmap='cmo.balance', levels=src.utils.make_cb_range(4, 0.4), extend='both', transform=ccrs.PlateCarree())
    _axs[0, 0].contourf(COMP.longitude, COMP.latitude, COMP.sel(year=1870), **kwargs_1)
    _axs[1, 0].contourf(COMP.longitude, COMP.latitude, COMP.sel(year=2080), **kwargs_1)
    _axs[2, 0].contourf(COMP.longitude, COMP.latitude, 4 * (COMP.sel(year=2080) - COMP.sel(year=1870)), **kwargs_1)
    for _ax in _axs.flatten():
        _ax.axhline(5, ls='--', c='k')
        _ax.axhline(-5, ls='--', c='k')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SSH
    """)
    return


@app.cell
def _():
    sel_ = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")
    return (sel_,)


@app.cell
def _(comps_cold_1, hov_kwargs, plot_comp_on_axs, plt, sel_):
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='ssh', amp=10, amp_diff=5, x0=(sel_(comps_cold_1) / comps_cold_1['T_3']).sel(year=1870), x1=(sel_(comps_cold_1) / comps_cold_1['T_3']).sel(year=2080), nlev=10, **hov_kwargs)
    for _ax in _axs:
        _ax.set_xlim([120, None])
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-2.5, 0, 2.5], label='K')
    plt.gca()
    return


@app.cell
def _(comps_cold_1, plt, sel_):
    plt.plot(comps_cold_1.longitude, (sel_(comps_cold_1['ssh']) / comps_cold_1['T_34']).sel(year=1870).sel(lag=2))
    plt.plot(comps_cold_1.longitude, (sel_(comps_cold_1['ssh']) / comps_cold_1['T_34']).sel(year=2080).sel(lag=2))
    return


@app.cell
def _(ccrs, comps_cold_1, plt, src):
    COMP_1 = (comps_cold_1['ssh'] / comps_cold_1['T_3']).sel(lag=0)
    _fig = plt.figure(figsize=(7.5, 3), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=15)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=1, format_func=_format_func)
    kwargs_2 = dict(cmap='cmo.balance', levels=src.utils.make_cb_range(7.5, 0.75), extend='both', transform=ccrs.PlateCarree())
    _axs[0, 0].contourf(COMP_1.longitude, COMP_1.latitude, COMP_1.sel(year=1870), **kwargs_2)
    _axs[1, 0].contourf(COMP_1.longitude, COMP_1.latitude, COMP_1.sel(year=2080), **kwargs_2)
    _axs[2, 0].contourf(COMP_1.longitude, COMP_1.latitude, 2 * (COMP_1.sel(year=2080) - COMP_1.sel(year=1870)), **kwargs_2)
    for _ax in _axs.flatten():
        _ax.axhline(5, ls='--', c='k')
        _ax.axhline(-5, ls='--', c='k')
    plt.gca()
    return


@app.cell
def _(comps_warm_1, hov_kwargs, plot_comp_on_axs, plt):
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='taux', amp=0.05, amp_diff=0.02, x0=comps_warm_1.sel(year=2010), x1=comps_warm_1.sel(year=2080), nlev=10, **hov_kwargs)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-2.5, 0, 2.5], label='K')
    for _ax in _axs:
        _ax.axhline(2, ls='--', c='w', lw=0.8)
    plt.gca()
    return


@app.cell
def _(ccrs, comps_warm_1, plt, src):
    COMP_2 = -comps_warm_1['tauy'].sel(lag=1)
    _fig = plt.figure(figsize=(7.5, 3), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=15)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=1, format_func=_format_func)
    kwargs_3 = dict(cmap='cmo.balance', levels=src.utils.make_cb_range(0.04, 0.004), extend='both', transform=ccrs.PlateCarree())
    _axs[0, 0].contourf(COMP_2.longitude, COMP_2.latitude, COMP_2.sel(year=1870), **kwargs_3)
    _axs[1, 0].contourf(COMP_2.longitude, COMP_2.latitude, COMP_2.sel(year=2080), **kwargs_3)
    _axs[2, 0].contourf(COMP_2.longitude, COMP_2.latitude, 2 * (COMP_2.sel(year=2080) - COMP_2.sel(year=1870)), **kwargs_3)
    for _ax in _axs.flatten():
        _ax.axhline(5, ls='--', c='k')
        _ax.axhline(-5, ls='--', c='k')
    plt.gca()
    return


@app.cell
def _(ccrs, clim, comps_warm_1, np, plt, src):
    COMP_3 = (clim['sst'] + comps_warm_1['sst']).sel(lag=0)
    _fig = plt.figure(figsize=(7.5, 3), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=15)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=1, format_func=_format_func)
    kwargs_4 = dict(cmap='cmo.thermal', levels=np.arange(20, 35, 1), extend='both', transform=ccrs.PlateCarree())
    _axs[0, 0].contourf(COMP_3.longitude, COMP_3.latitude, COMP_3.sel(year=1870), **kwargs_4)
    _axs[1, 0].contourf(COMP_3.longitude, COMP_3.latitude, COMP_3.sel(year=2080), **kwargs_4)
    _axs[2, 0].contourf(COMP_3.longitude, COMP_3.latitude, COMP_3.sel(year=2080) - COMP_3.sel(year=1870), cmap='cmo.balance', levels=src.utils.make_cb_range(2, 0.2), extend='both', transform=ccrs.PlateCarree())
    for _ax in _axs.flatten():
        _ax.axhline(5, ls='--', c='k')
        _ax.axhline(-5, ls='--', c='k')
    plt.gca()
    return (COMP_3,)


@app.cell
def _(COMP_3, plt):
    sel0 = lambda x: x.sel(latitude=slice(-5, 5)).mean('latitude')
    _sel = lambda x: sel0(x) - sel0(x).mean('longitude')
    plt.plot(COMP_3.longitude, _sel(COMP_3.isel(year=0)))
    plt.plot(COMP_3.longitude, _sel(COMP_3.isel(year=-1)))
    plt.axvline(190)
    return


@app.cell
def _(ccrs, comps_cold_1, plt, src):
    COMP_4 = comps_cold_1['pr'].sel(lag=1)
    _fig = plt.figure(figsize=(7.5, 3), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=15)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=1, format_func=_format_func)
    kwargs_5 = dict(cmap='cmo.balance_r', levels=src.utils.make_cb_range(0.00015, 1.5e-05), extend='max', transform=ccrs.PlateCarree())
    _axs[0, 0].contourf(COMP_4.longitude, COMP_4.latitude, COMP_4.sel(year=1870), **kwargs_5)
    _axs[1, 0].contourf(COMP_4.longitude, COMP_4.latitude, COMP_4.sel(year=2080), **kwargs_5)
    _axs[2, 0].contourf(COMP_4.longitude, COMP_4.latitude, COMP_4.sel(year=2080) - COMP_4.sel(year=1870), cmap='cmo.balance_r', levels=src.utils.make_cb_range(0.00015, 1.5e-05), extend='both', transform=ccrs.PlateCarree())
    for _ax in _axs.flatten():
        _ax.axhline(5, ls='--', c='k')
        _ax.axhline(-5, ls='--', c='k')
    plt.gca()
    return (COMP_4,)


@app.cell
def _(comps_warm_1, plt):
    max_lon_idx = comps_warm_1['pr'].sel(lag=0, latitude=slice(-5, 5)).mean('latitude').sel(longitude=slice(140, 280)).argmax('longitude')
    plt.plot(comps_warm_1.year, comps_warm_1.sel(longitude=slice(140, 280)).longitude.isel(longitude=max_lon_idx.values))
    return


@app.cell
def _(COMP_4, plt):
    plt.plot(COMP_4.longitude, COMP_4.isel(year=0).sel(latitude=slice(-5, 5)).mean('latitude'))
    plt.plot(COMP_4.longitude, COMP_4.isel(year=-1).sel(latitude=slice(-5, 5)).mean('latitude'))
    plt.axvline(190)
    return


@app.cell
def _(COMP_4, plt):
    plt.plot(COMP_4.latitude, COMP_4.sel(longitude=slice(140, 280)).mean('longitude').isel(year=0))
    plt.plot(COMP_4.latitude, COMP_4.sel(longitude=slice(140, 280)).mean('longitude').isel(year=-1))
    plt.xlim([-10, 10])
    return


@app.cell
def _(ccrs, comps_warm_1, plt, src):
    COMP_5 = comps_warm_1['nhf'].sel(lag=0)
    _fig = plt.figure(figsize=(7.5, 3), layout='constrained')
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=15)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=1, format_func=_format_func)
    kwargs_6 = dict(cmap='cmo.balance', levels=src.utils.make_cb_range(80, 8), extend='both', transform=ccrs.PlateCarree())
    _axs[0, 0].contourf(COMP_5.longitude, COMP_5.latitude, COMP_5.sel(year=1870), **kwargs_6)
    _axs[1, 0].contourf(COMP_5.longitude, COMP_5.latitude, COMP_5.sel(year=2080), **kwargs_6)
    _axs[2, 0].contourf(COMP_5.longitude, COMP_5.latitude, COMP_5.sel(year=2080) - COMP_5.sel(year=1870), cmap='cmo.balance', levels=src.utils.make_cb_range(40, 4), extend='both', transform=ccrs.PlateCarree())
    for _ax in _axs.flatten():
        _ax.axhline(5, ls='--', c='k')
        _ax.axhline(-5, ls='--', c='k')
    plt.gca()
    return


@app.cell
def _(comps_warm_1):
    h0 = comps_warm_1['ssh'].sel(year=1870).sel(longitude=slice(210, 270)).mean('longitude')
    h1 = comps_warm_1['ssh'].sel(year=2090).sel(longitude=slice(210, 270)).mean('longitude')
    return h0, h1


@app.cell
def _(h0, h1, plt, src):
    _LAG = 2
    _fig, _axs = plt.subplots(3, 1, figsize=(4, 7))
    _axs[0].contourf(h0.lag, h0.latitude, h0.differentiate('lag'), cmap='cmo.balance', levels=src.utils.make_cb_range(8, 0.8))
    _axs[1].contourf(h0.lag, h0.latitude, h1.differentiate('lag'), cmap='cmo.balance', levels=src.utils.make_cb_range(8, 0.8))
    _axs[2].contourf(h0.lag, h0.latitude, 2 * (h1 - h0).differentiate('lag'), cmap='cmo.balance', levels=src.utils.make_cb_range(8, 0.8))
    for _ax in _axs:
        _ax.set_ylim([-10, 10])
        _ax.axvline(0, ls='--', c='k')
        _ax.axhline(0, ls='--', c='k')
    plt.gca()
    return


@app.cell
def _(comps_warm_1, hov_kwargs, plot_comp_on_axs, plt):
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='sst', amp=3.5, x0=comps_warm_1.sel(year=1870), x1=comps_warm_1.sel(year=2070), nlev=10, **hov_kwargs)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-2.5, 0, 2.5], label='K')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### latitude hovmollers
    """)
    return


@app.cell
def _(comps_cold_1, comps_warm_1):
    # LON_RANGE=slice(150,240)
    LON_RANGE = slice(210, 270)
    # LON_RANGE=slice(240,275)
    lon_avg = lambda x: x.sel(longitude=LON_RANGE).mean('longitude')
    lat_hov_warm = lon_avg(comps_warm_1)
    ## get longitudinal means
    # lat_hov_warm = comps_warm.sel(longitude=slice(210,270)).mean("longitude")
    # lat_hov_cold = comps_cold.sel(longitude=slice(210,270)).mean("longitude")
    lat_hov_cold = lon_avg(comps_cold_1)
    return lat_hov_warm, lon_avg


@app.cell
def _(lat_hov_warm, plt, src):
    ## specify amplitude
    # AMP = 2e-2
    _AMP = 0.00015
    # AMP = 3
    HOV = lat_hov_warm['pr']
    _fig, _axs = plt.subplots(3, 1, figsize=(3, 6))
    for _ax, _y in zip(_axs, [2000, 2080]):
        _ax.contourf(HOV.lag, HOV.latitude, HOV.sel(year=_y), cmap='cmo.balance_r', levels=src.utils.make_cb_range(_AMP, _AMP / 5), extend='both')
    _axs[-1].contourf(HOV.lag, HOV.latitude, HOV.sel(year=2080) - HOV.sel(year=2000), cmap='cmo.balance_r', levels=src.utils.make_cb_range(_AMP / 4, _AMP / 20), extend='both')
    for _ax in _axs:
        _ax.set_ylim([-10, 10])
        _ax.set_xlim([-6, 6])
        _ax.axhline(0, ls='--', c='k', lw=0.8)
        _ax.axvline(2, ls='--', c='w')
    plt.gca()
    return


@app.cell
def _(clim, lon_avg, np, plt, src):
    HOV_1 = 86000.0 * lon_avg(clim['pr']).transpose('latitude', ...)
    _fig, _axs = plt.subplots(3, 1, figsize=(3, 6))
    for _ax, _y in zip(_axs, [2000, 2080]):
        _ax.contourf(HOV_1.lag, HOV_1.latitude, HOV_1.sel(year=_y), cmap='cmo.rain', levels=np.arange(0, 18, 2), extend='both')
    _axs[-1].contourf(HOV_1.lag, HOV_1.latitude, HOV_1.sel(year=2080) - HOV_1.sel(year=2000), cmap='cmo.balance_r', levels=src.utils.make_cb_range(8, 1), extend='both')
    for _ax in _axs:
        _ax.set_ylim([-15, 15])
        _ax.set_xlim([-6, 6])
        _ax.axhline(0, ls='--', c='k', lw=0.8)
        _ax.axvline(3, ls='--', c='w')
    plt.gca()
    return (HOV_1,)


@app.cell
def _(HOV_1, plt):
    _LAG = 3
    _fig, _ax = plt.subplots(figsize=(3, 2.5))
    _ax.plot(HOV_1.sel(year=1870, lag=_LAG), HOV_1.latitude)
    _ax.plot(HOV_1.sel(year=1980, lag=_LAG), HOV_1.latitude)
    _ax.set_ylim([-10, 10])
    _ax.axvline(0, ls='--', c='k')
    _ax.axhline(0, ls='--', c='k')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same, but including mean state
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Asymmetry
    """)
    return


@app.cell
def _():
    YEAR = 2070
    return (YEAR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SST
    """)
    return


@app.cell
def _(YEAR, comps_cold_1, comps_warm_1, hov_kwargs, plot_comp_on_axs, plt):
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='sst', amp=3.5, x1=comps_warm_1.sel(year=YEAR), x0=-comps_cold_1.sel(year=YEAR), nlev=10, **hov_kwargs)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-30, 30], label='K')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SSH
    """)
    return


@app.cell
def _(YEAR, comps_cold_1, comps_warm_1, hov_kwargs, plot_comp_on_axs, plt):
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='ssh', amp=25, x1=comps_warm_1.sel(year=YEAR), x0=-comps_cold_1.sel(year=YEAR), nlev=10, **hov_kwargs)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-30, 30], label='K')
    for _ax in _axs:
        _ax.set_xlim([120, None])
    plt.gca()
    return


@app.cell
def _(comps_cold_1, comps_warm_1, plt, src):
    sel__1 = lambda x, y: x.sel(year=y, lag=0, latitude=slice(-5, 5)).mean('latitude')
    _fig, _axs = plt.subplots(1, 3, figsize=(8, 2.5))
    for _ax, _y, _ls in zip(_axs[:-1], [1870, 2070], ['-', '--']):
        _ax.plot(comps_warm_1.longitude, -sel__1(comps_cold_1['ssh'], y=_y), ls=_ls)
        _ax.plot(comps_warm_1.longitude, sel__1(comps_warm_1['ssh'], y=_y), ls=_ls)
        _axs[-1].plot(comps_warm_1.longitude, sel__1(comps_warm_1['ssh'] + comps_cold_1['ssh'], y=_y), ls=_ls, c='k')
    for _ax in _axs:
        ref_kwargs = dict(ls='--', c='k', lw=0.8)
        _ax.axvline(210, **ref_kwargs)
        _ax.axhline(0, **ref_kwargs)
        _ax.set_xlim([120, 270])
    _axs[1].set_yticks([])
    src.utils.set_lims(_axs)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $\tau_x$
    """)
    return


@app.cell
def _():
    YEAR_1 = 2080
    return (YEAR_1,)


@app.cell
def _(YEAR_1, comps_cold_1, comps_warm_1, hov_kwargs, plot_comp_on_axs, plt):
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='taux', amp=0.05, x1=comps_warm_1.sel(year=YEAR_1), x0=-comps_cold_1.sel(year=YEAR_1), nlev=10, **hov_kwargs)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-30, 30], label='K')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $Z_{20}$
    """)
    return


@app.cell
def _(YEAR_1, comps_cold_1, comps_warm_1, hov_kwargs, plot_comp_on_axs, plt):
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='z20', amp=50, x1=comps_warm_1.sel(year=YEAR_1), x0=-comps_cold_1.sel(year=YEAR_1), nlev=10, **hov_kwargs)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-30, 30], label='K')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $Q$
    """)
    return


@app.cell
def _(YEAR_1, comps_cold_1, comps_warm_1, hov_kwargs, plot_comp_on_axs, plt):
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='Q', amp=1, x1=comps_warm_1.sel(year=YEAR_1), x0=-comps_cold_1.sel(year=YEAR_1), nlev=10, **hov_kwargs)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-2.5, 0, 2.5], label='K')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### budget
    """)
    return


@app.cell
def _(comps_warm_1, hov_kwargs, plot_comp_on_axs, plt):
    _comps = comps_warm_1
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 3), layout='constrained')
    _cf = plot_comp_on_axs(_axs, name='Q', amp=0.75, x0=_comps.sel(year=1990), x1=_comps.sel(year=2090), nlev=10, **hov_kwargs)
    _cb = _fig.colorbar(_cf, ax=_axs[2], ticks=[-2.5, 0, 2.5], label='K')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hovmollers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### by season
    """)
    return


@app.cell
def _(comps_cold_1, hov_kwargs, plot_comp_on_axs, plt):
    Y0_7 = 2010
    Y1_8 = 2080
    _COMPS = comps_cold_1 / comps_cold_1['T_3']
    _fig, _axs = plt.subplots(4, 3, figsize=(6, 9), layout='constrained')
    for _j, _n in enumerate(['THF', 'ZAF_ml', 'ddt_T_ml', 'Q']):
        print(_n)
        _cf = plot_comp_on_axs(_axs[_j, :], name=_n, amp=0.4, amp_diff=0.2, lat_bound=2, **dict(hov_kwargs, x0=_COMPS.sel(year=Y0_7), x1=_COMPS.sel(year=Y1_8)))
        _cb = _fig.colorbar(_cf, ax=_axs[_j, 2], ticks=[-1, 0, 1], label='K/mo')
    for _ax in _axs[:-1].flatten():
        _ax.set_xticks([])
        _ax.set_xlabel(None)
    for _ax in _axs[1:].flatten():
        _ax.set_title(None)
    for _ax in _axs.flatten():
        _ax.axhline(0, ls='--', c='k', lw=0.8)
    for _ax in _axs.flatten():
        _ax.set_ylim([-5, 5])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### over time
    """)
    return


@app.cell
def _(plt, src):
    def plot_hovs(comp, varname, amp0, amp1, lag=1, fn=src.utils.get_nino3, delta=False):
        """plot data on hovmoller"""

        ## get data for plot
        plot_data = comp[varname]

        ## make sure latitude is a dimension
        if "latitude" not in plot_data.dims:
            plot_data = plot_data.expand_dims("latitude")

        ## get data for plots
        x0 = plot_data.sel(lag=lag, latitude=slice(-5, 5)).mean("latitude")
        x0 = x0.transpose("year", ...)
        x1 = fn(plot_data).transpose("year", ...)

        ## get delta if necessary
        if delta:
            x0 = x0 - x0.isel(year=0)
            x1 = x1 - x1.isel(year=0)

        fig, axs = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")

        ## longitude on x-axis
        axs[0].contourf(
            x0.longitude,
            x0.year,
            x0,
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(amp0, amp0 / 10),
            extend="both",
        )

        ## lag on x-axis
        axs[1].contourf(
            plot_data.lag,
            plot_data.year,
            x1,
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(amp1, amp1 / 10),
            extend="both",
        )

        axs[0].set_xlim([140, 280])
        axs[0].axvline(210, ls="--", c="k", lw=0.8)
        axs[0].axvline(270, ls="--", c="k", lw=0.8)
        axs[1].axvline(0, ls="--", c="k", lw=0.8)
        axs[0].set_xticks([160, 210, 270])
        axs[0].set_yticks([1870, 1980, 2090])
        axs[1].set_xticks([-3, 0, -3])
        axs[1].set_xlim([-6, 6])
        axs[1].set_yticks([])

        return fig, axs

    return (plot_hovs,)


@app.cell
def _(comps_cold_1, plot_hovs, plt, src):
    _LAG = 0
    COMP_6 = comps_cold_1 / comps_cold_1['T_3']
    _FN = src.utils.get_nino34
    kwargs_7 = dict(amp0=0.2, amp1=0.2, lag=_LAG, fn=_FN, comp=COMP_6, delta=True)
    for _VARNAME in ['ddt_T_ml', 'ADV_ml', 'THF', 'EKM', 'ZAF_ml', 'DD_ml', 'Q']:
        print(f'\n\n{_VARNAME}')
        _fig, _axs = plot_hovs(varname=_VARNAME, **kwargs_7)
        plt.show()
    return


@app.cell
def _(comps_warm_1, plt):
    _sel = lambda x: x.sel(longitude=slice(190, 240)).mean('longitude')
    plt.plot(comps_warm_1.year, _sel((comps_warm_1 / comps_warm_1['T_3']).sel(lag=1)['THF']))
    plt.axvline(2030)
    return


@app.cell
def _(comps_warm_1, plt, src):
    plt.plot(comps_warm_1.year, src.utils.get_nino34((comps_warm_1 / comps_warm_1['T_3']).sel(lag=0)['Q']))
    plt.axvline(2030)
    return


@app.cell
def _(comps_cold_1, get_nino3_da, plt, src):
    _sel = lambda x: x.sel(lag=slice(-4, 0)).mean('lag')
    COMP_7 = comps_cold_1
    prep0 = lambda v: get_nino3_da(COMP_7[v]) / COMP_7['T_3']
    prep1 = lambda x: _sel(x - x.isel(year=0))
    _prep = lambda v: prep1(prep0(v))
    _fig, _axs = plt.subplots(1, 2, figsize=(7, 3))
    for f in ['ddt_T', 'ADV']:
        _axs[0].plot(COMP_7.year, _prep(f'{f}_ml'), label=f)
    for f in ['DD_ml', 'Q']:
        _axs[1].plot(COMP_7.year, -_prep(f'{f}'), label=f)
    _axs[0].legend(prop=dict(size=8))
    _axs[1].legend(prop=dict(size=8))
    for _ax in _axs:
        _ax.axhline(0, ls='--', c='k', lw=0.8)
    src.utils.set_lims(_axs)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot composites of $p$ and $Q$ for El Niño and La Niña
    """)
    return


@app.cell
def _(comps_cold_1, plot_hovs, plt, src):
    _LAG = 3
    COMP_8 = comps_cold_1 / comps_cold_1['T_34']
    _lon_max = COMP_8.longitude.isel(longitude=COMP_8['pr'].sel(latitude=slice(-5, 5), lag=_LAG).mean('latitude').argmax('longitude'))
    _FN = src.utils.get_nino34
    _VARNAME = 'Q'
    kwargs_8 = dict(amp0=0.3, amp1=0.3, lag=_LAG, fn=_FN, comp=COMP_8, delta=False)
    _fig, _axs = plot_hovs(varname=_VARNAME, **kwargs_8)
    _axs[0].plot(_lon_max, COMP_8.year, c='w', lw=1)
    plt.gca()
    return


@app.cell
def _(comps_warm_1, plot_hovs, plt, src):
    _LAG = 0
    COMP_9 = comps_warm_1 / comps_warm_1['sst']
    _lon_max = COMP_9.longitude.isel(longitude=COMP_9['pr'].sel(latitude=slice(-5, 5), lag=_LAG).mean('latitude').argmax('longitude'))
    _FN = src.utils.get_nino34
    _VARNAME = 'pr'
    kwargs_9 = dict(amp0=5e-05, amp1=5e-05, lag=_LAG, fn=_FN, comp=COMP_9, delta=False)
    _fig, _axs = plot_hovs(varname=_VARNAME, **kwargs_9)
    _axs[0].plot(_lon_max, COMP_9.year, c='w', lw=1)
    _axs[0].set_xlim([170, None])
    _axs[1].set_xlim([None, 1.5])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    formatting function for plots
    """)
    return


@app.function
def format_ax(ax):

    ## format func
    ax.set_xlim([140, 280])
    ax.axvline(210, ls="--", c="k", lw=0.8)
    ax.axvline(270, ls="--", c="k", lw=0.8)
    ax.set_xticks([150, 210, 270])
    ax.set_yticks([1880, 2010, 2080])
    ax.set_xlabel(r"Longitude")

    return ax


@app.cell
def _(comps_cold_1, comps_warm_1):
    ## specify composite type
    COMP_WARM = comps_warm_1 / comps_warm_1['T_34']
    COMP_COLD = comps_cold_1 / comps_cold_1['T_34']
    ## reduce func
    reduce = lambda x: x.sel(latitude=slice(-5, 5)).mean('latitude')
    return (reduce,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### precip
    """)
    return


@app.cell
def _(comps_cold_1, comps_warm_1, plt, reduce, src):
    COMP_WARM_1 = comps_warm_1 / comps_warm_1['T_34']
    COMP_COLD_1 = comps_cold_1 / comps_cold_1['T_34']
    _fig, _axs = plt.subplots(2, 2, figsize=(6, 4), layout='constrained')
    for _lag_idx, _LAG in enumerate([0, 2]):
        for _wc_idx, COMP_10 in enumerate([COMP_WARM_1, COMP_COLD_1]):
            _cp = _axs[_lag_idx, _wc_idx].contourf(COMP_10.longitude, COMP_10.year, 86000.0 * reduce(COMP_10['pr'].sel(lag=_LAG)), levels=src.utils.make_cb_range(6, 0.6), cmap='cmo.balance_r', extend='both')
            _axs[_lag_idx, _wc_idx].contour(COMP_10.longitude, COMP_10.year, reduce(COMP_10['Q'].sel(lag=_LAG)), levels=src.utils.make_cb_range(1, 0.1), colors='white', linewidths=1, alpha=0.5)
    for _ax in _axs.flatten():
        format_ax(_ax)
    for _ax in _axs[:, -1]:
        _ax.set_yticks([])
        _ax.yaxis.set_label_position('right')
    _axs[0, 1].set_ylabel('Dec')
    _axs[1, 1].set_ylabel('Feb')
    for _ax in _axs[0, :]:
        _ax.set_xticks([])
        _ax.set_xlabel(None)
    _axs[0, 0].set_title('El Niño')
    _axs[0, 1].set_title('La Niña')
    _fig.colorbar(_cp, ax=_axs[:, -1], ticks=[-6, 0, 6], label='$\\left(mm~day^{-1}\\right) ~K^{-1}$')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SST
    """)
    return


@app.cell
def _(Th, src):
    median = src.utils.get_windowed(Th, stride=120).median(["member", "time"])["T_34"]
    median_dec = src.utils.sel_month(src.utils.get_windowed(Th, stride=120), 12).median(
        ["member", "time"]
    )["T_34"]
    return


@app.cell
def _(comps_cold_1, comps_warm_1, plt, reduce, save, src):
    COMP_WARM_2 = comps_warm_1
    COMP_COLD_2 = comps_cold_1
    _fig, _axs = plt.subplots(3, 2, figsize=(5, 6), layout='constrained')
    for _lag_idx, _LAG in enumerate([4, 2, 0]):
        for _wc_idx, COMP_11 in enumerate([COMP_WARM_2, COMP_COLD_2]):
            _cp = _axs[_lag_idx, _wc_idx].contourf(COMP_11.longitude, COMP_11.year, reduce(COMP_11['sst'].sel(lag=_LAG)), levels=src.utils.make_cb_range(4, 0.4), cmap='cmo.balance', extend='both')
            _axs[_lag_idx, _wc_idx].plot(COMP_11['eli_total'].sel(lag=_LAG), COMP_11.year, c='w', ls='--')
    for _ax in _axs.flatten():
        format_ax(_ax)
    for _ax in _axs[:, -1]:
        _ax.set_yticks([])
    _axs[0, 0].set_ylabel('Apr(+1)')
    _axs[1, 0].set_ylabel('Feb(+1)')
    _axs[2, 0].set_ylabel('Dec(0)')
    for _ax in _axs[:-1, :].flatten():
        _ax.set_xticks([])
        _ax.set_xlabel(None)
    _axs[0, 0].set_title('El Niño')
    _axs[0, 1].set_title('La Niña')
    _fig.colorbar(_cp, ax=_axs[-1, :], ticks=[-4, 0, 4], label='K', orientation='horizontal', fraction=0.1)
    for _ax in _axs.flatten():
        _ax.axhline(2020, c='k', lw=0.8, ls='--')
    save(_fig, fname='sst_hov_over-time', dpi=300)
    plt.gca()
    return COMP_COLD_2, COMP_WARM_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Zonal
    """)
    return


@app.cell
def _(COMP_COLD_2, COMP_WARM_2, plt, src):
    sel_lag = lambda x: x.sel(lag=slice(-5, 3)).mean('lag')
    _fig, _axs = plt.subplots(1, 2, figsize=(6, 2.75), layout='constrained')
    for _ax, COMP_12 in zip(_axs, [COMP_WARM_2, COMP_COLD_2]):
        _COMP_ = COMP_12 / COMP_12['T_34']
        _cp = _ax.contourf(COMP_12.longitude, COMP_12.year, sel_lag(_COMP_['zonal']), levels=src.utils.make_cb_range(0.4, 0.04), cmap='cmo.balance', extend='both')
    for _ax in _axs:
        format_ax(_ax)
    _axs[1].set_yticks([])
    _axs[0].set_ylabel('Year')
    _axs[0].set_title('El Niño')
    _axs[1].set_title('La Niña')
    _fig.colorbar(_cp, ticks=[-5, 0, 5], label='month$^{-1}$')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Clean up fig.
    """)
    return


@app.cell
def _():
    def format_ax_1(ax):
        """format single ax"""
        ax.set_xlim([140, 280])
        ax_kwargs = dict(ls='--', c='w', lw=0.8)  ## format func
        ax.axvline(180, **ax_kwargs)
        ax.axvline(230, **ax_kwargs)
        ax.axhline(2010, **ax_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def format_axs(axs):
        """format all axs"""
        for ax in axs.flatten():
            format_ax_1(ax)
        axs[0, 0].set_title('El Niño')
        axs[0, 1].set_title('La Niña')
        for ax in axs[-1]:
            ax.set_xticks([145, 190, 240, 275])  ## format/labeling
            ax.set_xlabel('Longitude')
        for ax in axs[:, 0]:
            ax.set_yticks([1880, 2010, 2080])
        return  ## titles  ## xlabels  ## ylabels  # ax.set_ylabel("Year")

    return format_ax_1, format_axs


@app.cell
def _(comps_cold_1, comps_warm_1, format_axs, merimean, np, plt, src):
    sel_lag_1 = lambda x: x.sel(lag=slice(-2, 0)).mean('lag')
    sel_lag_diff = lambda x: sel_lag_1(x) - sel_lag_1(x).isel(year=0)
    for _sel, _scale in zip([sel_lag_1, sel_lag_diff], [1, 0.5]):
        _fig, _axs = plt.subplots(5, 2, figsize=(7, 10), layout='constrained')
        _cps = np.empty(shape=_axs.shape, dtype=object)
        _zz = (comps_warm_1.longitude, comps_warm_1.year)
        _plot_kwargs = dict(cmap='cmo.balance', extend='both')
        for _j, COMP_13 in enumerate([comps_warm_1, comps_cold_1]):
            _COMP_ = COMP_13 / COMP_13['T_34']
            _cps[0, _j] = _axs[0, _j].contourf(*_zz, merimean(_sel(COMP_13['sst'])), levels=src.utils.make_cb_range(4, 0.4) * _scale, **_plot_kwargs)
            _cps[-2, _j] = _axs[-2, _j].contourf(*_zz, merimean(_sel(_COMP_['pr'])) * 86000.0, levels=src.utils.make_cb_range(4, 0.4) * _scale, **dict(_plot_kwargs, cmap='cmo.balance_r'))
            _cps[-1, _j] = _axs[-1, _j].contourf(*_zz, merimean(_sel(_COMP_['taux'])) * 1000.0, levels=src.utils.make_cb_range(20, 2) * _scale, **dict(_plot_kwargs, cmap='cmo.balance'))
            for _i, _v in enumerate(['ddt_sst', 'Q'], start=1):
                _cps[_i, _j] = _axs[_i, _j].contourf(*_zz, 12 * merimean(_sel(_COMP_[_v])), levels=src.utils.make_cb_range(3, 0.3) * _scale, **_plot_kwargs)
        format_axs(_axs)
        units = ['K', 'yr$^{-1}$', 'yr$^{-1}$', 'mm day$^{-1}$ K$^{-1}$', 'mPa K$^{-1}$']
        amps = np.array([4, 3, 3, 10, 20]) * _scale
        _labels = ['SST', '$\\partial/\\partial t$(SST)$~/~T_{34}$', '$Q~/~T_{34}$', 'Precip. / $T_{34}$', '$\\tau_x$ / $T_{34}$']
        for _j, (u, a, _l) in enumerate(zip(units, amps, _labels)):
            kwargs_10 = dict(x=-0.5, y=0.4, size=14, horizontalalignment='center')
            _axs[_j, 0].text(s=_l, transform=_axs[_j, 0].transAxes, **kwargs_10)
            _fig.colorbar(_cps[_j, 1], ax=_axs[_j, 1], label=u, ticks=[-a, 0, a])
        plt.show()
    return sel_lag_1, sel_lag_diff


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Heat flux
    """)
    return


@app.cell
def _(
    comps_cold_1,
    comps_warm_1,
    format_axs,
    merimean,
    np,
    plt,
    sel_lag_1,
    sel_lag_diff,
    src,
):
    for _sel, _scale in zip([sel_lag_1, sel_lag_diff], [1, 0.5]):
        _fig, _axs = plt.subplots(4, 2, figsize=(7, 8), layout='constrained')
        _cps = np.empty(shape=_axs.shape, dtype=object)
        _zz = (comps_warm_1.longitude, comps_warm_1.year)
        _plot_kwargs = dict(cmap='cmo.balance', extend='both')
        for _j, COMP_14 in enumerate([comps_warm_1, comps_cold_1]):
            _COMP_ = COMP_14 / COMP_14['T_34']
            for _i, _v in enumerate(['Q_sw', 'Q_cloud', 'Q_lhf', 'Q_lw']):
                _cps[_i, _j] = _axs[_i, _j].contourf(*_zz, 12 * merimean(_sel(_COMP_[_v])), levels=src.utils.make_cb_range(3, 0.3) * _scale, **_plot_kwargs)
        format_axs(_axs)
        _labels = ['SW$~/~T_{34}$', 'SW (cloud) $~/~T_{34}$', 'LHF $~/~ T_{34}$', 'LW $~/~ T_{34}$']
        for _j, _l in enumerate(_labels):
            _fig.colorbar(_cps[_j, 1], label='yr$^{-1}$', ticks=[-3 * _scale, 0, 3 * _scale])
            kwargs_11 = dict(x=-0.5, y=0.4, size=14, horizontalalignment='center')
            _axs[_j, 0].text(s=_l, transform=_axs[_j, 0].transAxes, **kwargs_11)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Precip clim.
    """)
    return


@app.cell
def _(clim, comps_cold_1, comps_warm_1, format_ax_1, np, plt, xr):
    sel_lag_2 = lambda x: x.sel(lag=slice(-2, 0)).mean('lag')
    sel_pr = lambda x: x.sel(latitude=slice(-5, 5)).mean('latitude')
    _fig, _axs = plt.subplots(1, 3, figsize=(7, 2), layout='constrained')
    for _ax, comp_ in zip(_axs, [comps_warm_1, xr.zeros_like(comps_warm_1), comps_cold_1]):
        _cp = _ax.contourf(comp_.longitude, comp_.year, 86000.0 * sel_lag_2(sel_pr((clim + comp_)['pr'])), levels=np.arange(0, 12, 1), cmap='cmo.rain', extend='max')
        format_ax_1(_ax)
    _axs[0].set_ylabel('Year')
    _fig.colorbar(_cp, ticks=[0, 11], label='mm day$^{-1}$')
    plt.gca()
    return (sel_pr,)


@app.cell
def _(comps_cold_1, comps_warm_1, plt, sel_pr, src):
    sel_lag_3 = lambda x: x.sel(lag=slice(-2, 0)).mean('lag')
    _delta = lambda x: x - x.isel(year=0)
    _sel_helper = lambda x, lons: _delta(12 * sel_pr(sel_lag_3(x['Q'].sel(longitude=slice(*lons)).mean('longitude'))))
    _fig, _axs = plt.subplots(1, 3, figsize=(8, 2.5))
    for _ax, _lons in zip(_axs, [(180, 230), (210, 270), (230, 280)]):
        _ax.plot(comps_cold_1.year, _sel_helper(comps_cold_1, _lons))
        _ax.plot(comps_cold_1.year, -_sel_helper(comps_warm_1, _lons))
    src.utils.set_lims(_axs)
    for _ax in _axs[1:]:
        _ax.set_yticks([])
    plt.gca()
    return (sel_lag_3,)


@app.cell
def _(comps_cold_1, comps_warm_1, plt, sel_lag_3, sel_pr, src):
    _delta = lambda x: x - x.isel(year=0)
    _sel_helper = lambda x, lons: _delta(sel_pr(sel_lag_3(x['Q_lw'].sel(longitude=slice(*lons)).mean('longitude'))))
    _fig, _axs = plt.subplots(1, 3, figsize=(8, 2.5))
    for _ax, _lons in zip(_axs, [(180, 230), (210, 270), (230, 280)]):
        _ax.plot(comps_cold_1.year, _sel_helper(comps_cold_1, _lons))
        _ax.plot(comps_cold_1.year, -_sel_helper(comps_warm_1, _lons))
    src.utils.set_lims(_axs)
    for _ax in _axs[1:]:
        _ax.set_yticks([])
    plt.gca()
    return


@app.cell
def _(clim, comps_cold_1, comps_warm_1, plt, sel_lag_3, sel_pr):
    plt.plot(86000.0 * sel_pr(sel_lag_3((clim + comps_cold_1)['pr'])).sel(longitude=slice(180, 230)).mean('longitude'))
    plt.plot(86000.0 * sel_pr(sel_lag_3(comps_warm_1['pr'])).sel(longitude=slice(180, 230)).mean('longitude'))
    return


if __name__ == "__main__":
    app.run()
