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
    import xarray as xr
    import pathlib
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import xesmf
    import time
    import src.utils
    import copy

    ## specify filepath for data
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})

    ## bump up DPI for presentation
    mpl.rcParams["figure.dpi"] = 100
    return DATA_FP, np, os, pathlib, plt, src, tqdm, xesmf, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shared functions
    """)
    return


@app.function
def trim(
    data, lon_range=[130, 290], lat_range=[-5, 5], lon_name="TLONG", lat_name="TLAT"
):
    """select part of data in given longitude/latitude range"""

    ## helper function to check if 'x' is in 'x_range'
    isin_range = lambda x, x_range: (x_range[0] <= x) & (x <= x_range[1])

    ## get mask for data in given lon/lat range
    in_lon_range = isin_range(data[lon_name], lon_range)
    in_lat_range = isin_range(data[lat_name], lat_range)
    in_lonlat_range = in_lon_range & in_lat_range

    ## load to memory
    in_lonlat_range.load()

    ## Retain all points with at least one valid grid cell
    x_idx = in_lonlat_range.any("nlat")
    y_idx = in_lonlat_range.any("nlon")

    ## select given points
    return data.isel(nlon=x_idx, nlat=y_idx)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Subsurface ocean data
    """)
    return


@app.cell
def _(np, pathlib, tqdm, xr):
    def get_ensemble_ids():
        """get files for given variable name"""
        cesm2_fp = pathlib.Path('/glade/campaign/collections/rda/data/d651056/CESM2-LE/ocn/proc/tseries/month_1')
        data_fp = cesm2_fp / 'WVEL'  ## path to cesm2 lens data
        ensemble_ids = []
        for f in data_fp.glob('*.nc'):
            ensemble_ids.append(str(f)[-53:-28])
        ensemble_ids = sorted(list(set(ensemble_ids)))
        return ensemble_ids  ## path to FSNS (arbitrary, just want the ids)

    def _load_var(varname, ensemble_id):
        """Load variable for given ensemble ID"""  ## get list of ensemble ids
        cesm2_fp = pathlib.Path('/glade/campaign/collections/rda/data/d651056/CESM2-LE/ocn/proc/tseries/month_1')
        data_fp = cesm2_fp / varname
        if varname in ['WVEL', 'WTT']:
            z_coord = 'z_w_top'
            prep_fn = lambda x: x.where(np.isnan(x) | (x > 0), other=0.0)  ## get unique values and sort
        else:
            z_coord = 'z_t'
            prep_fn = lambda x: x
        data = xr.open_mfdataset(data_fp.glob(f'*{ensemble_id}*.nc'), decode_timedelta=True, chunks={'time': 18, z_coord: 60, 'nlat': 384, 'nlon': 320}, parallel=True)[varname]
        data = trim(data, lat_range=[-5, 5], lon_range=[120, 285])
        data = data.isel({z_coord: slice(None, 27), 'nlon': slice(None, None, 2)})
        data = prep_fn(data)
        data = data.assign_coords({'lon': data['TLONG'].mean('nlat')})
        data = data.mean('nlat')  ## get path to data
        return data

    def preprocess_ensemble(varname, temp_dir):
        """compute net heat flux for full ensemble. Save to temp directory"""
        ensemble_ids = get_ensemble_ids()  ## path to data
        for i in tqdm.tqdm(ensemble_ids):
            save_fp = pathlib.Path(temp_dir, f'{varname}_{i}.nc')
            if save_fp.is_file():  ## get z-coordinate name
                pass
            else:
                try:
                    data = _load_var(varname=varname, ensemble_id=i)
                    data.to_netcdf(save_fp)
                except:
                    print(f"warning: {i} didn't work...")
                    pass
        return  ## open data for ensemble member  ## trim to eq Pac  ## subset longitude and get top 300 m  ## zero out downwelling before meridional average  ## average over latitudes  ## get ensemble ids  ## loop through members  # idx 60 throwing "NetCDF: HDF error" for WVEL  # for i in tqdm.tqdm(ensemble_ids[61:]):  ## save filepath

    return get_ensemble_ids, preprocess_ensemble


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Initialize cluster
    """)
    return


@app.cell
def _():
    from dask.distributed import LocalCluster, Client
    _cluster = LocalCluster(n_workers=16)
    _client = Client(_cluster)
    _client
    return Client, LocalCluster


@app.cell
def _(DATA_FP, pathlib, preprocess_ensemble):
    # preprocess_ensemble(
    # varname="TEMP", temp_dir=pathlib.Path(DATA_FP, "cesm", "temp_05")
    # )

    # preprocess_ensemble(
    #     varname="WVEL", temp_dir=pathlib.Path(DATA_FP, "cesm", "wvel_05")
    # )

    # preprocess_ensemble(
    #     varname="ADV_3D_TEMP", temp_dir=pathlib.Path(DATA_FP, "cesm", "adv_05")
    # )

    # preprocess_ensemble(
    #     varname="TEND_TEMP", temp_dir=pathlib.Path(DATA_FP, "cesm", "ddt_T_05")
    # )

    preprocess_ensemble(varname="UVEL", temp_dir=pathlib.Path(DATA_FP, "cesm", "uvel_05"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### can't open one of the files; not sure why
    """)
    return


@app.cell
def _():
    # ids = get_ensemble_ids()
    # temp_dir = "wvel_temp"
    # save_fp = pathlib.Path(DATA_FP, "cesm", "wvel_temp", f"WVEL_{ids[60]}.nc")

    # ## load data
    # d = load_var("WVEL", ids[60])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test off-by-one and gradient
    """)
    return


@app.cell
def _(pathlib, xr):
    def load_var_test(varname, ensemble_id):
        """Load variable for given ensemble ID"""

        ## get path to data
        cesm2_fp = pathlib.Path(
            "/glade/campaign/collections/rda/data/d651056/CESM2-LE/ocn/proc/tseries/month_1"
        )

        ## path to data
        data_fp = cesm2_fp / varname

        ## open data for ensemble member
        data = xr.open_mfdataset(
            data_fp.glob(f"*{ensemble_id}*.nc"),
            decode_timedelta=True,
            # chunks={"time": 360, "z_t": 60, "nlat": 384, "nlon": 320},
            parallel=True,
        )[varname]

        ## trim to eq Pac
        # data = trim(data, lat_range=[-5,5], lon_range=[210, 270])
        data = trim(data, lat_range=[-1.5, 1.5], lon_range=[210, 270])

        return data.isel(z_t=0).mean(["nlat", "nlon"])

    return (load_var_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test off-by-one error with SST
    """)
    return


@app.cell
def _(get_ensemble_ids, load_var_test):
    T = load_var_test("TEMP", get_ensemble_ids()[0])
    T.load();
    return (T,)


@app.cell
def _(T, os, pathlib, xr):
    T_ = T.isel(time=slice(None, -1))

    Th = xr.open_dataset(pathlib.Path(os.environ["DATA_FP"], "cesm", "Th.nc"))
    T2 = Th["T_3"].sel(member=0, time=T_.time).compute()

    Th = xr.open_dataset(pathlib.Path(os.environ["DATA_FP"], "cesm", "Th_anom.nc"))
    T3 = Th["T_3"].sel(member=0, time=T_.time).compute()
    return T2, T_


@app.cell
def _(T2, T_):
    T_prime = T_.groupby("time.month") - T_.groupby("time.month").mean()
    T2_prime = T2.groupby("time.month") - T2.groupby("time.month").mean()
    return T2_prime, T_prime


@app.cell
def _(T2, T2_prime, T_, T_prime):
    from scipy.stats import pearsonr as r

    print(r(T_.values, T2.values)[0])
    print(r(T_.values[1:], T2.values[:-1])[0])
    print(r(T_.values[:-1], T2.values[1:])[0])
    print()
    print(r(T_prime.values, T2_prime.values)[0])
    print(r(T_prime.values[1:], T2_prime.values[:-1])[0])
    print(r(T_prime.values[:-1], T2_prime.values[1:])[0])
    return


@app.cell
def _(T, T2, plt):
    _shift = lambda x: x.isel(time=slice(1, None))
    t = dict(time=slice('1860', '1865'))
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(T.sel(t))
    _ax.plot(T2.sel(t))
    _ax.plot(_shift(T.sel(t)), ls='--')
    # ax.plot(T3.sel(t))
    plt.gca()
    return


@app.cell
def _(get_ensemble_ids, load_var_test):
    U = load_var_test("UVEL", get_ensemble_ids()[0])
    U.load();
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check $\frac{d}{dt}$
    """)
    return


@app.cell
def _(get_ensemble_ids, load_var_test, src):
    print('Loading T')
    T_1 = load_var_test('TEMP', get_ensemble_ids()[0])
    T_1.load()
    print('Loading ddt(T)')
    ddt_T = load_var_test('TEND_TEMP', get_ensemble_ids()[0])
    ddt_T.load()
    sec_per_month = 86400 * 30
    ddt_T = ddt_T * sec_per_month
    ## convert to K/mo
    T_1 = src.utils.align_pop_times(T_1.to_dataset(), pop_vars=['TEMP'])['TEMP']
    ## fix times
    ddt_T = src.utils.align_pop_times(ddt_T.to_dataset(), pop_vars=['TEND_TEMP'])['TEND_TEMP']
    return T_1, ddt_T


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load EOF data
    """)
    return


@app.cell
def _(src, xr):
    ## load spatial data
    forced, _anom = src.utils.load_consolidated()
    T_eof = (_anom['T'] + forced['T']).isel(member=0)
    ## get total
    T_eof = xr.merge([T_eof, forced['T_comp']])
    _fun_1d = lambda x: x.sel(longitude=slice(210, 270)).isel(z_t=0).mean('longitude').squeeze(drop=True)
    ## reconstruct
    T_recon = src.utils.reconstruct_fn(scores=T_eof['T'], components=T_eof['T_comp'], fn=_fun_1d)
    return (T_recon,)


@app.cell
def _(T_1, T_recon, ddt_T, plt, xr):
    _order = 2
    if _order == 2:
        _dT = T_1.values[2:] - T_1.values[:-2]
        ddt_T_hat = 0.5 * xr.ones_like(T_1.isel(time=slice(1, -1))) * _dT
        dT_recon = T_recon.values[2:] - T_recon.values[:-2]
        ddt_T_recon = 0.5 * xr.ones_like(T_recon.isel(time=slice(1, -1))) * dT_recon
    else:
        _dT = T_1.values[1:] - T_1.values[:-1]
        ddt_T_hat = xr.ones_like(T_1.isel(time=slice(1, None))) * _dT
        dT_recon = T_recon.values[1:] - T_recon.values[:-1]
        ddt_T_recon = xr.ones_like(T_recon.isel(time=slice(1, None))) * dT_recon
    _plot_idx = dict(time=slice('1851', '1854'))
    _shift = lambda x: x.isel(dict(time=slice(1, None)))
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(ddt_T.sel(_plot_idx), ls='-')
    _ax.plot(ddt_T_hat.sel(_plot_idx), ls='-')
    _ax.plot(ddt_T_recon.sel(_plot_idx), ls='--')
    plt.gca()
    return (ddt_T_hat,)


@app.cell
def _(ddt_T, ddt_T_hat, xr):
    print(xr.corr(ddt_T, ddt_T_hat).values.item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Check with anomalies
    """)
    return


@app.cell
def _(src, xr):
    _order = 2
    _, _anom = src.utils.load_consolidated()
    _anom = _anom[['T', 'T_comp']]
    _fun_1d = lambda x: x.sel(longitude=slice(210, 270)).isel(z_t=0).mean('longitude').squeeze(drop=True)
    T_recon_1 = src.utils.reconstruct_wrapper(_anom, fn=_fun_1d)['T']
    if _order == 2:
        _dT = T_recon_1.values[:, 2:] - T_recon_1.values[:, :-2]
        ddt_T_hat_1 = 0.5 * xr.ones_like(T_recon_1.isel(time=slice(1, -1))) * _dT
    else:
        _dT = T_recon_1.values[:, 1:] - T_recon_1.values[:, :-1]
        ddt_T_hat_1 = xr.ones_like(T_recon_1.isel(time=slice(None, -1))) * _dT
    _, budg_anom = src.utils.load_budget_data()
    ddt_T_1 = _fun_1d(budg_anom['TEND_TEMP']).rename('ddt_T')
    return ddt_T_1, ddt_T_hat_1


@app.cell
def _(ddt_T_1, ddt_T_hat_1, plt):
    _plot_idx = dict(time=slice('1851', '1854'), member=5)
    _shift = lambda x: x.isel(dict(time=slice(1, None)))
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(ddt_T_1.sel(_plot_idx), ls='-')
    _ax.plot(ddt_T_hat_1.sel(_plot_idx), ls='-')
    plt.gca()
    return


@app.cell
def _(ddt_T_1, ddt_T_hat_1, xr):
    print(xr.corr(ddt_T_1, ddt_T_hat_1).values.item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Surface data ($u$, $v$)
    """)
    return


@app.cell
def _(pathlib, tqdm, xr):
    def get_ensemble_ids_1():
        """get files for given variable name"""
        cesm2_fp = pathlib.Path('/glade/campaign/collections/rda/data/d651056/CESM2-LE/ocn/proc/tseries/month_1')
        data_fp = cesm2_fp / 'UVEL'
        ensemble_ids = []
        for f in data_fp.glob('*.nc'):
            ensemble_ids.append(str(f)[-53:-28])
        ensemble_ids = sorted(list(set(ensemble_ids)))
        return ensemble_ids

    def _load_var(varname, ensemble_id):
        """Load variable for given ensemble ID"""
        cesm2_fp = pathlib.Path('/glade/campaign/collections/rda/data/d651056/CESM2-LE/ocn/proc/tseries/month_1')
        data_fp = cesm2_fp / varname
        data = xr.open_mfdataset(data_fp.glob(f'*{ensemble_id}*.nc'), decode_timedelta=True, chunks={'time': 120, 'nlat': 384, 'nlon': 320}, parallel=True)[varname]
        data = data.isel(z_t=0)
        data = trim(data, lat_range=[-15, 15], lon_range=[120, 300], lon_name='ULONG', lat_name='ULAT')
        return data

    def preprocess_ensemble_1(varname, temp_dir):
        """compute net heat flux for full ensemble. Save to temp directory"""
        ensemble_ids = get_ensemble_ids_1()
        for i in tqdm.tqdm(ensemble_ids):
            save_fp = pathlib.Path(temp_dir, f'{varname}_{i}.nc')
            if save_fp.is_file():
                pass
            else:
                data = _load_var(varname=varname, ensemble_id=i)
                data.to_netcdf(save_fp)
        return

    return (preprocess_ensemble_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Initialize cluster
    """)
    return


@app.cell
def _(Client, LocalCluster):
    _cluster = LocalCluster(n_workers=12)
    _client = Client(_cluster)
    _client
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute
    """)
    return


@app.cell
def _(DATA_FP, pathlib, preprocess_ensemble_1):
    preprocess_ensemble_1(varname='UVEL', temp_dir=pathlib.Path(DATA_FP, 'cesm', 'uvel_temp'))
    preprocess_ensemble_1(varname='VVEL', temp_dir=pathlib.Path(DATA_FP, 'cesm', 'vvel_temp'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3D ocean data
    """)
    return


@app.cell
def _(np, pathlib, tqdm, xesmf, xr):
    def get_ensemble_ids_2():
        """get files for given variable name"""
        cesm2_fp = pathlib.Path('/glade/campaign/collections/rda/data/d651056/CESM2-LE/ocn/proc/tseries/month_1')
        data_fp = cesm2_fp / 'WVEL'
        ensemble_ids = []
        for f in data_fp.glob('*.nc'):
            ensemble_ids.append(str(f)[-53:-28])
        ensemble_ids = sorted(list(set(ensemble_ids)))
        return ensemble_ids

    def load_grid(lon_range, lat_range):
        """Create mask from OISST data on cloud"""
        sst = xr.open_dataset('http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2/new/sst.oisst.mon.ltm.1991-2020.nc', decode_times=False)
        sst = sst['sst'].isel(time=0).drop_vars('time')
        lsm = sst.where(np.isnan(sst), other=1.0)
        lsm = lsm.sel(lon=slice(*lon_range), lat=slice(*lat_range))
        lsm['mask'] = ~np.isnan(lsm)
        return lsm

    def _load_var(varname, ensemble_id):
        """Load variable for given ensemble ID"""
        cesm2_fp = pathlib.Path('/glade/campaign/collections/rda/data/d651056/CESM2-LE/ocn/proc/tseries/month_1')
        data_fp = cesm2_fp / varname
        if varname == 'WVEL':
            z_coord = 'z_w_top'
        else:
            z_coord = 'z_t'
        data = xr.open_mfdataset(data_fp.glob(f'*{ensemble_id}*.nc'), decode_timedelta=True, chunks={'time': 12, z_coord: 60, 'nlat': 384, 'nlon': 320}, parallel=True)[varname]
        data = trim(data, lat_range=[-6.5, 6.5], lon_range=[130, 290])
        data = data.isel({z_coord: slice(None, 27), 'nlon': slice(None, None, 2)})
        data = data.rename({'TLONG': 'lon', 'TLAT': 'lat'})
        grid = load_grid(lon_range=[130, 290], lat_range=[-6, 6])
        grid = grid.isel(lon=slice(None, None, 2))
        regridder = xesmf.Regridder(data, grid, 'bilinear')
        data_regrid = regridder(data).drop_vars('mask')
        return data_regrid

    def preprocess_ensemble_2(varname, temp_dir):
        """compute net heat flux for full ensemble. Save to temp directory"""
        ensemble_ids = get_ensemble_ids_2()
        for i in tqdm.tqdm(ensemble_ids):
            save_fp = pathlib.Path(temp_dir, f'{varname}_{i}.nc')
            if save_fp.is_file():
                pass
            else:
                data = _load_var(varname=varname, ensemble_id=i)
                data.to_netcdf(save_fp)
        return

    return (preprocess_ensemble_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Initialize cluster
    """)
    return


@app.cell
def _(Client, LocalCluster):
    _cluster = LocalCluster(n_workers=16)
    _client = Client(_cluster)
    _client
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### compute
    """)
    return


@app.cell
def _(DATA_FP, pathlib, preprocess_ensemble_2):
    # preprocess_ensemble(
    #     varname="VVEL", temp_dir=pathlib.Path(DATA_FP, "cesm", "vvel_3d_temp")
    # )
    #     varname="TEMP", temp_dir=pathlib.Path(DATA_FP, "cesm", "temp_3d_temp")
    #     varname="UVEL", temp_dir=pathlib.Path(DATA_FP, "cesm", "uvel_3d_temp")
    preprocess_ensemble_2(varname='WVEL', temp_dir=pathlib.Path(DATA_FP, 'cesm', 'wvel_3d_temp'))
    return


if __name__ == "__main__":
    app.run()
