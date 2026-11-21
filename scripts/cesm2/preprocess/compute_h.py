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
    # Compute $h$
    Compute thermocline depth from subsurface temperature profiles
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
    import pandas as pd

    # Import custom modules
    import src.utils

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return SAVE_FP, np, pathlib, plt, src, tqdm, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## cluster
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: need lots of memory to load data: ~16 GB / worker
    """)
    return


@app.cell
def _():
    from dask.distributed import LocalCluster, Client

    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)
    client
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(src, xr):
    ## load spatial data
    # forced, anom = src.utils.load_consolidated()
    _forced, _anom = src.utils.load_consolidated_05()
    T_scores = _forced['T'] + _anom['T']
    T = xr.merge([T_scores, _forced['T_comp']])
    ## chunk the data (for dask
    T = T.chunk({'member': 1, 'time': 120})
    return (T,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### threshold-version
    """)
    return


@app.cell
def _(SAVE_FP, T, pathlib, src, xr):
    ## set threshold for thermocline depth
    THRESH = 0.04
    _save_fp = pathlib.Path(SAVE_FP, 'h_ests', f'h_int_{int(THRESH * 1000.0)}.nc')
    ## get save filepath
    if _save_fp.is_file():
        print('File exists!')
        H = xr.open_dataarray(_save_fp)
    else:
        print('Computing')
        T_vals = src.utils.reconstruct_wrapper(T)
        H = src.utils.get_H_int(T_vals['T'], thresh=THRESH)
        H.to_netcdf(_save_fp)  ## save
    return (H,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Contour version
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Func to compute
    """)
    return


@app.cell
def _(SAVE_FP, np, pathlib, xr):
    def find_level(T_recon, level=20):
        """function to find depth of thermocline"""

        ## find index of closest vertical level
        level_idx = np.abs(T_recon - level).fillna(1e20).argmin("z_t")

        ## get corresponding depth
        level_est = T_recon.z_t.isel(z_t=level_idx)

        ## add back NaNs
        level_est = level_est.where(~np.isnan(T_recon.isel(z_t=0)), other=np.nan)

        return level_est


    def load_level(T_recon, lev):
        """wrapper function: try loading thermocline depth; otherwise compute and save"""

        ## get save filepath
        save_fp = pathlib.Path(SAVE_FP, "h_ests", f"z_{lev}.nc")

        if save_fp.is_file():

            print("File exists!")
            H = xr.open_dataarray(save_fp)

        else:

            ## compute
            H = find_level(T_recon, level=lev)

            ## save
            H.to_netcdf(save_fp)

        return H

    return find_level, load_level


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Test func
    """)
    return


@app.cell
def _(T, find_level, src):
    ## get data
    T_recon = src.utils.reconstruct_wrapper(T.isel(time=300, member=30))["T"].compute()

    ## find thermocline
    z20_est = find_level(T_recon)
    return T_recon, z20_est


@app.cell
def _(T_recon, np, plt, z20_est):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.contourf(T_recon.longitude, T_recon.z_t, T_recon, cmap='cmo.thermal', levels=np.arange(10, 34, 2), extend='both')
    _ax.contour(T_recon.longitude, T_recon.z_t, T_recon, colors='w', levels=[20])
    _ax.plot(z20_est.longitude, z20_est, ls='--', c='magenta')
    _ax.set_ylim([200, 5])
    _ax.set_xlim([120, 280])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load to memory (this part takes the longest)
    """)
    return


@app.cell
def _(T, src):
    T_recon_1 = src.utils.reconstruct_wrapper(T)['T'].compute()
    return (T_recon_1,)


@app.cell
def _(T_recon_1, load_level, np, tqdm):
    for lev in tqdm.tqdm(np.arange(20, 25)):
    # z20_est = find_level(T_recon, level=20)
    # z21_est = find_level(T_recon, level=21)
    # z22_est = find_level(T_recon, level=22)
    # z23_est = find_level(T_recon, level=23)
        load_level(T_recon_1, lev=lev)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot results to make sure they look ok
    """)
    return


@app.cell
def _(SAVE_FP, pathlib, xr):
    z20 = xr.open_dataarray(pathlib.Path(SAVE_FP, "h_ests", f"z_20.nc"))
    z22 = xr.open_dataarray(pathlib.Path(SAVE_FP, "h_ests", f"z_22.nc"))
    Z = xr.merge([z20.rename("z20"), z22.rename("z22")])
    return (Z,)


@app.cell
def _(T_recon_1, Z, np, plt):
    _sel = lambda x: x.isel(time=40, member=60)
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.contourf(T_recon_1.longitude, T_recon_1.z_t, _sel(T_recon_1), cmap='cmo.thermal', levels=np.arange(10, 34, 2), extend='both')
    _ax.contour(T_recon_1.longitude, T_recon_1.z_t, _sel(T_recon_1), colors='w', levels=[20])
    _ax.plot(Z.longitude, _sel(Z['z20']), ls='--', c='magenta')
    _ax.plot(Z.longitude, _sel(Z['z22']), ls='--', c='magenta')
    _ax.set_ylim([250, 5])
    _ax.set_xlim([120, 280])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### max-grad version
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### function to find max gradient
    """)
    return


@app.cell
def _(np):
    def find_maxgrad(T_recon):
        """function to find depth of thermocline"""

        ## find index of max gradient
        idx = T_recon.differentiate("z_t").fillna(1e20).argmin("z_t")

        ## get thermocline depth
        h = T_recon.z_t.isel(z_t=idx)

        ## fill in NaN values
        h = h.where(~np.isnan(T_recon.isel(z_t=0)), other=np.nan)

        return h

    return (find_maxgrad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute
    """)
    return


@app.cell
def _(SAVE_FP, T, find_maxgrad, pathlib, src, xr):
    _save_fp = pathlib.Path(SAVE_FP, 'h_ests', f'h_max-grad_05.nc')
    if _save_fp.is_file():
        print('File exists!')
        h = xr.open_dataarray(_save_fp)
    else:
        print('Computing')
        T_recon_2 = src.utils.reconstruct_wrapper(T)['T'].compute()
        h = find_maxgrad(T_recon_2)
        h.to_netcdf(_save_fp)
    return (h,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Test it makes sense
    """)
    return


@app.cell
def _(T, h, np, plt, src):
    idx = dict(time=100, member=6)
    T_ = src.utils.reconstruct_wrapper(T.isel(idx))['T'].compute()
    ## get data
    h_ = h.isel(idx)
    _fig, _ax = plt.subplots(figsize=(4, 3))
    ## get h
    _ax.contourf(T_.longitude, T_.z_t, T_, cmap='cmo.thermal', levels=np.arange(10, 34, 2), extend='both')
    _ax.contour(T_.longitude, T_.z_t, T_, colors='w', levels=[20])
    _ax.plot(h_.longitude, h_, c='k')
    _ax.set_ylim([250, 5])
    _ax.set_xlim([120, 270])
    ## plot 20 deg isotherm
    ## plot max grad thermocline
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### threshold max-grad version
    """)
    return


@app.cell
def _():
    # def find_maxgrad(T_recon):
    #     """function to find depth of thermocline"""

    #     ## find index of max gradient
    #     idx = T_recon.differentiate("z_t").fillna(1e20).argmin("z_t")

    #     ## get thermocline depth
    #     h = T_recon.z_t.isel(z_t=idx)

    #     ## fill in NaN values
    #     h = h.where(~np.isnan(T_recon.isel(z_t=0)), other=np.nan)

    #     return h
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load $Z_{20}$ (validation data)
    """)
    return


@app.cell
def _(src, xr):
    _forced, _anom = src.utils.load_consolidated()
    z20_scores = _forced['z20'] + _anom['z20']
    z20_1 = xr.merge([z20_scores, _forced['z20_comp']])
    z20_1 = z20_1.chunk({'member': 1, 'time': 180})
    z20_eq = src.utils.reconstruct_wrapper(z20_1, fn=lambda x: x.sel(latitude=slice(-1.5, 1.5)).mean('latitude'))
    z20_eq.load()
    return z20_1, z20_eq


@app.cell
def _(H, z20_eq):
    _sel = lambda x: x.isel(time=slice(None, 360)).mean(['member', 'time'])
    hbar = _sel(H)
    hbar1 = H.isel(time=slice(-360, None)).mean(['member', 'time'])
    z20_bar = _sel(z20_eq)
    return hbar, z20_bar


@app.cell
def _(hbar, plt, z20_bar):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(hbar.longitude, hbar)
    _ax.plot(z20_bar.longitude, z20_bar['z20'])
    # plt.plot(hbar.longitude, hbar0)
    # plt.plot(hbar.longitude, hbar1)
    _ax.set_ylim([200, 40])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spatial plot
    """)
    return


@app.cell
def _(src, z20_1):
    z_ = z20_1.isel(time=slice(None, 360)).mean(['time', 'member']).compute()
    z = src.utils.reconstruct_wrapper(z_)
    return z, z_


@app.cell
def _(src, z_):
    coslat_weights = src.utils.get_coslat_weights(z_.z20_comp)

    ## evaluate function on spatial components
    fn_eval = z_.z20_comp * 1 / coslat_weights
    return


@app.cell
def _(plt, src, z):
    import cartopy.crs as ccrs
    _fig = plt.figure(figsize=(5, 2.5), layout='constrained')
    format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=20)
    axs = src.utils.subplots_with_proj(_fig, nrows=1, ncols=1, format_func=format_func)
    axs[0, 0].contourf(z.longitude, z.latitude, z['z20'], cmap='cmo.thermal', transform=ccrs.PlateCarree())
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
