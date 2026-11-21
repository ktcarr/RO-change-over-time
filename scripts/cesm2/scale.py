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
    # Changes in thermal expansion coefficient
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
    import gsw

    # Import custom modules
    import src.utils

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return DATA_FP, gsw, np, os, pathlib, plt, src, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## funcs
    """)
    return


@app.function
def avg_upper_ocn(T, H=70, lon_range=[140, 280]):
    """average upper ocean above threshold"""
    T_ = T.sel(z_t=slice(None, H), longitude=slice(*lon_range))

    return T_.mean(["z_t", "longitude"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(src):
    ## load data
    _forced, _ = src.utils.load_consolidated()
    T = _forced[['T', 'T_comp']].sel(time=slice('1851', None))
    ## subset for temperature
    ## get windowed T
    T = src.utils.get_windowed(T, stride=120)
    return (T,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $\alpha$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Avg. temperature over near-surface layer
    """)
    return


@app.cell
def _(T, src):
    T_upper = src.utils.reconstruct_clim(T, fn=avg_upper_ocn)["T"]
    T_upper_mean = T_upper.mean("month")
    return T_upper, T_upper_mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute thermal expansion
    """)
    return


@app.cell
def _(T_upper, T_upper_mean, gsw, np, os, pathlib, src, xr):
    ## specify reference values
    SA = 25  # g/kg
    P = 35  # dbar
    H = 70  # m
    nyear = len(T_upper.year)
    ## reference values for salinity, pressure
    sa = SA * np.ones(nyear)
    p = P * np.ones(nyear)
    alpha = xr.ones_like(T_upper_mean)
    alpha.values = gsw.alpha(SA=sa, CT=T_upper_mean.values, p=p)
    ## compute alpha
    delta_a = src.utils.frac_change(alpha, inv=False)
    alpha_scale = alpha.isel(year=0) / alpha
    _save_fp = pathlib.Path(os.environ['SAVE_FP'], 'cesm_alpha_scale.nc')
    ## get fractional change
    if _save_fp.is_file():
        pass
    ## get scaling factor for RO
    else:
    ## save to file
        alpha_scale.to_netcdf(_save_fp)
    return (delta_a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot
    """)
    return


@app.cell
def _(T_upper, delta_a, plt):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(T_upper.year, delta_a)
    _ax.set_ylabel('Frac. change')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $h_w$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### funcs
    """)
    return


@app.cell
def _(xr):
    def lon_avg(x, lon_range):
        return x.sel(longitude=slice(*lon_range)).mean("longitude")


    def get_H_metrics(x, lons_e, lons_w):
        """compute metrics for thermocline"""

        metrics = xr.merge(
            [
                lon_avg(x, lons_w).rename("Hw"),
                lon_avg(x, lons_e).rename("He"),
                lon_avg(x, (lons_w[0], lons_e[1])).rename("Hbar"),
            ]
        )
        metrics["dHdx"] = metrics["He"] - metrics["Hw"]

        return metrics


    def frac_change(x, inv=False):
        """get fractional change"""

        if inv:
            x_ = 1 / x
        else:
            x_ = x

        return x_ / x_.isel(year=0) - 1

    return frac_change, get_H_metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load and compute
    """)
    return


@app.cell
def _(get_H_metrics, os, pathlib, src):
    ## specify longitude ranges
    LONS_W = (140, 210)
    LONS_E = (210, 270)
    H_mg_forced, _ = src.utils.load_h_data(max_grad=True)
    ## get maximum gradient of thermocline data
    H_mg_forced = src.utils.get_windowed(H_mg_forced, stride=120)
    H_mg_forced = get_H_metrics(H_mg_forced, lons_w=LONS_W, lons_e=LONS_E)
    ## remove first timestep for consistency with other data
    ## (don't do this for v2)
    # H_mg_forced = H_mg_forced.sel(time=slice("1851", None))
    Hbar = H_mg_forced['Hbar'].mean('time')
    ## get windowed
    Hbar_scale = Hbar / Hbar.isel(year=0)
    ssh_w = src.utils.load_cesm_indices()['h_w'].sel(time=slice('1851', None))
    ## get metrics
    ssh_w = src.utils.get_windowed(ssh_w, stride=120)
    _save_fp = pathlib.Path(os.environ['SAVE_FP'], 'cesm_Hbar_scale_v2.nc')
    ## compute Hbar (use this for scaling)
    if _save_fp.is_file():
        pass
    else:
    ## load ssh data
    ## save to file
        Hbar_scale.to_netcdf(_save_fp)
    return H_mg_forced, Hbar, ssh_w


@app.cell
def _(H_mg_forced, Hbar, frac_change, plt, ssh_w):
    _fig, _ax = plt.subplots(figsize=(3, 2.5))
    _ax.plot(H_mg_forced.year, frac_change(H_mg_forced['Hw'].mean('time'), inv=True))
    ## plot fractional change in h
    _ax.plot(H_mg_forced.year, frac_change(Hbar, inv=True))
    _ax.plot(ssh_w.year, frac_change(ssh_w.std(['time', 'member'])))
    ## plot change in sigma
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $\frac{\partial T}{\partial x}$
    (used for ELI scaling)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### v1
    """)
    return


@app.cell
def _(DATA_FP, os, pathlib, src, xr):
    ## Load T,h (total)
    # Th_total = xr.open_dataset(DATA_FP / "cesm" / "Th.nc").sel(time=slice("1851", "2100"))
    Th_total = xr.open_dataset(DATA_FP / 'cesm' / 'Th.nc')
    Th_total_forced = src.utils.separate_forced(Th_total)[0]
    ## get forced component
    _dTdx = Th_total_forced['T_4'] - Th_total_forced['T_3']
    _dTdx = src.utils.get_windowed(_dTdx, stride=120)
    ## compute dTdx
    _dTdx = _dTdx.groupby('time.month').mean()
    _save_fp = pathlib.Path(os.environ['SAVE_FP'], 'cesm_dTdx_v2.nc')
    ## get windowed
    if _save_fp.is_file():
        pass
    ## average in time
    else:
    ## save to file
        _dTdx.to_netcdf(_save_fp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### v2
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

    return avg(x.sel(**lon_range_w, **lat_range)) - avg(
        x.sel(**lon_range_e, **lat_range)
    )


@app.cell
def _(os, pathlib, src):
    ## load data
    _forced, _ = src.utils.load_consolidated()
    sst = _forced[['sst', 'sst_comp']]
    # ## subset for temperature
    # sst = forced[["sst", "sst_comp"]].sel(time=slice("1851", None))
    sst = src.utils.get_windowed(sst)
    # ## get windowed T
    # sst = src.utils.get_windowed(sst, stride=120)
    _dTdx = src.utils.reconstruct_wrapper(sst, get_ddx)['sst']
    ## subset for temperature
    _dTdx = _dTdx.groupby('time.month').mean()
    _save_fp = pathlib.Path(os.environ['SAVE_FP'], 'cesm_dTdx_Tw_v3.nc')
    ## get windowed T
    if _save_fp.is_file():
        pass
    ## compute grad
    else:
    ## average in time
    ## save to file
        _dTdx.to_netcdf(_save_fp)
    return


if __name__ == "__main__":
    app.run()
