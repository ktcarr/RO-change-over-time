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
    return DATA_FP, np, pathlib, plt, src, tqdm, xr


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
def _(np, pathlib, xr):
    def get_ensemble_ids():
        """get files for given variable name"""

        ## path to cesm2 lens data
        cesm2_fp = pathlib.Path(
            "/glade/campaign/collections/rda/data/d651056/CESM2-LE/ocn/proc/tseries/month_1"
        )

        ## path to FSNS (arbitrary, just want the ids)
        data_fp = cesm2_fp / "WVEL"

        ## get list of ensemble ids
        ensemble_ids = []
        for f in data_fp.glob("*.nc"):
            ensemble_ids.append(str(f)[-53:-28])

        ## get unique values and sort
        ensemble_ids = sorted(list(set(ensemble_ids)))

        return ensemble_ids


    def load_var(varname, ensemble_id):
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
            chunks={"time": 18, "z_t": 60, "nlat": 384, "nlon": 320},
            parallel=True,
        )[varname]

        ## trim to eq Pac
        data = trim(data, lat_range=[-5, 5], lon_range=[120, 285])

        ## subset longitude and get top 300 m
        data = data.isel({"z_t": slice(None, 27)})

        ## convert to regular coords and convert cm to m
        data = data.assign_coords(
            {
                "longitude": data["TLONG"].mean("nlat"), 
                "latitude": data["TLAT"].mean("nlon"),
                "z_t": data.z_t / 100,
            }
        )

        ## convert to coordinates
        data = data.swap_dims({"nlon":"longitude", "nlat":"latitude"})

        ## drop old grid
        data = data.drop_vars(["ULONG","ULAT","TLONG","TLAT"])

        return data

    def compute_ml_avg(x, mld=50):
        """Compute mixed layer average"""
        return x.sel(z_t=slice(None, mld)).mean(["z_t", "latitude"])

    def find_maxgrad(x):
        """function to find depth of thermocline"""

        ## find index of max gradient
        idx = x.differentiate("z_t").fillna(1e20).argmin("z_t")
    
        ## load indexer into memory
        idx.load()

        ## get thermocline depth
        h = x.z_t.isel(z_t=idx)

        ## fill in NaN values
        h = h.where(~np.isnan(x.isel(z_t=0)), other=np.nan)

        return h

    def compute_h(x):
        """function to find depth of thermocline"""

        return find_maxgrad(x).mean("latitude")

    return compute_h, compute_ml_avg, find_maxgrad, get_ensemble_ids, load_var


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Initialize cluster
    """)
    return


@app.cell
def _():
    from dask.distributed import LocalCluster, Client

    cluster = LocalCluster(n_workers=16)
    client = Client(cluster)
    client
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### test: load one file
    """)
    return


@app.cell
def _(get_ensemble_ids):
    ids = get_ensemble_ids()
    return (ids,)


@app.cell
def _(ids, load_var):
    d = load_var("TEMP", ids[0])

    ## trim in time
    d = d.isel(time=slice(None,18))
    return (d,)


@app.cell
def _(compute_ml_avg, d, find_maxgrad):
    h = find_maxgrad(d)
    T_ml_50 = compute_ml_avg(d, mld=50)
    T_ml_80 = compute_ml_avg(d, mld=80)
    return (h,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## check thermocline calc.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spatial plot
    """)
    return


@app.cell
def _(h, np, plt, src):
    import cartopy.crs as ccrs
    import cmocean
    _fig = plt.figure(figsize=(10, 2), layout='constrained')
    format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=5)
    axs = src.utils.subplots_with_proj(_fig, nrows=1, ncols=1, format_func=format_func)
    cp = axs[0, 0].contourf(h.longitude, h.latitude, h.mean('time'), transform=ccrs.PlateCarree(), levels=np.arange(30, 200, 20), extend='both', cmap='cmo.amp')
    axs[0, 0].set_xlim([-50, 90])
    axs[0, 0].set_aspect('auto')
    cb = _fig.colorbar(cp, label='depth')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### line plot
    """)
    return


@app.cell
def _(h, plt):
    _fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(h.longitude, h.mean(['latitude', 'time']))
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute for ensemble
    """)
    return


@app.cell
def _(
    compute_h,
    compute_ml_avg,
    get_ensemble_ids,
    load_var,
    pathlib,
    tqdm,
    xr,
):
    def preprocess_ensemble(save_dir):
        """compute net heat flux for full ensemble. Save to temp directory"""

        ## get ensemble ids
        ensemble_ids = get_ensemble_ids()

        ## loop through members
        for i in tqdm.tqdm(ensemble_ids):

            ## save filepath
            save_fp = pathlib.Path(save_dir, f"Th-3D_{i}.nc")

            if save_fp.is_file():
                pass

            else:
                try:
                    data = compute_member(id=i)
                    data.to_netcdf(save_fp)
                except:
                    print(f"warning: {i} didn't work...")
                    pass

        return

    def compute_member(id):
        """compute variables for given ensemble member"""

        ## load data
        x = load_var("TEMP", ensemble_id=id)

        ## compute mixed layer Temp and thermocline depth
        data = xr.merge(
            [
                compute_ml_avg(x, mld=50).rename("T_50"),
                compute_ml_avg(x, mld=80).rename("T_80"),
                compute_h(x).rename("h"),
            ]
        )

        return data

    return (preprocess_ensemble,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute
    """)
    return


@app.cell
def _(DATA_FP, pathlib, preprocess_ensemble):
    preprocess_ensemble(save_dir = pathlib.Path(DATA_FP, "cesm", "Th_3D"))
    return


if __name__ == "__main__":
    app.run()
