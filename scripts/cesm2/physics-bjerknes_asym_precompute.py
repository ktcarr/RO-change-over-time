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
    # Pre-compute regression coefficients
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

    # Import custom modules
    import src.utils

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return DATA_FP, SAVE_FP, np, os, pathlib, src, tqdm, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## specs
    """)
    return


@app.cell
def _(DATA_FP, pathlib):
    ## should we use mixed layer T?
    use_T_ml = True
    MLD = 80

    ## should we use OHC?
    USE_OHC = True

    ## should we subtract the median?
    SUBTRACT_MEDIAN = False

    ## specify save path
    BJERKNES_SAVE_DIR = pathlib.Path(DATA_FP, "bjerknes_ML_80")

    ## should we use "wide" data?
    USE_WIDE = True
    return BJERKNES_SAVE_DIR, MLD, SUBTRACT_MEDIAN, USE_OHC, USE_WIDE, use_T_ml


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funcs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Misc
    """)
    return


@app.cell
def _(os, pathlib, src, tqdm, xr):
    def window(x, subtract_median = False):
        """get data in windows"""
        x_windowed = src.utils.get_windowed(x, stride=120)

        ## handle median case
        if subtract_median:
            median = x_windowed.groupby("time.month").median(["time","member"])
            x_windowed = x_windowed.groupby("time.month") - median
    
        return x_windowed


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


    def regress_over_time(data_windowed, x_vars, y_vars, dims=["time"], subtract_median=False):
        """regression over time"""

        ## empty list to hold coefficients
        coefs = []

        ## shared args
        kwargs = dict(x_vars=x_vars, y_vars=y_vars, dims=dims)

        ## loop thru years
        for year in tqdm.tqdm(data_windowed.year):

            ## get grouped data
            data_y = data_windowed.sel(year=year).groupby("time.month")

            ## do regression
            coefs.append(data_y.map(src.utils.regress_xr_proj, **kwargs))

        return xr.concat(coefs, dim=data_windowed.year)

    return load_consolidated_wide, regress_over_time, window


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $T$, $h$
    """)
    return


@app.cell
def _(src):
    ## open data
    Th = src.utils.load_cesm_indices(load_z20=True)
    return (Th,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spatial data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### most data
    """)
    return


@app.cell
def _(Th, src, xr):
    ## load spatial data
    forced, anom = src.utils.load_consolidated()

    ## add T,h information
    anom = xr.merge([anom, Th])
    return anom, forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### "wide" subsurface data
    """)
    return


@app.cell
def _(USE_WIDE, anom, forced, load_consolidated_wide):
    ## load spatial data
    forced_wide, anom_wide = load_consolidated_wide()
    if USE_WIDE:
        for _v in list(forced_wide):
            forced[_v] = forced_wide[_v]
            anom[_v] = anom_wide[_v]
    return (anom_wide,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### max grad thermocline
    """)
    return


@app.cell
def _(src):
    h_mg_forced, h_mg = src.utils.load_h_data(max_grad=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### scaling for mean thermocline depth
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Mean thermocline depth
    """)
    return


@app.cell
def _(SAVE_FP, pathlib, xr):
    hbar_scale = xr.open_dataarray(
        pathlib.Path(SAVE_FP, "cesm_Hbar_scale_v2.nc"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute OHC
    """)
    return


@app.cell
def _():
    ## specify subsetting funcs
    LATS = dict(latitude=slice(-5, 5))
    LATS_H = dict(latitude=slice(-5, 5))
    LONS_E = dict(longitude=slice(210, 270))
    # LONS_W = dict(longitude=slice(120, 160))
    # LONS_W = dict(longitude=slice(120, 180))
    LONS_W = dict(longitude=slice(120, 170))
    LONS_TAU = dict(longitude=slice(150, 230))
    return LATS, LATS_H, LONS_E, LONS_TAU, LONS_W


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set funcs
    """)
    return


@app.cell
def _(LATS, LATS_H, LONS_E, LONS_TAU, LONS_W, MLD, anom, src, use_T_ml):
    ## helper func to select and avg
    def sel_helper(x, lats, lons):
        """helper func to avg over lats/lons"""

        ## first, average over lons
        x_avg = x.sel(lons).mean("longitude")

        if "latitude" in x_avg.dims:
            x_avg = x_avg.sel(lats).mean("latitude")

        return x_avg


    ## specify functions
    TAU_FN = lambda x: sel_helper(x, LATS, LONS_TAU)
    TAU_FN_3 = lambda x: sel_helper(x, LATS, LONS_E)
    He_FN = lambda x: sel_helper(x, LATS_H, LONS_E)
    Hw_FN = lambda x: sel_helper(x, LATS_H, LONS_W)
    Hgrad_FN = lambda x: He_FN(x) - Hw_FN(x)


    ## specify entrainment / ML averages
    LON_AVG = lambda x: x.sel(longitude=slice(210, 270)).mean("longitude")
    LON_AVG_34 = lambda x: x.sel(longitude=slice(190, 240)).mean("longitude")
    ENT_AVG = lambda x: x.sel(z_t=slice(50, 80)).mean("z_t")
    ML_AVG = lambda x: x.sel(z_t=slice(None, MLD)).mean("z_t")

    ## get T3 volume avg
    T3_ENT_AVG = lambda x: ENT_AVG(LON_AVG(x))
    T3_ML_AVG = lambda x: ML_AVG(LON_AVG(x))
    T34_ML_AVG = lambda x: ML_AVG(LON_AVG_34(x))

    if use_T_ml:
        anom["T_3"] = src.utils.reconstruct_wrapper(
            anom[["T", "T_comp"]],
            fn=T3_ML_AVG,
        )["T"]

        anom["T_34"] = src.utils.reconstruct_wrapper(
            anom[["T", "T_comp"]],
            fn=T34_ML_AVG,
        )["T"]
    return


@app.cell
def _(LONS_E, LONS_W, USE_OHC, anom, anom_wide, src):
    lon_avg = lambda x, lons: x.sel(lons).mean("longitude")
    lat_avg = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")

    if USE_OHC:

        ## compute ohc
        anom["h_w"] = src.utils.reconstruct_wrapper(
            anom_wide[["T", "T_comp"]],
            lambda x: lon_avg(x.integrate("z_t"), LONS_W) / 300,
        )["T"]
        anom["h_e"] = src.utils.reconstruct_wrapper(
            anom_wide[["T", "T_comp"]],
            lambda x: lon_avg(x.integrate("z_t"), LONS_E) / 300,
        )["T"]

    else:
        anom["h_w"] = src.utils.reconstruct_wrapper(
            anom[["ssh", "ssh_comp"]],
            lambda x: lat_avg(lon_avg(x, LONS_W)),
        )["ssh"]
        anom["h_e"] = src.utils.reconstruct_wrapper(
            anom[["ssh", "ssh_comp"]],
            lambda x: lat_avg(lon_avg(x, LONS_E)),
        )["ssh"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute regression coefs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute ddt
    """)
    return


@app.cell
def _(anom, np):
    ## compute ddt
    anom_ = anom[['sst', 'T', 'T_34', 'T_3', 'h_w', 'h']]
    anom_ = anom_.assign_coords({'t_idx': ('time', np.arange(len(anom_.time)))})
    anom_ = anom_.swap_dims({'time': 't_idx'})
    for _v in list(anom_):
        anom[f'ddt_{_v}'] = anom_[_v].differentiate('t_idx').swap_dims({'t_idx': 'time'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Subset data
    """)
    return


@app.cell
def _(anom):
    ## subset data
    _x_vars = ['T_3', 'T_34', 'h_w', 'h']
    y_vars = ['T', 'u', 'w', 'pr', 'taux', 'nhf', 'ddt_sst', 'ddt_T', 'ddt_T_3', 'ddt_T_34', 'ddt_h_w', 'ddt_h']
    anom_subset = anom[_x_vars + y_vars]
    return anom_subset, y_vars


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Should we subtract median?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### window the data
    """)
    return


@app.cell
def _(SUBTRACT_MEDIAN, anom_subset, window):
    anom_windowed = window(anom_subset, subtract_median=SUBTRACT_MEDIAN)
    return (anom_windowed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute
    """)
    return


@app.cell
def _(
    BJERKNES_SAVE_DIR,
    SUBTRACT_MEDIAN,
    anom_windowed,
    os,
    regress_over_time,
    y_vars,
):
    arg_list = [(['T_3'], 'coefs_xvars=T_3'), (['T_34'], 'coefs_xvars=T_34'), (['T_3', 'h_w'], 'coefs_xvars=T_3-h_w'), (['T_34', 'h_w'], 'coefs_xvars=T_34-h_w')]
    for _x_vars, subdir in arg_list:
        save_dir = BJERKNES_SAVE_DIR / subdir
        if save_dir.is_dir():
            print(f'{subdir} already computed')
        else:
            os.mkdir(save_dir)
            print(f'Computing {subdir}...')
            is_pos = anom_windowed[_x_vars[0]] > 0
            is_neg = anom_windowed[_x_vars[0]] < 0  ## specify savepath
            regress_kwargs = dict(x_vars=_x_vars, y_vars=y_vars, subtract_median=SUBTRACT_MEDIAN)
            coefs = regress_over_time(anom_windowed, **regress_kwargs)
            coefs.to_netcdf(save_dir / 'all.nc')  ## check it exists
            coefs_pos = regress_over_time(anom_windowed.where(is_pos), **regress_kwargs)
            coefs_pos.to_netcdf(save_dir / 'pos.nc')
            coefs_neg = regress_over_time(anom_windowed.where(is_neg), **regress_kwargs)
            coefs_neg.to_netcdf(save_dir / 'neg.nc')  ## make new directory  ## find pos/neg index  ## shared args for regression  ## all  ## pos  ## neg
    return


if __name__ == "__main__":
    app.run()
