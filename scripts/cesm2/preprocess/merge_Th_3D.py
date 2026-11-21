import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import xarray as xr
    import pathlib
    import numpy as np
    import os
    import src.utils
    import copy

    ## specify filepath for data
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    return DATA_FP, np, pathlib, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(DATA_FP, np, pathlib, xr):
    Th_3d = xr.open_mfdataset(
        sorted(list(pathlib.Path(DATA_FP / "cesm"/ "Th_3D").glob("*nc"))),
        combine="nested",
        concat_dim="member",
    )

    ## update coordinates to match
    Th_3d = Th_3d.assign_coords({"member":np.arange(100)}).compute()
    Th_3d = Th_3d.rename({"h":"thc"})
    return (Th_3d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Align times
    """)
    return


@app.cell
def _(Th_3d, xr):
    ## Get valid times
    valid_time = xr.date_range(start='1850-01-01', freq='MS', periods=3012, calendar='noleap', use_cftime=True)
    Th_3d_1 = Th_3d.assign_coords({'time': valid_time})
    return (Th_3d_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### compute indices
    """)
    return


@app.cell
def _(Th_3d_1, xr):
    def avg(x, lons):
        """average over longitudes"""
        return x.sel(longitude=slice(*lons)).mean(['longitude'])

    def get_Th_indices(x, T_var='T_50'):
        """ compute T,h indices using specified T-variable"""
        indices = xr.merge([avg(Th_3d_1[T_var], (210, 270)).rename('T_3'), avg(Th_3d_1[T_var], (190, 240)).rename('T_34'), avg(Th_3d_1[T_var], (160, 210)).rename('T_4'), avg(Th_3d_1[T_var], (230, 280)).rename('T_e'), avg(Th_3d_1[T_var], (180, 230)).rename('T_w'), avg(Th_3d_1['thc'], (120, 280)).rename('h'), avg(Th_3d_1['thc'], (120, 180)).rename('h_w')])
        return indices

    return (get_Th_indices,)


@app.cell
def _(Th_3d_1, get_Th_indices):
    Th_idx_50 = get_Th_indices(Th_3d_1, T_var='T_50')
    Th_idx_80 = get_Th_indices(Th_3d_1, T_var='T_80')
    return Th_idx_50, Th_idx_80


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save to file
    """)
    return


@app.cell
def _(DATA_FP, Th_3d_1, Th_idx_50, Th_idx_80, pathlib):
    for x, fname in [(Th_3d_1, 'Th_3D.nc'), (Th_idx_50, 'Th_50.nc'), (Th_idx_80, 'Th_80.nc')]:
        fp = pathlib.Path(DATA_FP, 'cesm', fname)
        if fp.is_file():
            pass
        else:
            x.to_netcdf(fp)
    return


if __name__ == "__main__":
    app.run()
