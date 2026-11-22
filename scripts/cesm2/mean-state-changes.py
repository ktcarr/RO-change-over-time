import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mean-state changes
    Look at how mean state changes over time
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
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return (
        DATA_FP,
        cartopy,
        cmocean,
        np,
        os,
        pathlib,
        pd,
        plt,
        scipy,
        sns,
        src,
        tqdm,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funcs
    """)
    return


@app.cell
def _(os, pathlib, xr):
    def get_ensemble_fn(data, fn):
        """apply python function to data using xr.apply_ufunc"""

        ## stack data
        data_stack = data.stack(sample=["time", "member"])

        ## kwargs for apply_ufunc
        kwargs = dict(
            input_core_dims=[["sample", "year"]],
            output_core_dims=[["year"]],
        )

        return xr.apply_ufunc(fn, data_stack, **kwargs)


    def get_ensemble_fn_bymonth(data, fn):
        """apply python function to data by month using xr.apply_ufunc"""
        return data.groupby("time.month").map(get_ensemble_fn, fn=fn)


    def get_nino3(x, z_t=70, lat_bound=5):
        """get Niño 3 average, with checks in place"""

        if "z_t" in x.dims:
            x = x.sel(z_t=z_t, method="nearest")

        if "latitude" in x.dims:
            x = x.sel(latitude=slice(-lat_bound, lat_bound)).mean("latitude")

        return x.sel(longitude=slice(210, 270)).mean("longitude")


    def get_nino3_helper(x, lat_bounds):
        """get Niño 3 average, with checks in place"""

        ## get index to select data
        idx = dict(latitude=slice(*lat_bounds), longitude=slice(210, 270))

        return x.sel(idx).mean(["longitude", "latitude"])


    def get_nino3_north(x):
        return get_nino3_helper(x, lat_bounds=(0, 10))


    def get_nino3_south(x):
        return get_nino3_helper(x, lat_bounds=(-10, 0))


    def save(fig, fname, dpi=300, do_save=False):
        """save figure to file"""

        ## get save directory
        save_dir = pathlib.Path(os.environ["SAVE_FP"], "ch3-outline")

        ## get fname
        fname = save_dir / f"{fname}.pdf"

        if do_save:
            fig.savefig(fname, dpi=dpi)
        
        return

    return (
        get_ensemble_fn_bymonth,
        get_nino3,
        get_nino3_north,
        get_nino3_south,
        save,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(get_nino3, get_nino3_north, get_nino3_south, src, xr):
    ## load spatial data
    forced, anom_ = src.utils.load_consolidated()

    ## get subset of data for total
    _VARNAMES = ["T", "w", "pr", "tauy", "sst"]
    total = anom_[_VARNAMES] + forced[_VARNAMES]
    total = xr.merge([forced[[f"{v}_comp" for v in _VARNAMES]], total])

    ## get nino3 averages
    total_n3 = src.utils.reconstruct_wrapper(total, fn=get_nino3)

    ## rename to avoid conflicts
    total_n3 = total_n3.rename({v: f"{v}_n3" for v in _VARNAMES})

    ## get precip North and south of equator
    pr_N = src.utils.reconstruct_wrapper(
        total[["pr", "pr_comp", "tauy", "tauy_comp"]],
        fn=get_nino3_north,
    ).rename({"pr": "pr_n", "tauy": "tauy_n"})

    pr_S = src.utils.reconstruct_wrapper(
        total[["pr", "pr_comp", "tauy", "tauy_comp"]],
        fn=get_nino3_south,
    ).rename({"pr": "pr_s", "tauy": "tauy_s"})

    ## merge
    total_n3 = xr.merge([total_n3, pr_N, pr_S])
    return anom_, forced, total, total_n3


@app.cell
def _(DATA_FP, forced, pathlib, src, xr):
    ## get windowed data
    forced_1 = src.utils.get_windowed(forced).groupby("time.month").mean()

    ## reconstruct data
    forced_1 = src.utils.reconstruct_wrapper(forced_1)

    # ## load tropical SST avg
    trop_sst = xr.open_dataset(pathlib.Path(DATA_FP, "cesm/trop_sst.nc"))

    ## Load T,h (total)
    Th_total = xr.open_dataset(DATA_FP / "cesm" / "Th.nc")
    Th_total = xr.merge([Th_total, trop_sst])

    ## custom h data
    h_mg_forced, h_mg_anom = src.utils.load_h_data(max_grad=True)
    return Th_total, forced_1


@app.cell
def _(DATA_FP, Th_total, forced_1, pathlib, src, total, total_n3, xr):
    ## compute diff. def of Tw
    get_Tw = lambda x: x.sel(latitude=slice(-5, 5), longitude=slice(120, 160)).mean(
        ["longitude", "latitude"]
    )
    Th_total["Tw"] = src.utils.reconstruct_wrapper(total[["sst", "sst_comp"]], get_Tw)[
        "sst"
    ]

    ## compute dTdx
    Th_total["dTdx"] = Th_total["T_4"] - Th_total["T_3"]
    Th_total["dTdx_v2"] = Th_total["Tw"] - Th_total["T_3"]

    ## load ELI
    eli = xr.open_dataset(pathlib.Path(DATA_FP, "cesm/eli.nc"))
    Th_total_1 = xr.merge([Th_total, eli, total_n3])


    ## windowed version
    Th_total_1 = src.utils.get_windowed(Th_total_1)

    ## relative SST
    for _v in ["T_3", "T_34", "T_4"]:
        Th_total_1[f"{_v}_rel"] = Th_total_1[_v] - Th_total_1["trop_sst_15"]

    ## ensemble mean
    Th_total_forced = Th_total_1.groupby("time.month").mean(["time", "member"])

    ## merge data
    forced_2 = xr.merge([forced_1, Th_total_forced])
    return Th_total_1, forced_2


@app.cell
def _(Th_total_1):
    ## compute quantiles
    quant = Th_total_1.groupby("time.season").quantile(
        q=[0.05, 0.25, 0.5, 0.75, 0.95], dim=["time", "member"]
    )
    quant = quant.rename({"quantile": "q"})

    ## compute difference
    _delta_q = quant - quant.isel(year=0)

    ## get spread
    _spread = quant.sel(q=0.95) - quant.sel(q=0.05)
    delta_spread = _spread - _spread.isel(year=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute variance
    """)
    return


@app.cell
def _(Th_total_1, src):
    ## open data
    Th = src.utils.load_cesm_indices()

    ## get windowed
    Th = src.utils.get_windowed(Th)
    for _v in ["05", "15", "20", "25", "30"]:
        Th[f"eli_{_v}"] = Th_total_1[f"eli_{_v}"] - Th_total_1[f"eli_{_v}"].mean("member")

    ## get variance by month
    _Th_sigma = Th.groupby("time.month").std(["time", "member"])

    ## fractional change
    Th_delta_sigma = (_Th_sigma - _Th_sigma.isel(year=0)) / _Th_sigma.isel(year=0)
    return Th, Th_delta_sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute skewness
    """)
    return


@app.cell
def _(Th, get_ensemble_fn_bymonth, scipy):
    Th_skew = get_ensemble_fn_bymonth(Th, fn=scipy.stats.skew)
    Th_delta_skew = Th_skew - Th_skew.isel(year=0)
    return (Th_delta_skew,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute quantiles
    """)
    return


@app.cell
def _(Th_total_1):
    ## compute quantiles
    quant_1 = Th_total_1.groupby("time.season").quantile(
        q=[0.02, 0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95, 0.98], dim=["time", "member"]
    )
    quant_1 = quant_1.rename({"quantile": "q"})

    ## compute difference
    _delta_q = quant_1 - quant_1.isel(year=0)

    ## get spread
    _spread = quant_1.sel(q=0.95) - quant_1.sel(q=0.05)
    delta_spread_1 = _spread - _spread.isel(year=0)
    return delta_spread_1, quant_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spatial plots
    """)
    return


@app.cell
def _(forced_2):
    forced0 = forced_2.sel(year=1870)
    forced1 = forced_2.sel(year=2010)
    forced2 = forced_2.sel(year=2080)
    return forced0, forced1, forced2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SST
    """)
    return


@app.cell
def _(forced0, forced1, forced2, np, plt, src):
    _v0 = "sst"
    v1 = "trop_sst_15"

    import cartopy.crs as ccrs

    sel_ = lambda x: x.sel(month=[12, 1, 2]).mean("month")

    _fig = plt.figure(figsize=(10, 4.375), layout="constrained")
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=30)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=2, format_func=_format_func)

    ## specify kwargs
    _plot_kwargs = dict(
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(10, 1),
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    for _i, _f, _f_ in zip(
        np.arange(3), [forced0, forced1, forced2], [forced0, forced0, forced1]
    ):

        ## plot relative SST
        _axs[_i, 0].contourf(
            _f.longitude,
            _f.latitude,
            sel_(_f[_v0] - _f[v1]),
            **_plot_kwargs,
        )

        ## difference
        _axs[_i, 1].contourf(
            _f.longitude,
            _f.latitude,
            5 * (sel_(_f[_v0] - _f[v1]) - sel_(_f_[_v0] - _f_[v1])),
            **_plot_kwargs,
        )


    plt.gca()
    return (ccrs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### precip
    """)
    return


@app.cell
def _(ccrs, forced0, forced1, forced2, np, plt, src):
    _v0 = "pr"



    # sel_ = lambda x: x.sel(month=[12, 1, 2]).mean("month")
    sel__1 = lambda x: x.mean("month")
    # sel_ = lambda x: x.sel(month=[3]).mean("month")

    _fig = plt.figure(figsize=(10, 4.375), layout="constrained")
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=30)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=2, format_func=_format_func)

    ## specify kwargs
    _plot_kwargs = dict(
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    for _i, _f, _f_ in zip(
        np.arange(3), [forced0, forced1, forced2], [forced0, forced0, forced1]
    ):

        ## plot relative SST
        _axs[_i, 0].contourf(
            _f.longitude,
            _f.latitude,
            8.6e4 * sel__1(_f[_v0]),
            cmap="cmo.rain",
            levels=np.arange(0, 16, 2),
            **_plot_kwargs,
        )

        ## difference
        _axs[_i, 1].contourf(
            _f.longitude,
            _f.latitude,
            8.6e4 * (sel__1(_f[_v0] - _f_[_v0])),
            cmap="cmo.balance_r",
            levels=src.utils.make_cb_range(4, 0.4),
            **_plot_kwargs,
        )

    for _ax in _axs.flatten():
        _ax.axhline(0, ls="--", c="w", lw=1)


    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### wind
    """)
    return


@app.cell
def _(ccrs, forced0, forced1, forced2, np, plt, src):
    # ## variable to plot
    # v0 = "taux"
    # SC=8e1

    ## variable to plot
    _v0 = "taux"
    SC = 8e1
    # SC = -16e1



    # sel_ = lambda x: x.sel(month=[3]).mean("month")
    sel__2 = lambda x: x.mean("month")

    _fig = plt.figure(figsize=(10, 4.375), layout="constrained")
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=30)
    _axs = src.utils.subplots_with_proj(_fig, nrows=3, ncols=2, format_func=_format_func)

    ## specify kwargs
    _plot_kwargs = dict(
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    for _i, _f, _f_ in zip(
        np.arange(3), [forced0, forced1, forced2], [forced0, forced0, forced0]
    ):

        ## plot relative SST
        _axs[_i, 0].contourf(
            _f.longitude,
            _f.latitude,
            SC * sel__2(_f[_v0]),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(10, 1),
            **_plot_kwargs,
        )

        ## difference
        _axs[_i, 1].contourf(
            _f.longitude,
            _f.latitude,
            SC * (sel__2(_f[_v0] - _f_[_v0])),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(2, 0.2),
            **_plot_kwargs,
        )

        _axs[_i, 1].axhline(0, ls="--", c="w")


    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Precip / $\tau_x$ on the equator
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Absolute
    """)
    return


@app.cell
def _(forced_2, np, plt, src):
    ## specify month and lat bound for averaging
    _MONTH = 5
    _LAT_BOUND = 5

    ## func to select data
    merimean = lambda x, lat_bound=5: x.sel(latitude=slice(-lat_bound, lat_bound)).mean(
        "latitude"
    )
    sel = lambda x: merimean(x, lat_bound=_LAT_BOUND).sel(month=_MONTH).transpose("year", ...)

    ## get longitude for convenience
    _LON = forced_2.longitude

    ## compute center of mass
    _forced_pac = forced_2.sel(longitude=slice(140, 280))
    _com = sel(_forced_pac * _LON).integrate("longitude") / sel(_forced_pac).integrate(
        "longitude"
    )

    ## get argmin
    _taux_argmin = _forced_pac.longitude.isel(
        longitude=sel(_forced_pac["taux"]).argmin("longitude")
    )

    _fig, _axs = plt.subplots(1, 2, figsize=(4, 4), layout="constrained")

    ## precip
    _axs[0].contourf(
        forced_2.longitude,
        forced_2.year,
        8.6e4 * sel(forced_2["pr"]),
        cmap="cmo.rain",
        levels=np.arange(0, 13.5, 1.5),
    )

    ## wind
    _axs[1].contourf(
        forced_2.longitude,
        forced_2.year,
        sel(forced_2["taux"]),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(0.075, 0.0075),
    )

    ## plot longitude of max grad
    _axs[1].plot(
        _taux_argmin,
        forced_2.year,
        c="w",
        ls="--",
        alpha=1,
        lw=1,
    )

    ## plot longitude of max grad
    _axs[0].plot(
        _com["pr"],
        forced_2.year,
        c="w",
        ls="--",
        alpha=1,
        lw=1,
    )


    ## formatting
    for _ax in _axs:
        _ax.set_xlim([140, 280])
    for _ax in _axs[1:]:
        _ax.set_yticks([])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Change
    """)
    return


@app.cell
def _(forced_2, plt, src):
    ## specify month and lat bound for averaging
    _MONTH = 5
    _LAT_BOUND = 5

    ## func to select data
    merimean_1 = lambda x, lat_bound=5: x.sel(latitude=slice(-lat_bound, lat_bound)).mean(
        "latitude"
    )
    diff = lambda x: (x - x.isel(year=0)) / x.isel(year=0)
    sel_1 = (
        lambda x: merimean_1(diff(x), lat_bound=_LAT_BOUND)
        .sel(month=_MONTH)
        .transpose("year", ...)
    )

    ## get longitude for convenience
    _LON = forced_2.longitude

    ## compute center of mass
    _forced_pac = forced_2.sel(longitude=slice(140, 280))
    _com = sel_1(_forced_pac * _LON).integrate("longitude") / sel_1(_forced_pac).integrate(
        "longitude"
    )

    ## get argmin
    _taux_argmin = _forced_pac.longitude.isel(
        longitude=sel_1(_forced_pac["taux"]).argmin("longitude")
    )

    _fig, _axs = plt.subplots(1, 2, figsize=(4, 4), layout="constrained")

    ## shared args
    _kwargs = dict(
        levels=src.utils.make_cb_range(1, 0.1),
        extend="both",
    )

    ## precip
    _axs[0].contourf(
        forced_2.longitude,
        forced_2.year,
        sel_1(forced_2["pr"]),
        cmap="cmo.balance_r",
        **_kwargs,
    )

    ## wind
    _axs[1].contourf(
        forced_2.longitude,
        forced_2.year,
        sel_1(forced_2["taux"]),
        cmap="cmo.balance",
        **_kwargs,
    )


    ## formatting
    for _ax in _axs:
        _ax.set_xlim([140, 280])
    for _ax in _axs[1:]:
        _ax.set_yticks([])

    plt.gca()
    return


@app.cell
def _(forced_2, np, plt, src):
    ## specify month and lat bound for averaging
    _MONTH = 2

    ## func to select data
    zonamean = lambda x: x.sel(longitude=slice(210, 270)).mean("longitude")
    sel_2 = lambda x: zonamean(x).sel(month=[2, 3, 4]).mean("month").transpose("year", ...)


    _fig, _axs = plt.subplots(1, 2, figsize=(4, 4), layout="constrained")

    ## precip
    _axs[0].contourf(
        forced_2.latitude,
        forced_2.year,
        8.6e4 * sel_2(forced_2["pr"]),
        cmap="cmo.rain",
        levels=np.arange(0, 10, 1),
        extend="both",
    )

    ## wind
    _axs[1].contourf(
        forced_2.latitude,
        forced_2.year,
        -sel_2(forced_2["tauy"]) + sel_2(forced_2["tauy"]).isel(year=0),
        cmap="cmo.balance",
        # levels=src.utils.make_cb_range(5e-2, 5e-3),
        levels=src.utils.make_cb_range(1e-2, 1e-3),
        extend="both",
    )

    ## formatting
    for _ax in _axs:
        _ax.axvline(0, c="w", ls="--", lw=0.8)
        _ax.set_xlim([-15, 15])
    for _ax in _axs[1:]:
        _ax.set_yticks([])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    rain
    """)
    return


@app.cell
def _(forced_2, np, plt, src):
    _y = 8.6e4 * forced_2["pr"].sel(longitude=slice(210, 270)).mean("longitude")
    # y = 8.6e4 * forced["pr"].sel(longitude=slice(150, 220)).mean("longitude")
    # y = 8.6e4 * forced["pr"].sel(longitude=slice(190, 240)).mean("longitude")
    # y = 3e2 * forced["taux"].sel(longitude=slice(190,240)).mean("longitude")

    # Y0 = 1870
    _Y0 = 2010
    _Y1 = 2080

    _fig, _axs = plt.subplots(3, 1, figsize=(3, 9), layout="constrained")
    _cp0 = _axs[0].contourf(
        _y.month,
        _y.latitude,
        _y.sel(year=_Y0).transpose("latitude", ...),
        cmap="cmo.rain",
        levels=np.arange(0, 20, 2),
        extend="both",
    )

    _axs[1].contourf(
        _y.month,
        _y.latitude,
        _y.sel(year=_Y1).transpose("latitude", ...),
        cmap="cmo.rain",
        levels=np.arange(0, 20, 2),
        extend="both",
    )

    _cp2 = _axs[2].contourf(
        _y.month,
        _y.latitude,
        (_y.sel(year=_Y1) - _y.sel(year=_Y0)).transpose("latitude", ...),
        cmap="cmo.balance_r",
        levels=src.utils.make_cb_range(10, 1),
        extend="both",
    )

    for _ax in _axs:
        _ax.set_ylim([-20, 20])
        _ax.set_xticks([3, 6, 9])
        _ax.set_ylabel("Latitude")
        for _t in [-5, 5]:
            _ax.axhline(_t, ls="--", c="gray", lw=0.8)

    _fig.colorbar(_cp0, ax=_axs[1], orientation="horizontal", label="mm/day")
    _fig.colorbar(_cp2, ax=_axs[2], orientation="horizontal", label="mm/day")
    _axs[0].set_title("Niño 3 precip")
    _axs[2].set_xlabel("Month")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\tau_x$
    """)
    return


@app.cell
def _(forced_2, plt, src):
    LON_RANGE = slice(140, 280)
    _y = 3e2 * forced_2["taux"].sel(longitude=LON_RANGE).mean("longitude")

    _Y0 = 2010
    _Y1 = 2080

    _fig, _axs = plt.subplots(3, 1, figsize=(4, 8))
    for _ax, _year in zip(_axs[:2], [_Y0, _Y1]):
        _ax.contourf(
            _y.month,
            _y.latitude,
            _y.sel(year=_year).transpose("latitude", ...),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(30, 3),
            extend="both",
        )

    _axs[2].contourf(
        _y.month,
        _y.latitude,
        (_y.sel(year=_Y1) - _y.sel(year=_Y0)).transpose("latitude", ...),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(10, 1),
        extend="both",
    )

    for _ax in _axs:
        _ax.set_ylim([-20, 20])
        for _t in [-5, 5]:
            _ax.axhline(_t, ls="--", c="gray", lw=0.8)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\tau_y$
    """)
    return


@app.cell
def _(forced_2, plt, src):
    _y = -2e2 * forced_2["tauy"].sel(longitude=slice(210, 270)).mean("longitude")
    # y = 2e2*forced["taux"].sel(longitude=slice(210,270)).mean("longitude")


    _fig, _axs = plt.subplots(2, 1, figsize=(5, 6))
    _axs[0].contourf(
        _y.month,
        _y.latitude,
        _y.sel(year=1870).transpose("latitude", ...),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(15, 1.5),
        extend="both",
    )

    _axs[1].contourf(
        _y.month,
        _y.latitude,
        (_y.sel(year=1975) - _y.sel(year=1870)).transpose("latitude", ...),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(1.5, 0.15),
        extend="both",
    )

    for _ax in _axs:
        _ax.set_ylim([-20, 20])
    plt.gca()
    return


@app.cell
def _(Th_total_1, plt, src):
    _y = src.utils.sel_month(Th_total_1, [2]).stack(s=["member", "time"])

    _fig, _axs = plt.subplots(1, 2, figsize=(6.5, 3))
    _axs[0].scatter(
        # y["T_3"].sel(year=1870),
        -_y["tauy_s"].sel(year=1870),
        (_y["pr_n"] - _y["pr_n3"]).sel(year=1870),
        s=5,
    )

    # axs[0].scatter(
    #     # y["T_3"].sel(year=1970),
    #     -y["tauy_s"].sel(year=1970),
    #     (y["pr_n"]-y["pr_n3"]).sel(year=1970),
    #     s=5,
    # )
    # src.utils.set_lims(axs)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stratification by season over time
    """)
    return


@app.cell
def _(src):
    def get_dTdz_sub(Tsub, mld, delta=25):
        """get velocity at base of mixed layer"""

        ## get temperature difference
        dT = src.utils.get_dT_sub(Tsub=Tsub, Hm=mld, delta=delta)

        ## get gradient
        dTdz = -dT / mld

        return dTdz

    return (get_dTdz_sub,)


@app.cell
def _(forced_2, get_dTdz_sub, plt, src):
    ## compute strat
    _dTdz = get_dTdz_sub(Tsub=forced_2.T, mld=50, delta=20)

    ## average over Niño 3
    dTdz_n3 = _dTdz.sel(longitude=slice(210, 270)).mean("longitude")

    ## get fractional change
    get_frac_change = lambda x: (x - x.isel(year=0)) / x.isel(year=0)


    ## make plot
    _fig, _ax = plt.subplots(figsize=(2, 4), layout="constrained")

    ## precip
    _ax.contourf(
        dTdz_n3.month,
        dTdz_n3.year,
        # (dTdz_n3 - dTdz_n3.isel(year=0)).transpose("year", ...),
        get_frac_change(dTdz_n3).transpose("year", ...),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(1.5, 0.15),
        extend="both",
    )

    ## formatting
    _ax.axvline(5, ls="--", lw=0.8, c="k")

    plt.gca()
    return (get_frac_change,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $w$ by season
    """)
    return


@app.cell
def _(forced_2, get_dTdz_sub, get_frac_change, plt, src):
    ## compute strat
    _dTdz = get_dTdz_sub(Tsub=forced_2.T, mld=50, delta=20)

    ## average over Niño 3
    dTdz_n3_1 = _dTdz.sel(longitude=slice(210, 270)).mean("longitude")

    ## make plot
    _fig, _ax = plt.subplots(figsize=(2, 4), layout="constrained")

    ## precip
    _ax.contourf(
        forced_2.month,
        forced_2.year,
        get_frac_change(forced_2["w_n3"]).transpose("year", ...),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(0.5, 0.05),
        extend="both",
    )

    ## formatting
    _ax.axvline(5, ls="--", lw=0.8, c="k")

    plt.gca()
    return (dTdz_n3_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $w\frac{dT}{dz}$ by season
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Timeseries
    """)
    return


@app.cell
def _(forced_2):
    ## specify reference coordinate and inflection pt
    ref = forced_2["trop_sst_15"]
    # ref = forced["T_3"] - forced["trop_sst_15"]
    ref_year = 1980

    ## specify plot style for reference line
    ref_kwargs = dict(ls="--", c="gray", lw=0.8)
    return ref, ref_kwargs, ref_year


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Tropical SST over time
    """)
    return


@app.cell
def _(forced_2, plt, ref, ref_kwargs, ref_year):
    sel_3 = lambda x: x.mean("month")
    # sel = lambda x : x.sel(month=12)

    _fig, _ax = plt.subplots(figsize=(4, 3))
    # ax.plot(forced.year, sel(forced["trop_sst_15"]), label="tropical SST", c="k")
    _ax.plot(forced_2.year, sel_3(ref), label="tropical SST", c="k")
    _ax.plot(forced_2.year, sel_3(forced_2["T_4"]), label="Niño 4", c="r")
    _ax.plot(forced_2.year, sel_3(forced_2["T_3"]), label="Niño 3", c="b")
    _ax.axvline(ref_year, **ref_kwargs)
    _ax.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Mean state quantities, plotted against tropical SST
    """)
    return


@app.cell
def _(dTdz_n3_1, forced_2, get_nino3_south, plt, ref, ref_kwargs, ref_year):
    sel_4 = lambda x: x.sel(month=[1]).mean("month")
    # sel = lambda x : x.mean("month")

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(forced_2.year, sel_4(forced_2["T_4"] - forced_2["T_3"]))
    # ax.plot(sel(ref), sel(forced["T_4"] - forced["T_3"]))
    # ax.axvline(sel(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.axvline(ref_year, **ref_kwargs)
    _ax.set_title(r"$\frac{dT}{dx}$")
    plt.show()

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(forced_2.year, 8.6e4 * sel_4(forced_2["pr_n"] - forced_2["pr_s"]))
    # ax.plot(sel(ref), 8.6e4 * sel(forced["pr_n"] - forced["pr_s"]))
    # ax.axvline(sel(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.set_title(r"$\frac{dp}{dy}$")
    plt.show()

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(sel_4(ref), -2e2 * sel_4(get_nino3_south(forced_2["tauy"])))
    _ax.axvline(sel_4(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.set_title(r"$\tau_y$")
    plt.show()


    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(sel_4(ref), 8.6e4 * sel_4(forced_2["pr_n3"]))
    _ax.axvline(sel_4(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.set_title(r"precip")
    plt.show()

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(forced_2.year, sel_4(forced_2["w_n3"]))
    # ax.plot(sel(ref), sel(forced["w_n3"]))
    # ax.axvline(sel(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.axvline(ref_year, **ref_kwargs)
    _ax.set_title(r"$w$")
    plt.show()

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(forced_2.year, sel_4(dTdz_n3_1))
    # ax.plot(sel(ref), sel(dTdz_n3))
    # ax.axvline(sel(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.axvline(ref_year, **ref_kwargs)
    _ax.set_title(r"$\frac{dT}{dz}$")
    plt.show()


    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(sel_4(ref), sel_4(forced_2["T_3"] - forced_2["trop_sst_15"]))
    _ax.axvline(sel_4(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.set_title(r"Niño 3 relative SST")
    # ax.set_ylim([0,None])
    plt.show()

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(forced_2.year, sel_4(forced_2["trop_sst_15"] - forced_2["trop_sst_05"]))
    # ax.plot(sel(ref), sel(forced["trop_sst_15"] - forced["trop_sst_05"]))
    # ax.axvline(sel(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.set_title(r"trop sst (15 minus 5)")
    _ax.axvline(ref_year, **ref_kwargs)
    # ax.set_ylim([0,None])

    plt.gca()
    return (sel_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Variance over time
    """)
    return


@app.cell
def _(Th_delta_sigma, cmocean, plt, ref, ref_kwargs, ref_year, sel_4):
    ## specify which variable to plot
    # VARNAME = "eli_30"
    VARNAME = "T_3"

    ## make colormap
    _CMAP = cmocean.cm.amp
    _CMAP_NORM = plt.Normalize(vmin=-1, vmax=3)

    #### plot variance over time
    _fig, _ax = plt.subplots(figsize=(4, 3))
    for _i, _m_ in enumerate([1, 4, 7, 10]):
        sel__3 = lambda x: x.sel(month=_m_)
        _ax.plot(
            sel_4(ref),
            sel__3(Th_delta_sigma[VARNAME]),
            c=_CMAP(_CMAP_NORM(_i)),
            label=f"Month {_m_}",
        )


    _ax.axvline(sel_4(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.set_title(r"Niño variance")
    # ax.set_ylim([0,None])
    plt.show()

    _fig, _ax = plt.subplots(figsize=(4, 3))
    for _i, _m_ in enumerate([1, 4, 7, 10]):
        sel__3 = lambda x: x.sel(month=_m_)
        _ax.plot(
            Th_delta_sigma.year,
            sel__3(Th_delta_sigma[VARNAME]),
            c=_CMAP(_CMAP_NORM(_i)),
            label=f"Month {_m_}",
        )


    # ax.axvline(sel(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.set_title(r"Niño variance")
    _ax.axvline(ref_year, **ref_kwargs)
    _ax.axvline(2030, **ref_kwargs)
    # ax.set_ylim([0,None])
    _ax.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PDFs
    """)
    return


@app.cell
def _(Th_total_1, np, plt, src):
    ## specify varname
    # VARNAME="eli_30"
    VARNAME_1 = "eli_30"

    ## get data
    _x = src.utils.sel_month(Th_total_1[VARNAME_1], [12]).stack(s=["member", "time"])


    ## compute PDFs
    edges = np.arange(120, 285, 2)
    pdf0, _ = src.utils.get_empirical_pdf(_x.sel(year=1870), edges=edges)
    pdf1, _ = src.utils.get_empirical_pdf(_x.sel(year=2085), edges=edges)

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.stairs(pdf0, edges)
    _ax.stairs(pdf1, edges)
    _ax.set_xlim([180, 215])
    plt.gca()
    return


@app.cell
def _(Th_total_1, np, plt, src):
    ## specify varname
    # VARNAME="eli_30"
    VARNAME_2 = "pr_n3"

    ## get data
    _x = src.utils.sel_month(Th_total_1[VARNAME_2], 12).stack(s=["member", "time"])


    ## compute PDFs
    edges_1 = np.arange(0, 30, 0.5)
    pdf0_1, _ = src.utils.get_empirical_pdf(8.6e4 * _x.sel(year=1870), edges=edges_1)
    pdf1_1, _ = src.utils.get_empirical_pdf(8.6e4 * _x.sel(year=2085), edges=edges_1)

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.stairs(pdf0_1, edges_1)
    _ax.stairs(pdf1_1, edges_1)
    _ax.set_xlim([0, 15])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Skewness over time
    """)
    return


@app.cell
def _(Th_delta_skew, cmocean, plt, ref_kwargs):
    ## specify which variable to plot
    VARNAME_3 = "eli_30"

    ## make colormap
    _CMAP = cmocean.cm.amp
    _CMAP_NORM = plt.Normalize(vmin=-1, vmax=3)

    #### plot variance over time
    _fig, _ax = plt.subplots(figsize=(4, 3))
    for _i, _m_ in enumerate([1, 4, 7, 10]):
        sel__4 = lambda x: x.sel(month=_m_)
        _ax.plot(
            # sel(ref),
            Th_delta_skew.year,
            sel__4(Th_delta_skew[VARNAME_3]),
            c=_CMAP(_CMAP_NORM(_i)),
            label=f"Month {_m_}",
        )


    # ax.axvline(sel(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    _ax.axhline(0, **ref_kwargs)
    _ax.set_title(r"Niño skewness")

    _ax.legend()

    # ax.set_ylim([0,None])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Quantiles over time
    """)
    return


@app.cell
def _(plt, quant_1):
    ## specify which variable to plot
    VARNAME_4 = "pr_n3"
    _SEASON = "JJA"
    sel__5 = lambda x: x[VARNAME_4].sel(season=_SEASON)
    # diff_ = lambda x : x-x.isel(year=0)
    # prep = lambda x : diff_(sel_(x))
    _prep = lambda x: sel__5(x)

    ## specify quantile to plot
    Q = 0.95

    #### plot variance over time
    _fig, _ax = plt.subplots(figsize=(4, 3))

    ## warm
    _ax.plot(
        # sel(ref),
        quant_1.year,
        _prep(quant_1).sel(q=Q),
        c="r",
    )

    ## median
    _ax.plot(
        # sel(ref),
        quant_1.year,
        _prep(quant_1).sel(
            q=0.5,
        ),
        c="k",
    )

    ## cold
    _ax.plot(
        # sel(ref),
        quant_1.year,
        _prep(quant_1).sel(q=1 - Q, method="nearest"),
        c="b",
    )

    # ax.axhline(0, **ref_kwargs)
    _ax.set_title(r"Niño quantiles")

    # ax.legend()

    # ax.set_ylim([0,None])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Quantile spread over time
    """)
    return


@app.cell
def _(cmocean, delta_spread_1, plt, quant_1):
    ## specify which variable to plot
    VARNAME_5 = "pr_n3"
    _SEASON = "DJF"

    ## make colormap
    _CMAP = cmocean.cm.amp
    _CMAP_NORM = plt.Normalize(vmin=-1, vmax=3)

    #### plot variance over time
    _fig, _ax = plt.subplots(figsize=(4, 3))
    for _i, _m_ in enumerate(quant_1.season):
        # sel_ = lambda x: x.sel(season=m_, q=.95) - x.sel(season=m_, q=.05)
        sel__6 = lambda x: x.sel(season=_m_)
        _ax.plot(
            # sel(ref),
            quant_1.year,
            sel__6(delta_spread_1[VARNAME_5]),
            c=_CMAP(_CMAP_NORM(_i)),
            label=f"{_m_.values.item()}",
        )


    # ax.axvline(sel(ref.sel(year=ref_year)).values.item(), **ref_kwargs)
    # ax.axhline(0, **ref_kwargs)
    _ax.set_title(r"Niño quantiles")

    _ax.legend()

    # ax.set_ylim([0,None])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### lower or upper width by season
    """)
    return


@app.cell
def _(cmocean, plt, quant_1):
    ## specify which variable to plot
    VARNAME_6 = "pr_n3"

    ## compute upper and lower width
    width_lower = quant_1.sel(q=0.5) - quant_1.sel(q=0.05)
    width_upper = quant_1.sel(q=0.95) - quant_1.sel(q=0.5)

    ## make colormap
    _CMAP = cmocean.cm.amp
    _CMAP_NORM = plt.Normalize(vmin=-1, vmax=3)

    #### plot variance over time
    _fig, _ax = plt.subplots(figsize=(4, 3))
    for _i, _m_ in enumerate(quant_1.season):
        _ax.plot(
            # sel(ref),
            quant_1.year,
            width_lower[VARNAME_6].sel(season=_m_),
            c=_CMAP(_CMAP_NORM(_i)),
            label=f"{_m_.values.item()}",
        )


    _ax.set_title(r"Niño quantiles")

    _ax.legend()

    # ax.set_ylim([0,None])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### warm vs. cold
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute PDF widths
    """)
    return


@app.cell
def _(quant_1):
    ## specify which variable to plot
    VARNAME_7 = "T_34"
    # VARNAME = "pr_n3"
    # VARNAME = "eli_30"
    _SEASON = "DJF"

    ## specify width bound
    wb = 0.05

    ## get func to select data
    sel__7 = lambda x: x[VARNAME_7].sel(season=_SEASON)

    ## compute upper and lower width
    width_lower_1 = quant_1.sel(q=0.5) - quant_1.sel(q=wb)
    width_upper_1 = quant_1.sel(q=1 - wb) - quant_1.sel(q=0.5)
    width_mean = 0.5 * (width_lower_1 + width_upper_1)
    return VARNAME_7, sel__7, width_lower_1, width_mean, width_upper_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute PDFs
    """)
    return


@app.cell
def _(Th, VARNAME_7, np, src):
    ## get data and subtract median
    _x = src.utils.sel_month(Th[VARNAME_7].resample({"time": "QS-DEC"}).mean(), 12)
    _x = _x.stack(s=["member", "time"])
    _x = _x - _x.median("s")

    ## compute PDFs
    edges_2 = np.linspace(-4.5, 4.5, 20)
    pdf0_2, _ = src.utils.get_empirical_pdf(_x.sel(year=1870).values, edges=edges_2)
    pdf1_2, _ = src.utils.get_empirical_pdf(_x.sel(year=2010).values, edges=edges_2)
    pdf2, _ = src.utils.get_empirical_pdf(_x.sel(year=2080).values, edges=edges_2)
    return edges_2, pdf0_2, pdf1_2, pdf2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot
    """)
    return


@app.cell
def _(
    edges_2,
    pdf0_2,
    pdf1_2,
    pdf2,
    plt,
    sel__7,
    width_lower_1,
    width_mean,
    width_upper_1,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(6.5, 3), layout="constrained")

    ## plot PDF
    _axs[0].stairs(pdf0_2, edges_2, fill=True, alpha=0.1, color="k", label=1870)
    _axs[0].stairs(pdf1_2, edges_2, lw=2, label=2010)
    _axs[0].stairs(pdf2, edges_2, lw=2, label=2080)
    # axs[0].set_xlim([-4.5, 4.3])
    _axs[0].axvline(ls="--", c="gray", lw=0.8)
    _axs[0].legend()
    _axs[0].set_xlabel(r"${\circ}$C")
    _axs[0].set_ylabel("Density")
    _axs[0].set_yticks([])


    ## plot width of distribution
    _axs[1].plot(width_lower_1.year, sel__7(width_lower_1), c="blue", label="Cold")
    _axs[1].plot(width_lower_1.year, sel__7(width_upper_1), c="r", label="Warm")
    _axs[1].plot(width_lower_1.year, sel__7(width_mean), c="k", label="Avg.")


    # ax.set_title(r"Distribution width (95% minus median)")
    _axs[1].set_ylabel(r"$^{\circ}$C")
    _axs[1].set_xlabel(r"Year")
    _axs[1].set_yticks([2, 2.5, 3])
    _axs[1].set_xticks([1870, 2010, 2080])
    _axs[1].axvline(2010, ls="--", lw=0.8, c="gray")
    _axs[1].legend()
    _axs[1].yaxis.tick_right()
    _axs[1].yaxis.set_label_position("right")

    ## save
    # save(fig, "pdf-over-time", dpi=300)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Second version of quantile plot
    """)
    return


@app.cell
def _(np):
    def get_intensity(x, get_max=True, q=0.9):
        """Get average intensity of extreme events"""

        ## compute median
        median = x.median(["time", "member"])

        ## switch sign if necessary
        if not get_max:
            x = -x
            median = -median

        ## get cutoff
        thresh = x.quantile(q, dim="time")

        ## get intensity of extreme events
        intensity_raw = x.where(x > thresh, other=np.nan).mean("time")

        return intensity_raw - median

    return (get_intensity,)


@app.cell
def _():
    # def get_ddx(x):
    #     """compute horizontal gradient"""

    #     ## ranges
    #     lat_range = dict(latitude=slice(-5,5))
    #     lon_range_w = dict(longitude=slice(120,160))
    #     lon_range_e = dict(longitude=slice(210,270))

    #     ## spatial avg
    #     avg = lambda x : x.mean(["latitude","longitude"])

    #     return avg(x.sel(**lon_range_w, **lat_range)) - avg(x.sel(**lon_range_e, **lat_range))

    # ## load data
    # _, anom = src.utils.load_consolidated()

    # ## subset for temperature
    # sst = anom[["sst", "sst_comp"]].sel(time=slice("1851", None))

    # ## get windowed T
    # sst = src.utils.get_windowed(sst)

    # ## compute grad
    # dTdx = src.utils.reconstruct_wrapper(sst, get_ddx)["sst"]

    # ## get DJF
    # dTdx_djf = src.utils.sel_month(
    #     dTdx.resample({"time":"QS-DEC"}).mean(), 12
    # )

    # ## get intensity
    # dTdx_max = get_intensity(dTdx_djf, get_max=True, q=0.9)
    # dTdx_min = get_intensity(dTdx_djf, get_max=False, q=0.9)
    return


@app.cell
def _(Th, get_intensity, plt, src):
    ## specify cutoff
    q = 0.9

    ## specify variable
    VARNAME_8 = "T_34"

    ## specify data
    Th_djf = src.utils.sel_month(Th.resample({"time": "QS-DEC"}).mean(), 12)

    ## shared args
    _kwargs = dict(x=Th_djf[VARNAME_8], q=q)

    ## compute widths
    width_upper_2 = get_intensity(get_max=True, **_kwargs)
    width_lower_2 = get_intensity(get_max=False, **_kwargs)
    width_mean_1 = 0.5 * (width_upper_2 + width_lower_2)

    ## how to select data
    mean = lambda x: x.mean("member")
    get_up = lambda x: x.quantile(q=0.9, dim="member")
    get_do = lambda x: x.quantile(q=0.1, dim="member")

    ## get year for convenience
    _year = width_lower_2.year

    ## get plot style for convenience
    bound_kwargs = dict(ls="--", lw=0.8)

    ## plot
    _fig, _axs = plt.subplots(1, 3, figsize=(8, 2.5), layout="constrained")

    ## plot width of distribution
    _axs[0].plot(_year, mean(width_lower_2), c="blue", label="Cold")
    _axs[0].plot(_year, mean(width_upper_2), c="r", label="Warm")
    _axs[0].plot(_year, mean(width_mean_1), c="k", label="Avg.")

    ## plot bounds for warm
    _axs[1].plot(_year, mean(width_mean_1), c="k")
    _axs[1].plot(_year, mean(width_upper_2), c="r")
    _axs[1].plot(_year, get_up(width_upper_2), c="r", **bound_kwargs)
    _axs[1].plot(_year, get_do(width_upper_2), c="r", **bound_kwargs)

    ## plot bounds for cold
    _axs[2].plot(_year, mean(width_mean_1), c="k")
    _axs[2].plot(_year, mean(width_lower_2), c="blue")
    _axs[2].plot(_year, get_up(width_lower_2), c="blue", **bound_kwargs)
    _axs[2].plot(_year, get_do(width_lower_2), c="blue", **bound_kwargs)


    # ax.set_title(r"Distribution width (95% minus median)")
    _axs[0].set_ylabel(r"$^{\circ}$C")
    _axs[0].set_xlabel(r"Year")
    _axs[0].set_yticks([2, 2.5, 3])
    _axs[0].legend()

    for _ax in _axs:
        _ax.set_xticks([1870, 2010, 2080])
        _ax.axvline(2010, ls="--", lw=0.8, c="gray")

    for _ax in _axs[1:]:
        _ax.set_yticks([])

    src.utils.set_lims(_axs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Precip
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot argmax vs. argmax (joint PDF/scatter)
    """)
    return


@app.cell
def _(src, total):
    sel_5 = lambda x: x.sel(latitude=slice(-20, 20), longitude=slice(210, 270)).mean(
        "longitude"
    )

    pr_cross = src.utils.get_windowed(total[["pr", "pr_comp"]], stride=120)
    pr_cross = pr_cross.sel(year=[1870, 1980, 2030, 2080])

    pr_cross = src.utils.reconstruct_wrapper(pr_cross, fn=sel_5)
    pr_cross = pr_cross["pr"]
    return (pr_cross,)


@app.cell
def _(pr_cross):
    ## subset data
    x0 = pr_cross.sel(year=1870)
    x1 = pr_cross.sel(year=1980)

    ## funcs to get quantiles
    get_quantiles = lambda x: x.quantile(
        q=[0.05, 0.25, 0.5, 0.75, 0.95], dim=["member", "time"]
    )
    get_quantiles_bymonth = lambda x: x.groupby("time.month").map(get_quantiles)

    ## compute
    q0 = get_quantiles_bymonth(x0)
    q1 = get_quantiles_bymonth(x1)
    return q0, q1, x0, x1


@app.cell
def _(src, x1):
    import xeofs as xe

    eofs = xe.single.EOF(use_coslat=True, n_modes=10)
    eofs.fit(src.utils.sel_month(x1, 1), dim=["member", "time"])
    return (eofs,)


@app.cell
def _(eofs, x0, x1):
    x0_proj = (eofs.components() * x0).sum("latitude")
    x1_proj = (eofs.components() * x1).sum("latitude")

    # plt.plot(x0_proj.std(["member","time"]))
    # plt.plot(x1_proj.std(["member","time"]))
    return


@app.cell
def _(eofs, plt, ref_kwargs, x0):
    _fig, _ax = plt.subplots(figsize=(3, 2.5))
    _ax.plot(x0.latitude, eofs.components().isel(mode=0))
    _ax.plot(x0.latitude, eofs.components().isel(mode=2))
    _ax.axvline(0, **ref_kwargs)
    plt.gca()
    return


@app.cell
def _(eofs, plt):
    plt.scatter(
        eofs.scores().isel(mode=0),
        eofs.scores().isel(mode=2),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    compute eofs!!
    """)
    return


@app.cell
def _(plt, q0, q1, ref_kwargs):
    _fig, _ax = plt.subplots(figsize=(3, 4))

    # ax.plot(q0.latitude, q0.sel(quantile=.75, month=1))
    _ax.plot(q0.latitude, q0.sel(quantile=0.95, month=1))

    # ax.plot(q0.latitude, q1.sel(quantile=.75, month=1))
    _ax.plot(q0.latitude, q1.sel(quantile=0.95, month=1))

    _ax.axvline(0, **ref_kwargs)
    _ax.set_xlim([-10, 10])
    plt.gca()
    return


@app.cell
def _(np, plt, pr_cross, src):
    ## specify month
    _MONTH = 12

    ## specify which metric to use
    USE_COM = False

    ## get center of mass, and restrict to month/lat range
    sel_6 = lambda x: src.utils.sel_month(x.sel(latitude=slice(-20, 20)), _MONTH)

    x0_ = sel_6(pr_cross.sel(year=2030))
    x1_ = sel_6(pr_cross.sel(year=2080))

    stack = lambda x: x.stack(sample=["time", "member"])

    ## compute
    get_com = lambda x: (x * x.latitude).sum("latitude") / x.sum("latitude")
    com0 = get_com(x0_)
    com1 = get_com(x1_)

    #### lat of max precip
    LAT = x0_.latitude
    max_lat0 = LAT.isel(latitude=x0_.argmax("latitude"))
    max_lat1 = LAT.isel(latitude=x1_.argmax("latitude"))


    ## make pdfs
    if USE_COM:

        edges_3 = np.arange(-10, 11, 1)
        pdf0_3, _ = src.utils.get_empirical_pdf(stack(com0), edges=edges_3)
        pdf1_3, _ = src.utils.get_empirical_pdf(stack(com1), edges=edges_3)

    else:

        edges_3 = np.append(LAT.values - 1.25, LAT.values[-1] + 1.25)
        pdf0_3, _ = src.utils.get_empirical_pdf(stack(max_lat0), edges=edges_3)
        pdf1_3, _ = src.utils.get_empirical_pdf(stack(max_lat1), edges=edges_3)

    _fig, _ax = plt.subplots(figsize=(2.5, 2.5))
    _ax.stairs(pdf0_3, edges_3, fill=True, alpha=0.2)
    _ax.stairs(pdf1_3, edges_3, lw=2)
    _ax.set_xlim([-15, 15])
    plt.gca()
    return com0, com1, stack, x0_, x1_


@app.cell
def _(Th_total_1, com0, com1, plt, src, stack, x0_, x1_):
    idx = dict(time=x1_.time, year=x1_.year)

    _fig, _axs = plt.subplots(1, 2, figsize=(5.5, 2.5), layout="constrained")

    _axs[0].scatter(
        stack(Th_total_1["eli_05"].sel(time=x0_.time, year=x0_.year)),
        stack(com0),
        # stack(max_lat0),
        s=1,
        alpha=0.5,
    )

    _axs[1].scatter(
        stack(Th_total_1["eli_05"].sel(time=x1_.time, year=x1_.year)),
        stack(com1),
        # stack(max_lat1),
        s=1,
        alpha=0.5,
    )
    _axs[1].set_yticks([])

    src.utils.set_lims(_axs)
    plt.gca()
    return


@app.cell
def _(Th_total_1, plt, src, x0_, x1_):
    _fig, _axs = plt.subplots(1, 2, figsize=(5.5, 2.5), layout="constrained")

    _axs[0].scatter(
        Th_total_1["T_3_rel"].sel(time=x1_.time, year=x0_.year),
        Th_total_1["eli_05"].sel(time=x1_.time, year=x0_.year),
        s=3,
        alpha=0.5,
    )

    _axs[1].scatter(
        Th_total_1["T_3_rel"].sel(time=x1_.time, year=x1_.year),
        Th_total_1["eli_05"].sel(time=x1_.time, year=x1_.year),
        s=3,
        alpha=0.5,
    )
    _axs[1].set_yticks([])

    src.utils.set_lims(_axs)
    plt.gca()
    return


@app.cell
def _(Th, Th_total_1, plt, src, x0_, x1_):
    _fig, _axs = plt.subplots(1, 2, figsize=(5.5, 2.5), layout="constrained")

    _axs[0].scatter(
        Th["T_3"].sel(time=x1_.time, year=x0_.year),
        # Th_total["dTdx_v2"].sel(time=x1_.time, year=x0_.year),
        src.utils.separate_forced(Th_total_1["dTdx_v2"])[1].sel(time=x1_.time, year=x0_.year),
        s=3,
        alpha=0.5,
    )

    _axs[0].scatter(
        Th["T_3"].sel(time=x1_.time, year=x1_.year),
        # Th_total["dTdx_v2"].sel(time=x1_.time, year=x1_.year),
        src.utils.separate_forced(Th_total_1["dTdx_v2"])[1].sel(time=x1_.time, year=x1_.year),
        s=3,
        alpha=0.5,
    )

    _axs[1].set_yticks([])

    src.utils.set_lims(_axs)
    plt.gca()
    return


@app.cell
def _(Th_total_1, src):
    dTdx_anom = src.utils.separate_forced(Th_total_1["dTdx_v2"])[1]
    return (dTdx_anom,)


@app.cell
def _(dTdx_anom):
    dTdx_sigma = dTdx_anom.groupby("time.month").std(["member", "time"])
    return (dTdx_sigma,)


@app.cell
def _(dTdx_sigma, plt):
    _f = lambda x: x / x.isel(year=0) - 1
    _f = lambda x: x - x.isel(year=0)
    _f = lambda x: x
    plt.plot(_f(dTdx_sigma.sel(month=[12]).mean("month")))
    plt.plot(_f(dTdx_sigma.sel(month=[1]).mean("month")))
    plt.plot(_f(dTdx_sigma.sel(month=[8]).mean("month")))
    # plt.plot(dTdx_sigma.sel(month=[3]).mean("month"))
    # plt.plot(dTdx_sigma.sel(month=[4]).mean("month"))
    # plt.plot(dTdx_sigma.sel(month=[8]).mean("month"))
    return


@app.cell
def _(Th_total_1, np, plt, src):
    _, _dTdx = src.utils.separate_forced(Th_total_1["dTdx_v2"])
    # dTdx = Th_total["dTdx_v2"]
    _dTdx = _dTdx.resample({"time": "QS-DEC"}).mean()

    ## select data
    _X = _dTdx.sel(year=1870)
    _Y = _dTdx.sel(year=2020)
    _Z = _dTdx.sel(year=2080)

    ## funcs to select data
    sel_7 = lambda x: src.utils.sel_month(x, 12)
    sel__8 = lambda x: sel_7(x).stack(s=["time", "member"])

    ## plot distributions
    _bins = np.linspace(-4.1, 3.5, 20)
    # bins=np.linspace(-2,10,20)
    _pdf_X, _ = src.utils.get_empirical_pdf(sel_7(_X), _bins)
    _pdf_Y, _ = src.utils.get_empirical_pdf(sel_7(_Y), _bins)
    _pdf_Z, _ = src.utils.get_empirical_pdf(sel_7(_Z), _bins)

    _fig, _ax = plt.subplots(figsize=(4, 3.5), layout="constrained")
    _ax.stairs(_pdf_X, _bins)
    _ax.stairs(_pdf_Y, _bins)
    _ax.stairs(_pdf_Z, _bins)
    # ax.set_xlim([-1.5,3.5])
    _ax.axvline(0, c="k", lw=1)
    plt.gca()
    return


@app.cell
def _(Th_total_1, np, plt, src):
    # _, dTdx = src.utils.separate_forced(Th_total["T_3_rel"])
    _dTdx = Th_total_1["T_4_rel"]
    _dTdx = _dTdx.resample({"time": "QS-DEC"}).mean()

    ## select data
    _X = _dTdx.sel(year=1870)
    _Y = _dTdx.sel(year=2020)
    _Z = _dTdx.sel(year=2080)

    ## funcs to select data
    sel_8 = lambda x: src.utils.sel_month(x, 12)
    sel__9 = lambda x: sel_8(x).stack(s=["time", "member"])

    ## plot distributions
    _bins = np.linspace(-5, 2.5, 25)
    # bins=np.linspace(-2,10,20)
    _pdf_X, _ = src.utils.get_empirical_pdf(sel_8(_X), _bins)
    _pdf_Y, _ = src.utils.get_empirical_pdf(sel_8(_Y), _bins)
    _pdf_Z, _ = src.utils.get_empirical_pdf(sel_8(_Z), _bins)

    _fig, _ax = plt.subplots(figsize=(4, 3.5), layout="constrained")
    _ax.stairs(_pdf_X, _bins)
    _ax.stairs(_pdf_Y, _bins)
    _ax.stairs(_pdf_Z, _bins)
    # ax.set_xlim([-1.5,3.5])
    _ax.axvline(0, c="k", lw=1)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try with eofs!!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figs
    """)
    return


@app.cell
def _(sns):
    ## colorbar
    sns.set_palette("colorblind")

    ## specify plot style for reference line
    ref_kwargs_1 = dict(ls="--", c="gray", lw=0.8)
    return (ref_kwargs_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Func to save
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mean state
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Timeseries
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Preprocessing funcs
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

    return (get_H_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Compute
    """)
    return


@app.cell
def _(forced_2, get_H_metrics, src):
    ## specify longitude ranges
    LONS_W = (140, 210)
    LONS_E = (210, 270)

    ## get maximum gradient of thermocline data
    H_mg = src.utils.load_h_data(max_grad=True)[0]
    H_mg = src.utils.get_windowed(H_mg).groupby("time.month").mean()

    ## get metrics
    _kwargs = dict(lons_w=LONS_W, lons_e=LONS_E)
    H_mg_stats = get_H_metrics(H_mg, **_kwargs)
    D20_stats = get_H_metrics(
        forced_2["z20"].sel(latitude=slice(-2, 2)).mean("latitude"), **_kwargs
    )
    return H_mg, H_mg_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, plot fractional change in mean-state quantities
    """)
    return


@app.cell
def _(ref_kwargs_1):
    def format_ax(ax):
        """add custom ticks"""
        ax.set_xticks([1870, 2010, 2085])
        ax.axvline(2010, **ref_kwargs_1)

        return

    return (format_ax,)


@app.cell
def _(H_mg_stats, dTdz_n3_1, forced_2, format_ax, plt, ref_kwargs_1):
    sel__10 = lambda x: x.mean("month")
    sel_9 = lambda x: (sel__10(x) - sel__10(x.isel(year=0))) / sel__10(x.isel(year=0))

    _fig, _ax = plt.subplots(figsize=(3.5, 3), layout="constrained")
    _ax.plot(forced_2.year, sel_9(dTdz_n3_1), label=r"$\Delta_z T$")
    # ax.plot(forced.year, sel(forced["T_4"] - forced["T_3"]), label=r"$\Delta_x T$")
    _ax.plot(forced_2.year, sel_9(forced_2["Tw"] - forced_2["T_3"]), label=r"$\Delta_x T$")
    _ax.plot(forced_2.year, sel_9(H_mg_stats["Hbar"]), label=r"$h$")
    _ax.plot(forced_2.year, sel_9(forced_2["w_n3"]), label=r"$w$")


    ## format/labelling
    _ax.set_yticks([-0.3, 0, 0.6])
    _ax.axhline(0, **ref_kwargs_1)
    format_ax(_ax)
    _ax.legend()
    _ax.set_ylabel("Fractional change")
    _ax.set_title(r"(e) Fractional changes", loc="left")

    ## save
    # save(fig, "mean-state_timeseries")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Second version, with error bounds
    """)
    return


@app.cell
def _(Th_total_1, plt, sns):
    sel2_ = lambda x: x.mean("time")
    get_bl = lambda x: x.isel(year=0).mean("member")
    # sel2 = lambda x: (sel2_(x) - sel2_(get_bl(x))) #/ sel2_(get_bl((x)))
    sel2 = lambda x: sel2_(x)

    ## get variables to plot
    _dTdx = sel2(Th_total_1["Tw"] - Th_total_1["T_3"])
    # dTdx = sel2(Th_total["Tw"]-Th_total["T_34"])

    color = sns.color_palette("colorblind")[2]
    color = "k"

    _fig, _ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    _ax.plot(_dTdx.year, _dTdx.mean("member"), color=color, lw=3)
    _ax.plot(_dTdx.year, _dTdx.quantile(q=0.9, dim="member"), color=color, ls="--", lw=1.5)
    _ax.plot(_dTdx.year, _dTdx.quantile(q=0.1, dim="member"), color=color, ls="--", lw=1.5)
    _ax.axvline(2010, ls="-", c="gray", lw=0.75)
    _ax.axhline(_dTdx.mean("member").isel(year=0), ls="-", c="gray", lw=0.75)
    _ax.set_xticks([1870, 2010, 2080])
    # ax.set_yticks([-1.5, -1, -.5, 0])
    _ax.set_ylabel(r"$^{\circ}$C")
    plt.gca()
    return


@app.cell
def _(ccrs, forced_2, plt, src):
    ## get relative SST
    forced_2["sst_rel"] = forced_2["sst"] - forced_2["trop_sst_15"]

    ## get difference
    x0_1 = forced_2.sel(year=1870)
    x1_1 = forced_2.sel(year=2085)
    dx = x1_1 - x0_1

    sel_10 = lambda x: x.mean("month")
    # sel = lambda x: x.sel(month=1)

    _fig = plt.figure(figsize=(8, 2.4), layout="constrained")
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=30, lon_range=(115, 285))
    _axs = src.utils.subplots_with_proj(_fig, nrows=1, ncols=1, format_func=_format_func)

    ## shared args
    _lonlat = (dx.longitude, dx.latitude)
    _kwargs = dict(extend="both", transform=ccrs.PlateCarree())

    ## plot relative SST
    _cp_sst = _axs[0, 0].contourf(
        *_lonlat,
        sel_10(dx["sst_rel"]),
        levels=src.utils.make_cb_range(2, 0.2),
        cmap="cmo.balance",
        **_kwargs,
    )

    ## relative SST baseline
    _axs[0, 0].contour(
        *_lonlat,
        sel_10(x0_1["sst_rel"]),
        levels=src.utils.make_cb_range(10, 2),
        colors="k",
        linewidths=1.0,
        **_kwargs,
    )
    _axs[0, 0].contour(
        *_lonlat,
        sel_10(x0_1["sst_rel"]),
        levels=[0],
        colors="gray",
        linewidths=1.0,
        **_kwargs,
    )

    ## colorbars
    _fig.colorbar(_cp_sst, label=r"$^{\circ}$C", ticks=[-2, 0, 2], pad=0.02)

    ## empty array to hold grid labels
    _gls = []

    ## boxes
    for _ax in _axs.flatten():
        src.utils.plot_nino3_box(_ax, c="w", lw=0.8)
        src.utils.plot_box(_ax, lons=[120, 160], lats=[-5, 5], c="w", lw=0.8)

        ## gridlines
        _gls.append(
            _ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=1,
                alpha=0,
                xlocs=[120, 160, -150, -90],
                ylocs=[-25, 0, 25],
            )
        )


    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, plot change in Niño indices
    """)
    return


@app.cell
def _(forced_2, format_ax, plt, save):
    sel_11 = lambda x: x.mean("month")
    # sel = lambda x : x.sel(month=12)

    _fig, _ax = plt.subplots(figsize=(3.5, 3), layout="constrained")
    _ax.plot(forced_2.year, sel_11(forced_2["trop_sst_15"]), label=r"Tropical avg.", c="k")
    # ax.plot(forced.year, sel(forced["T_4"]), label=r"$T_w$", c="r")
    _ax.plot(forced_2.year, sel_11(forced_2["Tw"]), label=r"$T_w$", c="r")
    _ax.plot(forced_2.year, sel_11(forced_2["T_3"]), label=r"$T_e$", c="b")

    ## formatting
    _ax.set_yticks([26, 29, 32])
    _ax.set_ylabel(r"$^{\circ}C$")
    format_ax(_ax)
    _ax.legend()
    _ax.set_title(r"(f) Absolute changes", loc="left")

    save(_fig, fname="nino-indices_overtime")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Combine into one plot
    """)
    return


@app.cell
def _(H_mg_stats, dTdz_n3_1, forced_2, format_ax, plt, ref_kwargs_1, save):
    sel__11 = lambda x: x.mean("month")
    sel_12 = lambda x: (sel__11(x) - sel__11(x.isel(year=0))) / sel__11(x.isel(year=0))

    _fig, _axs = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")

    ## fractional changes
    sel__11 = lambda x: x.mean("month")
    sel_12 = lambda x: (sel__11(x) - sel__11(x.isel(year=0))) / sel__11(x.isel(year=0))
    _axs[0].plot(forced_2.year, sel_12(dTdz_n3_1), label=r"$\Delta_z T$")
    _axs[0].plot(forced_2.year, sel_12(forced_2["Tw"] - forced_2["T_3"]), label=r"$\Delta_x T$")
    _axs[0].plot(forced_2.year, sel_12(H_mg_stats["Hbar"]), label=r"$h$")
    _axs[0].plot(forced_2.year, sel_12(forced_2["w_n3"]), label=r"$w$")

    ## other
    _axs[1].plot(forced_2.year, sel__11(forced_2["trop_sst_15"]), label=r"Tropical avg.", c="k")
    _axs[1].plot(forced_2.year, sel__11(forced_2["Tw"]), label=r"$T_w$", c="r")
    _axs[1].plot(forced_2.year, sel__11(forced_2["T_3"]), label=r"$T_e$", c="b")

    ## format/labelling
    _axs[0].set_yticks([-0.3, 0, 0.6])
    _axs[0].axhline(0, **ref_kwargs_1)
    _axs[0].set_ylabel("Fractional change")
    _axs[1].set_yticks([26, 29, 32])
    _axs[1].set_ylabel(r"$^{\circ}C$")
    _axs[1].yaxis.set_label_position("right")
    _axs[1].yaxis.tick_right()

    ## label
    _axs[0].set_title(r"(e) Fractional changes", loc="left")
    _axs[1].set_title(r"(f) Absolute changes", loc="left")

    for _ax in _axs:
        format_ax(_ax)
        _ax.legend()

    ## save
    save(_fig, "mean-state_timeseries_combined")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Spatial pattern
    """)
    return


@app.cell
def _(ccrs, forced_2, plt, save, src):
    ## get relative SST
    forced_2["sst_rel"] = forced_2["sst"] - forced_2["trop_sst_15"]

    ## get difference
    x0_2 = forced_2.sel(year=1870)
    x1_2 = forced_2.sel(year=2085)
    dx_1 = x1_2 - x0_2

    sel_13 = lambda x: x.mean("month")
    # sel = lambda x: x.sel(month=1)

    _fig = plt.figure(figsize=(8, 4.8), layout="constrained")
    _format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=30, lon_range=(115, 285))
    _axs = src.utils.subplots_with_proj(_fig, nrows=2, ncols=1, format_func=_format_func)

    ## shared args
    _lonlat = (dx_1.longitude, dx_1.latitude)
    _kwargs = dict(extend="both", transform=ccrs.PlateCarree())

    ## plot relative SST
    _cp_sst = _axs[0, 0].contourf(
        *_lonlat,
        sel_13(dx_1["sst_rel"]),
        levels=src.utils.make_cb_range(2, 0.2),
        cmap="cmo.balance",
        **_kwargs,
    )

    ## relative SST baseline
    _axs[0, 0].contour(
        *_lonlat,
        sel_13(x0_2["sst_rel"]),
        levels=src.utils.make_cb_range(10, 2),
        colors="k",
        linewidths=0.75,
        **_kwargs,
    )
    _axs[0, 0].contour(
        *_lonlat,
        sel_13(x0_2["sst_rel"]),
        levels=[0],
        colors="gray",
        linewidths=1.25,
        **_kwargs,
    )

    ## plot precip
    cp_pr = _axs[1, 0].contourf(
        *_lonlat,
        8.6e4 * sel_13(dx_1["pr"]),
        levels=src.utils.make_cb_range(5, 0.5),
        cmap="cmo.balance_r",
        **_kwargs,
    )

    ## baseline
    _axs[1, 0].contour(
        *_lonlat,
        8.6e4 * sel_13(x0_2["pr"]),
        levels=[6],
        colors="k",
        linewidths=0.75,
        **_kwargs,
    )

    ## colorbars
    _fig.colorbar(_cp_sst, label=r"$^{\circ}$C", ticks=[-2, 0, 2], pad=0.02)
    _fig.colorbar(cp_pr, label="mm / day", ticks=[-5, 0, 5], pad=0.02)

    ## empty array to hold grid labels
    _gls = []

    ## boxes
    for _ax in _axs.flatten():
        src.utils.plot_nino4_box(_ax, c="magenta", lw=0.8)
        src.utils.plot_nino3_box(_ax, c="magenta", lw=0.8)
        src.utils.plot_nino34_box(_ax, c="w", lw=0.8, ls="--")

        ## gridlines
        _gls.append(
            _ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=1,
                alpha=0,
                xlocs=[120, 160, -150, -90],
                ylocs=[-25, 0, 25],
            )
        )
    _gls[0].bottom_labels = False
    _gls[1].top_labels = False

    ## label
    src.utils.label_subplot(ax=_axs[0, 0], posn=(0.02, 0.05), label="(a)")
    src.utils.label_subplot(ax=_axs[1, 0], posn=(0.02, 0.05), label="(b)")

    ## save
    save(_fig, fname="mean-state-pattern_change_labeled")

    plt.gca()
    return dx_1, sel_13, x0_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Subsurface
    """)
    return


@app.cell
def _(dx_1, np, plt, save, sel_13, src, x0_2):
    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")

    ## temperature
    cp_T = _axs[0].contourf(
        dx_1["T"].longitude,
        dx_1["T"].z_t,
        sel_13(dx_1["T"]),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(5, 0.5),
        extend="both",
    )

    _axs[0].contour(
        x0_2["T"].longitude,
        x0_2["T"].z_t,
        sel_13(x0_2["T"]),
        colors="k",
        levels=np.arange(12, 34, 4),
        linewidths=0.5,
        extend="both",
    )

    ## vertical velocity
    cp_w = _axs[1].contourf(
        dx_1["w"].longitude,
        dx_1["w"].z_t,
        sel_13(dx_1["w"]),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(20, 2),
        extend="both",
    )

    _axs[1].contour(
        x0_2["w"].longitude,
        x0_2["w"].z_t,
        sel_13(x0_2["w"]),
        colors="k",
        levels=np.arange(0, 100, 10),
        linewidths=0.5,
        extend="both",
    )

    ## plot 20 degree isotherm on both panels
    for _ax in _axs:
        _ax.contour(
            x0_2.longitude,
            x0_2.z_t,
            sel_13(x0_2["T"]),
            levels=[20],
            colors="magenta",
            linestyles="--",
        )

    ## formatting
    src.utils.format_subsurf_axs(_axs)
    for _ax in _axs:
        _ax.set_xlim([140, 280])
        _ax.set_xticks([140, 210, 280])
        _ax.xaxis.set_label_position("top")
        _ax.xaxis.tick_top()

    ## colorbars
    _fig.colorbar(cp_T, ticks=[-5, 0, 5], label=r"$^{\circ}$C", orientation="horizontal")
    _fig.colorbar(cp_w, ticks=[-20, 0, 20], label=r"m / month", orientation="horizontal")

    ## label
    src.utils.label_subplot(ax=_axs[0], posn=(0.03, 0.05), label="(c)")
    src.utils.label_subplot(ax=_axs[1], posn=(0.03, 0.05), label="(d)")

    ## save
    save(_fig, fname="subsurface-change")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### warm pool (scratch)
    """)
    return


@app.cell
def _(forced_2, plt):
    merimean_2 = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")
    sel_14 = lambda x: merimean_2(x["taux"]).sel(month=[4]).mean("month")
    # prep = lambda x : sel(x)-sel(x).max()
    _prep = lambda x: sel_14(x)

    _fig, _axs = plt.subplots(1, 2, figsize=(5.5, 2.5), layout="constrained")

    for _year in [1870, 2080]:
        _axs[0].plot(
            forced_2.longitude,
            _prep(forced_2.sel(year=_year)),
        )

    for _ax in _axs:
        _ax.set_xlim([140, 280])
        _ax.set_ylim([-0.06, 0.01])
    plt.gca()
    return (merimean_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get equatorial precip
    """)
    return


@app.cell
def _(merimean_2, src, total):
    pr_windowed = src.utils.get_windowed(total[["pr", "pr_comp"]]).sel(year=[1870, 2080])

    pr_eq = src.utils.reconstruct_wrapper(pr_windowed, fn=merimean_2)
    return (pr_eq,)


@app.cell
def _(src):
    sel_15 = lambda x: src.utils.sel_month(x, months=[2, 3, 4]).groupby("time.year").mean()
    return (sel_15,)


@app.cell
def _(pr_eq, src):
    pr_eq_mam0 = (
        src.utils.sel_month(pr_eq.isel(year=0), months=[2, 3, 4])
        .groupby("time.year")
        .mean()
        .stack(sample=["year", "member"])["pr"]
    )
    pr_eq_mam1 = (
        src.utils.sel_month(pr_eq.isel(year=1), months=[2, 3, 4])
        .groupby("time.year")
        .mean()
        .stack(sample=["year", "member"])["pr"]
    )
    return pr_eq_mam0, pr_eq_mam1


@app.cell
def _(pr_eq_mam0):
    pr_eq_mam0
    return


@app.cell
def _(plt, pr_eq, pr_eq_mam0):
    _fig, _axs = plt.subplots(1, 2, figsize=(6.5, 3))

    for s in pr_eq_mam0.sample:
        _axs[0].plot(pr_eq.longitude, pr_eq_mam0.sel(sample=s), c="gray", lw=0.5, alpha=0.2)

    _axs[0].set_xlim([150, 270])
    plt.gca()
    return


@app.cell
def _(pr_eq_mam0, pr_eq_mam1):
    lon_idx_max = pr_eq_mam0.sel(longitude=slice(150, 270)).argmax("longitude")
    lon = pr_eq_mam0.sel(longitude=slice(150, 270)).longitude.isel(longitude=lon_idx_max)

    lon_idx_max1 = pr_eq_mam1.sel(longitude=slice(150, 270)).argmax("longitude")
    lon1 = pr_eq_mam1.sel(longitude=slice(150, 270)).longitude.isel(longitude=lon_idx_max1)
    return lon, lon1


@app.cell
def _(lon, lon1, plt):
    plt.hist(lon)
    plt.hist(lon1, alpha=0.5)
    return


@app.cell
def _(pr_eq, sel_15):
    pr_eq_mam = sel_15(pr_eq).stack(sample=["year", "member"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Heat flux
    """)
    return


@app.cell
def _():
    # ## func to select data
    # sel = lambda x : src.utils.sel_month(x, 12)

    # fig, axs = plt.subplots(1,2,figsize=(5.5,2.5), layout="constrained")

    # for ax, y_ in zip(axs, [1870,2080]):
    #     ax.scatter(
    #         sel(Th["T_3"].sel(year=y_)),
    #         sel(Th["T_3"].sel(year=y_)),
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variance
    **to-do: plot bounds**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute Niño 1+2 indices
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Time series
    """)
    return


@app.cell
def _(Th, anom_, src):
    def get_nino12(x):
        return x.sel(latitude=slice(0, 10), longitude=slice(270, 280)).mean(
            ["latitude", "longitude"]
        )


    ## get nino12
    nino12 = src.utils.reconstruct_wrapper(
        anom_[["sst", "sst_comp"]],
        fn=get_nino12,
    )["sst"]

    ## window it, and add to data
    Th["T_12"] = src.utils.get_windowed(nino12)
    return


@app.cell
def _(Th):
    ## get variance by month
    _Th_sigma = Th.groupby("time.season").std(["time", "member"])

    ## fractional change
    Th_delta_sigma_1 = (_Th_sigma - _Th_sigma.isel(year=0)) / _Th_sigma.isel(year=0)
    return (Th_delta_sigma_1,)


@app.cell
def _(Th_delta_sigma_1, cmocean, plt, ref_kwargs_1, sns, src):
    ## specify which variables to plot
    _VARNAMES = ["T_4", "T_34", "T_3", "T_12"]

    ## specify labels
    LABELS = ["Niño 4", "Niño 3.4", "Niño 3", "Niño 1+2"]

    ## specify months to plot
    SEASONS = ["DJF", "MAM", "SON"]

    ## get colors
    COLORS = [sns.color_palette()[i] for i in [0, 2, 1]]

    ## make colormap
    _CMAP = cmocean.cm.amp
    _CMAP_NORM = plt.Normalize(vmin=-1, vmax=2)

    _fig, _axs = plt.subplots(1, 4, figsize=(8.5, 2), layout="constrained")

    for _ax, _v, l in zip(_axs, _VARNAMES, LABELS):

        ## plot data
        for _season, _c in zip(SEASONS, COLORS):
            sel__12 = lambda x: x.sel(season=_season)
            _ax.plot(
                Th_delta_sigma_1.year,
                Th_delta_sigma_1[_v].sel(season=_season),
                c=_c,
                label=_season,
                lw=2,
            )

        ## add reference lines / label
        _ax.axvline(2030, **ref_kwargs_1)
        _ax.axhline(0, **ref_kwargs_1)
        _ax.set_yticks([-0.3, 0, 0.3])
        _ax.set_xticks([2030])
        _ax.set_title(l)


    ## label
    _axs[0].set_ylabel("Fractional change")
    for _ax in _axs[1:]:
        _ax.set_yticks([])

    # ax.set_ylim([0,None])
    _axs[-1].legend(loc=(1, 0.25))
    src.utils.set_lims(_axs)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hovmollers (by longitude)
    """)
    return


@app.cell
def _(cartopy, np, scipy, src, xr):
    def get_sigma_bygroup(data, grouper, varname="sst"):
        """get variance after grouping data"""

        var_bygroup = data.groupby(grouper).map(
            src.utils.reconstruct_var_wrapper,
        )

        return np.sqrt(var_bygroup)


    def get_sigma_byseason(data, varname="sst"):
        """Get variance by month"""

        return get_sigma_bygroup(data, grouper="time.season", varname=varname)


    def get_sigma_bymonth(data, varname="sst"):
        """Get variance by month"""

        return get_sigma_bygroup(data, grouper="time.month", varname=varname)


    def skew_helper(data):
        """get skew for data"""

        ## specify kwargs
        kwargs = dict(input_core_dims=[["sample"]], kwargs=dict(axis=-1))

        return xr.apply_ufunc(
            scipy.stats.skew, data.stack(sample=["time", "member"]), **kwargs
        )


    def get_skew_byseason(data):
        """compute skewness for equatorial strip by month"""

        ## get data on equator
        data_eq = src.utils.reconstruct_wrapper(
            data=data,
            fn=src.utils.merimean,
        )
        ## compute skewness
        return data_eq.groupby("time.season").map(skew_helper)


    def get_skew_bymonth(data):
        """compute skewness for equatorial strip by month"""

        ## get data on equator
        data_eq = src.utils.reconstruct_wrapper(
            data=data,
            fn=src.utils.merimean,
        )
        ## compute skewness
        return data_eq.groupby("time.month").map(skew_helper)


    def contour_cyclic(ax, data, **kwargs):
        """plot data on hovmoller with cyclic dependence on month"""

        ## add cyclic point
        data_cyclic, month = cartopy.util.add_cyclic_point(
            data.transpose("month", ...), data.month, axis=0
        )

        ## plot data
        cp = ax.contour(data.longitude, month, data_cyclic, **kwargs)

        return cp

    return (
        contour_cyclic,
        get_sigma_bymonth,
        get_sigma_byseason,
        get_skew_bymonth,
        get_skew_byseason,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Get windowed anomalies
    """)
    return


@app.cell
def _(anom_, src):
    anom_windowed = src.utils.get_windowed(anom_[["sst", "sst_comp"]])
    return (anom_windowed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Compute variance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By season
    """)
    return


@app.cell
def _(anom_windowed, get_sigma_byseason, pd, tqdm, xr):
    ## years to compute for
    _YEARS = anom_windowed.year

    ## empty list to hold result
    sigma_spatial_byseason = []

    for _year in tqdm.tqdm(_YEARS):
        sigma_spatial_byseason.append(get_sigma_byseason(anom_windowed.sel(year=_year)))

    ## convert back to XR
    sigma_spatial_byseason = xr.concat(
        sigma_spatial_byseason, dim=pd.Index(_YEARS, name="year")
    )

    ## Equatorial avg
    sigma_spatial_eq = sigma_spatial_byseason.sel(latitude=slice(-5, 5)).mean("latitude")
    delta_sigma_spatial_eq = sigma_spatial_eq - sigma_spatial_eq.isel(year=0)
    return (sigma_spatial_eq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By month
    """)
    return


@app.cell
def _(anom_windowed, get_sigma_bymonth, pd, tqdm, xr):
    # years to compute for
    _YEARS = [1870, 2010, 2080]

    ## empty list to hold result
    sigma_spatial_bymonth = []

    for _year in tqdm.tqdm(_YEARS):
        sigma_spatial_bymonth.append(get_sigma_bymonth(anom_windowed.sel(year=_year)))

    ## convert back to XR
    sigma_spatial_bymonth = xr.concat(
        sigma_spatial_bymonth, dim=pd.Index(_YEARS, name="year")
    )

    ## equatorial avg
    sigma_spatial_bymonth_eq = sigma_spatial_bymonth.sel(latitude=slice(-5, 5)).mean(
        "latitude"
    )
    return (sigma_spatial_bymonth_eq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute skewness
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By season
    """)
    return


@app.cell
def _(anom_windowed, get_skew_byseason, pd, tqdm, xr):
    ## years to compute for
    _YEARS = anom_windowed.year

    ## empty list to hold result
    skew_spatial_byseason = []

    for _year in tqdm.tqdm(_YEARS):
        skew_spatial_byseason.append(get_skew_byseason(anom_windowed.sel(year=_year)))

    ## convert back to XR
    skew_spatial_byseason = xr.concat(
        skew_spatial_byseason, dim=pd.Index(_YEARS, name="year")
    )

    ## change in skewness
    delta_skew = skew_spatial_byseason - skew_spatial_byseason.isel(year=0)
    return delta_skew, skew_spatial_byseason


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By month
    """)
    return


@app.cell
def _(anom_windowed, get_skew_bymonth, pd, tqdm, xr):
    # years to compute for
    _YEARS = [1870, 2010, 2080]

    ## empty list to hold result
    skew_spatial_bymonth = []

    for _year in tqdm.tqdm(_YEARS):
        skew_spatial_bymonth.append(get_skew_bymonth(anom_windowed.sel(year=_year)))

    ## convert back to XR
    skew_spatial_bymonth = xr.concat(skew_spatial_bymonth, dim=pd.Index(_YEARS, name="year"))
    return (skew_spatial_bymonth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### merge data
    """)
    return


@app.cell
def _(
    sigma_spatial_bymonth_eq,
    sigma_spatial_eq,
    skew_spatial_bymonth,
    skew_spatial_byseason,
    xr,
):
    stats_bymonth = xr.merge(
        [
            sigma_spatial_bymonth_eq.rename({"sst": "sigma"}),
            skew_spatial_bymonth.rename({"sst": "skew"}),
        ]
    )
    stats_byseason = xr.merge(
        [
            sigma_spatial_eq.rename({"sst": "sigma"}),
            skew_spatial_byseason.rename({"sst": "skew"}),
        ]
    )
    return stats_bymonth, stats_byseason


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### get differences
    """)
    return


@app.cell
def _(stats_bymonth, stats_byseason):
    delta_stats_byseason = stats_byseason - stats_byseason.isel(year=0)
    deltanorm_stats_byseason = delta_stats_byseason / stats_byseason.isel(year=0)
    delta_stats_bymonth = stats_bymonth - stats_bymonth.isel(year=0)
    return delta_stats_bymonth, delta_stats_byseason, deltanorm_stats_byseason


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Variance over time
    """)
    return


@app.cell
def _(
    delta_stats_byseason,
    deltanorm_stats_byseason,
    plt,
    ref_kwargs_1,
    save,
    src,
):
    _fig, _axs = plt.subplots(1, 4, figsize=(8, 3), layout="constrained")

    for _ax, _season, let in zip(_axs, ["DJF", "MAM", "JJA", "SON"], ["a", "b", "c", "d"]):
        _p = _ax.contourf(
            deltanorm_stats_byseason.longitude,
            deltanorm_stats_byseason.year,
            delta_stats_byseason["sigma"].sel(season=_season),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(0.4, 0.04),
        )
        _ax.set_xlim([140, 280])
        _ax.set_xticks([150, 210, 270])
        _ax.set_title(f"({let}) {_season}", loc="left")
        _ax.axhline(2030, **ref_kwargs_1)
        _ax.axvline(210, **ref_kwargs_1)
        _ax.set_xlabel("Lon.")


    ## label/format
    _fig.colorbar(_p, ax=_axs[-1], ticks=[-0.4, 0, 0.4], label="Frac. change")
    _axs[0].set_yticks([1870, 2030, 2080])
    for _ax in _axs[1:]:
        _ax.set_yticks([])

    save(_fig, "sigma-over-time_hov_simple")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Variance by season (v1)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Variance over time (v2)
    """)
    return


@app.cell
def _(
    contour_cyclic,
    delta_stats_bymonth,
    np,
    plt,
    save,
    skew_spatial_bymonth,
    src,
    stats_bymonth,
):
    # cp0 = src.utils.make_cycle_hov(axs[0], data=m_early, **kwargs)

    ## get plot data
    # baseline = sigma_spatial_bymonth_eq["sst"].sel(year=1870)
    # future = sigma_spatial_bymonth_eq["sst"].sel(year=2080)
    # change = future - baseline

    baseline = stats_bymonth.sel(year=1870)
    future = stats_bymonth.sel(year=2080)
    change = delta_stats_bymonth.sel(year=2080)

    ## for skew
    baseline_sk = skew_spatial_bymonth["sst"].sel(year=1870)
    future_sk = skew_spatial_bymonth["sst"].sel(year=2080)
    change_sk = future_sk - baseline_sk

    ## specify levels
    LEV = np.arange(0.4, 2.2, 0.2)
    LEV_DIFF = src.utils.make_cb_range(0.5, 0.05)

    ## shared args for plotting
    _plot_kwargs = dict(cmap="cmo.amp", extend="min", xticks=[210, 270])

    ## Set up plot
    _fig, _axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

    ## make hövmöllers
    # cp0 = src.utils.make_cycle_hov(axs[0], data=baseline["sigma"])
    _cp0 = src.utils.plot_cycle_hov(_axs[0], baseline["sigma"], levels=LEV, **_plot_kwargs)
    cp1 = src.utils.plot_cycle_hov(_axs[1], future["sigma"], levels=LEV, **_plot_kwargs)
    _cp2 = src.utils.plot_cycle_hov(
        _axs[2],
        change["sigma"],
        levels=LEV_DIFF,
        cmap="cmo.balance",
        extend="both",
        xticks=[210, 270],
    )

    ## superimpose skewness
    for _ax, _x, _ls in zip(_axs[:-1], [baseline, future], ["--", "-"]):
        for ax_ in [_ax, _axs[-1]]:
            contour_cyclic(
                ax=ax_,
                data=_x["skew"],
                levels=[-0.5],
                colors="k",
                zorder=10,
                linewidths=0.8,
                linestyles=_ls,
                alpha=0.8,
            )

    ## label
    _axs[0].set_title("(a) 1870", loc="left")
    _axs[1].set_title("(b) 2080", loc="left")
    _axs[2].set_title("(c) Difference", loc="left")
    _axs[1].set_xlabel("Longitude")
    # axs[-1].set_xticks([140, 190, 240])

    for _ax in _axs:
        _ax.set_xlim([140, 280])
        _ax.set_xticks([140, 210, 270])

    for _ax in _axs[1:]:
        _ax.set_yticks([])
        _ax.set_ylabel(None)

    ## add colorbars
    _kwargs = dict(ticks=[0.4, 1.2, 2], label=r"$^{\circ}\text{C}$")
    # cb0 = fig.colorbar(cp0, ax=axs[0], **kwargs)
    cb1 = _fig.colorbar(cp1, ax=_axs[1], **_kwargs)
    cb2 = _fig.colorbar(_cp2, ax=_axs[2], **dict(_kwargs, ticks=[-0.5, 0, 0.5]))

    ## save
    save(_fig, fname="sigma-cycle-change")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot skewness over time
    """)
    return


@app.cell
def _(delta_skew, plt, ref_kwargs_1, skew_spatial_byseason, src):
    _fig, _axs = plt.subplots(1, 4, figsize=(8, 3), layout="constrained")

    for _ax, _season in zip(_axs, ["DJF", "MAM", "JJA", "SON"]):
        _p = _ax.contourf(
            delta_skew.longitude,
            delta_skew.year,
            delta_skew["sst"].sel(season=_season),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(0.75, 0.075),
            extend="both",
        )
        _ax.set_xlim([140, 280])
        _ax.set_xticks([150, 270])
        _ax.set_title(_season)
        _ax.axhline(2030, **ref_kwargs_1)
        _ax.axvline(210, **ref_kwargs_1)
        _ax.set_xlabel("Lon.")

        ## plot climatological
        ax_cyc = _ax.twinx()
        for y_, c_, _ls in zip([1870, 2030, 2080], ["gray", "gray", "k"], ["-", "--", "-"]):
            ax_cyc.plot(
                skew_spatial_byseason.longitude,
                skew_spatial_byseason["sst"].sel(season=_season, year=y_),
                c=c_,
                ls=_ls,
                label=y_,
            )
        ax_cyc.set_ylim([-1.6, 4])
        ax_cyc.axhline(0, c="k", lw=0.8, alpha=0.5)

        ## labels to variance
        if _season == "DJF":
            ax_cyc.yaxis.set_label_position("left")
            ax_cyc.yaxis.tick_left()
            ax_cyc.set_yticks([-1.5, 0.0])
            ax_cyc.text(x=-0.7, y=0.14, s=r"Skew", color="gray", transform=ax_cyc.transAxes)
            _ax.text(x=-0.7, y=0.85, s=r"Year", color="k", transform=ax_cyc.transAxes)
            ax_cyc.tick_params(axis="y", colors="gray")
        else:
            ax_cyc.set_yticks([])


    ## label/format
    _fig.colorbar(_p, ax=_axs[-1], ticks=[-1, 0, 1], label=r"Skewness change")
    _axs[0].set_yticks([2030, 2080])
    for _ax in _axs[1:]:
        _ax.set_yticks([])


    # save(fig, fname="skew-over-time_hov")
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
