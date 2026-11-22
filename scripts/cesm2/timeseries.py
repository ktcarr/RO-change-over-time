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
    # Timeseries
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
    import src.utils
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import seaborn as sns
    import sklearn.linear_model
    import pathlib
    import os
    import sklearn.linear_model
    import cmocean
    import pandas as pd
    import copy
    import matplotlib as mpl

    ## RNG
    rng = np.random.default_rng()

    ## color palette
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")

    ## paths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    # SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return DATA_FP, np, os, pathlib, pd, plt, src, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## funcs
    """)
    return


@app.cell
def _(os, pathlib, src, xr):
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


    def load_h_ohc_idxs():
        """load OHC-based h indices"""

        ## specify subsetting funcs
        LONS_E = dict(longitude=slice(210, 270))
        LONS_W = dict(longitude=slice(120, 180))

        ## averaging funcs
        lon_avg = lambda x, lons: x.sel(lons).mean("longitude")
        lat_avg = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")

        ## load spatial data
        _, anom_wide = load_consolidated_wide()

        ## empty array to hold result
        h_idxs = xr.Dataset()

        ## compute ohc
        h_idxs["h_w_ohc"] = src.utils.reconstruct_wrapper(
            anom_wide[["T", "T_comp"]],
            lambda x: lon_avg(x.integrate("z_t"), LONS_W) / 300,
        )["T"]
        h_idxs["h_e_ohc"] = src.utils.reconstruct_wrapper(
            anom_wide[["T", "T_comp"]],
            lambda x: lon_avg(x.integrate("z_t"), LONS_E) / 300,
        )["T"]

        h_idxs["h_ohc"] = 0.5 * (h_idxs["h_w_ohc"] + h_idxs["h_e_ohc"])
        h_idxs["dh_ohc"] = h_idxs["h_e_ohc"] - h_idxs["h_w_ohc"]

        return h_idxs


    def save(fig, fname, dpi=800, do_save=False):
        """save figure to file"""

        ## get save directory
        save_dir = pathlib.Path(os.environ["SAVE_FP"], "egu_poster_figs")

        ## get fname
        fname = save_dir / f"{fname}.pdf"

        if do_save:
            fig.savefig(fname, dpi=dpi, format="pdf")
        
        return

    return (save,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## compute indices
    """)
    return


@app.cell
def _(src):
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


    def get_w_sub(w, mld=50, delta=30):
        """get velocity at base of mixed layer"""

        ## get temperature difference
        w_sub = w.sel(z_t=slice(mld, mld + delta)).mean("z_t")

        return w_sub


    def get_dTdz_sub(Tsub, mld=50, delta=30):
        """get difference between surface and subsurface temp"""

        ## get temperature difference
        dT = src.utils.get_dT_sub(Tsub=Tsub, Hm=mld, delta=delta)

        ## get gradient
        dTdz = -dT / mld

        return dTdz


    def get_dTdz_sub2(Tsub, mld=80):
        """get difference between surface and subsurface temp"""

        ## get temperature difference
        dT = Tsub.sel(z_t=mld, method="nearest") - Tsub.sel(z_t=0, method="nearest")

        ## get gradient
        dTdz = -dT / mld

        return dTdz

    return get_dTdz_sub, get_w_sub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### individual metrics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Niño 3 averages
    """)
    return


@app.cell
def _(src, xr):
    ## should we use "wide" data?
    USE_WIDE = True

    ## load spatial data
    forced, anom_ = src.utils.load_consolidated()

    if USE_WIDE:

        ## load wide data
        forced_05, anom_05 = src.utils.load_consolidated_05()

        for _v in list(forced_05):
            forced[_v] = forced_05[_v]
            anom_[_v] = anom_05[_v]

    ## get subset of data for total
    VARNAMES = ["T", "w", "sst", "taux"]
    total = anom_[VARNAMES] + forced[VARNAMES]
    total = xr.merge([forced[[f"{v}_comp" for v in VARNAMES]], total])
    return (total,)


@app.cell
def _(get_dTdz_sub, get_w_sub, src, total, xr):
    sel_n3 = lambda x: x.sel(longitude=slice(210, 270)).mean("longitude")

    ## compute stratification
    dTdz_n3 = src.utils.reconstruct_wrapper(
        total[["T", "T_comp"]],
        lambda x: get_dTdz_sub(sel_n3(x)),
        # lambda x: get_dTdz_sub2(x.sel(longitude=slice(210, 270)).mean("longitude")),
    )["T"]

    ## get velocity at base of mixed layer
    w_n3 = src.utils.reconstruct_wrapper(
        total[["w", "w_comp"]],
        lambda x: get_w_sub(sel_n3(x)),
    )["w"]

    ## merge results
    total_n3 = xr.merge([dTdz_n3.rename("dTdz_n3"), w_n3.rename("w_n3")])
    return (total_n3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $\partial T / \partial x$
    """)
    return


@app.cell
def _(src, total):
    ## compute diff. def of Tw
    sel_lat = lambda x: x.sel(latitude=slice(-5, 5)).mean(["latitude", "longitude"])
    get_Tw = lambda x: sel_lat(x.sel(longitude=slice(140, 190)))
    get_Te = lambda x: sel_lat(x.sel(longitude=slice(240, 280)))
    get_dTdx = lambda x: get_Tw(x) - get_Te(x)

    ## compute dTdx
    dTdx = src.utils.reconstruct_wrapper(total[["sst", "sst_comp"]], get_dTdx)
    dTdx = dTdx.rename({"sst": "dTdx"})
    return (dTdx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $\partial T / \partial y$
    """)
    return


@app.cell
def _(src, total):
    ## compute diff. def of Tw
    sel_lon = lambda x: x.sel(longitude=slice(170, 250)).mean(["latitude", "longitude"])
    get_T = lambda x: sel_lon(x.sel(latitude=slice(-5, 5)))
    get_Ts = lambda x: sel_lon(x.sel(latitude=slice(-15, -5)))
    get_dTdy = lambda x: get_Ts(x) - get_T(x)

    ## compute dTdx
    dTdy = src.utils.reconstruct_wrapper(total[["sst", "sst_comp"]], get_dTdy)
    dTdy = dTdy.rename({"sst": "dTdy"})
    return (dTdy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $\tau_x$
    """)
    return


@app.cell
def _(src, total):
    def get_taux(x):
        """function to compute area-averaged taux"""

        ## specify lon/lat range for averaging
        lon_range = slice(150, 270)
        lat_range = slice(-5, 5)
        idx = dict(longitude=lon_range, latitude=lat_range)

        ## compute
        return x.sel(idx).mean(["longitude", "latitude"])


    ## compute
    taux = src.utils.reconstruct_wrapper(total[["taux", "taux_comp"]], fn=get_taux)
    return (taux,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $h$ max-gradient stuff
    """)
    return


@app.cell
def _(DATA_FP, pathlib, src, xr):
    # ## load tropical SST avg
    trop_sst = xr.open_dataset(pathlib.Path(DATA_FP, "cesm/trop_sst.nc"))

    ## Load T,h (total)
    # Th_total = xr.open_dataset(DATA_FP / "cesm" / "Th.nc")
    # Th_total = xr.merge([Th_total, trop_sst])

    Th_total = src.utils.load_cesm_indices_3d(mld=50)
    Th_total = xr.merge([Th_total, trop_sst])

    ## compute anomalies
    Th_anom = Th_total - Th_total.mean("member")
    return Th_anom, Th_total, trop_sst


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute relative SST
    """)
    return


@app.cell
def _(Th_total, src, trop_sst, xr):
    ## compute relative sst
    for _n in ["T_3", "T_34", "T_4"]:
        Th_total[f"{_n}_rel"] = Th_total[_n] - trop_sst["trop_sst_10"]

    ## custom h data
    h_total = src.utils.load_cesm_3d()["thc"]
    # h_mg_forced, h_mg_anom = src.utils.load_h_data(max_grad=True, use_wide=USE_WIDE)
    # h_total = h_mg_forced + h_mg_anom

    ## compute h stats
    h = h_total.sel(longitude=slice(155, 270)).mean("longitude")
    h_e = h_total.sel(longitude=slice(240, 270)).mean("longitude")
    h_w = h_total.sel(longitude=slice(155, 210)).mean("longitude")
    h_idxs = xr.merge([h.rename("h_mg"), (h_w - h_e).rename("dhdx_mg")])
    return h_idxs, h_total


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot sample
    """)
    return


@app.cell
def _(h_total, plt):
    h_early = h_total.sel(time=slice("1850-01", "1889-12")).mean(["time", "member"])
    h_middle = h_total.sel(time=slice("1990-01", "2029-12")).mean(["time", "member"])
    h_late = h_total.sel(time=slice("2060-01", "2099-12")).mean(["time", "member"])

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(h_early.longitude, h_early, label="1870")
    _ax.plot(h_early.longitude, h_middle, label="2010")
    _ax.plot(h_early.longitude, h_late, label="2080")

    ## plot lines
    _ax_kwargs = dict(ls="--", lw=0.8, c="gray")
    ticks = [155, 210, 240, 270]
    _ax.set_xticks(ticks)
    for t in ticks:
        _ax.axvline(t, **_ax_kwargs)

    ## set ylimit
    # ax.set_ylim(ax.get_ylim()[::-1])
    _ax.set_ylabel("Depth")
    _ax.set_xlabel("Longitude")
    _ax.set_xlim([120,None])
    _ax.set_ylim([175,5])
    _ax.legend()

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Combine data
    """)
    return


@app.cell
def _(Th_anom, Th_total, dTdx, dTdy, h_idxs, src, taux, total_n3, xr):
    ## specify time index for plotting
    ## start index: use 3 for SON, 4 for DJF, etc.
    time_idx = slice(0, None, None)
    # time_idx = slice(0,None,4) ## for JFM

    ## combine data
    data = xr.merge(
        [Th_total, total_n3, h_idxs, taux, dTdx, dTdy, Th_anom["T_34"].rename("T_34a")]
    )

    ## get windowed version
    data = src.utils.get_windowed(
        data.isel(time=slice(None, -1)), window_size=480, stride=60
    )

    ## resample to DJF
    data = data.resample({"time": "QS-JAN"}).mean().isel(time=time_idx)

    ## compute ensemble mean and sigma
    data_mean = data.mean("time")
    data_sigma = data.std("time")
    return data, data_mean, data_sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Look at subset
    """)
    return


@app.cell
def _(data_mean, data_sigma, plt):
    ## specify year index
    yi = 7

    _fig, _axs = plt.subplots(1, 4, figsize=(10, 2.5), layout="constrained")
    for _ax, _v in zip(_axs, ["w_n3", "dTdx", "dTdz_n3", "T_34_rel"]):
        _ax.set_title(_v)
        _ax.scatter(
            data_sigma["T_34a"].isel(year=yi),
            data_mean[_v].isel(year=yi),
        )
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Remove nonlinear rect. effect
    """)
    return


@app.cell
def _(data_mean, data_sigma, src, xr):
    coefs = src.utils.estimate_rect(
        xr.merge([data_sigma["T_34a"].rename("sigma_T"), data_mean]),
        xvar="sigma_T",
        yvars=list(data_mean),
    )
    return (coefs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot
    """)
    return


@app.cell
def _():
    ## func to get fractional changes
    def sel(x, x0=None):
        """get fractional change"""

        ## get baseline if not specified
        if x0 is None:
            x0 = x.isel(year=0)

        ## compute
        return (x - x0) / x0


    labels = {
        "taux": r"$\tau_x$",
        "w_n3": r"$\overline{w}$",
        "dTdx": r"$\overline{T}_w-\overline{T}_e$",
        # "dTdz_n3": r"$\partial \overline{T} / \partial z$",
        "dTdz_n3": r"$\overline{T}_{\text{ml}}-\overline{T}_{\text{sub}}$",
        "h_mg": r"$h$",
        "dhdx_mg": r"$\partial h/\partial x$",
        "dTdy": r"$\partial T\partial y$",
        "T_3_rel": r"Niño 3 rel. SST",
        "T_34_rel": r"Niño 3.4 rel. SST",
    }
    return labels, sel


@app.cell
def _(coefs, data, labels, plt, sel):
    ## specify variables to plot
    _plot_vars = ["taux", "w_n3", "dTdx", "dTdz_n3", "h_mg", "dhdx_mg", "dTdy", "T_3_rel"]

    _fig, _axs = plt.subplots(
        # 1, len(plot_vars), figsize=(2.5 * len(plot_vars), 2.5), layout="constrained"
        2,
        4,
        figsize=(9, 4.5),
        layout="constrained",
    )

    for _ax, _v in zip(_axs.flatten(), _plot_vars):

        ## compute
        _data_v_clim = data[_v].mean("time")
        _median = _data_v_clim.quantile(q=0.5, dim="member")
        _baseline = _median.isel(year=0)

        ## plot each quantile
        for _q, _lw in zip([0.5, 0.05, 0.95], [2, 1, 1]):

            ## get data for quantile
            _data_q = sel(_data_v_clim.quantile(q=_q, dim="member"), x0=_baseline)

            ## plot
            _ax.plot(data.year, _data_q, c="k", lw=_lw)

        ## plot diagnosed
        _ax.plot(data.year, sel(coefs.sel(degree=0)[_v]))

        ## plot reference line for tau_x
        data_taux_clim = data["taux"].mean("time")
        median_taux = data_taux_clim.quantile(q=0.5, dim="member")
        plot_taux = sel(median_taux)
        if "dz" in _v:
            plot_taux = plot_taux * (-1)
        _ax.plot(data.year, plot_taux, c="magenta", ls="--", lw=1)

        _ax_kwargs = dict(c="gray", lw=0.5, ls="--")
        _ax.axvline(2010, **_ax_kwargs)
        _ax.axvline(2085, **dict(_ax_kwargs, ls="-"))
        _ax.set_title(labels[_v])
        _ax.axhline(0, **dict(_ax_kwargs, ls="-"))

        if "dz" in _v:
            _ax.set_ylim([-0.1, 0.4])
            _ax.axhline(0.1, **_ax_kwargs)
        else:
            _ax.set_ylim([0.1, -0.4])
            _ax.axhline(-0.1, **_ax_kwargs)

    for _ax in _axs[:, 1:].flatten():
        _ax.set_yticks([])
    for _ax in _axs[0]:
        _ax.set_xticks([])
    for _ax in _axs[-1]:
        _ax.set_xticks([1870, 2010, 2080])
    for _ax in _axs[:, 0]:
        _ax.set_yticks([-0.3, -0.1, 0.1])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    plot in temperature coords?
    """)
    return


@app.cell
def _():
    ## add Temperature coord...
    # data_T = copy.deepcopy(data)
    # data_T["T_coord"] = data["trop_sst_15"].mean(["time","member"])
    # data_T = data_T.swap_dims({"year":"T_coord"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    variables to include:
    - $w$, $dT/dx$, $dT/dz$
    - global warming index
    - nino 3.4 index
    - enso variance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare to ENSO stats
    """)
    return


@app.cell
def _(Th):
    Th
    return


@app.cell
def _(data, labels, src, xr):
    ## specify variable
    VARNAME = "T_34"
    MONTH0 = 12

    ## load Th data
    # Th = src.utils.load_cesm_indices()
    # Th = xr.merge([Th, load_h_ohc_idxs()])
    Th_ = src.utils.load_cesm_indices_3d(mld=80)
    Th = Th_-Th_.mean("member")

    Th = src.utils.get_windowed(Th, stride=60, window_size=480)

    ## specify thresh
    Q = 0.95

    ## remove median
    _median = Th[VARNAME].groupby("time.season").median(["member", "time"])
    # median = Th[VARNAME].groupby("time.season").mean(["member", "time"])
    Th_median = Th[VARNAME].groupby("time.season") - _median

    ## resample to DJF
    Th_median_djf = Th_median.resample({"time": "QS-DEC"}).mean()
    Th_median_djf = Th_median_djf.sel(time=Th_median_djf.time.dt.month == MONTH0)
    Th_median_djf = Th_median_djf.isel(time=slice(1, -1))

    ## get magnitude for extremes
    nino_mag = Th_median_djf.where(
        Th_median_djf > Th_median_djf.quantile(q=Q, dim="time").drop_vars("quantile")
    ).mean("time")
    nina_mag = -Th_median_djf.where(
        Th_median_djf < Th_median_djf.quantile(q=1 - Q, dim="time").drop_vars("quantile")
    ).mean("time")

    ## same, but compute over all ensemble members
    nino_mag_ens = Th_median_djf.quantile(q=Q, dim=["time", "member"]).drop_vars("quantile")
    nina_mag_ens = -Th_median_djf.quantile(q=1 - Q, dim=["time", "member"]).drop_vars(
        "quantile"
    )

    ## merge
    mag = xr.merge([nino_mag.rename("warm"), nina_mag.rename("cold")])
    mag_ens = xr.merge([nino_mag_ens.rename("warm"), nina_mag_ens.rename("cold")])

    ## add median
    mag["sigma"] = Th_median_djf.std("time")
    mag_ens["sigma"] = Th_median_djf.std(["time", "member"])

    ## compute quantiles
    mag_mean = mag.mean("member")
    mag_stats = mag.quantile(q=[0.1, 0.5, 0.9], dim="member")

    ## add to timeseries dataarray
    data["sigma_T"] = mag["sigma"]
    labels["sigma_T"] = r"$\sigma(T_{34})$"
    return Th, Th_median_djf, mag, mag_ens, mag_mean, mag_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Add to timeseries plots
    """)
    return


@app.cell
def _(data, labels, np, plt, sel):
    ## specify variables to plot
    _plot_vars = ["sigma_T", "dTdz_n3", "w_n3", "dTdx"]

    ## get baseline value (global warming index)
    _sel_ = lambda x : x-x.isel(year=0)
    _gwi = _sel_(data["trop_sst_15"].mean(["time","member"]))

    _fig, _axs = plt.subplots(
        1,
        4,
        figsize=(6.5, 2),
        layout="constrained",
    )

    _axsT = np.array([ax.twinx() for ax in _axs])
    for _i, (_ax, _axT, _v) in enumerate(zip(_axs, _axsT, _plot_vars)):

        ## compute
        _data_v_clim = data.mean("time")[_v]
        _median = _data_v_clim.quantile(q=0.5, dim="member")
        _baseline = _median.isel(year=0)

        ## plot each quantile
        for _q, _lw in zip([0.5, 0.05, 0.95], [2, 1, 1]):

            ## get data for quantile
            _data_q = sel(_data_v_clim.quantile(q=_q, dim="member"), x0=_baseline)

            ## plot
            _ax.plot(data.year, _data_q, c="k", lw=_lw)


        ## plot reference line for global warming index
        # axT = ax.twinx()
        _axT.plot(_gwi.year, _gwi, c="gray", ls="--", lw=2)
        _axT.set_ylim([-1.5,4.5])
        _axT.set_yticks([])
        _ax.set_xticks([1870, 2010])

        ## formatting
        _ax_kwargs = dict(c="gray", lw=0.5, ls="-")
        _ax.axvline(2010, **_ax_kwargs)
        _ax.set_title(labels[_v])
        _ax.axhline(0, **dict(_ax_kwargs, ls="-"))

        if _i<2:
            _ax.set_ylim([-0.15, 0.45])
            # ax.axhline(0.1, **ax_kwargs)
        else:
            _ax.set_ylim([0.15, -0.45])
            # ax.axhline(-0.1, **ax_kwargs)

    for _ax in _axs[1:]:
        _ax.set_yticks([])
    _axs[0].set_yticks([0,  .3])
    _axs[2].set_yticks([0,  -.3])

    ## add some ticks/labels
    _axsT[-1].set_yticks([0,3])
    _axsT[-1].yaxis.label.set_color('gray')
    _axsT[-1].tick_params(axis='y', colors='gray')
    _axs[0].set_ylabel("Fractional change")
    _axsT[-1].set_ylabel("Trop. warming (K)")
    for _ax, _axT in zip(_axs,_axsT):
        for _ax_ in [_ax, _axT]:
            _ax_.tick_params(axis="y", which="both", length=0)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same, but vertical
    """)
    return


@app.cell
def _(data, labels, np, plt, save, sel):
    ## specify color for reference line
    _ref_color = "magenta"

    ## specify variables to plot
    _plot_vars = ["sigma_T", "dTdz_n3", "w_n3", "dTdx"]

    ## get baseline value (global warming index)
    _sel_ = lambda x : x-x.isel(year=0)
    _gwi = _sel_(data["trop_sst_15"].mean(["time","member"]))

    _fig, _axs = plt.subplots(
        4,
        1,
        figsize=(2.5,7),
        layout="constrained",
    )

    ## twin axis
    _axsT = np.array([ax.twinx() for ax in _axs])

    ## loop thru axes
    prefixes = ["(d)", "(e)", "(f)", "(g)"]
    for _i, (_ax, _axT, _v, _pre) in enumerate(zip(_axs, _axsT, _plot_vars, prefixes)):

        ## compute
        _data_v_clim = data.mean("time")[_v]
        _median = _data_v_clim.quantile(q=0.5, dim="member")
        _baseline = _median.isel(year=0)

        ## plot each quantile
        for _q, _lw in zip([0.5, 0.05, 0.95], [2, 1, 1]):

            ## get data for quantile
            _data_q = sel(_data_v_clim.quantile(q=_q, dim="member"), x0=_baseline)

            ## get label
            if (_q==0.5):
                label="Median"
            elif (_q==.05):
                label="10-90%"
            else:
                label = None

            ## plot
            _ax.plot(data.year, _data_q, c="k", lw=_lw, label=label)


        ## plot reference line for global warming index
        # axT = ax.twinx()
        _axT.plot(_gwi.year, _gwi, c="magenta", ls="--", lw=2, label="Trop. SST")
        _axT.set_ylim([-1.5,4.5])
        _axT.set_yticks([])
        _ax.set_xticks([])

        ## formatting
        _ax_kwargs = dict(c="gray", lw=0.5, ls="-")
        _ax.axvline(2010, **_ax_kwargs)
        _ax.text(s=f"{_pre} {labels[_v]}", x=-0.01, y=0.85, transform=_ax.transAxes, zorder=10)
        _ax.axhline(0, **dict(_ax_kwargs, ls="-"))

        if _i<2:
            _ax.set_ylim([-0.15, 0.45])
        else:
            _ax.set_ylim([0.15, -0.45])

    for _ax in _axs[1:]:
        _ax.set_yticks([])

    for _ax in _axs[:2]:
        _ax.set_yticks([0,  .3], labels=["0%", r"+30%"])
    for _ax in _axs[2:]:
        _ax.set_yticks([0,  -.3], labels=[r"0%", r"$-$30%"])

    ## add some ticks/labels
    _axs[-1].set_xticks([1870, 2010])
    for _axT in _axsT:
        _axT.set_yticks([0,3], labels = [r"0$^{\circ}$C", r"3$^{\circ}$C"])
        _axT.yaxis.label.set_color(_ref_color)
        _axT.tick_params(axis='y', colors=_ref_color)
    
    for _ax, _axT in zip(_axs,_axsT):
        for _ax_ in [_ax, _axT]:
            _ax_.tick_params(axis="y", which="both", length=0)
            _ax_.grid(False)

    leg_kwargs = dict(prop=dict(size=9), frameon=False)
    _axs[2].legend(loc=(0,.53), **leg_kwargs)
    _axsT[2].legend(loc=(0, .41), **leg_kwargs)

    ## save 
    save(_fig, fname="timeseries", do_save=False)

    plt.gca()
    return (prefixes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## new version (for jerome)
    """)
    return


@app.cell
def _(data, labels, np, plt, prefixes, sel):
    ## specify color for reference line
    _ref_color = "magenta"

    ## get baseline value (global warming index)
    _sel_ = lambda x : x-x.isel(year=0)
    _gwi = _sel_(data["trop_sst_15"].mean(["time","member"]))

    nvars = len(list(labels))
    nloop = np.ceil(nvars/4).astype(int)

    for _i in range(nloop):
        _plot_vars = list(labels)[4*_i:4*(_i+1)]

        _fig, _axs = plt.subplots(
            4,
            1,
            figsize=(3.5,9.5),
            layout="constrained",
        )
    
        ## twin axis
        _axsT = np.array([ax.twinx() for ax in _axs])
    
        ## loop thru axes
        for _i, (_ax, _axT, _v, _pre) in enumerate(zip(_axs, _axsT, _plot_vars, prefixes)):
    
            ## compute
            _data_v_clim = data.mean("time")[_v]
            _median = _data_v_clim.quantile(q=0.5, dim="member")
            _baseline = _median.isel(year=0)
    
            ## plot each quantile
            for _q, _lw in zip([0.5, 0.05, 0.95], [2, 1, 1]):
    
                ## get data for quantile
                _data_q = sel(_data_v_clim.quantile(q=_q, dim="member"), x0=_baseline)
    
                ## plot
                _ax.plot(data.year, _data_q, c="k", lw=_lw)
    
    
            ## plot reference line for global warming index
            # axT = ax.twinx()
            _axT.plot(_gwi.year, _gwi, c="magenta", ls="--", lw=2)
            _axT.set_ylim([-1.5,7])
            _axT.set_yticks([])
            _ax.set_xticks([])
    
            ## formatting
            _ax_kwargs = dict(c="gray", lw=0.5, ls="-")
            _ax.axvline(2010, **_ax_kwargs)
            _ax.axvline(2085, **_ax_kwargs)
            _ax.text(s=f"{labels[_v]}", x=.01, y=0.6, transform=_ax.transAxes, zorder=10)
            _ax.axhline(0, **dict(_ax_kwargs, ls="-"))

            ## handle axes
            ## get max increase
            final_change = sel(
                _data_v_clim.quantile(q=0.5, dim="member"), x0=_baseline
            ).isel(year=-1).values.item()
            # final_change = median.isel(year=-1).values.item()
            if final_change < 0:
                _ax.set_ylim([0.15, -0.7])
                _ax.set_yticks([0,  -.3, -.6], labels=[r"0%", r"$-$30%", r"$-$60%"])
                _ax.axhline(-.3, **dict(_ax_kwargs, ls="-"))
            else:
                _ax.set_ylim([-0.15, 0.7])
                _ax.set_yticks([0,  .3, .6], labels=["0%", r"+30%", r"+60%"])
                _ax.axhline(.3, **dict(_ax_kwargs, ls="-"))

            ## label final change
            _ax.text(s=f"{final_change*100:.1f}%", x=2130, y=final_change,  zorder=10)

        ## add some ticks/labels
        _axs[-1].set_xticks([1870, 2010, 2085])
        for _axT in _axsT:
            _axT.set_yticks([0,3], labels = [r"0$^{\circ}$C", r"3$^{\circ}$C"])
            _axT.yaxis.label.set_color(_ref_color)
            _axT.tick_params(axis='y', colors=_ref_color)
        # axs[0].set_ylabel("Fractional change")
        # axsT[-1].set_ylabel("Trop. warming (K)")
        for _ax, _axT in zip(_axs,_axsT):
            for _ax_ in [_ax, _axT]:
                _ax_.tick_params(axis="y", which="both", length=0)
                _ax_.grid(False)
    
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot warm/cold amplitude over time
    """)
    return


@app.cell(hide_code=True)
def _(mag, mag_stats, plt, src):
    _fig, _axs = plt.subplots(1, 2, figsize=(5.5, 2.5))

    ## plot width of distribution
    for _ax, _n, _c in zip(_axs, ["warm", "cold"], ["r", "b"]):

        ## plot average
        _avg = 0.5 * (mag_stats["warm"] + mag_stats["cold"]).sel(quantile=0.5)
        _ax.plot(mag.year, _avg, c="k", ls="--")

        for _q, _lw in zip([0.5, 0.1, 0.9], [2.5, 1, 1]):
            _ax.plot(mag.year, mag_stats[_n].sel(quantile=_q), lw=_lw, c=_c)

        _ax.axvline(2010, ls="--", lw=0.8, c="gray")
        _ax.axvline(2050, ls="--", lw=0.8, c="gray")
        _ax.set_xticks([1880, 2010, 2080])


    # ax.set_title(r"Distribution width (95% minus median)")
    _axs[1].set_ylabel(r"$^{\circ}$C")
    _axs[0].set_xlabel(r"Year")
    # axs[0].set_yticks([2, 2.5, 3])
    # axs[0].set_yticks([0, 0.1, 0.2, 0.3])
    # axs[0].set_yticks([])
    _axs[1].yaxis.tick_right()
    _axs[1].yaxis.set_label_position("right")
    src.utils.set_lims(_axs[1:])
    # axs[0].set_title("(a) Niño 3.4 PDF", loc="left")
    _axs[0].set_title("(b) Strong El Niño", loc="left")
    _axs[1].set_title("(c) Strong La Niña", loc="left")
    src.utils.set_lims(_axs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same, but plot for all ensemble
    """)
    return


@app.cell
def _(mag_ens, plt):
    _fig, _ax = plt.subplots(figsize=(3, 2.5))

    ## plot average
    _avg = 0.5 * (mag_ens["warm"] + mag_ens["cold"])
    _ax.plot(mag_ens.year, _avg, c="k", ls="--")
    _ax.plot(mag_ens.year, mag_ens["warm"], c="r")
    _ax.plot(mag_ens.year, mag_ens["cold"], c="b")
    # ax.plot(mag_ens.year, mag_ens["warm"]-mag_ens["cold"]+2.5, c="magenta")

    _ax.axvline(2010, ls="--", lw=0.8, c="gray")
    _ax.axvline(2050, ls="--", lw=0.8, c="gray")
    _ax.set_xticks([1880, 2010, 2080])


    # ax.set_title(r"Distribution width (95% minus median)")
    _ax.set_ylabel(r"$^{\circ}$C")
    _ax.set_xlabel(r"Year")
    # ax.set_yticks([2, 2.5, 3])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Contribution from quantiles
    """)
    return


@app.cell
def _(Th_median_djf, mag_ens, np):
    ## Get T prime
    T = Th_median_djf - Th_median_djf.median(["member", "time"])
    Tbar = T.mean(["member", "time"])
    Tp = T - Tbar

    ## count samples over time
    N = (~np.isnan(Tp)).sum(["member", "time"])

    ## compute variance (total)
    Tp_var = 1 / N * (Tp**2).sum(["member", "time"])

    ## check computation matches earlier
    print(np.allclose(Tp_var, mag_ens["sigma"] ** 2))
    return N, T, Tbar, Tp, Tp_var


@app.cell
def _(N, T, Tbar, Tp_var, np):
    ## decompose
    p0 = 1 / N * (T**2).sum(["member", "time"])
    p1 = -1 / N * (2 * T * Tbar).sum(["member", "time"])
    p2 = Tbar**2

    ## check all close
    print(np.allclose(Tp_var, p0 + p1 + p2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternative decomp.
    """)
    return


@app.cell
def _(N, T, Tp, Tp_var, np, pd, xr):
    ## get bounds
    _bounds = Tp.quantile(dim=["member", "time"], q=[0.1, 0.25, 0.75, 0.9])

    ## find samples in each class
    _in_class0 = Tp < _bounds[0]  ## extreme La Niña
    _in_class1 = (_bounds[0] <= Tp) & (Tp < _bounds[1])  ## moderate La Niña
    _in_class2 = (_bounds[1] <= Tp) & (Tp <= _bounds[2])  ## neutral
    _in_class3 = (_bounds[2] < Tp) & (Tp <= _bounds[3])  ## moderate El Niño
    _in_class4 = _bounds[3] < Tp  ## extreme El Niño

    ## put in single dataarray
    _in_class = xr.concat(
        [_in_class0, _in_class1, _in_class2, _in_class3, _in_class4],
        dim=pd.Index(np.arange(5), name="intensity"),
        coords="minimal",
        compat="override",
    )

    ## compute variance from each
    _Tp_var_ = 1 / N * (Tp.where(_in_class) ** 2).sum(["time", "member"])
    Tp_var_med = 1 / N * (T.where(_in_class) ** 2).sum(["time", "member"])

    ## check decomposition works
    print(np.allclose(_Tp_var_.sum("intensity"), Tp_var))

    ## compute change over time
    delta_var = _Tp_var_ - _Tp_var_.isel(year=0)
    delta_var_med = Tp_var_med - Tp_var_med.isel(year=0)
    return delta_var, delta_var_med


@app.cell
def _(delta_var, delta_var_med, plt, src):
    year = delta_var.year
    _fig, _axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

    for _ax, _dvar in zip(_axs, [delta_var, delta_var_med, delta_var_med - delta_var]):
        _ax.plot(year, .5*_dvar.sum("intensity"), c="k", label="1/2 total")
        _ax.plot(
            year,
            _dvar.isel(intensity=slice(None, 2)).sum("intensity"),
            c="b",
            label="La Niña",
        )

        _ax.plot(
            year,
            _dvar.isel(intensity=slice(-2, None)).sum("intensity"),
            c="r",
            label="El Niño",
        )
    
        _ax.plot(year, _dvar.isel(intensity=2), c="gray", ls="-", label="Neutral")
        # ax.plot(year, dvar.isel(intensity=-2), c="r", ls="--", label="Moderate El Niño")
        # ax.plot(year, dvar.isel(intensity=-1), c="r", label="Strong El Niño")

    for _ax in _axs:
        _ax.axhline(0, lw=0.5, c="gray")
        _ax.axvline(2010, lw=0.5, c="gray")
        _ax.axvline(2060, lw=0.5, c="gray")
        _ax.set_yticks([])
        _ax.set_xticks([1870, 2010, 2060])

    src.utils.set_lims(_axs)
    _axs[0].set_yticks([-0.3, 0, 0.4, 0.8])
    _axs[-1].legend(prop=dict(size=10))
    _axs[0].set_title("a) Referenced to mean", loc="left")
    _axs[1].set_title("b) Referenced to median", loc="left")
    _axs[2].set_title("c) Difference (b-a)", loc="left")
    _axs[0].set_ylabel(r"K$^2$")
    plt.gca()
    return


@app.cell
def _(delta_var, delta_var_med, plt, src):
    year_1 = delta_var.year
    _fig, _axs = plt.subplots(2, 2, figsize=(5, 5), layout="constrained")

    for col_idx, _dvar in enumerate([delta_var, delta_var_med]):
    
        _axs[0,col_idx].plot(year_1, _dvar.isel(intensity=0),c="b",label="Strong La Niña", ls="--")
        _axs[1,col_idx].plot(year_1, _dvar.isel(intensity=-1),c="r",label="Strong El Niño", ls="--")

        _axs[0,col_idx].plot(year_1, _dvar.isel(intensity=1),c="b",label="Moderate La Niña")
        _axs[1,col_idx].plot(year_1, _dvar.isel(intensity=-2),c="r",label="Moderate El Niño")

        # ax.plot(
        #     year,
        #     dvar.isel(intensity=slice(-2, None)).sum("intensity"),
        #     c="r",
        #     label="El Niño",
        # )
    
        # ax.plot(year, dvar.isel(intensity=2), c="gray", ls="-", label="Neutral")
        # ax.plot(year, dvar.isel(intensity=-2), c="r", ls="--", label="Moderate El Niño")
        # ax.plot(year, dvar.isel(intensity=-1), c="r", label="Strong El Niño")

    for _ax in _axs.flatten():
        _ax.plot(year_1, .25*_dvar.sum("intensity"), c="k", label="1/4 total")
        _ax.axhline(0, lw=0.5, c="gray")
        _ax.axvline(2010, lw=0.5, c="gray")
        _ax.axvline(2060, lw=0.5, c="gray")
        _ax.set_yticks([])
        _ax.set_xticks([1870, 2010, 2060])


    _axs[0,0].set_yticks([-.3, 0, 0.3])
    _axs[1,0].set_yticks([-.3, 0, 0.3])
    _axs[0,0].legend(prop=dict(size=10))
    _axs[1,0].legend(prop=dict(size=10))
    _axs[0,0].set_title("Referenced to mean", loc="left")
    _axs[0,1].set_title("Referenced to Median", loc="left")
    src.utils.set_lims(_axs.flatten())
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Same, but with bounds
    """)
    return


@app.cell
def _(T, Tp, Tp_var, np, pd, xr):
    ## get bounds
    _bounds = Tp.quantile(dim=["member", "time"], q=[0.1, 0.25, 0.75, 0.9])

    ## find samples in each class
    _in_class0 = Tp < _bounds[0]  ## extreme La Niña
    _in_class1 = (_bounds[0] <= Tp) & (Tp < _bounds[1])  ## moderate La Niña
    _in_class2 = (_bounds[1] <= Tp) & (Tp <= _bounds[2])  ## neutral
    _in_class3 = (_bounds[2] < Tp) & (Tp <= _bounds[3])  ## moderate El Niño
    _in_class4 = _bounds[3] < Tp  ## extreme El Niño

    ## put in single dataarray
    _in_class = xr.concat(
        [_in_class0, _in_class1, _in_class2, _in_class3, _in_class4],
        dim=pd.Index(np.arange(5), name="intensity"),
        coords="minimal",
        compat="override",
    )

    ## compute variance from each
    Nt = (~np.isnan(Tp)).sum("time")
    _Tp_var_ = 1 / Nt * (Tp.where(_in_class) ** 2).sum("time")
    Tp_var_med_1 = 1 / Nt * (T.where(_in_class) ** 2).sum("time")

    ## check decomposition works
    print(np.allclose(_Tp_var_.sum(["intensity", "member"]), Tp_var))

    ## compute change over time
    delta_var_1 = _Tp_var_ - _Tp_var_.isel(year=0).mean("member")
    delta_var_med_1 = Tp_var_med_1 - Tp_var_med_1.isel(year=0).mean("member")
    return Tp_var_med_1, delta_var_1, delta_var_med_1


@app.cell
def _(delta_var_1, delta_var_med_1, plt, src):
    year_2 = delta_var_1.year
    _fig, _axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

    for _ax, _dvar in zip(_axs, [delta_var_1, delta_var_med_1, delta_var_med_1 - delta_var_1]):
        _ax.plot(year_2, 0.5 * _dvar.sum("intensity").median("member"), c="k", label="Total")
        _ax.plot(
            year_2,
            _dvar.isel(intensity=slice(None, 2)).sum("intensity").median("member"),
            c="b",
            label="La Niña",
        )

        sel_q = lambda x: x.sum("intensity").quantile(q=_q, dim="member")
        for _q in [0.1, 0.9]:
            _ax.plot(
                year_2,
                sel_q(_dvar.isel(intensity=slice(None, 2))),
                c="b",
                label="La Niña",
                ls="--",
            )

            _ax.plot(
                year_2,
                sel_q(_dvar.isel(intensity=[-1, -2])),
                ls="--",
                lw=1,
                c="r",
                label="Strong El Niño",
            )
        _ax.plot(
            year_2,
            _dvar.isel(intensity=[-1, -2]).sum("intensity").median("member"),
            c="r",
            label="El Niño",
        )


    for _ax in _axs:
        _ax.axhline(0, lw=0.5, c="gray")
        _ax.axvline(2010, lw=0.5, c="gray")
        _ax.set_yticks([])
        _ax.set_xticks([1870, 2010, 2080])

    src.utils.set_lims(_axs)
    _axs[0].set_yticks([-0.3, 0, 0.4, 0.8])
    _axs[-1].legend(prop=dict(size=10))
    _axs[0].set_title("a) Referenced to mean", loc="left")
    _axs[1].set_title("b) Referenced to median", loc="left")
    _axs[2].set_title("c) Difference (b-a)", loc="left")
    _axs[0].set_ylabel(r"K$^2$")
    plt.gca()
    return (year_2,)


@app.cell
def _(Tp_var_med_1, plt, save, src, year_2):
    _fig, _axs = plt.subplots(1, 2, figsize=(8*2/3, 2.75), layout="constrained")

    for _ax, color, intensity, na in zip(
        _axs, ["r", "b"], [[-2, -1], [0, 1]], ["El Niño", "La Niña"]
    ):

        ## get dvar
        # dvar = delta_var_med.isel(intensity=intensity).sum("intensity")
        _dvar = Tp_var_med_1.isel(intensity=intensity).sum("intensity")
        # Tp_var_med

        ## plot

        ## plot overall intensity (scaled by 0.5)
        # ax.plot(year, 0.5 * delta_var_med.sum("intensity").median("member"), c="k", label=r"1/2 total change")
        _ax.plot(
            year_2,
            0.5 * Tp_var_med_1.sum("intensity").median("member"),
            c="k",
            label=r"1/2 total",
        )

        ## plot median and bounds
        for _q, _lw, ls, lab in zip(
            [0.5, 0.1, 0.9], [2, 1, 1], ["-", "--", "--"], [f"Median", "10-90%", None]
        ):
            _ax.plot(
                _dvar.year,
                _dvar.quantile(q=_q, dim="member"),
                c=color,
                lw=_lw,
                ls=ls,
                label=lab,
            )

        ## formatting
        legend = _ax.legend(prop=dict(size=10), loc=(-.1,.68), frameon=True)
        frame = legend.get_frame()
        frame.set_color('white')
        _ax.axhline(1, lw=0.5, c="gray")
        _ax.set_yticks([])

    for _ax, tick in zip(_axs, [2010, 2060]):
        _ax.set_xticks([1870, tick])
        _ax.axvline(tick, lw=0.5, c="gray")
        _ax.set_xlabel("Year")

    src.utils.set_lims(_axs)
    _axs[0].set_title("b) El Niño contribution", loc="left")
    _axs[1].set_title("c) La Niña contribution", loc="left")
    _axs[1].set_ylabel(r"Variance (K$^2$)")

    # axs[1].set_yticks([-0.3, 0, 0.4, 0.8])
    _axs[1].set_yticks([0.5, 1, 1.5, 2])
    _axs[1].yaxis.set_label_position("right")
    _axs[1].yaxis.tick_right()
    _axs[1].tick_params(axis="y", which="both", length=0)

    save(_fig, fname="variance-contrib", do_save=False)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### skewness over time
    """)
    return


@app.cell
def _(Th):
    Th_djf = Th.resample({"time": "QS-DEC"}).mean()
    Th_djf = Th_djf.isel(time=slice(1, -1)).isel(year=slice(None, -1))
    Th_djf = Th_djf.sel(time=Th_djf.time.dt.month == 12)
    return (Th_djf,)


@app.cell
def _(Th_djf, np, xr):
    import scipy.stats

    ## compute stats
    get_stat = lambda x, fn: xr.apply_ufunc(
        fn, x, input_core_dims=[["time"]], kwargs={"axis": -1}
    )
    median_ot = get_stat(Th_djf, np.median)
    skew_ot = get_stat(Th_djf, scipy.stats.skew)
    std_ot = get_stat(Th_djf, np.std)
    m3_ot = skew_ot * std_ot ** (3 / 2)
    return m3_ot, median_ot, std_ot


@app.cell
def _(m3_ot, median_ot, plt, std_ot):
    ## specify variable to look at
    VAR = "T_34"
    YEARS = [1870, 2010, 2080]

    ## set up plot
    _fig, _axs = plt.subplots(1, 3, figsize=(9, 2.5))

    # for stat, c in zip([std_ot, median_ot], ["k","gray"]):

    #     for q, lw in zip([0.5, 0.1, 0.9], [2.5, 1, 1]):
    #         axs[0].plot(stat.year, stat[VAR].quantile(q=q, dim="member"), lw=lw, c=c)

    ## plot stats
    for _ax, stat in zip(_axs, [std_ot, median_ot, m3_ot]):

        for _q, _lw in zip([0.5, 0.1, 0.9], [2.5, 1, 1]):
            _ax.plot(stat.year, stat[VAR].quantile(q=_q, dim="member"), lw=_lw, c="k")

    for _ax in _axs:
        _ax.axvline(2010, ls="--", lw=0.8, c="gray")
        _ax.set_xticks([1870, 2010, 2080])

    for _ax in _axs[1:]:
        _ax.axhline(0, ls="--", lw=0.8, c="gray")

    ## formatting
    _axs[0].set_ylim([1, 2])
    _axs[1].set_ylim([-0.5, 0.5])
    _axs[0].set_yticks([1, 1.5, 2])
    _axs[1].set_yticks([-0.5, 0, 0.5])
    _axs[2].set_yticks([-1, 0, 1])

    ## label
    _axs[0].set_ylabel(r"$^{\circ}$C")
    _axs[1].set_ylabel(r"$^{\circ}$C")
    _axs[2].set_ylabel(r"$^{\circ}$C$^3$")
    _axs[2].set_ylabel(r"non-dim.")
    _axs[0].set_title(r"a) Standard deviation", loc="left")
    _axs[1].set_title(r"b) Median", loc="left")
    _axs[2].set_title(r"c) Skew", loc="left")

    _fig.subplots_adjust(wspace=0.75)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scatter plot of gradient vs. amplitude over time
    """)
    return


@app.cell
def _(coefs, data, mag, mag_mean, plt, sel):
    _USE_NORECT = False

    _fig, _axs = plt.subplots(3, 2, figsize=(4.5, 6), layout="constrained")

    ## specify x_data
    if _USE_NORECT:
        _x_data = coefs.sel(degree=0)
    else:
        _x_data = data.mean(["time", "member"])

    ## get plot data
    _x_data_plot = sel(_x_data)

    ## each row: (std, nino, nina)
    for _i, (_v, s) in enumerate(zip(["dTdx", "dTdz_n3"], [1, -1])):

        for j, data_ in enumerate(
            [mag["sigma"].mean("member"), mag_mean["warm"], mag_mean["cold"]]
            # [mag_ens["sigma"], mag_ens["warm"], mag_ens["cold"]]
        ):

            ## plot data
            _sc_kwargs = dict(s=35, c="gray", alpha=0.8)
            # sc_kwargs = dict(s=35, c=x_data.year, cmap="cmo.amp", alpha=0.8)
            _axs[j, _i].scatter(s*_x_data_plot[_v], data_, **_sc_kwargs)

            ## highlight points
            for _y, _c in zip([1870, 2010, 2085], ["k", "orange", "red"]):
                _sc = _axs[j, _i].scatter(
                    s*_x_data_plot[_v].sel(year=_y), data_.sel(year=_y), c=_c, s=100, label=_y
                )

    ## labeling
    for _ax in _axs[:, 1:].flatten():
        _ax.set_yticks([])
    for _ax in _axs[:-1, :].flatten():
        _ax.set_xticks([])
    _axs[1, 1].legend(prop=dict(size=10))

    ## add guideline
    for _ax in _axs.flatten():
        _ax.axvline(0, c="k", lw=0.5)

    for _ax in _axs[:, 0]:
        _ax.set_xlim([-0.3, 0.15])

    for _ax in _axs[:, 1]:
        # ax.set_xlim([-0.15, 0.9])
        _ax.set_xlim([-0.9, 0.15])

    _axs[-1, 0].set_xlabel(r"$\Delta \left(\frac{\partial T}{\partial x}\right)$")
    _axs[-1, 1].set_xlabel(r"$-\Delta \left(\frac{\partial T}{\partial z}\right)$")

    for _ax in _axs[1:].flatten():
        _ax.set_ylim([2, 3.4])
    _axs[0, 0].set_yticks([1.4, 1.6])
    _axs[1, 0].set_yticks([2, 3])
    _axs[2, 0].set_yticks([2, 3])

    _axs[0, 0].set_ylabel(r"$\sigma (K)$")
    _axs[1, 0].set_ylabel(r"El Niño (K)")
    _axs[2, 0].set_ylabel(r"La Niña (K)")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### vertical vs. horizontal gradient
    """)
    return


@app.cell
def _(coefs, data, plt, sel):
    _USE_NORECT = True

    _fig, _axs = plt.subplots(1, 2, figsize=(5, 2.5), layout="constrained")

    ## specify x_data
    if _USE_NORECT:
        _x_data = coefs.sel(degree=0)
    else:
        _x_data = data.mean(["time", "member"])

    ## get plot data
    _x_data_plot = sel(_x_data)

    ## each row: (std, nino, nina)
    for _i, no_rect in enumerate([True, False]):

        ## specify data
        if no_rect:
            _x_data = coefs.sel(degree=0)
        else:
            _x_data = data.mean(["time", "member"])

        ## get plot data
        _x_data_plot = sel(_x_data)

        ## plot data
        _sc_kwargs = dict(s=25, c="gray", alpha=0.8)
        # sc_kwargs = dict(s=35, c=x_data.year, cmap="cmo.amp", alpha=0.8)
        _axs[_i].scatter(_x_data_plot["dTdx"], _x_data_plot["dTdz_n3"], **_sc_kwargs)

        ## highlight points
        for _y, _c in zip([1870, 2010, 2085], ["k", "orange", "red"]):
            _sc = _axs[_i].scatter(
                _x_data_plot["dTdx"].sel(year=_y),
                _x_data_plot["dTdz_n3"].sel(year=_y),
                c=_c,
                s=100,
                label=_y,
            )

    ## formatting
    for _ax in _axs:
        _ax.axvline(0, c="k", lw=0.5)
        _ax.set_xlabel(r"$\Delta \left(\frac{\partial T}{\partial x}\right)$")
    _axs[0].set_ylabel(r"$\Delta \left(\frac{\partial T}{\partial z}\right)$")
    _axs[1].set_yticks([])
    _axs[1].legend(prop=dict(size=10))
    _axs[0].set_title("w/o rectification")
    _axs[1].set_title("w/ rectification")

    plt.gca()
    return


if __name__ == "__main__":
    app.run()
