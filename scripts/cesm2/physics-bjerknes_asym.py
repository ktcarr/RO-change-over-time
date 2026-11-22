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
    # Bjerknes feedback changes over time
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
    return DATA_FP, SAVE_FP, cmocean, copy, np, os, pathlib, plt, src, tqdm, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify kwargs
    """)
    return


@app.cell
def _():
    # Specify T variable
    T_VAR = "T_3"

    ## should we use OHC for RO's 'h' variable?
    USE_OHC = True

    ## should we subtract median?
    SUBTRACT_MEDIAN = True

    ## specify MLD
    MLD = 50

    ## should we use T_ml?
    USE_T_ML = False

    ## specify entrainment depth (meters)
    ENT_DEPTH_RANGE = slice(51, 80)

    ## MONTH FOR PLOTTING
    SEL_MONTH = lambda x: x.sel(month=slice(4,6)).mean("month")
    MONTH0 = 4

    ## WHETHER TO COMPUTE BY MEMBER
    BY_MEMBER = True

    if BY_MEMBER:
        DIMS = ["time"]
    else:
        DIMS = ["time", "member"]

    ## Get figure suffix
    FIG_SUFFIX = f'{T_VAR}{"_ML" if USE_T_ML else ""}_MLD{MLD}{"_OHC" if USE_OHC else ""}{"_nomedian" if SUBTRACT_MEDIAN else ""}'
    return (
        DIMS,
        ENT_DEPTH_RANGE,
        FIG_SUFFIX,
        MLD,
        MONTH0,
        SEL_MONTH,
        SUBTRACT_MEDIAN,
        T_VAR,
        USE_OHC,
        USE_T_ML,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load pre-computed regression coefficients
    """)
    return


@app.cell
def _(DATA_FP, SUBTRACT_MEDIAN, T_VAR, USE_OHC, USE_T_ML, xr):
    ## get filepath
    if SUBTRACT_MEDIAN:
        txt = "_zero_median"
    else:
        txt = ""
    
    if USE_OHC:
        suf = ""
    else:
        suf = "_ssh"

    if USE_T_ML:
        suf2 = "_ML"
    else:
        suf2 = ""

    ## build filepath
    bjerknes_fp = DATA_FP / f"bjerknes{txt}{suf}{suf2}"

    ## path where data is located
    save_fp = bjerknes_fp / f"coefs_xvars={T_VAR}-h_w"
    load = lambda n: xr.open_dataset(save_fp / f"{n}.nc")
    coefs = {n: load(n) for n in ["all", "pos", "neg"]}

    save_fp = bjerknes_fp / f"coefs_xvars={T_VAR}"
    load = lambda n: xr.open_dataset(save_fp / f"{n}.nc")
    coefs_T = {n: load(n) for n in ["all", "pos", "neg"]}
    return coefs, coefs_T


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
def _(
    FIG_SUFFIX,
    SAVE_FP,
    SUBTRACT_MEDIAN,
    copy,
    np,
    os,
    pathlib,
    plt,
    src,
    xr,
):
    def window(x, subtract_median=SUBTRACT_MEDIAN):
        """get data in windows"""
        x_windowed = src.utils.get_windowed(x, stride=120)

        ## handle median case
        if subtract_median:
            median = x_windowed.groupby("time.month").median(["time", "member"])
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


    def regress_over_time(
        data_windowed,
        x_vars,
        y_vars,
        dims=["time", "member"],
    ):
        """regression over time"""

        ## shared args
        kwargs = dict(x_vars=x_vars, y_vars=y_vars, dims=dims)

        ## do regression
        coefs = data_windowed.groupby("time.month").map(src.utils.regress_xr_proj, **kwargs)

        return coefs


    def regress_wrapper(data, x_vars, y_var, y_fn, dims=["time", "member"]):
        """regression over time"""

        ## prep data
        y_data = src.utils.reconstruct_wrapper(data[[f"{y_var}", f"{y_var}_comp"]], fn=y_fn)

        ## subset for data
        data_ = xr.merge([data[x_vars], y_data])

        return regress_over_time(data_, x_vars=x_vars, y_vars=[y_var], dims=dims)


    def frac_change(x):
        """fractional change"""
        return x / x.isel(year=0) - 1


    def check_dims(x):
        """make sure dimensions are ok before averaging"""
        ## check if latitude is in ssh
        if "latitude" in x.dims:
            x_ = copy.deepcopy(x)
        else:
            x_ = x.expand_dims("latitude")

        ## check if z_t is in ssh
        if "z_t" in x.dims:
            x_ = copy.deepcopy(x_)
        else:
            x_ = x_.expand_dims("z_t")

        return x_


    def get_ddt(data):
        """differentiate with respect to time"""
        data_ = copy.deepcopy(data)
        data_ = data_.assign_coords({"t_idx": ("time", np.arange(len(data_.time)))})
        data_ = data_.swap_dims({"time": "t_idx"})
        return data_.differentiate("t_idx").swap_dims({"t_idx": "time"})

    def save(fig, fname, do_save=False, **kwargs):
        """save figure to results."""

        ## set save filepath
        save_dir = SAVE_FP / "figs"

        if do_save:

            ## save figure
            fig.savefig(save_dir / f"{fname}_{FIG_SUFFIX}.pdf", format="pdf", dpi=500, **kwargs)

            ## close figure
            plt.close(fig)

        return

    return (
        check_dims,
        get_ddt,
        load_consolidated_wide,
        regress_over_time,
        save,
        window,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting
    """)
    return


@app.cell
def _(plt, src, xr):
    def make_scatter_ax(ax, anom_, xvar, yvar, month, label, verbose=True, by_season=True, **kwargs):
        """scatter plot of data for given month"""

        ## prep func
        if by_season:
            get_season = lambda x: src.utils.sel_month(
                x.resample({"time": "QS-JAN"}).mean(), month
            )

        else:
            get_season = lambda x: src.utils.sel_month(x, month)

        prep = lambda x: get_season(x).transpose("time", "member")

        ## get plot data
        plot_data = (prep(anom_[xvar]), prep(anom_[yvar]))

        ## get stats
        corr = xr.corr(*plot_data)
        cov = xr.cov(*plot_data)
        m = cov / plot_data[0].var()

        ## get label
        if verbose:
            legend=f"m = {m.item():.1e}\nr = {corr.item():.2f}"
        else:
            legend=None

        ## plot data
        ax.scatter(*plot_data, s=3, label=legend, **kwargs)
        ax.set_title(f"{label}")

        ## formatting
        ax_kwargs = dict(ls="--", c="gray", lw=0.5)
        ax.axvline(0, **ax_kwargs)
        ax.axhline(0, **ax_kwargs)

        if verbose:
            ax.legend(prop=dict(size=10))

        return ax


    def make_scatter(anom_, xvar, yvar, month, by_season=True, verbose=True, **kwargs):
        """scatter plot of data for given month"""

        fig, axs = plt.subplots(1, 4, figsize=(11, 2.5), layout="constrained")

        if "year" in anom_:
            for ax, year in zip(
                axs,
                [1870, 2010, 2050, 2080],
            ):

                ## scatter plot of data
                ax = make_scatter_ax(
                    ax,
                    anom_.sel(year=year, method="nearest"),
                    xvar=xvar,
                    yvar=yvar,
                    month=month,
                    by_season=by_season,
                    label=year,
                    verbose=verbose,
                    **kwargs,
                )

        else:

            for ax, t_idx, label in zip(
                axs,
                [["1850", "1879"], ["1995", "2024"], ["2035", "2064"], ["2071", "2100"]],
                ["1865", "2010", "2050", "2085"],
            ):

                ## helper func
                prep = lambda x: x.sel(time=slice(*t_idx))

                ## scatter plot of data
                ax = make_scatter_ax(
                    ax,
                    prep(anom_),
                    xvar=xvar,
                    yvar=yvar,
                    month=month,
                    by_season=by_season,
                    label=label,
                )

        ## format/scale axes
        src.utils.set_lims(axs)
        for ax in axs[1:]:
            ax.set_yticks([])

        return fig, axs


    def plot_timeseries(coefs, sel_fn=lambda x: x):
        """plot timeseries comparison"""

        fig, axs = plt.subplots(1, 3, figsize=(8, 2.5))

        ## loop thru pos and negative
        for i, (name, color) in enumerate(zip(["pos", "neg"], ["r", "b"])):

            ## plot median and bounds
            for q, lw in zip([0.5, 0.1, 0.9], [2, 0.6, 0.6]):

                ## plot neutral and pos/or neg
                for name_, color_ in zip(["all", name], ["k", color]):

                    ## finally, plot data
                    axs[i].plot(
                        coefs.year,
                        sel_fn(coefs)[name_].quantile(q=q, dim="member"),
                        c=color_,
                        lw=lw,
                    )

                ## plot on shared axs
                axs[2].plot(
                    coefs.year,
                    sel_fn(coefs)[name].quantile(q=q, dim="member"),
                    c=color,
                    lw=lw,
                )

        ## formatting
        for ax in axs:
            ax.set_xticks([1870, 2010, 2080])
            ax_kwargs = dict(ls="--", c="gray", lw=0.8)
            ax.axvline(2010, **ax_kwargs)
            ax.axhline(0, **ax_kwargs)
        src.utils.set_lims(axs)

        for ax in axs[1:]:
            ax.set_yticks([])

        return fig, axs


    def plot_zonal_structure(coefs, sel_fn=lambda x: x):
        """plot zonal structure of coefficients over time"""

        fig, axs = plt.subplots(1, 4, figsize=(10, 2.5), layout="constrained")

        for ax, y in zip(axs, [1865, 2010, 2050, 2085]):

            ## get data for year
            ax.set_title(y)
            coefs_y = sel_fn(coefs).sel(year=y, method="nearest")

            ## select data
            for n, color in zip(["all", "pos", "neg"], ["k", "r", "b"]):

                ax.plot(coefs.longitude, coefs_y[n].mean("member"), c=color)

        src.utils.set_lims(axs)
        for ax in axs[1:]:
            ax.set_yticks([])

        for ax in axs:
            ax.set_xticks([140, 190, 240, 280])
            ax_kwargs = dict(ls="--", c="gray", lw=0.8)
            ax.axhline(0, **ax_kwargs)

        return fig, axs


    def add_vticks(axs, xticks, xlines=None):
        """add vertical lines to axs"""

        ## specify line style
        ax_kwargs = dict(ls="--", c="gray", lw=0.8)

        ## loop thru axs
        for ax in axs:
            ax.set_xticks(xticks)
            if xlines is not None:
                for x0 in xlines:
                    ax.axvline(x0, **ax_kwargs)

        return


    def plot_vertical_structure(coefs, sel_fn):
        """plot vertical structure of coefficients over time"""

        fig, axs = plt.subplots(1, 4, figsize=(10, 2.5), layout="constrained")

        for ax, y in zip(axs, [1865, 2010, 2050, 2085]):

            ## get data for year
            ax.set_title(y)
            coefs_y = sel_fn(coefs).sel(year=y, method="nearest")

            ## select data
            for n, color in zip(["all", "pos", "neg"], ["k", "r", "b"]):

                ax.plot(coefs_y[n].mean("member"), coefs.z_t, c=color)

        for ax in axs:
            ax.set_ylim([150, 5])

            ax_kwargs = dict(ls="--", c="gray", lw=0.8)
            ax.axhline(50, **ax_kwargs)
            ax.axhline(80, **ax_kwargs)
            ax.axvline(0, **ax_kwargs)

        src.utils.set_lims(axs)
        axs[0].set_yticks([0, 50, 80])
        for ax in axs[1:]:
            ax.set_yticks([])
        return fig, axs

    return (
        add_vticks,
        make_scatter,
        make_scatter_ax,
        plot_timeseries,
        plot_zonal_structure,
    )


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
def _(anom, forced, load_consolidated_wide):
    ## should we use "wide" data?
    USE_WIDE = False

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
    LONS_W = dict(longitude=slice(120, 180))
    LONS_TAU = dict(longitude=slice(150, 230))
    return LATS, LATS_H, LONS_E, LONS_TAU, LONS_W


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set funcs
    """)
    return


@app.cell
def _(
    ENT_DEPTH_RANGE,
    LATS,
    LATS_H,
    LONS_E,
    LONS_TAU,
    LONS_W,
    MLD,
    T_VAR,
    USE_T_ML,
    anom,
    src,
):
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
    ENT_AVG = lambda x: x.sel(z_t=ENT_DEPTH_RANGE).mean("z_t")
    ML_AVG = lambda x: x.sel(z_t=slice(None, MLD)).mean("z_t")

    ## get T3 volume avg
    T3_ENT_AVG = lambda x: ENT_AVG(LON_AVG(x))
    T3_ML_AVG = lambda x: ML_AVG(LON_AVG(x))
    T34_ML_AVG = lambda x: ML_AVG(LON_AVG_34(x))

    if USE_T_ML:
        anom["T_3"] = src.utils.reconstruct_wrapper(
            anom[["T", "T_comp"]],
            fn=T3_ML_AVG,
        )["T"]

        anom["T_34"] = src.utils.reconstruct_wrapper(
            anom[["T", "T_comp"]],
            fn=T34_ML_AVG,
        )["T"]


    ## get averaging function
    if T_VAR == "T_3":
        REDUCE_X = LON_AVG
    else:
        REDUCE_X = LON_AVG_34
    return ENT_AVG, REDUCE_X


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
    ## handle rectification
    """)
    return


@app.cell
def _(Th, anom, forced, src, window):
    ## get variables to check
    rect_vars = ["u", "w", "T", "sst", "taux", "nhf"]

    ## get total
    total = forced[rect_vars] + anom[rect_vars]

    ## window it
    total = window(total, subtract_median=False).groupby("time.month").mean()

    ## add sigma T info
    total["sigma_T"] = window(Th["T_34"]).groupby("time.month").std()

    ## estimate rect. coefficients
    rect_coefs = src.utils.estimate_rect_bymonth(total, xvar="sigma_T", yvars=rect_vars)

    ## get climatology
    clim = total.mean("member")
    clim_norect = rect_coefs.sel(degree=0)
    return clim, clim_norect


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### T vs. ddt(SST)$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load coefficients
    """)
    return


@app.cell
def _(T_VAR, USE_T_ML, coefs, xr):
    ## specify which longitudes to reduce over
    if T_VAR == "T_34":
        ddt_LON_RANGE = [190, 240]

    else:
        ddt_LON_RANGE = [210, 270]

    ## specify which variable to use ("T" or "sst")
    if USE_T_ML:
        YVAR = "T"
    else:
        YVAR = "sst"

    ## function to select data
    sel_var = lambda x: x[f"ddt_{YVAR}"].sel(j=T_VAR)

    ## load coefs
    R_proj = 12 * xr.merge(sel_var(c).rename(n) for n, c in coefs.items())
    return R_proj, YVAR, ddt_LON_RANGE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot spatial structure
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    reconstruct spatial strucure
    """)
    return


@app.cell
def _(MLD, anom, ddt_LON_RANGE, src, xr):
    def reduce_yz(x):

        if "z_t" not in x.dims:
            x = x.expand_dims("z_t")

        if "latitude" not in x.dims:
            x = x.expand_dims("latitude")

        idx = dict(z_t=slice(None, MLD), latitude=slice(-5, 5))

        return x.sel(idx).mean(["z_t", "latitude"])


    def reduce_x(x):

        ## get averaging dims
        idx = dict(longitude=slice(*ddt_LON_RANGE))

        return x.sel(idx).mean("longitude")


    def recon_struct(coefs, v="T"):
        """reconstruct spatial structure of coefficients"""

        coefs_struct = src.utils.reconstruct_wrapper(
            xr.merge([coefs.rename(v), anom[f"{v}_comp"]]),
            fn=reduce_yz,
        )

        return coefs_struct[v]


    def recon_idx(coefs, v="T"):
        """reconstruct spatial structure of coefficients"""

        coefs_struct = src.utils.reconstruct_wrapper(
            xr.merge([coefs.rename(v), anom[f"{v}_comp"]]),
            fn=lambda y: reduce_x(reduce_yz(y)),
        )

        return coefs_struct[v]

    return recon_idx, recon_struct, reduce_x


@app.cell
def _(R_proj, YVAR, recon_struct, xr):
    ## load coefs
    R_struct = xr.merge(
        [recon_struct(R_proj[v], YVAR).rename(v) for v in ["all", "pos", "neg"]]
    )
    return (R_struct,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot
    """)
    return


@app.cell
def _(R_struct, SEL_MONTH, add_vticks, plot_zonal_structure, plt):
    fig, axs = plot_zonal_structure(R_struct, sel_fn=SEL_MONTH)
    for _ax in axs:
        _ax.set_xlim([140, 280])

    add_vticks(axs, xticks=[140, 190, 240, 280], xlines=[190, 240])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Timeseries
    """)
    return


@app.cell
def _(R_struct, plot_timeseries, plt, reduce_x):
    ## get index
    R = reduce_x(R_struct)

    # fig, axs = plot_timeseries(coefs=R, sel_fn=SEL_MONTH)
    fig_1, axs_1 = plot_timeseries(coefs=R, sel_fn=lambda x : x.mean("month"))
    plt.gca()
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter plot
    """)
    return


@app.cell
def _(MONTH0, T_VAR, YVAR, anom, get_ddt, make_scatter, plt, recon_idx, xr):
    scatter_data = xr.merge(
        [
            get_ddt(recon_idx(anom[YVAR], YVAR)).rename(f"ddt_{YVAR}"),
            anom[["T_3", "T_34", "h_w"]],
        ]
    )

    for _xvar in [T_VAR, "h_w"]:
        fig_2, axs_2 = make_scatter(scatter_data, xvar=_xvar, yvar=f"ddt_{YVAR}", month=MONTH0)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compare to RO
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Get data
    """)
    return


@app.cell
def _(T_VAR, Th, anom, get_ddt, window, xr):
    ## get data (remove median as well)
    x = xr.merge([Th[T_VAR], anom["h_w"]])
    # x /= x.std()
    x = window(x)
    y = get_ddt(x).rename({f"{v}": f"ddt_{v}" for v in list(x)})
    z = xr.merge([x, y])
    return x, y, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Version 1: piecewise linear
    """)
    return


@app.cell
def _(T_VAR, src, x, xr, y, z):
    ## do regression
    # z_pos = z.where(z[T_VAR]>0)

    _kwargs = dict(
        func=src.utils.regress_xr_proj,
        x_vars=list(x),
        y_vars=list(y),
        dims=["time", "member"],
    )

    regress_by_month = lambda x: 12 * x.groupby("time.month").map(**_kwargs)

    ## find pos/neg
    is_pos_ = z[T_VAR] > 0
    is_neg_ = z[T_VAR] < 0

    ## do regression
    coefs2_all = regress_by_month(z)
    coefs2_pos = regress_by_month(z.where(is_pos_))
    coefs2_neg = regress_by_month(z.where(is_neg_))
    coefs2 = [coefs2_all, coefs2_pos, coefs2_neg]

    ## get R, epsilon
    get_R = lambda x: x[f"ddt_{T_VAR}"].sel(j=T_VAR)
    get_eps = lambda x: -x["ddt_h_w"].sel(j="h_w")
    R2 = xr.merge([get_R(c).rename(n) for c, n in zip(coefs2, ["all", "pos", "neg"])])
    return (R2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check reconstruction matches
    """)
    return


@app.cell
def _(R, R2, add_vticks, plt, src):
    fig_3, axs_3 = plt.subplots(1, 3, figsize=(8.5, 2.5), layout="constrained")
    for _ax, _n in zip(axs_3, list(R)):
        _ax.plot(R2.year, R2[_n].mean("month"))
        _ax.plot(R.year, R[_n].mean(["month", "member"]), ls="--")
        _ax.axhline(0, ls="--", c="k", lw=0.8)
        _ax.set_title(_n)

    src.utils.set_lims(axs_3)
    for _ax in axs_3[1:]:
        _ax.set_yticks([])

    add_vticks(axs_3, xticks=[1870, 2010, 2090], xlines=[2010, 2090])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### version 2: cubic
    """)
    return


@app.cell
def _(tqdm, xr):
    import warnings


    def get_fits_over_time(data_rolling, model, by_member=False, **fit_kwargs):
        """Get RO fits for each ensemble member as a function of time."""

        ## empty list to hold results
        fits = []

        ## loop through years
        for y in tqdm.tqdm(data_rolling.year):

            ## get data for year
            data_y = data_rolling.sel(year=y)

            if by_member:

                ## separate fit for each ensemble member
                fits_ = []
                for m in data_rolling.member:
                    with warnings.catch_warnings(action="ignore"):
                        fits_.append(model.fit_matrix(data_y.sel(member=m), **fit_kwargs))
                fit = xr.concat(fits_, dim=data_y.member)

            else:

                ## fit for all ensemble members together
                with warnings.catch_warnings(action="ignore"):
                    fit = model.fit_matrix(data_y, **fit_kwargs)

            ## track fits
            fits.append(fit.drop_vars(["time", "X", "Y", "Yfit"]))

        ## put back in xarray
        fits = xr.concat(fits, dim=data_rolling.year)

        return fits

    return (get_fits_over_time,)


@app.cell
def _(get_fits_over_time, src, x):
    ## parameters for fitting
    MODEL = src.XRO.XRO(ncycle=12, ac_order=4, is_forward=True)
    fit_kwargs = dict(ac_mask_idx=None, maskNT=["T2", "T3"], maskNH=[])

    ## get fits
    fits = get_fits_over_time(
        x,
        model=MODEL,
        by_member=False,
        **fit_kwargs,
    )

    ## extract parameters
    params = src.utils.get_params(fits=fits, model=MODEL)

    ## get change from initial period
    delta_params = params - params.isel(year=0)
    return fits, params


@app.cell
def _(T_VAR, fits, np, x, xr):
    ## get coefficients
    coefs_nl = fits["NROT_Lac"].sel(nro_form=["T2", "T3"])
    R3 = fits["Lac"].isel(ranky=0, rankx=0).expand_dims(nro_form=["T"])
    eps3 = -fits["Lac"].isel(ranky=1, rankx=1)
    coefs3 = xr.concat([R3, coefs_nl], dim="nro_form").rename({"nro_form": "form"})

    ## get coefficient array
    # T0 = 1.5
    sigma = x[T_VAR].std().values.item()
    w = sigma * np.linspace(-3, 3)
    W = np.stack([w**i for i in range(3)], axis=1)
    W = xr.DataArray(W, coords=dict(T=w, form=coefs3.form), dims=["T", "form"])

    ## reconstruct nonlinear R
    R_nl = (coefs3 * W).sum("form")
    R_nl_pos = R_nl.where(R_nl["T"] > 0).mean("T")
    R_nl_neg = R_nl.where(R_nl["T"] < 0).mean("T")


    # T0 = sigma * 1.5
    T0 = 1.5
    R_nl_pos2 = R_nl.sel(T=T0, method="nearest")
    R_nl_neg2 = R_nl.sel(T=-T0, method="nearest")
    return R_nl, R_nl_neg, R_nl_neg2, R_nl_pos, R_nl_pos2, coefs_nl


@app.cell
def _(
    R,
    R2,
    R_nl,
    R_nl_neg,
    R_nl_neg2,
    R_nl_pos,
    R_nl_pos2,
    add_vticks,
    params,
    plt,
    src,
):
    fig_4, axs_4 = plt.subplots(1, 3, figsize=(8.5, 2.5), layout="constrained")
    for _ax, _n in zip(axs_4, list(R)):
        _ax.plot(R2.year, R2[_n].mean("month"))
        _ax.plot(R.year, R[_n].mean(["month", "member"]), ls="--")
        _ax.axhline(0, ls="--", c="k", lw=0.8)
        _ax.set_title(_n)

    axs_4[0].plot(params.year, params["R"].mean("cycle"), label=r"$R_0$")
    axs_4[0].plot(params.year, R_nl.mean(["cycle", "T"]), label=r"avg($R$)")


    axs_4[1].plot(params.year, R_nl_pos.mean("cycle"), c="gray")
    axs_4[1].plot(params.year, R_nl_pos2.mean("cycle"), c="gray", ls="--")

    axs_4[2].plot(params.year, R_nl_neg.mean("cycle"), c="gray")
    axs_4[2].plot(params.year, R_nl_neg2.mean("cycle"), c="gray", ls="--")

    axs_4[0].legend(prop=dict(size=8))
    src.utils.set_lims(axs_4)
    for _ax in axs_4[1:]:
        _ax.set_yticks([])

    add_vticks(axs_4, xticks=[1870, 2010, 2090], xlines=[2010, 2050, 2090])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    \begin{align}
        R &= R_0 + R_1T + R_2 T^2
    \end{align}
    """)
    return


@app.cell
def _(coefs_nl, params, plt):
    sel = lambda x: x.mean("cycle")  # .differentiate("year")
    fig_5, _ax = plt.subplots(figsize=(3, 2.5))
    _ax.plot(params.year, 0.2 * sel(params["R"]), c="k")
    _ax.plot(coefs_nl.year, sel(coefs_nl.sel(nro_form="T2")))
    _ax.plot(coefs_nl.year, sel(coefs_nl.sel(nro_form="T3")))
    _ax.axhline(0, ls="--", lw=0.8, c="gray")
    _ax.axvline(2050, ls="--", lw=0.8, c="gray")
    plt.gca()
    return


@app.cell
def _(R_nl, R_nl_neg, R_nl_neg2, R_nl_pos, R_nl_pos2, add_vticks, params, plt):
    fig_6, _ax = plt.subplots(figsize=(3, 2.5), layout="constrained")

    _ax.plot(params.year, R_nl.mean(["T", "cycle"]), c="k")
    _ax.plot(params.year, R_nl_pos.mean("cycle"), c="r")
    _ax.plot(params.year, R_nl_neg.mean("cycle"), c="b")

    _ax.plot(params.year, params["R"].mean("cycle"), c="k", ls="--")
    _ax.plot(params.year, R_nl_pos2.mean("cycle"), c="r", ls="--")
    _ax.plot(params.year, R_nl_neg2.mean("cycle"), c="b", ls="--")

    add_vticks([_ax], xticks=[1870, 2010, 2090], xlines=[2010, 2050])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    color by magnitude of anomaly...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### T vs. Tsub
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load coefficients
    """)
    return


@app.cell
def _(DIMS, T_VAR, anom, coefs, regress_over_time, window, xr):
    LOAD_GAMMA = True

    if LOAD_GAMMA:

        ## load coefs
        gamma_proj = xr.merge(c["T"].rename(n) for n, c in coefs.items())
        # gamma_proj = xr.merge(c["T"].rename(n) for n, c in coefs_T.items())

    else:

        ## get data for regression
        gamma_data = window(anom[["T", T_VAR, "h_w"]])
    
        _regress_kwargs = dict(x_vars=[T_VAR, "h_w"], y_vars=["T"], dims=DIMS)
        _regress_helper = lambda x: regress_over_time(x, **_regress_kwargs)
    
        ## do the regression
        print("regress all")
        gamma_proj_all = _regress_helper(gamma_data)
        print("regress pos")
        gamma_proj_pos = _regress_helper(gamma_data.where(gamma_data[T_VAR] > 0))
        print("regress neg")
        gamma_proj_neg = _regress_helper(gamma_data.where(gamma_data[T_VAR] < 0))


        ## merge into single dataarray
        gamma_proj = xr.merge(
            [gamma_proj_all.rename({"T":"all"}), 
             gamma_proj_pos.rename({"T":"pos"}), 
             gamma_proj_neg.rename({"T":"neg"}),
            ]
        )
    return (gamma_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot structure
    """)
    return


@app.cell(hide_code=True)
def _(ENT_DEPTH_RANGE, MLD, REDUCE_X, anom, src, xr):
    def reduce_x_1(x):
        """Get Tsub, but don't average over depth yet"""

        return REDUCE_X(x)


    def reduce_z(x, use_dT):
        """Get diff b/n Tsub and Tsurf, but don't average over longitude yet"""

        Tsurf = x.sel(z_t=slice(None, MLD)).mean("z_t")
        Tsub = x.sel(z_t=ENT_DEPTH_RANGE).mean("z_t")

        if use_dT:
            return Tsub - Tsurf

        else:
            return Tsub


    def recon_struct_1(coefs, use_dT=False):
        """post-process regression coefficients"""

        ## reconstruct mean over equatorial region
        coefs_ = src.utils.reconstruct_wrapper(
            xr.merge([coefs.rename("T"), anom["T_comp"]]),
            fn=lambda x : reduce_z(x, use_dT=use_dT),
        )
        ## drop extra variables
        mu = coefs_["T"].squeeze(drop=True)

        return mu


    def recon_idx_1(coefs, use_dT):
        """reconstruct spatial structure of coefficients"""

        coefs_struct = src.utils.reconstruct_wrapper(
            xr.merge([coefs["T"], anom[f"T_comp"]]),
            fn=lambda y: reduce_x_1(reduce_z(y, use_dT=use_dT)),
        )

        return coefs_struct["T"]


    def get_wsub(x):
        """Get upwelling at base of mixed layer"""

        wsub = x.sel(z_t=ENT_DEPTH_RANGE).mean("z_t")

        return wsub

    return get_wsub, recon_idx_1, recon_struct_1, reduce_x_1


@app.cell
def _(MLD, clim, forced, gamma_proj, get_wsub, recon_struct_1, src, xr):
    ## should we use dT?
    USE_dT = False

    ## get wbar
    w_bar_struct = src.utils.reconstruct_wrapper(
        xr.merge([clim["w"], forced["w_comp"]]),
        fn=get_wsub,
    )["w"]

    ## load coefs
    gamma_struct = xr.merge(
        [recon_struct_1(gamma_proj[v], use_dT=USE_dT).rename(v) for v in ["all", "pos", "neg"]]
    )

    ## handle dT
    # if not USE_dT:
    #     gamma_struct = gamma_struct-1

    # ## compute THF
    # first, scale upwelling by mixed layer depth (50 m) and convert from m/mo to m/yr
    w_H = 12 * w_bar_struct / MLD
    thf_struct = gamma_struct * w_H
    return gamma_struct, thf_struct, w_H, w_bar_struct


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make sure dimensions are ok
    """)
    return


@app.cell
def _(T_VAR, gamma_struct, thf_struct, w_H):
    if "j" not in list(gamma_struct.dims):
        gamma_struct_1 = gamma_struct.expand_dims(j=[T_VAR])
        w_H_1 = w_H.expand_dims(j=[T_VAR])
        thf_struct_1 = thf_struct.expand_dims(j=[T_VAR])
    return gamma_struct_1, thf_struct_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Subsurface $T$
    """)
    return


@app.cell
def _(SEL_MONTH, add_vticks, gamma_struct_1, plot_zonal_structure, plt):
    for _v in list(gamma_struct_1.j.values):

        fig_7, axs_5 = plot_zonal_structure(coefs=gamma_struct_1.sel(j=_v), sel_fn=SEL_MONTH)
        add_vticks(axs_5, xticks=[190, 240, 280], xlines=[190, 240])
        axs_5[0].set_ylabel(r"$T_{\text{sub}}~/~T-1$")

        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    thermocline feedback
    """)
    return


@app.cell
def _(
    SEL_MONTH,
    add_vticks,
    gamma_struct_1,
    plot_zonal_structure,
    plt,
    thf_struct_1,
):
    for _v in gamma_struct_1.j.values:
        print(_v)

        fig_8, axs_6 = plot_zonal_structure(coefs=thf_struct_1.sel(j=_v), sel_fn=SEL_MONTH)
        add_vticks(axs_6, xticks=[190, 240, 280], xlines=[190, 240])
        axs_6[0].set_ylabel(r"$T_{\text{sub}}~/~T-1$")

        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot timeseries
    """)
    return


@app.cell
def _(
    SEL_MONTH,
    T_VAR,
    gamma_struct_1,
    plot_timeseries,
    plt,
    reduce_x_1,
    thf_struct_1,
    w_bar_struct,
    xr,
):
    ## compute gamma
    print("gamma")
    gamma = reduce_x_1(gamma_struct_1).sel(j=T_VAR)
    fig_9, axs_7 = plot_timeseries(coefs=gamma, sel_fn=SEL_MONTH)
    plt.show()

    ## compute THF
    print(f"\nTHF")
    thf = reduce_x_1(thf_struct_1)
    fig_9, axs_7 = plot_timeseries(coefs=thf.sel(j=T_VAR), sel_fn=SEL_MONTH)
    plt.show()

    print("wbar")
    w_bar = reduce_x_1(w_bar_struct)
    fig_9, axs_7 = plot_timeseries(
        coefs=w_bar * xr.ones_like(gamma), sel_fn=SEL_MONTH
    )
    plt.gca()
    return gamma, thf, w_bar


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter
    """)
    return


@app.cell
def _(T_VAR, anom, make_scatter, plt, recon_idx_1, xr):
    for u in [True, False]:
        print(f"use_dT={u}")
        scatter_data_1 = xr.merge(
            [
                recon_idx_1(anom[["T"]], use_dT=u),
                anom[["T_3", "T_34", "h_w"]],
            ]
        )

        for _xvar in [T_VAR, "h_w"]:
            fig_10, axs_8 = make_scatter(scatter_data_1, xvar=_xvar, yvar=f"T", month=4, by_season=False)
            plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Damping
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load data
    """)
    return


@app.cell
def _(MLD, coefs_T, xr):
    ## get scaling factor to convert to damping rate (units of K/mo)
    sec_per_mo = 8.64e4 * 30
    rho = 1.02e3
    Cp = 4.2e3
    H0 = MLD
    scale_fac = sec_per_mo / (rho * Cp * H0)

    ## helper func to get data
    prep = lambda x: -12 * x["nhf"] * scale_fac

    ## load coefs
    alpha_proj = xr.merge(prep(c).rename(n) for n, c in coefs_T.items())
    return (alpha_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### structure
    """)
    return


@app.cell
def _(REDUCE_X, anom, src, xr):
    def reduce_x_2(x):
        """Get Tsub, but don't average over depth yet"""

        return REDUCE_X(x)
        # return LON_AVG_34(x)


    def reduce_y(x):
        """Get diff b/n Tsub and Tsurf, but don't average over longitude yet"""

        return x.sel(latitude=slice(-5, 5)).mean("latitude")


    def recon_struct_2(coefs):
        """post-process regression coefficients"""

        ## reconstruct mean over equatorial region
        coefs_ = src.utils.reconstruct_wrapper(
            xr.merge([coefs.rename("nhf"), anom["nhf_comp"]]),
            fn=reduce_y,
        )
        ## drop extra variables
        coefs_ = coefs_["nhf"].squeeze(drop=True)

        return coefs_


    def recon_idx_2(coefs):
        """reconstruct spatial structure of coefficients"""

        coefs_struct = src.utils.reconstruct_wrapper(
            xr.merge([coefs["nhf"], anom[f"nhf_comp"]]),
            fn=lambda z: reduce_x_2(reduce_y(z)),
        )

        return coefs_struct["nhf"]

    return recon_idx_2, recon_struct_2, reduce_x_2


@app.cell
def _(alpha_proj, recon_struct_2, xr):
    ## load coefs
    alpha_struct = xr.merge(
        [recon_struct_2(alpha_proj[v]).rename(v) for v in ["all", "pos", "neg"]]
    )
    return (alpha_struct,)


@app.cell
def _(add_vticks, alpha_struct, plot_zonal_structure, plt):
    sel_1 = lambda x : x.sel(month=slice(5,12)).mean("month")
    fig_11, axs_9 = plot_zonal_structure(coefs=alpha_struct, sel_fn=sel_1)
    add_vticks(axs_9, xticks=[190, 240, 280], xlines=[190, 240])
    axs_9[0].set_ylabel(r"$\alpha$")
    for _ax in axs_9:
        _ax.set_xlim([140, 280])
    plt.gca()
    return (sel_1,)


@app.cell
def _(alpha_struct, plt, sel_1, src):
    alph = alpha_struct.mean("member")
    fig_12,axs_10 = plt.subplots(1,3,figsize=(9,3))

    for _ax, _n in zip(axs_10, ["pos","all","neg"]):

        _ax.plot(alph.longitude, sel_1(alph[_n]).sel(year=1870, method="nearest"), c="k")
        _ax.plot(alph.longitude, sel_1(alph[_n]).sel(year=2000, method="nearest"), c="r")
        _ax.set_xlim([120,280])
        _ax.set_yticks([])
        _ax.axhline(0, ls="--", c="k", lw=1)
        _ax.set_title(_n)

    src.utils.add_vticks(axs_10, xticks=[210,270], xlines=[210,270])
    axs_10[0].set_yticks([0, 2])
    src.utils.set_lims(axs_10)

    # src.utils.
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Timeseries
    """)
    return


@app.cell
def _(alpha_struct, plot_timeseries, plt, reduce_x_2, sel_1):
    alpha = reduce_x_2(alpha_struct)

    # fig, axs = plot_timeseries(coefs=alpha, sel_fn=SEL_MONTH)
    fig_13, axs_11 = plot_timeseries(coefs=alpha, sel_fn=sel_1)
    plt.gca()
    return (alpha,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter
    """)
    return


@app.cell
def _(MONTH0, T_VAR, anom, make_scatter, plt, recon_idx_2, xr):
    scatter_data_2 = xr.merge(
        [
            recon_idx_2(anom[["nhf"]]),
            anom[["T_3", "T_34"]],
        ]
    )
    fig_14, axs_12 = make_scatter(scatter_data_2, xvar=T_VAR, yvar="nhf", month=MONTH0)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### wind stress
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load data
    """)
    return


@app.cell
def _(coefs_T, xr):
    ## load coefs
    mu_proj = xr.merge(c["taux"].rename(n) for n, c in coefs_T.items())
    return (mu_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### structure
    """)
    return


@app.cell
def _(anom, src, xr):
    def reduce_x_3(x):
        """Get Tsub, but don't average over depth yet"""

        return x.sel(longitude=slice(150, 270)).mean("longitude")
        # return x.sel(longitude=slice(230, 270)).mean("longitude")


    def reduce_y_1(x):
        """Get diff b/n Tsub and Tsurf, but don't average over longitude yet"""

        return x.sel(latitude=slice(-5, 5), longitude=slice(140, 280)).mean("latitude")


    def recon_struct_3(coefs):
        """post-process regression coefficients"""

        ## reconstruct mean over equatorial region
        coefs_ = src.utils.reconstruct_wrapper(
            xr.merge([coefs.rename("taux"), anom["taux_comp"]]),
            fn=reduce_y_1,
        )
        ## drop extra variables
        coefs_ = coefs_["taux"].squeeze(drop=True)

        return coefs_


    def recon_taux_idx(coefs):
        """reconstruct spatial structure of coefficients"""

        coefs_struct = src.utils.reconstruct_wrapper(
            xr.merge([coefs["taux"], anom["taux_comp"]]),
            fn=lambda z: reduce_x_3(reduce_y_1(z)),
        )

        return coefs_struct["taux"]

    return recon_struct_3, recon_taux_idx, reduce_x_3


@app.cell
def _(mu_proj, recon_struct_3, xr):
    ## load coefs
    mu_struct = xr.merge(
        [recon_struct_3(mu_proj[v]).rename(v) for v in ["all", "pos", "neg"]]
    )
    return (mu_struct,)


@app.cell
def _(SEL_MONTH, add_vticks, mu_struct, plot_zonal_structure, plt):
    fig_15, axs_13 = plot_zonal_structure(coefs=mu_struct, sel_fn=SEL_MONTH)

    add_vticks(axs_13, xticks=[150, 210, 270], xlines=[150, 210, 270])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Timeseries
    """)
    return


@app.cell
def _(SEL_MONTH, mu_struct, plot_timeseries, plt, reduce_x_3):
    mu = reduce_x_3(mu_struct)

    # fig, axs = plot_timeseries(coefs=alpha, sel_fn=SEL_MONTH)
    fig_16, axs_14 = plot_timeseries(coefs=mu, sel_fn=SEL_MONTH)
    plt.gca()
    return (mu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter
    """)
    return


@app.cell
def _(MONTH0, T_VAR, anom, make_scatter, plt, recon_taux_idx, xr):
    scatter_data_3 = xr.merge(
        [
            recon_taux_idx(anom[["taux"]]),
            anom[["T_3", "T_34"]],
        ]
    )
    fig_17, axs_15 = make_scatter(scatter_data_3, xvar=T_VAR, yvar="taux", month=MONTH0)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Scatter plot of $\tau_x$ vs. $w$
    """)
    return


@app.cell
def _():
    def get_taux_idx(x):
        idx = dict(longitude=slice(230,270), latitude=slice(-5,5))
        return x.sel(idx).mean(["longitude","latitude"])

    def get_w_idx(x):
        idx = dict(longitude=slice(230,270), z_t=slice(50,70))
        return x.sel(idx).mean(["longitude","z_t"])

    return get_taux_idx, get_w_idx


@app.cell
def _(anom, get_taux_idx, get_w_idx, src, xr):
    plot_data = xr.merge(
        [
            anom["T_3"],
            src.utils.reconstruct_wrapper(anom[["w","w_comp"]], get_w_idx),
            src.utils.reconstruct_wrapper(anom[["taux","taux_comp"]], get_taux_idx),
            src.utils.reconstruct_wrapper(anom[["sst","sst_comp"]], get_taux_idx),
        ]
    )
    return (plot_data,)


@app.cell
def _(MONTH0, make_scatter, plot_data, plt):
    for _xvar in ["T_3","sst","taux"]:
        print(f"\n{_xvar}")
        fig_18, axs_16 = make_scatter(plot_data, xvar=_xvar, yvar="w", month=MONTH0)
        plt.show()
    return


@app.cell
def _(plot_data, src):
    plot_windowed = src.utils.get_windowed(plot_data)
    sel_2 = lambda x : x["sst"].resample({"time":"QS-JAN"}).mean().isel(time=slice(4,-4,4)).values.flatten()
    return plot_windowed, sel_2


@app.cell
def _(np, plot_windowed, plt, sel_2, src):
    edges = np.linspace(-3.25,4,20)
    pdf0,_  = src.utils.get_empirical_pdf(sel_2(plot_windowed.sel(year=1870)), edges=edges)
    pdf1,_ = src.utils.get_empirical_pdf(sel_2(plot_windowed.sel(year=2010)), edges=edges)
    fig_19,_ax = plt.subplots(figsize=(6,3))
    _ax.stairs(pdf0, edges, lw=2)
    _ax.stairs(pdf1, edges, lw=2)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Add index to ```anom```
    """)
    return


@app.cell
def _(anom, recon_taux_idx):
    anom["taux_idx"] = recon_taux_idx(anom[["taux"]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sverdrup balance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute
    """)
    return


@app.cell
def _(anom, check_dims, src, xr):
    def reduce_y_2(x):
        """reduce over lat/lon/depth"""

        ## make sure dims are all there
        x_ = check_dims(x)

        ## specify indices
        idx = dict(longitude=slice(120, 280), latitude=slice(-5, 5))

        ## average over lats
        x_ = x_.sel(idx).mean("latitude")

        ## integrate (optionally)
        if len(x_.z_t) > 1:
            x_ = x_.integrate("z_t") / 300
        else:
            x_ = x_.squeeze(drop=True)

        return x_


    def postprocess_(coefs, T_var):
        """post-process regression coefficients"""

        ## get recon components
        comps = check_dims(anom[f"{T_var}_comp"])

        ## reconstruct mean over equatorial region
        coefs_ = src.utils.reconstruct_wrapper(
            xr.merge([coefs, comps]),
            fn=reduce_y_2,
        )

        ## drop extra variables
        coefs_ = coefs_[T_var].squeeze(drop=True)

        return coefs_

    return postprocess_, reduce_y_2


@app.cell
def _(DIMS, USE_OHC, anom, postprocess_, regress_over_time, window, xr):
    ## specify wind variable (one of "T_34", "taux_idx")
    x_var = "taux_idx"
    # x_var = T_VAR

    ## specify h-variable (one of "T", "ssh")
    if USE_OHC:
        y_var = "T"
    else:
        y_var = "ssh"

    ## get data for regression (OHC)
    beta_data = window(xr.merge([anom[y_var], anom[x_var]]))
    _regress_kwargs = dict(x_vars=[x_var], y_vars=[y_var], dims=DIMS)
    _regress_helper = lambda x: postprocess_(
        regress_over_time(x, **_regress_kwargs), T_var=y_var
    )

    ## do the regression
    print("regress all")
    _beta_all = _regress_helper(beta_data)
    print("regress pos")
    _beta_pos = _regress_helper(beta_data.where(beta_data[x_var] > 0))
    print("regress neg")
    _beta_neg = _regress_helper(beta_data.where(beta_data[x_var] < 0))

    ## merge into single dataarray
    beta_struct = xr.merge(
        [_beta_all.rename("all"), _beta_pos.rename("pos"), _beta_neg.rename("neg")]
    )
    return beta_data, beta_struct, x_var, y_var


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### spatial plot
    """)
    return


@app.cell
def _(SEL_MONTH, add_vticks, beta_struct, plot_zonal_structure, plt):
    fig_20, axs_17 = plot_zonal_structure(coefs=beta_struct, sel_fn=SEL_MONTH)

    add_vticks(axs_17, xticks=[120, 170, 210], xlines=[170, 210])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Compute inices
    """)
    return


@app.cell
def _(reduce_y_2):
    def get_he(x):
        """average over eastern Pacific"""

        ## lats/lons for averaging
        # idx = dict(longitude=slice(190, 240))
        idx = dict(longitude=slice(210, 270))

        return reduce_y_2(x).sel(idx).sum("longitude")


    def get_hw(x):
        """average over western Pacific"""

        ## lats/lons for averaging
        idx = dict(longitude=slice(120, 170))

        return reduce_y_2(x).sel(idx).sum("longitude")

        return


    def get_dh(x):
        """east-west difference"""

        return get_he(x) - get_hw(x)

    return get_dh, get_he, get_hw


@app.cell
def _(anom, beta_data, get_dh, get_he, get_hw, src, xr, y_var):
    # add to dataarray
    for _n, fn in zip(["he", "hw", "dh"], [get_he, get_hw, get_dh]):
        beta_data[_n] = src.utils.reconstruct_wrapper(
            xr.merge([beta_data[y_var], anom[f"{y_var}_comp"]]),
            fn=fn,
        )[y_var]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Make plot
    """)
    return


@app.cell
def _(beta_data, make_scatter, plt, x_var):
    ## specify kwargs
    _kwargs = dict(anom_=beta_data, xvar=x_var, month=7, by_season=True)

    for _yvar in ["dh", "he", "hw"]:
        print(_yvar)

        fig_21, axs_18 = make_scatter(yvar=_yvar, **_kwargs)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Timeseries
    """)
    return


@app.cell
def _(beta_struct, get_dh, get_he, plot_timeseries, plt):
    ## compute beta index
    beta_he = get_he(beta_struct)
    beta = get_dh(beta_struct)

    fig_22, axs_19 = plot_timeseries(coefs=beta, sel_fn=lambda x : x.mean("month"))
    plt.gca()
    return (beta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Thermal expansion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute
    """)
    return


@app.cell
def _(ENT_DEPTH_RANGE):
    ENT_DEPTH_RANGE
    return


@app.cell
def _(anom, src, xr):
    def reduce_z_1(x):
        """average over depth"""

        ## get indices
        # idx = dict(longitude=slice(120, 280), z_t=ENT_DEPTH_RANGE)
        idx = dict(longitude=slice(120, 280), z_t=slice(0,60,None))

        return x.sel(idx).mean("z_t")


    def postprocess_Tsub(coefs):
        """post-process regression coefficients"""

        ## reconstruct mean over equatorial region
        coefs_ = src.utils.reconstruct_wrapper(
            xr.merge([coefs, anom["T_comp"]]),
            fn=reduce_z_1,
        )

        ## drop extra variables
        mu = coefs_["T"].squeeze(drop=True)

        return mu

    return postprocess_Tsub, reduce_z_1


@app.cell
def _(anom, beta_data, window, xr):
    ## get data for regression
    ah_data = xr.merge([window(anom[["T","T_3", "h_w", "h_e"]]), beta_data["he"]])
    return (ah_data,)


@app.cell
def _(DIMS, ah_data, postprocess_Tsub, regress_over_time, xr):
    # regress_kwargs = dict(x_vars=["he"], y_vars=["T"], dims=DIMS)
    _regress_kwargs = dict(x_vars=["h_e"], y_vars=["T"], dims=DIMS)
    _regress_helper = lambda x: postprocess_Tsub(regress_over_time(x, **_regress_kwargs))

    ## do the regression
    print("regress all")
    ah_all = _regress_helper(ah_data)
    print("regress pos")
    ah_pos = _regress_helper(ah_data.where(ah_data["he"] > 0))
    print("regress neg")
    ah_neg = _regress_helper(ah_data.where(ah_data["he"] < 0))

    ## merge into single dataarray
    ah_struct = xr.merge([ah_all.rename("all"), ah_pos.rename("pos"), ah_neg.rename("neg")])
    return (ah_struct,)


@app.cell
def _(SEL_MONTH, add_vticks, ah_struct, plot_zonal_structure, plt):
    fig_23, axs_20 = plot_zonal_structure(coefs=ah_struct, sel_fn=SEL_MONTH)

    add_vticks(axs_20, xticks=[190, 240], xlines=[190, 240])

    plt.gca()
    return


@app.cell
def _(REDUCE_X, ah_data, anom, reduce_z_1, src, xr):
    def reduce_x_4(T):
        ## compute
        return REDUCE_X(T)


    def get_Tsub_idx(T):
        """compute Tsub over central/east Pacific"""

        ## specify averaging range
        idx = dict(longitude=slice(210, 270))

        ## compute
        return reduce_x_4(reduce_z_1(T))


    ## add to dataarray
    ah_data["Tsub"] = src.utils.reconstruct_wrapper(
        xr.merge([ah_data["T"], anom["T_comp"]]),
        fn=get_Tsub_idx,
    )["T"]
    return (reduce_x_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter
    """)
    return


@app.cell
def _(ah_data, make_scatter, plt):
    ## specify kwargs
    _kwargs = dict(anom_=ah_data, xvar="T_3", yvar="Tsub", month=10, by_season=True)

    fig_24, axs_21 = make_scatter(**_kwargs)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Time series
    """)
    return


@app.cell
def _(ah_struct, plot_timeseries, plt, reduce_x_4):
    SEL_MONTH_ah = lambda x : x.sel(month=slice(7,None)).mean("month")

    ## get index
    ah = reduce_x_4(ah_struct)

    fig_25, axs_22 = plot_timeseries(coefs=ah, sel_fn=SEL_MONTH_ah)
    for _ax in axs_22:
        _ax.set_ylim([1.3,None])
    plt.gca()
    return (ah,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### zonal velocity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load coefficients
    """)
    return


@app.cell
def _(coefs, xr):
    ## load coefs
    betau_proj = xr.merge(c["u"].rename(n) for n, c in coefs.items())
    return (betau_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot structure
    """)
    return


@app.cell
def _(MLD, REDUCE_X, anom, src, xr):
    def reduce_x_5(x):
        """Get Tsub, but don't average over depth yet"""

        return REDUCE_X(x)


    def reduce_z_2(x):
        """Average over ML don't average over longitude yet"""

        return x.sel(z_t=slice(None, MLD)).mean("z_t")


    def recon_struct_4(coefs):
        """post-process regression coefficients"""

        ## reconstruct mean over equatorial region
        coefs_ = src.utils.reconstruct_wrapper(
            xr.merge([coefs.rename("u"), anom["u_comp"]]),
            fn=reduce_z_2,
        )
        ## drop extra variables
        mu = coefs_["u"].squeeze(drop=True)

        return mu


    def recon_idx_3(coefs):
        """reconstruct spatial structure of coefficients"""

        coefs_struct = src.utils.reconstruct_wrapper(
            xr.merge([coefs["u"], anom[f"u_comp"]]),
            fn=lambda y: reduce_x_5(reduce_z_2(y)),
        )

        return coefs_struct["u"]

    return recon_idx_3, recon_struct_4, reduce_x_5, reduce_z_2


@app.cell
def _(betau_proj, clim_norect, forced, recon_struct_4, reduce_z_2, src, xr):
    ## load coefs
    betau_struct = xr.merge(
        [recon_struct_4(betau_proj[v]).rename(v) for v in ["all", "pos", "neg"]]
    )

    # ## Get zonal T profile
    T_bar = src.utils.reconstruct_wrapper(
        # xr.merge([clim["T"], forced["T_comp"]]),
        xr.merge([clim_norect["T"], forced["T_comp"]]),
        fn=lambda x: reduce_z_2(x),
    )["T"]

    ## compute zonal advective feedback
    ## mult. by 12 to convert from K/mo to K/yr
    zaf_struct = 12 * -src.utils.get_udTdx(u=betau_struct, T=T_bar)
    return betau_struct, zaf_struct


@app.cell
def _(SEL_MONTH, T_VAR, add_vticks, plot_zonal_structure, plt, zaf_struct):
    for _v in [T_VAR, "h_w"]:
        print(_v)

        fig_26, axs_23 = plot_zonal_structure(coefs=zaf_struct.sel(j=_v), sel_fn=SEL_MONTH)
        add_vticks(axs_23, xticks=[190, 240, 280], xlines=[190, 240])
        # axs[0].set_ylabel(r"$T_{\text{sub}}~/~T-1$")

        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot timeseries
    """)
    return


@app.cell
def _(SEL_MONTH, T_VAR, plot_timeseries, plt, reduce_x_5, zaf_struct):
    ## compute gamma
    zaf = reduce_x_5(zaf_struct)

    fig_27, axs_24 = plot_timeseries(coefs=zaf.sel(j=T_VAR), sel_fn=SEL_MONTH)
    plt.gca()
    return (zaf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter
    """)
    return


@app.cell
def _(MONTH0, T_VAR, anom, make_scatter, plt, recon_idx_3, xr):
    scatter_data_4 = xr.merge(
        [
            recon_idx_3(anom[["u"]]),
            anom[["T_3", "T_34", "h_w"]],
        ]
    )

    for _xvar in [T_VAR, "h_w"]:
        fig_28, axs_25 = make_scatter(scatter_data_4, xvar=_xvar, yvar="u", month=MONTH0)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Surface temp
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load coefficients
    """)
    return


@app.cell
def _(coefs, xr):
    ## load coefs
    betaT_proj = xr.merge(c["T"].rename(n) for n, c in coefs.items())
    return (betaT_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot structure
    """)
    return


@app.cell
def _(MLD, REDUCE_X, T_VAR, anom, src, xr):
    def reduce_x_6(x):
        """Get Tsub, but don't average over depth yet"""

        return REDUCE_X(x)


    def reduce_x_discrete(x):
        """Get Tsub, but don't average over depth yet"""

        if T_VAR == "T_3":
            lon_range = [210, 270]
        else:
            lon_range = [190, 240]

        return x.sel(longitude=lon_range[1], method="nearest") - x.sel(
            longitude=lon_range[0], method="nearest"
        )


    def reduce_z_3(x):
        """Average over ML don't average over longitude yet"""

        return x.sel(z_t=slice(None, MLD)).mean("z_t")


    def recon_struct_5(coefs):
        """post-process regression coefficients"""

        ## reconstruct mean over equatorial region
        coefs_ = src.utils.reconstruct_wrapper(
            xr.merge([coefs.rename("T"), anom["T_comp"]]),
            fn=reduce_z_3,
        )
        ## drop extra variables
        mu = coefs_["T"].squeeze(drop=True)

        return mu


    def recon_idx_4(coefs):
        """reconstruct spatial structure of coefficients"""

        coefs_struct = src.utils.reconstruct_wrapper(
            xr.merge([coefs["T"], anom[f"T_comp"]]),
            fn=lambda y: reduce_x_discrete(reduce_z_3(y)),
        )

        return coefs_struct["T"]

    return recon_idx_4, recon_struct_5, reduce_x_6, reduce_z_3


@app.cell
def _(betaT_proj, clim_norect, forced, recon_struct_5, reduce_z_3, src, xr):
    ## load coefs
    betaT_struct = xr.merge(
        [recon_struct_5(betaT_proj[v]).rename(v) for v in ["all", "pos", "neg"]]
    )

    ## Get u_bar
    u_bar = src.utils.reconstruct_wrapper(
        # xr.merge([clim["u"], forced["u_comp"]]),
        xr.merge([clim_norect["u"], forced["u_comp"]]),
        fn=lambda x: reduce_z_3(x),
    )["u"]

    ## compute dynamical damping
    # multiply by 12 to convert from K/mo to K/yr
    dd_struct = 12 * -src.utils.get_udTdx(u=u_bar, T=betaT_struct)
    return betaT_struct, dd_struct


@app.cell(hide_code=True)
def _(SEL_MONTH, T_VAR, add_vticks, dd_struct, plot_zonal_structure, plt):
    for _v in [T_VAR, "h_w"]:
        print(_v)

        fig_29, axs_26 = plot_zonal_structure(coefs=dd_struct.sel(j=_v), sel_fn=SEL_MONTH)
        add_vticks(axs_26, xticks=[190, 240, 280], xlines=[190, 240])
        # axs[0].set_ylabel(r"$T_{\text{sub}}~/~T-1$")

        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot timeseries
    """)
    return


@app.cell
def _(SEL_MONTH, T_VAR, dd_struct, plot_timeseries, plt, reduce_x_6):
    ## compute gamma
    dd = reduce_x_6(dd_struct)

    fig_30, axs_27 = plot_timeseries(coefs=dd.sel(j=T_VAR), sel_fn=SEL_MONTH)
    plt.gca()
    return (dd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter
    """)
    return


@app.cell
def _(MONTH0, T_VAR, anom, make_scatter, plt, recon_idx_4, xr):
    scatter_data_5 = xr.merge(
        [
            recon_idx_4(anom[["T"]]),
            anom[["T_3", "T_34", "h_w"]],
        ]
    )

    for _xvar in [T_VAR, "h_w"]:
        fig_31, axs_28 = make_scatter(scatter_data_5, xvar=_xvar, yvar="T", month=MONTH0)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### NDH_x
    """)
    return


@app.cell
def _(betaT_struct, betau_struct, src):
    ndh_struct = 12 * -src.utils.get_udTdx(T=betaT_struct, u=betau_struct)
    return (ndh_struct,)


@app.cell
def _(SEL_MONTH, T_VAR, add_vticks, ndh_struct, plot_zonal_structure, plt):
    for _v in [T_VAR, "h_w"]:
        print(_v)

        fig_32, axs_29 = plot_zonal_structure(coefs=ndh_struct.sel(j=_v), sel_fn=SEL_MONTH)
        add_vticks(axs_29, xticks=[190, 240, 280], xlines=[190, 240])
        # axs[0].set_ylabel(r"$T_{\text{sub}}~/~T-1$")

        plt.show()
    return


@app.cell
def _(SEL_MONTH, T_VAR, ndh_struct, plot_timeseries, plt, reduce_x_6):
    ## compute gamma
    ndh = reduce_x_6(ndh_struct)

    fig_33, axs_30 = plot_timeseries(coefs=ndh.sel(j=T_VAR), sel_fn=SEL_MONTH)
    plt.gca()
    return (ndh,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### sum of zonal
    """)
    return


@app.cell
def _(
    SEL_MONTH,
    T_VAR,
    add_vticks,
    dd_struct,
    ndh_struct,
    plot_zonal_structure,
    plt,
    zaf_struct,
):
    zonal_struct = zaf_struct + dd_struct + ndh_struct
    for _v in [T_VAR, "h_w"]:
        print(_v)

        fig_34, axs_31 = plot_zonal_structure(coefs=zonal_struct.sel(j=_v), sel_fn=SEL_MONTH)
        add_vticks(axs_31, xticks=[190, 240, 280], xlines=[190, 240])
        # axs[0].set_ylabel(r"$T_{\text{sub}}~/~T-1$")

        plt.show()
    return


@app.cell
def _(SEL_MONTH, T_VAR, dd, ndh, plot_timeseries, plt, zaf):
    fig_35, axs_32 = plot_timeseries(coefs=(zaf + dd + ndh).sel(j=T_VAR), sel_fn=SEL_MONTH)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### upwelling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load coefficients
    """)
    return


@app.cell
def _(coefs, xr):
    ## load coefs
    betaw_proj = xr.merge(c["w"].rename(n) for n, c in coefs.items())
    return (betaw_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot structure
    """)
    return


@app.cell
def _(anom, src, xr):
    def reduce_x_7(x):
        """Get Tsub, but don't average over depth yet"""

        # return REDUCE_X(x)
        return x.sel(longitude=slice(240,280)).mean("longitude")


    def reduce_z_4(x):
        """average over entrainment zone"""

        # return x.sel(z_t=ENT_DEPTH_RANGE).mean("z_t")
        return x.sel(z_t=slice(30,80)).mean("z_t")


    def recon_struct_6(coefs):
        """post-process regression coefficients"""

        ## reconstruct mean over equatorial region
        coefs_ = src.utils.reconstruct_wrapper(
            xr.merge([coefs.rename("w"), anom["w_comp"]]),
            fn=reduce_z_4,
        )
        ## drop extra variables
        mu = coefs_["w"].squeeze(drop=True)

        return mu


    def recon_idx_5(coefs):
        """reconstruct spatial structure of coefficients"""

        coefs_struct = src.utils.reconstruct_wrapper(
            xr.merge([coefs["w"], anom[f"w_comp"]]),
            fn=lambda y: reduce_x_7(reduce_z_4(y)),
        )

        return coefs_struct["w"]


    def get_dT(x):
        """Get diff b/n Tsub and Tsurf, but don't average over longitude yet"""

        # Tsurf = x.sel(z_t=slice(None, MLD)).mean("z_t")
        # Tsub = x.sel(z_t=ENT_DEPTH_RANGE).mean("z_t")
        Tsurf = x.sel(z_t=slice(None, 30)).mean("z_t")
        Tsub = x.sel(z_t=slice(30,80)).mean("z_t")

        return Tsub - Tsurf

    return get_dT, recon_idx_5, recon_struct_6, reduce_x_7


@app.cell
def _(MLD, betaw_proj, clim, forced, get_dT, recon_struct_6, src, xr):
    ## load coefs
    betaw_struct = xr.merge(
        [recon_struct_6(betaw_proj[v]).rename(v) for v in ["all", "pos", "neg"]]
    )

    ## get dTbar
    dT_bar_struct = src.utils.reconstruct_wrapper(
        xr.merge([clim["T"], forced["T_comp"]]),
        fn=get_dT,
    )["T"]

    ## compute ekman
    # first, scale upwelling by mixed layer depth (50 m) and convert from m/mo to m/yr
    w_H_2 = 12 * betaw_struct / MLD
    ekf_struct = dT_bar_struct * w_H_2
    return betaw_struct, dT_bar_struct, ekf_struct


@app.cell
def _(T_VAR, add_vticks, betaw_struct, ekf_struct, plot_zonal_structure, plt):
    SEL_MONTH_1 = lambda x : x.sel(month=slice(4,6)).mean("month")

    ## plot upwelling structure
    print("w")
    fig_36, axs_33 = plot_zonal_structure(coefs=betaw_struct.isel(j=0), sel_fn=SEL_MONTH_1)
    add_vticks(axs_33, xticks=[190, 240, 280], xlines=[190, 240])
    for _ax in axs_33:
        _ax.set_xlim([210, 285])
    add_vticks(axs_33, xticks=[240, 270], xlines=[240,270])
    # axs[0].set_ylabel(r"$T_{\text{sub}}~/~T-1$")

    plt.show()


    ## plot ekman structure
    for _v in [T_VAR, "h_w"]:
        print(f"ekf ({_v})")

        fig_36, axs_33 = plot_zonal_structure(coefs=ekf_struct.sel(j=_v), sel_fn=SEL_MONTH_1)
        add_vticks(axs_33, xticks=[240, 270], xlines=[240,270])
        for _ax in axs_33:
            _ax.set_xlim([210, 285])
        # axs[0].set_ylabel(r"$T_{\text{sub}}~/~T-1$")

        plt.show()
    return (SEL_MONTH_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot timeseries
    """)
    return


@app.cell
def _(
    SEL_MONTH_1,
    T_VAR,
    dT_bar_struct,
    ekf_struct,
    plot_timeseries,
    plt,
    reduce_x_7,
    xr,
):
    ## compute EKF
    ekf = reduce_x_7(ekf_struct)

    fig_37, axs_34 = plot_timeseries(coefs=ekf.sel(j=T_VAR), sel_fn=SEL_MONTH_1)
    plt.show()

    ## compute gamma
    dT_bar = reduce_x_7(dT_bar_struct)

    fig_37, axs_34 = plot_timeseries(
        coefs=xr.ones_like(ekf.sel(j=T_VAR)) * dT_bar, sel_fn=SEL_MONTH_1
    )
    plt.gca()
    return (ekf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter
    """)
    return


@app.cell
def _(MONTH0, T_VAR, anom, make_scatter, plt, recon_idx_5, xr):
    scatter_data_6 = xr.merge(
        [
            recon_idx_5(anom[["w"]]),
            anom[["T_3", "T_34", "h_w"]],
        ]
    )

    for _xvar in [T_VAR, "h_w"]:
        fig_38, axs_35 = make_scatter(scatter_data_6, xvar=_xvar, yvar=f"w", month=MONTH0)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### NDH_z
    """)
    return


@app.cell
def _(MLD, betaw_struct, gamma_struct_1):
    w_H_3 = betaw_struct / MLD
    ndh_z_struct = 12 * gamma_struct_1 * w_H_3
    return (ndh_z_struct,)


@app.cell
def _(SEL_MONTH_1, add_vticks, ndh_z_struct, plot_zonal_structure, plt):
    for _v in ndh_z_struct.j.values:
        print(_v)

        fig_39, axs_36 = plot_zonal_structure(coefs=ndh_z_struct.sel(j=_v), sel_fn=SEL_MONTH_1)
        add_vticks(axs_36, xticks=[190, 240, 280], xlines=[190, 240])
        # axs[0].set_ylabel(r"$T_{\text{sub}}~/~T-1$")

        plt.show()
    return


@app.cell
def _(SEL_MONTH_1, T_VAR, ndh_z_struct, plot_timeseries, plt, reduce_x_7):
    ## compute gamma
    ndh_z = reduce_x_7(ndh_z_struct)

    fig_40, axs_37 = plot_timeseries(coefs=ndh_z.sel(j=T_VAR), sel_fn=SEL_MONTH_1)
    plt.gca()
    return axs_37, fig_40


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### upwelling V2
    """)
    return


@app.cell
def _(anom, check_dims, src, xr):
    def reduce_y_3(x):
        """reduce over lat/lon/depth"""

        ## make sure dims are all there
        x_ = check_dims(x)

        ## specify indices
        idx = dict(longitude=slice(230, 280), z_t=slice(50,100))

        ## average over lats
        x_ = x_.sel(idx).mean("z_t")

        return x_


    def postprocess__1(coefs, T_var):
        """post-process regression coefficients"""

        ## get recon components
        comps = check_dims(anom[f"{T_var}_comp"])

        ## reconstruct mean over equatorial region
        coefs_ = src.utils.reconstruct_wrapper(
            xr.merge([coefs, comps]),
            fn=reduce_y_3,
        )

        ## drop extra variables
        coefs_ = coefs_[T_var].squeeze(drop=True)

        return coefs_

    return (postprocess__1,)


@app.cell
def _(DIMS, anom, postprocess__1, regress_over_time, window, xr):
    ## specify wind variable (one of "T_34", "taux_idx")
    x_var_1 = "taux_idx"
    # x_var = T_VAR

    ## specify h-variable (one of "T", "ssh")
    y_var_1 = "w"

    ## get data for regression (OHC)
    beta_data_1 = window(xr.merge([anom[y_var_1], anom[x_var_1]]))
    _regress_kwargs = dict(x_vars=[x_var_1], y_vars=[y_var_1], dims=DIMS)
    _regress_helper = lambda x: postprocess__1(
        regress_over_time(x, **_regress_kwargs), T_var=y_var_1
    )

    ## do the regression
    print("regress all")
    _beta_all = _regress_helper(beta_data_1)
    print("regress pos")
    _beta_pos = _regress_helper(beta_data_1.where(beta_data_1[x_var_1] > 0))
    print("regress neg")
    _beta_neg = _regress_helper(beta_data_1.where(beta_data_1[x_var_1] < 0))

    ## merge into single dataarray
    beta_struct_1 = xr.merge(
        [_beta_all.rename("all"), _beta_pos.rename("pos"), _beta_neg.rename("neg")]
    )
    return (beta_data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot all vars
    """)
    return


@app.cell
def _(axs_37, fig_40):
    def plot_timeseries_v2(ax, coefs, sel_fn=lambda x: x.mean("month"), norm=True, norm_center = False):
        """plot timeseries comparison"""

        ## normalize coefficients
        coefs_ = sel_fn(coefs)
        coefs_norm = coefs_ / coefs_.isel(year=0).mean("member")["pos"].values.item()

        ## select coefficients to plot
        if norm:
            plot_coefs = coefs_norm
        else:
            plot_coefs = coefs_

        ## loop thru pos and negative
        for i, (name, color) in enumerate(zip(["pos", "neg", "all"], ["r", "b", "k"])):

            ## plot median and bounds
            if name == "all":
                q_vals = [0.5]
            else:
                q_vals = [0.5, 0.1, 0.9]

            for q, lw in zip(q_vals, [2, 0.6, 0.6]):

                ## finally, plot data
                ax.plot(
                    plot_coefs.year,
                    plot_coefs[name].quantile(q=q, dim="member"),
                    c=color,
                    lw=lw,
                )

        ## formatting
        ax.set_xticks([1870, 2010, 2080])
        ax_kwargs = dict(ls="--", c="gray", lw=0.8)
        ax.axvline(2010, **ax_kwargs)
        ax.axhline(0, **ax_kwargs)
        # ax.set_yticks([])

        return fig_40, axs_37

    def plot_timeseries_v2_all(ax, coefs, sel_fn=lambda x: x.mean("month"), norm=True, norm_center=False):
        """plot timeseries comparison"""

        ## normalize coefficients
        coefs_ = sel_fn(coefs)
        coefs_norm = coefs_ / coefs_.isel(year=0).mean("member")["all"].values.item()
        coefs_norm_center = coefs_norm - 1

        ## select coefficients to plot
        if norm_center:
            plot_coefs = coefs_norm_center
        elif norm:
            plot_coefs = coefs_norm
        else:
            plot_coefs = coefs_

        for q, lw in zip([0.5, 0.1, 0.9], [2, 0.6, 0.6]):

            ## finally, plot data
            ax.plot(
                plot_coefs.year,
                plot_coefs["all"].quantile(q=q, dim="member"),
                c="k",
                lw=lw,
            )

        ## formatting
        ax.set_xticks([1870, 2010, 2080])
        ax_kwargs = dict(ls="--", c="gray", lw=0.8)
        ax.axvline(2010, **ax_kwargs)
        ax.axhline(0, **ax_kwargs)
        # ax.set_yticks([])

        return fig_40, axs_37

    return plot_timeseries_v2, plot_timeseries_v2_all


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, plot thermocline feedback estimates
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### THF
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Pt 1: components
    """)
    return


@app.cell
def _(MLD, ah, beta, gamma, mu, plot_timeseries_v2_all, plt, save, src, w_bar):
    ## should we save plots?
    do_save_THF = False

    for plot_months, plot_name in zip([slice(1,12), slice(10,12)], ["all", "OND"]):

        ## specify which months to plot
        _kwargs = dict(sel_fn = lambda x : x.sel(month=plot_months).mean("month"), norm_center=True)
    
        ## get normalized w
        w_bar_norm = w_bar / w_bar.isel(year=0)
    
        fig_41, axs_38 = plt.subplots(1, 3, figsize=(7.5, 2.5), layout="constrained")
    
        ## mu
        plot_timeseries_v2_all(axs_38[0], coefs=mu, **_kwargs)
    
        ## beta
        plot_timeseries_v2_all(axs_38[1], coefs=beta, **_kwargs)
    
        ## THERMAL expansion
        plot_timeseries_v2_all(axs_38[2], coefs=ah, **_kwargs)
    
        ## label
        _labels = [r"a) $\mu$", r"b) $\beta$", r"c) $a_h$"]
    
        for _ax, _label in zip(axs_38, _labels):
            _ax.set_title(_label, loc="left")
    
        ##formatting
        axs_38[0].set_yticks([-.2, 0, .2])
        src.utils.set_lims(axs_38)
        for _ax in axs_38[1:]:
            _ax.set_yticks([])
        axs_38[0].set_ylabel("Frac. change")
    
        save(fig_41, fname=f"THF_all_pt1_{plot_name}", do_save=do_save_THF)
    
        plt.show()
    
        ##### Pt 2: Direct vs. indirect
    
        ## specify which months to plot
        _kwargs = dict(sel_fn = lambda x : x.sel(month=plot_months).mean("month"))
    
        fig_41, axs_38 = plt.subplots(1, 3, figsize=(7.5, 2.5), layout="constrained")
    
        ## product
        plot_timeseries_v2_all(axs_38[0], coefs=mu * beta * ah-1, norm=False, **_kwargs)
    
        ## product
        plot_timeseries_v2_all(axs_38[1], coefs=12*w_bar/MLD * (mu * beta * ah-1), norm=False, **_kwargs)
    
        ## direct estimate
        plot_timeseries_v2_all(axs_38[2], coefs=12*w_bar/MLD * (gamma), norm=False, **_kwargs)
    
    
        ## label
        _labels = [
            r"d) $\hat{\eta} = \mu\cdot \beta \cdot a_h-1$",
            r"e) $\frac{\overline{w}}{H}\cdot\hat{\eta}$",
            r"f) $\frac{\overline{w}}{H}\cdot\eta$ (direct estimate)",
        ]
        for _ax, _label in zip(axs_38, _labels):
            _ax.set_title(_label, loc="left")
            # ax.axvline(2050)
    
        ##formatting
        src.utils.set_lims(axs_38[1:])
        axs_38[-1].set_yticks([])
        axs_38[0].set_ylabel(r"$K$")
        axs_38[1].set_ylabel(r"$K/yr$")
    
        save(fig_41, fname=f"THF_all_pt2_{plot_name}", do_save=do_save_THF)
    
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compare predictions of $T_{sub}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Is $\beta$ the problem? Could try again using $h_e$ instead of $\Delta_x h$
    """)
    return


@app.cell
def _(ENT_AVG, REDUCE_X, T_VAR, ah, anom, beta, gamma, mu, src, window):
    ## should we use dh in pred?
    USE_DH_PRED = False

    eta_hat = (mu * beta * ah).mean("member")
    eta = (gamma+1).mean("member")

    ## get data for regression
    T = window(anom[T_VAR])
    hw = window(anom["h_w"])

    ## get predictions
    Tsub_hat0 = eta_hat * T.groupby("time.month")
    Tsub_hat1 = eta * T.groupby("time.month")

    ## account for hw if desired
    if USE_DH_PRED:
    
        ## get dh prediction
        ah_ens = ah.mean("member")

        ## to-do: fix prediction for Tsub_hat1 to use estimated gamma coef
        Tsub_hat0 = Tsub_hat0 + (ah_ens * hw.groupby("time.month"))
        Tsub_hat1 = Tsub_hat1 + (ah_ens * hw.groupby("time.month"))

    ## get ground truth
    Tsub = window(
        src.utils.reconstruct_fn(
            scores=anom["T"],
            components=anom["T_comp"],
            fn = lambda x : REDUCE_X(ENT_AVG(x)),
        )
    )
    return T, Tsub, Tsub_hat0, Tsub_hat1, eta, eta_hat


@app.cell
def _(MONTH0, T, Tsub, Tsub_hat0, Tsub_hat1, make_scatter, np, plt, xr):
    ## specify which part of regression to plot
    reg_type = "pos"

    scatter_data_7 = xr.merge(
        [
            T.rename("T"),
            Tsub.rename("Tsub"), 
            Tsub_hat0[reg_type].rename("indirect"), 
            Tsub_hat1[reg_type].rename("direct"),
        ]
    )
    for y_1 in ["direct", "indirect"]:
        print(y_1)
        fig_42, axs_39 = make_scatter(scatter_data_7, xvar="Tsub", yvar=y_1, month=MONTH0)
        for _ax in axs_39:
            _zz = np.linspace(-4,4)
            _ax.plot(_zz,_zz, lw=1, c="k")
        plt.show()

    print("direct vs. indirect")
    fig_42, axs_39 = make_scatter(scatter_data_7, xvar="direct", yvar="indirect", month=MONTH0)
    for _ax in axs_39:
        _zz = np.linspace(-4,4)
        _ax.plot(_zz,_zz, lw=1, c="k")
        _ax.set_aspect("equal")
    plt.gca()
    return (scatter_data_7,)


@app.cell
def _(
    MONTH0,
    SEL_MONTH_1,
    T_VAR,
    eta,
    eta_hat,
    make_scatter,
    np,
    plt,
    save,
    scatter_data_7,
):
    ## pos/neg values for regression
    _zz_pos = np.linspace(0,4)
    _zz_neg = np.linspace(-4,0)

    ## helperfunc
    _get = lambda x : x.values.item()

    fig_43, axs_40 = make_scatter(
        scatter_data_7, xvar="T", yvar="Tsub", month=MONTH0, c="gray", alpha=.5, verbose=False, by_season=False
    )

    ## labels
    _labels = ["a)", "b)", "c)", "d)"]
    for _ax, label_ in zip(axs_40, _labels):

        ## get year
        y_2 = int(_ax.get_title())

        ## helper func to get data
        get_helper = lambda x : _get(SEL_MONTH_1(x).sel(year=y_2))
    
        for eta_, c, l in zip([eta_hat, eta], ["k", "magenta"], [r"indirect",r"direct"]):
            for _zz, _n, ls in zip([_zz_neg, _zz_pos], ["neg", "pos"], ["--","-"]):
                _label = None if (_n=="pos") else l
                _ax.plot(_zz, _zz*get_helper(eta_[_n]), c=c, ls=ls, lw=2.5, label=_label)

        ## do some formatting
        _ax.set_xticks([-4,0,4])
        _ax.set_xlabel(r"$T_{\text{surf}}$")

        ## update title
        _ax.set_title("")
        _ax.set_title(f"{label_} {y_2}", loc="left")

    ## do some formatting
    axs_40[0].set_ylabel(r"$T_{\text{sub}}$")
    axs_40[0].set_yticks([-4,0,4])
    axs_40[0].legend(prop=dict(size=8))

    ## save to file
    save(fig_43, fname=f"THF_bias_overtime_{T_VAR}", do_save=False)

    plt.gca()
    return (ls,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot components of THF estimate for 1870
    """)
    return


@app.cell
def _(
    ah,
    beta,
    beta_data_1,
    gamma,
    ls,
    make_scatter_ax,
    mu,
    np,
    plt,
    save,
    scatter_data_7,
    src,
    xr,
):
    scatter_data2 = xr.merge([
        scatter_data_7[["T","Tsub"]], beta_data_1[["taux_idx", "dh", "he", "hw"]],
    ])
    scatter_data2["dT"] = scatter_data2["Tsub"]-scatter_data2["T"]

    ## get hw * ah
    hw_dot_ah = scatter_data2["hw"].groupby("time.month") * ah["all"]
    scatter_data2["dT_prime"] = scatter_data2["dT"] - hw_dot_ah

    ## shared args
    _kwargs = dict(by_season=False, month=12, label="", verbose=False, alpha=.5, c="gray")

    ## specify things to plot
    plot_stuff = [
        ("T", "taux_idx", mu, 4, r"$\tau_x$", "$T$"),
        ("taux_idx", "dh", beta, .025, r"$h_e-h_w$", r"$\tau_x$"),
        # ("taux_idx", "he", beta_he, .025),
        ("he", "Tsub", ah, 50, r"$T_{sub}$", "$h_e$"),
        ("T", "dT", mu*beta*ah-1, 2.5, r"$T_{sub}-T$", "$T$"),
        # ("T", "dT_prime", mu*beta*ah-1, 2.5, r"$T_{sub}-T$", "$T$"),
        # ("T", "Tsub", mu*beta*ah, 4, r"$T_{sub}$", "$T$"),
    ]


    fig_44, axs_41 = plt.subplots(4, 4, figsize=(12,10), layout="constrained")

    for j, y_3 in enumerate([1870, 2010, 2050, 2080]):

        ## plot data
        for _ax, (_xvar, _yvar, _param, amp, ylabel, xlabel) in zip(axs_41[j], plot_stuff):
    
            ## scatter plot of data
            make_scatter_ax(_ax, xvar=_xvar, yvar=_yvar, anom_=scatter_data2.sel(year=y_3), **_kwargs)
    
            ## plot best fit
            _get = lambda x : x.sel(year=y_3, month=_kwargs["month"]).mean(["member"]).values.item()
    
            ## test points
            _zz_pos = np.linspace(0,amp)
            _zz_neg = np.linspace(-amp,0)
            zz_all = np.linspace(-amp, amp)
    
            ## plot pos/neg
            if ("dT" in _yvar):
                _ax.plot(zz_all, zz_all*_get(gamma["all"]), c="magenta", ls=ls, lw=2.5)

            else:
                pass
                # for zz, n, ls in zip([zz_neg, zz_pos], ["neg", "pos"], ["--","-"]):
                #     ax.plot(zz, zz*get(param[n]), c="k", ls=ls, lw=2.5)
            
    
            ## plot all
            _ax.plot(zz_all, zz_all*_get(_param["all"]), c="k", ls="--", lw=2.5, )
    
            _ax.set_xlabel(xlabel)
            _ax.set_ylabel(ylabel)

    for k in range(4):
        src.utils.set_lims(axs_41[:,k])
    for j in range(axs_41.shape[0]-1):
        for _ax in axs_41[j]:
            _ax.set_xticks([])

    save(fig_44, "thf_scatter_comp", do_save=False)
    
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Look at error correlation
    """)
    return


@app.cell
def _(T, anom, mu, src, window):
    def get_taux(x):
        idx = dict(latitude=slice(-5,5),longitude=slice(150,270))
        return x.sel(idx).mean(["latitude","longitude"])

    ## get predictions
    taux_hat = mu["pos"] * T.groupby("time.month")
    taux = window(
            src.utils.reconstruct_fn(
            scores=anom["taux"], 
            components=anom["taux_comp"], 
            fn = get_taux,
        )
    )

    error = taux_hat-taux
    return error, taux


@app.cell
def _(MONTH0, SEL_MONTH_1, T, error, make_scatter, mu, np, plt, taux, xr):
    scatter_data_8 = xr.merge(
        [
            error.rename("error"),
            taux.rename("taux"),
            T.rename("T"),
        ]
    )

    ## plot error
    fig_45, axs_42 = make_scatter(scatter_data_8.where(T>0), xvar="T", yvar="error", month=MONTH0)

    plt.show()

    ## plot prediction
    fig_45, axs_42 = make_scatter(scatter_data_8.where(T>0), xvar="T", yvar="taux", month=MONTH0)

    for _ax in axs_42:
        _zz = np.linspace(0,4)

        y_4 = int(_ax.get_title())
        mu_ = SEL_MONTH_1(mu["pos"]).sel(year=y_4).mean("member").values.item()
        _ax.plot(_zz, mu_*_zz, c="k")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### growth rate stuff
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load epsilon data
    """)
    return


@app.cell
def _(T_VAR, coefs, xr):
    get_coef = lambda n, v: 12 * coefs[n][f"ddt_{v}"].sel(j=v).isel(mode=0).rename(n)

    R_check = xr.merge(get_coef(n, T_VAR).rename(n) for n in list(coefs))
    eps = xr.merge(-get_coef(n, "h_w").rename(n) for n in list(coefs))


    get_F1 = lambda n: 12 * coefs[n][f"ddt_{T_VAR}"].sel(j="h_w").isel(mode=0).rename(n)
    get_F2 = lambda n: -12 * coefs[n][f"ddt_h_w"].sel(j=T_VAR).isel(mode=0).rename(n)
    F1 = xr.merge(get_F1(n).rename(n) for n in list(coefs))
    F2 = xr.merge(get_F2(n).rename(n) for n in list(coefs))
    return F1, F2, R_check, eps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check R matches
    """)
    return


@app.cell
def _(R, R_check, SEL_MONTH_1, coefs, plt, src):
    sel_3 = lambda x: SEL_MONTH_1(x).mean("member")
    # sel = lambda x: SEL_MONTH(x).mean("member")

    ## check R matches
    fig_46, axs_43 = plt.subplots(1, 3, figsize=(7.5, 2.5))
    for _ax, _n in zip(axs_43, list(coefs)):
        _ax.set_title(_n)
        _ax.plot(R.year, sel_3(R[_n]), c="k")
        _ax.plot(R.year, sel_3(R_check[_n]), c="orange", ls="--")
        # ax.plot(R.year, sel(eps[n]), c="green")
        # ax.axvline(2030)


    for _ax in axs_43:
        _ax.axhline(0, ls="--", c="k", lw=0.8)
    for _ax in axs_43[1:]:
        _ax.set_yticks([])
    src.utils.set_lims(axs_43)
    plt.gca()
    return


@app.cell
def _(plot_timeseries_v2, src):
    def plot_R_comparison(axs, R, eps, alpha, sel_fn):
        """plot comparison of R stuff on axs"""

        ## get scaled alpha

        ## get shared args
        kwargs = dict(norm=False, sel_fn=sel_fn)
    
        ## growth rate
        plot_timeseries_v2(axs[0], coefs=R - eps, **kwargs)
    
        ## Bjerknes
        plot_timeseries_v2(axs[1], coefs=R, **kwargs)
    
        ## damping
        plot_timeseries_v2(axs[2], coefs=alpha, **kwargs)
    
        ## difference
        plot_timeseries_v2(axs[3], coefs=R + alpha, **kwargs)
    
        ##formatting
        src.utils.set_lims(axs[1:])
        for ax in axs[2:]:
            ax.set_yticks([])
        axs[0].set_ylabel(r"yr$^{-1}$")

        return axs

    return (plot_R_comparison,)


@app.cell
def _(MLD, R, alpha, copy, eps, plot_R_comparison, plt, save):
    ## scale alpha by MLD
    MLD_ALPHA = copy.deepcopy(MLD)
    alpha_scaled = MLD / MLD_ALPHA * alpha

    SEL_ALL = lambda x: x.sel(month=slice(1, 12)).mean("month")
    SEL_OND = lambda x: x.sel(month=slice(10, 12)).mean("month")

    ## shared args

    fig_47, axs_44 = plt.subplots(2, 4, figsize=(10, 5), layout="constrained")

    axs_44[0] = plot_R_comparison(axs_44[0], R=R, eps=eps, alpha=alpha_scaled, sel_fn=SEL_ALL)
    axs_44[1] = plot_R_comparison(axs_44[1], R=R, eps=eps, alpha=alpha_scaled, sel_fn=SEL_OND)

    ## formatting
    _labels = [r"$R-\varepsilon$", r"$R$", r"$\alpha$", r"$R_o=R+\alpha$"]
    for _ax, _label in zip(axs_44[0], _labels):
        _ax.set_title(_label)
    
    for _ax in axs_44[0]:
        _ax.set_xticks([])
    
    axs_44[0,0].set_yticks([-1,0,1])
    axs_44[0,1].set_yticks([0, 2, 4])
    axs_44[1,0].set_yticks([0, -1, -2])
    axs_44[1,1].set_yticks([0, 1, 2])

    axs_44[0,-1].text(s="All months", x=1.1, y=0.5, transform=axs_44[0,-1].transAxes)
    axs_44[1,-1].text(s="OND", x=1.1, y=0.5, transform=axs_44[1,-1].transAxes)

    ## save to file
    save(fig_47, fname=f"GR_comp", do_save=True)

    plt.gca()
    return


@app.cell
def _(F1, F2, plot_timeseries_v2, plt):
    SEL = lambda x: x.mean("month")
    # SEL = lambda x : x.sel(month=[11,12,1,2]).mean("month")

    ## shared args
    _kwargs = dict(sel_fn=SEL, norm=False)

    fig_48, axs_45 = plt.subplots(1, 3, figsize=(7.5, 2.5), layout="constrained")

    ## F1
    plot_timeseries_v2(axs_45[0], coefs=F1, **_kwargs)

    ## F2
    plot_timeseries_v2(axs_45[1], coefs=F2, **_kwargs)

    ## Product
    plot_timeseries_v2(axs_45[2], coefs=F1 * F2, **_kwargs)


    ## label
    _labels = ["F1", "F2", "F1*F2"]
    for _ax, _label in zip(axs_45, _labels):
        _ax.set_title(_label)
    plt.gca()
    return


@app.cell
def _(
    R,
    R_check,
    T_VAR,
    ah,
    alpha,
    beta,
    cmocean,
    dd,
    ekf,
    eps,
    gamma,
    mu,
    ndh,
    np,
    plt,
    src,
    thf,
    zaf,
):
    j_var = T_VAR

    YEARS = [1870, 2010, 2050, 2080]
    colors = cmocean.cm.balance(np.linspace(0.2, 0.8, len(YEARS)))

    for _param, name in zip(
        [
            R - eps,
            R,
            mu,
            mu * beta * ah - 1,
            gamma,
            alpha,
            (zaf + dd + ndh).sel(j=j_var),
            ekf.sel(j=j_var),
            thf.sel(j=j_var),
            # dT_bar * xr.ones_like(thf.sel(j=j_var)),
            # w_bar * xr.ones_like(thf.sel(j=j_var)),
        ],
        [
            "R-eps",
            "R",
            "mu",
            "mu*beta*ah-1",
            r"$\gamma$",
            r"$\alpha$",
            "zonal",
            "EKF",
            "THF",
            # r"$\Delta_z\overline{T}$",
            # r"$\overline{w}$",
        ],
    ):

        print(name)

        fig_49, axs_46 = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

        for _ax, _n in zip(axs_46, ["all", "pos", "neg"]):
            _ax.set_title(_n)
            _ax.axhline(0, ls="--", c="k", lw=0.8)
            for i, y_5 in enumerate(YEARS):
                _ax.plot(
                    R_check.month, _param[_n].mean("member").sel(year=y_5), label=y_5, c=colors[i]
                )

        src.utils.set_lims(axs_46)
        for _ax in axs_46[1:]:
            _ax.set_yticks([])
        axs_46[0].legend(prop=dict(size=10))
        axs_46[0].set_ylabel(r"yr$^{-1}$")
        for _ax in axs_46:
            _ax.set_xticks([1,5,12], labels=["Jan","May","Dec"])

        plt.show()
    return


@app.cell
def _(R, T_VAR, alpha, plt, src, thf, zaf):
    ## what value to use for R
    R_VAL = R
    # R_VAL = R-eps

    ## plot amplitude
    _AMP = 1.5

    ## helper func. to get diff
    _get_delta = lambda x: x - x.isel(year=0)


    for _F, _label in zip(
        [
            R_VAL.expand_dims(j=[T_VAR]),
            -alpha.expand_dims(j=[T_VAR]),
            (R_VAL + alpha).expand_dims(j=[T_VAR]),
            thf,
        ],
        ["R", r"$-alpha$", r"$Ro$", "THF"],
    ):
        print(f"\n\n{_label}")
        fig_50, axs_47 = plt.subplots(1, 3, figsize=(6, 4))

        for _ax, _n in zip(axs_47, ["all", "pos", "neg"]):
            _ax.contourf(
                zaf.month,
                zaf.year,
                _get_delta(_F[_n].isel(j=0).mean("member")).transpose("year", ...),
                cmap="cmo.balance",
                levels=src.utils.make_cb_range(_AMP, _AMP / 10),
                extend="both",
            )

            _ax.axhline(2010, c="k", ls="--", lw=0.8)
            _ax.set_yticks([])
        axs_47[0].set_yticks([1870, 2010, 2085])
        plt.show()
    return


@app.cell
def _(R, T_VAR, ah, alpha, beta, ekf, eps, gamma, mu, plt, src, thf, zaf):
    ## specify feedback
    # F = (zaf+dd+ndh)
    # F = thf
    # F = ekf
    # F = R.expand_dims(j=[T_VAR])

    _get_delta = lambda x: x - x.isel(year=0)
    _AMP = 2
    # get_delta = lambda x : x
    # AMP = 2

    for _F, _label in zip(
        [
            (R-eps).expand_dims(j=[T_VAR]),
            thf,
            2 * (mu * ah * beta).expand_dims(j=[T_VAR]),
            4 * gamma.expand_dims(j=[T_VAR]),
            ekf,
            # zaf+dd+ndh,
            zaf,
            -alpha.expand_dims(j=[T_VAR]),
        ],
        ["R", "thf", "mu*beta*ah", "gamma", "ekf", "zonal", "-alpha"],
    ):
        print(f"\n\n{_label}")
        fig_51, axs_48 = plt.subplots(1, 3, figsize=(6, 4))

        for _ax, _n in zip(axs_48, ["all", "pos", "neg"]):
            _ax.contourf(
                zaf.month,
                zaf.year,
                _get_delta(_F[_n].isel(j=0).mean("member")).transpose("year", ...),
                cmap="cmo.balance",
                levels=src.utils.make_cb_range(_AMP, _AMP / 10),
                extend="both",
            )

            _ax.axhline(2010, c="k", ls="--", lw=0.8)
            _ax.set_yticks([])
        axs_48[0].set_yticks([1870, 2010, 2085])
        plt.show()
    return


if __name__ == "__main__":
    app.run()
