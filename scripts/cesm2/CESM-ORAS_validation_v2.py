import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
    import xesmf as xe
    import src.XRO_utils

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return DATA_FP, copy, np, os, pathlib, plt, sns, src, tqdm, xe, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## funcs
    """)
    return


@app.cell
def _(os, pathlib):
    def save(fig, fname, dpi=300):
        """save figure to file"""

        ## get save directory
        save_dir = pathlib.Path(os.environ["SAVE_FP"], "ch3-outline")

        ## get fname
        fname = save_dir / f"{fname}.pdf"

        # fig.savefig(fname, dpi=dpi)
        return

    return (save,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### CESM2
    """)
    return


@app.cell
def _(src):
    ## load spatial data
    forced, anom = src.utils.load_consolidated()

    ## trim to desired vars
    anom = anom[["sst", "pr", "ssh", "sst_comp", "pr_comp", "ssh_comp"]]
    return anom, forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reference data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ERRST
    """)
    return


@app.cell
def _(pathlib, xr):
    ## reference data
    _REF_FP = pathlib.Path("/glade/campaign/cgd/cas/asphilli/CVDP-OBS/")
    pr_ref = xr.open_dataset(_REF_FP / "gpcp.mon.mean.197901-202403.nc")["precip"]
    sst_ref = xr.open_dataset(_REF_FP / "ersstv5.185401-202412.nc")["ssta"]
    total_ref = xr.open_dataset(_REF_FP / "ersstv5.185401-202412.nc")["sst"]
    # sst_ref = xr.open_dataset(REF_FP / "hadisst.187001-202412.nc")

    ## assign time coord
    sst_ref["time"] = xr.date_range(start="1854-01", end="2024-12", freq="MS")
    total_ref["time"] = xr.date_range(start="1854-01", end="2024-12", freq="MS")
    # sst_ref["time"] = xr.date_range(start="1870-01", end="2024-12", freq="MS")
    return pr_ref, sst_ref, total_ref


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ORAS
    """)
    return


@app.cell
def _():
    # ## load data
    # data_oras = src.utils.load_oras_spatial_extended(
    #     DATA_FP / "oras", varnames=["tos"]
    # )["sst"]

    # ## estimate forced signal by removing 2nd-order polynomial from each calendar month
    # detrend_fn = lambda x: src.utils.detrend_dim(x, dim="time", deg=2)
    # sst_ref = data_oras.groupby("time.month").map(detrend_fn)
    # total_ref = data_oras - sst_ref
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trim to overlapping period
    """)
    return


@app.cell
def _(anom, forced, pr_ref, sst_ref, total_ref):
    ## specify index
    # time_idx = dict(time=slice("1958-01", "2024-06"))
    _time_idx = dict(time=slice("1958-01", "2023-12"))

    ## subset in time
    sst_ref_1 = sst_ref.sel(_time_idx)
    total_ref_1 = total_ref.sel(_time_idx)
    pr_ref_1 = pr_ref.sel(_time_idx)
    anom_1 = anom.sel(_time_idx)
    forced_1 = forced.sel(_time_idx)
    return anom_1, forced_1, pr_ref_1, sst_ref_1, total_ref_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### resample to seasonal
    """)
    return


@app.cell
def _(anom_1, copy, pr_ref_1, sst_ref_1, tqdm):
    ## func to resample with
    resample = lambda x: (x - x.mean("time")).resample({"time": "QS-NOV"}).mean("time")

    ## specify month for averaging
    MONTH = 11

    ## resample obs
    sst_ref_2 = resample(sst_ref_1)
    pr_ref_2 = resample(pr_ref_1)

    ## resample model
    anom_ = copy.deepcopy(anom_1[["sst_comp"]])
    for _v in tqdm.tqdm(["sst"]):
        anom_[_v] = resample(anom_1[_v])
    return MONTH, anom_, pr_ref_2, sst_ref_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regrid to match
    """)
    return


@app.cell
def _(anom_1, pr_ref_2, sst_ref_2, total_ref_1, xe):
    ## rename spatial coords
    if "lat" in sst_ref_2.coords:
        coord_dict = dict(lat="latitude", lon="longitude")
        pr_ref_3 = pr_ref_2.rename(coord_dict)
        sst_ref_3 = sst_ref_2.rename(coord_dict)

    ## regrid
    grid = anom_1["sst_comp"].isel(mode=0)
    sst_regridder = xe.Regridder(sst_ref_3, grid, "bilinear")
    sst_ref_3 = sst_regridder(sst_ref_3)
    total_regridder = xe.Regridder(total_ref_1, grid, "bilinear")
    total_ref_2 = total_regridder(total_ref_1)
    pr_regridder = xe.Regridder(pr_ref_3, grid, "bilinear")
    pr_ref_3 = pr_regridder(pr_ref_3)
    return coord_dict, pr_ref_3, sst_ref_3, total_ref_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### relative SST clim
    """)
    return


@app.cell
def _(DATA_FP, forced_1, pathlib, src, total_ref_2, xr):
    ## tropical sst for model
    model_trop_sst = xr.open_dataset(pathlib.Path(DATA_FP, "cesm/trop_sst.nc"))
    model_trop_sst = model_trop_sst["trop_sst_15"].sel(time=forced_1.time)
    model_trop_sst_clim = model_trop_sst.groupby("time.month").mean(["time", "member"])
    model_trop_sst_clim = model_trop_sst_clim.sel(month=[11, 12, 1]).mean("month")

    ## clim for model
    clim_model = src.utils.reconstruct_clim(forced_1[["sst", "sst_comp"]])
    clim_model = clim_model["sst"].sel(month=[11, 12, 1]).mean("month")

    ## relative sst for model
    rsst_model = clim_model - model_trop_sst_clim

    ## clim for obs
    clim_ref = total_ref_2.groupby("time.month").mean().sel(month=[11, 12, 1]).mean("month")

    ## get tropical avg
    ref_trop_sst = clim_ref.sel(latitude=slice(-15, 15)).mean(["latitude", "longitude"])
    rsst_ref = clim_ref - ref_trop_sst
    return rsst_model, rsst_ref


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Make composites
    """)
    return


@app.cell
def _(MONTH, anom_, pr_ref_3, src, sst_ref_3):
    ## specify index fn
    idx_fn = src.utils.get_nino3

    ## Get OND
    anom_ond = anom_.sel(time=anom_.time.dt.month == MONTH)
    sst_ref_ond = sst_ref_3.sel(time=sst_ref_3.time.dt.month == MONTH)
    pr_ref_ond = pr_ref_3.sel(time=pr_ref_3.time.dt.month == MONTH)

    ## Get index
    idx_ref_ond = idx_fn(sst_ref_ond)
    idx_anom_ond = src.utils.reconstruct_wrapper(anom_ond, idx_fn)["sst"]
    return anom_ond, idx_anom_ond, idx_ref_ond, sst_ref_ond


@app.cell
def _(anom_ond, idx_anom_ond, idx_ref_ond, pr_ref_3, sst_ref_ond, xr):
    def get_warm_composite(data, idx, q=0.75):
        """get warm composite"""

        ## get masked data
        data_ = data.where(idx > idx.quantile(q=q, dim="time"))

        return data_.mean("time").drop_vars("quantile")


    def get_cold_composite(data, idx, q=0.75):
        """get warm composite"""

        ## get masked data
        data_ = data.where(idx < idx.quantile(q=(1 - q), dim="time"))

        return data_.mean("time").drop_vars("quantile")


    def get_composites(data, idx, q=0.75):
        """get composites"""

        kwargs = dict(data=data, idx=idx, q=q)
        comps = xr.merge(
            [
                get_warm_composite(**kwargs).rename("warm"),
                get_cold_composite(**kwargs).rename("cold"),
            ],
        )

        return comps


    ## get composites
    q = 0.9
    comps_ref_sst = get_composites(sst_ref_ond, idx=idx_ref_ond, q=q)
    comps_anom_sst = get_composites(anom_ond["sst"], idx=idx_anom_ond, q=q).mean("member")

    comps_ref_pr = get_composites(pr_ref_3, idx=idx_ref_ond, q=q)
    # comps_anom_pr = 8.6e4 * get_composites(anom_["pr"], idx=idx_anom_, q=q).mean("member")
    return comps_anom_sst, comps_ref_sst


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reconstruct CESM2
    """)
    return


@app.cell
def _(anom_1, comps_anom_sst, comps_ref_sst, np, src):
    for n in ["warm", "cold"]:
        comps_anom_sst[n] = src.utils.reconstruct_fn(
            scores=comps_anom_sst[n],
            components=anom_1["sst_comp"],
            fn=lambda x: x,
        )

        # comps_anom_pr[n] = src.utils.reconstruct_fn(
        #     scores=comps_anom_pr[n],
        #     components=anom["pr_comp"],
        #     fn=lambda x: x,
        # )

    # ## drop mode coordinate
    # comps_anom_sst = comps_anom_sst.drop_vars("mode")
    # comps_anom_pr = comps_anom_pr.drop_vars("mode")

    ## normalize by El Niño
    get_scale = lambda x: np.abs(src.utils.get_nino3(x["warm"]))
    scale_ref = get_scale(comps_ref_sst)
    scale_anom = get_scale(comps_anom_sst)
    scale_ratio = scale_ref / scale_anom

    # comps_ref_norm = comps_ref_sst / scale_ref
    # comps_anom_norm = comps_anom_sst / scale_anom
    comps_anom_norm = scale_ratio * comps_anom_sst
    return comps_anom_norm, scale_anom, scale_ratio, scale_ref


@app.cell
def _(scale_ratio):
    print(scale_ratio.values.item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### funcs
    """)
    return


@app.cell
def _(np, src):
    import cartopy.crs as ccrs


    def make_cb_range(dx, nlevs):
        amp = dx * nlevs - dx / 2

        lev1 = np.arange(0, amp + dx / 2, dx) + dx / 2
        lev0 = -lev1[::-1]
        return np.concatenate([lev0, lev1])


    def plot_sst2(ax, data, dx=0.2, nlev=5):
        """plot data on ax"""

        cp = ax.contourf(
            data.longitude,
            data.latitude,
            data,
            cmap="cmo.balance",
            levels=make_cb_range(dx, nlev),
            transform=ccrs.PlateCarree(),
            extend="both",
        )

        return cp


    def plot_sst(ax, data, amp=2):
        """plot data on ax"""

        cp = ax.contourf(
            data.longitude,
            data.latitude,
            data,
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(amp, amp / 5),
            transform=ccrs.PlateCarree(),
            extend="both",
        )

        return cp


    def plot_pr(ax, data, amp=20):
        """plot data on ax"""

        ax.contour(
            data.longitude,
            data.latitude,
            data,
            colors="k",
            linewidths=0.8,
            alpha=0.75,
            levels=src.utils.make_cb_range(amp, amp / 5),
            transform=ccrs.PlateCarree(),
            extend="both",
        )

        return

    return ccrs, plot_sst, plot_sst2


@app.cell
def _(ccrs, src):
    ## specify lons/lats for E/W boxes
    LONS_E = [240, 280]
    LONS_W = [150, 190]


    ## funcs to plot boxes
    def plot_wbox(ax, **kwargs):
        src.utils.plot_box(ax, lons=LONS_W, lats=[-5, 5], **kwargs)


    def plot_ebox(ax, **kwargs):
        src.utils.plot_box(ax, lons=LONS_E, lats=[-5, 5], **kwargs)


    ## funcs to average over regions
    def get_eq_avg(data, lons):
        """average over equatorial region and longitudes"""
        return data.sel(longitude=slice(*lons), latitude=slice(-5, 5)).mean(
            ["longitude", "latitude"]
        )


    def get_e(data):
        return get_eq_avg(data, LONS_E)


    def get_w(data):
        return get_eq_avg(data, LONS_W)


    def get_dx(data):
        return get_e(data) - get_w(data)


    def add_gridlines(axs):
        """func to add gridlines to axs object"""

        for ax in axs[:-1, 0]:
            ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=1,
                alpha=0,
                xlocs=[],
                ylocs=[-20, 0, 20],
            )
        for ax in axs[-1]:
            gl_ = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=1,
                alpha=0,
                xlocs=[150, -170, -120, -80],
                ylocs=[-20, 0, 20],
            )
            gl_.top_labels = False
        gl_.left_labels = False

        return


    def plot_level(ax, data, level, ls="-", c="white"):
        """plot single level on hovmoller"""
        ax.contour(
            data.longitude,
            data.latitude,
            data,
            levels=[level],
            colors=c,
            linestyles=ls,
            zorder=10,
            transform=ccrs.PlateCarree(),
        )
        return

    return add_gridlines, get_dx, get_e, get_w, plot_ebox, plot_wbox


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spatial plot
    """)
    return


@app.cell
def _(
    add_gridlines,
    comps_anom_norm,
    comps_ref_sst,
    np,
    plot_ebox,
    plot_sst2,
    plot_wbox,
    plt,
    rsst_model,
    rsst_ref,
    save,
    src,
):
    ## specify intervals
    dxs = np.array(
        [
            [1, 0.3],
            [0.6, 0.6],
            [0.6, 0.6],
            [0.3, 0.3],
        ],
    )

    _fig = plt.figure(figsize=(10, 7), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=21, lon_range=(130, 285))
    _axs = src.utils.subplots_with_proj(_fig, nrows=4, ncols=2, format_func=_format_func)

    cp00 = plot_sst2(_axs[0, 0], rsst_ref, dx=dxs[0, 0])
    cp01 = plot_sst2(_axs[0, 1], rsst_model - rsst_ref, dx=dxs[0, 1])

    cp10 = plot_sst2(_axs[1, 0], comps_ref_sst["warm"], dx=dxs[1, 0])
    cp11 = plot_sst2(_axs[1, 1], comps_anom_norm["warm"], dx=dxs[1, 1])

    cp20 = plot_sst2(_axs[2, 0], comps_ref_sst["cold"], dx=dxs[2, 0])
    cp21 = plot_sst2(_axs[2, 1], comps_anom_norm["cold"], dx=dxs[2, 1])

    cp30 = plot_sst2(_axs[3, 0], comps_ref_sst["warm"] + comps_ref_sst["cold"], dx=dxs[3, 0])
    cp31 = plot_sst2(
        _axs[3, 1],
        comps_anom_norm["warm"] + comps_anom_norm["cold"],
        dx=dxs[3, 1],
    )

    ## add formatting to axs
    for _ax in _axs.flatten():
        plot_wbox(_ax, c="gray", lw=0.8)
        plot_ebox(_ax, c="gray", lw=0.8)
        src.utils.plot_nino34_box(_ax, c="w", ls="--", lw=0.8, alpha=0.75)

    ## label
    _axs[0, 0].set_title("(a) Relative SST mean (ERSST)", loc="left")
    _axs[0, 1].set_title(
        r"(b) Relative SST mean bias $(\text{CESM2} - \text{ERSST})$", loc="left"
    )
    _axs[1, 0].set_title("(c) El Niño (ERSST)", loc="left")
    _axs[1, 1].set_title("(d) El Niño (CESM2, scaled)", loc="left")
    _axs[2, 0].set_title("(e) La Niña (ERSST)", loc="left")
    _axs[2, 1].set_title("(f) La Niña (CESM2, scaled)", loc="left")
    _axs[3, 0].set_title("(g) El Niño + La Niña (ERSST)", loc="left")
    _axs[3, 1].set_title("(h) El Niño + La Niña (CESM2, scaled)", loc="left")


    ## add labels
    add_gridlines(_axs)

    bbox = dict(boxstyle="round", facecolor="white", alpha=1)
    legend_kwargs = dict(x=0.01, y=0.02, fontdict=dict(size=12, color="gray"), bbox=bbox)
    for _ax, dx in zip(_axs.flatten(), dxs.flatten()):
        _ax.text(s=f"Interval: {dx} K", transform=_ax.transAxes, **legend_kwargs)

    ## SAVE
    save(_fig, "spatial-valid")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scatter plot
    """)
    return


@app.function
def label_subplot(ax, label, posn):
    """Add label to subplot in custom style"""
    posnx, posny = posn

    ## kwargs for bounding box
    bbox = dict(boxstyle="round", facecolor="white", alpha=1)

    ## add label
    ax.text(
        posnx,
        posny,
        label,
        ha="center",
        va="center",
        bbox=bbox,
        transform=ax.transAxes,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### reconstruct indices
    """)
    return


@app.cell
def _(
    anom_,
    anom_ond,
    get_dx,
    get_e,
    get_w,
    np,
    plt,
    save,
    scale_anom,
    scale_ref,
    sns,
    src,
    sst_ref_3,
    sst_ref_ond,
):
    ## specify x func
    x_fn = src.utils.get_nino34

    ## get scale for CESM for comparable plotting
    scale_ratio_1 = scale_ref / scale_anom

    ## compute indices
    x_anom = scale_ratio_1 * src.utils.reconstruct_wrapper(anom_ond, x_fn)["sst"]
    Te_anom = scale_ratio_1 * src.utils.reconstruct_wrapper(anom_ond, get_e)["sst"]
    Tw_anom = scale_ratio_1 * src.utils.reconstruct_wrapper(anom_ond, get_w)["sst"]

    ## get grad (already scaled)
    dTdx_anom = Te_anom - Tw_anom

    ## compute PDFs

    # get indices
    pdf_idx_fn = src.utils.get_nino3
    # pdf_idx_fn = get_dx
    pdf_idx_anom = src.utils.reconstruct_wrapper(anom_, pdf_idx_fn)["sst"]
    pdf_idx_ref = pdf_idx_fn(sst_ref_3)

    ## compute
    bins = np.arange(-4.25, 4.75, 0.5)
    pdf_anom, _ = src.utils.get_empirical_pdf(pdf_idx_anom, bins)
    pdf_ref, _ = src.utils.get_empirical_pdf(pdf_idx_ref, bins)

    ## args for scatter plot
    kwargs = dict(s=25)

    ## make scatter plot
    _fig, _axs = plt.subplots(1, 3, figsize=(9, 2.5), layout="constrained")

    ## plot data
    _axs[0].scatter(x_fn(sst_ref_ond), get_dx(sst_ref_ond), s=25, c="gray")
    _axs[1].scatter(x_anom, dTdx_anom, alpha=0.3, s=1, c="gray")

    ## guidelines
    for _ax in _axs[:2]:
        _ax.axhline(0, ls="--", c="k", lw=0.5)
        _ax.axvline(0, ls="--", c="k", lw=0.5)
        _ax.set_xticks([-3, 0, 3])
        zz = np.linspace(-1, 3.5)
        _ax.plot(zz, zz, c="k", lw=0.8)
        _ax.set_xlabel("Niño 3.4 (K)")

    ## pdfs
    _axs[-1].stairs(pdf_ref, bins, fill=sns.color_palette()[0], alpha=0.3, label="ERSST")
    _axs[-1].stairs(pdf_anom, bins, label="CESM2", lw=2)
    _axs[-1].axvline(0, c="k", lw=0.8, ls="--")

    ## formatting
    src.utils.set_lims(_axs[:2])
    _axs[0].set_yticks([-1.5, 0, 1.5, 3])
    _axs[0].set_ylabel(r"$T_e-T_w$ (K)")
    _axs[1].set_yticks([])
    _axs[-1].legend()
    _axs[0].set_title("(i) ERSST", loc="left")
    _axs[1].set_title("(j) CESM2", loc="left")
    _axs[2].set_title("(k) PDFs", loc="left")
    _axs[2].set_xticks([-3, 0, 3])
    _axs[2].yaxis.set_label_position("right")
    _axs[2].yaxis.tick_right()
    _axs[2].set_yticks([0, 0.3, 0.6])
    _axs[2].set_xlim([-3.75, 3.75])
    _axs[2].set_xlabel("Niño 3 (K)")
    _axs[2].set_ylabel("Prob. density")

    # ## label with letters
    # label_subplot(axs[0], label="(i)", posn=(0.1, 0.9))
    # label_subplot(axs[1], label="(j)", posn=(0.1, 0.9))
    # label_subplot(axs[2], label="(k)", posn=(0.1, 0.9))

    ## save to file
    save(_fig, "spatial-asym-scatter")


    plt.gca()
    return pdf_anom, scale_ratio_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectral stuff
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load data
    """)
    return


@app.cell
def _(coord_dict, pathlib, src, xr):
    ## reference data
    _REF_FP = pathlib.Path("/glade/campaign/cgd/cas/asphilli/CVDP-OBS/")
    sst_ref_4 = xr.open_dataset(_REF_FP / "ersstv5.185401-202412.nc")[["ssta"]]

    ## assign time coord
    sst_ref_4["time"] = xr.date_range(start="1854-01", end="2024-12", freq="MS")
    sst_ref_4 = sst_ref_4.rename(coord_dict)

    # ## ccompute ref indices
    sst_ref_4["T_34"] = src.utils.get_nino34(sst_ref_4["ssta"])
    sst_ref_4["T_3"] = src.utils.get_nino3(sst_ref_4["ssta"])
    sst_ref_4 = sst_ref_4.drop_vars(["ssta", "latitude", "longitude"])

    # ## load data
    # data_oras = src.utils.load_oras_spatial_extended( DATA_FP / "oras", varnames=["tos"])
    # data_oras = data_oras[["sst"]]

    # ## estimate forced signal by removing 2nd-order polynomial from each calendar month
    # detrend_fn = lambda x: src.utils.detrend_dim(x, dim="time", deg=2)
    # sst_ref = data_oras.groupby("time.month").map(detrend_fn)

    # ## indices
    # sst_ref["T_34"] = src.utils.get_nino34(sst_ref["sst"])
    # sst_ref["T_3"] = src.utils.get_nino3(sst_ref["sst"])

    ## load CESM2
    Th = src.utils.load_cesm_indices()
    return Th, sst_ref_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get scaled metrics
    """)
    return


@app.cell
def _(Th, scale_ratio_1):
    for _v in ["T_3", "T_34"]:
        Th[f"{_v}_norm"] = Th[_v] * scale_ratio_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### overlapping times
    """)
    return


@app.cell
def _(Th, sst_ref_4):
    _time_idx = dict(time=slice("1958-01", "2023-12"))
    # time_idx = dict(time=slice("1958-01", "2020-12"))
    # time_idx = dict(time=slice("1979-01", "2020-12"))
    sst_mod = Th.sel(_time_idx)
    sst_ref_5 = sst_ref_4.sel(_time_idx)

    ## subtract mean
    sst_mod = sst_mod - sst_mod.mean()
    sst_ref_5 = sst_ref_5 - sst_ref_5.mean()
    return sst_mod, sst_ref_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute PSD
    """)
    return


@app.cell
def _(src, sst_mod, sst_ref_5):
    ## specify which variable to use
    varname = "T_34"

    ## specify args for psd
    psd_kwargs = dict(dim="time", dt=1 / 12, nw=5)

    ## compute PSD
    compute_psd = lambda x: src.XRO_utils.pmtm(x[varname], **psd_kwargs)
    psd_ref = compute_psd(sst_ref_5)
    psd_mod = compute_psd(sst_mod)
    return psd_mod, psd_ref


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute seasonal sync
    """)
    return


@app.cell
def _(sst_mod, sst_ref_5):
    ## reference
    sigma_ref = sst_ref_5.groupby("time.month").std()

    ## model
    sigma_mod = sst_mod.groupby("time.month").std()
    sigma_mod = sigma_mod.quantile(q=[0.1, 0.5, 0.9], dim="member")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute PDFs
    """)
    return


@app.cell
def _(np, src, sst_mod, sst_ref_5):
    def get_seasonal(x):
        x_seasonal = (x - x.mean()).resample({"time": "QS-NOV"}).mean()
        return x_seasonal


    ## compute
    bins_1 = np.arange(-4.25, 4.75, 0.5)
    pdf_mod, _ = src.utils.get_empirical_pdf(get_seasonal(sst_mod)["T_3"], bins_1)
    pdf_ref_1, _ = src.utils.get_empirical_pdf(get_seasonal(sst_ref_5)["T_3"], bins_1)
    return bins_1, pdf_ref_1


@app.cell
def _(bins_1, pdf_anom, pdf_ref_1, plt, psd_mod, psd_ref, sns, src):
    _fig, _axs = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")

    ## plot psd
    src.utils.plot_psd(_axs[0], psd_ref, label="ERSST")
    src.utils.plot_psd(_axs[0], psd_mod, label="CESM2")

    # ## plot seasonal sync.
    # axs[1].plot(sigma_ref.month, sigma_ref["T_34"], label="ERSST", lw=2)
    # p_ = axs[1].plot(sigma_mod.month, sigma_mod["T_34"].sel(quantile=0.5), label="CESM2", lw=2)
    # for q in [.1, .9]:
    #     kwargs = dict(lw=1, ls="--", c=p_[0].get_color())
    #     p_ = axs[1].plot(sigma_mod.month, sigma_mod["T_34"].sel(quantile=q), **kwargs)


    ## pdfs
    _axs[-1].stairs(pdf_ref_1, bins_1, fill=sns.color_palette()[0], alpha=0.3, label="ERSST")
    _axs[-1].stairs(pdf_anom, bins_1, label="CESM2", lw=2)


    ## label
    _axs[0].set_ylabel(r"PSD ($^{\circ}$C$^2$/cpm)")
    _axs[0].legend()
    _axs[0].set_xlim([None, 3])
    # axs[1].set_ylim([0,None])
    _axs[-1].axvline(0, c="k", lw=0.8, ls="--")
    _axs[-1].set_title("(k) PDFs", loc="left")
    _axs[-1].set_xticks([-3, 0, 3])
    _axs[-1].yaxis.set_label_position("right")
    _axs[-1].yaxis.tick_right()
    _axs[-1].set_yticks([0, 0.3, 0.6])
    _axs[-1].set_xlim([-3.75, 3.75])
    _axs[-1].set_xlabel("Niño 3 (K)")
    _axs[-1].set_ylabel("Prob. density")
    _axs[-1].legend()


    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### spaghetti
    """)
    return


@app.cell
def _(src):
    def get_spaghetti(idx, data, peak_month, event_type=None, q=0.95, is_warm=True):
        """
        Get hovmoller composite based on specified:
        - data: used to compute index/make composite
        - peak_month: month to center composite on
        - q: quantile threshold for composite
        """

        ## handle warm/cold case
        if is_warm:
            kwargs = dict(q=q, check_cutoff=lambda x, cut: x > cut)
        else:
            kwargs = dict(q=1 - q, check_cutoff=lambda x, cut: x < cut)

        ## kwargs for composite
        kwargs = dict(
            kwargs,
            avg=False,
            peak_month=peak_month,
            idx=idx,
            data=data,
            event_type=event_type,
        )

        ## composite of projected data
        spag = src.utils.make_composite(**kwargs)

        return spag

    return (get_spaghetti,)


@app.cell
def _(get_spaghetti, sst_mod, sst_ref_5):
    ## specify composite specs
    comp_kwargs = dict(
        event_type=None,
        peak_month=12,
        q=0.9,
    )

    ## specify variable to use for composite
    VARNAME = "T_34"

    ## do the compute
    spag_warm_ref = get_spaghetti(
        is_warm=True, idx=sst_ref_5[VARNAME], data=sst_ref_5, **comp_kwargs
    )
    spag_warm_mod = get_spaghetti(
        is_warm=True, idx=sst_mod[VARNAME], data=sst_mod, **comp_kwargs
    )

    spag_cold_ref = get_spaghetti(
        is_warm=False, idx=sst_ref_5[VARNAME], data=sst_ref_5, **comp_kwargs
    )
    spag_cold_mod = get_spaghetti(
        is_warm=False, idx=sst_mod[VARNAME], data=sst_mod, **comp_kwargs
    )

    ## concatenate for convenience
    spags_ref = [spag_warm_ref, spag_cold_ref]
    spags_mod = [spag_warm_mod, spag_cold_mod]
    return spags_mod, spags_ref


@app.cell
def _():
    def plot_spaghetti(ax, spags, n="T_34"):
        ## plot ensemble mean
        kwargs = dict(lw=5, ls="--")
        ax.plot(spags[0].lag, spags[0][n].mean("sample"), c="r", **kwargs)
        ax.plot(spags[1].lag, -spags[1][n].mean("sample"), c="b", **kwargs)

        ## plot samples
        spag_kwargs = dict(lw=1.5, alpha=0.7, ls="-")
        ax.plot(spags[0].lag, spags[0][n], c="r", **spag_kwargs)
        ax.plot(spags[1].lag, -spags[1][n], c="b", **spag_kwargs)
        return


    def plot_spaghetti_ensemble(ax, spags, n="T_34"):

        ## plot ensemble mean
        kwargs = dict(lw=5, ls="--")
        ax.plot(spags[0].lag, spags[0][n].quantile(q=0.5, dim="sample"), c="r", **kwargs)
        ax.plot(spags[1].lag, -spags[1][n].quantile(q=0.5, dim="sample"), c="b", **kwargs)

        ## plot bounds
        spag_kwargs = dict(lw=2, alpha=1, ls="-")
        for q in [0.1, 0.9]:
            ax.plot(
                spags[0].lag, spags[0][n].quantile(q=q, dim="sample"), c="r", **spag_kwargs
            )
            ax.plot(
                spags[1].lag, -spags[1][n].quantile(q=q, dim="sample"), c="b", **spag_kwargs
            )
        return

    return plot_spaghetti, plot_spaghetti_ensemble


@app.cell
def _(
    plot_spaghetti,
    plot_spaghetti_ensemble,
    plt,
    psd_mod,
    psd_ref,
    save,
    spags_mod,
    spags_ref,
    src,
):
    _fig, _axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

    ## reference
    plot_spaghetti(_axs[1], spags=spags_ref, n="T_34")
    plot_spaghetti_ensemble(_axs[2], spags=spags_mod, n="T_34_norm")

    ax_kwargs = dict(c="gray", lw=0.8)
    for _ax in _axs[1:]:
        _ax.set_xlim([-2, None])
        _ax.axhline(0, **ax_kwargs)
        _ax.axvline(0, **ax_kwargs)
        _ax.set_xticks([0, 6, 12])
        _ax.set_xlabel("Lag")

    ## format/label
    src.utils.set_lims(_axs)
    _axs[1].set_yticks([])
    _axs[2].set_yticks([-2.5, 0, 2.5])
    _axs[2].set_ylabel("Niño 3.4 (K)")
    _axs[1].set_title("(m) ERSST", loc="left")
    _axs[2].set_title("(n) CESM2 (scaled)", loc="left")
    _axs[-1].yaxis.set_label_position("right")
    _axs[-1].yaxis.tick_right()

    ## spectrum
    src.utils.plot_psd(_axs[0], psd_ref, label="ERSST")
    src.utils.plot_psd(_axs[0], psd_mod, label="CESM2")
    _axs[0].set_ylabel(r"PSD ($^{\circ}$C$^2$/cpm)")
    _axs[0].legend()
    _axs[0].set_xlim([None, 3])
    _axs[0].set_title("(l) Power spec. (Niño 3.4)", loc="left")

    label_subplot(_axs[0], label=r"                                ", posn=(0.5, 0.97))
    # axs[0].set_ylim([None,7])

    save(_fig, "psd_spaghetti")

    plt.gca()

    # break
    # ax.plot(spag_warm_ref.lag, -spag_cold_ref["T_34"].mean("sample"))
    # plt.plot(spag_warm_mod["T_34"].mean("sample"))
    return


if __name__ == "__main__":
    app.run()
