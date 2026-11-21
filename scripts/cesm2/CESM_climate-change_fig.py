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
    # Make climate change figure for poster
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
    import cartopy.crs as ccrs

    # Import custom modules
    import src.utils

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})
    sns.set_palette("colorblind")

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return DATA_FP, ccrs, mpl, np, os, pathlib, plt, src, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## funcs
    """)
    return


@app.cell
def _(os, pathlib):
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
    ## Load data
    """)
    return


@app.cell
def _(src):
    ## load spatial data
    forced, anom = src.utils.load_consolidated()
    return anom, forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### mean state / climatology
    """)
    return


@app.cell
def _():
    t0 = dict(time=slice("1851","1880"))
    t1 = dict(time=slice("2061","2090"))
    return t0, t1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### spatial patterns
    """)
    return


@app.cell
def _(forced, src, t0, t1):
    ## average
    clim0 = forced.sel(t0).mean("time")
    clim1 = forced.sel(t1).mean("time")
    dclim = clim1-clim0

    ## reconstruct
    var_list = ["sst", "T", "w"]
    var_list = var_list + [f"{v}_comp" for v in var_list]
    clim0_spatial = src.utils.reconstruct_wrapper(clim0[var_list])
    clim1_spatial = src.utils.reconstruct_wrapper(clim1[var_list])
    return clim0_spatial, clim1_spatial


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### tropical SST
    """)
    return


@app.cell
def _(DATA_FP, forced, pathlib, t0, t1, xr):
    ## tropical sst for model
    trop_sst = xr.open_dataset(pathlib.Path(DATA_FP, "cesm/trop_sst.nc"))
    trop_sst = trop_sst["trop_sst_15"].sel(time=forced.time)
    trop_sst0 = trop_sst.sel(t0).mean()
    trop_sst1 = trop_sst.sel(t1).mean()
    return trop_sst0, trop_sst1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### relative SST
    """)
    return


@app.cell
def _(clim0_spatial, clim1_spatial, trop_sst0, trop_sst1):
    ## get relative sst
    clim0_spatial["rsst"] = clim0_spatial["sst"] - trop_sst0
    clim1_spatial["rsst"] = clim1_spatial["sst"] - trop_sst1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get difference
    """)
    return


@app.cell
def _(clim0_spatial, clim1_spatial):
    dclim_spatial = clim1_spatial-clim0_spatial
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ENSO pattern
    """)
    return


@app.cell
def _(src, xr):
    def get_djf_enso(data):
        """Get DJF pattern for ENSO"""
        ## resample to djf
        data_seasonal = data["sst"].resample({"time":"QS-DEC"}).mean()

        ## get djf
        is_djf = data_seasonal.time.dt.month == 12
        data_djf = data_seasonal.sel(time=is_djf).isel(time=slice(1,-1))

        ## get Niño 3.4 index
        T34 = src.utils.reconstruct_fn(
            scores=data_djf,
            components=data["sst_comp"],
            fn=src.utils.get_nino34,
        )

        ## standardize it
        T34 = T34 / T34.std()

        ## merge data
        XY = xr.merge([T34.rename("T34"), data_djf])

        ## get coefficients
        m_proj = src.utils.regress_xr_proj(XY, x_vars=["T34"], y_vars=["sst"])
        m_proj = m_proj["sst"].squeeze(drop=True)

        ## reconstruct
        m = src.utils.reconstruct_fn(
            scores=m_proj,
            components=data["sst_comp"],
            fn =lambda x : x,
        )
        

        return m

    return (get_djf_enso,)


@app.cell
def _(anom, get_djf_enso, t0):
    #### Get ENSO pattern
    enso_pattern0 = get_djf_enso(anom.sel(t0))
    return (enso_pattern0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot SST
    """)
    return


@app.cell
def _(ccrs, np, src):
    ## specify lons/lats for E/W boxes
    LONS_E = [240, 280]
    LONS_W = [150, 190]

    def plot_wbox(ax, **kwargs):
    ## funcs to plot boxes
        src.utils.plot_box(ax, lons=LONS_W, lats=[-5, 5], **kwargs)

    def plot_ebox(ax, **kwargs):
        src.utils.plot_box(ax, lons=LONS_E, lats=[-5, 5], **kwargs)

    def get_eq_avg(data, lons):
        """average over equatorial region and longitudes"""
        return data.sel(longitude=slice(*lons), latitude=slice(-5, 5)).mean(['longitude', 'latitude'])
    ## funcs to average over regions

    def get_e(data):
        return get_eq_avg(data, LONS_E)

    def get_w(data):
        return get_eq_avg(data, LONS_W)

    def get_dx(data):
        return get_e(data) - get_w(data)

    def add_gridlines(axs):
        """func to add gridlines to axs object"""
        for ax in axs[:-1, 0]:
            ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, alpha=0, xlocs=[], ylocs=[-20, 0, 20])
        for ax in axs[-1]:
            gl_ = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, alpha=0, xlocs=[], ylocs=[-20, 0, 20])
            gl_.top_labels = False
        return

    def plot_level(ax, data, level, ls='-', c='white'):
        """plot single level on hovmoller"""
        ax.contour(data.longitude, data.latitude, data, levels=[level], colors=c, linestyles=ls, zorder=10, transform=ccrs.PlateCarree())
        return

    def make_cb_range(dx, nlevs):
        amp = dx * nlevs - dx / 2
        lev1 = np.arange(0, amp + dx / 2, dx) + dx / 2
        lev0 = -lev1[::-1]
        return np.concatenate([lev0, lev1])

    def plot_sst2(ax, data, dx=0.2, nlev=5):
        """plot data on ax"""
        cp = ax.contourf(data.longitude, data.latitude, data, cmap='cmo.balance', levels=make_cb_range(dx, nlev), transform=ccrs.PlateCarree(), extend='both')
        return cp

    def plot_sst_sigma(ax, data, lev_min=0.3, dx=0.2, nlev=5):
        """plot data on ax"""
        cp = ax.contourf(data.longitude, data.latitude, data, cmap='cmo.amp', levels=np.arange(lev_min, lev_min + dx * (nlev + 1), dx), transform=ccrs.PlateCarree(), extend='both')  # xlocs=[150, -170, -120, -80],
        return cp  # gl_.left_labels = False

    return add_gridlines, plot_ebox, plot_sst2, plot_wbox


@app.cell
def _(
    add_gridlines,
    ccrs,
    clim0_spatial,
    enso_pattern0,
    np,
    plot_ebox,
    plot_sst2,
    plot_wbox,
    plt,
    save,
    src,
):
    ## specify intervals
    dxs = np.array([0.8, 0.4])
    _fig = plt.figure(figsize=(6, 4.5), layout='constrained')
    format_func = lambda ax: src.utils.plot_setup_pac(ax, max_lat=21, lon_range=(140, 285))
    axs = src.utils.subplots_with_proj(_fig, nrows=2, ncols=1, format_func=format_func)
    cp00 = plot_sst2(axs[0, 0], clim0_spatial['rsst'], dx=dxs[0])
    cp10 = plot_sst2(axs[1, 0], enso_pattern0, dx=dxs[1])
    for _ax in axs.flatten():
        plot_wbox(_ax, c='gray', lw=1.5)
    # cp01 = plot_sst_sigma(axs[1, 0], sst_sigma0, dx=dxs[1])
        plot_ebox(_ax, c='gray', lw=1.5)
    ## add formatting to axs
        src.utils.plot_nino34_box(_ax, c='w', ls='--', lw=1.5, alpha=0.75)
    axs[0, 0].set_title('(a) Relative SST mean', loc='left')
    axs[1, 0].set_title('(b) SST ENSO pattern', loc='left')
    for loc, label, c in zip([(165, 7), (212.5, 7), (258, 7)], ['$T_w$', '$T_{34}$', '$T_e$'], ['w', 'k', 'k']):
        axs[0, 0].text(x=loc[0], y=loc[1], s=label, color=c, transform=ccrs.PlateCarree())
    ## label
    add_gridlines(axs)
    _bbox = dict(boxstyle='round', facecolor='white', alpha=1)
    _legend_kwargs = dict(x=0.01, y=0.02, fontdict=dict(size=12, color='gray'), bbox=_bbox)
    for _ax, dx in zip(axs.flatten(), dxs.flatten()):
        _ax.set_aspect('auto')
    ## label regions
    cb_kwargs = dict(label='$^{\\circ}$C', pad=0.03)
    _fig.colorbar(cp00, ax=axs[0, 0], ticks=[-3.6, 0, 3.6], **cb_kwargs)
    _fig.colorbar(cp10, ax=axs[1, 0], ticks=[-1.8, 0, 1.8], **cb_kwargs)
    save(_fig, 'spatial_plot', do_save=False)
    ## add labels
    ## colorbars
    ## SAVE
    # save(fig, "spatial-valid")
    plt.gca()  # ax.text(s=f"Interval: {dx} K", transform=ax.transAxes, **legend_kwargs)
    return


@app.cell
def _(clim0_spatial, mpl, np, plt, save, src):
    _fig, _ax = plt.subplots(figsize=(6.0, 2.5), layout='constrained')
    cp_T = _ax.contourf(clim0_spatial['T'].longitude, clim0_spatial.z_t, clim0_spatial['T'], cmap='cmo.thermal', levels=np.arange(10, 31, 3), extend='both')
    ## temperature
    cp_w = _ax.contour(clim0_spatial['w'].longitude, clim0_spatial.z_t, clim0_spatial['w'], colors='k', alpha=0.9, levels=src.utils.make_cb_range(100, 10), extend='both', linewidths=1)
    _ax.add_patch(mpl.patches.Rectangle(height=-80, linewidth=2, ls='-', xy=(210, 80), width=60, edgecolor='w', fill=None))
    _ax.axhline(50, ls='--', c='w', lw=1.5)
    _bbox = dict(edgecolor='none', facecolor='none')
    _legend_kwargs = dict(x=0.01, y=0.03, fontdict=dict(size=10, color='white'), bbox=_bbox)
    _ax.text(s=f'Upwelling interval: 10 m/month', transform=_ax.transAxes, **_legend_kwargs)  # levels=np.arange(12,30,2),
    _bbox = dict(edgecolor='none', facecolor='white', alpha=0)
    _legend_kwargs = dict(fontdict=dict(size=10, color='white'), bbox=_bbox)
    _ax.text(s='$T_{\\text{ml}}$', x=245, y=33, zorder=10, **_legend_kwargs)
    _ax.text(s='$T_\\text{sub}$, $w$', x=245, y=70, **_legend_kwargs)
    # ## upwelling
    _ax.set_xlim([140, 285])
    _ax.set_ylim([250, 5])
    _ax.set_xticks([140, 210, 280], labels=['140$^{\\circ}$E', '150$^{\\circ}$W', '80$^{\\circ}$W'])
    _ax.set_yticks([5, 50, 150, 250], labels=[0, 50, 150, 250])
    _ax.set_ylabel('Depth (m)')
    _ax.set_title('(c) Equatorial temperature & upwelling (1.5$^{\\circ}$S$-$1.5$^{\\circ}$N)', loc='left')
    cb = _fig.colorbar(cp_T, ticks=[10, 19, 28], label='$^{\\circ}$C', pad=-0.02)
    save(_fig, 'depth_plot', do_save=False)
    ## plot 80-m MLD
    ## legend for upwelling
    ## legend for mixed layer, w
    ## label / formatting
    ## colorbar
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
