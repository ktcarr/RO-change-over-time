import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # CESM2 mean state and its change under warming

    Paper figure `fig:cesm2-meanstate` (`paper-outline.tex` in the ch3 Overleaf project).

    - **(a)** Relative SST (SST minus the 15°S–15°N mean), averaged over 1851–1880
    - **(b)** ENSO SST pattern: DJF SST regressed on the standardized Niño 3.4 index, 1851–1880
    - **(c)** Equatorial temperature (shading) and vertical velocity (contours), 1851–1880
    - **(d–g)** Fractional change over time, in 40-year windows, of ENSO amplitude
      $\sigma(T_{34})$, stratification $\overline{T}_{\text{ml}}-\overline{T}_{\text{sub}}$,
      upwelling $\overline{w}$, and the zonal SST gradient $\overline{T}_w-\overline{T}_e$.
      The dashed magenta line is tropical-mean warming.

    **Inputs** (under `DATA_FP/cesm`): `consolidated/` and `consolidated_05/` (EOF-compressed
    fields, from `preprocess/consolidate.py`), `trop_sst.nc` (from `preprocess/preprocess_cesm.py`),
    and the Niño 3.4 index chosen by `T34_INDEX`: `Th_anom.nc` (SST-based, the default; from
    `preprocess/preprocess_cesm.py`) or `Th_50.nc`/`Th_80.nc` (from `preprocess/merge_Th_3D.py`).

    **Output:** `SAVE_FP/figures/cesm2-meanstate.pdf`

    Before this script, the figure was put together by hand from `spatial_plot.pdf`,
    `depth_plot.pdf` (from `CESM_climate-change_fig.py`, now in `explore/`), and
    `timeseries.pdf` (from `timeseries.py`, also now in `explore/`).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Imports
    """
    )
    return


@app.cell
def _():
    ## standard library
    import os
    import pathlib

    ## plotting and data libraries
    import cartopy.crs as ccrs
    import cmocean  # noqa: F401 (registers the "cmo.*" colormaps)
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    ## numerics, plot style, labeled arrays
    import numpy as np
    import seaborn as sns
    import xarray as xr

    ## this project's modules (shared figure helpers, general utilities)
    import src.figure_utils as fu
    import src.utils

    ## plotting style used for all paper figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")

    ## folder with the preprocessed data
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    return DATA_FP, ccrs, fu, mpl, np, plt, src, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Settings
    """
    )
    return


@app.cell
def _():
    ## early (preindustrial-like) period, used for the maps and depth section
    T_EARLY = dict(time=slice("1851", "1880"))

    ## Niño 3.4 index for ENSO statistics: "sst" (SST-based, used by the RO fits),
    ## or "ml50"/"ml80" (mixed-layer temperature over the top 50/80 m)
    T34_INDEX = "sst"

    ## window length and stride (months) for the changes over time in (d-g)
    WINDOW_SIZE = 480
    STRIDE = 60

    ## depth (m) of the mixed layer base and thickness of the layer below it, used
    ## for the stratification and upwelling indices in (e, f)
    MLD = 50
    DELTA = 30
    return DELTA, MLD, STRIDE, T34_INDEX, T_EARLY, WINDOW_SIZE


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data
    """
    )
    return


@app.cell
def _(src):
    ## consolidated fields for the maps and depth section (a-c); scores are split
    ## into the forced response (ensemble mean) and anomalies about it
    forced, anom = src.utils.load_consolidated()
    return anom, forced


@app.cell
def _(fu):
    ## total (forced + anomaly) fields for the indices in (e-g), using the wide
    ## consolidated data where available
    total = fu.load_total(["T", "w", "sst"])
    return (total,)


@app.cell
def _(DATA_FP, xr):
    ## tropical-mean SST, 15S-15N (used for relative SST and as a warming index)
    trop_sst = xr.open_dataset(DATA_FP / "cesm" / "trop_sst.nc")["trop_sst_15"]
    return (trop_sst,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (a–c) Early-period mean state and ENSO pattern
    """
    )
    return


@app.cell
def _(T_EARLY, forced, src):
    ## time mean of the forced scores over the early period
    clim0 = forced.sel(T_EARLY).mean("time")

    ## reconstruct spatial fields of SST, subsurface temperature, and vertical velocity
    var_list = ["sst", "T", "w", "sst_comp", "T_comp", "w_comp"]
    clim0_spatial = src.utils.reconstruct_wrapper(clim0[var_list])
    return (clim0_spatial,)


@app.cell
def _(T_EARLY, clim0_spatial, forced, trop_sst):
    ## tropical-mean SST over the early period (a single number: mean over time and members)
    trop_sst0 = trop_sst.sel(time=forced.time).sel(T_EARLY).mean()

    ## relative SST (a): SST minus the tropical mean
    rsst0 = clim0_spatial["sst"] - trop_sst0
    return (rsst0,)


@app.cell
def _(src, xr):
    def get_djf_enso(data):
        """
        DJF SST pattern of ENSO: SST regressed on the standardized Niño 3.4 index.

        Args:
            - data (xr.Dataset): anomaly scores with "sst" and "sst_comp"

        Returns:
            - pattern (xr.DataArray): SST change (K) per standard deviation of Niño 3.4
        """

        ## seasonal means, with seasons starting in December (DJF, MAM, ...)
        data_seasonal = data["sst"].resample({"time": "QS-DEC"}).mean()

        ## keep DJF, dropping the incomplete first and last seasons
        is_djf = data_seasonal.time.dt.month == 12
        data_djf = data_seasonal.sel(time=is_djf).isel(time=slice(1, -1))

        ## Niño 3.4 index from the DJF scores, then standardize it
        T34 = src.utils.reconstruct_fn(
            scores=data_djf,
            components=data["sst_comp"],
            fn=src.utils.get_nino34,
        )
        T34 = T34 / T34.std()

        ## regress the SST scores on the index (over members and time)
        XY = xr.merge([T34.rename("T34"), data_djf])
        m_proj = src.utils.regress_xr_proj(XY, x_vars=["T34"], y_vars=["sst"])
        m_proj = m_proj["sst"].squeeze(drop=True)

        ## reconstruct the regression coefficients as a spatial pattern
        pattern = src.utils.reconstruct_fn(
            scores=m_proj,
            components=data["sst_comp"],
            fn=lambda x: x,
        )

        return pattern

    return (get_djf_enso,)


@app.cell
def _(T_EARLY, anom, get_djf_enso):
    ## ENSO pattern (b) over the early period
    enso_pattern0 = get_djf_enso(anom.sel(T_EARLY))
    return (enso_pattern0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (d–g) Changes over time

    Each index is computed for every month and member, then cut into 40-year windows
    (5-year stride). The figure shows the window mean (or standard deviation, for ENSO
    amplitude) as a fractional change from the first window.
    """
    )
    return


@app.cell
def _(src):
    def get_dTdz_sub(T, mld, delta):
        """
        Vertical temperature gradient across the base of the mixed layer.

        Args:
            - T (xr.DataArray): temperature with a depth dimension "z_t" (m)
            - mld (float): mixed layer depth (m)
            - delta (float): thickness (m) of the layer below the mixed layer

        Returns:
            - dTdz (xr.DataArray): (mixed-layer mean minus sub-layer mean) / mld (K/m)
        """

        ## sub-layer mean minus mixed-layer mean (negative in stable stratification)
        dT = src.utils.get_dT_sub(Tsub=T, Hm=mld, delta=delta)

        ## gradient; positive when the mixed layer is warmer than the water below
        dTdz = -dT / mld

        return dTdz

    def get_w_sub(w, mld, delta):
        """
        Vertical velocity just below the mixed layer.

        Args:
            - w (xr.DataArray): vertical velocity with a depth dimension "z_t" (m)
            - mld (float): mixed layer depth (m)
            - delta (float): thickness (m) of the layer to average over

        Returns:
            - w_sub (xr.DataArray): w averaged from mld to mld + delta
        """

        ## average over the layer below the mixed layer
        w_sub = w.sel(z_t=slice(mld, mld + delta)).mean("z_t")

        return w_sub

    def get_n3_lon_avg(x):
        """
        Average over the Niño 3 longitudes (210-270E).

        Args:
            - x (xr.DataArray): data with a "longitude" dimension

        Returns:
            - avg (xr.DataArray): longitude average
        """
        return x.sel(longitude=slice(210, 270)).mean("longitude")

    return get_dTdz_sub, get_n3_lon_avg, get_w_sub


@app.cell
def _(DELTA, MLD, get_dTdz_sub, get_n3_lon_avg, src, total):
    ## stratification (e): reconstruct the Niño 3 average of dT/dz from the EOF scores
    dTdz_n3 = src.utils.reconstruct_wrapper(
        total[["T", "T_comp"]],
        lambda x: get_dTdz_sub(get_n3_lon_avg(x), mld=MLD, delta=DELTA),
    )["T"]
    return (dTdz_n3,)


@app.cell
def _(DELTA, MLD, get_n3_lon_avg, get_w_sub, src, total):
    ## upwelling (f): Niño 3 average of w just below the mixed layer
    w_n3 = src.utils.reconstruct_wrapper(
        total[["w", "w_comp"]],
        lambda x: get_w_sub(get_n3_lon_avg(x), mld=MLD, delta=DELTA),
    )["w"]
    return (w_n3,)


@app.cell
def _(src, total):
    def get_dTdx(x):
        """
        Zonal SST difference, western box (140-190E) minus eastern box (240-280E), 5S-5N.

        Note the western box here is wider than fu.LONS_W (150-190E), as in the
        original timeseries.py.

        Args:
            - x (xr.DataArray): field with "longitude" and "latitude"

        Returns:
            - dTdx (xr.DataArray): west-minus-east difference
        """

        ## average over 5S-5N and each box
        eq = x.sel(latitude=slice(-5, 5))
        T_w = eq.sel(longitude=slice(140, 190)).mean(["latitude", "longitude"])
        T_e = eq.sel(longitude=slice(240, 280)).mean(["latitude", "longitude"])

        return T_w - T_e

    ## zonal SST gradient (g), reconstructed from the EOF scores
    dTdx = src.utils.reconstruct_wrapper(total[["sst", "sst_comp"]], get_dTdx)["sst"]
    return (dTdx,)


@app.cell
def _(STRIDE, WINDOW_SIZE, dTdx, dTdz_n3, src, trop_sst, w_n3, xr):
    ## combine the mean-state indices with the warming index
    mean_idxs = xr.merge(
        [
            trop_sst,
            dTdz_n3.rename("dTdz_n3"),
            w_n3.rename("w_n3"),
            dTdx.rename("dTdx"),
        ]
    )

    ## cut into windows (dropping the last month, as in the original timeseries.py)
    mean_idxs_windowed = src.utils.get_windowed(
        mean_idxs.isel(time=slice(None, -1)), window_size=WINDOW_SIZE, stride=STRIDE
    )

    ## seasonal means (quarters starting in January), then the mean over each window
    mean_idxs_seasonal = mean_idxs_windowed.resample({"time": "QS-JAN"}).mean()
    mean_clim = mean_idxs_seasonal.mean("time")
    return (mean_clim,)


@app.cell
def _(STRIDE, T34_INDEX, WINDOW_SIZE, fu):
    ## Niño 3.4 anomalies (ensemble mean removed), cut into windows
    T34_windowed = fu.load_T34_windowed(
        index=T34_INDEX, window_size=WINDOW_SIZE, stride=STRIDE
    )

    ## DJF means, measured from the median of each window
    T34_djf = fu.get_djf_anom(T34_windowed)

    ## ENSO amplitude (d): standard deviation of DJF Niño 3.4 in each window and member
    sigma_T = T34_djf.std("time")
    return (sigma_T,)


@app.cell
def _(mean_clim, sigma_T):
    ## all four series in (d-g), on the same windows ("year" = window center)
    series = mean_clim[["dTdz_n3", "w_n3", "dTdx"]]
    series["sigma_T"] = sigma_T.reindex(year=mean_clim.year)

    ## warming index: tropical-mean SST change since the first window (ensemble mean)
    trop_warming = mean_clim["trop_sst_15"].mean("member")
    trop_warming = trop_warming - trop_warming.isel(year=0)
    return series, trop_warming


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting functions
    """
    )
    return


@app.cell
def _(ccrs, fu, src):
    def plot_maps(subfig, rsst, enso_pattern):
        """
        Panels (a) relative SST and (b) ENSO pattern, stacked vertically.

        Args:
            - subfig (matplotlib SubFigure): where to draw
            - rsst (xr.DataArray): relative SST map (K)
            - enso_pattern (xr.DataArray): ENSO SST pattern (K per std. of Niño 3.4)

        Returns:
            - axs (np.ndarray of GeoAxes): the two maps, shape (2, 1)
        """

        ## two maps of the tropical Pacific, centered on the dateline
        proj = ccrs.PlateCarree(central_longitude=190)
        axs = subfig.subplots(2, 1, subplot_kw=dict(projection=proj), squeeze=False)
        for ax in axs.flatten():
            src.utils.plot_setup_pac(ax, max_lat=21, lon_range=(140, 285))

        ## filled contours, with different intervals for the two fields
        dxs = [0.8, 0.4]
        cp_a = fu.plot_sst_map(axs[0, 0], rsst, dx=dxs[0])
        cp_b = fu.plot_sst_map(axs[1, 0], enso_pattern, dx=dxs[1])

        ## outline the T_w and T_e boxes (gray) and the Niño 3.4 box (dashed white)
        for ax in axs.flatten():
            fu.plot_wbox(ax, c="gray", lw=1.5)
            fu.plot_ebox(ax, c="gray", lw=1.5)
            src.utils.plot_nino34_box(ax, c="w", ls="--", lw=1.5, alpha=0.75)

        ## label the boxes on the top map
        box_labels = [
            ((165, 7), "$T_w$", "w"),
            ((212.5, 7), "$T_{34}$", "k"),
            ((258, 7), "$T_e$", "k"),
        ]
        for loc, label, c in box_labels:
            axs[0, 0].text(
                x=loc[0], y=loc[1], s=label, color=c, transform=ccrs.PlateCarree()
            )

        ## titles and latitude labels
        axs[0, 0].set_title("(a) Relative SST mean", loc="left")
        axs[1, 0].set_title("(b) SST ENSO pattern", loc="left")
        fu.add_lat_labels(axs)

        ## let the maps stretch to fill the space
        for ax in axs.flatten():
            ax.set_aspect("auto")

        ## colorbars
        cb_kwargs = dict(label="$^{\\circ}$C", pad=0.03)
        subfig.colorbar(cp_a, ax=axs[0, 0], ticks=[-3.6, 0, 3.6], **cb_kwargs)
        subfig.colorbar(cp_b, ax=axs[1, 0], ticks=[-1.8, 0, 1.8], **cb_kwargs)

        return axs

    return (plot_maps,)


@app.cell
def _(mpl, np, src):
    def plot_depth_section(subfig, T, w):
        """
        Panel (c): equatorial temperature (shading) and vertical velocity (contours).

        Args:
            - subfig (matplotlib SubFigure): where to draw
            - T (xr.DataArray): temperature (degC), dims (z_t, longitude)
            - w (xr.DataArray): vertical velocity, dims (z_t, longitude)

        Returns:
            - ax (matplotlib Axes): the depth section
        """

        ## one panel
        ax = subfig.subplots()

        ## temperature as filled contours
        cp_T = ax.contourf(
            T.longitude,
            T.z_t,
            T,
            cmap="cmo.thermal",
            levels=np.arange(10, 31, 3),
            extend="both",
        )

        ## vertical velocity as black contours
        ax.contour(
            w.longitude,
            w.z_t,
            w,
            colors="k",
            alpha=0.9,
            levels=src.utils.make_cb_range(100, 10),
            linewidths=1,
        )

        ## box for the subsurface (T_sub, w) region and line at the mixed layer base (50 m)
        box = mpl.patches.Rectangle(
            xy=(210, 80), width=60, height=-80, linewidth=2, edgecolor="w", fill=None
        )
        ax.add_patch(box)
        ax.axhline(50, ls="--", c="w", lw=1.5)

        ## contour interval for w, and labels for the mixed layer and subsurface regions
        ax.text(
            x=0.01,
            y=0.03,
            s="Upwelling interval: 10 m/month",
            transform=ax.transAxes,
            fontdict=dict(size=10, color="white"),
        )
        ax.text(
            s="$T_{\\text{ml}}$",
            x=245,
            y=33,
            zorder=10,
            fontdict=dict(size=10, color="white"),
        )
        ax.text(
            s="$T_\\text{sub}$, $w$", x=245, y=70, fontdict=dict(size=10, color="white")
        )

        ## axis limits: depth increases downward
        ax.set_xlim([140, 285])
        ax.set_ylim([250, 5])

        ## longitude ticks
        ax.set_xticks(
            [140, 210, 280],
            labels=["140$^{\\circ}$E", "150$^{\\circ}$W", "80$^{\\circ}$W"],
        )

        ## depth ticks and axis label
        ax.set_yticks([5, 50, 150, 250], labels=[0, 50, 150, 250])
        ax.set_ylabel("Depth (m)")

        ## title
        ax.set_title(
            "(c) Equatorial temperature & upwelling (1.5$^{\\circ}$S$-$1.5$^{\\circ}$N)",
            loc="left",
        )

        ## colorbar
        subfig.colorbar(
            cp_T, ax=ax, ticks=[10, 19, 28], label="$^{\\circ}$C", pad=-0.02
        )

        return ax

    return (plot_depth_section,)


@app.cell
def _(fu):
    def plot_timeseries(subfig, series, trop_warming):
        """
        Panels (d-g): fractional change over time of four indices, with tropical warming.

        Args:
            - subfig (matplotlib SubFigure): where to draw
            - series (xr.Dataset): "sigma_T", "dTdz_n3", "w_n3", "dTdx", dims (member, year)
            - trop_warming (xr.DataArray): tropical-mean SST change (K), dim year

        Returns:
            - axs (np.ndarray of Axes): the four panels
        """

        ## variables in plotting order, with their panel labels
        plot_vars = ["sigma_T", "dTdz_n3", "w_n3", "dTdx"]
        titles = [
            r"(d) $\sigma(T_{34})$",
            r"(e) $\overline{T}_{\text{ml}}-\overline{T}_{\text{sub}}$",
            r"(f) $\overline{w}$",
            r"(g) $\overline{T}_w-\overline{T}_e$",
        ]

        ## color of the warming reference line (and its axis)
        ref_color = "magenta"

        ## four stacked panels
        axs = subfig.subplots(4, 1)

        for i, (ax, v, title) in enumerate(zip(axs, plot_vars, titles)):

            ## baseline: ensemble median in the first window
            baseline = series[v].quantile(q=0.5, dim="member").isel(year=0)

            ## median (thick) and 5th/95th percentiles (thin) across members
            for q, lw, label in [
                (0.5, 2, "Median"),
                (0.05, 1, "5-95%"),
                (0.95, 1, None),
            ]:
                frac = fu.frac_change(
                    series[v].quantile(q=q, dim="member"), x0=baseline
                )
                ax.plot(series.year, frac, c="k", lw=lw, label=label)

            ## tropical warming on a twin axis
            axT = ax.twinx()
            axT.plot(
                trop_warming.year,
                trop_warming,
                c=ref_color,
                ls="--",
                lw=2,
                label="Trop. SST",
            )

            ## twin-axis range, ticks, and color
            axT.set_ylim([-1.5, 4.5])
            axT.set_yticks([0, 3], labels=[r"0$^{\circ}$C", r"3$^{\circ}$C"])
            axT.tick_params(axis="y", colors=ref_color)

            ## reference lines at zero change and at 2010
            line_kwargs = dict(c="gray", lw=0.5, ls="-")
            ax.axvline(2010, **line_kwargs)
            ax.axhline(0, **line_kwargs)

            ## panel label
            ax.text(s=title, x=-0.01, y=0.85, transform=ax.transAxes, zorder=10)

            ## (d, e) increase, (f, g) decrease: flip the y-axis for the decreasing ones
            if i < 2:
                ax.set_ylim([-0.15, 0.45])
                ax.set_yticks([0, 0.3], labels=["0%", r"+30%"])
            else:
                ax.set_ylim([0.15, -0.45])
                ax.set_yticks([0, -0.3], labels=[r"0%", r"$-$30%"])

            ## x ticks on the bottom panel only; no tick marks or grid
            if i == len(axs) - 1:
                ax.set_xticks([1870, 2010])
            else:
                ax.set_xticks([])
            for ax_ in [ax, axT]:
                ax_.tick_params(axis="y", which="both", length=0)
                ax_.grid(False)

            ## legend in panel (f)
            if i == 2:
                leg_kwargs = dict(prop=dict(size=9), frameon=False)
                ax.legend(loc=(0, 0.53), **leg_kwargs)
                axT.legend(loc=(0, 0.41), **leg_kwargs)
            else:
                pass

        return axs

    return (plot_timeseries,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(
    clim0_spatial,
    enso_pattern0,
    fu,
    plot_depth_section,
    plot_maps,
    plot_timeseries,
    plt,
    rsst0,
    series,
    trop_warming,
):
    ## figure split into a left column (a-c) and a narrower right column (d-g)
    fig = plt.figure(figsize=(9.4, 7.2), layout="constrained")
    subfig_left, subfig_right = fig.subfigures(1, 2, width_ratios=[2.6, 1])

    ## left column: maps on top (2/3 of the height), depth section below
    subfig_maps, subfig_depth = subfig_left.subfigures(2, 1, height_ratios=[2, 1])

    ## draw each part
    plot_maps(subfig_maps, rsst0, enso_pattern0)
    plot_depth_section(subfig_depth, clim0_spatial["T"], clim0_spatial["w"])
    plot_timeseries(subfig_right, series, trop_warming)

    ## save to SAVE_FP/figures
    fu.save_fig(fig, "cesm2-meanstate")

    fig
    return


if __name__ == "__main__":
    app.run()
