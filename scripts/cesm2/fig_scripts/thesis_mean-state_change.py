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
    # Mean-state changes in CESM2 (thesis)

    Thesis figure `fig:mean-state_change` (`chapter_enso-change.tex`). Changes in the
    ensemble mean (forced response) between an early period (1850-1889) and a late
    period (2065-2100):

    - **(a)** relative SST (SST minus the 15°S-15°N mean): change (colors) and early
      climatology (black contours every 2°C; gray is the zero contour). Magenta boxes are
      Niño 4 (left) and Niño 3 (right); the dashed white box is Niño 3.4.
    - **(b)** precipitation: change (colors) and the early 6 mm/day contour
    - **(c)** equatorial temperature (1.5°S-1.5°N): change (colors) and early climatology
      (black contours every 4°C); the dashed magenta line is the early 20°C isotherm
    - **(d)** same as (c), but for vertical velocity (black contours every 10 m/month)

    And over time, in 40-year windows every 5 years (labeled by their center; the last
    window, 2085, is cut short by the end of the simulation, so it covers 2065-2100;
    the periods above are the first and last windows):

    - **(e)** fractional change from the first window of: stratification
      $\Delta_z T$ (mean temperature over the top 50 m minus 50-70 m, over the Niño 3
      longitudes, divided by 50 m); the zonal SST difference $\Delta_x T = T_w - T_e$
      (west Pacific 120°E-160°E minus Niño 3, 5°S-5°N); the mean thermocline depth $h$
      (depth of the maximum vertical temperature gradient, averaged over 140°E-90°W);
      and vertical velocity $w$ at 70 m, averaged over the Niño 3 longitudes
    - **(f)** the tropical-mean SST, $T_w$, and $T_e$

    **Inputs:** under `DATA_FP/cesm`: `consolidated/forced.nc` (SST, precipitation,
    temperature, vertical velocity), `trop_sst.nc` (tropical-mean SST); and
    `SAVE_FP/h_ests/h_max-grad_05.nc` (thermocline depth, from
    `preprocess/compute_h.py`).

    **Output:** `SAVE_FP/figures/thesis_mean-state_change.pdf`

    Before this script, the thesis figure was put together in LaTeX from four PDFs made
    by `explore/mean-state-changes.py` (cells under "Figs"): `mean-state-pattern_change_labeled`
    (a, b), `subsurface-change` (c, d), `mean-state_timeseries` (e), and
    `nino-indices_overtime` (f). Changes:

    - All six panels are in one figure.
    - $T_e$ comes from the EOF-compressed SST, like $T_w$ (before, from the Niño 3 index
      in `Th.nc`).
    - Precipitation is converted to mm/day with 86,400 s/day (it was 86,000).
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
    ## standard library: paths
    import os
    import pathlib

    ## plotting, maps, and colormaps
    import cartopy.crs as ccrs
    import cmocean  # noqa: F401 (registers the "cmo.*" colormaps)
    import matplotlib.pyplot as plt

    ## numerics, plot style, labeled arrays
    import numpy as np
    import seaborn as sns
    import xarray as xr

    ## this project's modules: plotting, statistics
    import src.plot
    import src.stats

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all figures; colorblind palette for the line plots
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")

    ## folder with the preprocessed data
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    return DATA_FP, ccrs, np, plt, src, xr


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
    ## early and late periods for the maps and sections (a-d): the first and last
    ## 40-year windows of (e, f)
    PERIOD_EARLY = slice("1850", "1889")
    PERIOD_LATE = slice("2065", "2100")

    ## window length and stride (months) for the changes over time (e, f)
    WINDOW_SIZE = 480
    STRIDE = 60

    ## mixed-layer depth and thickness of the layer below it (m), for stratification
    MLD = 50
    DELTA = 20

    ## depth (m) of the vertical velocity in (e)
    Z_W = 70

    ## regions (degrees east): west Pacific box for T_w, Niño 3 longitudes, and the
    ## longitudes for the mean thermocline depth
    LONS_TW = [120, 160]
    LONS_N3 = [210, 270]
    LONS_H = [140, 270]

    ## years marked on the time axis, and the year marked with a line
    XTICKS = [1870, 2010, 2085]
    XLINE = 2010
    return (
        DELTA,
        LONS_H,
        LONS_N3,
        LONS_TW,
        MLD,
        PERIOD_EARLY,
        PERIOD_LATE,
        STRIDE,
        WINDOW_SIZE,
        XLINE,
        XTICKS,
        Z_W,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data

    All fields are the ensemble mean. The consolidated fields are EOF-compressed
    (scores and patterns); averaging the scores over time and then reconstructing gives
    the time mean of the fields.
    """
    )
    return


@app.cell
def _(DATA_FP, src, xr):
    ## ensemble-mean scores and patterns of SST, precipitation, temperature, and w
    _forced, _ = src.utils.load_consolidated()
    _names = ["sst", "pr", "T", "w"]
    patterns, scores = src.utils.split_components(
        _forced[_names + [f"{v}_comp" for v in _names]]
    )

    ## tropical-mean SST (15S-15N), ensemble mean; dims (time,)
    trop_sst = xr.open_dataset(DATA_FP / "cesm" / "trop_sst.nc")["trop_sst_15"]
    trop_sst = trop_sst.mean("member")

    ## thermocline depth (depth of max. dT/dz) on the equator, ensemble mean; dims
    ## (longitude, time)
    h, _ = src.utils.load_h_data(max_grad=True)
    return h, patterns, scores, trop_sst


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (a-d) Change between the early and late periods

    Relative SST is SST minus the tropical mean over the same period.
    """
    )
    return


@app.cell
def _(PERIOD_EARLY, PERIOD_LATE, patterns, scores, src, trop_sst):
    ## time-mean fields in each period
    clim = dict()
    for _name, _period in [("early", PERIOD_EARLY), ("late", PERIOD_LATE)]:
        _clim = src.utils.reconstruct_fn(
            components=patterns,
            scores=scores.sel(time=_period).mean("time"),
            fn=lambda x: x,
        )

        ## relative SST
        _clim["sst_rel"] = _clim["sst"] - trop_sst.sel(time=_period).mean()

        ## precipitation in mm/day (from m/s)
        _clim["pr"] = _clim["pr"] * 8.64e4
        clim[_name] = _clim

    ## change from the early to the late period
    delta = clim["late"] - clim["early"]
    return clim, delta


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (e, f) Indices over time

    Each index is averaged over each 40-year window (all months).
    """
    )
    return


@app.cell
def _(DELTA, LONS_N3, LONS_TW, MLD, Z_W, src):
    def get_indices(x):
        """
        Mean-state indices from reconstructed fields.

        Args:
            - x (xr.Dataset): "sst" (latitude, longitude), "T" and "w" (z_t,
              longitude; equatorial sections)

        Returns:
            - idx (xr.Dataset): "T_w", "T_e" (°C), "dTdz" (K/m), "w" (m/month)
        """

        ## SST over the west Pacific box and Niño 3 (5S-5N)
        _eq = x["sst"].sel(latitude=slice(-5, 5))
        T_w = _eq.sel(longitude=slice(*LONS_TW)).mean(["latitude", "longitude"])
        T_e = _eq.sel(longitude=slice(*LONS_N3)).mean(["latitude", "longitude"])

        ## Niño 3 averages of the equatorial sections
        T_n3 = x["T"].sel(longitude=slice(*LONS_N3)).mean("longitude")
        w_n3 = x["w"].sel(longitude=slice(*LONS_N3)).mean("longitude")

        ## stratification: (mixed-layer mean minus mean of the layer below) / MLD
        dTdz = -src.utils.get_dT_sub(Tsub=T_n3, Hm=MLD, delta=DELTA) / MLD

        ## vertical velocity at (the level nearest) Z_W
        w = w_n3.sel(z_t=Z_W, method="nearest", drop=True)

        ## all indices in one dataset: SST indices
        idx = T_w.to_dataset(name="T_w")
        idx["T_e"] = T_e

        ## subsurface indices
        idx["dTdz"] = dTdz
        idx["w"] = w

        return idx

    return (get_indices,)


@app.cell
def _(
    LONS_H,
    STRIDE,
    WINDOW_SIZE,
    get_indices,
    h,
    patterns,
    scores,
    src,
    trop_sst,
):
    ## window means of the scores; dims (year, mode)
    _kwargs = dict(window_size=WINDOW_SIZE, stride=STRIDE)
    _scores_windowed = src.utils.get_windowed(scores, **_kwargs).mean("time")

    ## reconstruct the fields, then the indices; dims (year,)
    indices = src.utils.reconstruct_fn(
        components=patterns[["sst", "T", "w"]],
        scores=_scores_windowed[["sst", "T", "w"]],
        fn=lambda x: x,
    )
    indices = get_indices(indices)

    ## zonal SST difference, west minus east
    indices["dTdx"] = indices["T_w"] - indices["T_e"]

    ## tropical-mean SST, window means
    indices["trop_sst"] = src.utils.get_windowed(trop_sst, **_kwargs).mean("time")

    ## mean thermocline depth, window means
    _h = h.sel(longitude=slice(*LONS_H)).mean("longitude")
    indices["h"] = src.utils.get_windowed(_h, **_kwargs).mean("time")

    ## fractional change from the first window
    frac = src.stats.frac_change(indices[["dTdz", "dTdx", "h", "w"]])
    return frac, indices


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting functions
    """
    )
    return


@app.cell
def _(ccrs, src):
    def plot_maps(subfig, delta, clim0):
        """
        Panels (a) relative SST and (b) precipitation: change (colors) and early
        climatology (contours), on maps of the tropical Pacific.

        Args:
            - subfig (matplotlib SubFigure): where to draw
            - delta (xr.Dataset): changes in "sst_rel" (°C) and "pr" (mm/day)
            - clim0 (xr.Dataset): early climatology of the same variables

        Returns:
            - axs (np.ndarray of GeoAxes): the two maps, shape (2, 1)
        """

        ## two maps of the tropical Pacific, centered on the dateline
        _format = lambda ax: src.utils.plot_setup_pac(
            ax, max_lat=30, lon_range=(115, 285)
        )
        axs = src.utils.subplots_with_proj(
            subfig, nrows=2, ncols=1, format_func=_format
        )

        ## shared arguments: grid and map projection of the data
        lonlat = (delta.longitude, delta.latitude)
        kwargs = dict(transform=ccrs.PlateCarree())

        ## (a) relative SST: change, early climatology every 2 degC, and its zero line
        cp_sst = axs[0, 0].contourf(
            *lonlat,
            delta["sst_rel"],
            levels=src.utils.make_cb_range(2, 0.2),
            cmap="cmo.balance",
            extend="both",
            **kwargs,
        )
        axs[0, 0].contour(
            *lonlat,
            clim0["sst_rel"],
            levels=src.utils.make_cb_range(10, 2),
            colors="k",
            linewidths=0.75,
            **kwargs,
        )
        axs[0, 0].contour(
            *lonlat,
            clim0["sst_rel"],
            levels=[0],
            colors="gray",
            linewidths=1.25,
            **kwargs,
        )

        ## (b) precipitation: change (wetter is blue) and the early 6 mm/day contour
        cp_pr = axs[1, 0].contourf(
            *lonlat,
            delta["pr"],
            levels=src.utils.make_cb_range(5, 0.5),
            cmap="cmo.balance_r",
            extend="both",
            **kwargs,
        )
        axs[1, 0].contour(
            *lonlat, clim0["pr"], levels=[6], colors="k", linewidths=0.75, **kwargs
        )

        ## colorbars
        subfig.colorbar(
            cp_sst, ax=axs[0, 0], label=r"$^{\circ}$C", ticks=[-2, 0, 2], pad=0.02
        )
        subfig.colorbar(
            cp_pr, ax=axs[1, 0], label="mm / day", ticks=[-5, 0, 5], pad=0.02
        )

        ## Niño 4 and Niño 3 boxes (magenta), Niño 3.4 box (dashed white)
        gls = []
        for ax in axs.flatten():
            src.utils.plot_nino4_box(ax, c="magenta", lw=0.8)
            src.utils.plot_nino3_box(ax, c="magenta", lw=0.8)
            src.utils.plot_nino34_box(ax, c="w", lw=0.8, ls="--")

            ## invisible gridlines, used only for their labels
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=1,
                alpha=0,
                xlocs=[120, 160, -150, -90],
                ylocs=[-25, 0, 25],
            )
            gls.append(gl)

        ## longitude labels on the top of (a) and the bottom of (b)
        gls[0].bottom_labels = False
        gls[1].top_labels = False

        ## panel labels
        src.utils.label_subplot(ax=axs[0, 0], posn=(0.02, 0.05), label="(a)")
        src.utils.label_subplot(ax=axs[1, 0], posn=(0.02, 0.05), label="(b)")

        return axs

    return (plot_maps,)


@app.cell
def _(np, src):
    def plot_sections(subfig, delta, clim0):
        """
        Panels (c) temperature and (d) vertical velocity on the equator: change
        (colors), early climatology (black contours), and the early 20°C isotherm
        (dashed magenta).

        Args:
            - subfig (matplotlib SubFigure): where to draw
            - delta (xr.Dataset): changes in "T" (°C) and "w" (m/month), dims (z_t,
              longitude)
            - clim0 (xr.Dataset): early climatology of the same variables

        Returns:
            - axs (np.ndarray of Axes): the two sections
        """

        ## two sections side by side
        axs = subfig.subplots(1, 2)

        ## (variable, color scale limit, early-climatology contour levels, units)
        panels = [
            ("T", 5, np.arange(12, 34, 4), r"$^{\circ}$C"),
            ("w", 20, np.arange(0, 100, 10), "m / month"),
        ]

        for ax, (varname, amp, levels, units) in zip(axs, panels):

            ## change
            cp = ax.contourf(
                delta.longitude,
                delta.z_t,
                delta[varname],
                cmap="cmo.balance",
                levels=src.utils.make_cb_range(amp, amp / 10),
                extend="both",
            )

            ## early climatology
            ax.contour(
                clim0.longitude,
                clim0.z_t,
                clim0[varname],
                colors="k",
                levels=levels,
                linewidths=0.5,
            )

            ## early 20 degC isotherm
            ax.contour(
                clim0.longitude,
                clim0.z_t,
                clim0["T"],
                levels=[20],
                colors="magenta",
                linestyles="--",
            )

            ## colorbar below each panel
            subfig.colorbar(
                cp, ax=ax, ticks=[-amp, 0, amp], label=units, orientation="horizontal"
            )

        ## depth axis pointing down
        src.utils.format_subsurf_axs(axs)

        ## longitude range and ticks
        for ax in axs:
            ax.set_xlim([140, 280])
            ax.set_xticks([140, 210, 280])

            ## longitude labels on top
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()

        ## panel labels
        src.utils.label_subplot(ax=axs[0], posn=(0.03, 0.05), label="(c)")
        src.utils.label_subplot(ax=axs[1], posn=(0.03, 0.05), label="(d)")

        return axs

    return (plot_sections,)


@app.cell
def _(XLINE, XTICKS):
    def plot_timeseries(subfig, frac, indices):
        """
        Panels (e) fractional changes of the mean-state indices and (f) SST indices,
        over time.

        Args:
            - subfig (matplotlib SubFigure): where to draw
            - frac (xr.Dataset): fractional changes of "dTdz", "dTdx", "h", "w",
              dims (year,)
            - indices (xr.Dataset): "trop_sst", "T_w", "T_e" (°C), dims (year,)

        Returns:
            - axs (np.ndarray of Axes): the two panels
        """

        ## two panels side by side
        axs = subfig.subplots(1, 2)

        ## guideline style
        ref_kwargs = dict(ls="--", c="gray", lw=0.8)

        ## (e) fractional changes
        labels = dict(dTdz=r"$\Delta_z T$", dTdx=r"$\Delta_x T$", h="$h$", w="$w$")
        for varname, label in labels.items():
            axs[0].plot(frac.year, frac[varname], label=label)

        ## (e) zero line
        axs[0].axhline(0, **ref_kwargs)

        ## (e) labels
        axs[0].set_yticks([-0.3, 0, 0.6])
        axs[0].set_ylabel("Fractional change")
        axs[0].set_title("(e) Fractional changes", loc="left")

        ## (f) SST indices
        axs[1].plot(indices.year, indices["trop_sst"], label="Tropical avg.", c="k")
        axs[1].plot(indices.year, indices["T_w"], label=r"$T_w$", c="r")
        axs[1].plot(indices.year, indices["T_e"], label=r"$T_e$", c="b")

        ## (f) labels
        axs[1].set_yticks([26, 29, 32])
        axs[1].set_ylabel(r"$^{\circ}$C")

        ## (f) y-axis on the right
        axs[1].yaxis.set_label_position("right")
        axs[1].yaxis.tick_right()
        axs[1].set_title("(f) Absolute changes", loc="left")

        ## time axis and legends
        for ax in axs:
            ax.set_xticks(XTICKS)
            ax.axvline(XLINE, **ref_kwargs)
            ax.legend()

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
    clim,
    delta,
    frac,
    indices,
    plot_maps,
    plot_sections,
    plot_timeseries,
    plt,
    src,
):
    ## three rows: maps (a, b), sections (c, d), and time series (e, f)
    fig = plt.figure(figsize=(8, 11.5), layout="constrained")
    _subfigs = fig.subfigures(3, 1, height_ratios=[4.8, 4, 3])

    ## draw each row
    plot_maps(_subfigs[0], delta, clim["early"])
    plot_sections(_subfigs[1], delta, clim["early"])
    plot_timeseries(_subfigs[2], frac, indices)

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "thesis_mean-state_change")

    fig
    return


if __name__ == "__main__":
    app.run()
