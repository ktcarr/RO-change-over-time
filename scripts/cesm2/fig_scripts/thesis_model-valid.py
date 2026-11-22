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
    # CESM2 model validation (thesis)

    Thesis figure `fig:model-valid` (`intro.tex`): CESM2 against ERSSTv5 over
    1958-2023.

    - **(a)** November-January (NDJ) mean relative SST in ERSST (SST minus its
      15°S-15°N mean)
    - **(b)** CESM2 minus ERSST relative SST (CESM2: ensemble mean, relative to its
      15°S-15°N mean)
    - **(c-f)** NDJ SST composites of strong El Niños (c, d) and La Niñas (e, f): NDJ
      Niño 3.4 above its 90th percentile or below its 10th. CESM2 (d, f) is scaled so
      its El Niño Niño 3.4 anomaly matches ERSST's (the scale factor, 0.76, is printed
      below). `MAP_INDEX` selects the index used for both
    - **(g, h)** El Niño plus La Niña composites (spatial asymmetry)
    - **(i, j)** NDJ Niño 3.4 against the zonal SST difference $T_e - T_w$ (gray
      boxes in a-h), for ERSST and CESM2 (scaled)
    - **(k)** PDFs of Niño 3 in 3-month means
    - **(l)** Power spectra of monthly Niño 3.4 (CESM2: mean over members, with their
      10th-90th percentile range shaded; dashed: 95% significance levels)
    - **(m, n)** Monthly Niño 3.4 during strong El Niños (red) and La Niñas (blue,
      sign reversed): December Niño 3.4 above its 90th percentile or below its 10th.
      (m) ERSST events (thin lines) and their mean (thick dashed); (n) CESM2 (scaled):
      the median (thick dashed) and 10th and 90th percentiles across events

    The white dashed boxes in (a-h) mark Niño 3.4.

    **Inputs:** ERSSTv5 (CVDP-OBS, see `src.data.load_ersst`); under `DATA_FP/cesm`, the
    SST EOFs (`consolidated/`), the 15°S-15°N mean SST (`trop_sst.nc`), and the Niño
    indices (`Th_anom.nc`).

    **Output:** `SAVE_FP/figures/thesis_model-valid.pdf`

    Before this script, the figure was three PDFs stacked in the thesis
    (`spatial-valid`, a-h; `spatial-asym-scatter`, i-k; `psd_spaghetti`, l-n), made by
    `explore/CESM-ORAS_validation_v2.py`. Now it is one figure. Changes:

    - (c-j, n): the thesis figure picked the (c-h) events, and scaled CESM2, with NDJ
      Niño 3 (its caption says Niño 3.4). The default here is Niño 3.4 for both
      (`MAP_INDEX = "nino34"`), as the caption says; `"nino3"` reproduces the thesis
      figure exactly. The patterns barely change (pattern correlations of at least
      0.98): one ERSST event of each sign swaps (El Niño: 1991 for 2009; La Niña:
      1970 for 1998). But the scale factor drops from 0.85 (the caption's value) to
      0.76, since CESM2's El Niño is relatively stronger in Niño 3.4 than in Niño 3,
      so every scaled CESM2 panel (d, f, h, j, n) is about 10% weaker than in the
      thesis.
    - The empty white box used to hide the band labels at the top of (l) is replaced
      by not drawing those labels.

    Other differences from the thesis caption:

    - (c-f): for CESM2, the percentiles are computed for each member.
    - (i, j): CESM2 is scaled by the same factor as (d, f); the caption doesn't say so.
    - (k): the 3-month means don't overlap (NDJ, FMA, MJJ, ASO); the caption says a
      running mean. CESM2 is not scaled here.
    - (m, n): events are picked with December monthly Niño 3.4, not the NDJ events
      of (c, e) as the caption says.
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
    ## paths
    import os
    import pathlib

    ## maps and plotting, plot style
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import seaborn as sns

    ## arrays, labeled arrays, regridding
    import numpy as np
    import xarray as xr
    import xesmf as xe

    ## this project's modules: loading data, indices
    import src.data
    import src.indices

    ## this project's modules: plotting, statistics
    import src.plot
    import src.stats

    ## older general utilities (to be split into the modules above), and the
    ## multitaper spectrum from the old RO code
    import src.utils
    import src.XRO_utils

    ## plotting style used for all figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})
    return ccrs, np, os, pathlib, plt, sns, src, xe, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Settings
    """
    )
    return


@app.cell
def _(np):
    ## period compared (ERSST and CESM2 overlap)
    PERIOD = dict(time=slice("1958-01", "2023-12"))

    ## NDJ: 3-month means labeled by their first month, November
    NDJ_START = 11

    ## (c-h) composites: NDJ index beyond this percentile (or 1 minus it)
    Q_MAP = 0.9

    ## index used to pick the (c-h) events and to scale CESM2: "nino34" (as the
    ## thesis caption says), or "nino3" (what the thesis figure used)
    MAP_INDEX = "nino34"

    ## (m, n) events: December Niño 3.4 beyond this percentile (or 1 minus it)
    Q_EVENT = 0.9
    PEAK_MONTH = 12

    ## (k) PDF bin edges (K)
    PDF_EDGES = np.arange(-4.25, 4.75, 0.5)

    ## (l) multitaper spectrum: monthly data (dt in years), time-bandwidth product
    PSD_KWARGS = dict(dim="time", dt=1 / 12, nw=5)

    ## (a-h) contour intervals (K), one per panel, in the layout of the maps
    MAP_DX = np.array([[1, 0.3], [0.6, 0.6], [0.6, 0.6], [0.3, 0.3]])
    return (
        MAP_DX,
        NDJ_START,
        PDF_EDGES,
        PEAK_MONTH,
        PERIOD,
        PSD_KWARGS,
        Q_EVENT,
        Q_MAP,
        MAP_INDEX,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load data
    """
    )
    return


@app.cell
def _(PERIOD, os, pathlib, src, xr):
    ## CESM2 SST EOFs: forced (ensemble mean) and anomalies, in the period
    _forced, _anom = src.utils.load_consolidated()
    forced = _forced[["sst", "sst_comp"]].sel(PERIOD)
    anom = _anom[["sst", "sst_comp"]].sel(PERIOD).load()

    ## CESM2 15S-15N mean SST (member, time)
    _fp = pathlib.Path(os.environ["DATA_FP"], "cesm", "trop_sst.nc")
    trop_sst = xr.open_dataset(_fp)["trop_sst_15"].sel(time=forced.time)

    ## CESM2 Niño 3 and 3.4 anomalies (member, time)
    nino_cesm = src.data.load_nino_anom("sst")[["T_3", "T_34"]].sel(PERIOD)

    ## ERSST anomaly and total SST (time, latitude, longitude)
    ersst_anom = src.data.load_ersst("ssta").sel(PERIOD)
    ersst_total = src.data.load_ersst("sst").sel(PERIOD)
    return anom, ersst_anom, ersst_total, forced, nino_cesm, trop_sst


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3-month means

    Non-overlapping 3-month means (NDJ, FMA, MJJ, ASO) of the SST anomalies, with
    ERSST regridded (bilinear) to the CESM2 grid.
    """
    )
    return


@app.cell
def _(NDJ_START, anom, ersst_anom, src, xe):
    ## CESM2 grid
    grid = anom["sst_comp"].isel(mode=0)

    ## CESM2: 3-month means of the EOF scores (member, time, mode)
    cesm_3mo = src.data.get_3month_means(anom["sst"])

    ## ERSST: 3-month means, regridded
    _ersst_3mo = src.data.get_3month_means(ersst_anom)
    ersst_3mo = xe.Regridder(_ersst_3mo, grid, "bilinear")(_ersst_3mo)

    ## NDJ only
    cesm_ndj = cesm_3mo.sel(time=cesm_3mo.time.dt.month == NDJ_START)
    ersst_ndj = ersst_3mo.sel(time=ersst_3mo.time.dt.month == NDJ_START)
    return cesm_3mo, cesm_ndj, ersst_3mo, ersst_ndj, grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (a, b) Relative SST

    NDJ mean SST minus the 15°S-15°N mean: for CESM2 the ensemble mean (forced)
    climatology, for ERSST the total SST (regridded).
    """
    )
    return


@app.cell
def _(ersst_total, forced, grid, src, trop_sst, xe):
    ## months averaged
    _ndj = [11, 12, 1]

    ## CESM2: forced SST climatology minus the 15S-15N mean, NDJ
    _clim = src.utils.reconstruct_clim(forced[["sst", "sst_comp"]])["sst"]
    _trop = trop_sst.groupby("time.month").mean(["time", "member"])
    rsst_cesm = _clim.sel(month=_ndj).mean("month") - _trop.sel(month=_ndj).mean(
        "month"
    )

    ## ERSST: total SST regridded, NDJ climatology
    _total = xe.Regridder(ersst_total, grid, "bilinear")(ersst_total)
    _clim_ref = _total.groupby("time.month").mean().sel(month=_ndj).mean("month")

    ## minus its 15S-15N mean
    _trop_ref = _clim_ref.sel(latitude=slice(-15, 15)).mean(["latitude", "longitude"])
    rsst_ersst = _clim_ref - _trop_ref
    return rsst_cesm, rsst_ersst


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (c-h) Composites

    NDJ SST averaged over the years when the NDJ index (`MAP_INDEX`) is beyond its 90th
    or 10th percentile (for CESM2, in each member, then averaged over members). CESM2 is
    then scaled so its El Niño index anomaly matches ERSST's.
    """
    )
    return


@app.cell
def _(MAP_INDEX, Q_MAP, anom, cesm_ndj, ersst_ndj, np, src):
    ## index function: Niño 3.4 or Niño 3 (box average)
    if MAP_INDEX == "nino34":
        _idx_fn = src.utils.get_nino34
    elif MAP_INDEX == "nino3":
        _idx_fn = src.utils.get_nino3
    else:
        raise ValueError(f"MAP_INDEX must be 'nino34' or 'nino3', not {MAP_INDEX!r}")

    ## NDJ index (CESM2 reconstructed from the EOFs)
    _idx_ersst = _idx_fn(ersst_ndj)
    _idx_cesm = src.utils.reconstruct_fn(anom["sst_comp"], cesm_ndj, _idx_fn)

    ## ERSST composites
    comps_ersst = src.stats.get_quantile_composites(ersst_ndj, _idx_ersst, q=Q_MAP)

    ## CESM2 composites of the scores, averaged over members, then reconstructed
    _comps = src.stats.get_quantile_composites(cesm_ndj, _idx_cesm, q=Q_MAP)
    _comps = _comps.mean("member")

    ## reconstruct each composite's SST field
    comps_cesm = _comps.copy()
    for _n in ["warm", "cold"]:
        comps_cesm[_n] = src.utils.reconstruct_fn(
            anom["sst_comp"], _comps[_n], lambda x: x
        )

    ## scale factor: ratio of the El Niño index anomalies (ERSST / CESM2)
    _idx_warm = lambda c: np.abs(_idx_fn(c["warm"]))  # noqa: E731
    scale = (_idx_warm(comps_ersst) / _idx_warm(comps_cesm)).item()
    print(f"CESM2 scale factor: {scale:.3f}")

    ## scaled CESM2 composites
    comps_cesm_scaled = scale * comps_cesm
    return comps_cesm_scaled, comps_ersst, scale


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (i-k) Zonal SST difference and PDFs
    """
    )
    return


@app.cell
def _(PDF_EDGES, anom, cesm_3mo, cesm_ndj, ersst_3mo, ersst_ndj, scale, src):
    ## ERSST: NDJ Niño 3.4 and T_e - T_w
    scatter_ersst = dict(
        x=src.utils.get_nino34(ersst_ndj), y=src.indices.get_dx(ersst_ndj)
    )

    ## CESM2 (scaled), reconstructed from the EOFs
    _recon = lambda fn: src.utils.reconstruct_fn(
        anom["sst_comp"], cesm_ndj, fn
    )  # noqa: E731
    scatter_cesm = dict(
        x=scale * _recon(src.utils.get_nino34),
        y=scale * (_recon(src.indices.get_e) - _recon(src.indices.get_w)),
    )

    ## PDFs of Niño 3 in all 3-month means (CESM2 unscaled, all members)
    _nino3_cesm = src.utils.reconstruct_fn(
        anom["sst_comp"], cesm_3mo, src.utils.get_nino3
    )
    pdf_cesm, _ = src.utils.get_empirical_pdf(_nino3_cesm, PDF_EDGES)
    pdf_ersst, _ = src.utils.get_empirical_pdf(
        src.utils.get_nino3(ersst_3mo), PDF_EDGES
    )
    return pdf_cesm, pdf_ersst, scatter_cesm, scatter_ersst


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (l-n) Monthly Niño 3.4: spectra and events

    Monthly indices, minus their mean over the period (and, for CESM2, over members).
    ERSST indices are computed on its own grid.
    """
    )
    return


@app.cell
def _(PSD_KWARGS, ersst_anom, nino_cesm, scale, src, xr):
    ## ERSST Niño 3.4 and 3
    _ersst = xr.Dataset(
        dict(
            T_34=src.utils.get_nino34(ersst_anom),
            T_3=src.utils.get_nino3(ersst_anom),
        )
    )

    ## CESM2, with a scaled Niño 3.4 for (n)
    _cesm = nino_cesm.copy()
    _cesm["T_34_scaled"] = scale * _cesm["T_34"]

    ## remove the mean over the period (and members)
    idx_ersst = _ersst - _ersst.mean()
    idx_cesm = _cesm - _cesm.mean()

    ## (l) power spectra of Niño 3.4 (CESM2 unscaled, one per member)
    psd_ersst = src.XRO_utils.pmtm(idx_ersst["T_34"], **PSD_KWARGS)
    psd_cesm = src.XRO_utils.pmtm(idx_cesm["T_34"], **PSD_KWARGS)
    return idx_cesm, idx_ersst, psd_cesm, psd_ersst


@app.cell
def _(PEAK_MONTH, Q_EVENT, idx_cesm, idx_ersst, src):
    def get_events(idx, is_warm):
        """
        Monthly indices from lag -12 to 12 around each strong event's December peak.

        Args:
            - idx (xr.Dataset): monthly indices, including "T_34" (used to pick events)
            - is_warm (bool): True for El Niño (above the Q_EVENT quantile), False for
              La Niña (below 1 - Q_EVENT)

        Returns:
            - events (xr.Dataset): dims (lag, sample), one sample per event
        """

        ## cutoff quantile and comparison
        if is_warm:
            q = Q_EVENT
            check_cutoff = lambda x, cut: x > cut  # noqa: E731
        else:
            q = 1 - Q_EVENT
            check_cutoff = lambda x, cut: x < cut  # noqa: E731

        ## every event, not averaged
        events = src.utils.make_composite(
            idx=idx["T_34"],
            data=idx,
            peak_month=PEAK_MONTH,
            q=q,
            check_cutoff=check_cutoff,
            avg=False,
        )

        return events

    ## El Niño and La Niña events in each dataset
    events_ersst = [get_events(idx_ersst, True), get_events(idx_ersst, False)]
    events_cesm = [get_events(idx_cesm, True), get_events(idx_cesm, False)]
    return events_cesm, events_ersst


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure parts
    """
    )
    return


@app.cell
def _(MAP_DX, ccrs, src):
    def plot_maps(subfig, rsst_ersst, rsst_cesm, comps_ersst, comps_cesm):
        """
        Panels (a-h): relative SST and ENSO composites, ERSST (left) and CESM2 (right).

        Args:
            - subfig (matplotlib SubFigure): where to draw
            - rsst_ersst (xr.DataArray): ERSST NDJ relative SST (K)
            - rsst_cesm (xr.DataArray): CESM2 NDJ relative SST (K)
            - comps_ersst (xr.Dataset): ERSST "warm" and "cold" composites (K)
            - comps_cesm (xr.Dataset): CESM2 composites (K), scaled

        Returns:
            - axs (np.ndarray of GeoAxes): the maps, shape (4, 2)
        """

        ## 4 x 2 maps of the tropical Pacific, centered on the dateline
        proj = ccrs.PlateCarree(central_longitude=190)
        axs = subfig.subplots(4, 2, subplot_kw=dict(projection=proj))
        for ax in axs.flatten():
            src.utils.plot_setup_pac(ax, max_lat=21, lon_range=(130, 285))

        ## field in each panel
        fields = [
            [rsst_ersst, rsst_cesm - rsst_ersst],
            [comps_ersst["warm"], comps_cesm["warm"]],
            [comps_ersst["cold"], comps_cesm["cold"]],
            [
                comps_ersst["warm"] + comps_ersst["cold"],
                comps_cesm["warm"] + comps_cesm["cold"],
            ],
        ]

        ## filled contours, each with its own interval
        for i in range(4):
            for j in range(2):
                src.plot.plot_sst_map(axs[i, j], fields[i][j], dx=MAP_DX[i, j])

        ## outline the T_w and T_e boxes (gray) and the Niño 3.4 box (dashed white)
        for ax in axs.flatten():
            src.plot.plot_wbox(ax, c="gray", lw=0.8)
            src.plot.plot_ebox(ax, c="gray", lw=0.8)
            src.utils.plot_nino34_box(ax, c="w", ls="--", lw=0.8, alpha=0.75)

        ## titles
        titles = [
            "(a) Relative SST mean (ERSST)",
            r"(b) Relative SST mean bias $(\text{CESM2} - \text{ERSST})$",
            "(c) El Niño (ERSST)",
            "(d) El Niño (CESM2, scaled)",
            "(e) La Niña (ERSST)",
            "(f) La Niña (CESM2, scaled)",
            "(g) El Niño + La Niña (ERSST)",
            "(h) El Niño + La Niña (CESM2, scaled)",
        ]
        for ax, title in zip(axs.flatten(), titles):
            ax.set_title(title, loc="left")

        ## latitude labels on the left, longitude labels on the bottom row
        src.plot.add_lat_labels(axs)
        src.plot.add_lon_labels(axs[-1])

        ## contour interval in the lower left of each panel
        bbox = dict(boxstyle="round", facecolor="white", alpha=1)
        text_kwargs = dict(fontdict=dict(size=12, color="gray"), bbox=bbox)
        for ax, dx in zip(axs.flatten(), MAP_DX.flatten()):
            ax.text(
                0.01, 0.02, f"Interval: {dx} K", transform=ax.transAxes, **text_kwargs
            )

        return axs

    return (plot_maps,)


@app.cell
def _(PDF_EDGES, np, src):
    def plot_scatter_pdfs(subfig, scatter_ersst, scatter_cesm, pdf_ersst, pdf_cesm):
        """
        Panels (i-k): Niño 3.4 against T_e - T_w, and PDFs of Niño 3.

        Args:
            - subfig (matplotlib SubFigure): where to draw
            - scatter_ersst (dict): ERSST "x" (Niño 3.4) and "y" (T_e - T_w), K
            - scatter_cesm (dict): same for CESM2 (scaled)
            - pdf_ersst (np.ndarray): ERSST PDF on PDF_EDGES
            - pdf_cesm (np.ndarray): CESM2 PDF on PDF_EDGES

        Returns:
            - axs (np.ndarray of Axes): the three panels
        """

        ## three panels in a row
        axs = subfig.subplots(1, 3)

        ## scatter plots: few large ERSST points; many small CESM2 points
        axs[0].scatter(scatter_ersst["x"], scatter_ersst["y"], s=25, c="gray")
        axs[1].scatter(scatter_cesm["x"], scatter_cesm["y"], alpha=0.3, s=1, c="gray")

        ## zero lines and a one-to-one line
        for ax in axs[:2]:
            ax.axhline(0, ls="--", c="k", lw=0.5)
            ax.axvline(0, ls="--", c="k", lw=0.5)

            ## one-to-one line
            zz = np.linspace(-1, 3.5)
            ax.plot(zz, zz, c="k", lw=0.8)

            ## x axis
            ax.set_xticks([-3, 0, 3])
            ax.set_xlabel("Niño 3.4 (K)")

        ## (k) PDFs: ERSST shaded, CESM2 as an outline
        kw = dict(fill=True, color="C0", alpha=0.3, label="ERSST")
        axs[2].stairs(pdf_ersst, PDF_EDGES, **kw)
        axs[2].stairs(pdf_cesm, PDF_EDGES, color="C1", lw=2, label="CESM2")

        ## (k) zero line and legend
        axs[2].axvline(0, c="k", lw=0.8, ls="--")
        axs[2].legend()

        ## (i, j) same axis ranges; y axis on (i) only
        src.utils.set_lims(axs[:2])
        axs[0].set_yticks([-1.5, 0, 1.5, 3])
        axs[0].set_ylabel(r"$T_e-T_w$ (K)")

        ## no y ticks on (j)
        axs[1].set_yticks([])

        ## (k) x axis
        axs[2].set_xticks([-3, 0, 3])
        axs[2].set_xlim([-3.75, 3.75])
        axs[2].set_xlabel("Niño 3 (K)")

        ## (k) y axis, on the right
        axs[2].set_yticks([0, 0.3, 0.6])
        axs[2].set_ylabel("Prob. density")

        ## move the label and ticks to the right
        axs[2].yaxis.set_label_position("right")
        axs[2].yaxis.tick_right()

        ## titles
        axs[0].set_title("(i) ERSST", loc="left")
        axs[1].set_title("(j) CESM2", loc="left")
        axs[2].set_title("(k) PDFs", loc="left")

        return axs

    return (plot_scatter_pdfs,)


@app.cell
def _(src):
    def plot_events(ax, events, varname, thin="all"):
        """
        Niño 3.4 during El Niño (red) and La Niña (blue, sign reversed) events.

        Args:
            - ax (matplotlib Axes): axis to draw on
            - events (list of xr.Dataset): El Niño and La Niña events, dims (lag,
              sample)
            - varname (str): index to plot
            - thin (str): "all" draws every event (thin) and the mean (thick dashed);
              "quantiles" draws the 10th and 90th percentiles and the median

        Returns:
            - None
        """

        ## El Niño as is, La Niña with the sign reversed
        for ev, sign, c in zip(events, [1, -1], ["r", "b"]):
            x = sign * ev[varname]

            ## every event and the mean ...
            if thin == "all":
                center = x.mean("sample")
                ax.plot(ev.lag, x, c=c, lw=1.5, alpha=0.7)

            ## ... or the 10th and 90th percentiles and the median
            else:
                center = x.quantile(q=0.5, dim="sample")
                for q in [0.1, 0.9]:
                    ax.plot(ev.lag, x.quantile(q=q, dim="sample"), c=c, lw=2)

            ## thick dashed center line
            ax.plot(ev.lag, center, c=c, lw=5, ls="--")

        return

    def plot_psd_events(subfig, psd_ersst, psd_cesm, events_ersst, events_cesm):
        """
        Panels (l-n): Niño 3.4 power spectra and event evolution.

        Args:
            - subfig (matplotlib SubFigure): where to draw
            - psd_ersst (tuple): ERSST spectrum from src.XRO_utils.pmtm
            - psd_cesm (tuple): CESM2 spectra (one per member)
            - events_ersst (list of xr.Dataset): ERSST El Niño and La Niña events
            - events_cesm (list of xr.Dataset): CESM2 events

        Returns:
            - axs (np.ndarray of Axes): the three panels
        """

        ## three panels in a row
        axs = subfig.subplots(1, 3)

        ## (m) ERSST events; (n) CESM2 (scaled) percentiles
        plot_events(axs[1], events_ersst, "T_34", thin="all")
        plot_events(axs[2], events_cesm, "T_34_scaled", thin="quantiles")

        ## (m, n) lags from -2, zero lines
        for ax in axs[1:]:
            ax.set_xlim([-2, None])
            ax.axhline(0, c="gray", lw=0.8)
            ax.axvline(0, c="gray", lw=0.8)

            ## x axis
            ax.set_xticks([0, 6, 12])
            ax.set_xlabel("Lag")

        ## (m, n) same ranges; y axis on the right of (n)
        src.utils.set_lims(axs[1:])
        axs[1].set_yticks([])

        ## y axis of (n), on the right
        axs[2].set_yticks([-2.5, 0, 2.5])
        axs[2].set_ylabel("Niño 3.4 (K)")

        ## move the label and ticks to the right
        axs[2].yaxis.set_label_position("right")
        axs[2].yaxis.tick_right()

        ## (l) spectra (without the band labels, which overlap the title)
        src.utils.plot_psd(axs[0], psd_ersst, label="ERSST", label_bands=False)
        src.utils.plot_psd(axs[0], psd_cesm, label="CESM2", label_bands=False)

        ## (l) axes
        axs[0].set_ylabel(r"PSD ($^{\circ}$C$^2$/cpm)")
        axs[0].set_xlim([None, 3])
        axs[0].legend()

        ## titles
        axs[0].set_title("(l) Power spec. (Niño 3.4)", loc="left")
        axs[1].set_title("(m) ERSST", loc="left")
        axs[2].set_title("(n) CESM2 (scaled)", loc="left")

        return axs

    return (plot_psd_events,)


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
    comps_cesm_scaled,
    comps_ersst,
    events_cesm,
    events_ersst,
    pdf_cesm,
    pdf_ersst,
    plot_maps,
    plot_psd_events,
    plot_scatter_pdfs,
    plt,
    psd_cesm,
    psd_ersst,
    rsst_cesm,
    rsst_ersst,
    scatter_cesm,
    scatter_ersst,
    src,
):
    ## maps (a-h) on top, then two rows of line plots (i-k, l-n)
    fig = plt.figure(figsize=(10, 12.5), layout="constrained")
    subfigs = fig.subfigures(3, 1, height_ratios=[7, 2.5, 3])

    ## draw each part
    plot_maps(subfigs[0], rsst_ersst, rsst_cesm, comps_ersst, comps_cesm_scaled)
    plot_scatter_pdfs(subfigs[1], scatter_ersst, scatter_cesm, pdf_ersst, pdf_cesm)
    plot_psd_events(subfigs[2], psd_ersst, psd_cesm, events_ersst, events_cesm)

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "thesis_model-valid")

    fig
    return


if __name__ == "__main__":
    app.run()
