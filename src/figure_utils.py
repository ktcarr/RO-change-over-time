"""
Shared helpers for the figure scripts in scripts/cesm2/fig_scripts.

Each figure script used to carry its own copy of these functions (e.g., eleven
copies of a `save` function). They live here so that every figure saves to the
same place and uses the same region definitions. Add a helper here when a
second figure script needs it; keep figure-specific code in the script itself.
"""

import os
import pathlib

import cartopy.crs as ccrs
import numpy as np
import xarray as xr

import src.utils

## longitude ranges (degrees east) for the western and eastern equatorial
## boxes used in the paper: T_w is averaged over LONS_W, T_e over LONS_E
LONS_W = [150, 190]
LONS_E = [240, 280]

## latitude range (degrees north) for "equatorial" averages
LATS_EQ = [-5, 5]


def save_fig(fig, name, subdir="figures", dpi=800):
    """
    Save a figure as a PDF under SAVE_FP/<subdir>.

    Args:
        - fig (matplotlib.figure.Figure): figure to save
        - name (str): file name without extension (use the figure's LaTeX label)
        - subdir (str): folder under SAVE_FP to save into
        - dpi (int): resolution for rasterized elements (e.g., filled contours)

    Returns:
        - fp (pathlib.Path): path of the saved file
    """

    ## build the output folder from the SAVE_FP environment variable, and create
    ## it if this is the first figure saved there
    save_dir = pathlib.Path(os.environ["SAVE_FP"], subdir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ## save as PDF (the format used in the Overleaf project)
    fp = save_dir / f"{name}.pdf"
    fig.savefig(fp, dpi=dpi, format="pdf")

    return fp


def get_eq_avg(data, lons):
    """
    Average over the equatorial band (LATS_EQ) and a range of longitudes.

    Args:
        - data (xr.DataArray or xr.Dataset): data with "longitude" and "latitude"
        - lons (list of float): [west, east] longitude bounds (degrees east)

    Returns:
        - avg (xr.DataArray or xr.Dataset): data averaged over the box
    """

    ## select the box, then average over both horizontal dimensions
    box = data.sel(longitude=slice(*lons), latitude=slice(*LATS_EQ))
    avg = box.mean(["longitude", "latitude"])

    return avg


def get_w(data):
    """
    Average over the western equatorial box (LONS_W).

    Args:
        - data (xr.DataArray or xr.Dataset): data with "longitude" and "latitude"

    Returns:
        - avg (xr.DataArray or xr.Dataset): western-box average
    """
    return get_eq_avg(data, LONS_W)


def get_e(data):
    """
    Average over the eastern equatorial box (LONS_E).

    Args:
        - data (xr.DataArray or xr.Dataset): data with "longitude" and "latitude"

    Returns:
        - avg (xr.DataArray or xr.Dataset): eastern-box average
    """
    return get_eq_avg(data, LONS_E)


def get_dx(data):
    """
    East-minus-west difference of equatorial box averages.

    Args:
        - data (xr.DataArray or xr.Dataset): data with "longitude" and "latitude"

    Returns:
        - dx (xr.DataArray or xr.Dataset): eastern-box minus western-box average
    """
    return get_e(data) - get_w(data)


def plot_wbox(ax, **kwargs):
    """
    Outline the western equatorial box on a map.

    Args:
        - ax (cartopy GeoAxes): map to draw on
        - kwargs: passed to ax.plot (e.g., c, lw, ls)

    Returns:
        - None
    """
    src.utils.plot_box(ax, lons=LONS_W, lats=LATS_EQ, **kwargs)
    return


def plot_ebox(ax, **kwargs):
    """
    Outline the eastern equatorial box on a map.

    Args:
        - ax (cartopy GeoAxes): map to draw on
        - kwargs: passed to ax.plot (e.g., c, lw, ls)

    Returns:
        - None
    """
    src.utils.plot_box(ax, lons=LONS_E, lats=LATS_EQ, **kwargs)
    return


def make_centered_levels(dx, nlevs):
    """
    Contour levels symmetric about zero, with a band centered on zero.

    For example, dx=1 and nlevs=2 gives [-1.5, -0.5, 0.5, 1.5]: the middle band
    (-0.5 to 0.5) is centered on zero, so near-zero values get the neutral color.

    Args:
        - dx (float): spacing between levels
        - nlevs (int): number of levels on each side of zero

    Returns:
        - levels (np.ndarray): 2 * nlevs levels, increasing
    """

    ## positive levels: dx/2, 3dx/2, ..., up to (nlevs - 1/2) * dx
    amp = dx * nlevs - dx / 2
    lev_pos = np.arange(0, amp + dx / 2, dx) + dx / 2

    ## negative levels mirror the positive ones
    lev_neg = -lev_pos[::-1]
    levels = np.concatenate([lev_neg, lev_pos])

    return levels


def plot_sst_map(ax, data, dx=0.2, nlev=5):
    """
    Filled contours of a lat/lon field on a map, with a diverging colormap.

    Args:
        - ax (cartopy GeoAxes): map to draw on
        - data (xr.DataArray): field with "longitude" and "latitude"
        - dx (float): contour interval
        - nlev (int): number of contour levels on each side of zero

    Returns:
        - cp (QuadContourSet): contour set (pass to fig.colorbar)
    """

    ## filled contours; data are on a regular lat/lon grid (PlateCarree)
    cp = ax.contourf(
        data.longitude,
        data.latitude,
        data,
        cmap="cmo.balance",
        levels=make_centered_levels(dx, nlev),
        transform=ccrs.PlateCarree(),
        extend="both",
    )

    return cp


def add_lat_labels(axs, ylocs=(-20, 0, 20)):
    """
    Label latitudes on the left of a column of maps.

    Args:
        - axs (np.ndarray of GeoAxes): maps, shape (nrows, ncols); only the first
          column is labeled
        - ylocs (tuple of float): latitudes to label

    Returns:
        - None
    """

    ## invisible gridlines (alpha=0) used only for their labels; no longitude labels
    for ax in axs[:, 0]:
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1,
            alpha=0,
            xlocs=[],
            ylocs=list(ylocs),
        )

        ## labels on the left only
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False

    return


def frac_change(x, x0=None):
    """
    Fractional change relative to a baseline, (x - x0) / x0.

    Args:
        - x (xr.DataArray): data with a "year" dimension
        - x0 (xr.DataArray or float): baseline; if None, use the first year of x

    Returns:
        - frac (xr.DataArray): fractional change (0.1 means +10%)
    """

    ## default baseline is the first window
    if x0 is None:
        x0 = x.isel(year=0)
    else:
        pass

    ## compute the fractional change
    frac = (x - x0) / x0

    return frac


def load_total(varnames):
    """
    Load CESM2 consolidated (EOF-compressed) fields as forced + anomaly.

    Fields that also exist in the "wide" consolidated data (consolidated_05,
    which covers a wider latitude band) are taken from there.

    Args:
        - varnames (list of str): variables to load (EOF scores, e.g. "sst", "T")

    Returns:
        - total (xr.Dataset): forced + anomaly scores for each variable, plus the
          EOF components ("<var>_comp") needed to reconstruct spatial fields
    """

    ## load the standard consolidated data, then the wide version
    forced, anom = src.utils.load_consolidated()
    forced_wide, anom_wide = src.utils.load_consolidated_05()

    ## replace each variable with its wide version where one exists
    for v in list(forced_wide):
        forced[v] = forced_wide[v]
        anom[v] = anom_wide[v]

    ## total = forced + anomaly (scores share the same EOF components)
    total = anom[varnames] + forced[varnames]

    ## add the EOF components so the fields can be reconstructed
    comp_names = []
    for v in varnames:
        comp_names.append(f"{v}_comp")
    total = xr.merge([forced[comp_names], total])

    return total


def load_T34_anom(index="sst"):
    """
    Monthly Niño 3.4 anomalies (ensemble mean removed) for each member.

    Args:
        - index (str): which Niño 3.4 index to use:
            - "sst": SST-based index (Th_anom.nc, via src.utils.load_cesm_indices);
              the one the RO fits use
            - "ml50" or "ml80": index from mixed-layer temperature averaged over the
              top 50 or 80 m (Th_50.nc or Th_80.nc)

    Returns:
        - T34_anom (xr.DataArray): anomalies, dims (member, time)
    """

    ## SST-based index (already stored as anomalies, but removing the ensemble
    ## mean again is harmless and keeps the two cases the same)
    if index == "sst":
        T34 = src.utils.load_cesm_indices()["T_34"]

    ## mixed-layer index; "ml80" -> Th_80.nc
    elif index in ["ml50", "ml80"]:
        T34 = src.utils.load_cesm_indices_3d(mld=int(index[2:]))["T_34"]

    ## anything else is a typo
    else:
        raise ValueError(f"index must be 'sst', 'ml50', or 'ml80', not {index!r}")

    ## anomalies: remove the ensemble mean (the forced response)
    T34_anom = T34 - T34.mean("member")

    return T34_anom


def load_T34_total(index="sst"):
    """
    Monthly Niño 3.4 index (total, forced response included) for each member.

    Args:
        - index (str): which Niño 3.4 index to use:
            - "sst": SST-based index (Th.nc)
            - "ml50" or "ml80": mixed-layer temperature averaged over the top 50 or
              80 m (Th_50.nc or Th_80.nc)

    Returns:
        - T34 (xr.DataArray): index (degrees C), dims (member, time), 1850-2100
    """

    ## SST-based index; drop the file's last time step (Jan 2101) so it ends in
    ## Dec 2100 like the other data (as src.utils.load_cesm_indices does)
    if index == "sst":
        fp = pathlib.Path(os.environ["DATA_FP"], "cesm", "Th.nc")
        T34 = xr.open_dataset(fp)["T_34"].isel(time=slice(None, -1))

    ## mixed-layer index; "ml80" -> Th_80.nc
    elif index in ["ml50", "ml80"]:
        T34 = src.utils.load_cesm_indices_3d(mld=int(index[2:]))["T_34"]

    ## anything else is a typo
    else:
        raise ValueError(f"index must be 'sst', 'ml50', or 'ml80', not {index!r}")

    return T34


def load_Q34(H=50, varname="nhf"):
    """
    Surface heat flux anomaly over Niño 3.4, as a temperature tendency.

    Q_34 = F_34 / (rho * Cp * H): the warming rate the heat flux F would give a
    mixed layer of depth H. All fluxes are positive into the ocean.

    Args:
        - H (float): mixed-layer depth (m)
        - varname (str): "nhf" (net surface heat flux, from consolidated/anom.nc),
          or one of its components from consolidated/flux_anom_with_clearsky.nc:
          "fsns" (net shortwave), "flns" (net longwave), "lhflx" (latent heat)

    Returns:
        - Q34 (xr.DataArray): anomalies (ensemble mean removed), K/yr, dims
          (member, time)
    """

    ## EOF scores and patterns of the flux anomaly
    if varname == "nhf":
        _, anom = src.utils.load_consolidated()
        sign = 1
    elif varname in ["fsns", "flns", "lhflx"]:
        _, anom = src.utils.load_flux_data()

        ## CESM's FLNS and LHFLX are positive upward: flip them to positive into the ocean
        if varname == "fsns":
            sign = 1
        else:
            sign = -1
    else:
        raise ValueError(f"unknown flux varname: {varname}")

    ## flux anomaly (W m^-2) averaged over Niño 3.4, from the EOF scores
    F_34 = src.utils.reconstruct_wrapper(
        anom[[varname, f"{varname}_comp"]], fn=src.utils.get_nino34
    )[varname]

    ## convert to a temperature tendency (K/yr), positive into the ocean
    Q34 = flux_to_tendency(sign * F_34, H=H)

    return Q34


def flux_to_tendency(F, H=50):
    """
    Convert a surface heat flux to the warming rate it gives a mixed layer.

    Q = F / (rho * Cp * H), in K/yr (with 30-day months, as in the old scripts).

    Args:
        - F (xr.DataArray or float): heat flux (W m^-2)
        - H (float): mixed-layer depth (m)

    Returns:
        - Q (xr.DataArray or float): temperature tendency (K/yr)
    """

    ## seawater density (kg m^-3), specific heat (J kg^-1 K^-1), seconds per year
    rho = 1.02e3
    cp = 4.2e3
    sec_per_yr = 12 * 30 * 8.64e4

    ## W m^-2 -> K/s, then K/yr
    Q = F * sec_per_yr / (rho * cp * H)

    return Q


def load_T34_windowed(index="sst", window_size=480, stride=60):
    """
    Niño 3.4 anomalies (ensemble mean removed), cut into overlapping windows.

    Args:
        - index (str): which Niño 3.4 index to use ("sst", "ml50", "ml80"; see
          load_T34_anom)
        - window_size (int): window length (months); the default is 40 years
        - stride (int): spacing between window centers (months); default 5 years

    Returns:
        - T34_windowed (xr.DataArray): dims (member, year, time); "year" is the
          window center and "time" the month within the window
    """

    ## monthly anomalies
    T34_anom = load_T34_anom(index)

    ## cut into windows
    T34_windowed = src.utils.get_windowed(
        T34_anom, window_size=window_size, stride=stride
    )

    return T34_windowed


def get_djf_anom(x_windowed):
    """
    DJF means of windowed data, measured from the median of each window.

    Anomalies are measured from the median (over members and years in each
    window) rather than the mean, because ENSO is skewed.

    Args:
        - x_windowed (xr.DataArray): monthly data with dims (member, year, time),
          e.g. from load_T34_windowed

    Returns:
        - x_djf (xr.DataArray): one DJF value per year of each window, minus the
          window's median
    """

    ## seasonal means, with seasons starting in December (DJF, MAM, ...)
    x_seasonal = x_windowed.resample({"time": "QS-DEC"}).mean()

    ## keep DJF; drop the first and last seasons, which are partial (the window
    ## runs Jan-Dec, so they hold only Jan-Feb and only Dec)
    x_djf = x_seasonal.sel(time=x_seasonal.time.dt.month == 12)
    x_djf = x_djf.isel(time=slice(1, -1))

    ## measure from the median over members and years in each window
    x_djf = x_djf - x_djf.median(["member", "time"])

    return x_djf


def get_window(data, year, window_size=480):
    """
    One window of monthly data, the same months src.utils.get_windowed labels
    with `year` (for data starting in January).

    Cutting out one window is cheaper than windowing the whole record, which
    matters for large fields such as EOF scores.

    Args:
        - data (xr.DataArray or xr.Dataset): monthly data with a "time" dimension
        - year (int): window center, as in get_windowed's "year" coordinate
        - window_size (int): window length (months); a multiple of 24

    Returns:
        - window (xr.DataArray or xr.Dataset): data from January of
          year - window_size / 24 to December of year + window_size / 24 - 1
          (e.g., 1990-2029 for year 2010 and 40-year windows). A window that
          runs past the end of the record is shorter here, where get_windowed
          pads it with NaN
    """

    ## half the window, in years
    half = window_size // 24

    ## select the window's months (keeping the real dates)
    window = data.sel(time=slice(f"{year - half}-01-01", f"{year + half - 1}-12-31"))

    return window


def get_window_month(data, year, month, window_size=480):
    """
    Values in one calendar month within one window (see get_window).

    Args:
        - data (xr.DataArray or xr.Dataset): monthly data, dims (member, time)
        - year (int): window center
        - month (int): calendar month (1-12)
        - window_size (int): window length (months); a multiple of 24

    Returns:
        - x_month (xr.DataArray or xr.Dataset): dims (member, time), one value
          per year in the window
    """

    ## cut out the window, then keep one calendar month
    window = get_window(data, year, window_size)
    x_month = window.sel(time=window.time.dt.month == month)

    return x_month.transpose("member", "time")


def get_composite_hov(
    anom,
    T34,
    varnames,
    years,
    window_size=480,
    peak_month=12,
    q=0.95,
    is_warm=True,
    event_type=None,
):
    """
    Equatorial (LATS_EQ) composites of strong ENSO events in each window.

    In each window, events are the months `peak_month` when T34 is beyond its q
    (El Niño) or 1 - q (La Niña) quantile over that calendar month; each event's
    sample runs from lag -12 to +12 months. The composite is the mean over events.

    Args:
        - anom (xr.Dataset): EOF-compressed anomalies (ensemble mean removed): for
          each name in varnames, the scores (member, time, mode) and the patterns
          in "<name>_comp"
        - T34 (xr.DataArray): Niño 3.4 anomalies (member, time), used to pick events
        - varnames (list of str): fields to composite
        - years (list of int): window centers (see get_window)
        - window_size (int): window length (months)
        - peak_month (int): calendar month of the event peak (lag 0)
        - q (float): quantile defining a strong event
        - is_warm (bool): True for El Niño, False for La Niña
        - event_type (int or None): None keeps all events; 1, 2, or 3 keeps only
          single-, double-, or triple-year events (consecutive peaks beyond the
          cutoff; see src.utils.sort_peak_idx). For a multi-year event, the
          composite uses its first peak

    Returns:
        - comp (xr.Dataset): dims (year, lag, longitude) for each field in varnames
          (meridional mean over LATS_EQ), plus the composite "T_34" (year, lag)
    """

    ## warm events are above the q quantile; cold events below the 1 - q quantile
    if is_warm:
        q_cut = q
        check_cutoff = lambda x, cut: x > cut  # noqa: E731
    else:
        q_cut = 1 - q
        check_cutoff = lambda x, cut: x < cut  # noqa: E731

    ## scores of the fields, plus the index used to pick the events
    scores = xr.merge([anom[varnames], T34.rename("T_34")])

    ## meridional mean, applied to the EOF patterns before reconstructing
    def eq_mean(x):
        """
        Mean over the equatorial band.

        Args:
            - x (xr.DataArray): field with a "latitude" dimension

        Returns:
            - x_eq (xr.DataArray): mean over LATS_EQ
        """
        return x.sel(latitude=slice(*LATS_EQ)).mean("latitude")

    ## composite for each window
    comps = []
    for year in years:

        ## this window's months
        window = get_window(scores, year, window_size)

        ## mean over the events in this window, dims (lag, ...)
        comp_scores = src.utils.make_composite(
            idx=window["T_34"],
            data=window,
            peak_month=peak_month,
            q=q_cut,
            check_cutoff=check_cutoff,
            event_type=event_type,
        )

        ## equatorial fields from the composite scores
        comp = xr.Dataset({"T_34": comp_scores["T_34"]})
        for varname in varnames:
            comp[varname] = src.utils.reconstruct_fn(
                components=anom[f"{varname}_comp"],
                scores=comp_scores[varname],
                fn=eq_mean,
            )
        comps.append(comp)

    ## stack the windows along a "year" dimension
    comp = xr.concat(comps, dim=xr.DataArray(years, dims="year", name="year"))

    return comp.transpose("year", "lag", ...)


def plot_hov_grid(warm, cold, rows, figsize, nino34_lons=(190, 240), year_line=2010):
    """
    Year-longitude plots of El Niño and La Niña composites: the values over
    time (left), and their difference from the first window (right).

    Args:
        - warm (xr.Dataset): El Niño composites, dims (year, longitude)
        - cold (xr.Dataset): La Niña composites, same variables and dims
        - rows (list of dict): one per row, with keys "var" (variable name),
          "label" (row label), "units" (colorbar label), "amp" (colorbar limit
          for the values; the differences use amp / 2), and optionally "cmap"
          (default "cmo.balance")
        - figsize (tuple): figure size (inches)
        - nino34_lons (tuple): longitudes of the dashed vertical lines (the
          Niño 3.4 region by default)
        - year_line (int): year of the dashed horizontal line

    Returns:
        - fig (matplotlib.figure.Figure): the figure
    """

    ## imported here so the data helpers above don't need matplotlib
    import matplotlib.pyplot as plt
    import cmocean  # noqa: F401  (registers the "cmo.*" colormaps)

    ## left: values over time; right: difference from the first window
    fig = plt.figure(figsize=figsize, layout="constrained")
    ## (the left half is wider, since it also holds the row labels)
    subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[1.25, 1])

    ## titles of the two halves
    year0 = warm.year.values[0]
    subfigs[0].suptitle("Anomaly over time", size=14)
    subfigs[1].suptitle(f"Difference from {year0}", size=14)

    ## one row per variable, El Niño and La Niña side by side
    nrows = len(rows)
    for subfig, is_diff in zip(subfigs, [False, True]):
        axs = subfig.subplots(nrows, 2, squeeze=False)

        for i, row in enumerate(rows):

            ## colorbar limit (smaller for the differences) and levels
            if is_diff:
                amp = row["amp"] / 2
            else:
                amp = row["amp"]
            levels = src.utils.make_cb_range(amp, amp / 10)

            for j, comp in enumerate([warm, cold]):

                ## values, or difference from the first window
                if is_diff:
                    x = comp[row["var"]] - comp[row["var"]].isel(year=0)
                else:
                    x = comp[row["var"]]

                ## filled contours, longitude on the x axis
                cp = axs[i, j].contourf(
                    comp.longitude,
                    comp.year,
                    x.transpose("year", "longitude"),
                    levels=levels,
                    cmap=row.get("cmap", "cmo.balance"),
                    extend="both",
                )

            ## one colorbar per row, next to the La Niña panel
            subfig.colorbar(cp, ax=axs[i, :], label=row["units"], ticks=[-amp, 0, amp])

        ## reference lines; ticks only on the outer panels (set below)
        for ax in axs.flatten():
            for lon in nino34_lons:
                ax.axvline(lon, ls="--", c="w", lw=0.8)
            ax.axhline(year_line, ls="--", c="w", lw=0.8)

            ## same longitude range everywhere; clear the ticks
            ax.set_xlim([140, 280])
            ax.set_xticks([])
            ax.set_yticks([])

        ## column titles
        axs[0, 0].set_title("El Niño")
        axs[0, 1].set_title("La Niña")

        ## longitude ticks on the bottom row
        for ax in axs[-1]:
            ax.set_xticks([145, 190, 240, 275])
            ax.set_xlabel("Longitude")

        ## year ticks on the left column of each half
        for ax in axs[:, 0]:
            ax.set_yticks([1880, 2010, 2080])

        ## row labels, left of the first half only
        if not is_diff:
            for ax, row in zip(axs[:, 0], rows):
                ax.set_ylabel(
                    row["label"], rotation=0, ha="right", va="center", size=12
                )

    return fig


def regress_by_sign(x, y, by_month=False):
    """
    Regression slope of y on x, separately for x > 0, x < 0, and all x.

    Each member and window gets its own fit: the regression is over "time" only.
    Fitting x > 0 and x < 0 separately gives a piecewise-linear relation, which
    captures El Niño/La Niña asymmetry.

    Args:
        - x (xr.DataArray): predictor, e.g. windowed Niño 3.4 anomalies with dims
          (member, year, time)
        - y (xr.DataArray): target, same dims as x
        - by_month (bool): if True, fit each calendar month separately (adds a
          "month" dimension)

    Returns:
        - slopes (xr.Dataset): slope of y on x for x > 0 ("pos"), x < 0 ("neg"),
          and all x ("all"), with x's dims other than "time" (plus "month")
    """

    ## predictor and target in one dataset, so masking applies to both
    data = xr.merge([x.rename("x"), y.rename("y")])

    ## subsets: all samples, warm (x > 0), and cold (x < 0)
    subsets = dict(
        all=data, pos=data.where(data["x"] > 0), neg=data.where(data["x"] < 0)
    )

    ## regression settings: one fit per member and window
    kwargs = dict(x_vars=["x"], y_vars=["y"], dims=["time"])

    ## fit each subset
    slopes = xr.Dataset()
    for name, subset in subsets.items():

        ## one fit for each calendar month, or one for all months together
        if by_month:
            coefs = subset.groupby("time.month").map(
                src.utils.regress_xr_proj, **kwargs
            )
        else:
            coefs = src.utils.regress_xr_proj(subset, **kwargs)

        ## keep the slope (dropping the predictor-index coordinate)
        slopes[name] = coefs["y"].drop_vars("j")

    return slopes


def get_lag_growth_rate(T, lag=1, month=None, by_month=False):
    """
    Growth rate of T from lag regression, separately for warm and cold starts.

    Regresses T(t + lag) on T(t), giving a slope S per member and window. If
    T(t + lag) = S T(t), then T grows at the rate R = log(S) / lag. Warm and cold
    subsets are chosen by the sign of T(t).

    Args:
        - T (xr.DataArray): windowed monthly anomalies, dims (member, year, time),
          as from src.utils.get_windowed
        - lag (int): months between the two times
        - month (int or None): calendar month of the starting time t; None uses
          all months
        - by_month (bool): if True, fit each starting month separately (adds a
          "month" dimension)

    Returns:
        - R (xr.Dataset): growth rate (yr^-1) for T(t) > 0 ("pos"), T(t) < 0
          ("neg"), and all T(t) ("all")
    """

    ## starting values, and the values `lag` months later on the same time axis;
    ## pairs stay within a window, since the shift is along "time"
    T0 = T.isel(time=slice(None, -lag))
    Tf = T.isel(time=slice(lag, None)).assign_coords(time=T0.time)

    ## (optional) keep only one starting month
    if month is None:
        T0_sel = T0
        Tf_sel = Tf
    else:
        T0_sel = src.utils.sel_month(T0, months=month)
        Tf_sel = src.utils.sel_month(Tf, months=month)

    ## lag-regression slope S
    S = regress_by_sign(T0_sel, Tf_sel, by_month=by_month)

    ## growth rate: log(S) per `lag` months, times 12 months per year
    R = 12 / lag * np.log(S)

    return R
