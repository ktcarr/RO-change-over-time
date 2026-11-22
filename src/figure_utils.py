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
