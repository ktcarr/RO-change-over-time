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
