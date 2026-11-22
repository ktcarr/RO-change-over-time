"""
Plotting: saving figures, maps, region boxes, contour levels, and Hovmöller grids.
"""

import os
import pathlib

import cartopy.crs as ccrs
import cartopy.util
import cmocean  # noqa: F401  (registers the "cmo.*" colormaps)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import src.constants
import src.utils


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


def plot_wbox(ax, **kwargs):
    """
    Outline the western equatorial box on a map.

    Args:
        - ax (cartopy GeoAxes): map to draw on
        - kwargs: passed to ax.plot (e.g., c, lw, ls)

    Returns:
        - None
    """
    src.utils.plot_box(
        ax, lons=src.constants.LONS_W, lats=src.constants.LATS_EQ, **kwargs
    )
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
    src.utils.plot_box(
        ax, lons=src.constants.LONS_E, lats=src.constants.LATS_EQ, **kwargs
    )
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


def plot_lag_hov_grid(comp0, comp1, rows, titles, box_color="k", figsize=(8, 8)):
    """
    Lag-longitude plots of a composite ENSO event in two windows: one row per
    variable, with the first window (left), the second (middle), and their
    difference (right). The ENSO longitude index is drawn on each panel (solid
    for the first window, dashed for the second).

    Args:
        - comp0 (xr.Dataset): composite in the first window, dims (lag, longitude)
          for each variable in rows, plus "eli" (lag): the total ENSO longitude
          index (degrees east)
        - comp1 (xr.Dataset): composite in the second window, same variables
        - rows (list of dict): one per row, with keys "var" (variable name),
          "label" (colorbar label), "amp" (colorbar limit; the differences use
          amp / 2), and optionally "cmap" (default "cmo.balance")
        - titles (list of str): titles of the three columns
        - box_color (str): color of the box (Niño 3.4 longitudes, lags -1 to 1)
          on the difference panels
        - figsize (tuple): figure size (inches)

    Returns:
        - fig (matplotlib.figure.Figure): the figure
    """

    ## one row per variable, three columns
    fig, axs = plt.subplots(len(rows), 3, figsize=figsize, layout="constrained")

    ## style of the reference lines (longitude ticks and lag 0)
    line_kwargs = dict(c="w", ls="--", lw=0.8, alpha=0.5)
    xticks = [210, 270]

    for i, row in enumerate(rows):

        ## values in each window, and the difference
        x0 = comp0[row["var"]]
        x1 = comp1[row["var"]]
        panels = [x0, x1, x1 - x0]

        ## colorbar limit for each panel (half for the difference)
        amps = [row["amp"], row["amp"], row["amp"] / 2]

        ## filled contours, longitude on the x axis, 20 levels per panel
        for ax, x, amp in zip(axs[i], panels, amps):
            cp = ax.contourf(
                x.longitude,
                x.lag,
                x.transpose("lag", "longitude"),
                levels=src.utils.make_cb_range(amp, amp / 10),
                cmap=row.get("cmap", "cmo.balance"),
                extend="both",
            )

            ## reference lines
            for tick in xticks:
                ax.axvline(tick, **line_kwargs)
            ax.axhline(0, **line_kwargs)

            ## colorbar for the second window (shared with the first) ...
            if ax is axs[i, 1]:
                fig.colorbar(cp, ax=ax, ticks=[-amp, 0, amp])
            ## ... and for the difference, with the row label
            elif ax is axs[i, 2]:
                fig.colorbar(cp, ax=ax, ticks=[-amp, 0, amp], label=row["label"])
            else:
                pass

        ## ENSO longitude index: solid for the first window, dashed for the second
        axs[i, 0].plot(comp0["eli"], comp0.lag, c="k", ls="-")
        axs[i, 1].plot(comp1["eli"], comp1.lag, c="k", ls="--")

        ## both on the difference
        axs[i, 2].plot(comp0["eli"], comp0.lag, c="k", ls="-")
        axs[i, 2].plot(comp1["eli"], comp1.lag, c="k", ls="--")

        ## box on the difference: Niño 3.4 longitudes, lags -1 to 1
        box = mpatches.Rectangle(
            xy=(190, -1), width=50, height=2, edgecolor=box_color, facecolor="none"
        )
        axs[i, 2].add_patch(box)

    ## same axis range everywhere (lags -6 to 6 shown)
    for ax in axs.flatten():
        ax.set_xlim([145, 280])
        ax.set_ylim([-6.5, 6.5])

        ## clear the ticks (added back on the outer panels below)
        ax.set_xticks([])
        ax.set_yticks([])

    ## longitude ticks on the bottom row only
    for ax in axs[-1]:
        ax.set_xticks(xticks)
        ax.set_xlabel("Lon")

    ## lag ticks on the left column (lag 0 = December)
    for ax in axs[:, 0]:
        ax.set_yticks([-6, 0, 6], labels=["Jun(-1)", "Dec(0)", "Jun(+1)"])

    ## column titles
    for ax, title in zip(axs[0], titles):
        ax.set_title(title, fontsize=12)

    ## panel labels (a), (b), ... in the lower left, row by row
    for k, ax in enumerate(axs.flatten()):
        ax.text(0.02, 0.05, f"({chr(ord('a') + k)})", transform=ax.transAxes, c="k")

    return fig


def plot_amplitudes(ax, amp):
    """
    Plot the amplitudes of warm (red) and cold (blue) events and their average
    (black) over time, as in the RO perturbation experiment figures.

    Args:
        - ax (matplotlib.axes.Axes): axes to plot on
        - amp (xr.Dataset): "warm", "cold", and "mean", dims (year,) (e.g. from
          src.ro.get_amplitudes, or differences between them)

    Returns:
        - None
    """

    ## warm and cold events, and their average
    ax.plot(amp.year, amp["warm"], c="r", label="El Niño")
    ax.plot(amp.year, amp["cold"], c="b", label="La Niña")
    ax.plot(amp.year, amp["mean"], c="k", label="Average")

    ## zero line
    ax.axhline(0, ls="--", c="k", lw=0.8)

    return


def plot_zonal_composites(clim, warm, cold, panels, flux_vars, lon_lines=(210, 270)):
    """
    Equatorial profiles of El Niño and La Niña composites, as in the
    Q-structure figures: totals (top row) and anomalies (bottom row), one
    column per variable.

    Top row: climatology (black), climatology plus the El Niño (red) and La Niña
    (blue) anomalies. Bottom row: the anomalies, with La Niña's sign reversed so
    the two can be compared. The flux anomaly panels share their y-limits, with
    the y-axis pointing down (so heat loss by the ocean is up).

    Args:
        - clim (xr.Dataset): climatological profiles, dims (longitude,)
        - warm (xr.Dataset): El Niño composite anomalies, dims (longitude,)
        - cold (xr.Dataset): La Niña composite anomalies, dims (longitude,)
        - panels (dict): variable name -> column title, in column order
        - flux_vars (list of str): variables in panels that are surface fluxes
        - lon_lines (tuple of float): longitudes marked with vertical lines

    Returns:
        - fig (matplotlib.figure.Figure): the figure
        - axs (np.ndarray of Axes): axes, shape (2, len(panels))
    """

    ## one column per variable
    fig, axs = plt.subplots(2, len(panels), figsize=(12, 4.5), layout="constrained")

    for j, (varname, title) in enumerate(panels.items()):

        ## top row: climatology
        axs[0, j].plot(clim.longitude, clim[varname], c="k")

        ## top row: climatology plus each composite anomaly
        axs[0, j].plot(clim.longitude, clim[varname] + warm[varname], c="r")
        axs[0, j].plot(clim.longitude, clim[varname] + cold[varname], c="b")
        axs[0, j].set_title(title)

        ## bottom row: anomalies (La Niña's sign reversed), with a zero line
        axs[1, j].plot(warm.longitude, warm[varname], c="r")
        axs[1, j].plot(cold.longitude, -cold[varname], c="b")
        axs[1, j].axhline(0, c="k", ls="--", lw=0.8, zorder=0.5)

    ## row labels to the right
    for ax, label in zip(axs[:, -1], ["Total", "Anom."]):
        ax.text(s=label, x=1.1, y=0.5, transform=ax.transAxes)

    ## longitude ticks and lines
    lon_ticks = [130, *lon_lines]
    src.utils.add_vticks(axs.flatten(), xticks=lon_ticks, xlines=lon_lines)

    ## tick labels on the bottom row only
    for ax in axs[0]:
        ax.set_xticks([])
    for ax in axs[1]:
        ax.set_xlabel("Longitude")

    ## flux anomaly panels: same limits
    is_flux = np.isin(list(panels), flux_vars)
    flux_axs = axs[1, is_flux]
    src.utils.set_lims(flux_axs)

    ## flux anomaly panels: y-axis pointing down
    for ax in flux_axs:
        ax.set_ylim(ax.get_ylim()[::-1])

    return fig, axs


def add_cyclic_month(data):
    """
    Repeat January after December, so a month-by-longitude Hovmöller plot
    covers the whole annual cycle (month 13 is the next January).

    Args:
        - data (xr.DataArray): data with dims (month, longitude), months 1-12

    Returns:
        - values (np.ndarray): data with the extra month, shape (13, n_longitude)
        - months (np.ndarray): months 1-13
    """

    ## month first, then append the first month's values at month 13
    data = data.transpose("month", "longitude")
    values, months = cartopy.util.add_cyclic_point(data, data.month, axis=0)

    return values, months
