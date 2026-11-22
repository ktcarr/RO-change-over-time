"""
Loading and preparing data: indices and fields from DATA_FP, unit conversions,
windows, and seasonal means.
"""

import os
import pathlib

import xarray as xr

import src.constants
import src.utils


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

    ## W m^-2 -> K/s (dividing by the heat capacity of the layer), then K/yr
    rho_cp = src.constants.RHO_SW * src.constants.CP_SW
    Q = F * src.constants.SEC_PER_YR / (rho_cp * H)

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


def get_djf(x):
    """
    DJF means of windowed (or simulated) monthly data, full seasons only.

    The windows run January to December, so the first DJF (January-February only)
    and the last (December only) are partial and are dropped; 39 remain.

    Args:
        - x (xr.Dataset or xr.DataArray): monthly data with a "time" dimension
          starting in January

    Returns:
        - x_djf (xr.Dataset or xr.DataArray): one value per full DJF season
    """

    ## seasonal means, with seasons starting in December (DJF, MAM, JJA, SON)
    x_seasonal = x.resample({"time": "QS-DEC"}).mean()

    ## every 4th season starting from the first full DJF (index 4); the last 4
    ## seasons hold the partial final DJF
    x_djf = x_seasonal.isel(time=slice(4, -4, 4))

    return x_djf
