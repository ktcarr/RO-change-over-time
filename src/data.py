"""
Loading and preparing data: indices and fields from DATA_FP, unit conversions,
windows, and seasonal means.
"""

import calendar
import os
import pathlib

import xarray as xr

import src.constants
import src.utils

## observational datasets from the CESM Climate Variability Diagnostics Package
CVDP_OBS_FP = pathlib.Path("/glade/campaign/cgd/cas/asphilli/CVDP-OBS")


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


def load_nino_anom(index="sst"):
    """
    Monthly Niño 4, 3.4, and 3 anomalies (ensemble mean removed) for each member.

    Args:
        - index (str): which indices to use:
            - "sst": SST-based indices (Th_anom.nc, via src.utils.load_cesm_indices);
              the ones the RO fits use
            - "ml50" or "ml80": indices from mixed-layer temperature averaged over the
              top 50 or 80 m (Th_50.nc or Th_80.nc)

    Returns:
        - nino_anom (xr.Dataset): "T_4", "T_34", "T_3", each with dims (member, time)
    """

    ## SST-based indices (already stored as anomalies, but removing the ensemble
    ## mean again is harmless and keeps the two cases the same)
    if index == "sst":
        nino = src.utils.load_cesm_indices()

    ## mixed-layer indices; "ml80" -> Th_80.nc
    elif index in ["ml50", "ml80"]:
        nino = src.utils.load_cesm_indices_3d(mld=int(index[2:]))

    ## anything else is a typo
    else:
        raise ValueError(f"index must be 'sst', 'ml50', or 'ml80', not {index!r}")

    ## anomalies: remove the ensemble mean (the forced response)
    nino = nino[["T_4", "T_34", "T_3"]]
    nino_anom = nino - nino.mean("member")

    return nino_anom


def load_T34_anom(index="sst"):
    """
    Monthly Niño 3.4 anomalies (ensemble mean removed) for each member.

    Args:
        - index (str): which Niño 3.4 index to use ("sst", "ml50", "ml80"; see
          load_nino_anom)

    Returns:
        - T34_anom (xr.DataArray): anomalies, dims (member, time)
    """
    return load_nino_anom(index)["T_34"]


def load_eli(varname="eli_05"):
    """
    Monthly ENSO longitude index (ELI) anomalies (ensemble mean removed): the
    average longitude of the equatorial Pacific grid points whose SST exceeds the
    equatorial-mean SST (computed by preprocess/preprocess_cesm.py).

    Args:
        - varname (str): variable in DATA_FP/cesm/eli.nc; "eli_05" uses the
          5S-5N mean SST as the threshold

    Returns:
        - eli_anom (xr.DataArray): anomalies (degrees longitude), dims
          (member, time)
    """

    ## ELI of each member
    fp = pathlib.Path(os.environ["DATA_FP"], "cesm", "eli.nc")
    eli = xr.open_dataset(fp)[varname]

    ## anomalies: remove the ensemble mean (the forced response)
    eli_anom = eli - eli.mean("member")

    return eli_anom


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
    Surface heat flux anomaly over Niño 3.4, as a temperature tendency (see
    load_Q).

    Args:
        - H (float): mixed-layer depth (m)
        - varname (str): "nhf", "fsns", "flns", or "lhflx" (see load_Q)

    Returns:
        - Q34 (xr.DataArray): anomalies (ensemble mean removed), K/yr, dims
          (member, time)
    """

    ## Niño 3.4 average
    Q34 = load_Q(H=H, varname=varname, region="34")

    return Q34


def load_eofs(varnames):
    """
    CESM2 consolidated (EOF-compressed) fields: the ensemble mean ("forced") and
    the anomalies from it, for the chosen variables.

    Args:
        - varnames (list of str): variables in consolidated/{forced,anom}.nc
          (e.g. "sst", "pr", "nhf") or in
          consolidated/flux_{forced,anom}_with_clearsky.nc ("fsns", "flns",
          "lhflx", "fsnsc"); CESM's units and sign conventions (see load_F)

    Returns:
        - forced (xr.Dataset): for each name, the scores (time, mode) and the
          patterns "<name>_comp" (mode, latitude, longitude)
        - anom (xr.Dataset): same, with scores (member, time, mode)
    """

    ## surface flux components are in their own files
    forced_main, anom_main = src.utils.load_consolidated()
    forced_flux, anom_flux = src.utils.load_flux_data()

    ## each variable's scores and patterns, from the file that has it
    forced_list = []
    anom_list = []
    for varname in varnames:
        names = [varname, f"{varname}_comp"]
        if varname in ["fsns", "flns", "lhflx", "fsnsc"]:
            forced_list.append(forced_flux[names])
            anom_list.append(anom_flux[names])
        else:
            forced_list.append(forced_main[names])
            anom_list.append(anom_main[names])

    ## one dataset for each
    forced = xr.merge(forced_list)
    anom = xr.merge(anom_list)

    return forced, anom


def load_region_avg(varname, region="34"):
    """
    CESM2 anomaly (ensemble mean removed) averaged over a Niño region,
    reconstructed from the consolidated (EOF-compressed) fields.

    Args:
        - varname (str): a variable in consolidated/anom.nc (e.g. "sst", "nhf")
          or in consolidated/flux_anom_with_clearsky.nc ("fsns", "flns",
          "lhflx", "fsnsc"); fluxes keep CESM's sign conventions (see load_F)
        - region (str): "3" (Niño 3), "34" (Niño 3.4), or "4" (Niño 4)

    Returns:
        - x (xr.DataArray): region average, dims (member, time)
    """

    ## EOF scores and patterns of the anomalies
    _, anom = load_eofs([varname])

    ## function that averages a field over the region
    if region == "3":
        region_fn = src.utils.get_nino3
    elif region == "34":
        region_fn = src.utils.get_nino34
    elif region == "4":
        region_fn = src.utils.get_nino4
    else:
        raise ValueError(f"region must be '3', '34', or '4', not {region!r}")

    ## region average, from the EOF scores
    x = src.utils.reconstruct_wrapper(anom[[varname, f"{varname}_comp"]], fn=region_fn)
    x = x[varname]

    return x


def load_F(varname="nhf", region="34"):
    """
    Surface heat flux anomaly (ensemble mean removed) averaged over a Niño
    region, positive into the ocean.

    Args:
        - varname (str): "nhf" (net surface heat flux, from consolidated/anom.nc),
          or one of its components from consolidated/flux_anom_with_clearsky.nc:
          "fsns" (net shortwave), "flns" (net longwave), "lhflx" (latent heat)
        - region (str): "3", "34", or "4" (see load_region_avg)

    Returns:
        - F (xr.DataArray): anomalies, W m^-2, dims (member, time)
    """

    ## sign that makes the flux positive into the ocean: CESM's FLNS and LHFLX
    ## are positive upward (nhf = fsns - flns - lhflx - shflx)
    if varname in ["nhf", "fsns"]:
        sign = 1
    elif varname in ["flns", "lhflx"]:
        sign = -1
    else:
        raise ValueError(f"unknown flux varname: {varname}")

    ## flux anomaly averaged over the region, positive into the ocean
    F = sign * load_region_avg(varname, region=region)

    return F


def load_Q(H=50, varname="nhf", region="34"):
    """
    Surface heat flux anomaly averaged over a Niño region, as a temperature
    tendency.

    Q = F / (rho * Cp * H): the warming rate the heat flux F would give a
    mixed layer of depth H. All fluxes are positive into the ocean.

    Args:
        - H (float): mixed-layer depth (m)
        - varname (str): "nhf", "fsns", "flns", or "lhflx" (see load_F)
        - region (str): "3", "34", or "4" (see load_region_avg)

    Returns:
        - Q (xr.DataArray): anomalies (ensemble mean removed), K/yr, dims
          (member, time)
    """

    ## flux anomaly (W m^-2), positive into the ocean
    F = load_F(varname=varname, region=region)

    ## convert to a temperature tendency (K/yr)
    Q = flux_to_tendency(F, H=H)

    return Q


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


def get_season(x, start_month):
    """
    Three-month seasonal means of monthly data for one season each year (e.g.
    Sep-Oct-Nov for start_month=9), full seasons only.

    Args:
        - x (xr.Dataset or xr.DataArray): monthly data with a "time" dimension
        - start_month (int): first month of the season (1-12)

    Returns:
        - x_season (xr.Dataset or xr.DataArray): one value per full season,
          labeled by its first month
    """

    ## three-month means, with seasons starting in start_month
    rule = f"QS-{calendar.month_abbr[start_month].upper()}"
    x_seasonal = x.resample({"time": rule}).mean()

    ## number of months in each season (partial at the start and end of the data)
    n_months = x["time"].resample({"time": rule}).count()

    ## keep the chosen season, full seasons only
    is_kept = (x_seasonal.time.dt.month == start_month) & (n_months == 3)
    x_season = x_seasonal.sel(time=is_kept.values)

    return x_season


def load_era5_flux():
    """
    ERA5 monthly surface heat fluxes (DATA_FP/era5_nhf_glade.nc), 1950-2022.

    Returns:
        - flux (xr.Dataset): net shortwave "sw", net longwave "lw", latent "lhf",
          sensible "shf", and their sum "nhf" (W m^-2, positive into the ocean,
          ECMWF's convention), dims (time, latitude, longitude), latitude
          ascending
    """

    ## open, dropping the unused date variable
    fp = pathlib.Path(os.environ["DATA_FP"], "era5_nhf_glade.nc")
    flux = xr.open_dataset(fp).drop_vars("utc_date")

    ## latitude ascending (as the Niño-region functions expect)
    flux = flux.reindex(latitude=flux.latitude[::-1])

    ## short names for the fluxes (mean surface net/latent/sensible fluxes)
    names = dict(MSNSWRF="sw", MSNLWRF="lw", MSLHF="lhf", MSSHF="shf")
    flux = flux[list(names)].rename(names)

    ## net surface heat flux
    flux["nhf"] = flux["sw"] + flux["lw"] + flux["lhf"] + flux["shf"]

    return flux


def load_ersst(varname="ssta"):
    """
    ERSSTv5 monthly SST (as provided by CVDP-OBS), 1854-2024.

    Args:
        - varname (str): "ssta" (anomaly from CVDP's monthly climatology) or
          "sst" (total)

    Returns:
        - sst (xr.DataArray): SST (anomaly in K, or total in °C), dims (time,
          latitude, longitude)
    """

    ## open and use this project's coordinate names
    sst = xr.open_dataset(CVDP_OBS_FP / "ersstv5.185401-202412.nc")[varname]
    sst = sst.rename({"lon": "longitude", "lat": "latitude"})

    ## monthly times at the start of each month (to match the other datasets)
    sst["time"] = xr.date_range(start="1854-01", end="2024-12", freq="MS")

    return sst


def load_gpcp():
    """
    GPCP monthly precipitation (as provided by CVDP-OBS), Jan 1979 - Mar 2024.

    Returns:
        - pr (xr.DataArray): precipitation (mm day^-1), dims (time, latitude,
          longitude)
    """

    ## open, using this project's names for the coordinates and the variable
    pr = xr.open_dataset(CVDP_OBS_FP / "gpcp.mon.mean.197901-202403.nc")["precip"]
    pr = pr.rename({"lon": "longitude", "lat": "latitude"}).rename("pr")

    return pr
