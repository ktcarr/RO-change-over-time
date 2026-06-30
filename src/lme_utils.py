import xarray as xr
import numpy as np
import pandas as pd
import xeofs as xe
import tqdm
import warnings
import pathlib
import cartopy.crs as ccrs
import cartopy.util
import matplotlib.pyplot as plt
import scipy.stats
import copy
import calendar
import os


def load_volc_forcing(cutoff=25):
    volc_url = r"https://climate.envsci.rutgers.edu/IVI2/IVI2TotalInjection_501-2000Version2.txt"
    volc_forcing = pd.read_csv(volc_url, sep=r"\s+", header=8, on_bad_lines="warn")

    ## convert to xr
    volc_forcing = volc_forcing.set_index("Year").to_xarray()
    volc_forcing = volc_forcing[["NH", "SH", "Global"]].rename({"Year": "year"})

    ## only show large
    volc_forcing = volc_forcing.where(volc_forcing > cutoff, other=np.nan)

    ## normalize
    for v in ["NH", "SH", "Global"]:
        volc_forcing[f"{v}_norm"] = volc_forcing[v] / volc_forcing["Global"].max()

    return volc_forcing.sel(year=slice(850, None))


def get_eli_helper(exceeds_thresh):
    """compute ELI from mask of threshold exceedance"""

    ## sum and count longitudes exceeding thresh
    lon = exceeds_thresh.longitude
    longitude_sum = (exceeds_thresh * lon).sum(["longitude", "latitude"])
    longitude_count = exceeds_thresh.sum(["longitude", "latitude"])

    ## eli is average longitude
    eli = longitude_sum / longitude_count

    return eli


def get_eli(rsst, rsst_thresh=0, max_lat=15):
    """compute ELI from tropical SST data"""

    ## get equatorial Pac. SST
    rsst_pac = rsst.sel(longitude=slice(120, 280), latitude=slice(-max_lat, max_lat))

    ## get boolean array where SST exceeds thresh
    exceeds_thresh0 = rsst_pac >= rsst_thresh

    ## compute initial ELI estimate
    eli0 = get_eli_helper(exceeds_thresh0)

    return eli0
