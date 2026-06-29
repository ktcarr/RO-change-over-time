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
