import xarray as xr
from src.XRO import XRO
import tqdm
import warnings

def get_RO_ensemble(data, T_var="T_3", h_var="h_w", verbose=False, fit_kwargs=dict(), model=None):
    """get RO params for each ensemble member"""

    ## initialize model
    if model is None:
        model = XRO(ncycle=12, ac_order=1, is_forward=True)

    ## empty list to hold model fits
    fits = []

    ## Loop thru ensemble members
    for m in tqdm.tqdm(data.member, disable=not (verbose)):

        ## select ensemble member and variables
        data_subset = data[[T_var, h_var]].sel(member=m)

        ## fit model
        with warnings.catch_warnings(action="ignore"):
            fits.append(model.fit_matrix(data_subset, **fit_kwargs))

    return model, xr.concat(fits, dim=data.member)
