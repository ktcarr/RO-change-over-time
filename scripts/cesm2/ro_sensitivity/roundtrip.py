"""
Round-trip test of how h_hat is made: does removing h's linear dependence on T
bias the fitted RO?

h_hat is h minus its regression on T (one regression per calendar month, or one
for all months). That removes all of h's lag-0 correlation with T, including any
that the RO's dynamics produce (e.g. through the nonlinear or seasonal terms),
not just a linear "instantaneous" dependence. To see what that does, this script
takes the RO fitted to CESM2 for one case (cached by simulate_case.py) as the
truth, simulates synthetic data the size of the CESM2 ensemble (100 members of
40 years) with several seeds, and fits the RO again to the synthetic T and to
its thermocline variable treated three ways:
    - "none": as simulated (only the fit's sampling error)
    - "hconst": minus one regression on T for all months
    - "hmonth": minus one regression on T for each calendar month
Each is centered and fitted with the case's settings. Comparing the refitted
parameters with the true ones shows the bias from each h_hat method.

The same refits show whether the seasonal cycles of the nonlinear terms (R_2,
R_3) are recovered at this sample size, where the lagged T^2 is a weak
instrument after the spring barrier.

Output: SAVE_FP/ro_sensitivity/<tag>/roundtrip.nc, with the true and refitted
parameters (src.ro.get_params), dims (year, seed, method, cycle).

Usage (from scripts/cesm2, with DATA_FP and SAVE_FP set):
    python ro_sensitivity/roundtrip.py <tag>
"""

import json
import os
import sys

import xarray as xr

import src.ro
import src.utils


## windows used as the truth, and the number of synthetic ensembles
YEARS = [1870, 2080]
SEEDS = range(5)

## the size of each synthetic ensemble: as many members and years as CESM2
NCOPY = 100
NYEAR = 40

## the h_hat methods compared
METHODS = ["none", "hconst", "hmonth"]


def make_h_hat(x, method):
    """
    The synthetic data's h_hat for one method, then centered like CESM2.

    Args:
        - x (xr.Dataset): synthetic T_3 and h_hat (the true thermocline
          variable), dims (time, member)
        - method (str): "none", "hconst" or "hmonth" (see the module docstring)

    Returns:
        - y (xr.Dataset): T_3 and h_hat for fitting, dims (time, member)
    """

    ## remove the regression on T (or not), pooling members and times
    y = x.copy()
    dims = ["time", "member"]
    if method == "hmonth":
        y["h_hat"] = src.utils.remove_sst_dependence_v2(
            Th=x, h_var="h_hat", T_var="T_3", dims=dims
        )
    elif method == "hconst":
        y["h_hat"] = src.utils.remove_sst_dependence_core(
            h=x["h_hat"], T=x["T_3"], dims=dims
        )
    else:
        pass

    ## center each calendar month, as in load_ro_data
    if case["remove_median"]:
        center = y.groupby("time.month").median(dims)
    else:
        center = y.groupby("time.month").mean(dims)
    y = (y.groupby("time.month") - center).drop_vars("month")

    return y


## this job's case, and the RO fitted to CESM2 for it
tag = sys.argv[1]
case_dir = f"{os.environ['SAVE_FP']}/ro_sensitivity/{tag}"
with open(f"{case_dir}/config.json") as f:
    case = json.load(f)
fits = xr.open_dataset(f"{case_dir}/fits.nc")
data = xr.open_dataset(f"{case_dir}/data.nc")
print(tag, case, flush=True)

## model and fit settings, as in simulate_case.py
model = src.ro.get_model(
    taus=case["taus"], ac_order=case["ac_order"], is_forward=case["is_forward"]
)
fit_kwargs = src.ro.get_fit_kwargs(
    fit_constant=case["fit_constant"],
    flat_constant=case["flat_constant"],
    flat_nonlinear=case["flat_nonlinear"],
)

## simulation settings: this case's noise, a synthetic ensemble the size of CESM2's
sim_kwargs = dict(
    src.ro.SIM_KWARGS, noise_scale=case["noise_scale"], ncopy=NCOPY, nyear=NYEAR
)
X0 = data[src.ro.VARNAMES].isel(year=0, member=0, time=0)

## true and refitted parameters for each window
truth = []
refit = []
for year in YEARS:
    fit_true = fits.sel(year=year)
    truth.append(src.ro.get_params(fit_true))

    ## refit synthetic ensembles from several seeds
    refit_year = []
    for seed in SEEDS:
        sim_kwargs["seed"] = seed
        x = model.simulate(fit_ds=fit_true, X0_ds=X0, **sim_kwargs)

        ## each h_hat method
        refit_seed = []
        for method in METHODS:
            y = make_h_hat(x[src.ro.VARNAMES], method)
            fit = src.ro.fit_window(
                y, model, fit_kwargs, nonpositive_T3=case["nonpositive_T3"]
            )
            refit_seed.append(src.ro.get_params(fit))
        refit_year.append(
            xr.concat(refit_seed, dim=xr.DataArray(METHODS, dims="method"))
        )
        print(year, seed, flush=True)
    refit.append(xr.concat(refit_year, dim=xr.DataArray(list(SEEDS), dims="seed")))

## stack the windows; true parameters get the prefix "true_"
years = xr.DataArray(YEARS, dims="year")
out = xr.concat(refit, dim=years)
truth = xr.concat(truth, dim=years)
for name in truth:
    out[f"true_{name}"] = truth[name]
out.to_netcdf(f"{case_dir}/roundtrip.nc")
print("done", flush=True)
