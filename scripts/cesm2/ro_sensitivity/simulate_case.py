"""
Fit and simulate the RO for one case of the sensitivity tests (cases.py), and
cache everything the diagnostics need, so they can change without rerunning
the simulations.

For the case's settings, this fits the RO in each 40-year window of CESM2,
simulates it (control), and runs the perturbation experiments of the
RO-experiments figures: fixing the noise, R, F1, or F2 and eps to their 1870
values (fig:RO-experiments), and removing the changes in the ocean part R_o or
in each heat flux's feedback alpha (fig:RO-experiments_Ro, _alpha).

Outputs, in SAVE_FP/ro_sensitivity/<tag>/:
    - config.json: the case's settings
    - data.nc: the CESM2 data the RO was fitted to (T_3, h_hat), and T_scale
    - fits.nc: the fitted parameters in each window
    - alpha.nc: the heat flux feedbacks, one variable per flux
    - experiments_R3.nc: the T^3 coefficient of each experiment's parameters
      (to check whether removing a part of R makes it positive)
    - sims_ctrl.nc: control simulations (T_3, h_hat)
    - sims_<experiment>.nc: each experiment's simulations (T_3 only)

Usage (from scripts/cesm2, with DATA_FP and SAVE_FP set):
    python ro_sensitivity/simulate_case.py <tag>
"""

import json
import os
import sys

import numpy as np
import xarray as xr

import src.ro

## the case table (cases.py, next to this script; Python puts the script's
## folder on the import path)
import cases


## the experiments' reference window and mixed-layer depth, as in the
## RO-experiments fig scripts
REF_YEAR = 1870
H_ML = 50


def save_sims(sims, path, varnames):
    """
    Save simulations as float32 (half the size; far more precise than needed).

    Args:
        - sims (xr.Dataset): simulated RO variables, dims (year, time, member)
        - path (str): output file
        - varnames (list of str): variables to keep

    Returns:
        - None
    """

    ## keep only these variables, in single precision
    sims[varnames].astype(np.float32).to_netcdf(path)


## this job's case
tag = sys.argv[1]
case = cases.CASES[tag]
print(tag, case, flush=True)

## output folder, with the settings saved next to the results
out_dir = f"{os.environ['SAVE_FP']}/ro_sensitivity/{tag}"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/config.json", "w") as f:
    json.dump(dict(case, ref_year=REF_YEAR, H_ML=H_ML), f, indent=2)

## CESM2 data, centered and with h_hat as in this case
data, T_scale = src.ro.load_ro_data(
    t_var="T_3", remove_median=case["remove_median"], h_by_month=case["h_by_month"]
)

## save the data, with T_scale to convert T back to K
data_out = data.copy()
data_out.attrs["T_scale"] = T_scale
data_out.to_netcdf(f"{out_dir}/data.nc")

## model and fit settings for this case
model = src.ro.get_model(
    taus=case["taus"], ac_order=case["ac_order"], is_forward=case["is_forward"]
)
fit_kwargs = src.ro.get_fit_kwargs(
    fit_constant=case["fit_constant"],
    flat_constant=case["flat_constant"],
    flat_nonlinear=case["flat_nonlinear"],
)

## fit each window
fits = src.ro.fit_over_time(
    data[src.ro.VARNAMES],
    model,
    fit_kwargs,
    fit_noise=case["fit_noise"],
    nonpositive_T3=case["nonpositive_T3"],
)
fits.to_netcdf(f"{out_dir}/fits.nc")
print("fitted", flush=True)

## heat flux feedbacks, from fluxes centered the same way as T
Q = src.ro.load_ro_flux_data(
    T_scale, flux_types=src.ro.FLUX_TYPES, H=H_ML, remove_median=case["remove_median"]
)
## with XRO's own fit (R's predictors and instruments), with the same form as
## R (harmonics, flat nonlinear terms if R's are), or one fit per calendar
## month as in the paper drafts
alpha = {}
for flux_type in src.ro.FLUX_TYPES:
    if case["alpha_xro"]:
        flux_fits = src.ro.fit_flux_feedback(
            data[src.ro.VARNAMES], Q[flux_type], model, fit_kwargs
        )
        alpha[flux_type] = src.ro.get_alpha(flux_fits)
    elif case["alpha_harmonic"]:
        alpha[flux_type] = src.ro.fit_alpha_harmonic(
            data["T_3"],
            Q[flux_type],
            cycle=fits["cycle"],
            ac_order=model.ac_order,
            flat_nonlinear=case["flat_nonlinear"],
        )
    else:
        alpha[flux_type] = src.ro.fit_alpha(
            data["T_3"], Q[flux_type], cycle=fits["cycle"]
        )

## save the feedbacks, one variable per flux
xr.Dataset(alpha).to_netcdf(f"{out_dir}/alpha.nc")

## the experiments' parameters: terms fixed to their REF_YEAR values
terms = src.ro.RO_TERMS
fits_fixed = dict(
    noise=src.ro.fix_terms(fits, terms["noise"], ref_year=REF_YEAR),
    R=src.ro.fix_terms(fits, terms["R"], ref_year=REF_YEAR),
    F1=src.ro.fix_terms(fits, terms["F1"], ref_year=REF_YEAR),
    F2_eps=src.ro.fix_terms(fits, terms["F2"] + terms["eps"], ref_year=REF_YEAR),
)

## the ocean part of R, R_o = R + alpha (net flux), without its changes
part_Ro = src.ro.get_R_coefs(fits) + alpha["nhf"].sel(param=src.ro.R_PARAMS)
fits_fixed["Ro"] = src.ro.remove_change_in_R(fits, part_Ro, ref_year=REF_YEAR)

## each flux's feedback (part -alpha of R, and of F1 if alpha has an h term)
## without its changes
for flux_type in src.ro.FLUX_TYPES:
    fits_fixed[f"alpha_{flux_type}"] = src.ro.remove_change_in_R(
        fits, -alpha[flux_type], ref_year=REF_YEAR
    )

## each experiment's T^3 coefficient (a bound on R doesn't hold for R minus a
## part), dims (experiment, year, cycle)
R3 = []
for fits_exp in fits_fixed.values():
    R3.append(fits_exp["NROT_Lac"].sel(nro_form="T3", drop=True))
experiments = xr.DataArray(list(fits_fixed), dims="experiment")
xr.concat(R3, dim=experiments).to_netcdf(f"{out_dir}/experiments_R3.nc")

## simulation settings: this case's noise scale
sim_kwargs = dict(src.ro.SIM_KWARGS, noise_scale=case["noise_scale"])

## initial condition, as in the fig scripts
X0 = data[src.ro.VARNAMES].isel(year=0, member=0, time=0)

## control: each window's fitted parameters
sims = src.ro.simulate_over_time(model, fits, X0, sim_kwargs=sim_kwargs)
save_sims(sims, f"{out_dir}/sims_ctrl.nc", src.ro.VARNAMES)
print("simulated: ctrl", flush=True)

## experiments: the perturbed parameters (T only, to save space)
for name, fits_exp in fits_fixed.items():
    sims = src.ro.simulate_over_time(model, fits_exp, X0, sim_kwargs=sim_kwargs)
    save_sims(sims, f"{out_dir}/sims_{name}.nc", ["T_3"])
    print(f"simulated: {name}", flush=True)

print("done", flush=True)
