"""
Compare three ways of estimating the heat flux feedback alpha, for each flux:
    - "monthly": one cubic fit per calendar month (src.ro.fit_alpha, as in the
      paper drafts)
    - "harmonic, seasonal": one fit per window, every coefficient with 3
      harmonics (src.ro.fit_alpha_harmonic), the form of R in the nlseas cases
    - "harmonic, flat": the same, but alpha_2 and alpha_3 without a seasonal
      cycle, the form of R in the nlflat cases
    - "XRO": XRO's own fit of the flux in place of dT/dt, with R's predictors
      (including h_hat), harmonics and lag-2 instrument
      (src.ro.fit_flux_feedback), as in the round-2 cases; it also has an h
      coefficient (the flux's part of F1)
The perturbation experiments use alpha's change since 1870, so the figure shows
each coefficient's annual mean over time, with its seasonal range shaded.

Uses the CESM2 T cached by simulate_case.py for one case (T is the same in all
cases with the mean removed) and loads the fluxes again.

Outputs, in SAVE_FP/ro_sensitivity/: alpha_forms.nc and alpha_forms.png/.pdf

Usage (from scripts/cesm2, with DATA_FP and SAVE_FP set):
    python ro_sensitivity/alpha_forms.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import xarray as xr

import src.ro


## the case whose cached T is used (mean removed, as in all the new cases)
TAG = "nlflat_hconst"

## the forms compared, and their line colors
FORMS = ["monthly", "harmonic, seasonal", "harmonic, flat", "XRO"]
COLORS = ["k", "tab:blue", "tab:orange", "tab:green"]

## readable names of the fluxes and coefficients
FLUX_NAMES = dict(nhf="net", fsns="shortwave", lhflx="latent", flns="longwave")
PARAM_NAMES = dict(
    T1=r"$\alpha_1$", T2=r"$\alpha_2$", T3=r"$\alpha_3$", h=r"$\alpha_h$"
)


## the cached CESM2 data and the fits' seasonal-cycle coordinate
out_dir = f"{os.environ['SAVE_FP']}/ro_sensitivity"
data = xr.open_dataset(f"{out_dir}/{TAG}/data.nc")
cycle = xr.open_dataset(f"{out_dir}/{TAG}/fits.nc")["cycle"]
T = data["T_3"].load()

## the round-2 RO's model and fit settings (seasonal nonlinear terms), for the
## XRO form
model = src.ro.get_model(taus=2)
fit_kwargs = src.ro.get_fit_kwargs(fit_constant=True, flat_constant=True)

## the fluxes, centered like T (mean removed)
with warnings.catch_warnings(action="ignore"):
    Q = src.ro.load_ro_flux_data(data.attrs["T_scale"], remove_median=False).load()

## alpha for each flux and form, dims (form, year, cycle, param)
alpha = xr.Dataset()
for flux_type in src.ro.FLUX_TYPES:
    a_monthly = src.ro.fit_alpha(T, Q[flux_type], cycle)
    a_seas = src.ro.fit_alpha_harmonic(T, Q[flux_type], cycle, flat_nonlinear=False)
    a_flat = src.ro.fit_alpha_harmonic(T, Q[flux_type], cycle, flat_nonlinear=True)
    flux_fits = src.ro.fit_flux_feedback(
        data[src.ro.VARNAMES].load(), Q[flux_type], model, fit_kwargs
    )
    a_xro = src.ro.get_alpha(flux_fits).assign_coords(cycle=cycle)

    ## stack the forms (only XRO's has an h coefficient; NaN for the others)
    forms = xr.DataArray(FORMS, dims="form")
    alpha[flux_type] = xr.concat(
        [a_monthly, a_seas, a_flat, a_xro], dim=forms, join="outer"
    )
alpha.to_netcdf(f"{out_dir}/alpha_forms.nc")

## one row per flux, one column per coefficient
fig, axs = plt.subplots(4, 4, figsize=(13, 10), sharex=True, layout="constrained")
for i, flux_type in enumerate(src.ro.FLUX_TYPES):
    for j, param in enumerate(src.ro.FLUX_PARAMS):
        ax = axs[i, j]

        ## each form: annual mean as a line, seasonal range shaded
        for form, color in zip(FORMS, COLORS):
            a = alpha[flux_type].sel(form=form, param=param)
            ax.plot(a.year, a.mean("cycle"), c=color, label=form)
            ax.fill_between(
                a.year, a.min("cycle"), a.max("cycle"), color=color, alpha=0.1, lw=0
            )

        ## zero line, and labels on the outer panels
        ax.axhline(0, c="gray", lw=0.5)
        ax.set_title(f"{FLUX_NAMES[flux_type]}, {PARAM_NAMES[param]}", fontsize=10)

## legend once, and the x label on the bottom row
axs[0, 0].legend(fontsize=8)
for ax in axs[-1]:
    ax.set_xlabel("window center")

## save as png and pdf
for ext in ["png", "pdf"]:
    fig.savefig(f"{out_dir}/alpha_forms.{ext}", dpi=150)
print("done", flush=True)
