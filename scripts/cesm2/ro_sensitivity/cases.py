"""
The RO settings compared in the sensitivity tests (see README.md). Each case is
a dict of the settings that differ between cases; simulate_case.py turns them
into arguments of the src/ro.py functions. Only settings live here, no code.

Settings:
    - taus (int): lag of the fit's covariances (src.ro.get_model)
    - remove_median (bool): remove each window's median (True) or mean (False)
      for each calendar month (src.ro.load_ro_data, load_ro_flux_data)
    - h_by_month (bool): regress h on T for each calendar month (True) or once
      for all months (False) to get h_hat (src.ro.load_ro_data)
    - fit_constant (bool): fit a constant in each equation (fit_matrix)
    - flat_constant (bool): no seasonal cycle on the constant
    - flat_nonlinear (bool): no seasonal cycle on the nonlinear terms
    - fit_noise (bool): memory-corrected noise from ro_tools.fit.set_fit_noise
      (src.ro.fit_over_time)
    - noise_scale (float): factor on the noise in the simulations
    - alpha_harmonic (bool): fit the heat flux feedback alpha with the same
      form as R, i.e. harmonics and, with flat_nonlinear, flat nonlinear terms
      (src.ro.fit_alpha_harmonic), so R_o = R + alpha has R's form; otherwise
      one fit per calendar month (src.ro.fit_alpha)
    - ac_order (int): number of harmonics of every seasonal coefficient (the
      operator, the nonlinear terms, the noise and alpha) (src.ro.get_model)
    - is_forward (bool): forward time differences (else centered)
    - nonpositive_T3 (bool): bound the T^3 coefficient of the T equation at zero
      in every month, so the cubic term always damps (src.ro.fit_over_time)
    - alpha_xro (bool): fit alpha with XRO itself, the flux's warming rate in
      place of dT/dt with R's predictors, harmonics and lagged covariances
      (src.ro.fit_flux_feedback), so R = R_o - alpha exactly and alpha has an h
      term (the flux's part of F1); overrides alpha_harmonic
"""

## the settings of the paper drafts so far: lag 1, median removed, no constant,
## noise from the fit's residuals scaled up by a hand-tuned factor
PUBLISHED = dict(
    taus=1,
    remove_median=True,
    h_by_month=True,
    fit_constant=False,
    flat_constant=False,
    flat_nonlinear=False,
    fit_noise=False,
    noise_scale=1.45,
    alpha_harmonic=False,
    ac_order=3,
    is_forward=False,
    nonpositive_T3=False,
    alpha_xro=False,
)


## settings chosen a priori: lag 2 (no shared noise with centered differences),
## mean removed and a constant fitted (closes the mean balance, so no drift),
## the constant without a seasonal cycle, memory-corrected noise, and alpha
## with the same functional form as R
BASE = dict(
    taus=2,
    remove_median=False,
    fit_constant=True,
    flat_constant=True,
    fit_noise=True,
    noise_scale=1.0,
    alpha_harmonic=True,
    ac_order=3,
    is_forward=False,
    nonpositive_T3=False,
    alpha_xro=False,
)


## round 2 (README): seasonal nonlinear terms and one h_hat regression, alpha
## from XRO's own fit, and the same number of harmonics for everything; varied:
## the number of harmonics, the T^3 bound, and centered differences with lag 2
## or forward differences with lag 1 (a stronger instrument after the spring
## barrier)
ROUND2 = dict(BASE, flat_nonlinear=False, h_by_month=False, alpha_xro=True)
FORWARD = dict(ROUND2, taus=1, is_forward=True)


## all cases, by tag; the four "nl*_h*" cases are the 2x2 still to decide:
## seasonal cycle on the nonlinear terms (nlseas) or not (nlflat), and h_hat
## from monthly regressions (hmonth) or one regression (hconst)
CASES = dict(
    published=PUBLISHED,
    ## reference: the a priori settings but without the constant
    mean_noconst=dict(
        BASE,
        fit_constant=False,
        flat_constant=False,
        flat_nonlinear=False,
        h_by_month=True,
    ),
    nlseas_hmonth=dict(BASE, flat_nonlinear=False, h_by_month=True),
    nlseas_hconst=dict(BASE, flat_nonlinear=False, h_by_month=False),
    nlflat_hmonth=dict(BASE, flat_nonlinear=True, h_by_month=True),
    nlflat_hconst=dict(BASE, flat_nonlinear=True, h_by_month=False),
    ## round 2: centered differences, lag 2
    ac3=dict(ROUND2, ac_order=3),
    ac2=dict(ROUND2, ac_order=2),
    ac1=dict(ROUND2, ac_order=1),
    ac3_neg=dict(ROUND2, ac_order=3, nonpositive_T3=True),
    ac2_neg=dict(ROUND2, ac_order=2, nonpositive_T3=True),
    ac1_neg=dict(ROUND2, ac_order=1, nonpositive_T3=True),
    ## round 2: forward differences, lag 1
    fwd_ac3=dict(FORWARD, ac_order=3),
    fwd_ac1=dict(FORWARD, ac_order=1),
    fwd_ac3_neg=dict(FORWARD, ac_order=3, nonpositive_T3=True),
    fwd_ac1_neg=dict(FORWARD, ac_order=1, nonpositive_T3=True),
)


## the comparisons shown together in diagnostics.py, by name
GROUPS = dict(
    round1=[
        "published",
        "mean_noconst",
        "nlseas_hmonth",
        "nlseas_hconst",
        "nlflat_hmonth",
        "nlflat_hconst",
    ],
    round2=["nlseas_hconst", "ac3", "ac2", "ac1", "ac3_neg", "ac2_neg", "ac1_neg"],
    round2_fwd=["ac3", "ac1", "fwd_ac3", "fwd_ac1", "fwd_ac3_neg", "fwd_ac1_neg"],
)
