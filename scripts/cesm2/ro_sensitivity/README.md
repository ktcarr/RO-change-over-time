# RO sensitivity tests

Tests of the recharge oscillator (RO) settings used in the paper's RO figures
(fig:RO-validation, fig:gr-over-time, fig:RO-experiments*). They're here so the choices can
be checked again and justified, e.g. in a supplement.

## Approach

The settings are chosen a priori where possible, preferring a simple, self-consistent model
over one that matches CESM2 more closely:

- **Lag taus = 2** for the fit's covariances. The fit uses centered differences, and with
  taus = 1 the lagged state overlaps the difference, so the two share noise and the fit is
  biased (too little damping).
- **Mean removed, with a constant fitted.** In the mean of the T equation,
  0 = c + L⟨x⟩ + b⟨T²⟩ + d⟨T³⟩: the quadratic and cubic terms have a nonzero mean, and the
  constant c balances it. Without it, the fit biases the nonlinear terms and the simulations
  drift. With the median removed instead, ⟨T⟩ ≠ 0 partly substitutes for c.
- **The constant has no seasonal cycle.** With one, it overfits (12 more parameters) and takes
  over some of the role of the T² term.
- **Noise from `ro_tools.fit.set_fit_noise`**, which corrects the fit residuals' variance for
  their memory. This replaces the hand-tuned noise factor of 1.45 (the implied factors are
  ~1.5).
- **The heat flux feedback α has the same functional form as R**
  (`src.ro.fit_alpha_harmonic`): a linear coefficient with 3 harmonics, and nonlinear
  coefficients that are flat exactly when R's are. α is one least-squares fit per window,
  pooling all months, so R_o = R + α has R's form. Averaging α's monthly fits over the year
  instead isn't equivalent: least squares weights months by how much they constrain each
  coefficient. Using harmonics instead of 12 separate months changes α by < 0.005. Flattening
  the nonlinear terms barely changes the net flux's α. It does change some components' changes
  from 1870 to 2080, e.g. shortwave α₂ goes from −0.07 to −0.02 and latent α₃ from −0.00 to
  −0.04.
- **Diagnostic: warm and cold contributions to the variance**
  (`src.stats.get_variance_contributions`, as in fig:variance-changes). They're measured from
  each window's own median, so they don't depend on how the data were centered or on any drift
  in the simulations. The earlier amplitudes (95th and minus the 5th percentile of DJF T,
  measured from zero) did.

Still to decide, as a 2×2:

- **Seasonal cycle on the nonlinear terms** (`nlseas`) or not (`nlflat`).
- **ĥ from one regression of h on T per calendar month** (`hmonth`) or one for all months
  (`hconst`). The monthly regression gives R₁ a clean meaning, but it removes all of h's lag-0
  correlation with T in each month, including correlation produced by the dynamics (e.g. the
  seasonal and nonlinear terms). `roundtrip.py` tests what each method does to the fit when the
  true RO is known.

## Cases (`cases.py`)

| tag | taus | centering | constant | nonlinear terms | ĥ | noise |
|---|---|---|---|---|---|---|
| `published` | 1 | median | none | seasonal | monthly | residuals × 1.45 |
| `mean_noconst` | 2 | mean | none | seasonal | monthly | set_fit_noise |
| `nlseas_hmonth` | 2 | mean | flat | seasonal | monthly | set_fit_noise |
| `nlseas_hconst` | 2 | mean | flat | seasonal | one regression | set_fit_noise |
| `nlflat_hmonth` | 2 | mean | flat | flat | monthly | set_fit_noise |
| `nlflat_hconst` | 2 | mean | flat | flat | one regression | set_fit_noise |

## Round 2: seasonal order, T³ bound, α from XRO, forward differences

The seasonal nonlinear terms (`nlseas`) match CESM2's skewness much better than flat ones,
but the fitted R₃ is strongly seasonal: about −1.6 in June and ≈ 0, even slightly positive,
in Dec–Jan. A study in `~/dlesym-pacemaker_intermodel` (ro_tools `docs/notes.md`, and that
repo's `docs/archive/TODO/log.md`) found that seasonal nonlinear terms are poorly
constrained there: the lagged T² is a weak instrument after the spring barrier (its
correlation with its value 2 months earlier falls from 0.96 in January to 0.37 in July).
Round 2 tests, keeping the seasonal nonlinear terms and one ĥ regression (`hconst`):

- **One seasonal order for everything** (`ac_order` = 3, 2, 1): the operator, the nonlinear
  terms, the noise (`set_fit_noise`) and α. The constant stays flat.
- **R₃ ≤ 0 in every month** (`_neg`; `ro_tools.validate.fit_with_T3_bound`, extended to a
  seasonal T³). A priori: a positive cubic term isn't bounded, so the RO isn't globally
  stable. α isn't bounded: it only splits R, and a flux's cubic term may have either sign.
  R_o = R + α is the rest, so it absorbs the bound's effect; `tables_round2.txt` shows
  both that effect and whether R₃ goes positive in the experiments.
- **α from XRO's own fit** (`alpha_xro`, `src.ro.fit_flux_feedback`): the flux's warming
  rate in place of dT/dt, with R's predictors (T, ĥ, T², T³, constant), harmonics and lag-2
  instrument. The fit is linear in the tendency, so R = R_o − α exactly (except where the
  bound binds). With lagged T alone as instrument, the other repo's flux part drifted with
  the lag (−1.06 to −0.78 yr⁻¹ from lag 0 to 2), a weak-instrument effect that went away
  with ĥ among the instruments. α now has an ĥ term (the flux's part of F₁), which the α
  experiments also fix.
- **Forward differences with a lag-1 instrument** (`fwd_`), a stronger instrument after
  the spring barrier; Q is averaged to the half step.
- The round trip (`roundtrip.py`) refits a known RO, so `tables_round2.txt` also shows
  whether the seasonal cycles of R₂ and R₃ are recovered at this sample size.

| tag | differences, lag | ac_order | R₃ ≤ 0 |
|---|---|---|---|
| `ac3`, `ac2`, `ac1` | centered, 2 | 3, 2, 1 | no |
| `ac3_neg`, `ac2_neg`, `ac1_neg` | centered, 2 | 3, 2, 1 | yes |
| `fwd_ac3`, `fwd_ac1` | forward, 1 | 3, 1 | no |
| `fwd_ac3_neg`, `fwd_ac1_neg` | forward, 1 | 3, 1 | yes |

`nlseas_hconst` (round 1) is `ac3` with the earlier α (one lag-0 least-squares fit on T).
The round-2 cache is on scratch (`results/ro_sensitivity/<tag>` are symlinks to
`/glade/derecho/scratch/kcarr/ro_sensitivity/<tag>`), because home was full.

## Running

From the repo root (one PBS job per case; each takes about an hour):

```
qsub -o /glade/derecho/scratch/kcarr/ro_sensitivity/logs/ scripts/cesm2/ro_sensitivity/run_cases.pbs
```

(by default round 2, `-J 6-15`; `-J 0-5` for round 1).

Each job runs `simulate_case.py <tag>` and then `roundtrip.py <tag>`. They cache the fits,
the control simulations and the experiment simulations in `SAVE_FP/ro_sensitivity/<tag>/`.
Then, on the current node (from `scripts/cesm2`):

```
python ro_sensitivity/diagnostics.py
```

reads only the cache and writes, for each comparison group of `cases.GROUPS` (`round1`,
`round2`, `round2_fwd`), `compare_<group>`, `experiments_<group>`, `baseline_<group>` and
`params_<group>` (png and pdf) and `tables_<group>.txt` to `SAVE_FP/ro_sensitivity/`.
Changing a diagnostic means rerunning only this script. `python ro_sensitivity/alpha_forms.py`
compares the ways of estimating α (`alpha_forms.png`).

## Conclusions (so far; Nov 22, 2026)

Numbers from `tables_round1.txt`, `tables_round2.txt`, `tables_round2_fwd.txt`.

**Round 1**
- Seasonal nonlinear terms (`nlseas`) match CESM2's skewness by month much better than flat
  ones (`nlflat`), and with flat terms the noise plays a bigger part in the early increase
  of El Niño. The late decrease's attribution doesn't depend on the form.
- ĥ from one regression (`hconst`) mainly changes the attribution, not the simulations; the
  round trip shows the monthly regression (`hmonth`) biases R₁ and ε (1870: R₁ 0.29 vs
  0.73 true, ε 1.09 vs 1.65).
- α with harmonics instead of 12 monthly fits changes α by < 0.005.

**Round 2** (all `nlseas`, `hconst`)
- **The R₃ ≤ 0 bound is nearly free:** it binds in 0–3 months per window, by small amounts;
  fit quality, experiment effects and R_o change by ≤ 0.07. It doesn't keep R₃ ≤ 0 inside
  the experiments (R − ΔR_o, R − Δα): there R₃ reaches 0.6 in some months for `ac3`,
  0.08 for `ac1`.
- **Fewer harmonics:** `ac1` fits σ and the warm contribution better (RMSE 0.065 vs 0.085,
  0.115 vs 0.141) and skewness a little worse (0.10 vs 0.085).
- **The skewness phase problem is late-century only:** the first harmonic of skew(T) by
  month peaks in February in 1870 in CESM2 and every seasonal case; in 2080 CESM2 peaks
  in Oct–Nov and every seasonal case (any order, either differences) in Mar–Apr. The flat
  cases of round 1 get 2080 right and 1870 wrong.
- **Weak instruments matter at ac_order 3:** in the round trip (refitting a known RO at
  CESM2's sample size), summer values come back biased (June R₂ 0.51 vs 0.20 true, July
  R₃ −1.59 vs −1.24, beyond the spread over seeds). At ac_order 1 R₂ and R₃ are recovered
  within 0.07 in every month. Forward differences at order 3 underestimate summer R₃
  (−0.53 vs −0.76).
- **α from XRO's fit** (`alpha_forms.png`): α₁ about the same, α₂ higher, the late decline
  of the net α₃ gone, and a large ĥ term (net ≈ 0.8; latent ≈ 0.5, shortwave 0.2–0.33).
  It changes the attribution: the net flux's effect on La Niña in 2080 goes from −0.39 to
  −0.63 K², R_o's on El Niño from 0.12 to 0.06 K² (`ac3` vs `nlseas_hconst`). Part of it
  is that the α experiments now also fix the flux's ĥ term.
- **Forward differences, lag 1:** σ about 0.06 K too low throughout, but the trend of the
  cold contribution is better (0.19–0.25 K² vs CESM2's 0.24; centered 0.31–0.43); smaller
  flux effects (net −0.51 / −0.38 K² in 2080).

**Suggested:** ac_order 1 for everything, with the R₃ bound (`ac1_neg` or `fwd_ac1_neg`):
the simplest form, recovered by the round trip, and the bound costs nothing.

**Still open:** centered (lag 2) vs forward (lag 1) differences; whether α keeps its ĥ
term (or an experiment separating it); what causes the late-century skewness phase
mismatch. Then step 2 of the plan (switch the `src/ro.py` defaults, contributions in the
fig scripts, rerun the RO figures) and step 3 (ch3 methods text).
