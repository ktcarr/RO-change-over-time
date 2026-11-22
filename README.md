# Recharge oscillator change over time
How do recharge oscillator parameters change over time in a large ensemble? What do these changes correspond to physically? Project started at ENSO 2025 winter school.

## Set-up
1. Clone/fork repository  (```git clone ...```)
2. Create virtual environment  (```mamba create ...```)
3. Activate virtual environment (```mamba activate ...```)
4. Install packages listed in ```env.yml``` file to virtual environment (```mamba env update --file env.yml```)
5. Install custom module to virtual environment (run ```pip install -e .``` in project's home directory)
6. Install the RO package [ro_tools](https://github.com/ktcarr/ro_tools) the same way, without dependencies (they come from ```env.yml```): ```pip install --no-deps -e <path to ro_tools>```
6. Download ORAS5 and MPI data from Google Drive folder (contact for access to shared folder)
7. Set filepaths for data and results, ```DATA_FP``` and ```SAVE_FP``` (```mamba env config vars set DATA_FP=...```)


## Description of project directory
- ```scripts```: code used for analysis/plots
  - ```cesm2```: analysis of the CESM2 large ensemble (the main focus of the paper; see below)
    - ```preprocess```: build $T$, $h$ indices and consolidated (EOF-compressed) fields from raw CESM2-LENS output
    - ```fig_scripts```: one script per paper (or thesis) figure, named after its LaTeX label
    - ```explore```: exploratory notebooks not needed to reproduce the paper's results
  - ```mpi```: analysis of the MPI grand ensemble (earlier phase of the project)
  - ```obs```: analysis and download of observational/reanalysis data
  - ```cesm1```: RO change over time in CESM1
  - ```lme```: analysis of the last millennium ensemble (separate side project)
  - ```scratch```: old experiments
- ```data```: raw and preprocessed data (*contact for access*)
- ```src```: custom module containing functions used in ```scripts```, organized by topic:
  - ```constants.py```: region boundaries and physical constants
  - ```data.py```: loading data, unit conversions, windows, and seasonal means
  - ```indices.py```: averages over the paper's regions
  - ```stats.py```: changes over time, composites, and regressions
  - ```plot.py```: saving figures, maps, region boxes, and Hovmöller grids
  - ```ro.py```: the recharge oscillator (RO) fitting and simulation steps shared by the RO figures; uses the RO model from ```ro_tools``` (```ro_tools.xro.XRO```), which with these settings reproduces the old copy in ```XRO.py``` to roundoff (```XRO.py``` stays for the scripts in ```explore/``` and ```src/utils.py```)
  - ```utils.py```: older general utilities, used by the scripts not yet split up (to be moved into the modules above)
- ```results```: figures and intermediate results (e.g., model parameters)
- ```tests```: check that certain functions work as expected (e.g., EOF reconstructions)

## Key CESM2 scripts
Listed in the order they should be run. Paths are relative to ```scripts/cesm2```.
```DATA_FP``` and ```SAVE_FP``` are environment variables (see Set-up).

### Running the scripts
The key scripts below are [marimo](https://marimo.io) notebooks, saved as plain ```.py``` files. (The exploratory notebooks in ```explore/``` and the other ```scripts/``` folders are still Jupyter ```.ipynb``` files.)
- Interactive: ```marimo edit <script>.py``` (run from ```scripts/cesm2```)
- Headless: ```MPLBACKEND=Agg python <script>.py``` runs every cell top to bottom

### 1. Preprocessing (writes intermediate data)
- ```preprocess/preprocess_cesm.py```
  - What it does: computes $T$, $h$ indices and anomalies (ensemble mean removed) from raw CESM2-LENS output
  - Main outputs: ```DATA_FP/cesm/{Th,Th_anom,cvdp_anom,trop_sst,eli}.nc```, per-member flux files
- ```preprocess/preprocess_ocean.py```
  - What it does: extracts per-member ocean budget variables (velocities, temperature, tendency)
  - Main outputs: per-member ocean budget files (used by ```consolidate```)
- ```preprocess/consolidate.py```
  - What it does: compresses spatial fields with EOFs; splits them into forced and anomaly components
  - Main outputs: ```DATA_FP/cesm/consolidated/{forced,anom,budg_*,flux_*_with_clearsky}.nc```
- ```preprocess/compute_h.py```
  - What it does: estimates thermocline depth
  - Main outputs: ```results/h_ests/```
- ```preprocess/get_T_ml.py```
  - What it does: computes mixed-layer temperature for each member
  - Main outputs: ```DATA_FP/cesm/Th_3D/```
- ```preprocess/merge_Th_3D.py```
  - What it does: merges the output of ```get_T_ml```
  - Main outputs: ```DATA_FP/cesm/{Th_3D,Th_50,Th_80}.nc```
- ```scale.py```
  - What it does: computes background-state scaling quantities (zonal SST gradient, $\bar{H}$, $\alpha$)
  - Main outputs: ```results/cesm_{dTdx_Tw_v3,Hbar_scale_v2,alpha_scale}.nc```

### 2. Figures
Figure labels refer to ```paper-outline.tex``` in the ch3 Overleaf project.

#### Figure scripts (```fig_scripts/```)
The scripts are being split up so that each figure comes from one script in ```fig_scripts/```, named after its LaTeX label. Thesis-only figures are ```fig_scripts/thesis_<label>.py```, named after the label in the thesis LaTeX (they save to ```SAVE_FP/figures/thesis_<label>.pdf```). Every figure script:
- states the figure, its panels, its inputs, and its output in the first cell;
- loads the data, computes each plotted quantity in its own cell, and draws one figure;
- saves the figure to ```SAVE_FP/figures/<label>.pdf``` (saving is on by default);
- uses the shared functions in ```src``` (see "Description of project directory"; tested in ```tests/test_<module>.py```); the RO figures share the fitting and simulation steps in ```src/ro.py```.

Run one from ```scripts/cesm2```, e.g. ```MPLBACKEND=Agg python fig_scripts/cesm2-meanstate.py```.

ENSO statistics use the SST-based Niño 3.4 index (```Th_anom.nc```, or ```Th.nc``` where the total is needed, as for relative SST), the one the RO fits use. The setting ```T34_INDEX``` at the top of a script switches to the mixed-layer index (```"ml50"``` or ```"ml80"```, from ```Th_50.nc```/```Th_80.nc```). With ```"ml80"```, the scripts reproduce the older notebooks exactly, except for the PDFs in ```fig:variance-changes``` (a), which now drop each window's partial first and last DJF seasons, as (b, c) do, and the ```fig:nhf_scatter``` fits, which change by less than 1e-5 K/yr after an unused fit parameter was dropped.

```fig:ddt-hov``` and ```fig:nhf-hov``` divide by the composite Niño 3.4 index in K; the old figures divided by the standardized index (standard deviation 1.26 K), so their normalized panels were larger by that factor. Both keep the paper's single-year events (```EVENT_TYPE = 1```), drop the 2090 window (which ran past 2100), and are each drawn as one figure where the halves used to be joined by hand.

```fig:seasonal-ann_comp``` is drawn as one figure (the two rows used to be joined by hand), with windows centered on 1870, ..., 2080; the old script dropped 1850, so its windows were centered on 1871, ..., 2081. On the same data, the new code reproduces the old growth rates and heat flux slopes exactly.

```fig:RO-validation``` shows the Niño 3 index in K: the RO is fitted to the index divided by its standard deviation (1.23 K), and the old panels plotted those non-dimensional values with a "K" label.

```fig:composite-nino``` now uses the same windows (2010, 2080) and events in all panels; (d, e) show percentiles of the difference between a random 2080 event and a random 2010 event, computed from all pairs of events in place of one arbitrary pairing (see the script's first cell).

| Script | Figure | Replaces |
| --- | --- | --- |
| ```fig_scripts/cesm2-meanstate.py``` | ```fig:cesm2-meanstate``` (a–g) | ```explore/CESM_climate-change_fig.py``` (a–c), ```explore/timeseries.py``` (d–g) |
| ```fig_scripts/variance-changes.py``` | ```fig:variance-changes``` (a–c) | ```explore/stats-changes.py``` (a), ```explore/timeseries.py``` (b, c) |
| ```fig_scripts/composite-nino.py``` | ```fig:composite-nino``` (a–e) | ```explore/composite-over-time.py``` (```spatial_comp```, a, b), ```explore/spaghetti_updated.py``` (```scalar_comp```, c–e) |
| ```fig_scripts/nhf_scatter.py``` | ```fig:nhf_scatter``` (a–c); also thesis ```nhf_rel-sst_triple``` | ```explore/stats-changes.py``` (```nhf_rel-sst_triple```) |
| ```fig_scripts/nhf_scatter_by_component.py``` | ```fig:nhf_scatter_by_component``` (formerly ```fig:nhf-scatter```) | ```explore/stats-changes.py``` (cell "Similar but for NHF components") |
| ```fig_scripts/ddt-hov.py``` | ```fig:ddt-hov``` | ```explore/composite-over-time.py``` (cell under "Clean up fig.") |
| ```fig_scripts/nhf-hov.py``` | ```fig:nhf-hov``` | ```explore/composite-over-time.py``` (cell under "Heat flux") |
| ```fig_scripts/RO-validation.py``` | ```fig:RO-validation``` (a–d) | ```explore/RO-change-over-time_NL.py``` (```RO-valid2_T_3```, a, b; ```RO-valid_T_3```, c, d) |
| ```fig_scripts/seasonal-ann_comp.py``` | ```fig:seasonal-ann_comp``` | ```explore/spaghetti_updated.py``` (cells under "Get $R$ over time") |
| ```fig_scripts/gr-over-time.py``` | ```fig:gr-over-time``` (a–c) | ```explore/RO-change-over-time_NL.py``` (```gr-over-time_ann```, cells under "Plot params over time") |
| ```fig_scripts/RO-experiments.py``` | ```fig:RO-experiments``` (a–f) | ```explore/RO-change-over-time_NL.py``` (```exp-comparison_general```, cell under "Check linearity") |
| ```fig_scripts/RO-experiments_Ro.py``` | ```fig:RO-experiments_Ro``` (a–d) | ```explore/RO-change-over-time_NL.py``` (```exp-comparison_Ro```, cell under "Check linearity of subset") |
| ```fig_scripts/RO-experiments_alpha.py``` | ```fig:RO-experiments_alpha``` (a–e) | ```explore/RO-change-over-time_NL.py``` (```exp-comparison_alpha```, second cell under "Check linearity of subset") |
| ```fig_scripts/RO-experiments_param_hov.py``` | ```fig:RO-experiments_param_hov``` (a–d) | ```explore/RO-change-over-time_NL.py``` (```param-hov```, cell under "Cleaned up plot") |
| ```fig_scripts/Q-comp.py``` | ```fig:Q-comp``` | ```explore/nhf_validation.py``` (```Q-validation```, cell under "Make plot") |
| ```fig_scripts/Q-structure_era5.py``` | ```fig:Q-structure_era5``` | ```explore/nhf_validation.py``` (ERA5 cells under "composites") |
| ```fig_scripts/Q-structure_cesm2.py``` | ```fig:Q-structure_cesm2``` | ```explore/nhf_validation.py``` (CESM2 cells under "composites") |
| ```fig_scripts/thesis_mean-state_change.py``` | thesis ```fig:mean-state_change``` (a–f) | ```explore/mean-state-changes.py``` (```mean-state-pattern_change_labeled```, a, b; ```subsurface-change```, c, d; ```mean-state_timeseries```, e; ```nino-indices_overtime```, f) |
| ```fig_scripts/thesis_sigma-over-time.py``` | thesis ```fig:sigma-over-time``` (a–d) | ```explore/mean-state-changes.py``` (```sigma-over-time_hov_simple```, cell under "Variance over time") |
| ```fig_scripts/thesis_sigma-by-season.py``` | thesis ```fig:sigma-by-season``` (a–c) | ```explore/mean-state-changes.py``` (```sigma-cycle-change```, cell under "Variance over time (v2)") |
| ```fig_scripts/thesis_pdf-over-time.py``` | thesis ```fig:pdf-over-time``` (a–c) | ```explore/stats-changes.py``` (```pdf-over-time_bounds```, cells under "similar, but with bounds") |
| ```fig_scripts/thesis_eli.py``` | thesis ```fig:eli``` (a–d) | ```explore/stats-changes.py``` (```eli-comparison```, cells under "ELI" and "Plot temperature as a function of ELI") |
| ```fig_scripts/thesis_seasonal.py``` | thesis ```fig:seasonal``` (a–f) | ```explore/CESM_climate-change.py``` (```seasonal-clim```, cell under "updated, combined fig") |
| ```fig_scripts/thesis_skew_by-season_over-time.py``` | thesis appendix ```fig:skew_by-season_over-time``` (a–c) | ```explore/CESM_climate-change.py``` (```skew_by-season_over-time```, cells under "Nicer plot, for thesis") |
| ```fig_scripts/thesis_skew-over-time.py``` | thesis appendix ```fig:skew-over-time``` (a–d) | ```explore/stats-changes.py``` (```nino34_djf_stats-over-time```, cells under "Variance and skewness over time") |
| ```fig_scripts/thesis_nhf-abs-sst.py``` | thesis appendix ```fig:nhf-abs-sst``` (a–c) | ```explore/stats-changes.py``` (```nhf_abs-sst_triple```, cells under "same, but with total SST") |
| ```fig_scripts/thesis_enso-hov.py``` | thesis ```fig:nino-hov``` and ```fig:nina-hov``` (two figures, a–l each) | ```explore/composite-over-time.py``` (```combined_hov```, ```combined_hov_nina```, cell under "SST") |
| ```fig_scripts/thesis_model-valid.py``` | thesis ```fig:model-valid``` (a–n; formerly three PDFs stacked in LaTeX) | ```explore/CESM-ORAS_validation_v2.py``` (```spatial-valid```, a–h; ```spatial-asym-scatter```, i–k; ```psd_spaghetti```, l–n) |

#### Scripts not yet split up
Once all of a script's figures have moved to ```fig_scripts/```, the script moves unchanged to ```explore/```.
- None: every paper and thesis figure now has its own script in ```fig_scripts/```.

### 3. Thesis figures
The published PhD thesis (chapters ```intro.tex``` and ```chapter_enso-change.tex```, and the appendix ```appendix_enso-change.tex```) used figures made by this repository. Each is now made by a ```fig_scripts/thesis_*.py``` script (or, for ```nhf_rel-sst_triple```, by ```fig_scripts/nhf_scatter.py```); see the table above. Each script's first cell lists where its figure differs from the thesis caption.

### 4. Supporting analyses (not in the current outline)
- ```physics-bjerknes_asym_precompute.py```
  - What it does: precomputes data for ```physics-bjerknes_asym```
  - Main outputs: ```DATA_FP/bjerknes_ML_80/```
- ```physics-bjerknes_asym.py```
  - What it does: decomposes the asymmetric (warm vs. cold) growth rate and heat flux
- ```alpha-over-time.py```
  - What it does: estimates heat flux damping $\alpha$ over time from surface flux components and cloud fraction

### Known issues
- Every paper and thesis figure is saved by its script in ```fig_scripts/``` (to ```SAVE_FP/figures/```). Outside ```fig_scripts/```, ```physics-bjerknes_asym.py``` saves only ```GR_comp``` (its other ```save``` calls have ```do_save=False```), and ```alpha-over-time.py``` saves no figures.
- ```explore/CESM_climate-change.py``` (exploratory; its two thesis figures are now in ```fig_scripts/```) stops headless at the skewness-by-month plot: it plots ```ssh``` and ```z20```, but ```SKEW_VARNAMES``` computes only ```sst```. The cells before it run.
- ```explore/spaghetti_updated.py``` (and ```explore/eli-change-over-time.ipynb```, ```explore/powerTransform.ipynb```) read ```results/cesm_dTdx_Tw.nc```, which no longer exists; ```scale.py``` now writes ```results/cesm_dTdx_Tw_v3.nc```.
- The marimo scripts outside ```fig_scripts/``` (```preprocess/```, the supporting analyses, ```scale.py```, and ```explore/```) don't yet follow the commenting and docstring conventions in ```CLAUDE.md```. They were converted from Jupyter with ```src/convert_marimo.py``` (```envs/bin/python -m src.convert_marimo NOTEBOOK.ipynb OUTPUT.py```), which wraps ```marimo convert``` so that comments survive (tested in ```tests/test_convert_marimo.py```). When a name was reused across cells, the converter numbered the copies (e.g., ```COMP_1```, ```kwargs_2```). Some cells also modify objects created in earlier cells (e.g., ```Th["x"] = ...```), which marimo doesn't track. Headless runs are unaffected because cells run top to bottom, but re-running a single cell in ```marimo edit``` may not update the cells downstream of it.

## Casper login
1. Go to jupyterhub.hpc.ucar.edu
2. Login with "UCAR-CIT" credentials (omit ```@ucar.edu```)
3. Switch to environment: ```Python [conda env:envs]```

## Data sources
CVDP data obtained from ```/glade/campaign/cesm/development/cvcwg/cvwg/CVDP-LE_Data_Repository/cesm2-lens_1850-2100.cvdp_data.tar```
