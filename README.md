# Recharge oscillator change over time
How do recharge oscillator parameters change over time in a large ensemble? What do these changes correspond to physically? Project started at ENSO 2025 winter school.

## Set-up
1. Clone/fork repository  (```git clone ...```)
2. Create virtual environment  (```mamba create ...```)
3. Activate virtual environment (```mamba activate ...```)
4. Install packages listed in ```env.yml``` file to virtual environment (```mamba env update --file env.yml```)
5. Install custom module to virtual environment (run ```pip install -e .``` in project's home directory)
6. Download ORAS5 and MPI data from Google Drive folder (contact for access to shared folder)
7. Set filepaths for data and results, ```DATA_FP``` and ```SAVE_FP``` (```mamba env config vars set DATA_FP=...```)


## Description of project directory
- ```scripts```: code used for analysis/plots
  - ```cesm2```: analysis of the CESM2 large ensemble (the main focus of the paper; see below)
    - ```preprocess```: build $T$, $h$ indices and consolidated (EOF-compressed) fields from raw CESM2-LENS output
    - ```explore```: exploratory notebooks not needed to reproduce the paper's results
  - ```mpi```: analysis of the MPI grand ensemble (earlier phase of the project)
  - ```obs```: analysis and download of observational/reanalysis data
  - ```cesm1```: RO change over time in CESM1
  - ```lme```: analysis of the last millennium ensemble (separate side project)
  - ```scratch```: old experiments
- ```data```: raw and preprocessed data (*contact for access*)
- ```src```: custom module containing functions used in ```scripts```
- ```results```: figures and intermediate results (e.g., model parameters)
- ```tests```: check that certain functions work as expected (e.g., EOF reconstructions)

## Key CESM2 scripts
Listed in the order they should be run. Paths are relative to ```scripts/cesm2```.
```DATA_FP``` and ```SAVE_FP``` are environment variables (see Set-up).

### 1. Preprocessing (writes intermediate data)
- ```preprocess/preprocess_cesm.ipynb```
  - What it does: computes $T$, $h$ indices and anomalies (ensemble mean removed) from raw CESM2-LENS output
  - Main outputs: ```DATA_FP/cesm/{Th,Th_anom,cvdp_anom,trop_sst,eli}.nc```, per-member flux files
- ```preprocess/preprocess_ocean.ipynb```
  - What it does: extracts per-member ocean budget variables (velocities, temperature, tendency)
  - Main outputs: per-member ocean budget files (used by ```consolidate```)
- ```preprocess/consolidate.ipynb```
  - What it does: compresses spatial fields with EOFs; splits them into forced and anomaly components
  - Main outputs: ```DATA_FP/cesm/consolidated/{forced,anom,budg_*,flux_*_with_clearsky}.nc```
- ```preprocess/compute_h.ipynb```
  - What it does: estimates thermocline depth
  - Main outputs: ```results/h_ests/```
- ```preprocess/get_T_ml.ipynb```
  - What it does: computes mixed-layer temperature for each member
  - Main outputs: ```DATA_FP/cesm/Th_3D/```
- ```preprocess/merge_Th_3D.ipynb```
  - What it does: merges the output of ```get_T_ml```
  - Main outputs: ```DATA_FP/cesm/{Th_3D,Th_50,Th_80}.nc```
- ```scale.ipynb```
  - What it does: computes background-state scaling quantities (zonal SST gradient, $\bar{H}$, $\alpha$)
  - Main outputs: ```results/cesm_{dTdx_Tw_v3,Hbar_scale_v2,alpha_scale}.nc```

### 2. Figures
Figure labels refer to ```paper-outline.tex``` in the ch3 Overleaf project.
- ```CESM_climate-change_fig.ipynb```
  - What it does: plots the mean state and its change under SSP3-7.0 (maps, depth sections)
  - Paper figures: ```fig:cesm2-meanstate``` (a–c)
- ```timeseries.ipynb```
  - What it does: plots mean-state timeseries and El Niño / La Niña contributions to variance
  - Paper figures: ```fig:cesm2-meanstate``` (d–g), ```fig:variance-changes``` (b, c)
- ```stats-changes.ipynb```
  - What it does: computes ENSO statistics over time and heat flux vs. relative SST
  - Paper figures: ```fig:variance-changes``` (a), ```fig:nhf_scatter```, ```fig:nhf-scatter```
- ```composite-over-time.ipynb```
  - What it does: computes El Niño composites in a sliding window, and Hovmöllers of SST tendency and heat flux
  - Paper figures: ```fig:composite-nino``` (spatial), ```fig:ddt-hov```, ```fig:nhf-hov```
- ```spaghetti_updated.ipynb```
  - What it does: plots event timeseries and the warm/cold growth rate and heat-flux feedback over time
  - Paper figures: ```fig:composite-nino``` (scalar), ```fig:seasonal-ann_comp```
- ```RO-change-over-time_NL.ipynb```
  - What it does: fits the nonlinear RO over time; validation, growth rate, and parameter perturbation experiments
  - Paper figures: ```fig:RO-validation```, ```fig:gr-over-time```, ```fig:RO-experiments*```, ```fig:RO-experiments_param_hov```
- ```nhf_validation.ipynb```
  - What it does: compares the heat flux feedback in ERA5 and CESM2
  - Paper figures: ```fig:Q-comp```, ```fig:Q-structure_era5```, ```fig:Q-structure_cesm2```

### 3. Supporting analyses (not in the current outline)
- ```physics-bjerknes_asym_precompute.ipynb```
  - What it does: precomputes data for ```physics-bjerknes_asym```
  - Main outputs: ```DATA_FP/bjerknes_ML_80/```
- ```physics-bjerknes_asym.ipynb```
  - What it does: decomposes the asymmetric (warm vs. cold) growth rate and heat flux
- ```alpha-over-time.ipynb```
  - What it does: estimates heat flux damping $\alpha$ over time from surface flux components and cloud fraction

### Known issues (to fix when restructuring scripts)
- Figure-saving calls are set to ```do_save=False```; paper figures were copied into the Overleaf project by hand.
- ```spaghetti_updated.ipynb``` (and ```explore/eli-change-over-time.ipynb```, ```explore/powerTransform.ipynb```) read ```results/cesm_dTdx_Tw.nc```, which no longer exists; ```scale.ipynb``` now writes ```results/cesm_dTdx_Tw_v3.nc```.

## Casper login
1. Go to jupyterhub.hpc.ucar.edu
2. Login with "UCAR-CIT" credentials (omit ```@ucar.edu```)
3. Switch to environment: ```Python [conda env:envs]```

## Data sources
CVDP data obtained from ```/glade/campaign/cesm/development/cvcwg/cvwg/CVDP-LE_Data_Repository/cesm2-lens_1850-2100.cvdp_data.tar```
