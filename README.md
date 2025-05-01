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
  - ```preprocess.ipynb```: remove external forcing signal (remove ensemble mean) and compute $T$, $h$ indices; also compress spatial data with EOFs
  - ```MPI-ORAS_validation.ipynb```: compare ENSO in MPI climate model to reanalysis
  - ```RO-MPI_validation.ipynb```: check if stats from RO simulations of ENSO match those from MPI
  - ```MPI_climate-change.ipynb```: evaluating how ENSO changes in MPI simulation
  - ```RO-change-over-time.ipynb```: compute change in RO parameters over time 
- ```data```: raw and preprocessed data
  - ```XRO_indices_oras5.nc```: precomputed $T$ and $h$ data from ORAS5 reanalysis
  - ```mpi_Th```: pre-computed $T$ and $h$ data from MPI climate model (50 ensemble members)
  - ```mpi/eofs300```: first 300 EOFs of MPI data (50 ensemble members)
  - ```oras5```: spatial data for SST and SSH in ORAS5
- ```src```: custom module containing functions used in ```scripts``` 
- ```results```: figures and intermediate results (e.g., model parameters)
- ```tests```: check that certain functions work as expected (e.g., EOF reconstructions)
