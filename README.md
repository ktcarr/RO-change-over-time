# Recharge oscillator change over time
How do recharge oscillator parameters change over time in a large ensemble? What do these changes correspond to physically? Project started at ENSO 2025 winter school.

## Set-up
1. Clone/fork repository  (```git clone ...```)
2. Create virtual environment  (```mamba create ...```)
3. Activate virtual environment (```mamba activate ...```)
4. Install packages listed in ```env.yml``` file to virtual environment (```mamba env update --file env.yml```)
5. Install custom module to virtual environment (run ```pip install -e .``` in project's home directory)


## Description of project directory
- ```scripts```: code used for analysis/plots
  - ```preprocess.ipynb```: remove external forcing signal (remove ensemble mean) and compute $T$, $h$ indices
  - ```RO-change-over-time.ipynb```: compute change in RO parameters over time
- ```src```: custom module containing functions used in ```scripts```  
- ```data```: raw and preprocessed data  
- ```results```: figures and intermediate results (e.g., model parameters)
