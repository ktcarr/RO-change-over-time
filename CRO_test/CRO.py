#!/usr/bin/env python
# coding: utf-8

# Origional code by B. Pagli

# Revised by Sen Zhao, 03/01/2025
# - Optimized solver execution for faster multi-member runs.
# - Added options for monthly mean and snapshot outputs.
# - Organized outputs as xarray dataset
# - Derive BWJ growth rate and periodicity from the linear RO parameters
# - 

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sc
from math import floor

import xarray as xr

def RO_par_load(ref="AN2020"):
    """
    Pre-load the parameters from known references.

    Parameters:
    -----------
    ref : str, optional
        Reference for the parameter set. Default is "AN2020".

    Returns:
    --------
    par : dict
        Dictionary containing the parameters.
    """
    if ref == "AN2020":
        R = -0.093
        F1 = 0.021
        F2 = 1.02
        epsilon = 0.028

        b_T = 0.012
        c_T = 0.
        d_T = 0.007
        b_h = 0

        sigma_T = 0.218
        sigma_h = 1.68
        B = 0.193
        m_T = 0.
        m_h = 0.
        n_T = 1
        n_h = 1
        n_g = 0

    par = {'R': R, 'F1': F1, 'F2': F2, 'epsilon': epsilon, 'b_T': b_T, 'c_T': c_T, 'd_T': d_T, 'b_h': b_h,
           'sigma_T': sigma_T, 'sigma_h': sigma_h, 'B': B, 'm_T': m_T, 'm_h': m_h, 'n_T': n_T, 'n_h': n_h, 'n_g': n_g}

    return par

def detect_par(p, t):
    """
    Detects the different sizes accepted for the parameters and applies the right method to match the simulation size.

    Parameters:
    -----------
    p : int, float, np.float32, np.float64, list, or np.ndarray
        The parameter to be processed.
    t : np.ndarray
        The time array for the simulation.

    Returns:
    --------
    np.ndarray
        The parameter adjusted to match the simulation size.
    """
    if isinstance(p, (int, float, np.floating)):  # Check for numpy float types as well
        if np.isnan(p):
            return np.zeros(t.size)
        else:
            return np.ones(t.size) * p
    elif isinstance(p, (list, np.ndarray)):  # Handle lists and numpy arrays
        if len(p) == 3:
            omega = 2.0 * np.pi / 12
            A = p[0]
            Aa = p[1]
            phia = p[2]
            return A + Aa * np.sin(omega * t + phia)
            # return p[0] + p[1] * np.cos(omega * t) + p[2] * np.cos(omega * t)
        elif len(p) == t.size:
            return p
        elif len(p) == int(t[-1] / (t[1] - t[0])):  # If p has length equal to number of months
            f = interp.interp1d(np.arange(0, len(p)), p, fill_value='extrapolate')
            return f(t)
        else:
            raise ValueError('ERR: Bad formats for inputs')
    else:
        raise ValueError('ERR: Input type not supported')


def RO_solver(par, IC, N, NE, NM="EH", dt=0.1, saveat=1.0, EF={'E_T': 0.0, 'E_h': 0.0}, 
              noise_custom=[], output_type='monthly_mean', seed=None, derivative='F'):
    """
    Solves the RO equations.

    Parameters:
    -----------
    par : dict
        Dictionary of length 16 containing the parameters.
    IC : array
        Array of size [2 x 1] setting the initial conditions for T and h.
    N : int
        Simulation time in months.
    NE : int
        Number of ensembles to generate.
    NM : str, optional
        Numerical scheme to solve the differential equation, "EM" or "EH". Default is "EH".
    dt : float, optional
        Timestep of the simulation in months. Default is 0.1.
    saveat : int, optional
        Save the output at each saveat month. Saveat needs to be a multiple of dt. Default is 1.0.
    EF : dict, optional
        Dictionary with 'E_T' and 'E_h'. Each can be an array of size [1 x N] or [1 x floor((N-1)/dt)+1] or [1 x 3].
    noise_custom : list, optional
        4D list of the noise to use in the integration of DT/dt (noise_custom[0]), dh/dt(noise_custom[1]), dxi_T/dt(noise_custom[2]), dh/dt(noise_custom[3]).
    output_type : str, optional
        Type of output, either 'snapshot' or 'monthly_mean'. Default is 'monthly_mean'.
    seed: seed to get option has idential stocastical simulations

    Returns:
    --------
    ds : xarray.Dataset
        Dataset containing the ensemble of time series for T and h.
    """
    # Set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Calculate the number of time steps per month
    steps_per_month = int(1 / dt)  # e.g., if dt = 0.1, steps_per_month = 10

    # Initialize arrays to store results
    if output_type == 'monthly_mean':
        TT = np.zeros((NE, N))  # Shape: (NE, N_months)
        hh = np.zeros((NE, N))  # Shape: (NE, N_months)
        xiT = np.zeros((NE, N))  # Shape: (NE, N_months)
        xih = np.zeros((NE, N))  # Shape: (NE, N_months)
    elif output_type == 'snapshot':
        # Calculate the number of snapshots
        num_snapshots = int(N / saveat) + 1
        TT = np.zeros((NE, num_snapshots))  # Shape: (NE, num_snapshots)
        hh = np.zeros((NE, num_snapshots))  # Shape: (NE, num_snapshots)
        xiT = np.zeros((NE, num_snapshots))  # Shape: (NE, num_snapshots)
        xih = np.zeros((NE, num_snapshots))  # Shape: (NE, num_snapshots)

    # Generate noise for all ensemble members and time steps
    if len(noise_custom) == 0:
        noise = np.random.normal(0, 1, size=(4, NE, N, steps_per_month))
        # Set first two rows equal to second two rows
        noise[2:4, ...] = noise[0:2, ...]
    else:
        noise = np.array(noise_custom)

    # Adjust the parameters to the right shape
    # t = np.arange(0, N * steps_per_month + 1) * dt  # Full time array

    # modified by Sen
    # when we fit using forward-differencing the resulted parameter should has 0.5 month shift, otherwise the phase locking has one month shift before
    # # See details in XRO model
    if derivative == 'F':
        shift = 0.5
        t = np.arange(0 - shift, N + 1 - shift, step=dt)  # Full time array
    else:
        t = np.arange(0, N + 1, step=dt)  # Full time array

    R = detect_par(par['R'], t)
    F1 = detect_par(par['F1'], t)
    F2 = detect_par(par['F2'], t)
    epsilon = detect_par(par['epsilon'], t)
    b_T = detect_par(par['b_T'], t)
    c_T = detect_par(par['c_T'], t)
    d_T = detect_par(par['d_T'], t)
    b_h = detect_par(par['b_h'], t)
    sigma_T = detect_par(par['sigma_T'], t)
    sigma_h = detect_par(par['sigma_h'], t)
    B = detect_par(par['B'], t)
    m_T = detect_par(par['m_T'], t)
    m_h = detect_par(par['m_h'], t)
    n_T = par['n_T']
    n_h = par['n_h']
    n_g = par['n_g']

    assert len(EF) == 2, 'Bad format for EF, needs to be a list of length 2'
    EF_T = detect_par(EF['E_T'], t)
    EF_h = detect_par(EF['E_h'], t)
    EF = [EF_T, EF_h]

    # Define the one_step function
    def one_step(T, h, xi_T, xi_h, j, dt, w_T, w_h, w_xi_T, w_xi_h):
        """
        Perform one step of the simulation.

        Parameters:
        -----------
        T : np.ndarray
            Temperature at the current time step.
        h : np.ndarray
            Height at the current time step.
        xi_T : np.ndarray
            Noise term for T at the current time step.
        xi_h : np.ndarray
            Noise term for h at the current time step.
        j : int
            Current time step index.
        dt : float
            Timestep of the simulation.
        w_T : np.ndarray
            Noise for T at the current time step.
        w_h : np.ndarray
            Noise for h at the current time step.
        w_xi_T : np.ndarray
            Noise for xi_T at the current time step.
        w_xi_h : np.ndarray
            Noise for xi_h at the current time step.

        Returns:
        --------
        T_new : np.ndarray
            Temperature at the next time step.
        h_new : np.ndarray
            Height at the next time step.
        xi_T_new : np.ndarray
            Noise term for T at the next time step.
        xi_h_new : np.ndarray
            Noise term for h at the next time step.
        """
        dw_T = np.sqrt(dt) * w_T
        dw_h = np.sqrt(dt) * w_h
        dw_xi_T = np.sqrt(dt) * w_xi_T
        dw_xi_h = np.sqrt(dt) * w_xi_h

        if NM == 'EM':
            T_new = T + f_T(T, h, xi_T, j) * dt + g_T(T, h, j) * dw_T
            h_new = h + f_h(T, h, xi_h, j) * dt + g_h(T, h, j) * dw_h
            xi_T_new = xi_T + f_xi(xi_T, m_T, j) * dt + g_xi(xi_T, m_T, j) * dw_xi_T if n_T == 0 else 0
            xi_h_new = xi_h + f_xi(xi_h, m_h, j) * dt + g_xi(xi_h, m_h, j) * dw_xi_h if n_h == 0 else 0
        elif NM == 'EH':
            T_hat_new = T + f_T(T, h, xi_T, j) * dt + g_T(T, h, j) * dw_T
            h_hat_new = h + f_h(T, h, xi_h, j) * dt + g_h(T, h, j) * dw_h
            xi_T_hat_new = xi_T + f_xi(xi_T, m_T, j) * dt + g_xi(xi_T, m_T, j) * dw_xi_T
            xi_h_hat_new = xi_h + f_xi(xi_h, m_h, j) * dt + g_xi(xi_h, m_h, j) * dw_xi_h

            T_new = T + 0.5 * (f_T(T, h, xi_T, j) + f_T(T_hat_new, h_hat_new, xi_T, j + 1)) * dt + 0.5 * (g_T(T, h, j) + g_T(T_hat_new, h_hat_new, j + 1)) * dw_T
            h_new = h + 0.5 * (f_h(T, h, xi_h, j) + f_h(T_hat_new, h_hat_new, xi_h, j + 1)) * dt + 0.5 * (g_h(T, h, j) + g_h(T_hat_new, h_hat_new, j + 1)) * dw_h
            xi_T_new = xi_T + 0.5 * (f_xi(xi_T, m_T, j) + f_xi(xi_T_hat_new, m_T, j + 1)) * dt + 0.5 * (g_xi(xi_T, m_T, j) + g_xi(xi_T_hat_new, m_T, j + 1)) * dw_xi_T if n_T == 0 else 0
            xi_h_new = xi_h + 0.5 * (f_xi(xi_h, m_h, j) + f_xi(xi_h_hat_new, m_h, j + 1)) * dt + 0.5 * (g_xi(xi_h, m_h, j) + g_xi(xi_h_hat_new, m_h, j + 1)) * dw_xi_h if n_h == 0 else 0

        return T_new, h_new, xi_T_new, xi_h_new

    # Define the functions f_T, g_T, f_h, g_h, f_xi, g_xi
    def f_T(T, h, xi_T, j):
        gt = B[j] * T if n_g == 0 else np.maximum(0, T) * B[j] * T
        if n_T == 0:
            return R[j] * T + F1[j] * h + b_T[j] * T ** 2 - c_T[j] * T ** 3 + d_T[j] * T * h + EF_T[j] + sigma_T[j] * (1.0 + gt) * xi_T
        elif n_T == 1:
            return R[j] * T + F1[j] * h + b_T[j] * T ** 2 - c_T[j] * T ** 3 + d_T[j] * T * h + EF_T[j]

    def g_T(T, h, j):
        gt = B[j] * T if n_g == 0 else np.maximum(0, T) * B[j] * T
        if n_T == 0:
            return 0
        elif n_T == 1:
            return sigma_T[j] * (1.0 + gt)

    def f_h(T, h, xi_h, j):
        if n_h == 0:
            return -F2[j] * T - epsilon[j] * h - b_h[j] * T ** 2 + EF_h[j] + sigma_h[j] * xi_h
        elif n_h == 1:
            return -F2[j] * T - epsilon[j] * h - b_h[j] * T ** 2 + EF_h[j]

    def g_h(T, h, j):
        if n_h == 0:
            return 0
        elif n_h == 1:
            return sigma_h[j]

    def f_xi(xi, m, j):
        return -m[j] * xi

    def g_xi(xi, m, j):
        return np.sqrt(2 * m[j])

    # Initialize T, h, xi_T, and xi_h for all ensemble members
    T = np.full(NE, IC[0])  # Shape: (NE,)
    h = np.full(NE, IC[1])  # Shape: (NE,)
    xi_T = np.zeros(NE)  # Shape: (NE,)
    xi_h = np.zeros(NE)  # Shape: (NE,)

    # Initialize snapshot counter
    snapshot_counter = 0

    # Loop over months
    for i in range(N):
        # Initialize accumulators for monthly means
        T_monthly_sum = np.zeros(NE)  # Shape: (NE,)
        h_monthly_sum = np.zeros(NE)  # Shape: (NE,)
        xiT_monthly_sum = np.zeros(NE)  # Shape: (NE,)
        xih_monthly_sum = np.zeros(NE)  # Shape: (NE,)

        # Loop over time steps within the month
        for j in range(steps_per_month):
            # Calculate the global time step index
            global_step = i * steps_per_month + j

            # Perform one step of the simulation
            T_new, h_new, xi_T_new, xi_h_new = one_step(T, h, xi_T, xi_h, global_step, dt, 
                                                       noise[0, :, i, j], noise[1, :, i, j], 
                                                       noise[2, :, i, j], noise[3, :, i, j])

            # Accumulate T and h for the monthly mean (if output_type == 'monthly_mean')
            if output_type == 'monthly_mean':
                T_monthly_sum += T_new
                h_monthly_sum += h_new
                xiT_monthly_sum += xi_T_new
                xih_monthly_sum += xi_h_new

            # Save snapshots at specified intervals (if output_type == 'snapshot')
            if output_type == 'snapshot' and global_step % int(saveat / dt) == 0:
                TT[:, snapshot_counter] = T_new
                hh[:, snapshot_counter] = h_new
                xiT[:, snapshot_counter] = xi_T_new
                xih[:, snapshot_counter] = xi_h_new
                snapshot_counter += 1

            # Update T, h, xi_T, and xi_h for the next step
            T, h, xi_T, xi_h = T_new, h_new, xi_T_new, xi_h_new

        # Calculate the monthly mean (if output_type == 'monthly_mean')
        if output_type == 'monthly_mean':
            TT[:, i] = T_monthly_sum / steps_per_month
            hh[:, i] = h_monthly_sum / steps_per_month
            xiT[:, i] = xiT_monthly_sum / steps_per_month
            xih[:, i] = xih_monthly_sum / steps_per_month

    # Create the xarray Dataset
    if output_type == 'monthly_mean':
        time = np.arange(0, N)  # Monthly time axis
    elif output_type == 'snapshot':
        time = np.arange(0, N * steps_per_month + 1, int(saveat / dt)) * dt  # Snapshot time axis

    ds = xr.Dataset(
        {
            'T': (['member', 'time'], TT),
            'h': (['member', 'time'], hh),
            # 'noise_T': (['member', 'time'], np.mean(noise[0, :, :, :], axis=2)),
            # 'noise_h': (['member', 'time'], np.mean(noise[1, :, :, :], axis=2)),
            'noise_xi_T': (['member', 'time'], xiT),
            'noise_xi_h': (['member', 'time'], xih),
        },
        coords={
            'member': np.arange(NE),
            'time': time,
        }
    )

    return ds


def RO_fitting(T, h, T_option, h_option, noise_option, method_fitting='LR-F', dt_fitting=1.0):
    """
    Fits the RO model to the given data.

    Parameters:
    -----------
    T : array
        Temperature data.
    h : array
        Height data.
    T_option : dict
        RO equation form for deterministic dT/dt part.
    h_option : dict
        RO equation form for deterministic dh/dt part.
    noise_option : dict
        RO noise option parameter.
    method_fitting : str, optional
        Fitting method. Default is 'LR-F'.
    dt_fitting : float, optional
        Timestep for fitting. Default is 1.0.
    start_month: float, optional
        start_month for fitting time, default is 1

    Returns:
    --------
    list_set_param : list
        List of dictionaries containing the fitted parameters.
    """
    #Detect if centered or forward are performed on T and h
    if '-F' in method_fitting:
        derivative='F'
    elif '-C' in method_fitting:
        derivative='C'
    
    N = T.shape[0]
    #Dealing with unique sample of time series to fit or multiple (model).
    if len(T.shape)==1:
        M = 1
        T=np.reshape(T,((T.shape[0]),1))
        h=np.reshape(h,((h.shape[0]),1))
    elif len(T.shape)==2: 
        M = T.shape[1]
    #Compute dT and dh according to the type of derivative
    if derivative=='F':
        dT = np.diff(T,axis=0) # difference of step i+1 and step i is the step i time's tendency
        dh = np.diff(h,axis=0)
    elif derivative=='C':
        #Index 0 will take the last element of the list !!
        dT = np.array([T[i+1,:]-T[i-1,:] for i in range(N-1)])/2
        dh = np.array([h[i+1,:]-h[i-1,:] for i in range(N-1)])/2
        #adjust for the index 0
        dT[0] = (T[1,:]-T[0,:])
        dh[0] = (h[1,:]-h[0,:])
    
    T = T[:-1]
    h = h[:-1]
    #Divide by dt_fitting 
    dT = dT/dt_fitting
    dh = dh/dt_fitting
    N = T.shape[0]

    if len(T.shape) == 1:
        M = 1
        T = np.reshape(T, (T.shape[0], 1))
        h = np.reshape(h, (h.shape[0], 1))
        dT = np.reshape(dT, (dT.shape[0], 1))
        dh = np.reshape(dh, (dh.shape[0], 1))
    elif len(T.shape) == 2:
        M = T.shape[1]

    assert T.shape == h.shape, "ERR: T and h don't have the same shape"

    list_set_param = []

    for i_sample in range(M):
        
        # print(f"Fit Sample {i_sample + 1}/{M}")

        Ti = T[:, i_sample]
        hi = h[:, i_sample]
        dTi = dT[:, i_sample]
        dhi = dh[:, i_sample]

        Ti2 = Ti * Ti
        Ti3 = Ti2 * Ti
        Thi = Ti * hi

        # time = np.arange(N)

        if derivative == 'F':
            shift = 0.5
            time = np.arange(0-shift, N-shift, step=1.) # we should shift 0.5 month as forward-differencing
        elif derivative == 'C':
            time = np.arange(N)

        omega = 2 * np.pi / 12

        R = Ti
        R_as = np.sin(omega * time) * R
        R_ac = np.cos(omega * time) * R

        F1 = hi
        F1_as = np.sin(omega * time) * F1
        F1_ac = np.cos(omega * time) * F1

        F2 = -Ti
        F2_as = np.sin(omega * time) * F2
        F2_ac = np.cos(omega * time) * F2

        epsilon = -hi
        epsilon_as = np.sin(omega * time) * epsilon
        epsilon_ac = np.cos(omega * time) * epsilon

        b_T = Ti2
        b_T_as = np.sin(omega * time) * b_T
        b_T_ac = np.cos(omega * time) * b_T

        c_T = -Ti3
        c_T_as = np.sin(omega * time) * c_T
        c_T_ac = np.cos(omega * time) * c_T

        d_T = Thi
        d_T_as = np.sin(omega * time) * d_T
        d_T_ac = np.cos(omega * time) * d_T

        b_h = -Ti2
        b_h_as = np.sin(omega * time) * b_h
        b_h_ac = np.cos(omega * time) * b_h

        dict_param_time_series_tot = {
            'R': R, 'R_as': R_as, 'R_ac': R_ac,
            'F1': F1, 'F1_as': F1_as, 'F1_ac': F1_ac,
            'F2': F2, 'F2_as': F2_as, 'F2_ac': F2_ac,
            'epsilon': epsilon, 'epsilon_as': epsilon_as, 'epsilon_ac': epsilon_ac,
            'b_T': b_T, 'b_T_as': b_T_as, 'b_T_ac': b_T_ac,
            'c_T': c_T, 'c_T_as': c_T_as, 'c_T_ac': c_T_ac,
            'd_T': d_T, 'd_T_as': d_T_as, 'd_T_ac': d_T_ac,
            'b_h': b_h, 'b_h_as': b_h_as, 'b_h_ac': b_h_ac,
        }

        list_param_to_fit_T = []
        for var, opt in T_option.items():
            if opt == 1:
                list_param_to_fit_T.append(var)
            elif opt == 3:
                list_param_to_fit_T.append(var)
                list_param_to_fit_T.append(var + '_as')
                list_param_to_fit_T.append(var + '_ac')

        list_param_to_fit_h = []
        for var, opt in h_option.items():
            if opt == 1:
                list_param_to_fit_h.append(var)
            elif opt == 3:
                list_param_to_fit_h.append(var)
                list_param_to_fit_h.append(var + '_as')
                list_param_to_fit_h.append(var + '_ac')

        XT = np.column_stack([dict_param_time_series_tot[var] for var in list_param_to_fit_T])
        Xh = np.column_stack([dict_param_time_series_tot[var] for var in list_param_to_fit_h])

        dict_param_result = {
            'R': 0, 'R_as': 0, 'R_ac': 0,
            'F1': 0, 'F1_as': 0, 'F1_ac': 0,
            'F2': 0, 'F2_as': 0, 'F2_ac': 0,
            'epsilon': 0, 'epsilon_as': 0, 'epsilon_ac': 0,
            'b_T': 0, 'b_T_as': 0, 'b_T_ac': 0,
            'c_T': 0, 'c_T_as': 0, 'c_T_ac': 0,
            'd_T': 0, 'd_T_as': 0, 'd_T_ac': 0,
            'b_h': 0, 'b_h_as': 0, 'b_h_ac': 0,
        }

        def fill_dict_param_result(res, list_name):
            for i, name in enumerate(list_name):
                if name in dict_param_result.keys():
                    dict_param_result[name] = res.params[i]
                else:
                    dict_param_result[name] = 0

        res_Ti = sm.OLS(dTi, XT).fit()
        predicted_Ti = res_Ti.predict()
        residual_Ti = dTi - predicted_Ti

        fill_dict_param_result(res_Ti, list_param_to_fit_T)

        res_hi = sm.OLS(dhi, Xh).fit()
        predicted_hi = res_hi.predict()
        residual_hi = dhi - predicted_hi

        fill_dict_param_result(res_hi, list_param_to_fit_h)

        dict_param_result_to_return = {}
        for option_dict in [T_option, h_option]:
            for var, opt in option_dict.items():
                if opt == 0:
                    dict_param_result_to_return[var] = np.nan
                elif opt == 1:
                    dict_param_result_to_return[var] = dict_param_result[var]
                elif opt == 3:
                    D3_list = [
                        dict_param_result[var],
                        np.sqrt(dict_param_result[var + '_as'] ** 2 + dict_param_result[var + '_ac'] ** 2),
                        np.arctan2(dict_param_result[var + '_ac'], dict_param_result[var + '_as'])
                        # dict_param_result[var + '_ac'],
                        # dict_param_result[var + '_as']
                    ]
                    dict_param_result_to_return[var] = D3_list

        def estimate_m_red(residual):
            temporal_deriv_residual = np.diff(residual, axis=0) / dt_fitting
            slope, _, _, _, _ = sc.stats.linregress(x=residual[:-1], y=temporal_deriv_residual)
            return -1.0 * slope

        def compute_variances_random(Tres, T, segment_length=120, num_ensembles=int(1e5)):
            N = len(Tres)
            var_Tres = []
            var_T = []

            for _ in range(num_ensembles):
                indices = np.random.choice(N, size=segment_length, replace=False)
                segment_Tres = Tres[indices]
                segment_T = T[indices]
                var_Tres.append(np.var(segment_Tres, ddof=1))
                var_T.append(np.var(segment_T, ddof=1))

            return np.array(var_Tres), np.array(var_T)

        def estimate_B_sigma_T_from_mult_residual(residual_X, X):
            var_residual_X, var_X = compute_variances_random(residual_X, X)
            slope, intercept, _, _, _ = sc.stats.linregress(x=var_X, y=var_residual_X)
            B = np.sqrt(slope / intercept)
            sigma_T = np.sqrt((slope + intercept) / (1 + B * B))
            return B, sigma_T

        if noise_option['T'] == "white" and noise_option['h'] == "white" and noise_option['T_type'] == "additive":
            B = np.nan
            n_T = 1
            n_h = 1
            n_g = 0
            m_T = np.nan
            m_h = np.nan
            sigma_T = np.std(residual_Ti)
            sigma_h = np.std(residual_hi)

        elif noise_option['T'] == "red" and noise_option['h'] == "red" and noise_option['T_type'] == "additive":
            B = np.nan
            n_T = 0
            n_h = 0
            n_g = 0
            sigma_T = np.std(residual_Ti)
            m_T = estimate_m_red(residual_Ti)
            sigma_h = np.std(residual_hi)
            m_h = estimate_m_red(residual_hi)

        elif noise_option['T'] == "white" and noise_option['h'] == "white" and (noise_option['T_type'] == "multi" or noise_option['T_type'] == "multi_H"):
            n_T = 1
            n_h = 1
            m_T = np.nan
            m_h = np.nan
            if noise_option['T_type'] == "multi_H":
                n_g = 1
                Ti_bis = Ti
                Ti_bis[np.where(Ti < 0)] = 0
                B, sigma_T = estimate_B_sigma_T_from_mult_residual(residual_Ti, Ti_bis)
            elif noise_option['T_type'] == "multi":
                n_g = 0
                B, sigma_T = estimate_B_sigma_T_from_mult_residual(residual_Ti, Ti)
            sigma_h = np.std(residual_hi)

        elif noise_option['T'] == "red" and noise_option['h'] == "red" and (noise_option['T_type'] == "multi" or noise_option['T_type'] == "multi_H"):
            n_T = 0
            n_h = 0
            if noise_option['T_type'] == "multi_H":
                n_g = 1
                Ti_bis = Ti
                Ti_bis[np.where(Ti < 0)] = 0
                B, sigma_T = estimate_B_sigma_T_from_mult_residual(residual_Ti, Ti_bis)
                m_T = estimate_m_red(residual_Ti / (1 + B * Ti_bis))
            elif noise_option['T_type'] == "multi":
                n_g = 0
                B, sigma_T = estimate_B_sigma_T_from_mult_residual(residual_Ti, Ti)
                m_T = estimate_m_red(residual_Ti / (1 + B * Ti))
            sigma_h = np.std(residual_hi)
            m_h = estimate_m_red(residual_hi)

        dict_param_result_to_return['sigma_T'] = sigma_T
        dict_param_result_to_return['sigma_h'] = sigma_h
        dict_param_result_to_return['B'] = B
        dict_param_result_to_return['m_T'] = m_T
        dict_param_result_to_return['m_h'] = m_h
        dict_param_result_to_return['n_T'] = n_T
        dict_param_result_to_return['n_h'] = n_h
        dict_param_result_to_return['n_g'] = n_g
        list_set_param.append(dict_param_result_to_return)

    if len(list_set_param) == 1:
        return list_set_param[0]
    else:
        return list_set_param


def RO_BWJ_analysis(fit_para, time=None, deriative='F'):
    """
    Computes the seasonal cycle of Recharge Oscillator (RO) parameters and Bjerknes-Wyrtki-Jin (BWJ) indices.
    
    Parameters:
    fit_para : dict
        Dictionary containing the fitted RO parameters.
        Each key represents a variable (e.g., 'R', 'F1', etc.), and the corresponding value
        is either a single value (constant) or a tuple (A, Aa, phia) defining a sinusoidal function.
    time : array-like, optional
        Time discretization for the seasonal cycle. If None, defaults to an array from 0 to 11.
    deriative : str, optional
        Unused parameter (default is 'F'). Can be removed in future implementations if unnecessary.
    
    Returns:
    xarray.Dataset
        Dataset containing the seasonal cycle of RO parameters and the computed BWJ indices.
        The dataset includes:
        - Seasonal cycle of input parameters ('R', 'F1', 'epsilon', 'F2', 'b_T', 'c_T', 'd_T', 'b_h')
        - Bjerknes index ('BJ')
        - Wave Feedback index ('WF')
        - Wave-Jin index ('WJ')
        
    Notes:
    - The function reconstructs the seasonal cycle using a sinusoidal approximation if parameters
      include amplitude and phase shift.
    - BWJ indices are calculated following Jin et al. (2020):
        * BJ = 0.5 * ( R - epsilon ) BJ index
        * WF = sqrt( F1 * F2 - ( R + epsilon )**2 / 4 )   Wyrtki frequency
        * WJ = 2 * pi / sqrt( F1 * F2 - ( R + epsilon )**2 / 4 ) Wyrtki periodcity
    """
    if time is None:
        if deriative == 'F':
            time_discr = np.arange(12) - 0.5
        else:
            time_discr = np.arange(12)
    else:
        time_discr = time

    def reconstruct_seasonal_param(var, time_discr):
        """
        Reconstructs the seasonal cycle for a given variable based on the provided parameters.
        
        Parameters:
        var : str
            Variable name in fit_para dictionary.
        time_discr : array-like
            Time discretization for seasonal reconstruction.
        
        Returns:
        np.ndarray
            Seasonal cycle values for the given variable.
        """
        try:
            A, Aa, phia = fit_para[var]
            sc = A + Aa * np.sin(2 * np.pi / 12 * time_discr + phia)
        except:
            A = fit_para[var]
            sc = A + 0. * np.sin(2 * np.pi / 12 * time_discr) 
        return sc

    fit_ds = xr.Dataset()
    for var in ['R', 'F1', 'epsilon', 'F2', 'b_T', 'c_T', 'd_T', 'b_h']:
        sc_var = reconstruct_seasonal_param(var, time_discr)
        fit_ds[var] = xr.DataArray(sc_var * 12, coords={'month': range(1, 13)})  # Convert units to year^-1

    # Calculate BWJ indices following Jin et al. (2020):
    fit_ds['BJ'] = 0.5 * (fit_ds['R'] - fit_ds['epsilon'])
    fit_ds['WF'] = np.sqrt(fit_ds['F1'] * fit_ds['F2'] - (fit_ds['R'] + fit_ds['epsilon'])**2 / 4)
    fit_ds['WJ'] = np.pi * 2 / np.sqrt(fit_ds['F1'] * fit_ds['F2'] - (fit_ds['R'] + fit_ds['epsilon'])**2 / 4)

    # Assign units to the indices
    fit_ds['BJ'].attrs['units'] = 'year**-1'
    fit_ds['WF'].attrs['units'] = 'year**-1'
    fit_ds['WJ'].attrs['units'] = 'year'
    return fit_ds