#!/usr/bin/env python
# coding: utf-8

#B.Pagli 04/2023
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as sc
from math import floor

#CRO package python version



def RO_par_load(ref="AN2020"):
    """
    Pre load the parameters from known references
    """
    
    if ref=="AN2020":
        R = -0.093
        F1 = 0.021
        F2 = 1.02
        epsilon = 0.028

        b_T=0.012
        c_T=0.
        d_T=0.007
        b_h=0

        sigma_T=0.218
        sigma_h=1.68
        B=0.193
        m_T=0.
        m_h=0.
        n_T=1
        n_h=1
        n_g=0
        
    par = {'R':R,'F1':F1,'F2':F2,'epsilon':epsilon,'b_T':b_T,'c_T':c_T,'d_T':d_T,'b_h':b_h,'sigma_T':sigma_T,'sigma_h':sigma_h,'B':B,'m_T':m_T,'m_h':m_h,'n_T':n_T,'n_h':n_h,'n_g':n_g}

    
    return par 

def RO_solver(par,IC,N,NE,NM="EH",dt=0.1,saveat=1.0,EF={'E_T':0.0,'E_h':0.0},noise_custom=[]):
    """
    Function that solves the RO equations (add ref of the equation)

    Parameters
        ----------
        par : dictionnary of length 16 containing the parameters : [R, F1, F2, epsilon, b_T, c, d, b_h, sigma_T, sigma_h, B, m_T, m_h, n_T, n_h,n_g]
            all the parameters can be : Single scalar value or array size of [1 x N] or [1 x floor((N-1)/dt)+1] or or [1 x 3]

        IC: array
            Array size of [2 x 1] setting the initial conditions for T and h

        N : int
            Simulation time in month

        NE : int
            Number of ensembles to generate

        NM : string 
            Numerical scheme to solve the differential equation, "EM" of "EH"

        dt : float
            Timestep of the simulation in month

        saveat : int
            Save the output at each saveat month. Saveat needs to be a multiple of dt.

        EF : dictionnary with 'E_T' and 'E_h'. Each can be array size of [1 x N] or [1 x floor((N-1)/dt)+1] or [1 x 3]
            External forcing applied to T and h

        noise_custom : 4D list of the noise to use in the integration of DT/dt (noise_custom[0]), dh/dt(noise_custom[1]), dxi_T/dt(noise_custom[2]), dh/dt(noise_custom[3]). Noises are given as array of size (1xfloor((N-1)/dt)+1)

    Outputs
        ----------
        T : array
            shape (NE,Nt) ensemble of time series generated for T
            where Nt =floor((N-1)/(saveat))+1

        h : array
            shape (NE,Nt) ensemble of time series generated for h
            where Nt=floor((N-1)/(saveat))+1


    """    
    
    t = np.array([j*dt for j in range(floor((N-1)/dt)+1)])

    def detect_par(p,t):
        """
        Function that detects the different sizes accepted for the parameters used as inputs in RO_solver() and 
        apply the right method to match the simulation size
        
        At the end, all the parameters are stored in 1D array of size floor((N-1)/dt)+1)
        
        """
        
        if isinstance(p,int) or isinstance(p,float):
            if np.isnan(p):
                return np.zeros(t.size)
            else:
                return np.ones(t.size)*p
        
        elif len(p) == 3:

            omega = 2.0*np.pi/12
            
            A = p[0]
            Aa = p[1]
            phia = p[2]

            return A + Aa*np.sin(omega*t+phia)

        elif len(p)==N:
            f = sc.interpolate.interp1d(np.arange(0,N),p,fill_value='extrapolate')
            full_time_p = f(t)
            return full_time_p
            
        elif len(p)==t.size:
            return p

        else:
            print('ERR : Bad formats for inputs')
            
    #Adjsut the parameters to the right shape according to their types using the function detect_par
    #For each parameters of par, detect its shape
    R = detect_par(par['R'],t)
    F1 = detect_par(par['F1'],t)
    F2 = detect_par(par['F2'],t)
    epsilon = detect_par(par['epsilon'],t)

    b_T=detect_par(par['b_T'],t)
    c_T=detect_par(par['c_T'],t)
    d_T=detect_par(par['d_T'],t)
    b_h=detect_par(par['b_h'],t)

    sigma_T=detect_par(par['sigma_T'],t)
    sigma_h=detect_par(par['sigma_h'],t)
    B=detect_par(par['B'],t)
    m_T=detect_par(par['m_T'],t)
    m_h=detect_par(par['m_h'],t)

    n_T=par['n_T']
    n_h=par['n_h']
    n_g=par['n_g']
    
    assert len(EF)==2,'Bad format for EF, needs to be a list of length 2'
    EF_T = detect_par(EF['E_T'],t)
    EF_h = detect_par(EF['E_h'],t)
    
    EF = [EF_T,EF_h]  
       
    def f_T(T,h,xi_T,j):
        if n_g==0:
            gt = B[j]*T
        elif n_g==1:
            gt = np.max([0,T])*B[j]*T

            
        if n_T==0:
            return R[j]*T +F1[j]*h + b_T[j]*T*T - c_T[j]*T*T*T + d_T[j]*T*h + EF_T[j] + sigma_T[j]*(1.0+gt)*xi_T
        elif n_T==1:
            return R[j]*T +F1[j]*h + b_T[j]*T*T - c_T[j]*T*T*T + d_T[j]*T*h + EF_T[j]
        
    def g_T(T,h,j):
        if n_g==0:
            gt = B[j]*T
        elif n_g==1:
            gt = np.max([0,T])*B[j]*T

        if n_T==0:
            return 0 
        elif n_T==1:
            return sigma_T[j]*(1.0+gt)
        
    def f_h(T,h,xi_h,j):
        if n_h==0:
            return -F2[j]*T - epsilon[j]*h - b_h[j]*T*T + EF_h[j] + sigma_h[j]*xi_h
        elif n_h==1:
            return -F2[j]*T - epsilon[j]*h - b_h[j]*T*T + EF_h[j] 
        
    def g_h(T,h,j):
        if n_h==0:    
            return 0
        elif n_h==1:
            return sigma_h[j]

    def f_xi(xi,m,j):
        return -m[j]*xi

    def g_xi(xi,m,j):
        return np.sqrt(2*m[j])

    #Function that makes one step in time 
    def one_step(T,h,xi_T,xi_h,j,dt,w_T,w_h,w_xi_T,w_xi_h):
            dw_T = np.sqrt(dt)*w_T
            dw_h = np.sqrt(dt)*w_h
            dw_xi_T = np.sqrt(dt)*w_xi_T
            dw_xi_h = np.sqrt(dt)*w_xi_h
        
            if NM=='EM':
                T_new = T + f_T(T,h,xi_T,j)*dt + g_T(T,h,j)*dw_T[j]
                h_new = h + f_h(T,h,xi_h,j)*dt + g_h(T,h,j)*dw_h[j]
                
                if n_T==0:
                    xi_T_new = xi_T + f_xi(xi_T,m_T,j)*dt + g_xi(xi_T,m_T,j)*dw_xi_T[j]
                if n_h==0:
                    xi_h_new = xi_h + f_xi(xi_h,m_h,j)*dt + g_xi(xi_h,m_h,j)*dw_xi_h[j]
                if n_T==1:
                    xi_T_new = 0
                if n_h==1:
                    xi_h_new = 0


            elif NM=='EH':
                T_hat_new = T + f_T(T,h,xi_T,j)*dt + g_T(T,h,j)*dw_T[j]
                h_hat_new = h + f_h(T,h,xi_h,j)*dt + g_h(T,h,j)*dw_h[j]
                
                xi_T_hat_new = xi_T + f_xi(xi_T,m_T,j)*dt + g_xi(xi_T,m_T,j)*dw_xi_T[j]
                xi_h_hat_new = xi_h + f_xi(xi_h,m_h,j)*dt + g_xi(xi_h,m_h,j)*dw_xi_h[j]
               
                T_new = T + 0.5*(f_T(T,h,xi_T,j) + f_T(T_hat_new,h_hat_new,xi_T,j+1))*dt + 0.5*(g_T(T,h,j)+g_T(T_hat_new,h_hat_new,j+1))*dw_T[j+1]
                h_new = h + 0.5*(f_h(T,h,xi_h,j) + f_h(T_hat_new,h_hat_new,xi_h,j+1))*dt + 0.5*(g_h(T,h,j)+g_h(T_hat_new,h_hat_new,j+1))*dw_h[j+1]

                if n_T==0:
                    xi_T_new = xi_T + 0.5*(f_xi(xi_T,m_T,j)+f_xi(xi_T_hat_new,m_T,j+1))*dt + 0.5*(g_xi(xi_T,m_T,j)+g_xi(xi_T_hat_new,m_T,j+1))*dw_xi_T[j+1]
                else:
                    xi_T_new = 0
                if n_h==0:
                    xi_h_new = xi_h + 0.5*(f_xi(xi_h,m_h,j)+f_xi(xi_h_hat_new,m_h,j+1))*dt + 0.5*(g_xi(xi_h,m_h,j)+g_xi(xi_h_hat_new,m_h,j+1))*dw_xi_h[j+1]
                else:
                    xi_h_new = 0
                    
            return T_new,h_new, xi_T_new, xi_h_new
    
    

    TT = []
    hh = []
    noise = [[] for _ in range(4)]
    
    

    #Generation of the NE members
    for i in range(NE):
        print("Sample {}/{}".format(i+1,NE))

        if len(noise_custom)==0:
            w_T=np.random.normal(0, 1, size = t.size)
            w_h=np.random.normal(0, 1, size = t.size)
            w_xi_T=np.random.normal(0, 1, size = t.size)
            w_xi_h=np.random.normal(0, 1, size = t.size)
        else:
            w_T = noise_custom[0]
            w_h = noise_custom[1]
            w_xi_T = noise_custom[2]
            w_xi_h = noise_custom[3]

        #Initialize T,h,xi_T,xi_h as empty lists
        T = []
        h = []
        xi_T = []
        xi_h = []

        #Initial conditions before time evolution
        T.append(IC[0])
        h.append(IC[1])
        #For noises initialize with a random white noise
        if n_T==0:
            xi_T.append(0)
        else:
            xi_T.append(0)
        if n_h==0:
            xi_h.append(0)
        else:
            xi_h.append(0)

        for j in np.arange(1,t.size):
            T_new,h_new, xi_T_new, xi_h_new = one_step(T[j-1],h[j-1],xi_T[j-1],xi_h[j-1],j-1,dt,w_T,w_h,w_xi_T,w_xi_h)
            
            T.append(T_new)
            h.append(h_new)
            xi_T.append(xi_T_new)
            xi_h.append(xi_h_new)

        TT.append(np.array(T[::int(round(saveat / dt))]))
        hh.append(np.array(h[::int(round(saveat / dt))]))
        noise[0].append(np.array(w_T[::int(round(saveat / dt))]))
        noise[1].append(np.array(w_h[::int(round(saveat / dt))]))
        noise[2].append(np.array(w_xi_T[::int(round(saveat / dt))]))
        noise[3].append(np.array(w_xi_h[::int(round(saveat / dt))]))


    
    return np.array(TT),np.array(hh),np.array(noise)




def RO_fitting(T,h,T_option, h_option, noise_option, method_fitting='LR-F', dt_fitting=1.0):

    """
    Inputs: 
    T,h has to be 1D or 2D (N,M))
    N = length of the time series
    M = number of samples (T,h) (e.g. M models CMIP )

    T_option : RO equation form for deterministic dT/dt part 
                T_option={“R”: X1, “F1”: X2, “b_T”: X3, “c_T”: X4, “d_T”: X5} 
                The elements of the cell array (Xi, i=1,2,3,4,5) are one of these values: 
                 0-not considered (Xi=0) 
                 1-seasonally-constant (Xi=X)
                 3-seasonally-varying-annual (Xi=X+Xasin(wt+φa)) (w=2π/12)
    h_option :  RO equation form for deterministic dh/dt part 
                T_option={“F2”: X1, “epsilon”: X2, “b_h”: X3} 
                The elements of the cell array (Xi, i=1,2,3) are one of these values:
                 0-not considered (X=0) 
                 1-seasonally-constant (X=X)
                 3-seasonally-varying-annual (X=X+Xasin(wt+φa)) (w=2π/12)

    noise_option :  RO noise option parameter 
                    noise_option={“T”: X1, “h”: X2, “T_type”: X3} 
                    X1 and X2 must be identical and are one of these values:
                     “white”-white noise RO (specify n_T=n_h=1)
                     “red”-red noise RO (specify n_T=n_h=0) 
                    X3 is one of these values: 
                     “additive”-additive noise RO (B is not considered) 
                     “multi”-multiplicative RO (specify n_g=0)
                     “multi-H”-multiplicaitve-heaviside RO (specify n_g=1)

    Note: The RO_solver function supports different values for n_T and n_h. 
    However, the RO_fitting function does not, as this option is not consistently 
    available for all supported methods and is impractical in implementation. 

    method_fitting : fitting method (default is LR-F) 
                     Option for specifying the fitting method for the RO. method_fitting is one of these values:
                     “LR-F”-linear regression (tendency with forward differencing scheme)
                     “LR-C”-linear regression (tendency with central differencing scheme)


    Output:
    return a list of dict of param for each couple (T,h) of length M (a set of param for each sample)
    
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
        dT = np.diff(T,axis=0)
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

    #Dealing with unique sample of time series to fit or multiple (model).
    if len(T.shape)==1:
        M = 1
        T=np.reshape(T,((T.shape[0]),1))
        h=np.reshape(h,((h.shape[0]),1))
        dT=np.reshape(dT,((dt.shape[0]),1))
        dh=np.reshape(dh,((dh.shape[0]),1))
    elif len(T.shape)==2: 
        M = T.shape[1]
    
    assert T.shape==h.shape, "ERR : t and h doesn't have the same shape"
    
    #list that will contain every set of param for each sample. If only one sample, list of length one
    list_set_param = []
    
    for i_sample in range(M):
        print("Fit Sample {}/{}".format(i_sample+1,M))
        
        Ti = T[:,i_sample]
        hi = h[:,i_sample]
        dTi = dT[:,i_sample]
        dhi = dh[:,i_sample]
    
        Ti2 = Ti*Ti
        Ti3 = Ti2*Ti
    
        Thi = Ti*hi
    
    
        time = np.arange(N)
        #Only annual cycle is supported in this version
        omega = 2*np.pi/12

        #Define all the possible terms in the equation of dT/dt and dh/dt, with the possible seasonal cycle on R,F1,F2,epsilon,b_T,c_T,d_T,b_h
        #Example for R : If we put a seasonal cycle on R :
        # R = (R + R_a*sin(omega*t + phi_a)) =  R + R_as*sin(omega*t) + R_ac*cos(omega*t)
        # The parameters that we estimate directly here are R, R_as, R_ac.
        # To recover the parameters that we will upload in RO_solver : 
        # R = R
        # R_a = sqrt(R_as^2 + R_ac^2)
        # phi_a = np.arctan2(R_ac/R_as)
        # each time series is named according to the parameters it is related
        
        R = Ti
        R_as = np.sin(omega*time)*R
        R_ac = np.cos(omega*time)*R
    
        F1 = hi
        F1_as = np.sin(omega*time)*F1
        F1_ac = np.cos(omega*time)*F1
    
        F2 = -Ti
        F2_as = np.sin(omega*time)*F2
        F2_ac = np.cos(omega*time)*F2
    
        epsilon = -hi
        epsilon_as = np.sin(omega*time)*epsilon
        epsilon_ac = np.cos(omega*time)*epsilon
    
        b_T = Ti2
        b_T_as = np.sin(omega*time)*b_T
        b_T_ac = np.cos(omega*time)*b_T
    
        c_T = -Ti3
        c_T_as = np.sin(omega*time)*c_T
        c_T_ac = np.cos(omega*time)*c_T
    
        d_T = Thi
        d_T_as = np.sin(omega*time)*d_T
        d_T_ac = np.cos(omega*time)*d_T
        
    
        b_h = -Ti2
        b_h_as = np.sin(omega*time)*b_h
        b_h_ac = np.cos(omega*time)*b_h

        #Dictionnary that will contains all the time series possible (including seasonal cycle on all deterministic variables)
        #The input parameters of the function will call the variables of interest
        dict_param_time_series_tot = {
                                'R' : R,
                                'R_as' : R_as,
                                'R_ac' : R_ac,
                                'F1' : F1,
                                'F1_as' : F1_as,
                                'F1_ac' : F1_ac,
                                'F2'  : F2,
                                'F2_as' : F2_as,
                                'F2_ac' : F2_ac,
                                'epsilon' : epsilon,
                                'epsilon_as' : epsilon_as,
                                'epsilon_ac' : epsilon_ac,
                                'b_T' : b_T,
                                'b_T_as' : b_T_as,
                                'b_T_ac' : b_T_ac,
                                'c_T' : c_T,
                                'c_T_as' : c_T_as,
                                'c_T_ac' : c_T_ac,
                                'd_T' : d_T,
                                'd_T_as' : d_T_as,
                                'd_T_ac' : d_T_ac,
                                'b_h' : b_h,
                                'b_h_as' : b_h_as,
                                'b_h_ac' : b_h_ac,
        }

        
        #According to the option indicated by the users in T_option and h_option, list the predictors
        #that will be taken in to account in the fit
        list_param_to_fit_T = []
        for var,opt in T_option.items():
            if opt==1:
                list_param_to_fit_T.append(var)
            elif opt==3:
                list_param_to_fit_T.append(var)
                list_param_to_fit_T.append(var+'_as')
                list_param_to_fit_T.append(var+'_ac')
        list_param_to_fit_h = []
        for var,opt in h_option.items():
            if opt==1:
                list_param_to_fit_h.append(var)
            elif opt==3:
                list_param_to_fit_h.append(var)
                list_param_to_fit_h.append(var+'_as')
                list_param_to_fit_h.append(var+'_ac')
                
        
        #Build the X matrices for T and h for the deterministic parameters. Contains the predictors of interest 
        XT = np.column_stack([dict_param_time_series_tot[var] for var in list_param_to_fit_T])
        Xh = np.column_stack([dict_param_time_series_tot[var] for var in list_param_to_fit_h])
        
        #dict of all parameters to be filled during the process
        dict_param_result = {'R' : 0,
                                'R_as' : 0,
                                'R_ac' : 0,
                                'F1' : 0,
                                'F1_as' : 0,
                                'F1_ac' : 0,
                                'F2'  : 0,
                                'F2_as' : 0,
                                'F2_ac' : 0,
                                'epsilon' : 0,
                                'epsilon_as' : 0,
                                'epsilon_ac' : 0,
                                'b_T' : 0,
                                'b_T_as' : 0,
                                'b_T_ac' : 0,
                                'c_T' : 0,
                                'c_T_as' : 0,
                                'c_T_ac' : 0,
                                'd_T' : 0,
                                'd_T_as' : 0,
                                'd_T_ac' : 0,
                                'b_h' : 0,
                                'b_h_as' : 0,
                                'b_h_ac' : 0,
        }
    
        def fill_dict_param_result(res,list_name):
            #function to fill the dictionnary of param with the parameters asked by the user
            for i,name in enumerate(list_name):
                if name in dict_param_result.keys():
                    dict_param_result[name] = res.params[i]
                else:
                    dict_param_result[name] = 0
    
        #Only the linear regression method is implemented in this version
        #Other methods are possible and sometimes more adapted to certain types of RO
        
        #Fit of the parameters for the T equation
        res_Ti = sm.OLS(dTi, XT).fit()   #Least square model between predictand dTi/dt and predictors XT
        predicted_Ti = res_Ti.predict()  #prediction
        residual_Ti = dTi - predicted_Ti #Compute the residual
    
        fill_dict_param_result(res_Ti,list_param_to_fit_T) #fill the dict of results with obtained parameters 
    
        #Fit of the parameters for the h equation
        res_hi = sm.OLS(dhi, Xh).fit()
        predicted_hi = res_hi.predict()
        residual_hi = dhi - predicted_hi #Compute the residual
    
        fill_dict_param_result(res_hi,list_param_to_fit_h)
        #For all variables listed in T_option and h_option, fill with parameters to be returned. 
        #For parameters asked as seasonally varying annually (3 dimensions list)
        #The fitted parameters X, X_as, X_ac are converted to X,Xa,phi_a (see explanations above)
        dict_param_result_to_return = {}
        for option_dict in [T_option, h_option]:
            for var,opt in option_dict.items():
                if opt==0:
                    dict_param_result_to_return[var]=np.nan
                elif opt==1:
                    dict_param_result_to_return[var]=dict_param_result[var]
                elif opt==3:
                    D3_list = []
                    D3_list.append(dict_param_result[var])
                    D3_list.append(np.sqrt(dict_param_result[var+'_as']**2+dict_param_result[var+'_ac']**2))
                    D3_list.append(np.arctan2(dict_param_result[var+'_ac'],dict_param_result[var+'_as']))
                    dict_param_result_to_return[var] = D3_list
                    
        #Functions defined for fitting noise parameters
        
        #Only for red noise, function estimating m_T or m_h by linear regression given a residual
        def estimate_m_red(residual):
            """
            (Res_i+1 - Res_i)/dt = -m*res_i
            m is estimated by linear regression
            """
            temporal_deriv_residual = np.diff(residual,axis=0)/dt_fitting
            slope, _,_,_,_ = sc.stats.linregress(x=residual[:-1],y=temporal_deriv_residual)
            return -1.0*slope

        #Only for multi T_type. In the case where we have a residual from the fit of T and T. 
        #Generates num_ensembles of the variance of segments of lenght segment_lenght of residual_T and T.
        #This used for estimatin sigma_T and B in the multi type.
        def compute_variances_random(Tres, T, segment_length=120, num_ensembles=int(1e5)):
            """Randomly samples segment_length non-consecutive points from Tres and T, then computes variances. num_ensembles
            of var(Tres) and var(T) are generated"""
            N = len(Tres)  # Length of time series
            var_Tres = []
            var_T = []
        
            for _ in range(num_ensembles):
                # Randomly select 120 unique indices
                indices = np.random.choice(N, size=segment_length, replace=False)
        
                # Extract random samples
                segment_Tres = Tres[indices]
                segment_T = T[indices]
        
                # Compute variance
                var_Tres.append(np.var(segment_Tres,ddof=1))  # ddof=1 for unbiased estimate
                var_T.append(np.var(segment_T,ddof=1))
    
            return np.array(var_Tres), np.array(var_T)
                
        #Only for multi T_type. Given the samples generated by compute_variances_random(), retreive B and sigma_T
        #thanks to a linear regression.
        def estimate_B_sigma_T_from_mult_residual(residual_X,X):
            """
            var(residual_T) = sigma_T^2 + sigma_T^2B^2var(T)
            Y = intercept + slope*X
            slope = sigma_T^2B^2
            intercept = sigma_T^2
            ==> B = sqrt(slppe/intercept), sigma_T = sqrt((slope+intercept)/(1+B^2))
            
            """
            var_residual_X, var_X = compute_variances_random(residual_X,X)
            slope, intercept,_,_,_ = sc.stats.linregress(x=var_X,y=var_residual_X)
            B = np.sqrt(slope/intercept)
            sigma_T = np.sqrt((slope+intercept)/(1+B*B))
            return B,sigma_T
    

            
        #According to the RO type, estimate noise parameters
        if noise_option['T']=="white" and noise_option['h']=="white" and noise_option['T_type']=="additive":
        #white-additive RO"
            B=np.nan
            n_T=1
            n_h=1
            n_g=0
            m_T=np.nan
            m_h=np.nan
            sigma_T = np.std(residual_Ti)
            sigma_h = np.std(residual_hi)
                            
        elif noise_option['T']=="red" and noise_option['h']=="red" and noise_option['T_type']=="additive":
        #red-additive RO"
            B=np.nan
            n_T=0
            n_h=0
            n_g=0
            sigma_T=np.std(residual_Ti)
            m_T = estimate_m_red(residual_Ti)
    
            sigma_h= np.std(residual_hi)
            m_h = estimate_m_red(residual_hi)
            
        elif noise_option['T']=="white" and noise_option['h']=="white" and (noise_option['T_type']=="multi" or noise_option['T_type']=="multi_H"):
        #white-multi RO"
            n_T=1
            n_h=1
            m_T=np.nan
            m_h=np.nan
            if noise_option['T_type']=="multi_H":
                n_g=1
                Ti_bis = Ti
                Ti_bis[np.where(Ti<0)]=0 #Heaviside(T)
                B, sigma_T = estimate_B_sigma_T_from_mult_residual(residual_Ti,Ti_bis)
    
            elif noise_option['T_type']=="multi":
                n_g=0
                B, sigma_T = estimate_B_sigma_T_from_mult_residual(residual_Ti,Ti)
            sigma_h=np.std(residual_hi)
            
        elif noise_option['T']=="red" and noise_option['h']=="red" and (noise_option['T_type']=="multi" or noise_option['T_type']=="multi_H"):
        #red-multi RO"
            n_T=0
            n_h=0
            if noise_option['T_type']=="multi_H":
                n_g=1
                Ti_bis = Ti
                Ti_bis[np.where(Ti<0)]=0
                B, sigma_T = estimate_B_sigma_T_from_mult_residual(residual_Ti,Ti_bis)
                m_T = estimate_m_red(residual_Ti/(1+B*Ti_bis))
    
            elif noise_option['T_type']=="multi":
                n_g=0
                B, sigma_T = estimate_B_sigma_T_from_mult_residual(residual_Ti,Ti)
                m_T = estimate_m_red(residual_Ti/(1+B*Ti))
                
            sigma_h = np.std(residual_hi)
            m_h = estimate_m_red(residual_hi)
    
    
        #Fill with corresponding nosie parameters adjusted
        dict_param_result_to_return['sigma_T']=sigma_T
        dict_param_result_to_return['sigma_h']=sigma_h
        dict_param_result_to_return['B']=B
        dict_param_result_to_return['m_T']=m_T
        dict_param_result_to_return['m_h']=m_h
        dict_param_result_to_return['n_T']=n_T
        dict_param_result_to_return['n_h']=n_h
        dict_param_result_to_return['n_g']=n_g
        list_set_param.append(dict_param_result_to_return)

        
    if len(list_set_param)==1:
        #Case where the fit is performed for only one pair (T,h)
        return list_set_param[0]
    else:
        #Case where the fit is performed for an ensemble of M pairs (T,h)
        return list_set_param

