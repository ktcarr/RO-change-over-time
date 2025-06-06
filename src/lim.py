# standard packages
import xarray as xr
import numpy as np
import scipy.spatial.distance
import scipy.linalg
import tqdm
import warnings
import copy
import math
import src.utils
import matplotlib.pyplot as plt
import seaborn as sns

## Set seaborn plotting style
sns.set(rc={"axes.facecolor": "white", "axes.grid": False})


#### Utility functions for LIM #####


def fourier_expand(X, T=12, sum_conjugates=False):
    """Decompose periodic signal into harmonics (i.e.,
    a Fourier series). Assumes 'X' is 2-d and time is
    zeroth axis.
    Args:
        'X' is the signal.
        'T' is the period of the signal in months
    Returns:
        'X_modal' is the decomposed signal. zeroth axis indexes the harmonic
            component.
        'mode_freq' is an array specifying the frequency associated with each
        mode. Note that len(mode_freq) == X_modal.shape[0]
    """

    ## get number of samples
    N = X.shape[0]

    ## get Nyquist freq (units: cycles/period)
    nyq_freq = np.floor(N / 2).astype(int)

    ## Compute FFT
    X_fft = np.fft.fft(X, axis=0)
    freq = np.fft.fftfreq(X_fft.shape[0])

    ## convert freq. from cyc/timestep to rad/month
    timesteps_per_month = N / T
    rad_per_cycle = 2 * np.pi
    freq_rad_month = freq * timesteps_per_month * rad_per_cycle

    ## custom ifft (get modal decomposition)
    t = np.arange(0, T, T / N)
    theta = np.einsum("f,t->ft", freq_rad_month, t)
    X_modal0 = 1 / N * np.einsum("f...,ft->ft...", X_fft, np.exp(1j * theta))

    ## sum over conjugate frequencies if desired
    if sum_conjugates:
        # empty array to hold result
        X_modal = np.zeros([nyq_freq + 1, N, *X.shape[1:]], dtype=complex)

        # handle zero-freq first
        X_modal[0] = copy.deepcopy(X_modal0[0])

        # check if even or odd
        if N % 2 == 0:
            # if even, handle nyquist freq. separately
            X_modal[nyq_freq] = copy.deepcopy(X_modal0[nyq_freq])

            # specify last index for summation
            last_idx_for_sum = nyq_freq - 1

        else:
            last_idx_for_sum = nyq_freq

        ## loop through pairs of modes (+/- freq. pairs)
        for mode_idx in np.arange(1, last_idx_for_sum + 1):
            X_modal[mode_idx] = X_modal0[mode_idx] + X_modal0[-mode_idx]

        # get (real) frequencies for each mode
        mode_freq = np.linspace(0, -freq[nyq_freq], nyq_freq + 1)

    else:
        X_modal = X_modal0
        mode_freq = freq

    return X_modal, mode_freq


def get_conjugate_indices(eigs):
    """Given a list of eigenvalues, return a list of paired
    indices. Eigenvalues without imaginary components correspond
    to a one-item list"""

    ## empty list to hold index pairs
    indices = []

    ## get array of indices to check
    indices_to_check = np.arange(len(eigs))

    ## only check non-zero eigenvalues
    indices_to_check = indices_to_check[eigs != 0]

    ## convert to list
    indices_to_check = list(indices_to_check)

    while len(indices_to_check) > 0:
        ## Get index to check
        idx = indices_to_check.pop(0)

        ## find index of complex conjugate
        match_idx = np.where(eigs.conj() == eigs[idx])[0]

        ## Check that complex conj. is in list.
        ## May not be true if eigenvalues/vectors are truncated
        ## Note that this IS true if eigenvalue has no imaginary part
        ## (b/c conj. of a purely real number is itself).
        if len(match_idx) > 0:
            ## Check if the eigenvalue has a complex conjugate.
            ## (otherwise, it's purely real)
            if match_idx != idx:
                indices.append([idx, match_idx.item()])
                indices_to_check.remove(match_idx)
            else:
                indices.append([idx])
        else:
            ## otherwise, put the index in a single-item list
            indices.append([idx])

    return indices


def pinv(X, k):
    """Take pseudoinverse of X, truncating to 'k' modes"""

    ## Perform SVD
    Z, Sigma, Qh = np.linalg.svd(X)

    ## truncate
    Zk = Z[:, :k]
    Sigmak = Sigma[:k]
    Qk = (Qh.conj().T)[:, :k]

    ## Invert singular values
    Sigmak_inv = 1 / Sigmak

    ## compute pseudoinverse
    pinv_ = Qk @ np.diag(Sigmak_inv) @ Zk.conj().T

    ## Compute condition number
    rcond = Sigmak[-1] / Sigmak[0] - np.finfo(float).eps

    return pinv_, rcond


###### kernel functions ######
########### utility functions for kernels ##########
def l2_sqr_loop(A, B):
    """
    Compute squared L2-distance between samples using a loop.
    Use to check 'dist' function is working properly.
    """
    n = B.shape[1]
    distmat = np.nan * np.zeros([n, n])
    for i in range(n):
        b = B[:, i : i + 1]
        distmat[:, i] = np.diag((A - b).T @ (A - b))
    return distmat


def l2_sqr_scipy(A, B):
    """Compute squared L2 distance between samples using Scipy function"""
    return scipy.spatial.distance.cdist(A.T, B.T, metric="sqeuclidean")


def l2_sqr(A, B):
    """Compute squared L2 distance between samples.
    A and B are both mxk matrices."""

    # Get difference between pairs (result is mxkxk matrix)
    diff = A[:, :, None] - B[:, None, :]

    # convert difference to L2 distance: square and sum along zeroth axis
    distmat = (diff**2).sum(0)

    return distmat


def kernel_linear(A, B):
    """Linear kernel. Take inner product of matrices/vectors A & B"""

    return A.T @ B


def kernel_polynomial(A, B, d, c=0):
    """Polynomial kernel"""
    gamma = 1 / A.shape[0]
    return (gamma * A.T @ B + c) ** d


def kernel_gaussian(A, B, sigma):
    """Gaussian kernel"""

    return np.exp(-1 / (2 * sigma**2) * l2_sqr_scipy(A, B))


def kernel_gaussian_grad(X, x, sigma):
    """Gradient of kernel with respect to x.
    Note: x must be shape mx1, where m=X.shape[0]"""
    kernel_eval = kernel_gaussian(X, x, sigma).flatten()
    return 1 / (sigma**2) * np.diag(kernel_eval) @ (X - x).T


def kernel_gaussian_precomp(l2_dist, sigma):
    """Gaussian kernel, but with pre-computed L2 distances.
    Useful for forming Gram matrices..."""

    return np.exp(-1 / (2 * sigma**2) * l2_dist)


def afunc(a):
    return a + 4


def psi_polynomial(W, gamma):
    """Lift data from original feature space to induced feature space.
    Assumes polynomial with degree 2 and c=1"""

    ## Get dimension of induced feature space
    m, n = W.shape
    m_k = math.comb(m + 2, 2)

    ## Empty array to hold result
    Psi_w = np.zeros([m_k, n])

    ## squared terms
    squared_terms = gamma * W**2

    ## linear terms
    linear_terms = np.concatenate(
        [
            np.sqrt(2 * gamma) * W,
            np.ones([1, n]),
        ],
        axis=0,
    )

    ## cross terms
    cross_terms = []
    for i in range(m - 1):
        cross_terms.append(W[i] * W[i + 1 :])
    cross_terms = np.sqrt(2) * gamma * np.concatenate(cross_terms, axis=0)

    ## assemble into single array
    Psi_w[:m] = squared_terms
    Psi_w[m : 2 * m + 1] = linear_terms
    Psi_w[2 * m + 1 :] = cross_terms

    return Psi_w


##### LIM-specific functions #####
###### "original" LIM, (following Penland & Sardeshmukh) #####
class LIM_orig:
    """
    Class to represent generalization of linear inverse model
    (i.e., with non-linear kernel).

    Attributes include:

    - K.T: propagator matrix
    - conj_idx: conjugate indices of eigenvalues
    - normal_modes: empirical normal modes (ENMs)
    - gamma, sigma, omega: timescales for ENMs
    """

    def __init__(
        self,
        X,
        Y,
        k=None,
        rcond=1e-10,
        lonlat_coord=None,
        Vtilde=None,
        no_truncate=False,
        truncate_Y=True,
        svd_idx=None,
        zero_idx=None,
    ):
        """
        'X' and 'Y' are matrices with training data.
        'k' is integer, specifying number of modes to truncate to.
        'kernel' is a function which takes in two matrices of equal size,
        and computes distance between pairs of columns.
        'rcond' is cutoff for singular values when doing a pseudoinverse.
        'lonlat_coord' is xarray coordinate to use for koopman modes.
        """

        self.X = X
        self.Y = Y
        self.k = k
        self.rcond = rcond
        self.lonlat_coord = lonlat_coord
        self.Vtilde = Vtilde
        self.no_truncate = no_truncate
        self.truncate_Y = truncate_Y
        self.svd_idx = svd_idx
        self.zero_idx = zero_idx

        ## Compute LIM components
        self.compute()

        return

    def compute(self):
        """pre-compute components of LIM"""

        ## First, project data onto leading EOFs
        Z = get_modes(X=self.X, idx=self.svd_idx, k=self.k)

        ## reset value of K if computing SVD modes separately on data subsest
        if self.svd_idx is not None:
            self.k = Z.shape[1]

        ## perform truncation if desired
        if self.no_truncate:
            Xhat = self.X
            Yhat = self.Y
        elif self.truncate_Y:
            Xhat = Z.T @ self.X
            Yhat = Z.T @ self.Y
        else:
            Xhat = Z @ Z.T @ self.X
            Yhat = self.Y

        ## Propagator
        KhatT = Yhat @ np.linalg.pinv(Xhat, rcond=self.rcond)
        Khat = KhatT.T

        ## Zero-out specified indices if desired
        if self.zero_idx is not None:
            Khat = self.mask_Khat(Khat)

        ## Eigenvalue decomposition of operator
        Vhat, Uhat, Lambda = eig_decomp(Khat)

        ## Manually truncate eigenvectors for comparison to kernel version
        if not self.truncate_Y:
            Vhat = Vhat[:, : self.k]
            Uhat = Uhat[:, : self.k]
            Lambda = Lambda[: self.k]

        ## Rescale right eigenvectors to match kernel LIM
        if self.Vtilde is not None:
            Vhat = rescale_right_eigs(Vhat=Vhat, Vtilde=self.Vtilde, Xhat=Xhat)

        ## Rescale left eigenvectors so that U^T @ V = I
        Uhat = rescale_left_eigs(V=Vhat, U=Uhat, Lambda=Lambda)

        ## Timescales for modes
        gamma, sigma, omega = get_timescales(Lambda)

        ## Get indices of complex conjugate eigenvalues
        conj_idx = get_conjugate_indices(Lambda)

        ## Compute Koopman modes (contain real and imaginary parts)
        if self.truncate_Y:
            Xi = Z @ Uhat
        else:
            Xi = Uhat

        ## convert to ENMs (only real parts)
        normal_modes = get_empirical_normal_modes(Xi, conj_idx, self.lonlat_coord)

        ## Propagator matrix
        def G(tau):
            ## Compute gamma*tau separately (to catch warning for mult. by -inf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                gamma_tau = gamma * tau

            return Uhat @ np.diag(np.exp(gamma_tau)) @ Vhat.T

        ## Save pre-computed components
        self.Xhat = Xhat
        self.Yhat = Yhat
        self.Khat = Khat
        self.Z = Z
        self.Vhat = Vhat
        self.Uhat = Uhat
        self.Lambda = Lambda
        self.gamma = gamma
        self.sigma = sigma
        self.omega = omega
        self.conj_idx = conj_idx
        self.Xi = Xi
        self.normal_modes = normal_modes
        self.G = G

        return

    def mask_Khat(self, Khat):
        """mask out specified elements of Ktilde"""

        ## Construct 'B' matrix; identity except for indices
        ## of elements to mask
        B = np.ones(self.k)
        B[self.zero_idx] = 0
        B = np.diag(B)

        ## mask out the propagator
        Khat_masked = B @ Khat

        return Khat_masked

    def get_optimal_helper(self, M):
        """Compute optimals given M matrix"""

        ## compute eigenvalue decomp. of operator
        V_M, _, Lambda_M = eig_decomp(M)

        ## Translate back into real space if necessary
        if self.truncate_Y:
            x_opt = self.Z @ V_M[:, 0]
        else:
            x_opt = V_M[:, 0]

        ## Put in xarray if lonlat coords specified
        if self.lonlat_coord is not None:
            x_opt = xr.DataArray(x_opt, coords={"lonlat": self.lonlat_coord}).unstack()

        ## rescale by standard deviation for convenience
        x_opt /= x_opt.std()

        return x_opt

    def get_optimal_IC(self, tau, N=None):
        """Compute optimal initial condition for specified lag and norm"""

        ## Specify identity norm as default
        if N is None:
            N = np.eye(self.Vhat.shape[0])

        ## Lag-dependent propagator matrix
        G = self.G(tau)

        ## Compute M matrix
        M = G.conj().T @ N @ G

        ## Compute optimal
        x0_opt = self.get_optimal_helper(M)

        return x0_opt

    def get_stochastic_optimal(self, tau, N=None, nt=20):
        """
        Compute optimal stochastic forcing pattern for specified lag and norm.
        'nt' is the number of sums to use when approximating the integral.
        """

        ## Specify identity norm as default
        if N is None:
            N = np.eye(self.Vhat.shape[0])

        ## specify bounds of integration and timestep
        dt = tau / nt  # time step for integral
        ti = dt / 2
        tf = tau - dt / 2

        ## Compute integral
        M = np.zeros_like(N)  # empty array to hold result
        for t in np.arange(ti, tf + dt, dt):
            G = self.G(tau - t)
            M = M + G.conj().T @ N @ G * dt

        ## Compute optimal
        f_opt = self.get_optimal_helper(M)

        return f_opt

    def predict(self, xt, tau):
        """
        Forecast x(t+tau) given initial conditions xt and
        lead times tau. Function computes Xi @ exp(Gamma*tau) @ xt
        - 'xt' is a 2-D array, where 1st dim is # of gridpoints
        and 2nd dim is number of initial conds.
        - 'tau' is a 1-D array (number of lead times to forecast for).
        For matrix multiplication/einsum, we represent dimensions
        using the following characters:
        m: space dimension
        k: eigenvalue dimension
        t: tau dimension
        x: init. cond. dimension
        """

        ## Compute gamma tau
        gamma_tau = np.einsum("k,t->kt", self.gamma, tau)

        ## truncate input if necessary
        if self.no_truncate:
            xt_proj = xt
        else:
            xt_proj = self.Z.T @ xt

        ## Perform matrix mult.
        forecast = np.einsum(
            "mk,kt,kx->mtx", self.Xi, np.exp(gamma_tau), self.Vhat.T @ xt_proj
        )

        return forecast


class LIM_CS:
    """
    Class to represent cyclostationary LIM
    (i.e., with non-linear kernel).

    Attributes include:

    - K.T: propagator matrix
    - conj_idx: conjugate indices of eigenvalues
    - normal_modes: empirical normal modes (ENMs)
    - gamma, sigma, omega: timescales for ENMs
    """

    def __init__(
        self,
        X,
        Y,
        month_labels,
        rcond=1e-10,
        split_harmonics=False,
        sum_conjugates=False,
    ):
        """
        'X' and 'Y' are matrices with training data.
        'k' is integer, specifying number of modes to truncate to.
        'month_labels' is array with length equal to number of training
            data points, representing month of each sample.
        'rcond' is cutoff used to compute pseudoinverse (for computing
            transition matrices).
        'split_harmonics' is bool specifying whether to decompose the
            cyclostationary (left) eigenvectors into harmonics (whose
            sum recovers the full eigenvector).
        'sum_conugates' only matters if(split_harmonics).
            if(sum_conjugates), sum over +/- pairs of harmonics.
        """

        self.X = X
        self.Y = Y
        self.month_labels = month_labels
        self.rcond = rcond
        self.split_harmonics = split_harmonics
        self.sum_conjugates = sum_conjugates

        ## Compute LIM components
        self.compute()

        return

    def compute(self):
        """pre-compute components of LIM"""

        # get dimension of data
        self.m = self.X.shape[0]

        ## Compute propagator matrices for each month
        K_m = self.compute_transition_mats()
        self.K_m = K_m

        ## Compute monodromy matrix for January start
        B0 = self.B(start_month=0)

        ## Eigenvalue decomposition of monodromy mat
        V0, U0, Lambda = eig_decomp(B0)

        ## Rescale left eigenvectors so that U^T @ V = I
        U0 = rescale_left_eigs(V=V0, U=U0, Lambda=Lambda)

        ## Timescales for modes
        gamma, sigma, omega = get_timescales(Lambda, tau=12)

        ## truncate by removing modes which decay fast.
        # 'Fast' modes defined as those which decay by
        # more than 1 e-fold per timestep. These fast modes
        # won't be important for forecasts and make testing
        # harder for numerical reasons.
        decays_fast = sigma < -1
        gamma = gamma[~decays_fast]
        sigma = sigma[~decays_fast]
        omega = omega[~decays_fast]
        Lambda = Lambda[~decays_fast]
        U0 = U0[:, ~decays_fast]
        V0 = V0[:, ~decays_fast]

        ## Get cyclostationary eigenvectors
        V, U = self.get_cyclo_eigenvecs(V0, U0, gamma)

        ## decompose (periodic) left eigenvectors into harmonics
        if self.split_harmonics:
            ## decompose U into harmonic components.
            ## Note zeroth axis indexes harmonic and first axis indexes month
            U, mode_freq = fourier_expand(U, sum_conjugates=self.sum_conjugates)
            self.mode_freq = mode_freq

            ## transpose U such that zeroth axis indexes month and
            ## first axis indices harmonic.
            U = np.einsum("ij...->ji...", U)

        # ## Save pre-computed components
        self.V = V
        self.U = U
        self.Lambda = Lambda
        self.gamma = gamma
        self.sigma = sigma
        self.omega = omega

        return

    def compute_transition_mats(self):
        """Compute propagators for each month separately"""

        ## empty dictionary to hold propagators for each month
        K = {}

        ## Iterate through different months
        for m in np.arange(0, 12):
            ## Get subset of data points from given month
            is_m = m == self.month_labels
            X_m = self.X[:, is_m]
            Y_m = self.Y[:, is_m]

            ## Compute propagator, and add to dictionary
            KT = Y_m @ np.linalg.pinv(X_m, rcond=self.rcond)
            K[m] = KT.T

        return K

    def K(self, start_month, tau):
        """Get fundamental matrix, needed to advance
        state forwards in time. 'start_month' is integer
        in [0,11], and tau in integer in [0,inf] specifying number
        of months to step the model forwards in time"""

        ## initialize as identity matrix
        KT = np.eye(self.m)

        if tau >= 1:
            ## Get array of months
            idx = np.array([start_month + tau_ for tau_ in np.arange(0, tau)])

            ## multiply propagators for each month
            months = idx % 12
            for m in months:
                KT = self.K_m[m].T @ KT

        return KT.T

    def get_cyclo_eigenvecs(self, V0, U0, gamma):
        """Get cyclostationary modes"""

        ## Arrays to hold cyclostationary eigenvectors.
        ## First dimension is month dimension
        V = np.zeros([12, *V0.shape], dtype=complex)
        U = np.zeros([12, *U0.shape], dtype=complex)

        ## cycle through months
        for m in np.arange(0, 12):
            ## get scaling factor for each set of eigenvecs
            U_scale = np.exp(-gamma * m)
            V_scale = np.exp(-gamma * (12 - m))

            ## compute eigenvectors
            U[m] = self.K(0, m).T @ U0 @ np.diag(U_scale)
            V[m] = self.K(m, 12 - m) @ V0 @ np.diag(V_scale)

        return V, U

    def B(self, start_month):
        """Compute monodromy matrix"""

        B = self.K(start_month=start_month, tau=12)

        return B

    def predict_single(self, xt, m, tau):
        """Make forecast for single initial condition.
        Allows for multiple timesteps"""

        ## make sure dimensions are okay
        if xt.ndim == 1:
            xt = xt[:, None]

        elif xt.shape[1] > 1:
            print("Error: this function only predicts 1 I.C.")

        if tau.ndim == 0:
            tau = np.array([tau])

        ## Get projection onto right eigenvectors
        proj_0 = self.V[m].T @ xt

        ## Make eigenfunction forecast
        gamma_tau = np.einsum("k,t->kt", self.gamma, tau)
        proj_tau = np.einsum("kt,kj->kt", np.exp(gamma_tau), proj_0)

        ## Multiply with cyclomodes (don't sum over modes yet)
        m_prime = np.mod(m + tau, 12)
        forecast = np.einsum("t...mk,kt->...mt", self.U[m_prime], proj_tau)

        return forecast

    def predict(self, xt, m, tau):
        """
        Forecast x(t+tau) given initial conditions xt and
        lead times tau. Function computes Xi @ exp(Gamma*tau) @ xt
        - 'xt' is a 2-D array (m x n), where 1st dim is # of gridpoints
        and 2nd dim is number of initial conds.
        - 'm' is 1-D array (m) representing month for each initial cond.
        - 'tau' is a 1-D array (t) representing lead times to forecast for.
        For matrix multiplication/einsum, we represent dimensions
        using the following characters:
        m: state space dimension
        t: lead-time dimension
        x: init. cond. dimension
        """

        ## make sure dimensions are okay
        if xt.ndim == 1:
            xt = xt[:, None]

        if tau.ndim == 0:
            tau = np.array([tau])

        ## Get projection onto right eigenvectors
        proj_0 = np.einsum("nmk,mn->kn", self.V[m], xt)

        ## Make eigenfunction forecast
        gamma_tau = np.einsum("k,t->kt", self.gamma, tau)
        proj_tau = np.einsum("kt,kn->nkt", np.exp(gamma_tau), proj_0)

        ## Next, get `end months' for prediction
        ## depends on initial condition and lag
        ## einsum units of m_prime: 'nt'
        m_prime = np.mod(m[:, None] + tau[None, :], 12)

        ## Make forecast (note `...` dim handles Fourier modes
        ## if `split_harmonics`==True
        forecast = np.einsum("nt...mk,nkt->n...mt", self.U[m_prime], proj_tau)

        return forecast


class LIM_CS_FOURIER(LIM_CS):
    """
    Class to represent CS-LIM with fourier decomp. applied
    to cyclostationary modes. This model's modes are separated
    into +/- combination tones.
    """

    def __init__(
        self,
        X,
        Y,
        month_labels,
        rcond=1e-10,
    ):
        self.X = X
        self.Y = Y
        self.month_labels = month_labels
        self.rcond = rcond
        self.split_harmonics = False
        self.sum_conjugates = False

        ## Compute CS-LIM components
        super().compute()

        ## do Fourier computation
        self.compute()

        return

    def compute(self):
        """Perform Fourier decomp. of cyclostationary modes"""

        ## Specify period and count # of timesteps in period
        T = 12  # period
        self.N = self.U.shape[0]  # timesteps per period

        ## get Nyquist freq (units: cycles/period)
        nyq_freq = np.floor(self.N / 2).astype(int)

        ## Compute FFT
        U_fft = np.fft.fft(self.U, axis=0)
        freq = np.fft.fftfreq(U_fft.shape[0])

        ## convert freq. from cyc/timestep to rad/month
        timesteps_per_month = self.N / T
        rad_per_cycle = 2 * np.pi
        freq = freq * timesteps_per_month * rad_per_cycle

        ## Compute augmented eigenvalues (combination frequencies!)
        self.gamma_tilde, sort_idx = self.get_gamma_tilde(j_omega=freq)

        ## Get mode amplitudes for each start month.
        self.Utilde = self.get_Utilde(U_fft, freq, sort_idx)

        ## Alternative decomp.
        self.Uhat, self.Vtilde = self.get_Uhat_Vtilde(U_fft, freq, sort_idx)

        ## for convenience, get sorted frequencies
        self.freq_sorted = freq[sort_idx]

        return

    def predict_single(self, x0, m0, tau, sum_idx=None):
        """Make prediction for single initial condition.
        Args:
            - x0: initial condition
            - m0: month during initial condition
            - tau: lags to forecast for (scalar or 1-D array)
            - sum_idx: indices of modes to sum over. If `none',
                don't sum over modes
        """

        ## to-do: instead of using ellipses, explicitly broadcast
        ## x0 and m0 to 1-D arrays, then denote that dimension with n

        ## Project init cond. on right eigenvectors
        c = np.einsum("mk,mn->kn", self.V[m0], x0)

        ## compute exponential term
        gamma_tau = np.einsum("fk,t->ftk", self.gamma_tilde, tau)

        ## Make forecast
        xt = np.einsum("fmk,ftk,k...->fmt...", self.Utilde[m0], np.exp(gamma_tau), c)

        return xt

    def predict(self, x0, m0, tau, sum_k=True):
        """Make prediction for multiple initial conditions.
        Args:
            - x0: initial conditions (mxn)
            - m0: month during initial conditions (n)
            - tau: lags to forecast for (scalar or 1-D array) (t)
            - sum_idx: should we sum over modes?

        Einsum indices:
        'n': sample index (# of of init. conds)
        'm': state dimension
        'k': eigenvalue/vector index (# of eigenvalues retained)
        'f': frequency index (for harmonics)
        't': lead time index (length of tau)
        """

        ## Project init cond. on right eigenvectors
        c = np.einsum("nfmk,mn->fkn", self.Vtilde[m0], x0)

        ## compute exponential term
        gamma_tau = np.einsum("fk,t->ftk", self.gamma_tilde, tau)

        ## get text for einsum
        if sum_k:
            final_dims = "nfmt"
        else:
            final_dims = "nfmkt"

        ## make forecast
        xt = np.einsum(f"fmk,ftk,fkn->{final_dims}", self.Uhat, np.exp(gamma_tau), c)

        return xt

    def get_Uhat_Vtilde(self, fft, j_omega, idx):
        """
        Function to get amplitudes of `updated` eigenvectors.
        In this model, we're pushing the time-dependence of the
        state's evolution to the augmented eigenvalues, `gamma_tilde`.
        In the original cs_lim, the state evolves as:
            x(t+tau) = U(t+tau) @ exp(gamma*tau) @ V(t).T @ x(t).
        We're writing it as:
            x(t+tau) = sum_j Uhat_j @ exp(gamma_tilde_j * tau) @ V_j(t) @ x(t).
        Args:
            - `fft`: fourier amplitudes for each harmonic
            - `j_omega`: frequency of each harmonic, in rad/month
            - `idx`: indices needed to sort harmonics to match
                the augmented eigenvalues, `gamma_tilde`
        Returns:
            - 'Uhat': array with dims 'fmk'
            - 'Vtilde': array with dims 'tfmk'
        """

        ## Get exponential term
        months = np.arange(12)
        theta = np.einsum("f,t->ft", j_omega, months)
        exp_theta = np.exp(1j * theta)

        ## Get freq-dependent right eigenvectors
        Uhat = 1 / self.N * fft
        Vtilde = np.einsum("ft,tmk->ftmk", exp_theta, self.V)

        ## Get sorting indices to match eigenvalue ordering
        U_broadcast_arr = np.ones(Uhat.shape, dtype=int)
        V_broadcast_arr = np.ones(Vtilde.shape, dtype=int)
        U_idx = np.einsum("fk,fmk->fmk", idx, U_broadcast_arr)
        V_idx = np.einsum("fk,ftmk->ftmk", idx, V_broadcast_arr)

        ## Sort eigenvectors
        Vtilde = np.take_along_axis(Vtilde, V_idx, axis=0)
        Uhat = np.take_along_axis(Uhat, U_idx, axis=0)

        ## Swap time & freq. dims on Vtilde ("ftmk->tfmk")
        Vtilde = np.einsum("ftmk->tfmk", Vtilde)

        return Uhat, Vtilde

    def get_Utilde(self, fft, j_omega, idx):
        """
        Function to gett amplitudes of `updated` eigenvectors.
        In this model, we're pushing all the time-dependence of the
        state's evolution to the augmented eigenvalues, `gamma_tilde`.
        Therefore, the
        `Utilde' modes are closely related to the Fourier coefficients.
        Note that these modes depend on initial conditions (i.e.,
        the phasing depends on the start month for the forecast).
        Therefore, after forming the augmented eigenvalues, we absorb
        the phase-dependency of the initial condition in the Utilde
        modes. Then, as the state evolves from this initial condition,
        the Utilde modes remain fixed. In practice, this means that we
        have a different set of Utilde modes for each phase in the
        seasonal cycle, theta = (t/12) * 2pi. We could specify
        this fn to returns another fn (which takes in the month of
        the initial condition and returns the corresponding modes).
        However, for convenience, we pre-compute Utilde at integer
        values of months (which are used for forecasting).
        Args:
            - `fft`: fourier amplitudes for each harmonic
            - `j_omega`: frequency of each harmonic, in rad/month
            - `idx`: indices needed to sort harmonics to match
                the augmented eigenvalues, `gamma_tilde`
        """

        ## first, broadcast indices for sorting
        broadcast_arr = np.ones(fft.shape, dtype=int)
        idx_ = np.einsum("fk,fmk->fmk", idx, broadcast_arr)

        ## Get Fourier coefficients for reconstruction.
        ## Note: these depend on initial condition!
        ## Therefore, these coefficients are a fn. of start time
        ## (where start time `t' has a unit of months since Jan 1).
        Utilde_fn0 = (
            lambda t: 1
            / self.N
            * np.einsum("f...,f->f...", fft, np.exp(1j * j_omega * t))
        )

        ## sort eigenvectors to match augmented eigenvalues
        Utilde_fn = lambda t: np.take_along_axis(Utilde_fn0(t), indices=idx_, axis=0)

        ## in theory, U_hat is continuous (we can evaluate at any
        ## time!) In practice, with monthly data we'll only evaluate
        ## at integer months, we may as well pre-compute them
        ## (this will make things easier later on).
        Utilde = np.stack([Utilde_fn(t) for t in np.arange(12)], axis=0)

        return Utilde

    def get_gamma_tilde(self, j_omega):
        """
        Function computes `augmented` eigenvalues, which result
        from adding `fundamental` frequencies to harmonics of
        the seasonal cycle. Resulting eigenvalues are sorted by
        their imaginary component (frequency) in order of
        increasing magnitude. This results in mode ordering of:
            [zeroth (fund.), 1st diff, 1st sum, 2nd diff, 2nd sum, ...].
        Args:
            - `j_omega` is 1D-array containing (harmonic) frequencies
                of the seasonal cycle (i.e., `j` times `omega_s`).
        """

        ## get augmented eigenvalues (combination frequencies!)
        gamma_tilde = 1j * j_omega[:, None] + self.gamma[None, :]

        ## get indices to sort them in frequency
        sort_idx = np.argsort(np.abs(gamma_tilde.imag), axis=0)

        ## Do the sorting
        # orders as: [zero (fund.), 1st diff, 1st sum, 2nd diff, 2nd sum, ...]
        gamma_tilde = np.take_along_axis(gamma_tilde, indices=sort_idx, axis=0)

        ## fix frequencies above the nyquist frequency
        omega_tilde = copy.deepcopy(gamma_tilde.imag)
        idx_to_fix = np.abs(omega_tilde) > np.pi
        omega_tilde[idx_to_fix] = -np.sign(omega_tilde[idx_to_fix]) * (
            2 * np.pi - np.abs(omega_tilde[idx_to_fix])
        )
        gamma_tilde = gamma_tilde.real + 1j * omega_tilde

        return gamma_tilde, sort_idx

    def U_recon(self, months=np.arange(12), mode_idx=None):
        """
        Reconstruct cyclic eigenmodes from CS_LIM using
        specified Fourier modes. Don't need this to forecast;
        function is for convenience.
        Args:
            - 'months' is array specifying which months to reconstruct.
                Defaults to integers, [0,1,...,11].
            - 'mode_idx' is list of modes to use in reconstruction.
                If 'none', all modes are used.
        """

        ## evaluate Fourier functions at each month
        theta = np.einsum("fk,t->ftk", self.freq_sorted, months)
        exp_theta = np.exp(1j * theta)

        ## reconstruct eigenvectors in modal form
        U_recon_modal = np.einsum("fmk,ftk->ftmk", self.Uhat, exp_theta)

        ## sum over specified indices
        if mode_idx is None:
            return U_recon_modal.sum(0)

        else:
            return U_recon_modal[mode_idx].sum(0)

    def U_recon_from_Utilde(self, m0, tau=np.arange(0, 12), mode_idx=None):
        """
        Reconstruct cyclic eigenmodes from CS_LIM using
        specified Fourier modes. Don't need this to forecast;
        function is for convenience.
        Args:
            - m0 is starting month (scalar).
            - tau is array of lags at which to evaluate the mode.
            - mode_idx is list of modes to use in reconstruction.
                If 'none', all modes are used.
        """

        ## evaluate Fourier functions at specified lags
        omega_tau = np.einsum("fk,t->ftk", self.freq_sorted, tau)

        ## reconstruct eigenvectors in modal form
        U_recon_modal = np.einsum(
            "fmk,ftk->ftmk", self.Utilde[m0], np.exp(1j * omega_tau)
        )

        ## sum over specified indices
        if mode_idx is None:
            return U_recon_modal.sum(0)

        else:
            return U_recon_modal[mode_idx].sum(0)


################ LIM class ###################
class LIM:
    """
    Class to represent generalization of linear inverse model
    (i.e., with non-linear kernel).

    Attributes include:

    - Gxx, Gxy: Gram matrices.
    - Ktilde: projection of "K" matrix in standard LIM.
    - Vtilde, Utilde, Lambda: eig. decomp. of Ktilde.
    - conj_idx: conjugate indices of eigenvalues.
    - normal_modes: empirical normal modes (ENMs).
    - gamma, sigma, omega: timescales for ENMs.
    """

    def __init__(
        self,
        X,
        Y,
        kernel,
        kernel_grad=None,
        k=None,
        rcond=1e-10,
        lonlat_coord=None,
        kernel_reg=False,
        zero_idx=None,
        Gxx=None,
        Gxy=None,
    ):
        """
        'X' and 'Y' are matrices with training data.
        'k' is integer, specifying number of modes to truncate to.
        'kernel' is a function which takes in two matrices of equal size,
        and computes distance between pairs of columns.
        'kernel_grad' is function which computes the gradient of the kernel
        at specified point.
        'rcond' is cutoff for singular values when doing a pseudoinverse.
        'lonlat_coord' is xarray coordinate to use for koopman modes.
        'kernel_reg' is bool specifying whether to do kernel regression-style
        fitting. If True, choose koopman modes to minimize training loss;
        otherwise, choose modes to minimize data reconstruction error.
        Gxx and Gxy are pre-computed gram matrices.
        """

        self.X = X
        self.Y = Y
        self.k = k
        self.kernel = kernel
        self.kernel_grad = kernel_grad
        self.rcond = rcond
        self.lonlat_coord = lonlat_coord
        self.kernel_reg = kernel_reg
        self.zero_idx = zero_idx
        self.Gxx = Gxx
        self.Gxy = Gxy

        ## Compute LIM components
        self.compute()

        return

    def compute(self):
        """pre-compute components of LIM"""

        ## Compute gram matrices if necessary
        if self.Gxx is None:
            self.Gxx = self.kernel(self.X, self.X)

        if self.Gxy is None:
            self.Gxy = self.kernel(self.X, self.Y)

        ## Compute pseudo inverse of Gxx
        if self.k is None:
            Gxx_pinv = np.linalg.pinv(self.Gxx, rcond=self.rcond)
        else:
            Gxx_pinv, rcond = pinv(self.Gxx, k=self.k)
            self.rcond = rcond

        ## Ktilde
        Ktilde = Gxx_pinv @ self.Gxy.T

        ## Zero-out specified indices if desired.
        ## (note: current implementation only works for special
        ## case where data is pre-projected to EOFs)
        if self.zero_idx is not None:
            Ktilde = self.mask_Ktilde(Ktilde, Gxx_pinv=Gxx_pinv)

        ## Eigenvalue decomposition of operator
        Vtilde, Utilde, Lambda = eig_decomp(Ktilde)

        ## Rescale left eigenvectors so that U^T @ V = I
        Utilde = rescale_left_eigs(V=Vtilde, U=Utilde, Lambda=Lambda)

        ## Truncate eigs
        if self.k is not None:
            Vtilde = Vtilde[:, : self.k]
            Utilde = Utilde[:, : self.k]
            Lambda = Lambda[: self.k]

        ## Timescales for modes
        gamma, sigma, omega = get_timescales(Lambda)

        ## Get indices of complex conjugate eigenvalues
        conj_idx = get_conjugate_indices(Lambda)

        ## Compute Koopman modes (contain real and imaginary parts)
        if self.kernel_reg:
            Xi = self.Y @ Gxx_pinv @ Utilde @ np.diag(1 / Lambda)
        else:
            Xi = self.X @ Gxx_pinv @ Utilde

        ## convert to ENMs (only real parts)
        normal_modes = get_empirical_normal_modes(Xi, conj_idx, self.lonlat_coord)

        ## Define 'propagator' matrix (used for computing optimals)
        def P(tau):
            ## compute gamma*tau separately to catch runtime warning (mult. by -inf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                gamma_tau = gamma * tau

            return Xi @ np.diag(np.exp(gamma_tau)) @ Vtilde.T

        ## Save pre-computed components
        self.Gxx_pinv = Gxx_pinv
        self.Ktilde = Ktilde
        self.Vtilde = Vtilde
        self.Utilde = Utilde
        self.Lambda = Lambda
        self.gamma = gamma
        self.sigma = sigma
        self.omega = omega
        self.conj_idx = conj_idx
        self.Xi = Xi
        self.normal_modes = normal_modes
        self.P = P

        return

    def mask_Ktilde(self, Ktilde, Gxx_pinv):
        """mask out specified elements of Ktilde"""

        ## First, construct 'B' matrix
        B = np.ones(self.X.shape[0])
        B[self.zero_idx] = 0
        B = np.diag(B)

        ## Next, compute 'A' matrix
        A = Gxx_pinv @ self.X.T @ B @ self.X

        ## finally, mask Ktilde
        Ktilde_masked = A @ Ktilde

        return Ktilde_masked

    def varphi(self, x):
        """
        Evaluate eigenfunctions at specified datapoint
        varphi(x) = \tilde{V}^T K(X,x)
        """

        ## Make sure x is column vector
        if x.ndim == 1:
            x = x[:, None]

        return self.Vtilde.T @ self.kernel(self.X, x)

    def varphi_linearized(self, x, xbar):
        """
        Linearize varphi function about a specified mean state, xbar
        """

        ## Make sure x, xbar are column vectors
        if x.ndim == 1:
            x = x[:, None]

        if xbar.ndim == 1:
            xbar = xbar[:, None]

        ## Get linearized kernel
        dx = x - xbar
        Kbar = self.kernel(self.X, xbar)
        dK = self.kernel_grad(self.X, xbar) @ dx

        return self.Vtilde.T @ (Kbar + dK)

    def predict(self, xt, tau, Xi=None, xbar=None, lonlat_coord=None):
        """
        Forecast x(t+tau) given initial conditions xt and
        lead times tau. Function computes Xi @ exp(Gamma*tau) @ xt
        - 'xt' is a 2-D array, where 1st dim is # of gridpoints
            and 2nd dim is number of initial conds.
        - 'tau' is a 1-D array (number of lead times to forecast for).
        - 'Xi' are normal modes, needed to reconstruct 'real space'
            variables from eigenfunctions. These default to self.Xi.
        - 'xbar' is base state to linearize around. If specified,
            a linearized forecast is made. Otherwise, it is ignored.
        - 'lonlat_coord' is longitude/latitude coordinate (used
            for putting result in xr.dataarray)
        For matrix multiplication/einsum, we represent dimensions
        using the following characters:
        m: space dimension
        k: eigenvalue dimension
        t: tau dimension
        x: init. cond. dimension
        """

        ## set normal modes if unspecified
        if Xi is None:
            Xi = self.Xi

        ## set lonlat coord if unspecified
        if lonlat_coord is None:
            lonlat_coord = self.lonlat_coord

        ## Evaluate eigenfunctions at datapoints.
        ## If basic state is provided, evaluate linearized
        ## eigenfunctions instead.
        if xbar is None:
            varphi = self.varphi(xt)

        else:
            varphi = self.varphi_linearized(xt, xbar)

        ## loop method (faster if number of timesteps is relatively low)
        predict_onetime = lambda t: (Xi * np.exp(self.gamma * t)) @ varphi
        forecast = np.stack([predict_onetime(t) for t in tau], axis=1)

        # ## "pure" way to forecast (note einsum is slow...)
        # gamma_tau = np.einsum("k,t->kt", self.gamma, tau)
        # forecast = np.einsum(
        #     "mk,kt,kx->mtx", Xi, np.exp(gamma_tau), varphi
        # )

        ## Put in xarray if desired
        if lonlat_coord is not None:
            forecast = xr.DataArray(
                forecast,
                coords={
                    "lonlat": lonlat_coord,
                    "tau": tau,
                    "xt_idx": np.arange(xt.shape[1]),
                },
                dims=["lonlat", "tau", "xt_idx"],
            )
            forecast = forecast.unstack()

        return forecast

    def predict_decoupled(self, xt, tau, Xi, xbar=None, lonlat_coord=None, idx=None):
        """
        Wrapper function for 'predict'. Decouples the model into two components.
        One component is indexed by 'idx'; the other component contains the
        remaining elements. Note: Xi must be specified; for evolving the state
        vector forwards in time, the specified Xi should correspond to the
        """

        ## infer dimensions from inputs
        m, ni = xt.shape  # length of state vector and number of initial conds.
        nt = len(tau)  # number of lead times to forecast for
        m_out = Xi.shape[0]  # number of output variables

        ## Convert indices to boolean array
        idx_bool = np.zeros(m, dtype=bool)
        idx_bool[idx] = True

        ## make empty array to hold result
        forecast = np.zeros([m_out, nt, ni])

        ## loop through model components
        for idx_ in [idx_bool, ~idx_bool]:
            ## decouple output if Xi==None is specified.
            ## if Xi is provided (e.g., for variable not in state vector)
            ## use the full Xi.
            if Xi is None:
                print("indexing Xi here")
                Xi_ = self.Xi[idx_]
            else:
                Xi_ = Xi

            ## To `decouple' the input, set elements not indexed by 'idx_'
            ## to zero.
            xt_ = copy.deepcopy(xt)
            xt_[~idx_, :] = 0.0

            print(f"idx_ shape: {idx_.shape}")
            print(f"idx_ sum: {idx_.sum()}")
            print(f"Xi_ shape: {Xi_.shape}")
            print(f"xt_ shape: {xt_.shape}")
            print(f"\n")

            ## make forecast
            forecast[idx_, ...] = self.predict(
                xt=xt_, tau=tau, Xi=Xi_, xbar=xbar, lonlat_coord=None
            )

        ## Put in xarray if desired
        if lonlat_coord is not None:
            forecast = xr.DataArray(
                forecast,
                coords={
                    "lonlat": lonlat_coord,
                    "tau": tau,
                    "xt_idx": np.arange(xt.shape[1]),
                },
                dims=["lonlat", "tau", "xt_idx"],
            )
            forecast = forecast.unstack()

        return forecast

    def predict_single(self, xt, tau):
        """Make forecast for single initial condition and lead time.
        Used to verify the vectorized 'predict' function"""

        forecast = self.Xi @ np.diag(np.exp(self.gamma * tau)) @ self.varphi(xt)

        return forecast

    def predict_loop(self, xt, tau):
        """Wrapper function to make forecast for multiple initial conditions
        and lead times (compare to 'predict' function)"""

        forecast = []
        for ti, t in enumerate(tau):
            for xi in range(xt.shape[1]):
                forecast.append(self.predict_single(xt[:, xi : xi + 1], tau[ti]))
        forecast = np.stack(forecast, axis=1).reshape(-1, len(tau), xt.shape[1])

        ## Put in xarray if desired
        if self.lonlat_coord is not None:
            forecast = xr.DataArray(
                forecast,
                coords={
                    "lonlat": self.lonlat_coord,
                    "tau": tau,
                    "xt_idx": np.arange(xt.shape[1]),
                },
                dims=["lonlat", "tau", "xt_idx"],
            )
            forecast = forecast.unstack()

        return forecast

    def get_optimal_helper(self, M):
        """
        Helper function for 'get_optimal_IC' and 'get_stochastic_optimal.
        Performs eigenvalue decomposition on 'M' matrix and converts
        back to  'real' space.
        """

        ## Compute Z matrix
        Z = self.Gxx @ M @ self.Gxx

        ## compute eigenvalue decomp. (generalized eigenvalue problem)
        eps = np.finfo(np.float32).eps * np.std(np.diag(self.Gxx))
        eps_diag = eps * np.eye(self.Gxx.shape[0])
        B = self.Gxx + eps_diag
        V_Z, Lambda_Z = eigh_decomp(A=Z, B=B)

        ## Get leading mode
        alpha = V_Z[:, 0]

        ## convert back to real space
        if self.kernel_reg:
            x_opt = (
                self.X @ self.Gxx_pinv @ self.Utilde @ self.Vtilde.T @ self.Gxx @ alpha
            )
        else:
            x_opt = self.Xi @ self.Vtilde.T @ self.Gxx @ alpha

        ## Put in xarray if desired
        if self.lonlat_coord is not None:
            x_opt = xr.DataArray(x_opt, coords={"lonlat": self.lonlat_coord}).unstack()

        ## rescale by standard deviation for convenience
        x_opt /= x_opt.std()

        return x_opt

    def get_optimal_IC(self, tau, N=None):
        """Compute optimal initial condition for specified lag and norm"""

        ## Specify norm if none is provided
        if N is None:
            N = np.ones(self.X.shape[0])

        ## Lag-dependent propagator matrix
        P = self.P(tau)

        ## Compute M matrix
        if N.ndim == 2:
            M = P.conj().T @ N @ P
        else:
            M = P.conj().T @ (N[:, None] * P)

        ## Get optimal
        x0_opt = self.get_optimal_helper(M)

        return x0_opt

    def get_stochastic_optimal(self, tau, N=None, nt=20, verbose=True):
        """
        Compute optimal stochastic forcing pattern for specified lag and norm.
        'nt' is the number of sums to use when approximating the integral.
        """

        ## Specify norm if none is provided
        if N is None:
            N = np.ones(self.X.shape[0])

        ## specify bounds of integration and timestep
        dt = tau / nt  # time step for integral
        ti = dt / 2
        tf = tau - dt / 2

        ## Compute integral
        M = np.zeros([self.X.shape[1], self.X.shape[1]])  # empty array to hold result
        for t in tqdm.tqdm(np.arange(ti, tf + dt, dt), disable=(not verbose)):
            ## get propagator
            P = self.P(tau - t)

            ## if norm is provided in vector form, handle multiplication differently
            ## (much faster compared to matrix form)
            if N.ndim == 2:
                M = M + P.conj().T @ N @ P * dt
            else:
                M = M + P.conj().T @ (N[:, None] * P) * dt

        ## Get optimal
        f_opt = self.get_optimal_helper(M)

        return f_opt


class LIM_reg(LIM):
    """
    Different solution method for LIM, based on regularization.
    """

    def __init__(
        self,
        X,
        Y,
        kernel,
        epsilon=1e-3,
        compute_backwards=False,
    ):
        """
        'X' and 'Y' are matrices with training data.
        'kernel' is a function which takes in two matrices of equal size,
            and computes distance between pairs of columns.
        'epsilon' is regularization factor.
        'compute_backwards' specifies whether to stabilize approximation
            by estimating K_tilde^{-1} and averaging with K_tilde
        """

        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.epsilon = epsilon
        self.compute_backwards = compute_backwards

        ## Compute LIM components
        self.compute()

        return

    def compute(self):
        """pre-compute components of LIM"""

        ## Compute gram matrices
        Gxx = self.kernel(self.X, self.X)
        Gxy = self.kernel(self.X, self.Y)

        ## Ktilde
        n = Gxx.shape[0]
        Gxx_reg = Gxx + n * self.epsilon * np.eye(n)
        Ktilde = scipy.linalg.solve(Gxx_reg, Gxy.T, assume_a="sym")

        ## compute backwards approximation if desired
        if self.compute_backwards:
            ## get backwards approximation
            Gxy_reg = Gxy + n * self.epsilon * np.eye(n)
            Ktilde_inv = scipy.linalg.solve(Gxy_reg.T, Gxx, assume_a="sym")
            Ktilde_backwards = np.linalg.inv(Ktilde_inv)

            ## update approxmation for Ktilde
            Ktilde = 0.5 * (Ktilde + Ktilde_backwards)

        ## Eigenvalue decomposition of operator
        Vtilde, _, Lambda = eig_decomp(Ktilde)

        ## Timescales for modes
        gamma, sigma, omega = get_timescales(Lambda)

        ## Update units of sigma, omega for convenience.
        # sigma: 1/month => 1/year
        # omega: rad/month => 1/year
        sigma = 12 * copy.deepcopy(sigma)
        omega = 12 / (2 * np.pi) * copy.deepcopy(omega)

        ## Get indices of slowly-decaying components
        ## def. as anything with e-fold > 2 months (Nyquist freq)
        efold = -1 / sigma
        decays_slowly = efold > 1 / 6

        ## Save pre-computed components
        self.Vtilde = Vtilde[:, decays_slowly]
        self.Lambda = Lambda
        self.gamma = gamma
        self.sigma = sigma
        self.omega = omega

        ## initialize decoder
        self.decoder = None

        return

    def predict_varphi(self, X_0, tau):
        """
        Make *eigenfunction* predictions from initial conditions X_0
        at specified lead times tau. Assumes self.enso_idx is not None.
        I.e., returns eigenfunction predictions for single eigenfunction

        Args:
            - X_0: initial conditions, with shape (m x n)
            - tau: lead times, with shape (t)

        Returns:
            - varphi_tau: (t x n) array of eigenfunction predictions
        """

        ## Evaluate eigenfunctions at initial condition.
        ## Shape: n, where 'n' is number of initial conditions.
        varphi_0 = self.varphi(X_0, eig_idx=self.enso_idx).flatten()

        ## get eigenvalue for enso index
        gamma = self.gamma[self.enso_idx]

        ## propagate forwards in time
        varphi_tau = np.einsum("t,n->tn", np.exp(gamma * tau), varphi_0)

        return varphi_tau

    def predict(self, X_0, tau, month_idx):
        """
        Make predictions from initial conditions X_0 at specified lead times tau.
        Args:
            - X_0: initial conditions, with shape (m_in x n)
            - tau: lead times, with shape (t)
            - month_idx: month indices at each forecast verification time (t x n)

        Returns:
            - X_tau: (m_out x t x n) array of Niño 3.4 predictions
        """

        if self.decoder is None:
            print("No decoder!")
            return

        ## get eigenfunction predictions
        varphi_tau = self.predict_varphi(X_0=X_0, tau=tau)

        ## decode the eigenvalue forecasts
        X_tau = []
        for tau_i, _ in enumerate(tau):
            ## get data for given lead time
            kwargs = dict(c_e=varphi_tau[tau_i], month=month_idx[tau_i])

            ## make prediction
            X_tau.append(self.decoder.predict(**kwargs))

        ## Convert back to numpy array: stack 't' (m_out x n)  arrays,
        ## so resulting shape is (m_out x t x n)
        X_tau = np.stack(X_tau, axis=1)

        return X_tau

    def varphi(self, x, eig_idx=None):
        """
        Evaluate eigenfunctions at specified datapoint:
            varphi(x) = V_tilde^T @ K(X,x)
        'eig_idx' is either None or a scalar. If a scalar, only evaluate
            eigenfunction at the specified index
        """

        ## Make sure x is column vector
        if x.ndim == 1:
            x = x[:, None]

        ## subset eigenfunctions if necessary
        if eig_idx is None:
            Vtilde = self.Vtilde

        else:
            Vtilde = self.Vtilde[:, eig_idx : eig_idx + 1]

        return Vtilde.T @ self.kernel(self.X, x)

    def fit_decoder(
        self,
        nmodes_e,
        nmodes_c,
        zero_mode,
        X,
        y,
        eig_idx,
        month_idx,
        q,
        eigfn_eval=None,
        nonlinear_r=False,
    ):
        """
        Fit decoder object based on specified number of modes.
        Args:
            - nmodes_e, nmodes_c: number of fourier modes for fundamental/seasonal cycle
            - zero_mode: should we include zeroth fourier mode for enso? (boolean)
            - X: (m_in x n) array of state vector data used to predict Niño 3.4
            - y: (m_out x n) array of target data for decoder
            - eig_idx: index of ENSO eigenfunction
            - month_idx: (n) array containing index of month for each sample
            - q: cutoff percentile for c_e when fitting to data (discard data below
                this percentile)
        """

        ## initialize decoder
        U = decoder(
            nmodes_e=nmodes_e,
            nmodes_c=nmodes_c,
            zero_mode=zero_mode,
            nonlinear_r=nonlinear_r,
        )

        ## Evaluate eigenfunctions at data
        if eigfn_eval is None:
            print("Evaluating eigenfunction...")
            eigfn_eval = self.varphi(X, eig_idx=eig_idx)

        ## fit to data
        U.fit(y=y, c_e=eigfn_eval.flatten(), month=month_idx, cutoff_perc=q)

        ## save decoder and corresponding eigenfunction index
        self.decoder = U
        self.enso_idx = eig_idx

        return

    def plot_decoder(
        self,
        k_idx,
        j_idx,
        y_idx=0,
        show_traj=False,
        month_idx=11,
        r_mag=1.0,
        amp=None,
        fig=None,
        ax=None,
    ):
        """plot decoder"""

        fig, ax, plot_data = self.decoder.plot(
            k_idx,
            j_idx,
            y_idx=y_idx,
            show_traj=show_traj,
            omega_e=self.omega[self.enso_idx],
            month_idx=month_idx,
            r_mag=r_mag,
            amp=amp,
            fig=fig,
            ax=ax,
        )

        return fig, ax, plot_data

    def plot_composite(
        self, dataset, eig_idx, n_slice_comp=16, q=60, use_train=True, c_e=None
    ):
        """
        Plot composite of Niño 3.4 on eigenfunction.
        Args:
            - dataset: lim_dataset object
            - eig_idx: index of eigenvalue
            - n_slice_comp: number of slices for the composite
            - q: percentile cutoff for composite
            - use_train: should we use training data? If not, use validation
        Returns fig,ax
        """

        ## get data for composite
        if use_train:
            XY = dataset.XY_train
            n34 = dataset.n34_train
        else:
            XY = dataset.XY_valid
            n34 = dataset.n34_valid

        ## Get evaluated eigenfunction, then mag. and phase
        if c_e is None:
            c_e = self.varphi(XY, eig_idx=eig_idx).flatten()

        ## get magnitude/angle
        mag_e = np.abs(c_e)
        theta_e = src.utils.get_angle(c_e)

        ## Get composite
        theta_plot, comp_mean, comp_std = src.utils.get_nino_comp(
            n34=n34, theta=theta_e, mag=mag_e, n_slice=n_slice_comp, cutoff_perc=q
        )

        ## shift so max occurs at pi/2
        # first, get index for current Niño and pi/2
        nino_idx = np.argmax(comp_mean)
        pi_2_idx = np.argmin(np.abs(theta_plot - np.pi / 2))

        # next, roll timeseries
        nroll = pi_2_idx - nino_idx + 1
        comp_mean = np.roll(comp_mean, nroll)
        comp_std = np.roll(comp_std, nroll)

        ## func to get range of array
        get_range = lambda x: x.max() - x.min()

        #### make plot
        fig, ax = plt.subplots(figsize=(4, 3))

        ## Plot Nino3.4 composite
        ax.plot(theta_plot, comp_mean, label="composite", c="k")
        ax.fill_between(
            theta_plot, comp_mean + comp_std, comp_mean - comp_std, alpha=0.1, color="k"
        )

        ## Plot sin(theta)
        amp = 1 / 2 * get_range(comp_mean)
        ax.plot(
            theta_plot,
            amp * np.sin(theta_plot),
            label=f"$\sin(\\theta)$",
            c="k",
            ls=":",
        )

        ## set axis labels/ticks
        ax.axhline(0, ls="-", c="k", lw=0.5)
        ax.set_xticks(
            [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
            labels=["0", f"$\pi/2$", f"$\pi$", f"$3\pi/2$", f"$2\pi$"],
        )
        ax.set_xlabel("Phase")
        ax.set_ylabel("Nino 3.4")
        ax.legend(prop={"size": 7})

        ## Set axis limits
        amp = np.max(np.abs(comp_mean[:, None] + comp_std[None, :]))
        ax.set_ylim([-amp, amp])

        return fig, ax

    def get_traj(
        self,
        month_idx,
        k_sum,
        j_sum,
        normalize=True,
        theta_0=None,
        decay=False,
        r0=1.0,
    ):
        """
        Get trajectory from peak Niño/Niña given initial condition in given month.
        Args:
            - month_idx: month to start the trajectory from
            - theta_0: eigenfunction phase to start trajectory from
            - k_sum: Fourier modes to sum over for theta_e
            - j_sum: Fourier modes to sum over for theta_s
            - normalize: should we set max(abs(output))=2?
            - decay: should we damp eigenfunction? (TO-DO: implement this)
        Returns:
            - time: array
            - (m x n) array representing evolution of each target variable
        """

        ## specify number of points in trajectory
        n = 50

        ## set theta initial condition if not specified
        if theta_0 is None:
            theta_0 = self.decoder.get_nino_theta(month_idx=month_idx, r_mag=r0)

        ## get theta trajectory
        theta_traj = np.mod(np.linspace(theta_0, theta_0 + 2 * np.pi, n), 2 * np.pi)

        ## get corresponding eigenfunction trajectory
        c_e_traj = r0 * np.exp(1j * theta_traj)

        ## Get month trajectory and elapsed time in years
        period_years = 1 / self.omega[self.enso_idx]
        period_months = period_years * 12
        month_traj = np.mod(np.linspace(month_idx, month_idx + period_months, n), 12)
        ## Get elapsed time in years
        time_elapsed_years = np.linspace(0, period_years, n)

        ## add decay if specified
        if decay:
            c_e_traj *= np.exp(self.sigma[self.enso_idx] * time_elapsed_years)

        ## Get trajectory of target variable
        y_traj = self.decoder.predict(
            c_e=c_e_traj, month=month_traj, k_sum=k_sum, j_sum=j_sum
        )

        ## Map to Niño 3.4 if desired
        if self.decoder.get_n34 is not None:
            y_traj = self.decoder.get_n34(
                y_traj.squeeze(), month=month_traj, k_sum=k_sum, j_sum=j_sum
            )

        ## rescale so max of each variable is 2 over trajectory
        if normalize:
            y_traj *= 2 / np.max(np.abs(y_traj), axis=1, keepdims=True)

        return y_traj, time_elapsed_years

    def plot_eig_evo(
        self, dataset, start_idx, n, eig_idx=None, is_train=True, c_e=None
    ):
        """
        Plot evolution of eigenfunction in complex plane.
        Args:
            - start_idx: index (in time) for initial point
            - n: number of points to plot
            - eig_idx: index of eigenfunction for plotting
            - is_train: should we plot evaluation on training data?
        """

        ## get indices to subset relevant part of the data
        idx = np.arange(start_idx, start_idx + n)

        ## evaluate eigenfunctions if necessary
        if c_e is None:
            ## get relevant part of the data
            XY = dataset.XY_train if is_train else dataset.XY_val
            XY_ = XY[:, idx]

            ## Set eigenfunction index if not specified
            if eig_idx is None:
                eig_idx = self.enso_idx

            ## evaluate eigenfunction at datapoints
            c_e_ = self.varphi(XY_, eig_idx=eig_idx).flatten()

        else:
            c_e_ = c_e[idx]

        ## Make plot
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_aspect("equal")

        ## plot data
        ax.plot(c_e_.real, c_e_.imag, lw=1)

        ## Plot mean, start, and end points
        ax.scatter(c_e_.real.mean(), c_e_.imag.mean(), c="k", s=50, label="mean")
        for i, c, label in zip([0, -1], ["g", "r"], ["start", "end"]):
            ax.scatter(c_e_.real[i], c_e_.imag[i], c=c, s=50, label=label)

        ## remove xticks and add legend
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(prop={"size": 8})

        return fig, ax


class LIM_reg_explicit(LIM_reg):
    """Non-kernel version of LIM_reg"""

    def __init__(self, X, Y, psi, epsilon):
        """
        'X' and 'Y' are matrices with training data.
        'psi' is a function which transforms input data
        'epsilon' is regularization factor.
        """

        self.X = X
        self.Y = Y
        self.psi = psi
        self.epsilon = epsilon

        ## Compute LIM components
        self.compute()

        return

    def compute(self):
        """fit components of LIM to data"""

        ## Transform inputs
        Psi_x = self.psi(self.X)
        Psi_y = self.psi(self.Y)

        ## Get covariance matrices
        Cxx = Psi_x @ Psi_x.T
        Cxy = Psi_x @ Psi_y.T

        ## add regularization
        n = Cxx.shape[0]
        Cxx_reg = Cxx + n * self.epsilon * np.eye(n)

        ## Get least squares solution for K.
        ## Note:   Y = K.T @ X
        ##    => Y.T = X.T @ K
        ##  => X@Y.T = X@X.T @ K
        ##  => Y@X.T =
        K = scipy.linalg.solve(a=Cxx_reg, b=Cxy, assume_a="sym")

        ## eig decomp. of operator
        V, U, Lambda = src.lim.eig_decomp(K)

        ## rescale left eigs such that U.T@V = V.T@U = I
        U = src.lim.rescale_left_eigs(V=V, U=U, Lambda=Lambda)

        ## get timescales
        gamma, sigma, omega = src.lim.get_timescales(Lambda)

        ## Update units of sigma, omega for convenience:
        # sigma: 1/month => 1/year
        sigma = 12 * copy.deepcopy(sigma)

        # omega: rad/month => 1/year
        omega = 12 / (2 * np.pi) * copy.deepcopy(omega)

        ## save results
        self.K = K
        self.V = V
        self.U = U
        self.Lambda = Lambda
        self.gamma = gamma
        self.sigma = sigma
        self.omega = omega

        return

    def varphi(self, x, eig_idx=None):
        """
        Evaluate eigenfunctions at specified datapoint:
            varphi(x) = V^T @ psi(x)
        'eig_idx' is either None or a scalar. If a scalar, only evaluate
            eigenfunction at the specified index
        """

        ## Make sure x is column vector
        if x.ndim == 1:
            x = x[:, None]

        ## subset eigenfunctions if necessary
        if eig_idx is None:
            V = self.V

        else:
            V = self.V[:, eig_idx : eig_idx + 1]

        return V.T @ self.psi(x)


class LIM_stripped:
    """
    Stripped-down version of LIM_orig used to make predictions for specified
    linear operator
    """

    def __init__(
        self,
        L,
        lonlat_coord=None,
        Z=None,
    ):
        """
        'L' is linear operator (a matrix)
        'lonlat_coord' is xarray coordinate to use for koopman modes.
        """

        self.L = L
        self.lonlat_coord = lonlat_coord
        self.Z = Z

        ## Compute LIM components
        self.compute()

        return

    def compute(self):
        """pre-compute components of LIM"""

        ## Eigenvalue decomposition of operator
        V, U, gamma = eig_decomp(self.L.T)

        ## Rescale left eigenvectors so that U^T @ V = I
        U = rescale_left_eigs(V=V, U=U, Lambda=gamma)

        ## Get koopman modes
        if self.Z is None:
            Xi = U
        else:
            Xi = self.Z @ U

        ## Save pre-computed components
        self.V = V
        self.U = U
        self.gamma = gamma
        self.Xi = Xi

        return

    def predict(self, xt, tau, lonlat_coord=None):
        """
        Forecast x(t+tau) given initial conditions xt and
        lead times tau. Function computes U @ exp(Gamma*tau) @ V.T @ xt
        - 'xt' is a 2-D array, where 1st dim is # of gridpoints
            and 2nd dim is number of initial conds.
        - 'tau' is a 1-D array (number of lead times to forecast for).
        - 'lonlat_coord' is longitude/latitude coordinate (used
            for putting result in xr.dataarray)
        """

        ## set lonlat coord if unspecified
        if lonlat_coord is None:
            lonlat_coord = self.lonlat_coord

        ## Compute gamma tau
        gamma_tau = np.einsum("k,t->kt", self.gamma, tau)

        ## evaluate eigenfuncs. at data points
        varphi = self.V.T @ xt

        ## loop method (faster if number of timesteps is relatively low)
        predict_onetime = lambda t: (self.Xi * np.exp(self.gamma * t)) @ varphi
        forecast = np.stack([predict_onetime(t) for t in tau], axis=1)

        ## Put in xarray if desired
        if lonlat_coord is not None:
            forecast = xr.DataArray(
                forecast,
                coords={
                    "lonlat": lonlat_coord,
                    "tau": tau,
                    "xt_idx": np.arange(xt.shape[1]),
                },
                dims=["lonlat", "tau", "xt_idx"],
            )
            forecast = forecast.unstack()

        return forecast


def get_modes(X, idx=None, k=None):
    """Get modes for projection. Written specifically for
    used with a 'LIM_proj' object. 'idx' is a dictionary specifying
    which indices to use for SVD decomp. Useful for performing
    SVD on sepaate variables, for example.
    To-do: understand why special truncation needed"""

    ## if 'idx' not supplied, perform single SVD
    if idx is None:
        idx = {"all": np.arange(X.shape[0])}

    ## empty matrix to hold modes
    Z = []

    ## Get truncation
    rank = np.linalg.matrix_rank(X)
    k_ = np.floor(rank / len(idx)).astype(int)
    if k is None:
        k = k_
    else:
        k = np.min([k, k_])

    ## compute svd for each list of indices
    for idx_ in idx.values():
        ## empty array to hold spatial modes
        dim0 = X.shape[0]
        dim1 = np.min([X.shape[1], len(idx_)])
        Z_ = np.zeros([dim0, dim1])

        ## Get relevant subset of data
        X_ = X[idx_]

        ## Perform SVD on data subset
        Z_[idx_, :], _, _ = np.linalg.svd(X_, full_matrices=False)

        ## Truncate result
        Z_ = Z_[:, :k]

        ## append to results
        Z.append(Z_)

    ## Concatenate to form new modes
    Z = np.concatenate(Z, axis=1)

    return Z


class LIM_proj(LIM):
    """Modified version of kernel LIM in which data is manually projected
    onto specified modes. To-do: write tests!"""

    def __init__(
        self,
        X,
        Y,
        k,
        svd_idx=None,
        zero_idx=None,
        rcond=1e-10,
        Gxx=None,
        Gxy=None,
        kernel=None,
        kernel_grad=None,
        k_proj=None,
    ):
        """
        'k' is number of modes to truncate model at.
        'k_proj' is number of modes to truncate data at for the EOF projection.
        """

        ## Get modes for EOF-based truncation
        self.Z = get_modes(X, idx=svd_idx, k=k_proj)
        self.k = k

        ## project data onto specified modes
        self.X = self.Z.T @ X
        self.Y = self.Z.T @ Y

        ## specify other parameters
        self.kernel = kernel
        self.kernel_grad = kernel_grad
        self.rcond = 1e-10
        self.lonlat_coord = None
        self.kernel_reg = False
        self.zero_idx = zero_idx
        self.rcond = rcond
        self.Gxx = Gxx
        self.Gxy = Gxy

        ## compute
        super().compute()

        ## convert normal modes back to 'real' space
        self.Xi = self.Z @ self.Xi

        return

    def project(self, x):
        """convenience function to project data into EOF space"""

        ## convert 1-D input into vector
        if x.ndim == 1:
            x = x[:, None]

        return self.Z.T @ x

    def varphi(self, x):
        """project input data, before computing"""

        ## project
        x_proj = self.project(x)

        return super().varphi(x_proj)

    def varphi_linearized(self, x, xbar):
        """project input data before computing"""

        ## project
        x_proj = self.project(x)
        xbar_proj = self.project(xbar)

        return super().varphi_linearized(x=x_proj, xbar=xbar_proj)

    def predict(self, xt, tau, Xi=None, xbar=None):
        """make prediction in 'real' space"""

        ## get forecast in PC space
        args = {"xt": xt, "tau": tau, "Xi": Xi, "xbar": xbar}
        forecast = super().predict(**args)

        return forecast


class LIM_poly_explicit(LIM_orig):
    """Equivalent to kernel LIM with polynomial of degree 2, but
    with explicit feature space. Used for interaction shutoff
    experiments.
    - 'psi' is a function which lifts data to induced feature space"""

    def __init__(
        self,
        X,
        Y,
        k,
        psi,
        k_proj=None,
        svd_idx=None,
        zero_idx=None,
    ):
        ## Get modes for EOF-based truncation
        Z = get_modes(X, idx=svd_idx, k=k_proj)

        ## Linearly project data onto specified modes
        self.X = psi(Z.T @ X)
        self.Y = psi(Z.T @ Y)

        ## Set other parameters
        # Note: truncate_Y doesn't do anything; setting for backwards
        # compatibility.
        self.psi = psi
        self.k = k
        self.k_proj = k_proj
        self.lonlat_coord = None
        self.no_truncate = True
        self.truncate_Y = False
        self.svd_idx = svd_idx
        self.zero_idx = zero_idx
        self.rcond = 1e-10
        self.Vtilde = None

        ## Compute
        super().compute()

        ## override LIM_orig's setting for the following vars:
        self.Z = Z
        # self.Xi = X @ np.linalg.pinv(
        #     self.Vhat.T @ self.X, rcond=self.rcond
        # )
        self.Xi = (self.Z @ self.Z.T @ X) @ np.linalg.pinv(
            self.Vhat.T @ self.X, rcond=self.rcond
        )

        return

    def varphi(self, xt):
        """evaluate eigenfunctions at datapoints"""

        ## compute projection on eigenvectors
        x_proj = self.Z.T @ xt
        varphi_eval = self.Vhat.T @ self.psi(x_proj)

        return varphi_eval

    def predict(self, xt, tau, Xi=None, lonlat_coord=None):
        """
        Forecast x(t+tau) given initial conditions xt and
        lead times tau. Function computes Xi @ exp(Gamma*tau) @ xt
        - 'xt' is a 2-D array, where 1st dim is # of gridpoints
            and 2nd dim is number of initial conds.
        - 'tau' is a 1-D array (number of lead times to forecast for).
        - 'Xi' are normal modes, needed to reconstruct 'real space'
            variables from eigenfunctions. These default to self.Xi.
        - 'lonlat_coord' is longitude/latitude coordinate (used
            for putting result in xr.dataarray)
        For matrix multiplication/einsum, we represent dimensions
        using the following characters:
        m: space dimension
        k: eigenvalue dimension
        t: tau dimension
        x: init. cond. dimension
        """

        ## set normal modes if unspecified
        if Xi is None:
            Xi = self.Xi

        ## set lonlat coord if unspecified
        if lonlat_coord is None:
            lonlat_coord = self.lonlat_coord

        ## evaluate init conds' proj on eigenfunctions
        varphi_eval = self.varphi(xt)

        ## loop method (faster if number of timesteps is relatively low)
        predict_onetime = lambda t: (Xi * np.exp(self.gamma * t)) @ varphi_eval
        forecast = np.stack([predict_onetime(t) for t in tau], axis=1)

        ## Put in xarray if desired
        if lonlat_coord is not None:
            forecast = xr.DataArray(
                forecast,
                coords={
                    "lonlat": lonlat_coord,
                    "tau": tau,
                    "xt_idx": np.arange(xt.shape[1]),
                },
                dims=["lonlat", "tau", "xt_idx"],
            )
            forecast = forecast.unstack()

        return forecast


def get_timescales(Lambda, tau=1):
    """Given Lambda and tau, estimate gamma, sigma, and omega.
    We assume relationship of the form:
        Lambda = exp(gamma * tau)
        gamma = sigma + i * omega

        => gamma = 1/tau * ln(Lambda)
    """

    ## empty array to hold gamma
    gamma = np.zeros_like(Lambda)

    ## find indices where Lambda==0, and set these to minus inf.
    zero_idx = Lambda == 0
    gamma[zero_idx] = -np.inf
    gamma[~zero_idx] = 1 / tau * np.log(Lambda[~zero_idx])

    ## for convenience, separate real and imag. parts
    sigma = gamma.real
    omega = gamma.imag

    return gamma, sigma, omega


def argsort_eigs(eigs):
    """Get indices to sort eigenvalues"""

    ## very small number (used to break ties in sorting)
    eps = np.finfo(float).eps

    ## Get magnitude of eigenvalues
    eig_mag = eigs * eigs.conj()

    ## is the imaginary part of the eigenvalue positive?
    imag_is_pos = eigs.imag > 0

    ## sort by magnitude, then by imaginary part of eigenvalue
    idx = np.argsort(eig_mag + eps * imag_is_pos)[::-1]

    # ## old version: only sort by magnitude
    # idx = np.argsort(eigs * eigs.conj())[::-1]

    return idx


def eig_decomp(A):
    """eigenvector/eigenvalue decomposition of matrix A.
    If k is specified, Only keep the top 'k' modes."""

    ## Eigenvalue decomposition
    Lambda, U, V = scipy.linalg.eig(A, right=True, left=True)

    ## Scipy orders left eigs such that V^H U = U^H V = I
    ## We take comp. conj. to relabel them such that V^T U = U^T V = I
    U = U.conj()

    ## Get indices to sort eigenvalues in descending order
    idx = argsort_eigs(Lambda)
    Lambda = Lambda[idx]
    U = U[:, idx]
    V = V[:, idx]

    return V, U, Lambda


def eigh_decomp(A, B=None):
    """
    Generalized eigenvector/eigenvalue decomposition of hermitian matrix A.
    B is optional RHS matrix (also hermitian); default to identity
    """

    ## Eigenvalue decomposition
    Lambda, V = scipy.linalg.eigh(a=A, b=B)

    ## Get indices to sort eigenvalues in descending order
    idx = argsort_eigs(Lambda)
    Lambda = Lambda[idx]
    V = V[:, idx]

    return V, Lambda


def rescale_left_eigs(V, U, Lambda):
    """
    Rescale left eigenvectors U such that vi^T @ ui = 1.
    'V': right eigenvectors
    'U': (unscaled) left eigenvectors
    'Lambda': corresponding eigenvalues
    Returns re-scaled left eigenvectors.
    """

    ## Get magnitude of diagonal, for rescaling
    mag = np.diag(V.T @ U).copy()

    ## Get scale factor (add eps to avoid div. by zero)
    scale = 1 / (mag + np.finfo(float).eps)

    ## zero out eigenvectors with small eigenvalues
    cond = np.abs(Lambda) / np.max(np.abs(Lambda))
    scale[cond < 1e-12] = 0

    ## do the rescaling
    U_scaled = U * scale[None, :]

    return U_scaled


def rescale_right_eigs(Vhat, Vtilde, Xhat):
    """
    Rescale right eigenvectors Vhat such that (vhat_i == Xhat @ vtilde_i) .
    'Vhat': right eigenvectors (to be rescaled)
    'Vtilde': reference right eigenvectors
    'Xhat': data matrix
    """

    ## Get scale of reference eigs.
    Vtilde_scale = Xhat[:1, :] @ Vtilde

    ## Get scaling factor
    factor = Vtilde_scale / Vhat[:1, :]

    ## return re-scaled eigenvectors
    return Vhat * factor


def get_empirical_normal_modes(Xi, conj_indices, lonlat_coord=None):
    """
    Define empirical normal modes (ENMs) as the real/imaginary
    parts of Koopman modes (Xi).
    Put result in a dataarray if lonlat_coord is specified.
    Compare ENMs to Penland & Sardeshmukh (1995).
    """

    ## For complex conjugate pairs, set one mode to real part
    ## and the other to the imaginary part.
    normal_modes = np.zeros(Xi.shape)
    for pair in conj_indices:
        # Set the first mode in the pair to the real component
        normal_modes[:, pair[0]] = Xi[:, pair[0]].real

        ## if mode is part of pair, set second mode to imag. component
        if len(pair) > 1:
            normal_modes[:, pair[1]] = Xi[:, pair[1]].imag

    ## Put in xr.DataArray if lonlat coord is passed
    if lonlat_coord is not None:
        ## Put results in xr.DataArray
        normal_modes = xr.DataArray(
            normal_modes,
            coords={"lonlat": lonlat_coord, "mode": np.arange(Xi.shape[1])},
            dims=["lonlat", "mode"],
        )

        ## Unstack lon/lat coordinates
        normal_modes = normal_modes.unstack("lonlat")

    return normal_modes


class fft_1d:
    def __init__(self, data, x):
        ## Get data size, stepsize, and offset
        N = len(x)
        dx = x[1] - x[0]
        offset = x[0]

        ## Get frequency
        freq = np.fft.fftfreq(N, d=dx)

        ## do fft
        fft = np.fft.fft(data)

        ## handle even number of elements
        if np.mod(N, 2) == 0:
            ## Get updated frequency
            freq_shifted = np.fft.fftshift(freq)
            freq_shifted = np.insert(freq_shifted, obj=N, values=-freq_shifted[0])
            freq = np.fft.ifftshift(freq_shifted)

            ## build augmented fft
            fft_aug = np.zeros([N + 1], dtype=complex)
            fft_aug[:N] = np.fft.fftshift(fft)

            ## add conjugate row/col for positive Nyquist frequency
            fft_aug[0] /= 2
            fft_aug[-1] = fft_aug[0].conj()

            ## shift back
            fft = np.fft.ifftshift(fft_aug)

        ## save results
        self.fft = fft
        self.freq = freq
        self.N = N
        self.offset = offset

        return

    def f(self, x):
        """reconstruct data at specified point"""

        ## get frequency grid
        k = 2 * np.pi * self.freq
        ktheta_x = np.einsum("f,x->fx", k, x - self.offset)

        ## get reconstruction
        recon = 1 / self.N * np.einsum("f,ft", self.fft, np.exp(1j * ktheta_x))

        ## set imag. part to zero
        if np.allclose(recon.imag, np.zeros_like(recon.imag)):
            recon = recon.real
        else:
            print("warning: imaginary part not close to zero!")

        return recon


class fft_2d:
    def __init__(self, data, x, y):
        """
        Takes in data with shape 'n x n' and two 1-D arrays
        with length 'n', representing x and y coordinates.
        """

        ## Get data size, stepsize, and offset
        N = len(x)
        dx = x[1] - x[0]
        offset = x[0]

        ## Get frequency
        freq = np.fft.fftfreq(N, d=dx)

        ## do fft
        fft = np.fft.fft2(data)

        ## handle even number of elements
        if np.mod(N, 2) == 0:
            ## Get updated frequency
            freq_shifted = np.fft.fftshift(freq)
            freq_shifted = np.insert(freq_shifted, obj=N, values=-freq_shifted[0])
            freq = np.fft.ifftshift(freq_shifted)

            ## build augmented fft
            fft_aug = np.zeros([N + 1, N + 1], dtype=complex)
            fft_aug[:N, :N] = np.fft.fftshift(fft)

            ## add conjugate row/col for positive Nyquist frequency
            fft_aug[0, :] /= 2
            fft_aug[1:, 0] /= 2
            fft_aug[-1, :] = fft_aug[0, ::-1].conj()
            fft_aug[:, -1] = fft_aug[::-1, 0].conj()

            ## shift back
            fft = np.fft.ifftshift(fft_aug)

        ## save results
        self.fft = fft
        self.freq = freq
        self.N = N
        self.offset = offset

        return

    def f(self, x, y):
        """reconstruct function at given test points"""

        ## get integer multiples
        j = self.freq * 2 * np.pi
        k = self.freq * 2 * np.pi

        ## Get exponent term
        ktheta_y = np.einsum("f,y->fy", k, y - self.offset)
        jtheta_x = np.einsum("g,x->gx", j, x - self.offset)

        ## Sum arrays with broadcasting.
        ## Einsum notation: "fy,gx->fygx"
        jktheta = ktheta_y[..., None, None] + jtheta_x[None, None, ...]

        ## Get reconstruction
        recon = self.N**-2 * np.einsum("fg,fygx->yx", self.fft, np.exp(1j * jktheta))

        ## set imag. part to zero
        if np.allclose(recon.imag, np.zeros_like(recon.imag)):
            recon = recon.real
        else:
            print("warning: imaginary part not close to zero!")

        return recon


class decoder:
    def __init__(
        self, nmodes_e, nmodes_c, nonlinear_r=False, zero_mode=True, get_n34=None
    ):
        """object to represent 'decoder', which maps from eigenfunction to
        observable."""

        ## check that number of c-modes is an odd number
        if np.mod(nmodes_c, 2) == 0:
            print("Warning: even number of c-modes specified.")

        ## get coefficients for terms in exponential
        get_coefs = lambda N: np.fft.fftfreq(n=N, d=1 / N)
        k = get_coefs(N=nmodes_e * 2 + 1)
        j = get_coefs(N=nmodes_c)

        ## remove zero mode if specified
        if not zero_mode:
            k = k[1:]

        ## save coefficients
        self.k = k
        self.j = j
        self.nonlinear_r = nonlinear_r
        self.zero_mode = zero_mode
        self.get_n34 = None

        return

    def get_theta(self, theta_e, theta_s):
        ## get terms in exponent to be broadcast
        k_theta_e = np.einsum("k,n->kn", self.k, theta_e)
        j_theta_s = np.einsum("j,n->jn", self.j, theta_s)

        ## make arrays broadcast-able (kn/jn->kjn)
        k_theta_e = k_theta_e[:, None, :]
        j_theta_s = j_theta_s[None, :, :]

        ## do the broadcasting
        theta = k_theta_e + j_theta_s

        return theta

    def get_r_coefs(self, r):
        """get coefficients for 'r'.
        Basically, raise r^k, and broadcast to match 'j' dimension"""

        ## broadcast to match 'k' dimension
        if self.nonlinear_r:
            ## raise r to k^th power (n,k->kn)
            r_coefs = r[None, :] ** np.abs(self.k[:, None])

        else:
            r_coefs = r[None, :] * np.ones_like(self.k)[:, None]

        ## broadcast to match 'j' dimension
        r_coefs = np.einsum("kn,j->kjn", r_coefs, np.ones_like(self.j))

        return r_coefs

    def get_feature_mat(self, c_e, month):
        """build feature matrix from eigenfunction projections (c_e) and
        month of year"""

        ## Get magnitude and angle for eigenfn projections
        r = np.abs(c_e)
        theta_e = src.utils.get_angle(c_e)
        theta_s = (month / 12) * 2 * np.pi

        ## get theta and corresponding r coefficients
        theta = self.get_theta(theta_e=theta_e, theta_s=theta_s)
        r_coefs = self.get_r_coefs(r=r)

        return r_coefs * np.exp(1j * theta)

    def fit(self, y, c_e, month, cutoff_perc=None):
        """
        Fit reconstruction coefficients to data. Args:
            - y: (m x n) array of target variables
            - c_e: (n) array of ENSO eigenfn projections
            - month: month of year for each sample
            - cutoff_perc: only use data with magnitude exceeds this percentile
        """

        ## for convenience, check if target is real
        self.real_target = np.allclose(y.imag, np.zeros_like(y.imag))

        ## Get feature/target matrices
        E = self.get_feature_mat(c_e=c_e, month=month)

        ## For fitting model, reshape to 2-D matrix, kjn->(k*j)n
        n = y.shape[1]
        X = E.reshape(-1, n)

        if cutoff_perc is not None:
            ## Find high-magnitude samples
            r = np.abs(c_e)
            is_himag = r > np.percentile(r, cutoff_perc)

            ## subset data
            X = X[:, is_himag]
            y = y[:, is_himag]

        ## Get best fit
        Xi = y @ np.linalg.pinv(X)

        ## reshape: m(k*j)->mkj
        self.Xi = Xi.reshape(-1, len(self.k), len(self.j))

        return

    def predict(self, c_e, month, k_sum=None, j_sum=None):
        """Given ENSO eigenfn projection and month, predict Niño 3.4.
        'k_sum' is array-like of integers in the range [0,max(self.k)]
        'j_sum' is array-like of integers in [self.j]
        """

        ## Get feature matrix (kjn)
        feature_mat = self.get_feature_mat(c_e=c_e, month=month)

        ## get predictions
        # first option: sum over all Fourier modes.
        # otherwise: sum over specified indices
        if k_sum is None:
            preds = np.einsum("mkj,kjn->mn", self.Xi, feature_mat)

        else:
            ## check provided indices are valid
            valid_k = np.unique(np.abs(self.k))
            valid_j = self.j
            check_inlist = lambda l1, l2: all([(l_ in l2) for l_ in l1])
            if not (check_inlist(k_sum, valid_k) & check_inlist(j_sum, valid_j)):
                print("Warning: invalid sum indices!")
                print("valid_k", valid_k)
                print("k", k_sum)
                print()
                print("valid_j", valid_j)
                print("j", j_sum)
                print()

            ## empty array to hold results
            ntargets = self.Xi.shape[0]
            nsamples = feature_mat.shape[-1]
            preds = np.zeros([ntargets, nsamples], dtype=complex)

            ## multiply fourier matrix, but don't sum over it yet
            preds_ = np.einsum("mkj,kjn->kjmn", self.Xi, feature_mat)

            ## Pad along axis if zero_mode=False. This helps us avoid indexing
            ## issues. E.g., if zero_mode=False, then the 1st mode is in the 0th posn.
            ## Padding the kth dim with zeros makes it possible to use the same
            ## summation/indexing as for the zero_mode=True case.
            if self.zero_mode == False:
                ## add the padding of zeros in first element of k^th dimension
                preds_ = np.pad(
                    preds_,
                    pad_width=((1, 0), (0, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )

            ## loop through values of k (enso mode indices)
            for k_ in k_sum:
                ## if k_==0, make sure 'j' indices are positive to avoid
                ## double counting
                j_sum_ = j_sum if k_ > 0 else np.unique(np.abs(j_sum))

                ## loop through values of j (seasonal cycle indices)
                for j_ in j_sum_:
                    ## handle zero-mode case separately
                    ## (again, to avoid double-counting)
                    if k_ == 0 & j_ == 0:
                        preds += preds_[0, 0]

                    ## otherwise, sum complex-conjugate pair
                    else:
                        preds += preds_[k_, j_] + preds_[-k_, -j_]

        ## make sure imaginary part is small
        eps = np.finfo(np.float32).eps
        if np.all(np.abs(preds.imag) < eps):
            preds = preds.real

        elif self.real_target:
            print("Warning: large imaginary parts")

        return preds

    def get_nino_theta(
        self, month_idx=0, warm=True, y_idx=0, j_sum=None, k_sum=None, r_mag=1.0
    ):
        """Get theta_e corresponding to max Niño 3.4 index conditioned on given month"""

        ## get theta_e and theta_s
        theta_e = np.linspace(0, 2 * np.pi, 1000)
        month_idx_ = month_idx * np.ones_like(theta_e)

        ## Get recon
        c_e = r_mag * np.exp(1j * theta_e)
        recon = self.predict(c_e=c_e, month=month_idx_, k_sum=k_sum, j_sum=j_sum)

        ## get Niño 3.4 reconstruction
        if self.get_n34 is not None:
            y = self.get_n34(recon.squeeze(), month=month_idx_)

        else:
            y = recon[y_idx]

        ## find angle for peak Niño/Niña
        if warm:
            return theta_e[np.argmax(y)]

        else:
            return theta_e[np.argmin(y)]

    def plot(
        self,
        k_idx,
        j_idx,
        y_idx=0,
        ne=51,
        ns=50,
        show_traj=False,
        omega_e=None,
        month_idx=11,
        r_mag=1.0,
        amp=None,
        fig=None,
        ax=None,
    ):
        """
        Plot decoder function.
        Args:
            - k_idx, j_idx: which Fourier modes to sum over
            - y_idx: index of target variable to plot
            - ne, ns: number of points in y,x discretization
            - show_traj: should we plot a sample trajectory?
            - omega_e: frequency of ENSO mode; needed for slope of trajectory
            - month_idx: month to start trajectory, if desired
            - r_mag: magnitude of fundamental eigenfunction
            - amp: colobar max value
        Returns fig, ax objects
        """

        ## Get coordinates for grid
        month = np.linspace(0, 12, ns)
        theta_e = np.linspace(0, 2 * np.pi, ne)

        ## find angle for peak Niño during start of trajectory
        kwargs = dict(month_idx=month_idx, r_mag=r_mag, j_sum=j_idx, k_sum=k_idx)
        theta_max = self.get_nino_theta(**kwargs)

        ## get phase shift so that peak Niño displays at pi/2
        phi = theta_max - np.pi / 2
        theta_plot = theta_e + phi

        ## get eigenfunction coordinate for grid
        c_e = r_mag * np.exp(1j * theta_plot)

        ## get grids for plotting
        xx, yy = np.meshgrid(c_e, month)
        xx = xx.flatten()
        yy = yy.flatten()

        ## evaluate again for the plot, but with shifted theta_e values
        recon = self.predict(
            c_e=xx.flatten(),
            month=yy.flatten(),
            k_sum=k_idx,
            j_sum=j_idx,
        )

        ## map from possibly multi-dimensional recon to 1-d recon
        if (np.abs(recon.imag) > 0).any():
            recon = self.get_n34(
                recon.squeeze(), month=yy.flatten(), k_sum=k_idx, j_sum=j_idx
            )

        else:
            ## select relevant target index, and reshape into grid
            recon = recon[y_idx]

        ## put back on grid
        recon = recon.reshape(ns, ne)

        ## get scale for plotting if not specified
        if amp is None:
            amp = np.abs(recon).max()

        ## Set up plotting background if not specified
        if fig is None:
            fig = plt.figure(figsize=(4, 4))

        if ax is None:
            ax = fig.add_subplot()

        ## Plot the data
        plot_data = ax.pcolormesh(
            theta_e,
            month,
            recon,
            vmax=amp,
            vmin=-amp,
            shading="nearest",
            cmap="cmo.balance",
        )

        ## plot trajectory if desired
        if show_traj:
            ## compute trajectory
            n = 500  # number of points in trajectory
            theta_e_traj = np.mod(
                np.linspace(1 / 2 * np.pi, 5 / 2 * np.pi, n), 2 * np.pi
            )
            month_traj = np.mod(
                np.linspace(month_idx, month_idx + 12 * 1 / omega_e, n), 12
            )

            ## plot trajectory
            ax.scatter(
                theta_e_traj,
                month_traj,
                s=0.5,
                c=np.arange(len(theta_e_traj)),
                cmap="cmo.gray_r",
            )

            ## plot start/end months
            ax.scatter(theta_e_traj[0], month_traj[0], c="w", s=50)
            ax.scatter(theta_e_traj[-1], month_traj[-1], c="k", s=50)

        ## set axis labels/ticks
        xticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        xlabels = ["0", f"$\pi/2$", f"$\pi$", f"$3\pi/2$", f"$2\pi$"]
        ax.set_xticks(xticks, labels=xlabels)
        ax.set_xlabel(r"$\theta_e$")

        ax.set_ylabel("Month")
        ax.set_yticks([0, 4, 8, 12], labels=["Jan", "May", "Sep", "Jan"])

        return fig, ax, plot_data
