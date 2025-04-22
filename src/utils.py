import xarray as xr
import numpy as np
import pandas as pd
import src.XRO
import tqdm
import warnings


def get_RO_ensemble(
    data, T_var="T_3", h_var="h_w", verbose=False, fit_kwargs=dict(), model=None
):
    """get RO params for each ensemble member"""

    ## initialize model
    if model is None:
        model = src.XRO.XRO(ncycle=12, ac_order=1, is_forward=True)

    ## empty list to hold model fits
    fits = []

    ## Loop thru ensemble members
    for m in tqdm.tqdm(data.member, disable=not (verbose)):

        ## select ensemble member and variables
        data_subset = data[[T_var, h_var]].sel(member=m)

        ## fit model
        with warnings.catch_warnings(action="ignore"):
            fits.append(model.fit_matrix(data_subset, **fit_kwargs))

    return model, xr.concat(fits, dim=data.member)


def generate_ensemble(
    model, params, sampling_type="ensemble_mean", **simulation_kwargs
):
    """Generate ensemble of RO simulations given parameters estimated
    from each MPI ensemble member."""

    if sampling_type == "ensemble_mean":
        RO_ensemble = model.simulate(fit_ds=params.mean("member"), **simulation_kwargs)

    else:
        print("Not a valid sampling type")

    return RO_ensemble


def get_ensemble_stats(ensemble, center_method="mean", bounds_method="std"):
    """compute ensemble 'center' and 'bounds' for plotting"""

    ## compute ensemble 'center'
    if center_method == "mean":
        center = ensemble.mean("member")
    else:
        print("Not implemented")

    ## compute bounds
    if bounds_method == "std":
        sigma = ensemble.std("member")
        upper_bound = center + sigma
        lower_bound = center - sigma
    else:
        print("Not implemented")

    ## concatenate into single array
    posn_dim = pd.Index(["center", "upper", "lower"], name="posn")
    stats = xr.concat([center, upper_bound, lower_bound], dim=posn_dim)

    return stats


def get_timescales(model, fit):
    """Get annual cycle of growth rate and period estimate from RO parameters"""

    ## get parameters
    params = model.get_RO_parameters(fit)

    ## extract BJ index
    BJ_ac = params["BJ_ac"]

    ## extract period (based on annual avg.)
    L0 = fit["Lcomp"].isel(ac_rank=0, cycle=0)
    w, _ = np.linalg.eig(L0)
    T = 2 * np.pi / w[0].imag

    return BJ_ac, T


def get_timescales_ensemble(model, fit_ensemble):
    """Get timescales for ensemble"""

    ## empty lists to hold results
    BJ_ac_ensemble = []
    T_ensemble = []

    ## loop through ensemble members
    for m in fit_ensemble.member:
        BJ_ac, T = get_timescales(model, fit_ensemble.sel(member=m))
        BJ_ac_ensemble.append(BJ_ac)
        T_ensemble.append(T)

    ## put in arrays
    BJ_ac_ensemble = xr.concat(BJ_ac_ensemble, dim=fit_ensemble.member)
    T_ensemble = np.array(T_ensemble)

    return BJ_ac_ensemble, T_ensemble


def get_empirical_pdf(x, edges=None):
    """
    Estimate the "empirical" probability distribution function for the data x.
    In this case the result is a normalized histogram,
    Normalized means that integrating over the histogram yields 1.
    Returns the PDF (normalized histogram) and edges of the histogram bins
    """

    ## compute histogram
    hist, bin_edges = np.histogram(x, bins=edges)

    ## normalize to a probability distribution (PDF)
    bin_width = bin_edges[1:] - bin_edges[:-1]
    pdf = hist / (hist * bin_width).sum()

    return pdf, bin_edges


def get_cov(X, Y=None):
    """compute covariance"""

    if Y is None:
        Y = X

    ## remove means
    X_prime = X - X.mean(1, keepdims=True)
    Y_prime = Y - Y.mean(1, keepdims=True)

    ## number of samples
    n = X_prime.shape[1]

    return 1 / n * X_prime @ Y_prime.T


def make_cossin_mask(x_idx, ac_order, rankx=2):
    """
    Get mask for removing annual cycle from specified parameter.
    Args:
    - x_idx: input index of parameter to zero out
    - ac_order: order of annual cycle in the model
    - ranky, rankx: number of output and input variables

    Returns:
    - array of shape rankx * (1 + 2 * ac_order)
    """

    ## mask annual cycle for input variables
    ac_mask = np.ones([rankx, ac_order + 1])

    ## set seasonality terms to zero
    ac_mask[x_idx, 1:] = 0

    ## reshape into cossin format
    cossin_mask = np.concat(
        [ac_mask[..., :1], ac_mask[..., 1:], ac_mask[..., 1:]], axis=-1
    )

    ## reshape to match shape of G/C
    cossin_mask = cossin_mask.reshape(-1, order="F").astype(bool)

    return cossin_mask


def make_GC_masks(x_idx, ac_order, rankx=2):
    """
    Make annual cycle masks for <Y, X> and <X, X> covariance matrices.
    Args:
    - x_idx: input index of parameter to zero out
    - ac_order: order of annual cycle in the model
    - ranky, rankx: number of output and input variables

    Returns:
    - G_mask: mask for <Y, X> matrix, with shape (1xp)
    - C_mask: mask for <X, X> matrix, with shape (pxp),
    where p = 1 + 2 * ac_order
    """

    ## Get cossin mask
    cossin_mask = make_cossin_mask(x_idx=x_idx, ac_order=ac_order, rankx=rankx)

    ## empty arrays for G and C masks
    p = len(cossin_mask)
    G_mask = np.zeros([1, p])
    C_mask = np.zeros([p, p])

    ## fill with ones where values are retained
    G_mask[:, cossin_mask] = 1.0
    C_mask[np.ix_(cossin_mask, cossin_mask)] = 1.0

    return G_mask, C_mask
