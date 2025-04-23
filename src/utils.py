import xarray as xr
import numpy as np
import pandas as pd
import xeofs as xe
import src.XRO
import tqdm
import warnings
import pathlib
import cartopy.crs as ccrs


def spatial_avg(data):
    """get spatial average, weighting by cos of latitude"""

    ## get weighted data
    weights = np.cos(np.deg2rad(data.latitude))

    return data.weighted(weights).mean(["latitude", "longitude"])


def get_RO_ensemble(
    data, T_var="T_3", h_var="h_w", verbose=False, model=None, ac_mask_idx=None
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
            fits.append(model.fit_matrix(data_subset, ac_mask_idx=ac_mask_idx))

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


def unzip_tuples_to_arrays(list_of_tuples):
    """convert list of tuples to two arrays"""

    ## unzip list of (y,x) tuples to list-of-y and list-of-x
    y_list, x_list = list(zip(*list_of_tuples))

    ## convert to array
    y_arr = np.array(y_list)
    x_arr = np.array(x_list)

    return y_arr, x_arr


def reconstruct_fn(components, scores, fn):
    """reconstruct function of spatial data from PCs"""

    ## get latitude weighting for components
    coslat_weights = np.sqrt(np.cos(np.deg2rad(components.latitude)))

    ## evaluate function on spatial components
    fn_eval = fn(components * 1 / coslat_weights)

    ## reconstruct
    recon = (fn_eval * scores).sum("mode")

    return recon


def get_nino34(data):
    """compute Niño 3.4 index from data"""

    ## trim to nino3.4 region
    data_trimmed = data.sel(latitude=slice(-5, 5), longitude=slice(190, 240))

    return src.utils.spatial_avg(data_trimmed)


def load_eofs(eofs_fp):
    """
    Load pre-computed EOFs.
    Args:
        - eofs_fp: pathlib.Path object; path to saved EOFs
    """

    ## initialize EOF model
    eofs_kwargs = dict(n_modes=150, standardize=False, use_coslat=True, center=False)
    eofs = xe.single.EOF(**eofs_kwargs)

    ## Load pre-computed model if it exists
    if pathlib.Path(eofs_fp).is_file():
        return eofs.load(eofs_fp, engine="netcdf4")

    else:
        print("Error: file doesn't exist! Please run preprocessing script")
        return


def reconstruct_var(scores, components):
    """reconstruct spatial variance from projected data"""

    ## remove mean
    scores_anom = scores - scores.mean(["member"])

    ## compute outer product (XX^T)
    outer_prod = xr.dot(
        scores_anom, scores_anom.rename({"mode": "mode_out"}), dim=["time", "member"]
    )

    ## get scaling
    n = len(scores.time) * len(scores.member)

    ## get covariance of projected data
    scores_cov = 1 / n * outer_prod

    ## now reconstruct spatial field (U @ SVt @ VS) @ U
    spatial_cov = xr.dot(
        xr.dot(components, scores_cov, dim="mode"),
        components.rename({"mode": "mode_out"}),
        dim="mode_out",
    )

    ## get coslat weights
    coslat_weights = np.cos(np.deg2rad(components.latitude))

    return spatial_cov / coslat_weights


def plot_setup(fig, lon_range, lat_range):
    """Add a subplot to the figure with the given map projection
    and lon/lat range. Returns an Axes object."""

    ## Create subplot with given projection
    proj = ccrs.PlateCarree(central_longitude=190)
    ax = fig.add_subplot(projection=proj)

    ## Subset to given region
    extent = [*lon_range, *lat_range]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ## draw coastlines
    ax.coastlines(linewidth=0.3)

    return ax
