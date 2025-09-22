import xarray as xr
import numpy as np
import pandas as pd
import xeofs as xe
import src.XRO
import tqdm
import warnings
import pathlib
import cartopy.crs as ccrs
import cartopy.util
import matplotlib.pyplot as plt
import scipy.stats
import copy
import calendar
import os
import src.XRO


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

    elif center_method == "median":
        center = ensemble.quantile(q=0.5, dim="member")

    else:
        print("Not implemented")

    ## compute bounds
    if bounds_method == "std":
        sigma = ensemble.std("member")
        upper_bound = center + sigma
        lower_bound = center - sigma

    elif bounds_method == "quantile":
        lower_bound = ensemble.quantile(q=0.1, dim="member")
        upper_bound = ensemble.quantile(q=0.9, dim="member")

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


def get_coslat_weights(components):
    """helper function: return coslat weights if 'latitude' is in components;
    otherwise, return 1"""

    ## Cosine-latitude weights
    if "latitude" in components.coords:
        return np.sqrt(np.cos(np.deg2rad(components.latitude)))

    elif "lat" in components.coords:
        return np.sqrt(np.cos(np.deg2rad(components.lat)))

    else:
        return 1.0


def reconstruct_fn_da(components, scores, fn):
    """reconstruct function of spatial data from PCs. Assumes
    components and scores are xr.DataArrays"""

    ## get latitude weighting for components
    coslat_weights = get_coslat_weights(components)

    ## evaluate function on spatial components
    fn_eval = fn(components * 1 / coslat_weights)

    ## reconstruct
    recon = (fn_eval * scores).sum("mode")

    return recon


def reconstruct_fn(components, scores, fn):
    """reconstruct given function based on EOF components and scores"""

    ## handle Dataset case
    if type(components) is xr.Dataset:
        varnames = list(components)
        recon = xr.merge(
            [
                reconstruct_fn_da(components[n], scores[n], fn).rename(n)
                for n in varnames
            ]
        )

    ## handle DataArray case
    else:
        recon = reconstruct_fn_da(components, scores, fn)

    return recon


def get_spatial_avg_idx(data, lon_range, lat_range):
    """get index which is a spatial average over data subset"""

    ## get indexer
    indexer = dict(longitude=slice(*lon_range), latitude=slice(*lat_range))

    return spatial_avg(data.sel(indexer))


def get_nino4(data):
    """compute Niño 4 index from data"""

    ## get indexer
    idx = dict(latitude=slice(-5, 5), longitude=slice(160, 210))

    return spatial_avg(data.sel(idx))


def get_nino34(data):
    """compute Niño 3.4 index from data"""

    ## get indexer
    idx = dict(latitude=slice(-5, 5), longitude=slice(190, 240))

    return spatial_avg(data.sel(idx))


def get_nino3(data):
    """compute Niño 3 index from data"""

    ## get indexer
    idx = dict(latitude=slice(-5, 5), longitude=slice(210, 270))

    return spatial_avg(data.sel(idx))


def get_RO_h(data):
    """compute RO's 'h' index from data"""

    ## get indexer
    idx = dict(latitude=slice(-5, 5), longitude=slice(120, 280))

    return spatial_avg(data.sel(idx))


def get_RO_hw(data):
    """compute RO's 'h' index from data"""

    ## get indexer
    idx = dict(latitude=slice(-5, 5), longitude=slice(120, 210))

    return spatial_avg(data.sel(idx))


def get_RO_T_indices(sst_data):
    """compute 'T' indices for RO model"""
    funcs = [get_nino3, get_nino34, get_nino4]
    names = ["T_3", "T_34", "T_4"]

    ## compute indices
    indices = xr.merge([f(sst_data).rename(n) for f, n in zip(funcs, names)])

    return indices


def get_RO_h_indices(ssh_data):
    """compute 'T' indices for RO model"""
    funcs = [get_RO_h, get_RO_hw]
    names = ["h", "h_w"]

    ## compute indices
    indices = xr.merge([f(ssh_data).rename(n) for f, n in zip(funcs, names)])

    return indices


def get_RO_indices(data, h_var="ssh"):
    """compute possible indices for RO from given dataset"""

    ## specify functions and variables
    funcs = [get_nino3, get_nino34, get_nino4, get_RO_h, get_RO_hw]
    vars_ = ["sst", "sst", "sst", h_var, h_var]
    names = ["T_3", "T_34", "T_4", "h", "h_w"]

    ## compute indices
    indices = xr.merge([f(data[v]).rename(n) for f, v, n in zip(funcs, vars_, names)])

    return indices


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


def reconstruct_std(**kwargs):
    """Reconstruct standard deviation"""

    return np.sqrt(reconstruct_var(**kwargs))


def reconstruct_var(scores, components, fn=None):
    """reconstruct spatial variance based on EOF components and scores"""

    ## handle Dataset case
    if type(components) is xr.Dataset:
        varnames = list(components)
        recon = xr.merge(
            [
                reconstruct_var_da(
                    scores=scores[n], components=components[n], fn=fn
                ).rename(n)
                for n in varnames
            ]
        )

    ## handle DataArray case
    else:
        recon = reconstruct_var_da(scores=scores, components=components, fn=fn)

    return recon


def reconstruct_var_da(scores, components, fn):
    """reconstruct spatial variance from projected data"""

    ## remove mean
    scores_anom = scores - scores.mean(["member", "time"])

    ## compute outer product (XX^T)
    outer_prod = xr.dot(
        scores_anom, scores_anom.rename({"mode": "mode_out"}), dim=["time", "member"]
    )

    ## get scaling
    n = len(scores.time) * len(scores.member)

    ## get covariance of projected data
    scores_cov = 1 / n * outer_prod

    ## get latitude weighting for reconstructino
    coslat_weights = get_coslat_weights(components)

    ## apply function to components
    if fn is None:
        fn = lambda x: x
    fn_eval = fn(components * 1 / coslat_weights)

    ## now reconstruct spatial field (U @ SVt @ VS) @ U
    fn_var = xr.dot(
        xr.dot(fn_eval, scores_cov, dim="mode"),
        fn_eval.rename({"mode": "mode_out"}),
        dim="mode_out",
    )

    return fn_var


def reconstruct_cov_da(
    V_x,
    U_x,
    fn_x=lambda x: x,
    V_y=None,
    U_y=None,
    fn_y=None,
):
    """reconstruct local covariance from projected data"""

    ## handle variance case:
    if V_y is None:
        V_y = V_x
        U_y = U_x
    if fn_y is None:
        fn_y = fn_x

    ## remove means
    V_x_ = V_x - V_x.mean(["member", "time"])
    V_y_ = V_y - V_y.mean(["member", "time"])

    ## compute outer product (XX^T)
    outer_prod = xr.dot(V_x_, V_y_.rename({"mode": "mode_out"}), dim=["time", "member"])

    ## get scaling
    n = len(V_x.time) * len(V_x.member)

    ## get covariance of projected data
    V_cov = 1 / n * outer_prod

    ## get latitude weighting for reconstructino
    weights_x = get_coslat_weights(components=U_x)
    weights_y = get_coslat_weights(components=U_y)

    ## apply function to components
    fn_x_eval = fn_x(U_x * 1 / weights_x)
    fn_y_eval = fn_y(U_y * 1 / weights_y)

    ## now reconstruct spatial field (U @ SVt @ VS) @ U
    fn_cov = xr.dot(
        xr.dot(fn_x_eval, V_cov, dim="mode"),
        fn_y_eval.rename({"mode": "mode_out"}),
        dim="mode_out",
    )

    return fn_cov


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


def subplots_with_proj(
    fig,
    proj=ccrs.PlateCarree(central_longitude=190),
    nrows=1,
    ncols=1,
    format_func=None,
):
    """create subplots with proj"""

    ## get number of total plots
    size = nrows * ncols

    for row in range(nrows):
        for col in range(ncols):
            ## get index for plot
            idx = row * ncols + col + 1

            ## add subplot
            ax = fig.add_subplot(nrows, ncols, idx, projection=proj)

            ## format if desired
            if format_func is not None:
                format_func(ax)

    return np.array(fig.get_axes()).reshape(nrows, ncols)


def plot_setup_pac(ax, max_lat=30):
    """Plot Pacific region"""

    ## trim and add coastlines
    ax.coastlines(linewidth=0.3)
    ax.set_extent([100, 300, -max_lat, max_lat], crs=ccrs.PlateCarree())

    return ax


def detrend_dim(data, dim="time", deg=1):
    """
    Detrend along a single dimension.
    'data' is an xr.Dataset or xr.DataArray
    """
    ## First, check if data is a dataarray. If so,
    ## convert to dataset
    is_dataarray = type(data) is xr.DataArray
    if is_dataarray:
        data = xr.Dataset({data.name: data})

    ## Compute the linear best fit
    p = data.polyfit(dim=dim, deg=deg, skipna=True)
    fit = xr.polyval(data[dim], p)

    ## empty array to hold results
    data_detrended = np.nan * xr.ones_like(data)

    ## Subtract the linear best fit for each variable
    varnames = list(data)
    for varname in varnames:
        data_detrended[varname] = data[varname] - fit[f"{varname}_polyfit_coefficients"]

    if is_dataarray:
        varname = list(data_detrended)[0]
        data_detrended = data_detrended[varname]

    return data_detrended


def load_oras_spatial_extended(oras_fp, varnames=["tos", "ssh", "d20"]):
    """Load spatial data for SST and SSH from ORAS5"""

    ## funcs to trim and load data
    trim = lambda x: x.sel(lat=slice(-20, 20), lon=slice(100, 300))
    load = lambda f, v: trim(xr.open_dataarray(f)).rename(v)

    ## loop through variables
    data = []
    for v in varnames:

        ## get filename
        f = list(oras_fp.glob(f"{v}*.nc"))[0]

        ## load data
        data.append(load(f, v))

    ## merge
    data = xr.merge(data)

    ## update varnames if desired
    if "tos" in varnames:
        data = data.rename({"tos": "sst"})

    return update_oras_coords(data)


def update_oras_coords(data):
    """update coordinates on ORAS dataset"""

    return data.rename({"lon": "longitude", "lat": "latitude", "time_counter": "time"})


def load_oras_spatial(sst_fp, ssh_fp, extended_record=False, use_d20=False):
    """Load spatial data for SST and SSH from ORAS5"""

    ## func to trim data
    trim = lambda x: x.sel(lat=slice(-30, 30), lon=slice(100, 300))

    ## open data for 1979 - 2018
    sst_oras = xr.open_mfdataset(sst_fp.glob("*.nc"), preprocess=trim)
    ssh_oras = xr.open_mfdataset(ssh_fp.glob("*.nc"), preprocess=trim)

    ## merge sst/ssh data
    data_oras = xr.merge([sst_oras, ssh_oras]).compute()

    ## rename coords
    data_oras = data_oras.rename(
        {
            "sosstsst": "sst",
            "sossheig": "ssh",
            "lon": "longitude",
            "lat": "latitude",
            "time_counter": "time",
        }
    )

    return data_oras


def make_cb_range(amp, delta):
    """Make colorbar_range for cmo.balance"""
    return np.concatenate(
        [np.arange(-amp, 0, delta), np.arange(delta, amp + delta, delta)]
    )


def plot_box(ax, lons, lats, **kwargs):
    """plot box on given ax object"""

    ## get box coords
    lon_coords = [lons[0], lons[1], lons[1], lons[0], lons[0]]
    lat_coords = [lats[0], lats[0], lats[1], lats[1], lats[0]]

    ax.plot(
        lon_coords,
        lat_coords,
        transform=ccrs.PlateCarree(),
        **kwargs,
    )

    return


def plot_nino4_box(ax, **kwargs):
    """outline Niño 4 region"""
    return plot_box(ax, lons=[160, 210], lats=[-5, 5], **kwargs)


def plot_nino34_box(ax, **kwargs):
    """outline Niño 3.4 region"""
    return plot_box(ax, lons=[190, 240], lats=[-5, 5], **kwargs)


def plot_nino3_box(ax, **kwargs):
    """outline Niño 3.4 region"""
    return plot_box(ax, lons=[210, 270], lats=[-5, 5], **kwargs)


def plot_hw_box(ax, **kwargs):
    """outliner h_w region"""
    return plot_box(ax, lats=[-5, 5], lons=[120, 210], **kwargs)


def get_gaussian_best_fit(x):
    """Get gaussian best fit to data, and evaluate
    probabilities over the range of the data."""

    ## get normal distribution best fit
    gaussian = scipy.stats.norm(loc=x.mean(), scale=x.std())

    ## evaluate over range of data
    amp = np.max(np.abs(x))
    x_eval = np.linspace(-amp, amp)
    pdf_eval = gaussian.pdf(x_eval)

    return pdf_eval, x_eval


def sel_month(x, months=None):
    """select months in specified range.
    Args:
        - x: xr.DataArray or xr.Dataset
        - months: one of {None, int, list-of-int}
    """
    if months is None:
        return x

    else:
        data_months = x.time.dt.month

        if type(months) is int:
            is_month = data_months == months

        else:
            is_month = np.array([m in months for m in data_months])

        return x.isel(time=is_month)


def get_rolling_fn(x, fn, n=0, reduce_ensemble_dim=True):
    """Apply function to rolling set of data. Applies to window of
    size (2n+1), centered at each datapoint."""

    ## get rolling object
    x_rolling = x.rolling({"time": 2 * n + 1}, center=True).construct("window")

    if reduce_ensemble_dim:

        ## stack member,time dimensinos if reducing
        x_rolling = x_rolling.stack(sample=["member", "window"])

    else:

        ## otherwise, just rename 'window' dim to sample
        x_rolling = x_rolling.rename({"window": "sample"})

    ## apply function over sample dimension
    fn_rolling = xr.apply_ufunc(
        fn, x_rolling, input_core_dims=[["sample"]], kwargs={"axis": -1}
    )

    ## trim if necessary
    if n > 0:
        fn_rolling = fn_rolling.isel(time=slice(n, -n))

    return fn_rolling


def get_rolling_fn_bymonth(x, fn, n=0, reduce_ensemble_dim=True):
    """apply function to rolling set of data by month. 'n' has units of years"""

    ## apply function to data grouped by month
    kwargs = dict(fn=fn, n=n, reduce_ensemble_dim=reduce_ensemble_dim)
    return x.groupby("time.month").map(lambda z: get_rolling_fn(z, **kwargs))


def separate_forced(data, n=0):
    """
    Get forced component of ensemble. 'n' specifies number of years
    to average over when computing "forced" component. I.e., average
    n time over a window size of [t-n, t+n], so total window size
    is 2n + 1.
    Returns:
        - forced: 'externally forced' component of ensemble
        - anom: total minus forced component
    """

    ## forced response defined as ensemble mean, smoothed in time
    forced = get_rolling_fn_bymonth(data, fn=np.mean, n=n)

    ## anomaly is residual of total minus forced
    anom = data.sel(time=forced.time) - forced

    return forced, anom


def get_rolling_avg(x, n, dim="time"):
    """Get rolling average over 2n+1 timesteps, and remove NaNs"""

    ## compute rolling average
    x_rolling = x.rolling({dim: 2 * n + 1}, center=True).mean()

    ## remove NaNs
    if n != 0:
        x_rolling = x_rolling.isel({dim: slice(n, -n)})

    return x_rolling


def separate_forced(data, n=0):
    """
    Get forced component of ensemble. 'n' specifies number of years
    to average over when computing "forced" component. I.e., average
    n time over a window size of [t-n, t+n], so total window size
    is 2n + 1.
    Returns:
        - forced: 'externally forced' component of ensemble
        - anom: total minus forced component
    """

    ## compute ensemble mean
    ensemble_mean = data.mean("member")

    ## group by month, then computing rolling mean over years
    get_rolling_avg_ = lambda x: get_rolling_avg(x, n=n)
    forced = ensemble_mean.groupby("time.month").map(get_rolling_avg_)

    ## make sure time axis is in order
    forced = forced.isel(time=forced.time.argsort().values)

    ## compute anomalies
    anom = data.sel(time=forced.time) - forced

    return forced, anom


def unstack_month_and_year(data):
    """Function 'unstacks' month and year in a dataframe with 'time' dimension
    The 'time' dimension is separated into a month and a year dimension
    This increases the number of dimensions by 1"""

    ## copy data to avoid overwriting
    data_ = copy.deepcopy(data)

    ## Get year and month
    year = data_.time.dt.year.values
    month = data_.time.dt.month.values

    ## update time index
    new_idx = pd.MultiIndex.from_arrays([year, month], names=("year", "month"))
    new_idx = xr.Coordinates.from_pandas_multiindex(new_idx, "time")
    data_ = data_.assign_coords(new_idx)

    return data_.unstack("time")


def make_variance_subplots(
    fig,
    axs,
    var0,
    var1,
    amp,
    amp_diff,
    show_colorbars,
    cbar_label=None,
    amp_min=0,
):
    """make 3-paneled subplots showing variance in ORAS5 and MPI; and (normalized) difference)"""

    ## shared arguments
    kwargs = dict(cmap="cmo.amp", transform=ccrs.PlateCarree())

    ## get levels for individual plots
    levels = np.linspace(amp_min, amp, 9)
    levels_diff = make_cb_range(amp_diff, amp_diff / 8)

    ## should we extend "both" or "max"?
    extend = "max" if (amp_min == 0) else "both"

    ## plot variance in ORAS5
    plot_data0 = axs[0, 0].contourf(
        var0.longitude, var0.latitude, var0, levels=levels, extend=extend, **kwargs
    )
    axs[0, 0].set_title("ORAS5")

    ## plot variance in MPI
    plot_data1 = axs[1, 0].contourf(
        var1.longitude, var1.latitude, var1, levels=levels, extend=extend, **kwargs
    )
    axs[1, 0].set_title("MPI")

    ## plot difference
    bias = var1 - var0
    kwargs.update(dict(cmap="cmo.balance", levels=levels_diff))
    plot_data2 = axs[2, 0].contourf(
        bias.longitude, bias.latitude, bias, extend="both", **kwargs
    )
    axs[2, 0].set_title("Bias")

    ## add colorbars if desired
    if show_colorbars:
        fig.colorbar(plot_data0, ax=axs[0, 0], ticks=[amp_min, amp], label=cbar_label)
        fig.colorbar(plot_data1, ax=axs[1, 0], ticks=[amp_min, amp], label=cbar_label)
        fig.colorbar(
            plot_data2, ax=axs[2, 0], ticks=[-amp_diff, 0, amp_diff], label=cbar_label
        )

    return fig, axs


def spatial_comp(
    xhat,
    x,
    kwargs=dict(),
    diff_kwargs=dict(),
    figsize=(3, 3),
    format_func=plot_setup_pac,
    show_colorbars=True,
    add_titles=True,
):
    """make 3-paneled subplots showing truth (x), prediction (xhat), and diff (xhat-x)"""

    ## set up plotting canvas
    fig = plt.figure(figsize=figsize, layout="constrained")
    axs = subplots_with_proj(fig, nrows=3, ncols=1, format_func=format_func)

    ## for convenience, get lon/lat
    lon, lat = x.longitude, x.latitude

    ## plot ground truth
    plot_data0 = axs[0, 0].contourf(lon, lat, x, transform=ccrs.PlateCarree(), **kwargs)

    ## Plot prediction
    plot_data1 = axs[1, 0].contourf(
        lon, lat, xhat, transform=ccrs.PlateCarree(), **kwargs
    )

    ## plot difference
    plot_data2 = axs[2, 0].contourf(
        lon, lat, xhat - x, transform=ccrs.PlateCarree(), **diff_kwargs
    )

    ## concatenate plot data
    plot_data = [plot_data0, plot_data1, plot_data2]

    ## make colorbars
    colorbars = []
    if show_colorbars:
        for j, ax in enumerate(axs):
            levels = kwargs["levels"] if j < 2 else diff_kwargs["levels"]
            colorbars.append(
                fig.colorbar(plot_data[j], ax=ax, ticks=[levels.min(), levels.max()])
            )

    ## add labels
    if add_titles:
        labels = ["truth", "recon", "error"]
        for j, (ax, t) in enumerate(zip(axs, labels)):
            labels = ["ground truth", "reconstruction", "error"]
            axs[j, 0].set_title(t)

    return fig, axs, plot_data, colorbars


def plot_cycle_hov(ax, data, **kwargs):
    """plot hovmoller of SST on the equator"""

    ## make sure month is first
    data = data.transpose("month", ...)

    ## Get cyclic point for plotting
    data_cyclic, month_cyclic = cartopy.util.add_cyclic_point(data, data.month, axis=0)

    ## plot data
    plot_data = ax.contourf(data.longitude, month_cyclic, data_cyclic, **kwargs)

    ## plot Niño 3.4 region
    kwargs = dict(ls="--", c="w", lw=0.85)
    ax.axvline(190, **kwargs)
    ax.axvline(240, **kwargs)

    ## labels/style
    ax.set_yticks([1, 5, 9, 13], labels=["Jan", "May", "Sep", "Jan"])
    ax.set_ylabel("Month")
    ax.set_xticks([])
    ax.set_xlim([130, 280])

    return plot_data


def plot_cycle_hov_v2(ax, data, amp, is_filled=True):
    """plot data on ax object"""

    ## specify shared kwargs
    shared_kwargs = dict(levels=make_cb_range(amp, amp / 5), extend="both")

    ## specify kwargs
    if is_filled:
        plot_fn = ax.contourf
        kwargs = dict(cmap="cmo.balance")

    else:
        plot_fn = ax.contour
        kwargs = dict(colors="k", linewidths=0.8)

    def merimean(x):
        """helper func. to compute meridional mean"""
        kwargs = dict(longitude=slice(140, 285), latitude=slice(-5, 5))
        return x.sel(**kwargs).mean("latitude")

    ## do the plotting
    cp = plot_fn(
        merimean(data).longitude,
        merimean(data).month,
        merimean(data),
        **kwargs,
        **shared_kwargs,
    )

    ## format ax object
    xticks = [160, 210]
    kwargs = dict(c="w", ls="--", lw=1)
    ax.set_xlabel("Lon")
    ax.set_xticks(xticks)
    for tick in xticks:
        ax.axvline(tick, **kwargs)

    return cp


def plot_cycle_hov_v3(ax, data, amp, is_filled=True):
    """plot data on ax object"""

    ## specify shared kwargs
    shared_kwargs = dict(levels=make_cb_range(amp, amp / 5), extend="both")

    ## specify kwargs
    if is_filled:
        plot_fn = ax.contourf
        kwargs = dict(cmap="cmo.balance")

    else:
        plot_fn = ax.contour
        kwargs = dict(colors="k", linewidths=0.8)

    def merimean(x):
        """helper func. to compute meridional mean"""
        kwargs = dict(longitude=slice(140, 285), latitude=slice(-5, 5))
        return x.sel(**kwargs).mean("latitude")

    ## do the plotting
    cp = plot_fn(
        merimean(data).longitude,
        merimean(data).month,
        merimean(data),
        **kwargs,
        **shared_kwargs,
    )

    ## xticks
    ax.set_xticks([])
    xticks = [160, 210, 240]
    kwargs = dict(c="w", ls="--", lw=1)
    for tick in xticks:
        ax.axvline(tick, **kwargs)

    ## yticks
    ax.set_yticks([3, 10], labels=["Mar", "Oct"])

    return cp


def format_hov_v3(axs):
    """format plot for plot_cycle_hov_v3"""

    axs[0].set_xticks([160, 210, 240])
    axs[0].set_xlabel("Lon")
    axs[0].xaxis.set_label_position("top")
    axs[0].xaxis.tick_top()
    axs[0].set_ylabel("Coefficients")
    axs[1].set_ylabel("Bias")

    for ax in axs:
        ax.yaxis.set_label_position("right")

    return axs


def make_pcolor_plot(ax, x, amp, sel=lambda x: x.mean("month")):
    """plot data on ax object"""
    cp = ax.pcolormesh(
        x.longitude,
        x.latitude,
        sel(x),
        vmax=amp,
        vmin=-amp,
        cmap="cmo.balance",
        transform=ccrs.PlateCarree(),
    )

    return cp0


def make_contour_plot(ax, x, amp, sel=lambda x: x.mean("month")):
    """plot data on ax object"""
    cp = ax.contourf(
        x.longitude,
        x.latitude,
        sel(x),
        levels=make_cb_range(amp, amp / 10),
        cmap="cmo.balance",
        transform=ccrs.PlateCarree(),
        extend="both",
    )

    return cp


def get_bias(data, ref_data):
    """get bias, accounting for grid differences
    1. reverse direction of latitude
    2. get longitudes to match
    """

    ## get target grid
    grid = dict(longitude=ref_data.longitude, latitude=ref_data.latitude)

    return data.interp(grid) - ref_data


def make_scatter(ax, data, x_var, y_var, fn_x, fn_y, scale=1):
    """scatter plot data on axis"""

    ## evaluate functions
    if "mode" in data.dims:
        fn_x_eval = reconstruct_fn(
            scores=data[x_var], components=data[f"{x_var}_comp"], fn=fn_x
        )
        fn_y_eval = reconstruct_fn(
            scores=data[y_var], components=data[f"{y_var}_comp"], fn=fn_y
        )

        ## stack member/time dim
        stack = lambda x: x.stack(sample=["member", "time"])
        fn_x_eval = stack(fn_x_eval)
        fn_y_eval = stack(fn_y_eval)
        dim = "sample"

    else:
        fn_x_eval = fn_x(data[x_var])
        fn_y_eval = fn_y(data[y_var])
        dim = "time"

    ## compute slope for best fit line
    slope = regress_core(X=fn_x_eval, Y=scale * fn_y_eval, dim=dim)

    ## convert to numpy
    slope = slope.values.item()

    ## plot data
    ax.scatter(fn_x_eval, scale * fn_y_eval, s=0.5)

    ## plot best fit
    xtest = np.linspace(fn_x_eval.values.min(), fn_x_eval.values.max())
    ax.plot(xtest, slope * xtest, c="k", lw=1)

    ## plot some guidelines
    ax.axhline(0, ls="--", lw=0.8, c="k")
    ax.axvline(0, ls="--", lw=0.8, c="k")

    return slope


def apply_along_axis(data, fn, axis_name):
    """apply function along given axis"""
    return xr.apply_ufunc(fn, data, input_core_dims=[[axis_name]], kwargs={"axis": -1})


def plot_seasonal_cyc(ax, x, varname, fn=np.std, use_quantile=False, **plot_kwargs):
    """print seasonal cycle of data in x on specified ax object"""

    ## specify arguments for computing confidence bounds
    if use_quantile:
        kwargs = dict(center_method="median", bounds_method="quantile")
    else:
        kwargs = dict(center_method="mean", bounds_method="std")

    ## func to compute stats
    inner_fn = lambda x: apply_along_axis(x, fn=fn, axis_name="time")
    outer_fn = lambda x: x.groupby("time.month").map(inner_fn)
    get_stats = lambda x: get_ensemble_stats(outer_fn(x[varname]), **kwargs)

    ## compute std for each dataset
    x_plot = get_stats(x)

    ## months (x-coordinate for plotting
    months = np.arange(1, 13)

    ## plot MPI ensemble mean
    plot_data = ax.plot(np.arange(1, 13), x_plot.sel(posn="center"), **plot_kwargs)

    ## plot MPI bounds
    kwargs = dict(c=plot_data[0].get_color(), ls="--", lw=1)
    for bound in ["upper", "lower"]:
        ax.plot(months, x_plot.sel(posn=bound), **kwargs)

    return plot_data


def plot_seasonal_comp(
    ax,
    x0,
    x1,
    varname="T_34",
    fn=np.std,
    use_quantile=False,
    plot_kwargs0=dict(),
    plot_kwargs1=dict(),
):
    """Plot comparison of seasonal cycle between 2 ensembles"""

    ## plot data
    plot_data0 = plot_seasonal_cyc(
        ax, x0, varname=varname, fn=fn, use_quantile=use_quantile, **plot_kwargs0
    )
    plot_data1 = plot_seasonal_cyc(
        ax, x1, varname=varname, fn=fn, use_quantile=use_quantile, **plot_kwargs1
    )

    ## format ax
    ax.set_xticks([1, 8, 12], labels=["Jan", "Aug", "Dec"])
    ax.set_xlabel("Month")
    ax.set_ylim([0.4, 1.2])
    ax.set_yticks([])

    return plot_data0, plot_data1


def ls_fit_xr(Y_data, idx):
    """applies least-squares fit to index and xr.Dataset/xr.DataArray"""
    return xr_wrapper(ls_fit_xr_da, ds=Y_data, idx=idx)


def rho_xr(Y_data, idx):
    """applies correlation coefficient to index and xr.Dataset/xr.DataArray"""
    return xr_wrapper(rho_xr_da, ds=Y_data, idx=idx)


def xr_wrapper(xr_da_func, ds, **kwargs):
    """wrapper function which applies dataarray function to dataset"""
    if type(ds) is xr.DataArray:
        return xr_da_func(ds, **kwargs)

    else:
        results = []
        varnames = list(ds)
        for varname in varnames:
            result = xr_da_func(ds[varname], **kwargs)
            results.append(result)

        return xr.merge(results)


def ls_fit_xr_da(Y_data, **kwargs):
    """xarray wrapper for ls_fit (dataarray version).
    'E_data' and 'y_data' are both dataarrays"""

    # Parse inputs
    idx = kwargs["idx"]

    ## Identify regression dimension
    regr_dim = idx.dims[0]
    other_dims = [d for d in Y_data.dims if d != regr_dim]

    ## Stack the non-regression dimensions, if they exist
    if len(other_dims) > 0:
        Y_stack = Y_data.stack(other_dims=other_dims)
    else:
        Y_stack = Y_data

    ## Empty array to hold results
    coefs = Y_stack.isel({regr_dim: slice(None, 2)})
    coefs = coefs.rename({regr_dim: "coef"})
    coefs["coef"] = pd.Index(["m", "b"], name="coef")

    ## perform regression
    coefs.values = ls_fit(idx.values[:, None], Y_stack.values)
    return coefs.unstack()


def rho_xr_da(Y_data, **kwargs):
    """xarray wrapper for rho (dataarray version).
    'idx' and 'y_data' are both dataarrays"""

    # Parse inputs
    idx = kwargs["idx"]

    ## Stack the non-regression dimensions
    regr_dim = idx.dims[0]
    other_dims = [d for d in Y_data.dims if d != regr_dim]
    Y_stack = Y_data.stack(other_dims=other_dims)

    ## Empty array to hold results
    coef = Y_stack.isel({regr_dim: 0}, drop=True)

    ## compute correlation
    coef.values = get_rho(idx.values, Y_stack.values)

    return coef.unstack()


def add_constant(x):
    """Add a column of ones to matrix, representing constant coefficient for mat. mult."""
    x = np.append(x, np.ones([x.shape[0], 1]), axis=1)
    return x


def get_rho(x, Y):
    """Get correlation between many timeseries (Y) and single timeseries (x)"""
    Y = Y - np.nanmean(Y, axis=0)
    x = x - np.nanmean(x)
    Y[np.isnan(Y)] = 0.0
    x[np.isnan(x)] = 0.0
    rho = (x.T @ Y) / np.sqrt(np.sum(Y**2, axis=0) * np.dot(x, x))
    return rho


def ls_fit(x, Y):
    """Function solves least-squares to get coefficients for linear projection of x onto Y
    returns coef, a vector of the best-fit coefficients"""
    # Add array of ones to feature vector (to represent constant term in linear model)
    x = add_constant(x)
    coef = np.linalg.inv(x.T @ x) @ x.T @ Y
    return coef


def seasonal_embed(data, month_idxs, fill=0.0):
    """given m x n array 'data' and length-n array 'month_idxs',
    embed data to form (m x 12) x n array 'data_tilde', such that
    data_tilde[12*m:12*(m+1), i] = data[:,i] if (month_idxs[i]==m)
    and 'fill' otherwise"""

    ## Get size of state vector
    m = data.shape[0]

    ## allocate array to hold embedded data
    data_tilde = fill * np.ones([m * 12, *data.shape[1:]], dtype=data.dtype)

    ## loop through row block
    for m_idx in range(12):
        ## Get sample indices occuring in given month
        is_month = m_idx == month_idxs

        ## Put this data into the relevant rows of embedded data
        ## think these indices are wrong...
        data_tilde[m * m_idx : m * (m_idx + 1), is_month] = data[:, is_month]

    return data_tilde


def reverse_seasonal_embed(data_tilde, month_idxs):
    """reverse seasonal embedding"""

    m_tilde, n = data_tilde.shape
    m = np.round(m_tilde / 12).astype(int)

    data = np.zeros([m, n])

    ## reverse seasonal embedding
    for m_idx in range(12):
        ## Get sample indices occuring in given month
        is_month = m_idx == month_idxs

        ## Put this data into the relevant rows of embedded data
        ## think these indices are wrong...
        data[:, is_month] = data_tilde[m * m_idx : m * (m_idx + 1), is_month]

    return data


def get_angle(c):
    """get angle of complex number (or array of complex numbers).
    Wrapper for np.angle which returns angle in [0,2pi) instead
    of (-pi,pi]"""

    ## compute angle in (-pi,pi]
    phi = np.angle(c)

    ## switch range from (-pi,pi] to [0,2pi)
    phi[phi < 0] = phi[phi < 0] + 2 * np.pi

    return phi


def format_psd_ax(ax):
    """format PSD plot's axis as desired"""

    ## set limits
    ax.set_xlim([1 / 20, 5])
    ax.set_ylim([1e-3, 10])

    ## get ENSO and 1st combination frequencies
    Mfreq_enso = np.array([1 / 2.5, 1 / 5.5])
    Mfreq_fplus = 1 + Mfreq_enso
    Mfreq_fmins = 1 - Mfreq_enso

    ## shade these regions
    for Mfreq in [Mfreq_enso, Mfreq_fplus, Mfreq_fmins]:
        ax.axvspan(*Mfreq, fc="gray", alpha=0.1)

    ## label them
    y_max = ax.get_ylim()[1]
    kwargs = dict(ha="center", va="top", size=6)
    ax.text(np.mean(Mfreq_enso) * 0.9, y_max, "$f_{E}$", **kwargs)
    ax.text(np.mean(Mfreq_fplus) + 0.1, y_max, "$1+f_{E}$", **kwargs)
    ax.text(np.mean(Mfreq_fmins), y_max, "$1-f_{E}$", **kwargs)

    ## x-axis labels
    per = [10, 5.5, 2.5, 1, 0.5]
    xt = 1.0 / np.array(per)
    ax.set_xticks(xt)
    ax.set_xticklabels(map(str, per))
    ax.set_xlabel("Period (year)")

    return


def plot_psd(ax, psd_results, color=None, label=None):
    """plot PSD stuff for given data on specified ax"""

    ## "unzip" psd results
    psd, psd_sig, psd_ci = psd_results

    ## make sure 'member' is dimension
    if "member" not in psd.dims:
        psd = psd.expand_dims("member")
        psd_sig = psd_sig.expand_dims("member")

    ## plot ensemble mean
    plot_data = ax.loglog(psd.freq, psd.mean("member"), lw=2, label=label, c=color)

    ## plot ensemble spread
    ax.fill_between(
        psd.freq,
        psd.quantile(0.1, dim="member"),
        psd.quantile(0.9, dim="member"),
        fc=plot_data[0].get_color(),
        alpha=0.3,
    )

    ## plot 95% confidence levels
    ax.semilogx(
        psd_sig.freq,
        psd_sig.mean("member"),
        color=plot_data[0].get_color(),
        linestyle="--",
        lw=1,
    )

    ## format axis
    format_psd_ax(ax)

    return


def plot_xcorr(ax, data, **plot_kwargs):
    """plot mean and bounds for data"""

    ## center
    plot_data = ax.plot(data.lag, data.sel(posn="center"), **plot_kwargs)

    ## bounds
    ax.fill_between(
        data.lag,
        data.sel(posn="upper"),
        data.sel(posn="lower"),
        color=plot_data[0].get_color(),
        alpha=0.2,
    )

    ## set axis specs
    ax.set_ylim([-0.5, 1.05])
    ax.set_xlim([-18, 18])
    ax.set_xticks([])
    ax.set_yticks([])
    axis_kwargs = dict(c="k", lw=0.5, alpha=0.5)
    ax.axhline(0, **axis_kwargs)
    ax.axvline(0, **axis_kwargs)

    return


def make_composite_helper(
    idx,
    data,
    peak_month=1,
    q=0.85,
    check_cutoff=lambda x, cut: x > cut,
    event_type=None,
):
    """get samples for composite (but don't average yet)"""

    ## add ensemble dimension if it doesn't exist
    if "member" not in data.dims:
        idx = idx.expand_dims("member")
        data = data.expand_dims("member")

    ## stack member and time dimension
    idx_ = idx.stack(sample=["member", "time"])
    data_ = data.stack(sample=["member", "time"])
    year = idx_.time.dt.year
    month = idx_.time.dt.month

    ## get valid years and month
    valid_year = (year > year[0]) & (year < year[-1])
    is_peak_month = peak_month == month
    valid_idx = is_peak_month & valid_year

    ## get cutoff
    cutoff = idx_[valid_idx].quantile(q=q).values.item()

    ## find peak indices
    meets_cutoff = check_cutoff(x=idx_, cut=cutoff)
    peak_idx = np.where(meets_cutoff & valid_idx)[0]

    ## filter peaks by event type (e.g., 1-year v 2-year)
    if event_type is not None:
        peak_idx = sort_peak_idx(peak_idx)[event_type]

    ## create composite by pulling relevant samples
    comp = []
    for j in peak_idx:
        comp_ = data_.isel(sample=slice(j - 12, j + 13))
        comp_ = comp_.drop_vars(("sample", "member", "time"))
        comp_ = comp_.rename({"sample": "lag"})
        comp_["lag"] = np.arange(-12, 13)
        comp.append(comp_)

    ## put in xarray format
    comp = xr.concat(comp, dim=pd.Index(np.arange(len(peak_idx)), name="sample"))

    return comp


def sort_peak_idx(peak_idxs):
    """sort into single-year, double-year, or triple-year"""

    ## empty lists to hold results
    peak_idxs1 = []
    peak_idxs2 = []
    peak_idxs3 = []

    ## loop counter
    i = 0

    ## loop through peak indices
    while i < len(peak_idxs):

        ## get peak index variable
        peak_idx = peak_idxs[i]

        ## check for triple-year
        if ((peak_idx + 24) in peak_idxs) & ((peak_idx + 12) in peak_idxs):
            peak_idxs3.append(peak_idx)
            i += 3

        ## check for double-year
        elif (peak_idx + 12) in peak_idxs:
            peak_idxs2.append(peak_idx)
            i += 2

        ## otherwise, count as single-year
        else:
            peak_idxs1.append(peak_idx)
            i += 1

    ## convert to numpy array
    peak_idxs1 = np.array(peak_idxs1)
    peak_idxs2 = np.array(peak_idxs2)
    peak_idxs3 = np.array(peak_idxs3)

    return {1: peak_idxs1, 2: peak_idxs2, 3: peak_idxs3}


def make_composite(
    idx,
    data,
    peak_month=1,
    q=0.85,
    check_cutoff=lambda x, cut: x > cut,
    avg=True,
    event_type=None,
):
    """get composite for data"""

    ## Get samples for composite
    kwargs = dict(
        idx=idx,
        data=data,
        peak_month=peak_month,
        q=q,
        check_cutoff=check_cutoff,
        event_type=event_type,
    )
    comp = make_composite_helper(**kwargs)

    ## average to create composite (otherwise, make spaghetti)
    if avg:
        comp = comp.mean("sample")

    return comp.transpose("lag", ...)


def reconstruct_helper(scores, model, other_coord=None):
    """reconstruct scores for given model"""

    ## get weights
    weights = get_weight(model).values

    ## put raw scores into xarray
    n = scores.shape[1]
    scores_xr = xr.zeros_like(model.scores().isel(time=slice(None, n)))
    scores_xr.values = scores / weights

    ## do inverse transform
    data = model.inverse_transform(scores_xr)

    ## rename coord
    if other_coord is None:
        other_coord = pd.Index(np.arange(n), name="n")
    data = data.rename({"time": other_coord.name})
    data[other_coord.name] = other_coord.values

    return data


def get_weight(model):
    """Get weights for data (used for LIM)"""

    ## get explained variance of projected data relative to total
    expvar_ratio_total = model.explained_variance_ratio().sum()
    expvar_first = model.explained_variance()[0]

    return expvar_ratio_total / np.sqrt(expvar_first)


def get_W_tropics(model):
    """get weights to subset EOFs to tropics"""

    ## get weighting matrix
    L = xr.ones_like(model.components().isel(mode=0))
    L = L.where((L.latitude <= 5) & (L.latitude >= -5), other=0)

    ## get components (fill NaNs with zeros)
    U = model.components().where(~np.isnan(model.components()), other=0)

    ## get weighting matrix
    W = np.einsum("kij,ij,lij", U, L, U)

    return W


def get_W_tropics_multi(models):
    """get weights to subset EOFs to tropics"""

    ## Get W matrix for each model
    Ws = [get_W_tropics(model) for model in models]

    return scipy.linalg.block_diag(*Ws)


def plot_hov(ax, x, beta=1):
    """plot hovmoller of meridional avg on equator"""

    ## get colorbar levels
    sst_lev = beta * make_cb_range(3, 0.3)
    ssh_lev = beta * make_cb_range(17.5, 2.5)

    ## plot SST
    cf = ax.contourf(
        x.longitude,
        x.lag,
        x["sst"],
        cmap="cmo.balance",
        levels=sst_lev,
        extend="both",
    )

    ## thermocline
    co = ax.contour(
        x.longitude,
        x.lag,
        x["ssh"],
        colors="k",
        levels=ssh_lev,
        extend="both",
        linewidths=0.8,
        alpha=0.8,
    )

    ## set xlimit and add guidelines for Niño 3.4 region
    ax.set_xlim([120, 280])
    kwargs = dict(lw=0.5, c="w", ls="--", alpha=0.5)
    ax.axhline(0, **kwargs)
    ax.axvline(190, **kwargs)
    ax.axvline(240, **kwargs)

    return cf, co


def label_hov_yaxis(ax, peak_mon):
    """label y-axis with ±1 year"""

    ## get corresponding month name
    mon_name = calendar.month_name[peak_mon][:3]

    ## add labels
    ax.set_yticks(
        [-12, 0, 12], labels=[f"{mon_name}(-1)", f"{mon_name}(0)", f"{mon_name}(+1)"]
    )

    return


def remove_sst_dependence(ds, sst_idx, remove_from_sst=False):
    """function to remove linear dependence of variables on SST"""

    ## get h-variables and T-variables
    if remove_from_sst:
        h_vars = list(ds)
        T_vars = []
    else:
        h_vars = [n for n in list(ds) if ("h" in n) or ("d20" in n)]
        T_vars = list(set(list(ds)) - set(h_vars))

    ## create array to hold results
    ds_hat = copy.deepcopy(ds[h_vars])

    ## add "member" dimension if not included
    if "member" not in ds.dims:
        ds_hat = ds_hat.expand_dims("member")
        sst_idx = sst_idx.expand_dims("member")

    ## make dimensions align
    ds_hat = ds_hat.transpose("member", ...)
    sst_idx = sst_idx.transpose("member", ...)

    ## add SST index to array
    dims = ("member", "time")
    ds_hat = ds_hat.assign_coords(dict(sst_idx=(dims, sst_idx.data)))

    ## form sample dim
    ds_hat = ds_hat.stack(sample=["member", "time"])

    ## function to remove linear dependence
    fn = lambda x: detrend_dim(x, deg=1, dim="sst_idx")

    ## remove linear dependence for each month separately
    ds_hat = ds_hat.groupby("time.month").map(fn).unstack("sample")

    ## drop sst index as dim and add as variable
    ds_hat = ds_hat.drop_vars("sst_idx")

    ## merge with T-data
    ds_hat = xr.merge([ds_hat, ds[T_vars]])

    return ds_hat.squeeze()


def get_merimean_composite(idx, data, lat_range=[-5, 5], components=None, **kwargs):
    """Get meridional-mean composite for given index and data"""

    ## raw composite
    comp = make_composite(idx=idx, data=data, **kwargs)

    ## func to get meridional mean
    get_merimean = lambda x: x.sel(latitude=slice(*lat_range)).mean("latitude")

    ## handle EOF reconstruction
    if "mode" in data:
        comp_merimean = reconstruct_fn(
            components=components, scores=comp, fn=get_merimean
        )

    else:
        comp_merimean = get_merimean(comp)

    return comp_merimean.transpose("lag", ...)


def get_composites(idx, data, **kwargs):
    """Get set of three composites for given dataset"""

    ## get data with linear dependence removed
    data_hat = remove_sst_dependence(data, idx, remove_from_sst=True)

    ## create composites (full data)
    comp = get_merimean_composite(idx=idx, data=data, **kwargs)

    ## create composites (no Niño 3.4 dependence)
    comp_hat = get_merimean_composite(idx=idx, data=data_hat, **kwargs)

    ## create composites (only Niño 3.4 dependence)
    comp_tilde = comp - comp_hat

    return comp, comp_tilde, comp_hat


def load_cesm_indices():
    """Load cesm indices"""

    ## get filepath for cesm data
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    cesm_fp = DATA_FP / "cesm"

    ## Load T,h
    Th = xr.open_dataset(cesm_fp / "Th_anom.nc")

    ## load cvdp indices
    cvdp = xr.open_dataset(cesm_fp / "cvdp_anom.nc", decode_times=False)

    ## remove last timestep for Th data and fix times on cvdp
    Th = Th.isel(time=slice(None, -1))
    cvdp["time"] = Th.time

    return xr.merge([Th, cvdp])


def regress_core(Y, X, dim="time"):
    """compute regression for Y onto X"""

    ## compute covariance
    cov = xr.cov(X, Y, dim=dim, ddof=0)

    ## get variance (and add small number to prevent overflow)
    var_X = X.var(dim=dim) + np.finfo(np.float32).eps

    return cov / var_X


def regress(data, y_var, x_var, fn_x=lambda x: x):
    """compute regression for y_var onto x_var"""

    ## apply fn_x to x data
    fn_x_eval = fn_x(data[x_var])

    return regress_core(Y=data[y_var], X=fn_x_eval, dim="time")


def regress_proj(data, y_var, x_var, fn_x=lambda x: x, fn_y=lambda x: x):
    """compute regression for y_var onto x_var, from projected data"""

    ## compute covariance
    cov = reconstruct_cov_da(
        V_y=data[y_var],
        V_x=data[x_var],
        U_y=data[f"{y_var}_comp"],
        U_x=data[f"{x_var}_comp"],
        fn_x=fn_x,
        fn_y=fn_y,
    )

    ## compute variance
    var_X = reconstruct_var(
        scores=data[x_var],
        components=data[f"{x_var}_comp"],
        fn=fn_x,
    )

    return cov / var_X


def multi_regress(data, y_var, x_vars):
    """multiple linear regression"""

    ## Get covariates and targets
    X = data[x_vars].to_dataarray(dim="i")
    Y = data[y_var]

    ## compute covariance matrices
    YXt = xr.cov(Y, X, dim=["member", "time"])
    XXt = xr.cov(X, X.rename({"i": "j"}), dim=["member", "time"])

    ## invert XX^T
    XXt_inv = xr.zeros_like(XXt)
    XXt_inv.values = np.linalg.inv(XXt.values)

    ## get least-squares fit, YX^T @ (XX^T)^{-1}
    m_proj = (YXt * XXt_inv).sum("i")

    ## reconstruct spatial fields
    # get name of component variable
    if "ddt_" in y_var:
        comp_name = y_var[4:]
    else:
        comp_name = y_var

    # do reconstruction
    m = reconstruct_fn(
        scores=m_proj,
        components=data[f"{comp_name}_comp"],
        fn=lambda x: x,
    )

    return m.to_dataset(dim="j")


def multi_regress_bymonth(data, y_var, x_vars):
    """do multiple linear regression for each month separately"""
    return data.groupby("time.month").map(multi_regress, y_var=y_var, x_vars=x_vars)


def regress_xr(data, y_vars, x_vars, helper_fn=None, **helper_kwargs):
    """
    Regress projected y_vars onto x_vars in given dataset,
    and reconstruct.
    """

    ## get helper function if not specified
    if helper_fn is None:
        helper_fn = regress_xr_proj

    ## compute projected coefficients
    m_proj = helper_fn(data, y_vars=y_vars, x_vars=x_vars, **helper_kwargs)

    ## loop thru variables to reconstruct
    m = []
    for n in list(m_proj):

        ## check if 'ddt' is in name
        if "ddt_" in n:
            comp_name = n[4:]
        else:
            comp_name = n

        ## reconstruct
        m_ = src.utils.reconstruct_fn(
            scores=m_proj[n],
            components=data[f"{comp_name}_comp"],
            fn=lambda x: x,
        )

        ## append to list
        m.append(m_.rename(n))

    return xr.merge(m)


def regress_xr_proj(data, y_vars, x_vars):
    """
    Regress projected y_vars onto x_vars in given dataset
    """

    ## Get covariates and targets
    X = data[x_vars].to_dataarray(dim="i")
    Y = data[y_vars].to_dataarray(dim="k")

    ## compute covariance matrices
    YXt = xr.cov(Y, X, dim=["member", "time"])
    XXt = xr.cov(X, X.rename({"i": "j"}), dim=["member", "time"])

    # ## invert XX^T
    XXt_inv = xr.zeros_like(XXt)
    XXt_inv.values = np.linalg.inv(XXt.values)

    ## get least-squares fit, YX^T @ (XX^T)^{-1}
    m_proj = (YXt * XXt_inv).sum("i")

    ## convert to dataset
    m_proj = m_proj.to_dataset(dim="k")

    return m_proj


def regress_xr_bymonth(data, y_vars, x_vars):
    """
    apply 'regress_xr' function by month
    """

    kwargs = dict(y_vars=y_vars, x_vars=x_vars)
    return data.groupby("time.month").map(regress_xr, **kwargs)


def get_F(month, max_order=5):
    """function to get fourier matrix"""

    ## coefficients to build fourier matrix
    order_np = np.arange(0, max_order + 1)

    ## sin
    sin_order_coord = xr.Coordinates(dict(order=[f"sin_{n}" for n in order_np[1:]]))
    sin_order = xr.DataArray(order_np[1:], coords=sin_order_coord)

    ## cos
    cos_order_coord = xr.Coordinates(dict(order=[f"cos_{n}" for n in order_np]))
    cos_order = xr.DataArray(order_np, coords=cos_order_coord)

    ## convert month to theta
    theta = (month - 0.5) / 12 * (2 * np.pi)

    ## build the matrix
    F = xr.concat(
        [np.cos(cos_order * theta), np.sin(sin_order * theta)],
        dim="order",
    )

    return F


def regress_harm(data, y_vars, x_vars):
    """
    Regress projected y_vars onto x_vars in given dataset
    """

    ## Get covariates and targets
    X = data[x_vars].to_dataarray(dim="ell")
    Y = data[y_vars].to_dataarray(dim="k")

    ## stack harmonic dim
    X = X.stack(i=["ell", "order"])
    harm_coord = copy.deepcopy(X.i)

    ## compute covariance matrices
    X_ = X.rename({"i": "j"}).drop_vars(["order", "ell"])
    XXt = xr.cov(X, X_, dim=["member", "time"])
    YXt = xr.cov(Y, X, dim=["member", "time"])

    # ## invert XX^T
    XXt_inv = xr.zeros_like(XXt)
    XXt_inv.values = np.linalg.inv(XXt.values)

    ## get least-squares fit, YX^T @ (XX^T)^{-1}
    m_proj = (YXt * XXt_inv).sum("i")

    ## convert to dataset
    m_proj = m_proj.to_dataset(dim="k")

    ## add back harmonic coord and unstack
    m_proj = m_proj.rename({"j": "i"}).assign_coords(i=harm_coord).unstack("i")

    return m_proj


def regress_harm_wrapper(data, y_vars, x_vars, max_order=3):
    """
    Regress projected y_vars onto x_vars in given dataset
    """

    ## matrix to project data on harmonics
    F = get_F(month=data.time.dt.month, max_order=max_order)

    ## updated data
    data_proj = xr.merge([F * data[x_vars], data[y_vars]])

    ## compute coefficients
    coefs_proj = regress_harm(data=data_proj, y_vars=y_vars, x_vars=x_vars)

    ## inverse Fourier transform
    month = xr.DataArray(np.arange(1, 13), coords=dict(month=np.arange(1, 13)))
    # coefs = scores = (get_F(month - 0.5) * coefs_proj).sum("order")
    coefs = scores = (get_F(month) * coefs_proj).sum("order")

    return coefs


def remove_sst_dependence_core(h, T, dims=["time", "member"]):
    """function to remove linear dependence of h on T"""

    ## get covariance/variance
    cov = xr.cov(h, T, dim=dims, ddof=0)
    var = T.var(dim=dims)

    ## remove linear dependence
    return h - cov / var * T


def remove_sst_dependence_v2(Th, T_var="T_3", h_var="h_w", dims=["time", "member"]):
    """remove sst dependence by month"""

    helper_fn = lambda x: remove_sst_dependence_core(h=x[h_var], T=x[T_var], dims=dims)

    ## apply to each month
    return Th.groupby("time.month").map(helper_fn)


def get_ddt(ds, is_forward=True):
    """compute time derivative for each variable in dataset"""

    ## transpose so time is last dimension
    ds = ds.transpose(..., "time")

    ## loop through variables
    for n in list(ds):

        if f"ddt" in n:
            pass

        elif "comp" in n:
            ds[f"ddt_{n}"] = ds[n]

        else:
            ## create empty variable and fill with gradient
            ds[f"ddt_{n}"] = xr.zeros_like(ds[n])
            ds[f"ddt_{n}"].values = src.XRO.gradient(
                ds[n].values, is_forward=is_forward
            )

    return ds


def get_THF(bar, prime):
    """thermocline feedback"""
    return get_wdTdz(w=bar["w"], T=prime["T"])


def get_EKM(bar, prime):
    """thermocline feedback"""
    return get_wdTdz(T=bar["T"], w=prime["w"])


def get_ZAF(bar, prime, u_var="u", T_var="T"):
    """zonal advective feedback"""
    return -get_udTdx(T=bar[T_var], u=prime[u_var])


def get_DD(bar, prime, u_var="u", T_var="T"):
    """dynamical damping"""
    return -get_udTdx(T=prime[T_var], u=bar[u_var])


def get_MAF(bar, prime, v_var="v", T_var="T"):
    """meridional advection"""

    return -get_vdTdy(T=bar[T_var], v=prime[v_var])


def get_DDM(bar, prime, v_var="v", T_var="T"):
    """meridional advection"""

    return -get_vdTdy(T=prime[T_var], v=bar[v_var])


def get_feedbacks(bar, prime):
    """
    Given bar and prime for {u,w,T}, compute following feedbacks:
        - thermocline
        - Ekman
        - zonal advective
        - dynamical damping
    """

    ## initialize dataset
    feedbacks = xr.Dataset()

    ## compute
    feedbacks["THF"] = get_THF(bar, prime)
    feedbacks["EKM"] = get_EKM(bar, prime)
    feedbacks["ZAF"] = get_ZAF(bar, prime)
    feedbacks["DD"] = get_DD(bar, prime)
    feedbacks["ADV"] = (
        feedbacks["THF"] + feedbacks["EKM"] + feedbacks["ZAF"] + feedbacks["DD"]
    )

    return feedbacks


def get_surface_feedbacks(bar, prime, u_var="uvel", v_var="vvel", T_var="sst"):
    """
    Given bar and prime for {u,v,T}, compute following feedbacks:
        - zonal/meridional advective
        - dynamical damping (zonal/meridional)
    """

    ## initialize dataset
    feedbacks = xr.Dataset()

    ## compute
    feedbacks["ZAF_"] = get_ZAF(bar, prime, u_var=u_var, T_var=T_var)
    feedbacks["DD_"] = get_DD(bar, prime, u_var=u_var, T_var=T_var)
    feedbacks["MAF_"] = get_MAF(bar, prime, v_var=v_var, T_var=T_var)
    feedbacks["DDM_"] = get_DDM(bar, prime, v_var=v_var, T_var=T_var)
    feedbacks["ADV_"] = (
        feedbacks["ZAF_"] + feedbacks["DD_"] + feedbacks["MAF_"] + feedbacks["DDM_"]
    )

    return feedbacks


def decompose_feedback_change(
    bar_early,
    bar_late,
    prime_early,
    prime_late,
    feedback_fn,
    prefix="",
):
    """
    Decompose feedback change into:
        - "mean" component
        - "anomaly" component
        - "nonlinear" component
    """

    ## empty dataset to hold result
    feedback_change = xr.Dataset()

    ## compute differences
    delta_bar = bar_late - bar_early
    delta_prime = prime_late - prime_early

    ## compute
    feedback_change[f"{prefix}_mean"] = feedback_fn(delta_bar, prime_early)
    feedback_change[f"{prefix}_anom"] = feedback_fn(bar_early, delta_prime)
    feedback_change[f"{prefix}_nl"] = feedback_fn(delta_bar, delta_prime)

    return feedback_change


def decompose_feedback_changes(
    bar_early,
    bar_late,
    prime_early,
    prime_late,
):
    """
    Wrapper function to decompose feedback changes into:
        - "mean" component
        - "anomaly" component
        - "nonlinear" component
    """

    ## specify feedback names and functions
    names = ["THF", "EKM", "ZAF", "DD"]
    fns = [get_THF, get_EKM, get_ZAF, get_DD]

    ## shared args
    kwargs = dict(
        bar_early=bar_early,
        bar_late=bar_late,
        prime_early=prime_early,
        prime_late=prime_late,
    )

    ## empty list to hold result
    feedbacks = []

    ## loop thru feedbacks
    for n, fn in zip(names, fns):
        feedbacks.append(
            decompose_feedback_change(
                feedback_fn=fn,
                prefix=f"delta_{n}",
                bar_early=bar_early,
                bar_late=bar_late,
                prime_early=prime_early,
                prime_late=prime_late,
            ),
        )

    return xr.merge(feedbacks)


def reconstruct_wrapper(data, fn=lambda x: x):
    """
    Convenience function to reconstruct data which has both scores
    and components
    """

    ## first, get list of components
    comp_names = [v for v in list(data) if "_comp" in v]
    scores_names = [v[:-5] for v in comp_names]
    rename_dict = {n0: n1 for n0, n1 in zip(comp_names, scores_names)}

    ## reconstruct
    recon = reconstruct_fn(
        scores=data[scores_names],
        components=data[comp_names].rename(rename_dict),
        fn=fn,
    )

    return recon


def reconstruct_var_wrapper(data, fn=lambda x: x):
    """
    Convenience function to reconstruct data which has both scores
    and components
    """

    ## first, get list of components
    comp_names = [v for v in list(data) if "_comp" in v]
    scores_names = [v[:-5] for v in comp_names]
    rename_dict = {n0: n1 for n0, n1 in zip(comp_names, scores_names)}

    ## reconstruct
    recon = reconstruct_var(
        scores=data[scores_names],
        components=data[comp_names].rename(rename_dict),
        fn=fn,
    )

    return recon


def reconstruct_clim(data, fn=lambda x: x):
    """
    Convenience function to reconstruct monthly climatology from projected data
    """
    return reconstruct_wrapper(data.groupby("time.month").mean(), fn=fn)


def get_ml_avg_ds(ds, **ml_kwargs):
    """wrapper function to compute mixed layer average for each
    variable in given xr.dataset"""

    for v in list(ds):
        if "z_t" in ds[v].coords:
            ds[f"{v}_ml"] = get_ml_avg(ds[v], **ml_kwargs)

    return ds


def get_ml_avg(data, Hm, delta=5, H0=None):
    """func to average data from surface to Hm + delta"""

    ## tweak integration bounds
    if H0 is None:
        Hm = Hm + delta

    else:
        Hm = H0

    ## average over everything above the mixed layer
    return data.where(data.z_t <= Hm).mean("z_t")


def get_wdTdz(w, T):
    """function to get vertical flux (handles diff. w/T grids)"""

    ## get dTdz (convert from 1/cm to 1/m)
    dTdz = T.differentiate("z_t")

    return w * dTdz


def get_udTdx(u, T):
    """zonal advection"""

    ## number of meters in 1 deg longitude on equator
    m_per_deglon = get_dx(lat_deg=0, dlon_deg=1.0)

    ## differentiate and convert units to K/yr
    u_dTdx = u * T.differentiate("longitude") * 1 / m_per_deglon
    return u_dTdx


def get_vdTdy(v, T):
    """zonal advection"""

    ## number of meters in 1 deg latitude on equator
    m_per_deglat = get_dy(dlat_deg=1)

    ## differentiate and convert units to K/yr
    v_dTdy = v * T.differentiate("latitude") * 1 / m_per_deglat
    return v_dTdy


def get_dy(dlat_deg):
    """get spacing between latitudes in meters"""

    ## convert from degrees to radians
    dlat_rad = dlat_deg / 180.0 * np.pi

    ## multiply by radius of earth
    R = 6.378e6  # earth radius (centimeters)
    dlat_meters = R * dlat_rad

    return dlat_meters


def get_dx(lat_deg, dlon_deg):
    """get spacing between longitudes in meters"""

    ## convert from degrees to radians
    dlon_rad = dlon_deg / 180.0 * np.pi
    lat_rad = lat_deg / 180 * np.pi

    ## multiply by radius of earth
    R = 6.378e6  # earth radius (meters)
    dlon_meters = R * np.cos(lat_rad) * dlon_rad

    return dlon_meters


def get_dydx(data):
    """get dy and dx for given data"""

    ## empty array to hold result
    grid = xr.Dataset(
        coords=dict(
            latitude=data["latitude"].values,
            longitude=data["longitude"].values,
        ),
    )

    grid["dlat"] = grid["latitude"].values[1] - grid["latitude"].values[0]
    grid["dlon"] = grid["longitude"].values[1] - grid["longitude"].values[0]

    grid["dlat_rad"] = grid["dlat"] / 180.0 * np.pi
    grid["dlon_rad"] = grid["dlon"] / 180.0 * np.pi
    R = 6.378e8  # earth radius (centimeters)

    ## height of gridcell doesn't depend on longitude
    grid["dy"] = R * grid["dlat_rad"]  # unit: meters
    grid["dy"] = grid["dy"] * xr.ones_like(grid["latitude"])

    ## Compute width of gridcell
    grid["lat_rad"] = grid["latitude"] / 180 * np.pi  # latitude in radians
    grid["dx"] = R * np.cos(grid["lat_rad"]) * grid["dlon_rad"]

    return grid[["dy", "dx"]]


def regress_v2(data, y_var, x_vars):
    """multiple linear regression"""

    ## Get covariates and targets
    X = data[x_vars].to_dataarray(dim="i")
    Y = data[y_var]

    ## compute covariance matrices
    YXt = xr.cov(Y, X, dim=["member", "time"])
    XXt = xr.cov(X, X.rename({"i": "j"}), dim=["member", "time"])

    ## invert XX^T
    XXt_inv = xr.zeros_like(XXt)
    XXt_inv.values = np.linalg.inv(XXt.values)

    ## get least-squares fit, YX^T @ (XX^T)^{-1}
    m = (YXt * XXt_inv).sum("i")

    return m.to_dataset(dim="j")


def regress_bymonth(data, y_var, x_vars):
    """do multiple linear regression for each month separately"""
    return data.groupby("time.month").map(regress_v2, y_var=y_var, x_vars=x_vars)


def get_dT_sub(Tsub, Hm, delta=25):
    """
    Get temperature difference b/n entrainment zone and mixed layer.
    (positive if entrainment zone is warmer than ML)
    """

    ## find indices in ML and entrainment zone (ez)
    in_ml = Tsub.z_t <= Hm
    in_ez = (Tsub.z_t > Hm) & (Tsub.z_t < (delta + Hm))

    ## get Tbar and Tplus (following Frankignoul et al paper)
    Tbar = Tsub.where(in_ml).mean("z_t")
    Tplus = Tsub.where(in_ez).mean("z_t")

    ## get gradient
    dT = Tplus - Tbar

    return dT


def split_components(data):
    """separate components from given dataset"""

    ## get list of component variables
    comp_vars = [v for v in list(data) if "_comp" in v]

    ## get components and everything else
    components = data[comp_vars]
    other = data.drop_vars(comp_vars)

    ## rename variables in components
    components = components.rename({c: c[:-5] for c in comp_vars})

    return components, other


def format_subsurf_axs(axs):
    """add labels/formatting to 3-panel axs"""

    ## loop thru axs
    for ax in axs:
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlim([None, 281])
        ax.set_yticks([])
        ax.set_xlabel("Longitude")
    axs[0].set_yticks([300, 150, 0])
    axs[0].set_ylabel("Depth (m)")

    return


def format_hov_axs(axs):
    """put hovmoller axs in standardized format"""

    ## set fontsize
    font_kwargs = dict(size=8)
    axs[0].set_ylabel("Month", **font_kwargs)
    axs[0].set_title("Early", **font_kwargs)
    axs[1].set_title("Late", **font_kwargs)
    axs[2].set_title("Difference (x2)", **font_kwargs)

    axs[1].set_yticks([])
    axs[2].set_yticks([])
    axs[0].set_yticks([1, 5, 9, 12], labels=["Jan", "May", "Sep", "Dec"])

    for ax in axs:
        # ax.set_xlim([190, None])
        ax.set_xticks([190, 240])
        ax.axvline(240, ls="--", c="w", lw=1)
        ax.axvline(190, ls="--", c="w", lw=1)

    return


def merimean(x, lat_bound=5):
    """get meridional mean"""

    ## get bounds for latitude averaging
    coords = dict(
        longitude=slice(140, 285),
        latitude=slice(-lat_bound, lat_bound),
    )

    return x.sel(coords).mean("latitude")


def make_cycle_hov(ax, data, amp, is_filled=True, xticks=[190, 240], lat_bound=5):
    """plot data on ax object"""

    ## specify shared kwargs
    shared_kwargs = dict(levels=src.utils.make_cb_range(amp, amp / 5), extend="both")

    ## specify kwargs
    if is_filled:
        plot_fn = ax.contourf
        kwargs = dict(cmap="cmo.balance")

    else:
        plot_fn = ax.contour
        kwargs = dict(colors="k", linewidths=0.8)

    ## average over latitudes (if necessary)
    if "latitude" in data.coords:
        plot_data = merimean(data, lat_bound=lat_bound)
    else:
        plot_data = data

    ## do the plotting
    cp = plot_fn(
        plot_data.longitude,
        plot_data.month,
        plot_data.transpose("month", "longitude"),
        **kwargs,
        **shared_kwargs,
    )

    ## format ax object
    kwargs = dict(c="w", ls="--", lw=1)
    ax.set_xlim([145, 280])
    ax.set_xlabel("Lon")
    ax.set_xticks(xticks)
    for tick in xticks:
        ax.axvline(tick, **kwargs)

    return cp


def set_ylims(axs):
    lims = np.stack([ax.get_ylim() for ax in axs.flatten()], axis=0)

    lb = lims[:, 0].min()
    ub = lims[:, 1].max()

    for ax in axs:
        ax.set_ylim([lb, ub])

    return


def set_xlims(axs):
    lims = np.stack([ax.get_xlim() for ax in axs.flatten()], axis=0)

    lb = lims[:, 0].min()
    ub = lims[:, 1].max()

    for ax in axs:
        ax.set_xlim([lb, ub])

    return


def set_lims(axs):
    """set both x and y limits on pair of axes objects"""

    set_xlims(axs)
    set_ylims(axs)
    return


def eval_fn(data, varname, fn=None, months=None):
    """helper to get evaluated function"""

    ## subset for months and data
    data_subset = src.utils.sel_month(data[varname], months=months)

    ## reconstruct data
    if fn is None:
        fn_eval = data_subset

    else:

        ## get name of components
        if "ddt_" in varname:
            n = varname[4:]
        else:
            n = varname

        fn_eval = src.utils.reconstruct_fn(
            scores=data_subset, components=data[f"{n}_comp"], fn=fn
        )

    return fn_eval


def make_scatter2(ax, data, x_var, y_var, scale=1, fn_x=None, fn_y=None, months=None):
    """scatter plot data on axis"""

    ## helper function to stack time/member dims
    stack = lambda x: x.stack(sample=["member", "time"])

    ## evaluate data
    x = stack(eval_fn(data, x_var, fn=fn_x, months=months))
    y = stack(eval_fn(data, y_var, fn=fn_y, months=months)) * scale

    ## compute slope for best fit line
    slope = src.utils.regress_core(X=x, Y=y, dim="sample")

    ## convert to numpy
    slope = slope.values.item()

    ## plot data
    ax.scatter(x, y, s=0.5)

    ## plot best fit
    xtest = np.linspace(x.values.min(), x.values.max())
    ax.plot(xtest, slope * xtest, c="k", lw=1)

    ## plot some guidelines
    ax.axhline(0, ls="--", lw=0.8, c="k")
    ax.axvline(0, ls="--", lw=0.8, c="k")

    return slope


def plot_hov2(fig, ax, data, amp, label=None):
    """Plot hovmoller of longitude vs. year"""

    plot_data = ax.contourf(
        data.month,
        data.year,
        data.T,
        cmap="cmo.balance",
        extend="max",
        levels=src.utils.make_cb_range(amp, amp / 10),
    )
    cb = fig.colorbar(
        plot_data,
        orientation="horizontal",
        ticks=[-amp, 0, amp],
        label=label,
    )
    cb.ax.tick_params(labelsize=8)

    ## remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return


def align_pop_times(x, pop_vars=["uvel", "vvel", "u", "w", "T", "nhf"]):
    """fix misalignment b/n MMLEA and POP"""

    ## get variables in POP and MMLEA
    mmlea_vars = list(set(list(x)) - set(pop_vars))

    ## Get valid times
    valid_time = xr.date_range(
        start="1850-01-01",
        freq="MS",
        periods=3012,
        calendar="noleap",
        use_cftime=True,
    )

    ## fix time coord
    if len(mmlea_vars) > 0:
        x_pop = x[pop_vars].isel(time=slice(1, None)).assign_coords(time=valid_time)
        x_ = xr.merge([x_pop, x[mmlea_vars].sel(time=valid_time)])

    else:
        x_ = x[pop_vars].assign_coords(time=valid_time)

    return x_


def load_budget_data():
    """utility function to load budget data. (note: pop times already aligned!)"""

    ## directory with data
    CONS_DIR = pathlib.Path(os.environ["DATA_FP"], "cesm", "consolidated")

    ## open data and align pop times
    forced = xr.open_dataset(CONS_DIR / "budg_forced.nc")
    anom = xr.open_dataset(CONS_DIR / "budg_anom.nc")

    return forced, anom


def load_consolidated():
    """utility function to load consolidated data"""

    ## directory with data
    CONS_DIR = pathlib.Path(os.environ["DATA_FP"], "cesm", "consolidated")

    ## open data and align pop times
    forced = align_pop_times(xr.open_dataset(CONS_DIR / "forced.nc"))
    anom = align_pop_times(xr.open_dataset(CONS_DIR / "anom.nc"))

    return forced, anom


def get_windowed(data, window_size=480, stride=60):
    """Get windowed version of data (used for computing parameter values over time)"""

    ## get windowed data
    data_rolling = data.rolling({"time": window_size}, center=True)
    data_windowed = data_rolling.construct(window_dim="sample", stride=stride)

    ## trim (remove nan values)
    n_trim = int(window_size / 2 / stride)
    data_windowed = data_windowed.isel(time=slice(n_trim, -(n_trim - 1)))

    ## rename window coord from time to year
    year_coord = dict(year=data_windowed.time.dt.year.values)
    data_windowed = data_windowed.rename({"time": "year"}).assign_coords(year_coord)

    ## rename sample coord to time (use arbitrary time, used for seasonality)
    time_coord = dict(
        time=xr.date_range(start="1850-01-01", freq="MS", periods=window_size)
    )
    data_windowed = data_windowed.rename({"sample": "time"}).assign_coords(time_coord)

    return data_windowed


def get_params(fits, model):
    """Get parameters from fits dataarray"""

    ## get parameters from fits
    params = model.get_RO_parameters(fits)

    ## get normalized noise stats
    fix_coords = lambda x: x.assign_coords({"cycle": params.cycle})
    params["xi_T_norm"] = fix_coords(fits["normxi_stdac"].isel(ranky=0))
    params["xi_h_norm"] = fix_coords(fits["normxi_stdac"].isel(ranky=1))

    ## get wyrtki index
    sign = np.sign(params["F1"] * params["F2"])
    params["wyrtki"] = sign * np.sqrt(np.abs(params["F1"] * params["F2"]))

    return params.squeeze()


def get_H(T):
    """compute thermocline depth"""

    ## find index for max (negative gradient)
    min_idx = T.differentiate("z_t").argmin("z_t")

    return T.z_t.isel(z_t=min_idx)


def get_H_int(T, thresh=0):
    """get H using a different method"""

    ## compute vertical grad
    dTdz = T.differentiate("z_t")

    ## find where grad exceeds thresh
    dTdz = dTdz.where(np.abs(dTdz) > thresh, other=0)

    ## compute numerator and denom
    num = (dTdz.z_t * dTdz).integrate("z_t")
    den = dTdz.integrate("z_t")

    return num / den


def frac_change(x, inv=True):
    """get fractional change"""

    ## get inverse if desired
    if inv:
        x_ = 1 / x
    else:
        x_ = x

    ## compute initial value and change
    x0_ = x_.isel(year=0)

    return (x_ - x0_) / x0_
