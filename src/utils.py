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


def reconstruct_fn_da(components, scores, fn):
    """reconstruct function of spatial data from PCs. Assumes
    components and scores are xr.DataArrays"""

    ## get latitude weighting for components
    coslat_weights = np.sqrt(np.cos(np.deg2rad(components.latitude)))

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
    coslat_weights = np.sqrt(np.cos(np.deg2rad(components.latitude)))

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


def plot_nino34_box(ax, **kwargs):
    """outline Niño 3.4 region"""
    return plot_box(ax, lons=[190, 240], lats=[-5, 5], **kwargs)


def plot_nino3_box(ax, **kwargs):
    """outline Niño 3.4 region"""
    return plot_box(ax, lons=[210, 270], lats=[-5, 5], **kwargs)


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
    data_["time"] = new_idx

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
    extend = "min" if (amp_min == 0) else "both"

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


def plot_seasonal_cyc(ax, x, varname, use_quantile=False, **plot_kwargs):
    """print seasonal cycle of data in x on specified ax object"""

    ## specify arguments for computing confidence bounds
    if use_quantile:
        kwargs = dict(center_method="median", bounds_method="quantile")
    else:
        kwargs = dict(center_method="mean", bounds_method="std")

    ## func to compute stats
    get_std = lambda x: x.groupby("time.month").std("time")
    get_stats = lambda x: get_ensemble_stats(get_std(x[varname]), **kwargs)

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
    use_quantile=False,
    plot_kwargs0=dict(),
    plot_kwargs1=dict(),
):
    """Plot comparison of seasonal cycle between 2 ensembles"""

    ## plot data
    plot_data0 = plot_seasonal_cyc(
        ax, x0, varname=varname, use_quantile=use_quantile, **plot_kwargs0
    )
    plot_data1 = plot_seasonal_cyc(
        ax, x1, varname=varname, use_quantile=use_quantile, **plot_kwargs1
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
    idx, data, peak_month=1, q=0.85, check_cutoff=lambda x, cut: x > cut
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


def make_composite(
    idx, data, peak_month=1, q=0.85, check_cutoff=lambda x, cut: x > cut
):
    """get composite for data"""

    ## Get samples for composite
    kwargs = dict(
        idx=idx, data=data, peak_month=peak_month, q=q, check_cutoff=check_cutoff
    )
    comp = make_composite_helper(**kwargs)

    ## average to create composite
    comp = comp.mean("sample").transpose("lag", ...)

    return comp


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
