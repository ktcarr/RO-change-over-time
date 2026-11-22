import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RO change over time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import datetime
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import seaborn as sns
    import xarray as xr
    import tqdm
    import pathlib
    import cmocean
    import os
    import src.XRO
    import copy
    import scipy.stats
    import warnings
    import calendar
    import pandas as pd

    # import gsw

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return (
        DATA_FP,
        cmocean,
        copy,
        np,
        os,
        pathlib,
        pd,
        plt,
        scipy,
        sns,
        src,
        tqdm,
        warnings,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## specify args
    """)
    return


@app.cell
def _():
    ## should we remove median?
    REMOVE_MEDIAN = True

    ## specify T region
    T_REGION = "3"

    ## should we use ML avg?
    USE_ML_AVG = False

    ## use these...
    varnames = [f"T_{T_REGION}", "h_hat"]
    return REMOVE_MEDIAN, T_REGION, USE_ML_AVG, varnames


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Functions
    """)
    return


@app.cell
def _(
    REMOVE_MEDIAN,
    copy,
    np,
    os,
    pathlib,
    pd,
    src,
    tqdm,
    varnames,
    warnings,
    xr,
):
    def var_by_quant(x):
        """compute variance by quantiles"""

        ## get bounds
        bounds = x.quantile(dim=["member", "time"], q=[0.25, 0.75]).rename(
            {"quantile": "q"}
        )

        ## find samples in each class
        in_class0 = x < bounds.isel(q=0)  ## extreme La Niña
        in_class1 = (bounds.isel(q=0) <= x) & (x <= bounds.isel(q=1))  ## neutral
        in_class2 = bounds.isel(q=1) < x  ## extreme El Niño

        ## put in single dataarray
        in_class = xr.concat(
            [in_class0, in_class1, in_class2],
            dim=pd.Index(np.arange(3), name="intensity"),
            coords="minimal",
            compat="override",
        )

        ## zero out NaNs
        # is_invalid = np.abs(x) > (x.std() * 7)
        # if is_invalid.to_dataarray().sum() > 0:

        #     ## count fraction of invalid
        #     frac = is_invalid.to_dataarray().mean().values.item()
        #     print(f"Warning: {frac:.1e} frac. large values")

        #     ## fill large values with NaN
        #     x = x.where(~is_invalid, other=np.nan)

        ## compute variance from each
        N = (~np.isnan(x)).sum(["member", "time"])
        x_var = 1 / N * (x.where(in_class) ** 2).sum(["time", "member"])

        return x_var


    def get_rolling_std(data, n=20):
        """
        Get standard deviation, computing over time and ensemble member. To increase
        sample size for variance estimate, compute over time window of 2n+1
        years, centered at given year.
        """

        ## do the computation
        kwargs = dict(fn=np.std, n=n, reduce_ensemble_dim=False)
        data_std = src.utils.get_rolling_fn_bymonth(data, **kwargs)

        ## unstack year and month
        data_std = src.utils.unstack_month_and_year(data_std)

        return data_std


    def get_fits_over_time(data_rolling, model, by_member=False, **fit_kwargs):
        """Get RO fits for each ensemble member as a function of time."""

        ## empty list to hold results
        fits = []

        ## loop through years
        for y in tqdm.tqdm(data_rolling.year):

            ## get data for year
            data_y = data_rolling.sel(year=y)

            if by_member:

                ## separate fit for each ensemble member
                fits_ = []
                for m in data_rolling.member:
                    with warnings.catch_warnings(action="ignore"):
                        fits_.append(model.fit_matrix(data_y.sel(member=m), **fit_kwargs))
                fit = xr.concat(fits_, dim=data_y.member)

            else:

                ## fit for all ensemble members together
                with warnings.catch_warnings(action="ignore"):
                    fit = model.fit_matrix(data_y, **fit_kwargs)

            ## track fits
            # fits.append(fit.drop_vars(["time", "X", "Y", "Yfit"]))
            fits.append(fit)

        ## put back in xarray
        fits = xr.concat(fits, dim=data_rolling.year)

        return fits


    def get_fits_over_time_wrapper(
        data_rolling, model, by_member=False, fname=None, **fit_kwargs
    ):
        """wrapper function to handle saving/loading"""

        ## function to compute fits
        get_fits = lambda: get_fits_over_time(
            data_rolling, model=model, by_member=by_member, **fit_kwargs
        )

        ## if fname not specified, compute without loading/saving
        if fname is None:
            fits = get_fits()

        else:

            ## full path to file
            fp = pathlib.Path(os.environ["SAVE_FP"], "fits_cesm", fname)

            ## load if it already exists
            if fp.is_file():
                fits = xr.open_dataset(fp)

            ## otherwise, compute and save
            else:
                fits = get_fits()
                fits.to_netcdf(fp)

        return fits


    def get_sims_over_time(model, params, **simulation_kwargs):
        """Compute stats over time"""

        ## take ensemble mean if necessary
        if "member" in params.dims:
            params = params.mean("member")

        ## empty list to hold result
        sims = []

        ## loop through years
        for y in tqdm.tqdm(params.year):

            ## do simulation
            sim_y = model.simulate(fit_ds=params.sel(year=y), **simulation_kwargs)

            ## append
            sims.append(sim_y)

        ## put back in xarray
        sims = xr.concat(sims, dim=params.year)

        return sims


    def save(fig, fname, dpi=800, do_save=False):
        """save figure to file"""

        ## get save directory
        save_dir = pathlib.Path(os.environ["SAVE_FP"], "june-updates")

        ## get fname
        fname = f"{fname}-{varnames[0]}-subtract_median_{REMOVE_MEDIAN}.pdf"

        if do_save:
            fig.savefig(save_dir / fname, dpi=dpi, format="pdf")

        return


    def get_perturbed_NRO(params, nro_form_idx=None, nro_type="NROT_Lac"):
        """
        Fix values of R parameter set.
        if 'fix_others' is True, then other parameters are fixed to their
        initial value. Otherwise, given parameter is fixed to its initial value
        """

        ## initialize empty array to hold parameters
        params_new = copy.deepcopy(params)
        params_new[nro_type] = params_new[nro_type].transpose("year", "nro_form", ...)

        ## get numpy version of linear operator
        NRO_Lac = params_new[nro_type].values

        ## update parameter
        if nro_form_idx is None:
            NRO_Lac = NRO_Lac[:1]

        else:
            NRO_Lac[:, nro_form_idx] = NRO_Lac[:1, nro_form_idx]

        ## add back to parameters
        params_new[nro_type] = xr.ones_like(params_new[nro_type]) * NRO_Lac

        return params_new


    def get_ddt(data):
        """differentiate with respect to time"""
        data_ = copy.deepcopy(data)
        data_ = data_.assign_coords({"t_idx": ("time", np.arange(len(data_.time)))})
        data_ = data_.swap_dims({"time": "t_idx"})

        ## differentiate
        ddt_data = data_.differentiate("t_idx").swap_dims({"t_idx": "time"})

        ## mult. by 12 to convert from 1/mo to 1/yr
        return 12 * ddt_data

    return (
        get_ddt,
        get_fits_over_time_wrapper,
        get_perturbed_NRO,
        get_sims_over_time,
        save,
        var_by_quant,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### T, h
    """)
    return


@app.cell
def _(src):
    ## open data
    Th = src.utils.load_cesm_indices(load_z20=True, load_h_cust=True, max_grad=True)
    Th = Th.fillna(0)
    return (Th,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load ELI
    """)
    return


@app.cell
def _(DATA_FP, Th, pathlib, src, xr):
    ## Load ELI
    eli_all = xr.open_dataset(pathlib.Path(DATA_FP, "cesm/eli.nc"))
    eli_forced, eli = src.utils.separate_forced(eli_all)

    ## add to data
    Th_1 = xr.merge([Th, eli])
    return (Th_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Scale it
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Merged Niño index
    """)
    return


@app.cell(hide_code=True)
def _(Th_1, src):
    ## function to get merged nino3/34 regions
    def get_nino_merged(x):
        idx = dict(latitude=slice(-5, 5), longitude=slice(190, 270))
        return x.sel(idx).mean(["longitude", "latitude"])


    def get_Te(x):
        idx = dict(latitude=slice(-5, 5), longitude=slice(230, 280))
        return x.sel(idx).mean(["longitude", "latitude"])


    def get_Tw(x):
        idx = dict(latitude=slice(-5, 5), longitude=slice(180, 230))
        return x.sel(idx).mean(["longitude", "latitude"])


    ## spatial data
    forced, anom = src.utils.load_consolidated()

    ## compute
    Th_1["T_m"] = src.utils.reconstruct_wrapper(anom[["sst", "sst_comp"]], get_nino_merged)[
        "sst"
    ]
    Th_1["T_e"] = src.utils.reconstruct_wrapper(anom[["sst", "sst_comp"]], get_Te)["sst"]
    Th_1["T_w"] = src.utils.reconstruct_wrapper(anom[["sst", "sst_comp"]], get_Tw)["sst"]
    return anom, get_Te, get_nino_merged


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OHC
    """)
    return


@app.cell(hide_code=True)
def _(os, pathlib, src, xr):
    def load_consolidated_wide():
        """utility function to load consolidated data"""

        ## directory with data
        CONS_DIR = pathlib.Path(os.environ["DATA_FP"], "cesm", "consolidated_05")

        ## function to align and open
        kwargs = dict(pop_vars=["u", "u_comp", "T", "T_comp", "w", "w_comp"])
        align_and_open = lambda fp: src.utils.align_pop_times(xr.open_dataset(fp), **kwargs)

        ## open data and align pop times
        kwargs = dict(pop_vars=["u", "u_comp", "T", "T_comp", "w", "w_comp"])
        forced = align_and_open(CONS_DIR / "forced.nc")
        anom = align_and_open(CONS_DIR / "anom.nc")

        return forced, anom


    forced_wide, anom_wide = load_consolidated_wide()
    return (anom_wide,)


@app.cell(hide_code=True)
def _(Th_1, anom_wide, src):
    lon_avg = lambda x, lons: x.sel(longitude=slice(*lons)).mean("longitude")

    ## compute ohc
    Th_1["h_w_ohc"] = src.utils.reconstruct_wrapper(
        anom_wide[["T", "T_comp"]],
        lambda x: lon_avg(x.integrate("z_t"), (120, 180)),
    )["T"]

    ## compute ohc
    Th_1["h_ohc"] = src.utils.reconstruct_wrapper(
        anom_wide[["T", "T_comp"]],
        lambda x: lon_avg(x.integrate("z_t"), (120, 280)),
    )["T"]


    ## compute ml temp
    Th_1["T_3_ml"] = src.utils.reconstruct_wrapper(
        anom_wide[["T", "T_comp"]],
        lambda x: lon_avg(x.sel(z_t=slice(None, 50)).integrate("z_t"), (210, 270)),
    )["T"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Surface heat flux
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load
    """)
    return


@app.cell
def _(anom, src):
    flux_forced, flux_anom = src.utils.load_flux_data()

    ## switch sign convection to be positive into the ocean
    for _v in ["flns", "lhflx"]:
        flux_anom[_v] = -1 * flux_anom[_v]
        flux_forced[_v] = -1 * flux_forced[_v]

    ## add net heat flux data for convenience
    flux_anom[["nhf", "nhf_comp"]] = anom[["nhf", "nhf_comp"]]

    ## load into memory
    flux_anom.load();
    return (flux_anom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute regional averages
    """)
    return


@app.cell
def _(src):
    def smooth_121(x):
        """apply 1-2-1 smoothing"""

        ## get 1-1-1 avg
        x_111_sum = x.rolling({"time": 3}, center=True, min_periods=1).sum()

        return 1 / 4 * (x_111_sum + x)


    def get_Q(data, label, fn):
        """get heat flux over region using specified fn"""

        ### reconstruct flux over region
        hf = src.utils.reconstruct_wrapper(data, fn=fn)

        ## rolling mean (more comparable to dT/dt estimates?)
        # hf = smooth_121(hf)

        ## convert units to K/yr
        sec_per_mo = 8.64e4 * 30
        sec_per_yr = 12 * sec_per_mo
        rho = 1.02e3
        Cp = 4.2e3
        H = 50
        # H = 80

        ## apply scaling fac.
        Q = hf * sec_per_yr / (rho * Cp * H)

        return Q.rename({v: f"{v}_{label}" for v in list(hf)})

    return (get_Q,)


@app.cell
def _(Th_1, flux_anom, get_Q, get_Te, get_nino_merged, src, tqdm, xr):
    ## empty list to hold result
    Q = []

    for fn, _label in tqdm.tqdm(
        [
            (src.utils.get_nino3, "3"),
            (src.utils.get_nino34, "34"),
            (get_Te, "e"),
            (get_nino_merged, "m"),
        ]
    ):
        Q.append(get_Q(flux_anom, label=_label, fn=fn))

    ## list to xr
    Q = xr.merge(Q)

    ## add to T, h data
    Th_2 = xr.merge([Th_1, Q])
    return Q, Th_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### load 3-D data
    """)
    return


@app.cell
def _(DATA_FP, Th_2, np, pathlib, xr):
    Th_3d = xr.open_mfdataset(
        sorted(list(pathlib.Path(DATA_FP / "cesm" / "Th_3D").glob("*nc"))),
        combine="nested",
        concat_dim="member",
    )

    ## update coordinates to match
    Th_3d = Th_3d.assign_coords({"member": np.arange(100)}).compute()
    Th_3d = Th_3d.assign_coords({"time": Th_2.time})
    Th_3d = Th_3d.rename({"h": "thermocline"})
    return (Th_3d,)


@app.cell
def _(Th_3d):
    MLD = 50
    avg = lambda x, lons: x.sel(longitude=slice(*lons)).mean(["longitude"])
    Th_3d["T_3"] = avg(Th_3d[f"T_{MLD}"], (210, 270))
    Th_3d["T_34"] = avg(Th_3d[f"T_{MLD}"], (190, 240))
    Th_3d["T_4"] = avg(Th_3d[f"T_{MLD}"], (160, 210))
    Th_3d["T_e"] = avg(Th_3d[f"T_{MLD}"], (230, 280))
    Th_3d["T_w"] = avg(Th_3d[f"T_{MLD}"], (180, 230))
    Th_3d["h"] = avg(Th_3d["thermocline"], (120, 280))
    Th_3d["h_w"] = avg(Th_3d["thermocline"], (120, 180))
    Th_3d_1 = Th_3d.drop_vars(["thermocline", "T_50", "T_80"])
    return (Th_3d_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standardize/window
    """)
    return


@app.cell
def _(Q, Th_2, Th_3d_1, USE_ML_AVG, src):
    if USE_ML_AVG:
        for _v in ["T_3", "T_34", "T_4", "T_e", "T_w"]:
            Th_2[_v] = Th_3d_1[_v]

    ## update thermocline variable
    Th_2["h_mg"] = Th_3d_1["h"]
    Th_2["h_w_mg"] = Th_3d_1["h_w"]

    ## non-dimensionalize
    T_scale = Th_2["T_3"].std().values.item()
    h_scale = Th_2["h_w"].std().values.item()

    ## non-dim T variables
    for _v in ["T_3", "T_34", "T_4", "T_e", "T_m"] + list(Q):
        Th_2[_v] = Th_2[_v] / T_scale


    ## non-dim h variables
    for _v in ["h", "h_w", "h_w_ohc", "h_ohc"]:
        Th_2[_v] = Th_2[_v] / h_scale

    ## get windowed data (used to estimate change in parameters over time)
    Th_rolling = src.utils.get_windowed(Th_2, window_size=480, stride=120)
    # Th_rolling = src.utils.get_windowed(Th_3d, window_size=480, stride=120)

    ## drop last year because of NaNs
    Th_rolling = Th_rolling.sel(year=slice(None, 2080))

    ## compute grad
    Th_rolling["dTdx"] = Th_rolling["T_3"] - Th_rolling["T_4"]
    return (Th_rolling,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Remove linear dependence
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: power point results use ```T_var=T_m```
    """)
    return


@app.cell
def _(Th_rolling, src, tqdm, varnames):
    ## remove SST dependence
    for h_var in tqdm.tqdm(["h", "h_w", "h_ohc", "h_w_ohc", "h_mg", "h_w_mg"]):
        Th_rolling[f"{h_var}_hat"] = src.utils.remove_sst_dependence_v2(
            Th=Th_rolling,
            h_var=h_var,
            T_var=varnames[0],
            dims=["time", "member", "year"],
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### remove median
    """)
    return


@app.cell
def _(REMOVE_MEDIAN, Th_rolling, xr):
    ## remove median (optional)
    if REMOVE_MEDIAN:
        median = Th_rolling.groupby("time.month").median(["time", "member"])
        Th_rolling_1 = Th_rolling.groupby("time.month") - median

    else:
        mean = Th_rolling_1.groupby("time.month").mean(["time", "member"])
        Th_rolling_1 = Th_rolling_1.groupby("time.month") - mean

    ## add constant
    Th_rolling_1["ones"] = xr.ones_like(Th_rolling_1["T_3"])
    return (Th_rolling_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute time-varying RO parameters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Filenames:
    - "T_3-h_w_ohc-zero_median_all-members.nc": ```maskb=["h"], maskNT=["T2", "T3", "TH"], maskNH=["T2"]```
    - "T_3-h_w_ohc-zero_median.nc": same, but separate fits for each ensemble member
    - "T_3-h_w_ohc-zero_median_simple.nc": ```maskNT=["T2", "T3"]```
    """)
    return


@app.cell
def _(Th_rolling_1, get_fits_over_time_wrapper, np, src, varnames):
    ## parameters for fitting
    # MODEL = src.XRO.XRO(
    #     ncycle=12, ac_order=3, taus=np.zeros(4, dtype=int), is_forward=False
    # )
    MODEL = src.XRO.XRO(ncycle=12, ac_order=3, is_forward=False)

    ## nonlinear R
    fit_kwargs = dict(
        ac_mask_idx=None,
        # maskNT=["T2", "T3", "H2"],
        maskNT=["T2", "T3"],
        # maskNT=["T2", "T3", "H2"],
        maskNH=["T2"],
    )

    ## get fits
    fits = get_fits_over_time_wrapper(
        Th_rolling_1[varnames],
        model=MODEL,
        by_member=False,
        fname=None,
        # fname=f"{varnames[0]}-{varnames[1]}-zero_median_simple.nc",
        **fit_kwargs,
    )

    ## extract parameters
    params = src.utils.get_params(fits=fits, model=MODEL)

    ## get change from initial period
    delta_params = params - params.isel(year=0)

    ## compute error
    fits["error"] = fits["Yfit"] - fits["Y"]
    fits["error_amp"] = np.sqrt(fits["error"] ** 2)

    ## get corrected coords
    time_coord = Th_rolling_1.stack(s=["member", "time"]).s
    fits = fits.rename({"time": "s"}).assign_coords({"s": time_coord}).unstack("s")
    return MODEL, fits, params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Look at covariance between parameters
    """)
    return


@app.cell
def _(Th_rolling_1, np, plt, src, varnames):
    _fig, axs = plt.subplots(3, 4, figsize=(10, 7.5), layout="constrained")


    ## loop thru years
    for _yi, _y in enumerate([1870, 2010, 2080]):

        ## get data for given year
        # Th_y = Th_rolling.sel(year=y).resample({"time":"QS-JAN"}).mean()
        Th_y = Th_rolling_1.sel(year=_y)

        ## loop thru months
        for mi, m_idx in enumerate(np.arange(1, 13, 3)):

            ## subset for month
            is_m = Th_y.time.dt.month == m_idx
            Th_ym = Th_y.isel(time=is_m)

            ## plot data
            axs[_yi, mi].scatter(Th_ym[varnames[0]], Th_ym[varnames[1]], s=2)

            ## formatting
            _ax_kwargs = dict(ls="--", lw=1, c="gray")
            axs[_yi, mi].axvline(0, **_ax_kwargs)
            axs[_yi, mi].axhline(0, **_ax_kwargs)

    src.utils.set_lims(axs)
    for _ax in axs[:-1].flatten():
        _ax.set_xticks([])
    for _ax in axs[:, 1:].flatten():
        _ax.set_yticks([])
    plt.gca()
    return (mi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### validate
    """)
    return


@app.cell
def _(Th_rolling_1, varnames):
    # simulation specs
    simulation_kwargs = dict(
        nyear=40,
        ncopy=500,
        seed=1000,
        X0_ds=Th_rolling_1[varnames].isel(year=0, member=0, time=0),
        # X0_ds = None,
        noise_type="white",
        use_noise_cov=True,
        is_xi_stdac=True,
        # noise_scale = .6,
        # noise_scale=1.25,  ## use for h_hat, is_forward=True
        # noise_scale=1.35, ## use for h_w
        noise_scale=1.45,  ## use for h_hat
    )
    return (simulation_kwargs,)


@app.cell
def _(
    MODEL,
    Th_rolling_1,
    fits,
    get_sims_over_time,
    simulation_kwargs,
    varnames,
):
    ## do simulations
    sims = get_sims_over_time(model=MODEL, params=fits, **simulation_kwargs)

    ## resample to DJF
    get_djf = lambda x: x.resample({"time": "QS-DEC"}).mean().isel(time=slice(4, -4, 4))

    ## resample to seasonal
    sims_djf = get_djf(sims)
    Th_djf = get_djf(Th_rolling_1[varnames])
    return Th_djf, sims, sims_djf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute stats
    """)
    return


@app.cell(hide_code=True)
def _(xr):
    def get_std(x):
        return x.std(["time", "member"])


    def get_quantiles(x):
        return x.quantile(dim=["time", "member"], q=[0.05, 0.1, 0.5, 0.9, 0.95])


    def get_stats(x):
        """compute relevant stats for given dataset"""

        ## empty dataset to hold results
        stats = xr.Dataset()

        ## helper func to convert to xr.DataArray
        to_da = lambda x: x.to_dataarray(dim="v")

        ## compute
        stats["sigma"] = to_da(get_std(x))
        stats["q"] = to_da(get_quantiles(x))

        return stats

    return (get_stats,)


@app.cell
def _(Th_djf, get_stats, sims_djf):
    stats_ro = get_stats(sims_djf)
    stats_gt = get_stats(Th_djf)
    return stats_gt, stats_ro


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Quantiles
    """)
    return


@app.cell
def _(plt, save, src, stats_gt, stats_ro, varnames):
    for varname in varnames:

        _fig, axs_1 = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")
        for _ax, _stats_ in zip(axs_1, [stats_gt, stats_ro]):
            _q = _stats_["q"].sel(v=varname)

            _ax.plot(_stats_.year, _q.sel(quantile=0.95), c="r", label="El Niño")
            _ax.plot(_stats_.year, -_q.sel(quantile=0.05), c="b", label="La Niña")
            _ax.plot(
                _stats_.year,
                0.5 * (_q.sel(quantile=0.95) - _q.sel(quantile=0.05)),
                label="Average",
                c="k",
            )

            _ax_kwargs = dict(ls="--", c="gray", lw=0.8)
            src.utils.add_vticks(axs_1, xticks=[1870, 2010, 2080], xlines=[2010])
            # ax.set_ylim([0, None])

        src.utils.set_lims(axs_1)
        axs_1[1].set_yticks([])
        axs_1[1].legend()
        axs_1[0].set_title("CESM2")
        axs_1[1].set_title("RO")
        axs_1[0].set_ylabel(r"K")

        save(_fig, fname=f"RO-valid_{varname}", do_save=False)
        plt.show()
    return


@app.cell
def _(Th_rolling_1, plt, src):
    Th_ = Th_rolling_1.resample({"time": "QS-JAN"}).mean().isel(time=slice(4, -4, 4))

    _fig, axs_2 = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    for _ax, _yr in zip(axs_2, [2010, 2080]):
        axs_2[0].scatter(
            Th_["T_e"].sel(year=_yr),
            Th_["T_w"].sel(year=_yr),
            s=3,
        )
        axs_2[1].scatter(
            Th_["T_e"].sel(year=_yr),
            Th_["T_e"].sel(year=_yr) - Th_["T_w"].sel(year=_yr),
            s=3,
        )

    src.utils.set_lims(axs_2)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### PDFs
    """)
    return


@app.cell
def _(np, src, varnames):
    def get_pdf(x, year, varname=varnames[0], amp=4):
        """compute pdf for given data"""

        ## get edges
        edges = np.linspace(-amp, amp, 20)

        ## reshape data
        x_ = x[varname].sel(year=year).stack(s=["time", "member"]).values

        ## compute
        pdf, _ = src.utils.get_empirical_pdf(x_, edges=edges)

        return pdf, edges

    return (get_pdf,)


@app.cell
def _(Th_djf, get_pdf, plt, sims_djf, src, varnames):
    for PLOT_VAR, a in zip(varnames, [4, 2.5]):

        _fig, axs_3 = plt.subplots(1, 2, figsize=(6, 2.5), layout="constrained")

        for _ax, x, _title in zip(axs_3, [Th_djf, sims_djf], ["CESM2", "RO"]):

            for _y, _kwargs in zip(
                [1870, 2010, 2080],
                [dict(color="gray", fill=True, alpha=0.2), dict(lw=2), dict(lw=2)],
            ):

                ## compute pdf for given year
                pdf_y, edges = get_pdf(x, year=_y, varname=PLOT_VAR, amp=a)

                ## plot
                _ax.stairs(pdf_y, edges, label=_y, **_kwargs)

            ## label/format
            _ax.set_title(_title)
            # ax.set_xlim([-3.3, 3.3])
            # ax.set_xticks([-3, 0, 3])
            _ax_kwargs = dict(ls="--", c="k", lw=0.8)
            _ax.axvline(0, **_ax_kwargs)
            _ax.axvline(-1.5, **_ax_kwargs)

        ## formatting
        src.utils.set_lims(axs_3)
        axs_3[1].set_yticks([])
        axs_3[1].legend(prop=dict(size=10))
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Variance/skewness over time
    """)
    return


@app.cell
def _(Th_djf, sims_djf):
    get_sigma = lambda x: x.std("time").quantile(dim="member", q=[0.1, 0.5, 0.9])
    sigma_mod = get_sigma(Th_djf)
    sigma_sim = get_sigma(sims_djf)
    return sigma_mod, sigma_sim


@app.cell
def _(Th_djf, scipy, sims_djf, xr):
    ## compute stats
    get_stat = lambda x, fn: xr.apply_ufunc(
        fn, x, input_core_dims=[["time"]], kwargs={"axis": -1}
    )
    get_stat_wrapper = lambda x, fn: get_stat(x, fn).quantile(
        dim="member", q=[0.1, 0.5, 0.9]
    )

    # median_ot = get_stat(Th_djf, np.median)
    skew_mod = get_stat_wrapper(Th_djf, scipy.stats.skew)
    skew_sim = get_stat_wrapper(sims_djf, scipy.stats.skew)
    # std_ot = get_stat(Th_djf, np.std)
    # m3_ot = skew_ot * (std_ot**3)
    return skew_mod, skew_sim


@app.cell
def _(plt, save, sigma_mod, sigma_sim, skew_mod, skew_sim, sns, src, varnames):
    ## specify colors
    _colors = sns.color_palette()

    for PLOT_VAR_1 in varnames:

        ## plot setup
        _fig, axs_4 = plt.subplots(1, 2, figsize=(6.5, 3), layout="constrained")

        ## plot data
        for _ax, stats in zip(axs_4, [[sigma_mod, sigma_sim], [skew_mod, skew_sim]]):
            for stat, _c, ls, _label in zip(stats, _colors, ["-", "--"], ["CESM2", "RO"]):
                for _q, lw in zip([0.5, 0.1, 0.9], [2, 1, 1]):
                    label_ = _label if (_q == 0.5) else None
                    _ax.plot(
                        stat.year,
                        stat[PLOT_VAR_1].sel(quantile=_q),
                        c=_c,
                        lw=lw,
                        ls=ls,
                        label=label_,
                    )

        src.utils.add_vticks(axs_4, xticks=[1870, 2010, 2080], xlines=[2010])

        axs_4[0].legend()
        axs_4[1].axhline(0, ls="--", c="k", lw=0.8)
        axs_4[0].set_ylabel("K")
        axs_4[1].set_ylabel("non-dim")
        axs_4[0].set_title(r"$\sigma(T)$ over time (DJF)")
        axs_4[1].set_title(r"skew$(T)$ over time (DJF)")

        save(_fig, fname=f"RO-valid2_{PLOT_VAR_1}", do_save=False)
        plt.show()
    return


@app.cell
def _(Th_rolling_1, np, plt, sims, src, varnames):
    PLOT_VAR_2 = varnames[0]

    print(f"CESM2:")
    _fig, axs_5 = plt.subplots(2, 1, figsize=(8, 3), layout="constrained")
    axs_5[0].plot(np.arange(0, 40, 1 / 12), Th_rolling_1[PLOT_VAR_2].isel(member=9, year=10))
    axs_5[1].plot(np.arange(0, 40, 1 / 12), Th_rolling_1[PLOT_VAR_2].isel(member=10, year=-1))
    src.utils.set_lims(axs_5)
    for _ax in axs_5:
        _ax.axhline(0, ls="--", c="k", lw=0.8)
    plt.show()

    print(f"\nSimulated:")
    _fig, axs_5 = plt.subplots(2, 1, figsize=(8, 3), layout="constrained")
    axs_5[0].plot(np.arange(0, 40, 1 / 12), sims[PLOT_VAR_2].isel(member=9, year=10))
    axs_5[1].plot(np.arange(0, 40, 1 / 12), sims[PLOT_VAR_2].isel(member=10, year=-1))
    src.utils.set_lims(axs_5)
    for _ax in axs_5:
        _ax.axhline(0, ls="--", c="k", lw=0.8)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## look at params
    """)
    return


@app.cell
def _(fits, params, plt, src):
    # delta =

    _xi = fits["xi_stdac"].isel(ranky=0)
    _R = fits["Lac"].isel(ranky=0, rankx=0) - fits["Lac"].isel(ranky=1, rankx=1)
    F1 = fits["Lac"].isel(ranky=0, rankx=1)
    F2 = -fits["Lac"].isel(ranky=1, rankx=0)
    # beta= fits["NROT_Lac"].isel(nro_form=-1)
    beta = fits["NROH_Lac"].isel(nro_form=0)
    dxi = _xi / _xi.isel(year=0) - 1
    dR = _R / _R.isel(year=0) - 1

    _fig, axs_6 = plt.subplots(1, 4, figsize=(10, 2.5))
    for _ax, _p in zip(axs_6, [_R, _xi, F1, beta]):

        for _y in [1870, 1910, 1950, 1990]:

            _ax.plot(params.cycle, _p.sel(year=_y))

    axs_6[1].set_ylim([0, None])
    axs_6[2].set_ylim([0, None])
    src.utils.add_vticks(axs_6, xlines=[4, 6], xticks=[4, 6])
    plt.gca()
    return (F1,)


@app.cell
def _(fits, plt, src):
    _BJ = fits["Lac"].isel(rankx=0, ranky=0) + fits["Lac"].isel(rankx=1, ranky=1)

    _fig, axs_7 = plt.subplots(10, 1, figsize=(3, 11), layout="constrained")
    for _y in [1880, 1990]:  # , 2030, 2080]:
        axs_7[0].plot(fits.cycle, fits["Lac"].isel(rankx=0, ranky=0).sel(year=_y), label=_y)
        axs_7[1].plot(fits.cycle, fits["NROT_Lac"].isel(nro_form=0).sel(year=_y), label=_y)
        axs_7[2].plot(fits.cycle, fits["NROT_Lac"].isel(nro_form=2).sel(year=_y), label=_y)
        axs_7[3].plot(fits.cycle, fits["Lac"].isel(ranky=0, rankx=1).sel(year=_y), label=_y)
        axs_7[4].plot(fits.cycle, fits["NROT_Lac"].isel(nro_form=-1).sel(year=_y), label=_y)
        axs_7[5].plot(fits.cycle, fits["xi_stdac"].isel(ranky=0).sel(year=_y), label=_y)
        axs_7[6].plot(fits.cycle, -fits["Lac"].isel(ranky=1, rankx=0).sel(year=_y), label=_y)
        axs_7[7].plot(fits.cycle, -fits["NROH_Lac"].isel(nro_form=0).sel(year=_y), label=_y)
        axs_7[8].plot(fits.cycle, -fits["Lac"].isel(rankx=1, ranky=1).sel(year=_y), label=_y)
        # axs[9].plot(fits.cycle, -fits["NLb_Lac"].isel(ranky=1).sel(year=y), label=y)
        axs_7[9].plot(fits.cycle, _BJ.sel(year=_y), label=_y)


    src.utils.add_vticks(axs_7, xlines=fits.cycle.values[[3, 5]], xticks=[])
    _kwargs = dict(x=1.0, y=0.5)
    axs_7[0].text(s=r"$R$", transform=axs_7[0].transAxes, **_kwargs)
    axs_7[1].text(s=r"$\frac{dT}{dt}\propto T^2$", transform=axs_7[1].transAxes, **_kwargs)
    axs_7[2].text(s=r"$\frac{dT}{dt}\propto T^3$", transform=axs_7[2].transAxes, **_kwargs)
    axs_7[3].text(s=r"$F_1$", transform=axs_7[3].transAxes, **_kwargs)
    axs_7[4].text(s=r"$\frac{dT}{dt}\propto h^2$", transform=axs_7[4].transAxes, **_kwargs)
    axs_7[5].text(s=r"$\xi_T$", transform=axs_7[5].transAxes, **_kwargs)
    axs_7[6].text(s=r"$F_2$", transform=axs_7[6].transAxes, **_kwargs)
    axs_7[7].text(s=r"$\frac{dh}{dt}\propto -T^2$", transform=axs_7[7].transAxes, **_kwargs)
    axs_7[8].text(s=r"$\varepsilon$", transform=axs_7[8].transAxes, **_kwargs)
    # axs[9].text(s=r"$\frac{dh}{dt}\propto -h^2$", transform=axs[9].transAxes, **kwargs)
    axs_7[9].text(s=r"$R-\varepsilon$", transform=axs_7[9].transAxes, **_kwargs)
    for _ax in axs_7:
        _ax.axhline(0, ls="--", c="gray", lw=0.8)
    # ax.legend()

    axs_7[-1].set_xticks(fits.cycle.values[[3, 5]], labels=[4, 6])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hovmoller plot of $R$ and $\varepsilon$
    """)
    return


@app.cell
def _(fits):
    R_ = fits["Lac"].isel(rankx=0, ranky=0)
    F1_ = fits["Lac"].isel(rankx=1, ranky=0)
    R2 = fits["NROT_Lac"].isel(nro_form=0)
    R3 = fits["NROT_Lac"].isel(nro_form=2)
    eps_ = fits["Lac"].isel(rankx=1, ranky=1)
    error_sigma = fits["xi_stdac"].isel(ranky=0)
    # error_bar = fits["error"].groupby("time.month").mean(["time","member"]).isel(ranky=0)
    # error_sigma =  fits["error"].groupby("time.month").std(["time","member"]).isel(ranky=0).rename({"month":"cycle"}).assign_coords({"cycle":R3.cycle})
    # error_bar = (error_bar/error_sigma).rename({"month":"cycle"}).assign_coords({"cycle":R3.cycle})
    return R2, R3, R_, eps_, error_sigma


@app.cell
def _(F1, R2, R3, R_, eps_, error_sigma, plt, src):
    _delta = lambda x: x - x.isel(year=0)
    _fig, axs_8 = plt.subplots(1, 6, figsize=(12, 2), layout="constrained")

    for _ax, _p, _l in zip(
        axs_8,
        [R_, R_ + eps_, F1, R3, R2, error_sigma],
        [
            r"$R$",
            r"$R-\varepsilon$",
            r"$F_1$",
            r"$\frac{dT}{dt}\propto T^3$",
            r"$\frac{dT}{dt}\propto T^2$",
            "error",
        ],
    ):

        _ax.contourf(
            _p.cycle,
            _p.year,
            _delta(_p),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(0.75, 0.075),
            extend="both",
        )
        _ax.axhline(1990, ls="--", c="gray", lw=1)
        _ax.set_yticks([])
        _ax.set_title(_l)

    src.utils.add_vticks(
        axs_8,
        xticks=[],
        xlines=[_p.cycle[i] for i in [0, 3, 5]],
    )


    axs_8[0].set_yticks([1870, 1990, 2080])
    plt.gca()
    return


@app.cell
def _(Th_rolling_1, fits, get_ddt, plt, varnames):
    ## check fits make sense
    idx = dict(year=2, member=0, time=slice(None, 120))

    ## make plot
    _fig, _ax = plt.subplots(figsize=(3, 2.5))
    _ax.plot(fits["Y"].isel(ranky=0, **idx))
    _ax.plot(get_ddt(Th_rolling_1[varnames[0]].isel(**idx)), ls="--")
    plt.gca()
    return


@app.cell
def _(fits):
    ## resample to AMJ
    fit_amj = fits.resample({"time": "QS-JAN"}).mean().isel(time=slice(1, -4, 4))
    fit_a = fits.isel(time=slice(3, -12, 12))
    fit_amj0 = fit_amj.sel(year=1880)
    fit_amj1 = fit_amj.sel(year=1990)
    fit_a0 = fit_a.sel(year=1880)
    fit_a1 = fit_a.sel(year=1990)
    return fit_a0, fit_a1, fit_amj, fit_amj0, fit_amj1


@app.cell
def _(fit_a0, fit_a1, fit_amj0, fit_amj1, plt, src, xr):
    for fit_amj_, fit_a_ in zip([fit_amj0, fit_amj1], [fit_a0, fit_a1]):

        _fig, axs_9 = plt.subplots(2, 2, figsize=(5, 5), layout="constrained")

        ## loop thru (actual, error)
        for _j, _v in enumerate(["Y", "error"]):

            for _xi, _ax in enumerate(axs_9[_j]):

                x0 = fit_a_["X"].isel(rankx=_xi)
                # x1 = fit_a_[v].isel(ranky=0).transpose(*x0.dims)
                # x0 = fit_amj_["X"].isel(rankx=xi)
                x1 = fit_amj_[_v].isel(ranky=0).transpose(*x0.dims)

                _ax.scatter(x0, x1, s=3, alpha=0.5)
                corr = xr.corr(x0, x1).values.item()
                _ax.set_title(f"Corr: {corr:.2f}")

                ## format
                _ax_kwargs = dict(ls="--", c="k", lw=0.8)
                _ax.axhline(0, **_ax_kwargs)
                _ax.axvline(0, **_ax_kwargs)

            src.utils.set_lims(axs_9[_j])
            axs_9[_j, 1].set_yticks([])

        for _ax in axs_9[0]:
            _ax.set_xticks([])

        plt.show()
    return


@app.cell
def _(fit_amj, np, plt, src):
    _fig, axs_10 = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")

    for _ax, year in zip(axs_10, [1880, 1990]):
        _ax.scatter(
            fit_amj["Y"].isel(ranky=0).sel(year=year),
            fit_amj["Yfit"].isel(ranky=0).sel(year=year),
            s=3,
            alpha=0.5,
        )

        ## format
        _ax_kwargs = dict(ls="--", c="k", lw=0.8)
        _ax.axhline(0, **_ax_kwargs)
        _ax.axvline(0, **_ax_kwargs)
        zz = np.linspace(-10, 5)
        _ax.plot(zz, zz, ls="-", lw=0.8, c="k")

    axs_10[1].set_yticks([])
    src.utils.set_lims(axs_10)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extract params
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get coefficients
    """)
    return


@app.cell
def _(fits, xr):
    ## get coefficients
    coefs_nl = fits["NROT_Lac"].sel(nro_form=["T2", "T3"])
    _R = fits["Lac"].isel(ranky=0, rankx=0).expand_dims(nro_form=["T"])
    eps = -fits["Lac"].isel(ranky=1, rankx=1)
    coefs = xr.concat([_R, coefs_nl], dim="nro_form").rename({"nro_form": "form"})
    return coefs, eps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute feature matrix for reconstructing nonlinear $R$
    """)
    return


@app.cell
def _(Th_rolling_1, coefs, fits, np, xr):
    ## how should we scale the feature matrix?
    ## One of {"constant", "by_month", "sliding_by_month"}
    SCALE_TYPE = "by_month"

    ## get scale
    sigma_T = Th_rolling_1["T_3"].groupby("time.month").std(["time", "member"])

    ## average as needed
    if SCALE_TYPE == "constant":
        sigma_T = sigma_T.mean(["year", "month"]) * xr.ones_like(sigma_T)

    elif SCALE_TYPE == "by_month":
        sigma_T = sigma_T.mean("year") * xr.ones_like(sigma_T)

    ## update coordinate names to match
    sigma_T = sigma_T.rename({"month": "cycle"}).assign_coords({"cycle": fits["cycle"]})

    ## scale feature mat by sigma_T
    z = np.arange(-3, 3.1, 0.1)
    z = xr.DataArray(z, coords=dict(sigma=z))
    Z = sigma_T * z.expand_dims({"year": fits.year, "cycle": sigma_T.cycle})

    ## expand to get coefficients
    Z = xr.concat([Z**i for i in range(3)], dim=coefs.form)

    ## check everything makes sense
    print(np.allclose(Z.sel(form="T2").sel(sigma=1, method="nearest"), sigma_T))
    return (Z,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Do the reconstruction
    """)
    return


@app.cell
def _(Z, coefs, eps):
    ## reconstruct nonlinear R
    R_nl = (coefs * Z).sum("form")
    R_nl_ann = R_nl.mean("cycle")
    eps_ann = eps.mean("cycle")
    return R_nl, R_nl_ann


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute time-varying $R_o$ and $\alpha$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### $\alpha$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Do fitting (or load pre-computed)
    """)
    return


@app.cell
def _(np, pd, src, xr):
    def regress_cov_lagged(X, x_vars, y_var):
        """do nonlinear regression"""

        ## prep data
        prep = lambda x: x.stack(s=["time", "member"]).transpose("year", ..., "s")

        ## get vars
        x_vars_plus = [f"{v}_plus" for v in x_vars]
        x_vars_minus = [f"{v}_minus" for v in x_vars]
        Y_ = prep(X[f"{y_var}_plus"])

        X_plus_ = prep(X[x_vars_plus].to_dataarray(dim="param"))
        X_minus_ = prep(X[x_vars_minus].to_dataarray(dim="param"))
        Y_plus_ = prep(X[f"{y_var}_plus"])

        ## empty array to hold results
        m = xr.DataArray(
            coords={"year": X.year, "param": x_vars},
            dims=["year", "param"],
        )

        ## covariance
        Cyx = np.einsum("yi,yji->yj", Y_plus_.values, X_minus_.values)
        Cxx = np.einsum("yji,yki->yjk", X_plus_.values, X_minus_.values)
        Cxx_inv = np.linalg.inv(Cxx)
        m.values = np.einsum("yj,yji->yi", Cyx, Cxx_inv)

        return m


    def regress_cov_lagged_wrapper(X, x_vars, y_var, by_month=True):
        """helper func to get covariance"""

        ## Get xplus, xminus
        X_minus = X.isel(time=slice(None, -1))
        X_plus = X.isel(time=slice(1, None))
        X_plus = X_plus.assign_coords({"time": X_minus.time})
        X_minus = X_minus.rename({v: f"{v}_minus" for v in list(X_minus)})
        X_plus = X_plus.rename({v: f"{v}_plus" for v in list(X_plus)})
        X_ = xr.merge([X_plus, X_minus])

        ## now do regression
        kwargs = dict(x_vars=x_vars, y_var=y_var)
        if by_month:
            return X_.groupby("time.month").map(regress_cov_lagged, **kwargs)

        else:
            return regress_cov_lagged(X_, **kwargs)


    def regress_pinv(X, x_vars, y_var):
        """do nonlinear regression"""

        ## prep data
        # prep = lambda x: x.transpose("member", ...)
        prep = lambda x: x.stack(s=["time", "member"]).transpose("year", ..., "s")

        X_ = prep(X[x_vars].to_dataarray(dim="v"))
        Y_ = prep(X[y_var])

        ## empty array to hold results
        m = xr.DataArray(
            coords={"year": X.year, "v": x_vars},
            dims=["year", "v"],
        )

        ## do regression
        X_pinv = np.linalg.pinv(X_.values)
        m.values = np.einsum("yi,yij->yj", Y_.values, X_pinv)

        return m.rename({"v": "param"})


    def regress_pinv_wrapper(X, x_vars, y_var):
        return X.groupby("time.month").map(regress_pinv, x_vars=x_vars, y_var=y_var)


    def regress_smooth(X, x_vars, y_var):
        """fit with harmonics"""

        ## get coefficients
        m = src.utils.regress_harm_wrapper(
            data=X,
            y_vars=[y_var],
            x_vars=x_vars,
            max_order=3,
            dims=["time", "member"],
        )

        ## convert from dataset to dataarray
        m = m.to_dataarray().squeeze(drop=True)

        return m.rename({"ell": "param"})


    def get_alpha(
        X, x_vars, regress_fn, T_REGION="3", flux_types=["nhf", "fsns", "lhflx", "flns"]
    ):
        """compute alpha over time"""

        ## empty array to hold results
        alpha = []

        ## loop thru flux types
        for ftype in flux_types:

            kwargs = dict(x_vars=["T1", "T2", "T3"], y_var=f"{ftype}_{T_REGION}")
            alpha.append(regress_fn(X, **kwargs))

        ## conver to xr
        alpha = xr.concat(alpha, dim=pd.Index(flux_types, name="flux_type"))

        ## get points for evaluating alpha
        sigma = np.arange(-3, 3.1, 0.1)
        z = xr.DataArray(sigma, coords=dict(sigma=sigma))
        zz = xr.concat([z**i for i in range(1, 4)], dim=alpha.param)

        ## evaluate
        psi_eval = (zz * alpha).sum("param")

        ## merge
        alpha = xr.merge(
            [
                -alpha.rename("params"),
                psi_eval.rename("psi_eval"),
                -psi_eval.differentiate("sigma").rename("eval"),
            ]
        )

        ## reconstruct net heat flux from fsns and lhflx
        nhf_recon = (
            alpha.sel(flux_type="fsns")
            + alpha.sel(flux_type="lhflx")
            + alpha.sel(flux_type="flns")
        )
        nhf_recon = nhf_recon.expand_dims({"flux_type": ["nhf_recon"]})
        alpha = xr.concat([alpha, nhf_recon], dim="flux_type")

        return alpha

    return get_alpha, regress_cov_lagged_wrapper, regress_pinv_wrapper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To-do: clean this up
    """)
    return


@app.cell
def _(
    Th_rolling_1,
    get_alpha,
    regress_cov_lagged_wrapper,
    regress_pinv_wrapper,
    xr,
):
    ## get flux types
    flux_types = ["nhf", "fsns", "lhflx", "flns"]

    ## merge T and Q data
    _T_data = xr.merge([(Th_rolling_1["T_3"] ** i).rename(f"T{i}") for i in range(4)])
    flux_data = Th_rolling_1[[f"{ftype}_3" for ftype in flux_types]]
    X = xr.merge([_T_data, flux_data])

    ## regression with lagged covariance matrix
    alpha_lagged = get_alpha(
        X=X,
        x_vars=["T1", "T2", "T3"],
        regress_fn=regress_cov_lagged_wrapper,
    )


    ## regression with no lag in cov. matrices
    alpha0 = get_alpha(
        X=X,
        x_vars=["T1", "T2", "T3"],
        regress_fn=regress_pinv_wrapper,
    )

    ## regression using harmonics
    # alpha_smooth = get_alpha(
    #     X=X,
    #     x_vars=["T1", "T2", "T3"],
    #     regress_fn=regress_smooth,
    # )
    return (alpha0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot results
    """)
    return


@app.cell
def _(alpha0, fits):
    ## specify which alpha to use
    ALPHA = alpha0

    ## update month index
    ALPHA = ALPHA.rename({"month": "cycle"}).assign_coords({"cycle": fits.cycle})
    return (ALPHA,)


@app.cell
def _(ALPHA, T_REGION, Th_rolling_1, mi, plt, save, src, varnames):
    ## specify year indices
    yis = [0, 12, -1]

    _fig, _axs_ = plt.subplots(4, 3, figsize=(8, 10), layout="constrained")

    for ji, _yi in enumerate(yis):

        ## get axes for plotting
        axs_11 = _axs_[:, ji]
        axs_11[0].set_title(Th_rolling_1.year.values[_yi])

        ## indx for scatter
        idx_sc = dict(year=_yi, time=slice(mi, None, 12))

        ## get T variable
        _T_data = Th_rolling_1[varnames[0]].isel(idx_sc)

        for _ax, flux_type in zip(axs_11, ALPHA.flux_type.values[:-1]):

            ## plot data
            _ax.scatter(
                _T_data,
                Th_rolling_1[f"{flux_type}_{T_REGION}"].isel(idx_sc),
                s=5,
            )

            ## plot fit
            _ax.plot(
                ALPHA.sigma,
                ALPHA["psi_eval"].sel(flux_type=flux_type).isel(year=_yi, cycle=mi),
                c="magenta",
                ls="-",
            )

            ## plot fit for first period
            _ax.plot(
                ALPHA.sigma,
                ALPHA["psi_eval"].sel(flux_type=flux_type).isel(year=0, cycle=mi),
                c="k",
                ls="--",
            )

        for _ax in axs_11:
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.axhline(0, ls="--", c="gray", lw=1)
            _ax.axvline(0, ls="--", c="gray", lw=1)

    ## label/format axes
    src.utils.set_lims(_axs_)
    for _ax in _axs_[-1]:
        _ax.set_xticks([-3, 0, 3])
        _ax.set_xlabel(r"$T$ (K)")
    for _ax in _axs_[:, 0]:
        _ax.set_yticks([-5, 0, 5])
        _ax.set_ylabel(r"$Q$ (K yr$^{-1}$)")
    for _ax, _label in zip(
        _axs_[:, -1],
        ["Net", "Shortwave", "Latent", "Longwave"],
    ):
        _ax.text(s=_label, x=1.1, y=0.5, transform=_ax.transAxes)

    save(_fig, fname="Q-comparison", do_save=False)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Scratch (compute sum...)
    """)
    return


@app.cell
def _():
    # ## plot sum of latent and shortwave
    # nhf_recon = (
    #     Th_rolling[f"fsns_{T_REGION}"]
    #     + Th_rolling[f"lhflx_{T_REGION}"]
    #     + Th_rolling[f"flns_{T_REGION}"]
    # )
    # axs[-1].scatter(T_data, nhf_recon.isel(idx_sc), s=5)

    # # ## plot reconstruction
    # # alpha_recon = alpha.sel(flux_type="fsns") + alpha.sel(flux_type="lhflx")
    # axs[-1].plot(
    #     ALPHA.sigma,
    #     ALPHA["psi_eval"].isel(year=yi, cycle=mi).sel(flux_type="nhf_recon"),
    #     c="k",
    # )
    # axs[-1].set_title("fsns + lhflx")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute $R_o$
    """)
    return


@app.cell(hide_code=True)
def _(xr):
    def get_fits_alpha(fits, alpha):
        """put alpha in fits format"""

        ## empty fits dataset
        fits_alpha = xr.zeros_like(fits)
        fits_alpha["var_names"] = fits["var_names"]

        ## helper func to get params
        get_p = lambda x: x["params"].transpose("year", "cycle").values

        ## update linear
        alpha_Lac = fits_alpha["Lac"].transpose("year", "cycle", "ranky", "rankx")
        alpha_Lac.values[..., 0, 0] = get_p(alpha.sel(param="T1"))

        ## update nonlinear
        alpha_non = fits_alpha["NROT_Lac"].transpose("year", "cycle", "nro_form")
        alpha_non.values[..., 0] = get_p(alpha.sel(param="T2"))
        alpha_non.values[..., 2] = get_p(alpha.sel(param="T3"))

        ## update fits
        fits_alpha["Lac"] = alpha_Lac.transpose(*fits_alpha["Lac"].dims)
        fits_alpha["NROT_Lac"] = alpha_non.transpose(*fits_alpha["NROT_Lac"].dims)

        return fits_alpha


    def op_fits(fits1, fits0, op):
        """Handle operation on two fits objects, handling non-numeric data"""

        ## get list of valid variables
        valid_vars = [x for x in list(fits0) if (x != "var_names")]

        ## subtract
        res = op(fits1[valid_vars], fits0[valid_vars])

        ## update other variables
        res["var_names"] = fits0["var_names"]

        return res


    def sum_fits(fits1, fits0):
        """Get difference between fits (fits1-fits0).
        Handles non-numeric data"""

        return op_fits(fits1, fits0, op=lambda x1, x0: x1 + x0)


    def diff_fits(fits1, fits0):
        """Get difference between fits (fits1-fits0).
        Handles non-numeric data"""

        return op_fits(fits1, fits0, op=lambda x1, x0: x1 - x0)

    return diff_fits, get_fits_alpha, sum_fits


@app.cell
def _(ALPHA, R_nl, fits, get_fits_alpha, sum_fits):
    ## evaluated
    Ro = R_nl + ALPHA["eval"].sel(flux_type="nhf")

    ## get alpha and Ro in fits format
    fits_alpha = get_fits_alpha(fits=fits, alpha=ALPHA.sel(flux_type="nhf"))
    fits_Ro = sum_fits(fits1=fits, fits0=fits_alpha)

    ## same, but for shortwave and lhflx
    fits_alpha_sw = get_fits_alpha(fits=fits, alpha=ALPHA.sel(flux_type="fsns"))
    fits_minus_sw = sum_fits(fits1=fits, fits0=fits_alpha_sw)

    fits_alpha_lhf = get_fits_alpha(fits=fits, alpha=ALPHA.sel(flux_type="lhflx"))
    fits_minus_lhf = sum_fits(fits1=fits, fits0=fits_alpha_lhf)

    fits_alpha_lw = get_fits_alpha(fits=fits, alpha=ALPHA.sel(flux_type="flns"))
    fits_minus_lw = sum_fits(fits1=fits, fits0=fits_alpha_lw)
    return (
        Ro,
        fits_Ro,
        fits_alpha,
        fits_alpha_lhf,
        fits_alpha_lw,
        fits_alpha_sw,
        fits_minus_lhf,
        fits_minus_lw,
        fits_minus_sw,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check it works
    """)
    return


@app.cell
def _(ALPHA, fits, fits_Ro, fits_alpha, np):
    print(
        np.allclose(
            ALPHA["params"].sel(param="T1", flux_type="nhf").transpose("year", ...),
            (fits_Ro["Lac"] - fits["Lac"]).isel(rankx=0, ranky=0),
        )
    )
    print(
        np.allclose(
            ALPHA["params"].sel(param="T2", flux_type="nhf").transpose("year", ...),
            (fits_Ro["NROT_Lac"] - fits["NROT_Lac"]).isel(nro_form=0),
        )
    )
    print(
        np.allclose(
            ALPHA["params"].sel(param="T3", flux_type="nhf").transpose("year", ...),
            (fits_Ro["NROT_Lac"] - fits["NROT_Lac"]).isel(nro_form=2),
        )
    )

    print(
        np.allclose(
            fits_alpha[["Lac", "NROT_Lac"]].to_dataarray(),
            (fits_Ro[["Lac", "NROT_Lac"]] - fits[["Lac", "NROT_Lac"]]).to_dataarray(),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hovmoller plot of $R_o$ and $\varepsilon$
    """)
    return


@app.cell
def _(fits_Ro, fits_alpha):
    Ro_ = fits_Ro["Lac"].isel(rankx=0, ranky=0)
    Ra_ = fits_alpha["Lac"].isel(rankx=0, ranky=0)
    # F1_ = fits_Ro["Lac"].isel(rankx=1,ranky=0)
    Ro2 = fits_Ro["NROT_Lac"].isel(nro_form=0)
    Ro3 = fits_Ro["NROT_Lac"].isel(nro_form=2)
    return Ra_, Ro2, Ro3, Ro_


@app.cell
def _(R_, Ra_, Ro2, Ro3, Ro_, plt, src):
    _delta = lambda x: x - x.isel(year=0)
    _fig, axs_12 = plt.subplots(1, 5, figsize=(10, 2), layout="constrained")

    for _ax, _p, _l in zip(
        axs_12,
        [R_, Ro_, -Ra_, Ro3, Ro2],
        [
            r"$R$",
            r"$R_o$",
            r"$-\alpha$",
            r"$\frac{dT}{dt}\propto T^3$",
            r"$\frac{dT}{dt}\propto T^2$",
        ],
    ):

        _ax.contourf(
            _p.cycle,
            _p.year,
            _delta(_p),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(0.75, 0.075),
            extend="both",
        )
        _ax.axhline(1990, ls="--", c="gray", lw=1)
        _ax.set_yticks([])
        _ax.set_title(_l)

    src.utils.add_vticks(
        axs_12,
        xticks=[],
        xlines=[_p.cycle[i] for i in [0, 3, 5]],
    )


    axs_12[0].set_yticks([1870, 1990, 2080])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cleaned up plot
    """)
    return


@app.cell
def _(R_, Ra_, Ro_, error_sigma, fits_alpha, plt, save, src):
    _delta = lambda x: x - x.isel(year=0)
    _fig, axs_13 = plt.subplots(1, 4, figsize=(10, 2.5), layout="constrained")

    Ra_3 = fits_alpha["NROT_Lac"].isel(nro_form=0)

    for _ax, _p, _l, _amp in zip(
        axs_13,
        [R_, Ro_, -Ra_, error_sigma],
        [
            r"$R$",
            r"$R_o$",
            r"$-\alpha$",
            r"Noise amplitude",
        ],
        [1, 1, 1, 0.5],
    ):

        cp = _ax.contourf(
            _p.cycle,
            _p.year,
            _delta(_p),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(_amp, _amp / 10),
            extend="both",
        )
        _ax.axhline(1990, ls="--", c="gray", lw=1)
        _ax.set_yticks([])
        _ax.set_title(_l)
        _ax.set_xticks([_p.cycle[i] for i in [0, 5]], labels=["Jan", "Jun"])
        _ax.axvline(_p.cycle[5], ls="--", c="gray", lw=0.8)

    ## label
    _fig.colorbar(cp, ax=axs_13[-2], ticks=[-1, 0, 1], label=r"yr$^{-1}$")
    _fig.colorbar(cp, ax=axs_13[-1], ticks=[-0.5, 0, 0.5], label=r"K$^{-1}$ yr$^{-1}$")
    axs_13[0].set_yticks([1870, 1990, 2080])

    ## Save
    save(_fig, fname="param-hov", do_save=False)

    plt.gca()
    return


@app.cell
def _(Ro_, fits, fits_Ro, plt, src):
    _BJ = fits_Ro["Lac"].isel(rankx=0, ranky=0) + fits_Ro["Lac"].isel(rankx=1, ranky=1)

    _fig, axs_14 = plt.subplots(10, 1, figsize=(3, 11), layout="constrained")
    for _y in [1880, 1990, 2030]:  # , 2030, 2080]:
        # axs[0].plot(fits.cycle, fits_Ro.sel(year=y), label=y)
        axs_14[0].plot(fits.cycle, Ro_.sel(year=_y), label=_y)
        axs_14[1].plot(fits.cycle, fits_Ro["NROT_Lac"].isel(nro_form=0).sel(year=_y), label=_y)
        axs_14[2].plot(fits.cycle, fits_Ro["NROT_Lac"].isel(nro_form=2).sel(year=_y), label=_y)
        axs_14[3].plot(fits.cycle, fits_Ro["Lac"].isel(ranky=0, rankx=1).sel(year=_y), label=_y)
        axs_14[4].plot(fits.cycle, fits_Ro["NROT_Lac"].isel(nro_form=-1).sel(year=_y), label=_y)
        axs_14[5].plot(fits.cycle, fits_Ro["xi_stdac"].isel(ranky=0).sel(year=_y), label=_y)
        axs_14[6].plot(fits.cycle, -fits_Ro["Lac"].isel(ranky=1, rankx=0).sel(year=_y), label=_y)
        axs_14[7].plot(fits.cycle, -fits_Ro["NROH_Lac"].isel(nro_form=0).sel(year=_y), label=_y)
        axs_14[8].plot(fits.cycle, -fits_Ro["Lac"].isel(rankx=1, ranky=1).sel(year=_y), label=_y)
        # axs[9].plot(fits.cycle, -fits["NLb_Lac"].isel(ranky=1).sel(year=y), label=y)
        axs_14[9].plot(fits.cycle, _BJ.sel(year=_y), label=_y)


    src.utils.add_vticks(axs_14, xlines=fits.cycle.values[[3, 5]], xticks=[])
    _kwargs = dict(x=1.0, y=0.5)
    axs_14[0].text(s=r"$R$", transform=axs_14[0].transAxes, **_kwargs)
    axs_14[1].text(s=r"$\frac{dT}{dt}\propto T^2$", transform=axs_14[1].transAxes, **_kwargs)
    axs_14[2].text(s=r"$\frac{dT}{dt}\propto T^3$", transform=axs_14[2].transAxes, **_kwargs)
    axs_14[3].text(s=r"$F_1$", transform=axs_14[3].transAxes, **_kwargs)
    axs_14[4].text(s=r"$\frac{dT}{dt}\propto h^2$", transform=axs_14[4].transAxes, **_kwargs)
    axs_14[5].text(s=r"$\xi_T$", transform=axs_14[5].transAxes, **_kwargs)
    axs_14[6].text(s=r"$F_2$", transform=axs_14[6].transAxes, **_kwargs)
    axs_14[7].text(s=r"$\frac{dh}{dt}\propto -T^2$", transform=axs_14[7].transAxes, **_kwargs)
    axs_14[8].text(s=r"$\varepsilon$", transform=axs_14[8].transAxes, **_kwargs)
    # axs[9].text(s=r"$\frac{dh}{dt}\propto -h^2$", transform=axs[9].transAxes, **kwargs)
    axs_14[9].text(s=r"$R-\varepsilon$", transform=axs_14[9].transAxes, **_kwargs)
    for _ax in axs_14:
        _ax.axhline(0, ls="--", c="gray", lw=0.8)
    # ax.legend()

    axs_14[-1].set_xticks(fits.cycle.values[[3, 5]], labels=[4, 6])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot params over time
    """)
    return


@app.function
def add_vticks(axs, xticks, xlines=None):
    """add vertical lines to axs"""

    ## specify line style
    ax_kwargs = dict(ls="--", c="gray", lw=0.8)

    ## loop thru axs
    for ax in axs:
        ax.set_xticks(xticks)
        if xlines is not None:
            for x0 in xlines:
                ax.axvline(x0, **ax_kwargs)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot snapshots of growth rate over time. Also, plot seasonal cycle.
    """)
    return


@app.cell
def _(R_nl):
    ## specify where to evaluate R value
    T0 = 1.5

    if T0 is None:
        """for this case: average over all pos/neg values"""
        R_pos = R_nl.where(R_nl["sigma"] > 0).mean("sigma")
        R_neg = R_nl.where(R_nl["sigma"] < 0).mean("sigma")
        R_all = R_nl.mean("sigma")

    else:
        """for this case: find R at given sigma value"""
        R_pos = R_nl.sel(sigma=T0, method="nearest")
        R_neg = R_nl.sel(sigma=-T0, method="nearest")
        R_all = R_nl.sel(sigma=0, method="nearest")
    return R_all, R_neg, R_pos, T0


@app.cell
def _(R_neg, R_nl, R_nl_ann, R_pos, T0, cmocean, np, params, plt, save, src):
    YEARS = np.array([1880, 2010, 2050, 2080])
    _colors = cmocean.cm.balance(np.linspace(0.2, 0.8, len(YEARS)))

    _fig, axs_15 = plt.subplots(1, 3, figsize=(9, 2.5), layout="constrained")

    for _c, _y in zip(_colors, YEARS):

        ## plot R nonlinearity
        axs_15[0].plot(R_nl["sigma"], R_nl_ann.sel(year=_y), c=_c, label=_y)
        # axs[0].plot(R_nl["sigma"], (xr.ones_like(R_nl_ann)-eps.mean("cycle")).sel(year=y), c=c, label=y)
        # axs[0].plot(R_nl["sigma"], (R_nl_ann - eps.mean("cycle")).sel(year=y), c=c, label=y)

        ## plot R seasonal cycle
        axs_15[1].plot(
            # params.cycle,
            # 0.5 * (R_pos - eps).sel(year=y).squeeze(drop=True),
            # # -0.5 * (eps).sel(year=y).squeeze(drop=True),
            # c=c,
            # label=y,
            params.cycle,
            R_pos.sel(year=_y).squeeze(drop=True),
            c=_c,
            label=_y,
        )
        axs_15[2].plot(
            # params.cycle,
            # 0.5 * (R_neg - eps).sel(year=y).squeeze(drop=True),
            # c=c,
            # label=y,
            params.cycle,
            R_neg.sel(year=_y).squeeze(drop=True),
            c=_c,
            label=_y,
        )


    _ax_kwargs = dict(ls="--", c="k", lw=0.8)
    axs_15[0].axvline(0, **_ax_kwargs)

    axs_15[0].set_xlabel(r"$T$")
    axs_15[0].set_ylabel(r"Growth rate (yr$^{-1}$)")
    axs_15[0].set_title(r"$R(T)$, annual mean")
    axs_15[1].set_title(f"R($T=${T0:.1f})")
    axs_15[2].set_title(f"R($T=-${T0:.1f})")
    axs_15[2].legend(loc=(1.1, 0.15))

    for _ax in axs_15:
        _ax.axhline(0, **_ax_kwargs)

    add_vticks(axs_15[1:], xticks=[1, 7, 12], xlines=[7])
    for _ax in axs_15[1:]:
        _ax.set_xlabel("Month")
        _ax.set_xticks([1, 7, 12], labels=["Jan", "Jul", "Dec"])

    src.utils.set_lims(axs_15[1:])
    if T0 is not None:
        add_vticks(axs_15[:1], xticks=[-T0, 0, T0], xlines=[-T0, 0, T0])

    save(_fig, fname="gr-cycle", do_save=False)

    plt.gca()
    return (YEARS,)


@app.cell
def _(R_all, R_neg, R_pos, T0, YEARS, eps, params, plt, save):
    ## select cycle
    # SEL_CYCLE = lambda x: x.isel(cycle=slice(9, 12)).mean("cycle")
    # SEL_CYCLE = lambda x: x.isel(cycle=[7,8,9]).mean("cycle")
    # SEL_CYCLE = lambda x: x.isel(cycle=slice(6,None)).mean("cycle")
    SEL_CYCLE = lambda x: x.mean("cycle")

    _fig, axs_16 = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")

    ## plot R and R-epsilon
    for R__1, _c, _label in zip(
        [R_pos, R_neg, R_all], ["r", "b", "k"], [f"$T=${T0}", f"$T=-${T0}", "All"]
    ):

        ## get shared args
        plot_kwargs = dict(c=_c, label=_label)

        axs_16[0].plot(R__1.year, SEL_CYCLE(R__1), **plot_kwargs)
        axs_16[1].plot(R__1.year, 0.5 * SEL_CYCLE(R__1 - eps), **plot_kwargs)

    ## plot noise over time
    _delta = lambda x: x / x.isel(year=1) - 1
    axs_16[2].plot(params.year, _delta(params["xi_T"].mean("cycle")), label=r"$T$")
    axs_16[2].plot(params.year, _delta(params["xi_h"].mean("cycle")), label=r"$h$")
    # axs[2].plot(params.year, params["xi_T"].mean("cycle"), label=r"$T$")
    # axs[2].plot(params.year, params["xi_h"].mean("cycle"), label=r"$h$")

    ## label
    labels = [
        r"$R$",
        r"$\frac{1}{2}\left(R-\varepsilon\right)$",
        r"$\Delta$ (stochastic forcing amp.)",
    ]
    for _ax, _title in zip(axs_16, labels):
        _ax.set_title(_title)
    axs_16[0].legend()
    axs_16[2].legend()
    # axs[2].set_yticks([0, 0.1, 0.2, 0.3])

    ## formatting
    axs_16[0].set_ylabel(r"Growth rate (yr$^{-1}$)")
    axs_16[2].set_ylabel(r"Fraction")
    _ax_kwargs = dict(ls="--", c="k", lw=0.8)
    add_vticks(axs_16, xticks=YEARS[:-1], xlines=YEARS[:-1])
    # for ax in axs[-1:]:
    #     ax.axhline(0, **ax_kwargs)

    axs_16[0].axhline(0, **_ax_kwargs)
    axs_16[0].set_yticks([-1, 0])
    axs_16[1].set_yticks([-1, -0.5])
    axs_16[2].set_yticks([0, 0.1])

    save(_fig, fname="gr-over-time_ann", do_save=False)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Comparison plot between $R,R_o,\alpha$
    """)
    return


@app.cell(hide_code=True)
def _(cmocean, np, src):
    def plot_p(ax, p, months):
        """plot change over time for parameter"""

        ## get month indexes
        month_idxs = np.array(months) - 1

        ## helper funcs
        sel_month = lambda x: x.isel(cycle=month_idxs).mean("cycle")
        center = lambda x: x - x.isel(year=0)
        get = lambda x: center(sel_month(x))
        # get = lambda x : sel_month(x)

        ## get colors for plotting
        colors = cmocean.cm.balance(np.linspace(0.2, 0.8, 4))
        colors = [colors[i] for i in range(colors.shape[0])]
        colors = colors[:2] + ["k"] + colors[2:]

        ## plot different T values
        for s_, color in zip([-1, -0.5, 0, 0.5, 1], colors):

            ## plot data
            ax.plot(
                p.year, get(p.sel(sigma=s_, method="nearest")), c=color, label=f"$T=${s_}"
            )

        return


    def plot_p_comp(axs, p0, p1, p2, p3, months):
        """plot change over time for parameters"""

        ## plot total R, damping, and ocean contribution
        for ax, p in zip(axs, [p0, p1, p2, p3]):

            plot_p(ax, p, months=months)
            ax.set_ylim([-1.4, 1.4])

        axs[0].set_yticks([-1, 0, 1])

        src.utils.set_lims(axs)
        axs[0].set_ylabel(r"yr$^{-1}$")
        for ax in axs:
            ax.axhline(0, ls="--", c="k", lw=0.8)
        for ax in axs[1:]:
            ax.set_yticks([])

        src.utils.add_vticks(axs, xlines=[2010, 2060], xticks=[1870, 2010, 2060])

        return

    return (plot_p_comp,)


@app.cell
def _(ALPHA, R_nl, Ro, eps, np, plot_p_comp, plt, save):
    _fig, axs_17 = plt.subplots(5, 4, figsize=(10, 9), layout="constrained")

    for _j, (_months, _label, _amp) in enumerate(
        zip(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], np.arange(1, 13)],
            ["JFM", "AMJ", "JAS", "OND", "All"],
            [2, 2, 1.5, 1, 1],
            # [[12,1,2], [3,4,5], [6,7,8], [9,10,11], np.arange(1, 13)],
            # ["DJF", "MAM", "JJA", "SON", "All"],
            # [2, 2, 1.5, 1, 1],
        )
    ):

        plot_p_comp(
            axs_17[_j],
            p0=0.5 * (R_nl - eps),
            p1=R_nl,
            p2=-ALPHA["eval"].sel(flux_type="nhf"),
            p3=Ro,
            months=_months,
        )
        axs_17[_j, -1].text(s=_label, x=1.1, y=0.5, transform=axs_17[_j, -1].transAxes)

    for _axs_ in axs_17[:-1]:
        for _ax in _axs_:
            _ax.set_xticks([])

    axs_17[0, 0].set_title(r"$\frac{1}{2}\left(R-\varepsilon\right)$")
    axs_17[0, 1].set_title(r"$R$")
    axs_17[0, 2].set_title(r"$-\alpha$")
    axs_17[0, 3].set_title(r"$R_o=R+\alpha$")
    axs_17[0, 0].legend(prop=dict(size=8), loc="lower left")


    # src.utils.set_lims(axs.flatten())

    save(_fig, "gr-over-time", do_save=False)

    plt.gca()
    return


@app.cell
def _(ALPHA, R_nl, Ro, eps, np, plot_p_comp, plt):
    _fig, axs_18 = plt.subplots(3, 4, figsize=(10, 6), layout="constrained")

    for _j, (_months, _label) in enumerate(
        zip(
            [[1, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], np.arange(1, 13)],
            ["Jan-Jun", "Jul-Dec", "All"],
        )
    ):

        plot_p_comp(
            axs_18[_j],
            p0=0.5 * (R_nl - eps),
            p1=R_nl,
            p2=-ALPHA["eval"].sel(flux_type="nhf"),
            p3=Ro,
            months=_months,
        )
        axs_18[_j, -1].text(s=_label, x=1.1, y=0.5, transform=axs_18[_j, -1].transAxes)

    for _axs_ in axs_18[:-1]:
        for _ax in _axs_:
            _ax.set_xticks([])

    axs_18[0, 0].set_title(r"$\frac{1}{2}\left(R-\varepsilon\right)$")
    axs_18[0, 1].set_title(r"$R$")
    axs_18[0, 2].set_title(r"$-\alpha$")
    axs_18[0, 3].set_title(r"$R_o=R+\alpha$")


    # src.utils.set_lims(axs.flatten())

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot in a slightly different way
    """)
    return


@app.cell
def _(fits, fits_alpha):
    # fits_Ro = fits[["Lac","NROT_Lac"]] - fits_fixed_Ro[["Lac","NROT_Lac"]]
    fits_Ro_ = fits[["Lac", "NROT_Lac"]] + fits_alpha[["Lac", "NROT_Lac"]]
    return (fits_Ro_,)


@app.cell
def _(fits, fits_Ro_, fits_alpha, plt, src):
    _yr = fits.year
    _sel = lambda x: x[["Lac", "NROT_Lac"]].mean("cycle")
    _delta = lambda x: x - x.isel(year=0)
    sel2 = lambda x: _delta(_sel(x))

    ## funcs to select params
    sel_sym = lambda x: x["Lac"].isel(rankx=0, ranky=0)
    sel_asym = lambda x: x["NROT_Lac"].isel(nro_form=0)
    sel_cub = lambda x: x["NROT_Lac"].isel(nro_form=2)


    _fig, axs_19 = plt.subplots(1, 3, figsize=(9, 2.5), layout="constrained")

    for _ax, sel_fn in zip(axs_19, [sel_sym, sel_asym, sel_cub]):

        ## get func to select all data
        sel_ = lambda x: sel_fn(sel2(x))

        ## plot total
        _ax.plot(_yr, sel_(fits), c="k", label="$R$")

        ##
        for fits_, _l in zip(
            [fits_Ro_, -fits_alpha[["Lac", "NROT_Lac"]]], [r"$R_o$", r"$-\alpha$"]
        ):
            _ax.plot(_yr, sel_(fits_), label=_l)

    for _ax in axs_19:
        _ax_kwargs = dict(ls="--", c="k", lw=0.8)
        _ax.axhline(0, **_ax_kwargs)

    src.utils.add_vticks(axs_19, xticks=[1870, 2010], xlines=[2010])
    # axs[1].set_yticks([])

    src.utils.set_lims(axs_19)
    axs_19[0].set_ylabel(r"yr$^{-1}$")
    axs_19[1].set_ylabel(r"K$^{-1}$ yr$^{-1}$")
    axs_19[2].set_ylabel(r"K$^{-2}$ yr$^{-1}$")
    axs_19[0].legend()
    axs_19[0].set_title("Linear")
    axs_19[1].set_title("Quadratic")
    axs_19[2].set_title("Cubic")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ablation tests
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Funcs to tweak params
    """)
    return


@app.cell(hide_code=True)
def _(copy, fits, get_perturbed_NRO, np, params_fixed_F1, src, xr):
    def fix_R0(params, **kwargs):
        """Fix R parameter"""

        ## fix T (linear R)
        pparams = src.utils.get_perturbed_multi(params=params, idxs=[(0, 0)], **kwargs)

        return pparams


    def fix_R2(params, **kwargs):
        """Fix quadratic R parameter"""

        ## fix T (quadratic R)
        pparams = get_perturbed_NRO(nro_form_idx=0, params=params, nro_type="NROT_Lac")

        return pparams


    def fix_R3(params, **kwargs):
        """Fix quadratic R parameter"""

        ## fix T (quadratic R)
        pparams = get_perturbed_NRO(nro_form_idx=2, params=params, nro_type="NROT_Lac")

        return pparams


    def fix_R(params, **kwargs):
        """Fix R parameter"""

        ## fix T (linear R)
        pparams0 = fix_R0(params=params, **kwargs)

        ## fix T2
        pparams1 = fix_R2(params=pparams0)

        ## fix T3
        pparams = fix_R3(params=pparams1)

        return pparams


    def fix_F1(params, **kwargs):
        """Fix F1 parameter"""

        ## fix linear param (depependence on H)
        pparams0 = src.utils.get_perturbed_multi(params=params, idxs=[(0, 1)], **kwargs)

        ## fix dependence on H2
        pparams1 = get_perturbed_NRO(params=pparams0, nro_form_idx=-1, nro_type="NROT_Lac")

        ## fix dependence on T, H
        pparams = get_perturbed_NRO(params=pparams1, nro_form_idx=1, nro_type="NROT_Lac")

        return pparams


    def fix_F2(params, **kwargs):
        """Fix F1 parameter"""

        ## fix linear param (depependence on H)
        pparams0 = src.utils.get_perturbed_multi(params=params, idxs=[(1, 0)], **kwargs)

        ## fix dependence on T2
        pparams = get_perturbed_NRO(params=pparams0, nro_form_idx=0, nro_type="NROH_Lac")

        return pparams


    def fix_BJ(params, **kwargs):
        """fix R and epsilon"""

        ## get fixed R
        pparams_fixed_R = fix_R(params, **kwargs)

        ## fix epsilon
        pparams = src.utils.get_perturbed_multi(
            params=pparams_fixed_R, idxs=[(1, 1)], **kwargs
        )

        return pparams


    def fix_wyrtki(params, **kwargs):
        """fix F1 and F2"""

        ## fix F1
        pparams_fixed_F1 = fix_F1(params, **kwargs)

        ## fix F2
        pparams = fix_F2(params=params_fixed_F1, **kwargs)

        return pparams


    def fix_ddt_T(params, **kwargs):
        """fix R and F1"""

        ## get fixed R
        pparams_fixed_R = fix_R(params, **kwargs)

        ## fix F1
        pparams = fix_F1(params=pparams_fixed_R, **kwargs)

        return pparams


    def fix_ddt_h(params, **kwargs):
        """fix epsilon and F2"""

        ## get fixed R
        pparams_fixed_F2 = fix_F2(params, **kwargs)

        ## fix F1
        pparams = fix_eps(params=pparams_fixed_F2, **kwargs)

        return pparams


    def fix_eps(params, **kwargs):
        """fix epsilon"""
        pparams = src.utils.get_perturbed_multi(params=params, idxs=[(1, 1)], **kwargs)

        return pparams


    def fix_cycle(params, pparams, cycle_idxs):
        """only keep perturbation for part of seasonal cycle"""

        ## these are param names to modify
        pnames = [
            "Lac",
            "NLb_Lac",
            "NLc_Lac",
            "NROT_Lac",
            "NROH_Lac",
            "xi_stdac",
            "xi_std",
            "xi_cov",
            "xi_covac",
        ]

        ## get difference in params
        dparams = fits[pnames] - pparams[pnames]

        ## get months to return to baseline values (difference b/n sets)
        idxs_to_zero = set(np.arange(12)) - set(cycle_idxs)

        ## convert from set to array
        idxs_to_zero = np.array(list(idxs_to_zero))

        ## now remove param perturbations from these indices
        dparams_updated = xr.merge(
            [
                dparams.isel(cycle=cycle_idxs).drop_vars("xi_cov"),
                xr.zeros_like(dparams.isel(cycle=idxs_to_zero)),
            ]
        )

        ## get updated param perturbations
        ## apply updated perturbation to original parameter set
        pparams_updated = copy.deepcopy(pparams)
        pparams_updated[pnames] = fits[pnames] - dparams_updated[pnames]

        return pparams_updated

    return (
        fix_F1,
        fix_R,
        fix_R0,
        fix_R2,
        fix_R3,
        fix_cycle,
        fix_ddt_T,
        fix_ddt_h,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Test they work
    """)
    return


@app.cell
def _(fits, fix_R, fix_cycle, np, plt, src):
    fits_R = fix_R(fits)
    fits_R0 = fix_cycle(params=fits, pparams=fits_R, cycle_idxs=np.arange(6))

    _sel = lambda x: x["Lac"].isel(rankx=0, ranky=0).mean("cycle")

    _fig, axs_20 = plt.subplots(1, 2, figsize=(5, 2))

    for _ax, cyc_idx in zip(axs_20, [np.arange(0, 6), np.arange(6, 12)]):

        _ax.plot(fits.year, _sel(fits.isel(cycle=cyc_idx)), c="k")
        _ax.plot(fits.year, _sel(fits_R.isel(cycle=cyc_idx)))
        _ax.plot(fits.year, _sel(fits_R0.isel(cycle=cyc_idx)), ls="--")

    src.utils.set_lims(axs_20)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Specify ablation setup
    """)
    return


@app.cell
def _(
    diff_fits,
    fits,
    fits_Ro,
    fits_alpha,
    fits_alpha_lhf,
    fits_alpha_lw,
    fits_alpha_sw,
    fits_minus_lhf,
    fits_minus_lw,
    fits_minus_sw,
    fix_F1,
    fix_R,
    fix_R0,
    fix_R2,
    fix_R3,
    fix_ddt_T,
    fix_ddt_h,
    get_perturbed_NRO,
    src,
):
    ## should we fix the given parameter? or all others?
    FIX_OTHERS = False
    FIX_NOISE = False

    ## shared args
    pparam_kwargs = dict(params=fits, fix_others=FIX_OTHERS, fix_noise=FIX_NOISE)

    ## get fixed params
    fits_fixed_cubic = get_perturbed_NRO(nro_form_idx=2, params=fits, nro_type="NROT_Lac")
    fits_fixed_quad = get_perturbed_NRO(nro_form_idx=0, params=fits, nro_type="NROT_Lac")
    fits_fixed_R0 = fix_R0(**pparam_kwargs)
    fits_fixed_R = fix_R(**pparam_kwargs)
    fits_fixed_ddt_h = fix_ddt_h(**pparam_kwargs)
    fits_fixed_ddt_T = fix_ddt_T(**pparam_kwargs)
    fits_fixed_noise = src.utils.get_perturbed_noise(fits, fix_others=FIX_OTHERS)

    ## ocean/atm experiments
    fits_fixed_Ro_F1 = diff_fits(
        fix_F1(fix_R(params=fits_Ro)),
        fits_alpha,
    )

    fits_fixed_Ro = diff_fits(
        fix_R(params=fits_Ro),
        fits_alpha,
    )

    fits_fixed_Ro_lin = diff_fits(
        fix_R0(params=fits_Ro),
        fits_alpha,
    )

    fits_fixed_Ro_cub = diff_fits(
        fix_R3(params=fits_Ro),
        fits_alpha,
    )

    fits_fixed_Ro_F1_lin = diff_fits(
        fix_F1(fix_R0(params=fits_Ro)),
        fits_alpha,
    )

    fits_fixed_alpha = diff_fits(
        fits_Ro,
        fix_R(fits_alpha),
    )

    fits_fixed_sw = diff_fits(
        fits_minus_sw,
        fix_R(fits_alpha_sw),
    )

    fits_fixed_lhf = diff_fits(
        fits_minus_lhf,
        fix_R(fits_alpha_lhf),
    )

    fits_fixed_lw = diff_fits(
        fits_minus_lw,
        fix_R(fits_alpha_lw),
    )

    ## fix ocean terms
    fits_fixed_cubic = fix_R3(**pparam_kwargs)
    fits_fixed_quad = fix_R2(**pparam_kwargs)
    fits_fixed_R0 = fix_R0(**pparam_kwargs)
    fits_fixed_F1 = fix_F1(**pparam_kwargs)

    ## shared args (fix cycle)
    # cyc_kwargs = dict(params=fits, pparams=fits_fixed_Ro)
    # cyc_kwargs = dict(params=fits, pparams=fits_fixed_R0)
    # cyc_kwargs = dict(params=fits, pparams=fits_fixed_alpha)
    cyc_kwargs = dict(params=fits, pparams=fits_fixed_noise)

    ## specify params for experiments
    param_set_dict = {
        "Control": fits,
        "Fix noise": fits_fixed_noise,
        r"Fix $R$, $F_1$": fits_fixed_ddt_T,
        r"Fix $R_o$, $F_1$": fits_fixed_Ro_F1,
        # r"Fix $R_o$, $F_1$": fits_fixed_Ro,
        r"Fix $R_o$": fits_fixed_Ro,
        r"Fix $R$": fits_fixed_R,
        r"Fix $F1$": fits_fixed_F1,
        r"Fix $F_2$, $\varepsilon$": fits_fixed_ddt_h,
        # r"Fix lin.": fits_fixed_R0,
        # r"Fix quad": fits_fixed_quad,
        # r"Fix cubic": fits_fixed_cubic,
        r"Fix $\alpha$": fits_fixed_alpha,
        r"Fix shortwave": fits_fixed_sw,
        r"Fix latent": fits_fixed_lhf,
        r"Fix longwave": fits_fixed_lw,
        # r"Fix $F_1$": fix_F1(**pparam_kwargs),
        # r"Fix $R$, $F_1$ (AMJ)": fix_cycle(cycle_idxs=np.arange(4,7), params=fits, pparams=fits_fixed_Ro_F1),
        # r"Fix $R_o$ (AMJ)": fix_cycle(cycle_idxs=np.arange(4,7), params=fits, pparams=fits_fixed_Ro),
        # r"Fix $R_0$ (lin, AMJ)": fix_cycle(cycle_idxs=np.arange(4,7), params=fits, pparams=fits_fixed_Ro_lin),
        # r"Fix $R_0$ (cub, AMJ)": fix_cycle(cycle_idxs=np.arange(4,7), params=fits, pparams=fits_fixed_Ro_cub),
        # r"Fix $F_1$ (AMJ)": fix_cycle(cycle_idxs=np.arange(4,7), params=fits, pparams=fits_fixed_F1),
        # r"Fix $R$, $F_1$ (JFM)": fix_cycle(cycle_idxs=np.arange(0, 3), **cyc_kwargs),
        # r"Fix $R$, $F_1$ (AMJ)": fix_cycle(cycle_idxs=np.arange(3, 6), **cyc_kwargs),
        # r"Fix $R$, $F_1$ (JAS)": fix_cycle(cycle_idxs=np.arange(6, 9), **cyc_kwargs),
        # r"Fix $R$, $F_1$ (OND)": fix_cycle(cycle_idxs=np.arange(9, 12), **cyc_kwargs),
        # r"Fix $R$, $F_1$ (Jul-Dec)": fix_cycle(
        #     params=fits,
        #     pparams=fix_ddt_T(**pparam_kwargs),
        #     cycle_idxs=np.arange(6, 12),
        # ),
        # r"Fix $\varepsilon$": fix_eps(**pparam_kwargs),
        # r"Fix BJ": fits_fixed_BJ,
        # r"Fix $R_0$": fits_fixed_R0,
        # r"Fix Wyrtki": fits_fixed_wyrtki,
        # r"Fix $b$": get_perturbed_NRO(nro_form_idx=0, params=fits, nro_type="NROT_Lac"),
        # r"Fix $R_o$": fits_fixed_Ro,
        # r"Fix $\alpha$": fits_fixed_alpha,
        # r"Fix $\varepsilon$": fits_fixed_eps,
        # r"Fix Jan-Jun": fits_cyc0,
        # r"Fix Jul-Dec": fits_cyc1,°
        # r"Fixed $R_3$": src.utils.get_perturbed_multi(idxs=[(2, 2)], **pparam_kwargs),
        # r"Fix $L_o$": fits_fixed_Lo,
    }

    # ## get list of param sets and labels
    param_sets = list(param_set_dict.values())
    labels_1 = list(param_set_dict.keys())
    return labels_1, param_sets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot parameter perturbations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Helper func to plot param set
    """)
    return


@app.function(hide_code=True)
def plot_param_set(ax, params, model):
    """plot parameter set over time for given experiment"""

    ## get named named params (nnual mean)
    params_ = params.mean("cycle")

    ## plot core params
    for p in ["R", "epsilon", "F1", "F2", "bT_1", "cT_1", "bH_1"]:
        plot = ax.plot(params_.year, params_[p], label=p)
        ax.axhline(params_[p].isel(year=0), ls="--", c=plot[0].get_color(), lw=0.5)

    ## plot noise
    for p, c in zip(["xi_T", "xi_h"], ["gray", "lightgray"]):
        ax.plot(params_.year, params_[p], label=p, c=c)
        ax.axhline(params_[p].isel(year=0), ls="--", c=c, lw=0.5)

    ## format ax
    ax.set_xticks([])
    ax.legend(loc=(1.3, 0.1), prop=dict(size=6))

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the param set
    """)
    return


@app.cell
def _(MODEL, labels_1, param_sets, plt, src):
    _fig, axs_21 = plt.subplots(
        len(labels_1), 1, figsize=(4.5, 2.5 * len(labels_1)), layout="constrained"
    )

    ## plot control
    params0 = MODEL.get_RO_parameters(param_sets[0])
    plot_param_set(axs_21[0], params0, model=MODEL)
    axs_21[0].set_title(labels_1[0])

    ## plot diff from control
    for _ax, param_set, _label in zip(axs_21[1:], param_sets[1:], labels_1[1:]):

        params_ = MODEL.get_RO_parameters(param_set)

        plot_param_set(_ax, params_ - params0, model=MODEL)
        _ax.set_title(_label)

    ## formatting
    src.utils.set_lims(axs_21[1:])
    for _ax in axs_21[1:]:
        _ax.set_yticks([])


    ## legend
    # axs[-1].legend(loc=(1.3, 0.1), prop=dict(size=8))
    src.utils.add_vticks(axs_21, xticks=[], xlines=[2010])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run simulations
    """)
    return


@app.cell
def _(MODEL, get_sims_over_time, param_sets, sims, simulation_kwargs):
    exp_kwargs = dict(**simulation_kwargs, model=MODEL)
    sims_exp = [sims] + [get_sims_over_time(params=p, **exp_kwargs) for p in param_sets[1:]]
    return (sims_exp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot some samples
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute stats from simulations
    """)
    return


@app.cell
def _(get_stats, sims_exp, var_by_quant, xr):
    ## stat
    get_q = lambda x: x.quantile(q=[0.1, 0.5, 0.9], dim="member").rename({"quantile": "q"})

    ## funcs to get stats over time
    # season = xr.groupers.SeasonGrouper(["JFM", "AMJ", "JAS", "OND"])
    season = xr.groupers.SeasonGrouper(["DJF", "MAM", "JJA", "SON"])
    get_fn_ot = lambda exps, fn: [get_q(exp.groupby(time=season).map(fn)) for exp in exps]
    get_fn_ot2 = lambda exps, fn: [exp.groupby(time=season).map(fn) for exp in exps]

    ## compute
    sigma_exp = get_fn_ot(sims_exp, lambda x: x.std("time"))
    quantiles_exp = get_fn_ot2(sims_exp, get_stats)
    var_exp = get_fn_ot2(sims_exp, var_by_quant)

    ## SPECIFY PLOT SEASON
    PLOT_SEASON = "DJF"
    return PLOT_SEASON, quantiles_exp, sigma_exp, var_exp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Absolute stats from each experiments
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Variance
    """)
    return


@app.cell(hide_code=True)
def _(np, sns, varnames):
    def format_row(axs, y0, season):
        """format row of plot in comparison"""

        ## label
        axs[0].set_yticks([np.round(y0, 1), np.round(y0 + 0.5, 1)])
        for ax in axs[1:]:
            ax.set_ylabel(None)
        axs[-1].set_ylabel(season)
        axs[-1].yaxis.set_label_position("right")
        for ax in axs:
            ax.axhline(y0, c="gray", lw=1, ls="--")
            ax.set_ylim([y0 - 0.3, y0 + 0.7])

        return


    def format_subplots(axs, labels):
        """format all subplots"""
        for ax in axs[:-1].flatten():
            ax.set_xticks([])
            ax.set_xlabel(None)

        for ax in axs[:, 1:].flatten():
            ax.set_yticks([])

        for j in range(axs.shape[1]):
            axs[0, j].set_title(labels[j])

        return


    def plot_stats_comp(ax, list_of_stats, labels, colors=None, n=varnames[0]):
        """plot comparison of variance over time"""

        if colors is None:
            colors = sns.color_palette()[: len(list_of_stats)]

        for stats, label, c in zip(list_of_stats, labels, colors):

            ## plot median
            mplot = ax.plot(stats.year, stats[n].sel(q=0.5), lw=2.5, label=label, c=c)

            ## plot lower/upper quantiles
            kwargs = dict(c=mplot[0].get_color(), lw=0.8)
            for q in stats.q:
                if q != 0.5:
                    ax.plot(stats.year, stats[n].sel(q=q), **kwargs)

        ## label and set plotting specs
        ax.set_xlabel("Year")
        ax.set_ylabel(r"$\sigma_T$ ($^{\circ}$C)")
        ax.set_ylim([0.3, 1.7])
        ax.set_xticks([1870, 2010, 2080])
        ax.set_yticks([0.6, 1.2])

        return

    return format_row, format_subplots, plot_stats_comp


@app.cell
def _(
    format_row,
    format_subplots,
    labels_1,
    np,
    plot_stats_comp,
    plt,
    sigma_exp,
    sns,
    src,
    varnames,
):
    ## specify which variable to plot
    PLOT_VAR_3 = varnames[0]

    ## plot results
    _n = len(sigma_exp) - 1
    _fig, axs_22 = plt.subplots(4, _n, figsize=(_n * 2, 7), layout="constrained")

    for _i, s in enumerate(sigma_exp[0].season.values):

        ## loop thru experiments
        for _j in np.arange(len(sigma_exp) - 1):
            plot_stats_comp(
                axs_22[_i, _j],
                [sigma_exp[0].sel(season=s), sigma_exp[_j + 1].sel(season=s)],
                labels=[labels_1[0], labels_1[_j + 1]],
                colors=["k", sns.color_palette()[np.mod(_j, 10)]],
                n=PLOT_VAR_3,
            )

        ## get baseline value
        _sigma0 = sigma_exp[0].sel(season=s)[PLOT_VAR_3].isel(q=1, year=0).values.item()

        ## format
        format_row(axs=axs_22[_i], y0=_sigma0, season=s)

    ## format all subplots
    format_subplots(axs_22, labels=labels_1[1:])
    src.utils.add_vticks(axs_22.flatten(), xticks=[1870, 2010], xlines=[2010])

    # save(fig, fname="ro-experiments")

    plt.gca()
    return (PLOT_VAR_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Quantiles
    """)
    return


@app.cell
def _(PLOT_SEASON, PLOT_VAR_3, labels_1, plt, quantiles_exp, save, src):
    ## SPECIFY SEASON
    _SEASON = PLOT_SEASON

    _fig, axs_23 = plt.subplots(
        1, len(quantiles_exp), figsize=(3 * len(quantiles_exp), 3), layout="constrained"
    )
    for _ax, _stats_, _title in zip(axs_23, quantiles_exp, labels_1):
        _q = _stats_.sel(season=_SEASON)["q"].sel(v=PLOT_VAR_3)

        _ax.plot(_stats_.year, _q.sel(quantile=0.95), c="r", label="El Niño")
        _ax.plot(_stats_.year, -_q.sel(quantile=0.05), c="b", label="La Niña")
        _ax.plot(
            _stats_.year,
            0.5 * (_q.sel(quantile=0.95) - _q.sel(quantile=0.05)),
            label="Average",
            c="k",
        )

        _ax_kwargs = dict(ls="--", c="gray", lw=0.8)
        add_vticks(axs_23, xticks=[1870, 2010, 2050], xlines=[2010, 2050])
        _ax.set_title(_title)

    src.utils.set_lims(axs_23[:-1])
    for _ax in axs_23[1:]:
        axs_23[1].set_yticks([])
    axs_23[0].legend()
    axs_23[-1].set_ylim(axs_23[0].get_ylim())

    ## save to file
    save(_fig, fname="RO-experiments", do_save=False)

    plt.gca()
    return (axs_23,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot effect
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute effect
    """)
    return


@app.cell
def _(quantiles_exp, var_exp):
    ## for quantiles
    ctrl_ = quantiles_exp[0]
    quantiles_eff = [ctrl_ - q for q in quantiles_exp]

    ## for variance
    ctrl_var = var_exp[0]
    var_eff = [ctrl_var - q for q in var_exp]
    return ctrl_, quantiles_eff, var_eff


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plotting function
    """)
    return


@app.cell
def _(axs_23, src):
    def plot_quantiles(ax, stats, season, plot_var, lev=0.95):
        """plot quantiles on ax"""

        ## extract quantile data for plotting
        q = stats.sel(season=season)["q"].sel(v=plot_var)

        ## get el nino/la nina quantiles
        q_warm = q.sel(quantile=lev, method="nearest")
        q_cold = -q.sel(quantile=(1 - lev), method="nearest")

        ## plot data
        ax.plot(stats.year, q_warm, c="r", label="El Niño")
        ax.plot(stats.year, q_cold, c="b", label="La Niña")

        ## plot mean
        ax.plot(stats.year, 0.5 * (q_warm + q_cold), label="Average", c="k")

        ## formatting
        src.utils.add_vticks(axs_23, xticks=[1870, 2010, 2080], xlines=[2010])
        ax.axhline(0, ls="--", c="k", lw=0.8)
        ax.set_yticks([])

        return

    return (plot_quantiles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Quantiles
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Effect of each experiment
    """)
    return


@app.cell
def _(
    PLOT_SEASON,
    PLOT_VAR_3,
    ctrl_,
    labels_1,
    plot_quantiles,
    plt,
    quantiles_eff,
    quantiles_exp,
    src,
):
    ## specify seaons
    _SEASON = PLOT_SEASON

    ## number of figures in panel
    _n = len(quantiles_exp) - 1

    ## get control
    ctrl = ctrl_.sel(season=_SEASON)["q"].sel(v=PLOT_VAR_3)

    ## make plot
    _fig, axs_24 = plt.subplots(1, _n, figsize=(3 * (_n), 3), layout="constrained")
    for _ax, _stats_, _title in zip(axs_24, quantiles_eff[1:], labels_1[1:]):

        plot_quantiles(_ax, stats=_stats_, season=_SEASON, plot_var=PLOT_VAR_3)

        _ax.set_title(f"Effect of {_title[4:]}")
        # ax.set_ylim([-0.3, 0.4])
        # ax.set_ylim([-0.05, 0.2])

    src.utils.set_lims(axs_24)
    axs_24[0].set_yticks([-0.8, -0.4, 0, 0.4])

    axs_24[-1].set_ylim(axs_24[0].get_ylim())
    axs_24[0].legend()

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check linearity
    """)
    return


@app.cell
def _(
    PLOT_SEASON,
    PLOT_VAR_3,
    labels_1,
    plot_quantiles,
    plt,
    quantiles_eff,
    quantiles_exp,
    save,
    src,
    xr,
):
    _SEASON = PLOT_SEASON

    ## specify indices
    _idxs = [1, 5, 6, 7]

    ## compute number of panels for plot
    _n = len(_idxs) + 2

    ## get plot vars and labels
    _plot_vars = [quantiles_eff[i] for i in _idxs]
    _plot_labs = [labels_1[i] for i in _idxs]

    ## get control effect
    _ctrl_eff = quantiles_exp[0] - quantiles_exp[0].isel(year=0)

    ## get sum
    _sum_ = xr.zeros_like(quantiles_eff[0])
    for _i in _idxs:
        _sum_ = _sum_ + quantiles_eff[_i]

    _fig, axs_25 = plt.subplots(1, _n, figsize=(2 * _n, 2), layout="constrained")

    ## plot experiments
    for _ax, _stats_, _title in zip(axs_25, _plot_vars, _plot_labs):

        ## plot data
        plot_quantiles(_ax, _stats_, season=_SEASON, plot_var=PLOT_VAR_3)
        _ax.set_title(f"Effect of {_title[4:]}")

    ## plot sum
    plot_quantiles(axs_25[-2], _sum_, season=_SEASON, plot_var=PLOT_VAR_3)
    axs_25[-2].set_title("Sum")

    ## plo tcontrol
    plot_quantiles(axs_25[-1], _ctrl_eff, season=_SEASON, plot_var=PLOT_VAR_3)
    axs_25[-1].set_title("Control (all params)")

    ## formatting
    src.utils.set_lims(axs_25)
    axs_25[0].set_yticks([-0.4, 0, 0.4])
    axs_25[0].set_ylabel(r"$^{\circ}$C")
    axs_25[0].legend(prop=dict(size=8), loc="lower left")

    ## save
    save(_fig, fname="exp-comparison_general", do_save=False)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check linearity of subset
    """)
    return


@app.cell
def _(
    PLOT_SEASON,
    PLOT_VAR_3,
    labels_1,
    plot_quantiles,
    plt,
    quantiles_eff,
    save,
    src,
    xr,
):
    _SEASON = PLOT_SEASON

    ## specify indices
    _idxs = [4, -4]

    ## get control index
    _ctrl_idx = 5

    ## compute number of panels for plot
    _n = len(_idxs) + 2

    ## get plot vars and labels
    _plot_vars = [quantiles_eff[i] for i in _idxs]
    _plot_labs = [labels_1[i] for i in _idxs]

    ## get sum
    _sum_ = xr.zeros_like(quantiles_eff[0])
    for _i in _idxs:
        _sum_ = _sum_ + quantiles_eff[_i]

    _fig, axs_26 = plt.subplots(1, _n, figsize=(2.5 * _n, 2.5), layout="constrained")

    ## plot experiments
    for _ax, _stats_, _title in zip(axs_26, _plot_vars, _plot_labs):

        ## plot data
        plot_quantiles(_ax, _stats_, season=_SEASON, plot_var=PLOT_VAR_3)
        _ax.set_title(f"Effect of {_title[4:]}")

    ## plot sum
    plot_quantiles(axs_26[-2], _sum_, season=_SEASON, plot_var=PLOT_VAR_3)
    axs_26[-2].set_title("Sum")

    ## plo tcontrol
    plot_quantiles(axs_26[-1], quantiles_eff[_ctrl_idx], season=_SEASON, plot_var=PLOT_VAR_3)
    axs_26[-1].set_title("Control (effect of $R$)")

    ## formatting
    src.utils.set_lims(axs_26)
    axs_26[0].set_yticks([-0.8, -0.4, 0, 0.4])
    axs_26[0].set_ylabel(r"$^{\circ}$C")
    axs_26[0].legend()

    ## save to file
    save(_fig, fname="exp-comparison_Ro", do_save=False)

    plt.gca()
    return


@app.cell
def _(
    PLOT_SEASON,
    PLOT_VAR_3,
    labels_1,
    plot_quantiles,
    plt,
    quantiles_eff,
    save,
    src,
    xr,
):
    _SEASON = PLOT_SEASON

    ## specify indices
    _idxs = [-3, -2, -1]

    ## get control index
    _ctrl_idx = -4

    ## compute number of panels for plot
    _n = len(_idxs) + 2

    ## get plot vars and labels
    _plot_vars = [quantiles_eff[i] for i in _idxs]
    _plot_labs = [labels_1[i] for i in _idxs]

    ## get sum
    _sum_ = xr.zeros_like(quantiles_eff[0])
    for _i in _idxs:
        _sum_ = _sum_ + quantiles_eff[_i]

    _fig, axs_27 = plt.subplots(1, _n, figsize=(2.5 * _n, 2.5), layout="constrained")

    ## plot experiments
    for _ax, _stats_, _title in zip(axs_27, _plot_vars, _plot_labs):

        ## plot data
        plot_quantiles(_ax, _stats_, season=_SEASON, plot_var=PLOT_VAR_3)
        _ax.set_title(f"Effect of {_title[4:]}")

    ## plot sum
    plot_quantiles(axs_27[-2], _sum_, season=_SEASON, plot_var=PLOT_VAR_3)
    axs_27[-2].set_title("Sum")

    ## plo tcontrol
    plot_quantiles(axs_27[-1], quantiles_eff[_ctrl_idx], season=_SEASON, plot_var=PLOT_VAR_3)
    axs_27[-1].set_title(r"Control (effect of $\alpha$)")

    ## formatting
    src.utils.set_lims(axs_27)
    axs_27[0].set_yticks([-0.8, -0.4, 0, 0.4])
    axs_27[0].set_ylabel(r"$^{\circ}$C")
    axs_27[0].legend()


    ## save to file
    save(_fig, fname="exp-comparison_alpha", do_save=False)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Variance
    """)
    return


@app.cell
def _(PLOT_SEASON, labels_1, plt, src, var_eff, var_exp, varnames, xr):
    _SEASON = PLOT_SEASON

    _idxs = [1, 3, 4]
    # idxs = [1,2]
    _n = len(_idxs) + 2
    _plot_vars = [var_eff[i] for i in _idxs]
    _plot_labs = [labels_1[i] for i in _idxs]

    ## get control effect
    _ctrl_eff = var_exp[0] - var_exp[0].isel(year=0)

    ## get sum
    _sum_ = xr.zeros_like(var_eff[0])
    for _i in _idxs:
        _sum_ = _sum_ + var_eff[_i]

    _fig, axs_28 = plt.subplots(1, _n, figsize=(2.5 * _n, 2.5), layout="constrained")
    for _ax, _stats_, _title in zip(
        axs_28, _plot_vars + [_sum_] + [_ctrl_eff], _plot_labs + ["sum"] + ["Control"]
    ):
        _q = _stats_[varnames[0]].sel(season=_SEASON)

        _ax.plot(
            _stats_.year,
            _q.isel(intensity=2),
            c="r",
            label="El Niño",
        )
        _ax.plot(
            _stats_.year,
            _q.isel(intensity=0),
            c="b",
            label="La Niña",
        )
        _ax.plot(
            _stats_.year,
            0.5 * _q.sum("intensity"),
            label="Average",
            c="k",
        )

        _ax_kwargs = dict(ls="--", c="gray", lw=0.8)
        add_vticks(axs_28, xticks=[1870, 2010, 2050], xlines=[2010, 2050])
        _ax.axhline(0, ls="--", c="k", lw=0.8)
        _ax.set_title(f"Effect of {_title[4:]}")
        # ax.set_ylim([-0.2, 0.28])

    axs_28[-2].set_title("Sum")
    axs_28[-1].set_title("Control")
    src.utils.set_lims(axs_28)
    for _ax in axs_28[1:]:
        axs_28[1].set_yticks([])
    axs_28[0].legend()

    plt.gca()
    return


if __name__ == "__main__":
    app.run()
