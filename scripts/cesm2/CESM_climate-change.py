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
    # Look at climate change in CESM
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
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import cartopy.crs as ccrs
    import numpy as np
    import scipy.stats
    import seaborn as sns
    import xarray as xr
    import tqdm
    import pathlib
    import cmocean
    import os
    import cartopy.util
    import copy

    # Import custom modules
    import src.utils
    from src.XRO import XRO, xcorr

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})

    ## bump up DPI
    mpl.rcParams["figure.dpi"] = 100

    ## get filepaths
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])
    SAVE_FP = pathlib.Path(os.environ["SAVE_FP"])
    return (
        DATA_FP,
        XRO,
        cartopy,
        ccrs,
        copy,
        np,
        os,
        pathlib,
        pd,
        plt,
        scipy,
        src,
        tqdm,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Functions
    """)
    return


@app.cell
def _(
    cartopy,
    copy,
    dlat,
    eli_anom,
    eli_forced,
    np,
    os,
    pathlib,
    pd,
    src,
    tqdm,
    xr,
):
    def plot_hov(ax, data, amp, label=None, *, fig, axs):
        """Plot hovmoller of longitude vs. year"""

        # kwargs = dict(levels=src.utils.make_cb_range(3, 0.3), cmap="cmo.balance", extend="both")
        plot_data = ax.contourf(
            data.longitude,
            data.year,
            data.T,
            cmap="cmo.balance",
            extend="both",
            levels=src.utils.make_cb_range(amp, amp / 10),
        )
        cb = fig.colorbar(
            plot_data, orientation="horizontal", ticks=[-amp, 0, amp], label=label
        )

        ## label
        kwargs = dict(ls="--", c="w", lw=0.8)
        for ax in axs:
            ax.set_xlabel("Longitude")
            ax.set_xticks([190, 240])
            ax.set_yticks([])
            ax.axvline(190, **kwargs)
            ax.axvline(240, **kwargs)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

        return


    def plot_hov2(ax, data, amp, label=None, *, fig, axs):
        """Plot hovmoller of longitude vs. year"""

        # kwargs = dict(levels=src.utils.make_cb_range(3, 0.3), cmap="cmo.balance", extend="both")
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
            # plot_data, orientation="horizontal", ticks=[], label=None
        )

        ## label
        kwargs = dict(ls="--", c="w", lw=0.8)
        for ax in axs:
            # ax.set_xlabel("Month")
            # ax.set_xticks([1, 12])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

        return


    def get_rolling_var(data, n=10):
        """
        Get variance, computing over time and ensemble member. To increase
        sample size for variance estimate, compute over time window of 2n+1
        years, centered at given year.
        """

        return src.utils.get_rolling_fn_bymonth(data, fn=np.var, n=n)


    def get_ml_avg(data, Hm, delta=5, H0=None):
        """func to average data from surface to Hm + delta"""

        ## interpolate MLD onto data grid
        Hm_ = Hm.rename({"longitude": "lon"}).interp({"lon": data.lon})

        ## tweak integration bounds
        if H0 is None:
            Hm_ = Hm_ + delta

        else:
            Hm_ = H0 * xr.ones_like(Hm_)

        ## average over everything above the mixed layer
        return data.where(data.z_t <= Hm_).mean("z_t")


    def get_ml_avg_wrapper(data, Hm, delta=5):
        """wrapper function to format data for plotting"""

        ## first, compute mixed layer average
        ml_avg = get_ml_avg(data=data, Hm=Hm, delta=delta)

        ## rename coord and tranpose
        return ml_avg.rename({"lon": "longitude"}).transpose("month", ...)


    def plot_mld_bounds(ax, clim, m):
        """Plot MLD climatology and ± bounds"""

        ## clim
        ax.plot(clim.longitude, clim, c="k")

        ## El Niño
        ax.plot(clim.longitude, clim + m, c="r")

        ## La Niña
        ax.plot(clim.longitude, clim - m, c="b")

        return


    def get_wT(w, T):
        """function to get vertical flux (handles diff. w/T grids)"""

        ## rename w grid
        w_ = copy.deepcopy(w).rename({"z_w_top": "z_t"})
        w_ = w_.assign_coords({"z_t": T.z_t})

        return w_ * T


    def get_wdTdz(w, T):
        """function to get vertical flux (handles diff. w/T grids)"""

        ## rename w grid
        w_ = copy.deepcopy(w).rename({"z_w_top": "z_t"})
        w_ = w_.assign_coords({"z_t": T.z_t})

        ## get dTdz (convert from 1/cm to 1/m)
        dTdz = T.differentiate("z_t")

        return w_ * dTdz


    def get_udTdx(u, T):
        """zonal advection"""

        ## get grid spacing
        dlon_deg = T.lon.values[1] - T.lon.values[0]
        lat_deg = 0.0

        ## get grid spacing
        dx_m = get_dx(lat_deg=lat_deg, dlon_deg=dlon_deg)

        ## differentiate
        u_dfdx_ = u * T.differentiate("lon") * 1 / dx_m

        return u_dfdx_


    def get_u_adv(u, T):
        """zonal advection"""

        ## get grid spacing
        dlon_deg = T.lon.values[1] - T.lon.values[0]
        lat_deg = 0.0

        ## get grid spacing
        dx_m = get_dx(lat_deg=lat_deg, dlon_deg=dlon_deg)

        ## differentiate and convert units to K/yr
        mo_per_yr = 12
        u_dfdx_ = u * T.differentiate("lon") * 1 / dx_m * mo_per_yr

        return -u_dfdx_


    def recon_clim(data, components, varname="sst"):
        """reconstruct climatology for data"""

        ## get climatolgoy in PC space
        monthly_clim = data.groupby("time.month").mean()

        ## function to compute equatorial mean
        equatorial_mean = lambda x: x.sel(latitude=slice(-2, 2)).mean("latitude")

        ## reconstruct
        recon = src.utils.reconstruct_fn(
            components[varname], monthly_clim[varname], fn=equatorial_mean
        )

        ## fill zero values with NaN
        recon.values[recon.values == 0] = np.nan

        return recon


    def get_monthly_eli(t_bnds):

        ## get eli for period
        eli_ = eli_forced.isel(time=slice(*t_bnds)).groupby("time.month").mean()

        return eli_


    def get_monthly_eli_std(t_bnds):

        ## get eli for period
        eli_ = (
            eli_anom.isel(time=slice(*t_bnds)).groupby("time.month").std(["time", "member"])
        )

        return eli_


    def plot_cyclic(ax, data, sigma=None, **kwargs):
        """plot data on hovmoller with cyclic dependence on month"""

        ## add cyclic point
        data_cyclic, dim_cyclic = cartopy.util.add_cyclic_point(data, data.month, axis=0)

        ## plot data
        ax.plot(data_cyclic, dim_cyclic, **kwargs)

        ## plot bounds if they exist
        if sigma is not None:
            sigma_cyclic, _ = cartopy.util.add_cyclic_point(sigma, data.month, axis=0)

            ## plot data
            ax.plot(data_cyclic + sigma_cyclic, dim_cyclic, **kwargs, lw=0.8)
            ax.plot(data_cyclic - sigma_cyclic, dim_cyclic, **kwargs, lw=0.8)

        return


    def plot_cyclic_quantiles(ax, data, quantiles=[0.5, 0.15, 0.85], **kwargs):
        """plot data on hovmoller with cyclic dependence on month"""

        ## compute quantiles
        q = data.groupby("time.month").quantile(q=quantiles, dim=["time", "member"])
        # q = q.rename({"quantile":"q"})

        ## convert to numpy
        month = q.month.values
        q = q.transpose("quantile", "month").values

        ## add cyclic point
        q_cyclic, dim_cyclic = cartopy.util.add_cyclic_point(q, month, axis=1)

        ## plot median
        ax.plot(q_cyclic[0], dim_cyclic, **kwargs)

        ## plot other quantiles
        if len(quantiles) > 1:
            for j in range(1, len(quantiles)):
                ax.plot(q_cyclic[j], dim_cyclic, lw=0.8, **kwargs)

        return


    def format_subsurf_axs(axs):
        """add labels/formatting to 3-panel axs"""

        ## loop thru axs
        for ax in axs:
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.set_xlim([140, 281])
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


    def get_w_int(w):
        """get vertical velocity integrated over top 200 m"""
        return w.sel(z_t=slice(None, 200)).mean("z_t")


    def get_dTdz(Tsub):
        """get vertical velocity integrated over top 200 m"""
        T_surf = Tsub.sel(z_t=0, method="nearest").squeeze(drop=True)
        T_subsurf = Tsub.sel(z_t=200, method="nearest").squeeze(drop=True)

        return T_surf - T_subsurf


    def get_diags(data):
        """get diagnostics"""
        diags = xr.merge(
            [get_dTdz(data["T"]).rename("dTdz"), get_w_int(data["w"]).rename("w_int")]
        )
        return diags


    def get_dTdz_sub(Tsub, mld, delta=25):
        """get velocity at base of mixed layer"""

        ## get temperature difference
        dT = src.utils.get_dT_sub(Tsub=Tsub, Hm=mld, delta=delta)

        ## get gradient
        dTdz = -dT / mld

        return dTdz


    def get_nino34(data):
        return data.sel(longitude=slice(190, 240)).mean("longitude")


    def get_w_int_idx(data):
        """get nino3.4 w-int"""
        return get_nino34(get_w_int(data))


    def get_dTdz_idx(data):
        """get nino3.4 w-int"""
        return get_nino34(get_dTdz(data))


    def reconstruct_fn(components, scores, fn):

        ## reconstruct
        recon = (fn(components) * scores).sum("mode")

        return recon


    def eq_avg(x):
        return x.sel(latitude=slice(-5, 5), longitude=slice(125, 279)).mean("latitude")


    def get_var(data, year_center, n=15, fn=None):
        """
        Reconstruct variance for given time period.
        Estimate variance based on window centered on `year_center'.
        Window encompasses all samples within 'n' years of year_center.
        """

        ## get indices of samples in window
        in_window = np.abs(data.time.dt.year - year_center) <= n

        ## get variance of samples in window
        kwargs = dict(data=data.isel(time=in_window), fn=fn)
        data_var = src.utils.reconstruct_var_wrapper(**kwargs)

        return data_var


    def get_var_bymonth(data, year_center, n=15, fn=None):
        """Get variance by month"""

        ## function to apply to each month
        kwargs = dict(year_center=year_center, n=n, fn=fn)
        get_var_ = lambda x: get_var(x, **kwargs)
        return data.groupby("time.month").map(get_var_)


    def get_var_for_periods(data, periods, n=15, fn=None, by_month=True):
        """get variance for specified periods"""

        ## get arguments for variance reconstruction
        kwargs = dict(data=data, n=n, fn=fn)

        ## get variance recon func
        get_var_fn = get_var_bymonth if by_month else get_var
        var_by_period = [get_var_fn(year_center=y, **kwargs) for y in tqdm.tqdm(periods)]

        ## get dimension to represent period_centers
        period_dim = pd.Index(periods, name="period")

        ## put in array
        return xr.concat(var_by_period, dim=period_dim)


    def avg_mon_range(data, m0, m1):
        """average data each year over specified month range"""

        ## find indices for month range
        month = data.time.dt.month
        is_season = (month >= m0) & (month <= m1)

        ## get avg avg
        data_season = data.isel(time=is_season).groupby("time.year").mean()

        return data_season.rename({"year": "time"})


    def get_mam(data):
        """subset for MAM months"""

        return avg_mon_range(data, m0=3, m1=5)


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


    def get_dy(dlat_deg):
        """get spacing between latitudes in meters"""

        ## convert from degrees to radians
        dlat_rad = dlat / 180.0 * np.pi

        ## multiply by radius of earth
        R = 6.378e8  # earth radius (centimeters)
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


    def u_dfdx(u, f):
        """zonal advection"""

        ## get grid spacing
        dx_cm = get_dydx(f)["dx"]
        sec_per_year = 86400 * 365
        factor = sec_per_year / dx_cm

        u_dfdx_ = u * f.differentiate("longitude") * factor

        return u_dfdx_


    def v_dfdy(v, f):
        """meridional advection"""

        ## get grid spacing
        dy_cm = get_dydx(f)["dy"]
        sec_per_year = 86400 * 365
        factor = sec_per_year / dy_cm

        v_dfdy_ = v * f.differentiate("latitude") * factor

        return v_dfdy_


    def get_adv(uv, T):
        """
        Compute T tendency from horizontal advection.
        Equal to:
            (u,v) dot grad(-T)
        """

        ## compute grad T
        u_dTdx = u_dfdx(u=uv["uvel"], f=T)
        v_dTdy = v_dfdy(v=uv["vvel"], f=T)

        ## get

        return -(u_dTdx + v_dTdy)


    def save(fig, fname, dpi=300):
        """save figure to file"""

        ## get save directory
        save_dir = pathlib.Path(os.environ["SAVE_FP"], "ch3-outline")

        ## get fname
        fname = save_dir / f"{fname}.pdf"

        # fig.savefig(fname, dpi=dpi)
        return

    return (
        avg_mon_range,
        eq_avg,
        format_hov_axs,
        format_subsurf_axs,
        get_adv,
        get_dTdz_idx,
        get_dTdz_sub,
        get_diags,
        get_dx,
        get_mam,
        get_ml_avg_wrapper,
        get_rolling_var,
        get_u_adv,
        get_var_for_periods,
        get_w_int_idx,
        get_wdTdz,
        plot_cyclic_quantiles,
        plot_hov2,
        plot_mld_bounds,
        reconstruct_fn,
        save,
        set_xlims,
        set_ylims,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Change in $T$, $h$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load data
    And compute variance/skewness
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Load ELI
    """)
    return


@app.cell
def _(DATA_FP, pathlib, src, xr):
    ## load ELI data
    eli = xr.open_dataset(pathlib.Path(DATA_FP, "cesm/eli.nc"))

    ## get forced/anomalous component
    eli_forced, eli_anom = src.utils.separate_forced(eli)
    return eli, eli_anom, eli_forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Load $T$, $h$, and climate mode indices
    """)
    return


@app.cell
def _(src):
    ## open data
    Th = src.utils.load_cesm_indices(load_z20=True)

    ## rename indices for convenience
    Th = Th.rename(
        {
            "north_tropical_atlantic": "natl",
            "atlantic_nino": "nino_atl",
            "tropical_indian_ocean": "iobm",
            "indian_ocean_dipole": "iod",
            "north_pacific_meridional_mode": "npmm",
            "south_pacific_meridional_mode": "spmm",
        }
    )

    ## add ELI data
    # Th = xr.merge([Th, eli_anom["eli_15"]])
    return (Th,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute relative SST for $T$ indices
    """)
    return


@app.cell
def _(DATA_FP, Th, pathlib, xr):
    ## load tropical SST avg
    trop_sst = xr.open_dataset(pathlib.Path(DATA_FP, "cesm/trop_sst.nc"))

    ## Load T,h (total)
    Th_total = xr.open_dataset(DATA_FP / "cesm" / "Th.nc")

    ## compute relative sst
    for _n in ["T_3", "T_34", "T_4"]:
        Th[f"{_n}_rel"] = Th_total[_n] - trop_sst["trop_sst_10"]
    return Th_total, trop_sst


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute stats
    """)
    return


@app.cell
def _(Th, get_rolling_var, scipy, src):
    ## get rolling variance, by month
    Th_var = get_rolling_var(Th, n=15)
    Th_var_bymonth = src.utils.unstack_month_and_year(Th_var)

    ## get rolling skew, by month
    Th_skew = src.utils.get_rolling_fn_bymonth(Th, fn=scipy.stats.skew, n=20)
    Th_skew_bymonth = src.utils.unstack_month_and_year(Th_skew)

    ## Get % increase in variance
    _baseline = Th_var_bymonth.isel(year=slice(None, 30)).mean("year")
    Th_var_bymonth_pct = 100 * (Th_var_bymonth - _baseline) / _baseline
    return Th_skew_bymonth, Th_var_bymonth, Th_var_bymonth_pct


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hövmöller
    """)
    return


@app.cell
def _(Th, Th_var_bymonth_pct, plot_hov2, plt):
    ## setup plot
    fig, axs = plt.subplots(
        1, len(list(Th)), figsize=(0.9 * len(list(Th)), 2), layout="constrained"
    )

    ## plot T data
    for _ax, _n in zip(axs, list(Th)):
        plot_hov2(_ax, Th_var_bymonth_pct[_n].T, amp=100, fig=fig, axs=axs)
        _ax.set_title(_n, fontsize=9)
        _ax.axvline(8, ls="--", c="w", lw=1)


    plt.gca()
    return axs, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare August and December variance over time
    """)
    return


@app.cell
def _(Th_var_bymonth, plt):
    fig_1, _ax = plt.subplots(figsize=(2.5, 2))

    ## plot data
    _ax.plot(
        Th_var_bymonth.year, Th_var_bymonth["T_34"].sel(month=8), label="Aug", c="k", ls="-"
    )
    _ax.plot(
        Th_var_bymonth.year,
        Th_var_bymonth["T_34"].sel(month=11),
        label="Dec",
        c="gray",
        ls="--",
    )

    ## label and style
    _ax.set_title(r"$\sigma^2\left(\text{Niño 3.4}\right)$")
    _ax.set_ylim([0, None])
    _ax.legend(prop=dict(size=8))
    _ax.set_ylabel(r"$^{\circ}\text{C}^2$")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Skewness
    """)
    return


@app.cell
def _(Th_skew_bymonth, plot_hov2, plt):
    ## setup plot
    fig_2, axs_1 = plt.subplots(1, 9, figsize=(11, 4), layout="constrained")

    ## plot T data
    for _i, _n in enumerate(
        ["T_4", "T_34", "T_3", "h", "h_w", "h_w_z20", "h_z20", "T_34_rel", "T_3_rel"]
    ):
        plot_hov2(axs_1[_i], Th_skew_bymonth[_n].T, amp=1.5, fig=fig_2, axs=axs_1)
        axs_1[_i].set_title(_n, fontsize=9)
        axs_1[_i].axvline(5, ls="--", c="w", lw=1, alpha=0.5)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Nicer plot, for thesis
    """)
    return


@app.cell
def _(src):
    def plot_hov_helper(ax, data, amp, label=None):
        """Plot hovmoller of longitude vs. year"""

        plot_data = ax.contourf(
            data.month,
            data.year,
            data.T,
            cmap="cmo.balance",
            extend="both",
            levels=src.utils.make_cb_range(amp, amp / 10),
        )

        ## label
        kwargs = dict(ls="--", c="w", lw=1, alpha=0.5)
        ax.set_xlabel("Month")
        ax.set_xticks([1.6, 5, 11.4], labels=["Jan", "May", "Dec"])
        ax.axhline(2010, **kwargs)
        ax.axvline(5, **kwargs)
        ax.set_yticks([])

        return plot_data

    return (plot_hov_helper,)


@app.cell
def _(Th_skew_bymonth, plot_hov_helper, plt, save):
    ## setup plot
    fig_3, axs_2 = plt.subplots(1, 3, figsize=(7, 3), layout="constrained")

    ## plot T data
    for _i, (_n, title) in enumerate(
        zip(
            ["T_4", "T_34", "T_3"],
            ["(a) Niño 4", "(b) Niño 3.4", "(c) Niño 3"],
        )
    ):
        _cp = plot_hov_helper(axs_2[_i], Th_skew_bymonth[_n].T, amp=1.5)
        axs_2[_i].set_title(title, loc="left")

    fig_3.colorbar(_cp, ax=axs_2[-1], ticks=[-1.5, 0, 1.5], label="Skew (non-dim.)")
    axs_2[0].set_yticks([1870, 2010, 2080])

    save(fig_3, fname="skew_by-season_over-time", dpi=300)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ELI
    """)
    return


@app.cell
def _(eli_anom, eli_forced, get_rolling_var, scipy, src):
    ## get mean by month (smoothed)
    eli_mean_bymonth = src.utils.unstack_month_and_year(eli_forced)
    eli_mean_bymonth = src.utils.get_rolling_avg(eli_mean_bymonth, n=15, dim="year")
    delta_eli = eli_mean_bymonth - eli_mean_bymonth.isel(year=0)

    ## get rolling variance, by month
    eli_var = get_rolling_var(eli_anom, n=15)
    eli_var_bymonth = src.utils.unstack_month_and_year(eli_var)

    ## Get % increase in variance
    _baseline = eli_var_bymonth.isel(year=slice(None, 30)).mean("year")
    eli_var_bymonth = 100 * (eli_var_bymonth - _baseline) / _baseline

    ## get rolling skew, by month
    eli_skew = src.utils.get_rolling_fn_bymonth(eli_anom, fn=scipy.stats.skew, n=15)
    eli_skew_bymonth = src.utils.unstack_month_and_year(eli_skew)
    return delta_eli, eli_skew_bymonth, eli_var_bymonth


@app.cell
def _(delta_eli, eli_skew_bymonth, eli_var_bymonth, plot_hov2, plt):
    ## specify which ELI index to plot
    eli_idx = "eli_05"

    ## setup plot
    fig_4, axs_3 = plt.subplots(1, 3, figsize=(3.16, 3), layout="constrained")

    ## plot mean
    plot_hov2(axs_3[0], delta_eli[eli_idx].T, amp=15, fig=fig_4, axs=axs_3)
    plot_hov2(axs_3[1], eli_var_bymonth[eli_idx].T, amp=300, fig=fig_4, axs=axs_3)
    plot_hov2(axs_3[2], eli_skew_bymonth[eli_idx].T, amp=3, fig=fig_4, axs=axs_3)

    ## label
    _kwargs = dict(size=8)
    axs_3[0].set_title(r"$\Delta$ mean", **_kwargs)
    axs_3[1].set_title(r"$\Delta$ variance (%)", **_kwargs)
    axs_3[2].set_title("skew", **_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### stats
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Scatter ELI vs. Niño indices
    """)
    return


@app.cell
def _(Th, Th_total, eli, eli_anom, xr):
    ## select data
    data = xr.merge(
        [
            Th[["T_3_rel", "T_34_rel", "T_4_rel", "T_4", "T_3", "T_34", "h_w_z20"]],
            eli_anom["eli_05"],
        ]
    )
    data_early = data.isel(time=slice(None, 360))
    data_late = data.isel(time=slice(-360, None))
    # data_late = data.sel(time=slice("1980","2009"))

    ## get corresponding data for total
    total = xr.merge([Th_total, eli])
    total_early = total.sel(time=data_early.time)
    total_late = total.sel(time=data_late.time)
    return data_early, data_late, total_early, total_late


@app.cell
def _(data_early, data_late, plt, set_xlims, set_ylims):
    ## specify which T-variable to plot
    _T_var = "T_3_rel"

    ## specify month
    _month = 4

    ## function to select month
    sel_mon = lambda x: x.isel(time=x.time.dt.month == _month)

    ## plot kwargs
    _kwargs = dict(s=2, alpha=0.8)

    fig_5, axs_4 = plt.subplots(2, 1, figsize=(3, 5), layout="constrained")

    axs_4[0].scatter(
        sel_mon(data_early[_T_var]),
        sel_mon(data_early["eli_05"]),
        **_kwargs,
    )

    axs_4[1].scatter(
        sel_mon(data_late[_T_var]),
        sel_mon(data_late["eli_05"]),
        **_kwargs,
    )

    ## format
    axs_4[0].set_xticks([])
    axs_4[1].set_xlabel(r"$T$")
    set_ylims(axs_4)
    set_xlims(axs_4)
    for _ax in axs_4:
        _kwargs = dict(c="k", lw=0.8, ls="--")
        _ax.axhline(0, **_kwargs)
        _ax.axvline(0, **_kwargs)
        _ax.set_ylabel(r"ELI")

    plt.gca()
    return (sel_mon,)


@app.cell
def _(plt, sel_mon, set_xlims, set_ylims, total_early, total_late):
    fig_6, axs_5 = plt.subplots(1, 2, figsize=(5.5, 2.5), layout="constrained")

    ## kwargs
    sc_kwargs = dict(s=2, alpha=0.6)

    ## scatter all data
    axs_5[0].scatter(
        sel_mon(total_early["eli_05"]),
        sel_mon(total_early["T_4"] - total_early["T_3"]),
        **sc_kwargs,
    )

    axs_5[1].scatter(
        sel_mon(total_late["eli_05"]),
        sel_mon(total_late["T_4"] - total_late["T_3"]),
        **sc_kwargs,
    )

    set_ylims(axs_5)
    set_xlims(axs_5)

    for _ax in axs_5:
        _kwargs = dict(c="k", lw=0.8, ls="--")
        _ax.axhline(0, **_kwargs)
        _ax.set_xlabel("ELI")
        _ax.set_xticks([190, 240])
        for _t in _ax.get_xticks():
            _ax.scatter(_t, 0, c="k", s=100, marker="|", linewidths=3)

    axs_5[0].set_ylabel(r"(Niño 4) – (Niño 3)")
    axs_5[1].set_yticks([])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Look at change relative to median
    """)
    return


@app.cell
def _(data_early, data_late, np, plt, scipy, src):
    ## specify plot variable
    _plot_var = "T_34"

    ## function to select month
    sel_mon_1 = lambda x: x.isel(time=x.time.dt.month == 1)

    if "eli" in _plot_var:
        _edges = np.arange(-64, 68, 4)
        _tick_kwargs = dict(ticks=[-50, 0, 50], labels=["-50", "median", "+50"])
        # edges = np.arange(-25,27.5, 2.5)

    elif "h" in _plot_var:
        _edges = np.arange(-34.5, 37.5, 3)
        _tick_kwargs = dict(ticks=[-30, 0, 30], labels=["-30", "median", "+30"])

    else:
        _edges = np.arange(-4.05, 4.05 + 1 / 3, 1 / 3)
        _tick_kwargs = dict(ticks=[-3, 0, 3], labels=["-3", "median", "+3"])


    ## get data
    _y0 = sel_mon_1(data_early)[_plot_var].values.flatten()
    _y1 = sel_mon_1(data_late)[_plot_var].values.flatten()

    ## compute pdfs
    _pdf0, _ = src.utils.get_empirical_pdf(_y0, edges=_edges)
    _pdf1, _ = src.utils.get_empirical_pdf(_y1, edges=_edges)

    ## compute skewness
    s0 = scipy.stats.skew(_y0)
    s1 = scipy.stats.skew(_y1)

    ## make plot
    fig_7, _ax = plt.subplots(figsize=(4, 3), layout="constrained")
    # ax.stairs(pdf0, edges - np.median(y0), label=f"early: skew = {s0:.2f}")
    # ax.stairs(pdf1, edges - np.median(y1), label=f"late: skew = {s1:.2f}")
    _ax.stairs(_pdf0, _edges - np.median(_y0), label=f"Baseline")
    _ax.stairs(_pdf1, _edges - np.median(_y1), label=f"Future")
    _ax.axvline(0, c="k", lw=1)
    _ax.set_xticks(**_tick_kwargs)

    _ax.legend(prop=dict(size=12))

    plt.gca()
    return s0, s1, sel_mon_1


@app.cell
def _(np, plt, sel_mon_1, src, total_early, total_late):
    ## get data for histogram
    _plot_var = "eli_05"

    ## get data
    _y0 = sel_mon_1(total_early)[_plot_var].values.flatten()
    _y1 = sel_mon_1(total_late)[_plot_var].values.flatten()

    ## compute pdfs
    _edges = np.arange(140, 280, 5)
    _pdf0, _ = src.utils.get_empirical_pdf(_y0, edges=_edges)
    _pdf1, _ = src.utils.get_empirical_pdf(_y1, edges=_edges)

    ## make plot
    fig_8, _ax = plt.subplots(figsize=(3, 2.5), layout="constrained")
    _ax.stairs(_pdf0, _edges, label=f"early")
    _ax.stairs(_pdf1, _edges, label=f"late")
    _ax.set_xticks(ticks=[190, 240])
    _ax.set_ylim([-2e-3, None])
    _ax.set_xlim([None, 255])
    _ax.set_yticks([])
    _ax.set_ylabel("Prob. density")
    _ax.set_xlabel("ELI (longitude)")

    for _t in _ax.get_xticks():
        _ax.scatter(_t, 0, marker="|", s=100, linewidths=3, c="k")


    _ax.legend(prop=dict(size=8))

    plt.gca()
    return


@app.cell
def _(np, plt, s0, s1, src, total_early, total_late):
    ## specify plot variable
    _plot_var = "T_34"

    ## function to select month
    sel_mon_2 = lambda x: x.isel(time=x.time.dt.month == 12)

    _edges = np.arange(22, 34, 0.5)

    ## get data
    _y0 = sel_mon_2(total_early)[_plot_var].values.flatten()
    _y1 = sel_mon_2(total_late)[_plot_var].values.flatten()

    ## compute pdfs
    _pdf0, _ = src.utils.get_empirical_pdf(_y0, edges=_edges)
    _pdf1, _ = src.utils.get_empirical_pdf(_y1, edges=_edges)


    ## make plot
    fig_9, _ax = plt.subplots(figsize=(4, 3), layout="constrained")
    _ax.stairs(_pdf0, _edges, label=f"early: skew = {s0:.2f}")
    _ax.stairs(_pdf1, _edges, label=f"late: skew = {s1:.2f}")
    # ax.axvline(0, c="k", lw=1)
    # ax.set_xticks(**tick_kwargs)

    _ax.legend(prop=dict(size=8))

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Change in spatial patterns
    """)
    return


@app.cell
def _(src):
    ## load spatial data
    forced, anom = src.utils.load_consolidated()
    return anom, forced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Change in mean
    """)
    return


@app.cell
def _(src):
    def merimean_safe(data, lat_bound=2):
        """get meridional mean"""
        if "latitude" in data.coords:
            return src.utils.merimean(data, lat_bound)

        else:
            return data

    return (merimean_safe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Computation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get climatologies
    """)
    return


@app.cell
def _(anom, eli, forced, merimean_safe, src):
    ## specify latitude bound for hovmollers
    LAT_BOUND = 5

    ## split into early/late periods
    t_early = dict(time=slice("1850", "1890"))
    # t_late = dict(time=slice("1980", "2009"))
    # t_early = dict(time=slice("1985", "2014"))
    t_late = dict(time=slice("2060", "2100"))

    ## split surface data
    anom_early = anom.sel(t_early)
    anom_late = anom.sel(t_late)

    ## climatolgoies
    _fn = lambda x: merimean_safe(x, lat_bound=LAT_BOUND)
    clim_early = src.utils.reconstruct_clim(forced.sel(t_early), fn=_fn)
    clim_late = src.utils.reconstruct_clim(forced.sel(t_late), fn=_fn)

    for _clim in [clim_early, clim_late]:
        _clim["dx_sst"] = 100 * _clim["sst"].differentiate("longitude")

    ## get difference
    clim_diff = clim_late - clim_early

    ## compute ELI
    eli_early = eli.sel(t_early)
    eli_late = eli.sel(t_late)
    return (
        anom_early,
        anom_late,
        clim_diff,
        clim_early,
        clim_late,
        eli_early,
        eli_late,
        t_early,
        t_late,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### SST
    """)
    return


@app.cell
def _(
    clim_diff,
    clim_early,
    clim_late,
    eli_early,
    eli_late,
    np,
    plot_cyclic_quantiles,
    plt,
    src,
):
    ## specify which ELI index to plot
    _ELI = "eli_05"

    ## make hövmöllers
    fig_10, axs_6 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.thermal", levels=np.arange(23, 32), extend="both")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_6[0], clim_early["sst"], **_kwargs)

    ## plot late
    _kwargs["levels"] = _kwargs["levels"]  # + 3
    _cp1 = src.utils.plot_cycle_hov(axs_6[1], clim_late["sst"], **_kwargs)

    ## plot bias
    _cp2 = src.utils.plot_cycle_hov(
        axs_6[2],
        clim_diff["sst"],
        cmap="cmo.amp",
        levels=np.arange(2.6, 5.2, 0.2) - 2.6,
        extend="both",
    )

    ## plot eli
    plot_cyclic_quantiles(ax=axs_6[0], data=eli_early[_ELI], c="w")
    plot_cyclic_quantiles(ax=axs_6[1], data=eli_late[_ELI], c="k")
    plot_cyclic_quantiles(ax=axs_6[2], data=eli_early[_ELI], c="w")
    plot_cyclic_quantiles(ax=axs_6[2], data=eli_late[_ELI], c="k")

    ## label
    axs_6[0].set_title("Early")
    axs_6[1].set_title("Late")
    axs_6[2].set_title("Difference")
    axs_6[-1].set_xlabel("Longitude")
    axs_6[-1].set_xticks([140, 190, 240])

    _cb0 = fig_10.colorbar(_cp0, ax=axs_6[0], ticks=[23, 27, 31], label=r"$^{\circ}C$")
    _cb1 = fig_10.colorbar(_cp1, ax=axs_6[1], ticks=[26, 30, 34], label=r"$^{\circ}C$")
    _cb2 = fig_10.colorbar(_cp2, ax=axs_6[2], ticks=[3, 4, 5], label=r"$^{\circ}C$")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Gradient
    """)
    return


@app.cell
def _(
    clim_diff,
    clim_early,
    clim_late,
    eli_early,
    eli_late,
    plot_cyclic_quantiles,
    plt,
    src,
):
    ## specify which ELI index to plot
    _ELI = "eli_05"

    ## get colorbar
    _kwargs = dict(
        cmap="cmo.balance", extend="both", levels=src.utils.make_cb_range(15, 1.5)
    )
    kwargs_diff = dict(
        cmap="cmo.balance", extend="both", levels=src.utils.make_cb_range(7.5, 0.75)
    )


    ## make hövmöllers
    fig_11, axs_7 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_7[0], clim_early["dx_sst"], **_kwargs)

    ## plot late
    # kwargs["levels"] = kwargs["levels"] + 3
    _cp1 = src.utils.plot_cycle_hov(axs_7[1], clim_late["dx_sst"], **_kwargs)

    ## plot bias
    _cp2 = src.utils.plot_cycle_hov(axs_7[2], clim_diff["dx_sst"], **kwargs_diff)

    ## plot eli
    plot_cyclic_quantiles(ax=axs_7[0], data=eli_early[_ELI], c="w")
    plot_cyclic_quantiles(ax=axs_7[1], data=eli_late[_ELI], c="k")
    plot_cyclic_quantiles(ax=axs_7[2], data=eli_early[_ELI], c="w")
    plot_cyclic_quantiles(ax=axs_7[2], data=eli_late[_ELI], c="k")

    ## label
    axs_7[0].set_title("Early")
    axs_7[1].set_title("Late")
    axs_7[2].set_title("Difference")
    axs_7[-1].set_xlabel("Longitude")
    axs_7[-1].set_xticks([140, 190, 240])

    LABEL = r"$K~\left(100^{\circ}\text{lon}\right)^{-1}$"
    _cb0 = fig_11.colorbar(_cp0, ax=axs_7[0], ticks=[-15, 0, 15], label=LABEL)
    _cb1 = fig_11.colorbar(_cp1, ax=axs_7[1], ticks=[-15, 0, 15], label=LABEL)
    _cb2 = fig_11.colorbar(_cp2, ax=axs_7[2], ticks=[-7.5, 0, 7.5], label=LABEL)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Relative SST
    """)
    return


@app.function
def plot_level(
    ax, data, level, ls="-", lw=1, c="w", x_coord="longitude", y_coord="month", **kwargs
):
    """plot single level on hovmoller"""
    ax.contour(
        data[x_coord],
        data[y_coord],
        data,
        levels=[level],
        colors=c,
        linestyles=ls,
        linewidths=lw,
        **kwargs,
    )
    return


@app.cell
def _(t_early, t_late, trop_sst):
    ## compute tropical sst clim
    compute_clim = lambda x: x.groupby("time.month").mean(["time", "member"])
    trop_sst_clim_early = compute_clim(trop_sst.sel(t_early))
    trop_sst_clim_late = compute_clim(trop_sst.sel(t_late))
    return trop_sst_clim_early, trop_sst_clim_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    average sst over wider area?
    """)
    return


@app.cell
def _(
    clim_early,
    clim_late,
    plt,
    src,
    trop_sst_clim_early,
    trop_sst_clim_late,
):
    ## specify which trop_sst variable index to use
    trop_sst_var = "trop_sst_15"

    ## compute
    rel_sst_early = clim_early["sst"] - trop_sst_clim_early[trop_sst_var]
    rel_sst_late = clim_late["sst"] - trop_sst_clim_late[trop_sst_var]
    diff = rel_sst_late - rel_sst_early

    ## make hövmöllers
    fig_12, axs_8 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.balance", levels=src.utils.make_cb_range(4, 0.5), extend="both")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_8[0], rel_sst_early, **_kwargs)

    ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_8[1], rel_sst_late, **_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_8[2],
        (rel_sst_late - rel_sst_early),
        **dict(_kwargs, levels=_kwargs["levels"] / 2),
    )

    ## contour zero levels
    _c_kwargs = dict(level=0, c="k")
    plot_level(axs_8[0], rel_sst_early, **_c_kwargs)
    plot_level(axs_8[1], rel_sst_early, **_c_kwargs)
    plot_level(axs_8[1], rel_sst_late, ls="--", **_c_kwargs)


    ## label
    axs_8[0].set_title("Early")
    axs_8[1].set_title("Late")
    axs_8[-1].set_xlabel("Longitude")
    axs_8[-1].set_xticks([140, 190, 240])

    _cb0 = fig_12.colorbar(_cp0, ax=axs_8, ticks=[-4, 0, 4], label=r"$^{\circ}C$")
    # cb1 = fig.colorbar(cp1, ax=axs[1], ticks=[26, 30, 34], label=r"$^{\circ}C$")
    # cb2 = fig.colorbar(cp2, ax=axs[2], ticks=[3, 4, 5], label=r"$^{\circ}C$")

    plt.gca()
    return rel_sst_early, rel_sst_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Precip
    """)
    return


@app.cell
def _(clim_early, clim_late, np, plt, rel_sst_early, rel_sst_late, src):
    ## specify non-dimensional scale
    SCALE = 8.6e4

    ## make hövmöllers
    fig_13, axs_9 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.rain", levels=np.arange(0, 16.5, 1.5), extend="max")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_9[0], SCALE * clim_early["pr"], **_kwargs)

    ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_9[1], SCALE * clim_late["pr"], **_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_9[2],
        SCALE * (clim_late - clim_early)["pr"],
        cmap="cmo.balance_r",
        levels=src.utils.make_cb_range(7.5, 0.75),
        extend="both",
    )

    ## contour zero levels
    _c_kwargs = dict(level=0, c="k")
    plot_level(axs_9[0], rel_sst_early, **_c_kwargs)
    plot_level(axs_9[1], rel_sst_early, **_c_kwargs)
    plot_level(axs_9[1], rel_sst_late, ls="--", **_c_kwargs)


    ## label
    axs_9[0].set_title("Early")
    axs_9[1].set_title("Late")
    axs_9[-1].set_xlabel("Longitude")
    axs_9[-1].set_xticks([140, 190, 240])

    _cb0 = fig_13.colorbar(_cp0, ax=axs_9[:-1], ticks=[0, 7.5, 15], label=r"mm / day")
    _cb2 = fig_13.colorbar(_cp2, ax=axs_9[-1], ticks=[-7.5, 7.5], label=r"mm / day")
    # cb1 = fig.colorbar(cp1, ax=axs[1], ticks=[26, 30, 34], label=r"$^{\circ}C$")
    # cb2 = fig.colorbar(cp2, ax=axs[2], ticks=[3, 4, 5], label=r"$^{\circ}C$")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $\tau_x$
    """)
    return


@app.cell
def _(clim_early, clim_late, plt, rel_sst_early, rel_sst_late, src):
    ## make hövmöllers
    fig_14, axs_10 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(
        cmap="cmo.balance", levels=src.utils.make_cb_range(0.08, 0.01), extend="both"
    )

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_10[0], clim_early["taux"], **_kwargs)

    ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_10[1], clim_late["taux"], **_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_10[2],
        (clim_late - clim_early)["taux"],
        **dict(_kwargs, levels=_kwargs["levels"] / 2)
    )

    ## contour zero levels
    _c_kwargs = dict(level=0, c="k")
    plot_level(axs_10[0], rel_sst_early, **_c_kwargs)
    plot_level(axs_10[1], rel_sst_early, **_c_kwargs)
    plot_level(axs_10[1], rel_sst_late, ls="--", **_c_kwargs)


    ## label
    axs_10[0].set_title("Early")
    axs_10[1].set_title("Late")
    axs_10[-1].set_xlabel("Longitude")
    axs_10[-1].set_xticks([140, 190, 240])

    _cb0 = fig_14.colorbar(_cp0, ax=axs_10[:-1], ticks=[-0.08, 0.08], label=r"Pa")
    _cb2 = fig_14.colorbar(_cp2, ax=axs_10[-1], ticks=[-0.04, 0.04], label=r"Pa")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### updated, combined fig
    """)
    return


@app.cell
def _(clim_early, clim_late, np, plt, rel_sst_early, rel_sst_late, save, src):
    ## shared args for plotting


    ## Set up plot
    fig_15, axs_11 = plt.subplots(2, 3, figsize=(9, 5), layout="constrained")

    ## plot relative SST
    LEV = src.utils.make_cb_range(4, 0.4)
    LEV_DIFF = src.utils.make_cb_range(2, 0.2)
    _plot_kwargs = dict(cmap="cmo.balance", extend="both", xticks=[210, 270])
    _cp0 = src.utils.plot_cycle_hov(axs_11[0, 0], rel_sst_early, levels=LEV, **_plot_kwargs)
    _cp1 = src.utils.plot_cycle_hov(axs_11[0, 1], rel_sst_late, levels=LEV, **_plot_kwargs)
    _cp2 = src.utils.plot_cycle_hov(
        axs_11[0, 2], rel_sst_late - rel_sst_early, levels=LEV_DIFF, **_plot_kwargs
    )

    ## plot precip
    LEV = np.arange(0, 13, 1)
    LEV_DIFF = src.utils.make_cb_range(6, 0.6)
    _plot_kwargs = dict(cmap="cmo.rain", extend="max", xticks=[210, 270])
    _cp0_ = src.utils.plot_cycle_hov(
        axs_11[-1, 0], 8.6e4 * clim_early["pr"], levels=LEV, **_plot_kwargs
    )
    _cp1_ = src.utils.plot_cycle_hov(
        axs_11[-1, 1], 8.6e4 * clim_late["pr"], levels=LEV, **_plot_kwargs
    )
    cp2_ = src.utils.plot_cycle_hov(
        axs_11[-1, 2],
        8.6e4 * (clim_late - clim_early)["pr"],
        levels=LEV_DIFF,
        **dict(_plot_kwargs, cmap="cmo.balance_r", extend="both"),
    )


    ## label
    axs_11[0, 0].set_title("1870")
    axs_11[0, 1].set_title("2080")
    axs_11[0, 2].set_title("Difference")
    for _ax in axs_11[-1]:
        _ax.set_xlabel("Longitude")

    for _ax in axs_11.flatten():
        _ax.set_xlim([140, 280])
        _ax.set_xticks([140, 210, 270])

    for _ax in axs_11[:, 1:].flatten():
        _ax.set_yticks([])
        _ax.set_ylabel(None)

    for _ax in axs_11[:-1].flatten():
        _ax.set_xticks([])
        _ax.set_xlabel(None)

    ## add colorbars
    _kwargs = dict(ticks=[-4, 0, 4], label=r"$^{\circ}\text{C}$")
    _cb1 = fig_15.colorbar(_cp1, ax=axs_11[0, 1], **dict(_kwargs, label=None))
    _cb2 = fig_15.colorbar(_cp2, ax=axs_11[0, 2], **dict(_kwargs, ticks=[-2, 0, 2]))

    _kwargs = dict(ticks=[0, 6, 12], label=r"mm day$^{-1}$")
    _cb1 = fig_15.colorbar(_cp1_, ax=axs_11[-1, 1], **dict(_kwargs, label=None))
    _cb2 = fig_15.colorbar(cp2_, ax=axs_11[-1, 2], **dict(_kwargs, ticks=[-6, 0, 6]))

    ## add labels
    ## kwargs for bounding box
    bbox = dict(boxstyle="round", facecolor="white", alpha=0)
    posnx, posny = (0.02, 0.05)
    labels = np.array([["a", "b", "c"], ["d", "e", "f"]])
    colors = np.array([["w", "w", "w"], ["w", "w", "k"]])

    for _ax, _label, c in zip(axs_11.flatten(), labels.flatten(), colors.flatten()):

        ## add label
        _ax.text(
            posnx,
            posny,
            s=f"({_label})",
            bbox=bbox,
            transform=_ax.transAxes,
            c=c,
        )

    ## save
    save(fig_15, fname="seasonal-clim")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SSH
    """)
    return


@app.cell
def _(clim_early, clim_late, plt, rel_sst_early, rel_sst_late, src):
    ## compute ssh anom
    get_anom = lambda x: x - x.mean("longitude")
    ssh_early = get_anom(clim_early["ssh"])
    ssh_late = get_anom(clim_late["ssh"])
    diff_1 = ssh_late - ssh_early

    ## make hövmöllers
    fig_16, axs_12 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.balance", levels=src.utils.make_cb_range(24, 3), extend="both")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_12[0], ssh_early, **_kwargs)

    ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_12[1], ssh_late, **_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_12[2], diff_1, **dict(_kwargs, levels=_kwargs["levels"] / 4)
    )

    ## contour zero levels
    _c_kwargs = dict(level=0, c="k")
    plot_level(axs_12[0], rel_sst_early, **_c_kwargs)
    plot_level(axs_12[1], rel_sst_early, **_c_kwargs)
    plot_level(axs_12[1], rel_sst_late, ls="--", **_c_kwargs)


    ## label
    axs_12[0].set_title("Early")
    axs_12[1].set_title("Late")
    axs_12[-1].set_xlabel("Longitude")
    axs_12[-1].set_xticks([140, 190, 240])

    _cb0 = fig_16.colorbar(_cp0, ax=axs_12[:-1], ticks=[-24, 24], label=r"cm")
    _cb2 = fig_16.colorbar(_cp2, ax=axs_12[-1], ticks=[-6, 6], label=r"cm")
    # cb1 = fig.colorbar(cp1, ax=axs[1], ticks=[26, 30, 34], label=r"$^{\circ}C$")
    # cb2 = fig.colorbar(cp2, ax=axs[2], ticks=[3, 4, 5], label=r"$^{\circ}C$")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $\frac{d}{dx}$(SSH)
    """)
    return


@app.cell
def _(clim_early, clim_late, plt, rel_sst_early, rel_sst_late, src):
    ## compute ssh anom
    ssh_grad_early = clim_early["ssh"].differentiate("longitude")
    ssh_grad_late = clim_late["ssh"].differentiate("longitude")
    diff_2 = ssh_grad_late - ssh_grad_early

    ## make hövmöllers
    fig_17, axs_13 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.balance", levels=src.utils.make_cb_range(1, 0.1), extend="both")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_13[0], ssh_grad_early, **_kwargs)

    ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_13[1], ssh_grad_late, **_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_13[2], diff_2, **dict(_kwargs, levels=_kwargs["levels"] / 2)
    )

    ## contour zero levels
    _c_kwargs = dict(level=0, c="k")
    plot_level(axs_13[0], rel_sst_early, **_c_kwargs)
    plot_level(axs_13[1], rel_sst_early, **_c_kwargs)
    plot_level(axs_13[1], rel_sst_late, ls="--", **_c_kwargs)


    ## label
    axs_13[0].set_title("Early")
    axs_13[1].set_title("Late")
    axs_13[-1].set_xlabel("Longitude")
    axs_13[-1].set_xticks([140, 190, 240])

    _cb0 = fig_17.colorbar(_cp0, ax=axs_13[:-1], ticks=[-1, 1], label=r"cm / $^{\circ}$lon")
    _cb2 = fig_17.colorbar(_cp2, ax=axs_13[-1], ticks=[-0.5, 0.5], label=r"cm / $^{\circ}$lon")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Mixed layer depth
    """)
    return


@app.cell
def _(clim_diff, clim_early):
    # ## compute clims
    # kwargs = dict(components=components, varname="mld")
    # mld_clim_early = recon_clim(forced.isel(time=slice(None, 360)), **kwargs)
    # mld_clim_late = recon_clim(forced.isel(time=slice(-360, None)), **kwargs)

    # ## get difference
    # mld_clim_diff = mld_clim_late - mld_clim_early

    ## get fractional chnage
    mld_clim_diff_pct1 = 100 * (clim_diff / clim_early)["mld"]
    mld_clim_diff_pct2 = 100 * (-clim_diff / (clim_early + clim_diff))["mld"]
    return (mld_clim_diff_pct2,)


@app.cell
def _(clim_early, clim_late, mld_clim_diff_pct2, np, plt, src):
    ## make hövmöllers
    fig_18, axs_14 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.amp", levels=np.arange(0, 65, 5), extend="max")
    _cb_kwargs = dict(ticks=[0, 30, 60], label=r"m")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_14[0], clim_early["mld"], **_kwargs)
    _cb0 = fig_18.colorbar(_cp0, ax=axs_14[0], **_cb_kwargs)

    ## plot late
    _kwargs["levels"] = _kwargs["levels"] + 3
    _cp1 = src.utils.plot_cycle_hov(axs_14[1], clim_late["mld"], **_kwargs)
    _cb1 = fig_18.colorbar(_cp1, ax=axs_14[1], **_cb_kwargs)

    ## plot % pct change
    _cp2 = src.utils.plot_cycle_hov(
        axs_14[2],
        mld_clim_diff_pct2,
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(20, 2),
        extend="both",
    )
    _cb2 = fig_18.colorbar(_cp2, ax=axs_14[2], ticks=[-80, 0, 80], label="% change")

    ## label
    axs_14[0].set_title("Early")
    axs_14[1].set_title("Late")
    axs_14[2].set_title("Difference")
    axs_14[-1].set_xlabel("Longitude")
    axs_14[-1].set_xticks([140, 190, 240])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $Z_{20}$
    """)
    return


@app.cell
def _(clim_diff, clim_early, clim_late, np, plt, src):
    z20_clim_diff_pct2 = 100 * (-clim_diff / (clim_early + clim_diff))["z20"]

    ## make hövmöllers
    fig_19, axs_15 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.amp", levels=np.arange(40, 210, 10), extend="max")
    _cb_kwargs = dict(ticks=[40, 200, 120], label=r"m")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_15[0], clim_early["z20"], **_kwargs)
    _cb0 = fig_19.colorbar(_cp0, ax=axs_15[0], **_cb_kwargs)

    ## plot late
    _kwargs["levels"] = _kwargs["levels"] + 3
    _cp1 = src.utils.plot_cycle_hov(axs_15[1], clim_late["z20"], **_kwargs)
    _cb1 = fig_19.colorbar(_cp1, ax=axs_15[1], **_cb_kwargs)

    ## plot % pct change
    _cp2 = src.utils.plot_cycle_hov(
        axs_15[2],
        z20_clim_diff_pct2,
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(10, 1),
        extend="both",
    )
    _cb2 = fig_19.colorbar(_cp2, ax=axs_15[2], ticks=[-50, 0, 50], label="% change")

    ## label
    axs_15[0].set_title("Early")
    axs_15[1].set_title("Late")
    axs_15[2].set_title("Difference")
    axs_15[-1].set_xlabel("Longitude")
    axs_15[-1].set_xticks([140, 190, 240])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### zonal velocity
    """)
    return


@app.cell
def _(clim_diff, clim_early, clim_late, plt, src):
    ## make hövmöllers
    fig_20, axs_16 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(
        cmap="cmo.balance", levels=src.utils.make_cb_range(2e6, 2e5), extend="both"
    )

    _cb_kwargs = dict(ticks=[-2e6, 2e6], label=r"m (month)$^{-1}$")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_16[0], clim_early["uvel"], **_kwargs)
    _cb0 = fig_20.colorbar(_cp0, ax=axs_16[0], **_cb_kwargs)

    ## plot late
    _kwargs["levels"] = _kwargs["levels"] + 3
    _cp1 = src.utils.plot_cycle_hov(axs_16[1], clim_late["uvel"], **_kwargs)
    _cb1 = fig_20.colorbar(_cp1, ax=axs_16[1], **_cb_kwargs)

    ## plot % pct change
    _cp2 = src.utils.plot_cycle_hov(
        axs_16[2],
        clim_diff["uvel"],
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(1e6, 1e5),
        extend="both",
    )
    _cb2 = fig_20.colorbar(_cp2, ax=axs_16[2], ticks=[-1e6, 1e6], label="Difference")

    ## label
    axs_16[0].set_title("Early")
    axs_16[1].set_title("Late")
    axs_16[2].set_title("Difference")
    axs_16[-1].set_xlabel("Longitude")
    axs_16[-1].set_xticks([140, 190, 240])
    for _ax in axs_16:
        _ax.axhline(7, ls="--", c="k", lw=1)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### spatial pattern for precip/relative SST
    """)
    return


@app.cell
def _(src):
    def recon_sst_pr_clim(data, trop_sst):
        """reconstruct sst and precip climatology,
        and compute relative SST"""

        ## specify vars to reconstruct
        recon_vars = [
            "sst",
            "pr",
            "sst_comp",
            "pr_comp",
            "taux",
            "taux_comp",
            "tauy",
            "tauy_comp",
        ]

        ## reconstruct
        clim = src.utils.reconstruct_clim(data[recon_vars])

        ## compute mean tropical SST
        trop_sst_clim = trop_sst.groupby("time.month").mean(["time", "member"])

        ## get relative SST
        clim["sst_rel"] = clim["sst"] - trop_sst_clim

        ## swap sign of tauy (currently it represents stress on atmosphere...)
        clim["tauy"] = -clim["tauy"]

        return clim

    return (recon_sst_pr_clim,)


@app.cell
def _(forced, recon_sst_pr_clim, t_early, t_late, trop_sst):
    ## reconstruct
    spatial_clim_early = recon_sst_pr_clim(
        forced.sel(t_early),
        trop_sst["trop_sst_05"].sel(t_early),
    )
    spatial_clim_late = recon_sst_pr_clim(
        forced.sel(t_late),
        trop_sst["trop_sst_05"].sel(t_late),
    )
    return spatial_clim_early, spatial_clim_late


@app.cell
def _(ccrs, np, src):
    def plot_sst_rel(ax, data):
        """plot relative sst on ax object"""

        cp = ax.contourf(
            data.longitude,
            data.latitude,
            data["sst_rel"],
            cmap="cmo.balance",
            transform=ccrs.PlateCarree(),
            levels=src.utils.make_cb_range(10, 1),
            extend="both",
        )

        return cp


    def plot_contour(ax, data, lev, lw, c="k"):
        """plot relative sst on ax object"""

        cp = ax.contour(
            data.longitude,
            data.latitude,
            data["sst_rel"],
            transform=ccrs.PlateCarree(),
            levels=[lev],
            linewidths=lw,
            colors=c,
        )

        return cp


    def plot_pr(ax, data):
        """plot precip on ax object"""

        ## convert from kg / m / s to mm/day
        ## 1e-3 (m3 / kg) * 1e3 (mm / m) * 8.6e4 (s / day)
        factor = 8.6e4

        cp = ax.contourf(
            data.longitude,
            data.latitude,
            factor * data["pr"],
            cmap="cmo.rain",
            transform=ccrs.PlateCarree(),
            levels=np.arange(0, 21, 3),
            extend="max",
        )

        return cp

    return plot_contour, plot_pr, plot_sst_rel


@app.cell
def _(
    ccrs,
    plot_contour,
    plot_pr,
    plot_sst_rel,
    plt,
    spatial_clim_early,
    spatial_clim_late,
    src,
):
    ## specify which month to look at
    _sel = lambda x: x.sel(month=12)
    # sel = lambda x :x.mean("month")

    fig_21 = plt.figure(figsize=(12, 3.5), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_17 = src.utils.subplots_with_proj(fig_21, nrows=3, ncols=2, format_func=_format_func)

    ## plot early
    _cp0 = plot_sst_rel(axs_17[0, 0], _sel(spatial_clim_early))
    _cp1 = plot_pr(axs_17[0, 1], _sel(spatial_clim_early))

    ## plot late
    plot_sst_rel(axs_17[1, 0], _sel(spatial_clim_late))
    plot_pr(axs_17[1, 1], _sel(spatial_clim_late))

    ## plot diff
    _cp0_ = axs_17[2, 0].contourf(
        spatial_clim_early.longitude,
        spatial_clim_early.latitude,
        _sel((spatial_clim_late - spatial_clim_early)["sst_rel"]),
        cmap="cmo.balance",
        transform=ccrs.PlateCarree(),
        levels=src.utils.make_cb_range(0.5, 0.05),
        extend="both",
    )

    _cp1_ = axs_17[2, 1].contourf(
        spatial_clim_early.longitude,
        spatial_clim_early.latitude,
        8.6e4 * _sel((spatial_clim_late - spatial_clim_early)["pr"]),
        cmap="cmo.balance_r",
        transform=ccrs.PlateCarree(),
        levels=src.utils.make_cb_range(1.2, 0.12),
        extend="both",
    )

    ## plot convective bounds
    for _ax in axs_17[0, :]:
        plot_contour(_ax, _sel(spatial_clim_early), lev=0, lw=2)
        plot_contour(_ax, _sel(spatial_clim_early), lev=-1, lw=1)

    for _ax in axs_17[1, :]:
        plot_contour(_ax, _sel(spatial_clim_late), lev=0, lw=2)
        plot_contour(_ax, _sel(spatial_clim_late), lev=-1, lw=1)

    for _ax in axs_17[2]:
        plot_contour(_ax, _sel(spatial_clim_early), lev=-1, lw=1, c="gray")
        plot_contour(_ax, _sel(spatial_clim_early), lev=-1, lw=1)

    for _ax in axs_17.flatten():
        _ax.axhline(0, c="magenta", lw=1, alpha=0.5)

    ## colorbars
    fig_21.colorbar(_cp0, ax=axs_17[:2, 0], ticks=[-10, 0, 10])
    fig_21.colorbar(_cp1, ax=axs_17[:2, 1], ticks=[0, 18, 9])
    fig_21.colorbar(_cp0_, ax=axs_17[2, 0], ticks=[-2, 0, 2])
    fig_21.colorbar(_cp1_, ax=axs_17[2, 1], ticks=[-12, 0, 12])

    plt.gca()
    return


@app.cell
def _(Th_total, plt, src):
    # plt.plot(src.utils.sel_month(Th["T_3_rel"].mean("member"),6))
    # plt.plot(src.utils.sel_month(Th["T_3_rel"].mean("member"),6).rolling({"time":50}).mean())
    # plt.plot(src.utils.sel_month(Th["T_3_rel"].mean("member"),7).rolling({"time":50}).mean())
    # for m in np.arange(1,13).astype(int):
    #     print(m)

    MONTH = 7
    dTdx = (Th_total["T_4"] - Th_total["T_3"]).mean("member")
    dTdx = src.utils.sel_month(dTdx, MONTH).rolling({"time": 50}).mean()

    fig_22, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(dTdx.time.dt.year, dTdx)
    _ax.set_xlim([None, 2010])
    _ax.set_ylim([3.4, None])
    plt.gca()
    return


@app.cell
def _(Th_var_bymonth, plt):
    plt.plot(Th_var_bymonth["T_3"].sel(month=8))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Spatial pattern for SST
    """)
    return


@app.cell
def _(ccrs, np):
    def plot_sst(ax, data):
        """plot relative sst on ax object"""

        cp = ax.contourf(
            data.longitude,
            data.latitude,
            data["sst"],
            cmap="cmo.thermal",
            transform=ccrs.PlateCarree(),
            levels=np.arange(22, 34),
            extend="both",
        )

        return cp

    return (plot_sst,)


@app.cell
def _(ccrs, plot_sst, plt, spatial_clim_early, spatial_clim_late, src):
    ## specify which month to look at
    _sel = lambda x: x.sel(month=9)
    # sel = lambda x :x.mean("month")

    fig_23 = plt.figure(figsize=(12, 3.5), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_18 = src.utils.subplots_with_proj(fig_23, nrows=3, ncols=1, format_func=_format_func)

    ## plot early
    _cp0 = plot_sst(axs_18[0, 0], _sel(spatial_clim_early))
    _cp1 = plot_sst(axs_18[1, 0], _sel(spatial_clim_late))

    for _ax in axs_18.flatten():
        _ax.axhline(0, c="magenta", lw=1, alpha=0.5)

    ## plot difference
    _cp = _ax.contourf(
        spatial_clim_late.longitude,
        spatial_clim_late.latitude,
        _sel(spatial_clim_late - spatial_clim_early)["sst"],
        cmap="cmo.balance",
        transform=ccrs.PlateCarree(),
        levels=src.utils.make_cb_range(1, 0.1),
        extend="both",
    )

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### spatial pattern for $\tau_x$
    """)
    return


@app.cell
def _(ccrs, clim_early, plt, spatial_clim_early, spatial_clim_late, src):
    ## specify which month to look at
    _sel = lambda x: x.sel(month=2)

    fig_24 = plt.figure(figsize=(6, 3.5), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_19 = src.utils.subplots_with_proj(fig_24, nrows=3, ncols=1, format_func=_format_func)

    ## plot diff
    for _ax, clim_ in zip(axs_19[:2, 0], [spatial_clim_early, spatial_clim_late]):
        _cp = _ax.contourf(
            clim_.longitude,
            clim_.latitude,
            _sel(clim_)["taux"],
            cmap="cmo.balance",
            transform=ccrs.PlateCarree(),
            levels=src.utils.make_cb_range(0.12, 0.012),
            extend="both",
        )
    _cp_diff = axs_19[-1, 0].contourf(
        clim_.longitude,
        clim_.latitude,
        _sel(spatial_clim_late - spatial_clim_early)["taux"],
        cmap="cmo.balance",
        transform=ccrs.PlateCarree(),
        levels=src.utils.make_cb_range(0.06, 0.006),
        extend="both",
    )


    ## kwargs for contour plots
    _c_kwargs = dict(
        level=6, lw=1.5, y_coord="latitude", transform=ccrs.PlateCarree(), c="k"
    )
    plot_level(ax=axs_19[0, 0], data=8.6e4 * _sel(spatial_clim_early["pr"]), **_c_kwargs)
    plot_level(ax=axs_19[1, 0], data=8.6e4 * _sel(spatial_clim_late["pr"]), ls="--", **_c_kwargs)
    plot_level(ax=axs_19[2, 0], data=8.6e4 * _sel(spatial_clim_early["pr"]), **_c_kwargs)
    plot_level(ax=axs_19[2, 0], data=8.6e4 * _sel(spatial_clim_late["pr"]), ls="--", **_c_kwargs)

    for _ax in axs_19.flatten():
        _ax.axhline(5, c="magenta", lw=1, alpha=0.5)
        _ax.axhline(-5, c="magenta", lw=1, alpha=0.5)

    plt.show()


    ## plot taux/pr along equator
    sel_eq = lambda x: _sel(x).sel(latitude=slice(-5, 5)).mean("latitude")

    for _v in ["taux", "pr"]:
        fig_24, _ax = plt.subplots(figsize=(2.5, 2))
        _ax.plot(clim_early.longitude, sel_eq(spatial_clim_early[_v]), label="early")
        _ax.plot(clim_early.longitude, sel_eq(spatial_clim_late[_v]), label="late")
        _ax.set_xlim([140, 280])
        _ax.axhline(0, ls="--", c="k", lw=0.8)
        _ax.legend(prop=dict(size=6))
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Subsurface
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mixed layer depth
    """)
    return


@app.cell
def _():
    # def get_eq_mld(data):
    #     """Get equatorial mixed layer depth"""

    #     data_ = data.sel(latitude=slice(-1.5, 1.5)).mean("latitude")

    #     return data_.sel(longitude=slice(140, 280))


    # def recon_eq_mld(t_bnds):
    #     """reconstruct equatorial MLD"""
    #     return src.utils.reconstruct_fn(
    #         scores=forced["mld"].isel(time=slice(*t_bnds)).groupby("time.month").mean(),
    #         components=components["mld"],
    #         fn=get_eq_mld,
    #     )


    # eq_mld_early = recon_eq_mld((None, 360))
    # eq_mld_late = recon_eq_mld((-360, None))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot vertical velocity and temperature grad
    """)
    return


@app.cell
def _(clim_early, clim_late, format_subsurf_axs, np, plt, src):
    ## specify which period/month to plot
    # sel = lambda x: x.mean("month")
    _sel = lambda x: x.sel(month=12)

    fig_25, axs_20 = plt.subplots(1, 3, figsize=(8, 2.5), layout="constrained")

    for _ax, _clim in zip(axs_20[:2], [clim_early, clim_late]):

        ## temperature
        _ax.contourf(
            _clim["T"].longitude,
            _clim["T"].z_t,
            _sel(_clim["T"]),
            cmap="cmo.thermal",
            levels=np.arange(10, 34, 2),
            extend="both",
        )

        ## vertical velocity
        _ax.contour(
            _clim["w"].longitude,
            _clim["w"].z_t,
            _sel(_clim["w"]),
            colors="k",
            levels=src.utils.make_cb_range(70, 7),
            extend="both",
            linewidths=1,
        )


    ## plot difference
    diff_3 = clim_late - clim_early

    ## temperature
    axs_20[2].contourf(
        _clim["T"].longitude,
        _clim["T"].z_t,
        _sel(diff_3["T"]),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(1, 0.1),
        extend="both",
    )

    ## vertical velocity
    axs_20[2].contour(
        _clim["w"].longitude,
        _clim["w"].z_t,
        _sel(diff_3["w"]),
        colors="k",
        levels=src.utils.make_cb_range(10, 1),
        extend="both",
        linewidths=1,
    )

    ## plot MLD
    _lon = clim_early.longitude
    axs_20[0].plot(_lon, _sel(clim_early["mld"]), c="w")
    axs_20[1].plot(_lon, _sel(clim_late["mld"]), c="w", ls="--")
    axs_20[2].plot(_lon, _sel(clim_early["mld"]), c="w")
    axs_20[2].plot(_lon, _sel(clim_late["mld"]), c="w", ls="--")

    ## label
    format_subsurf_axs(axs_20)
    for _ax in axs_20:
        _ax.axhline(70, c="magenta", ls="--")


    plt.gca()
    return (diff_3,)


@app.cell
def _(clim_early, clim_late, diff_3, format_subsurf_axs, get_dx, np, plt, src):
    ## specify which period/month to plot
    _sel = lambda x: x.sel(month=[5, 6, 7]).mean("month")


    ## get aspect ratio (for scaling arrows)
    _dz = 300  # units: m
    _dx = get_dx(lat_deg=0, dlon_deg=150)
    _aspect = _dx / _dz

    fig_26, axs_21 = plt.subplots(1, 3, figsize=(12, 3), layout="constrained")

    for _ax, _clim in zip(axs_21[:2], [clim_early, clim_late]):

        ## temperature
        _cp = _ax.contourf(
            _clim["T"].longitude,
            _clim["T"].z_t,
            _sel(_clim["T"]),
            cmap="cmo.thermal",
            levels=np.arange(10, 34, 2),
            extend="both",
        )

        ## u and w
        _ax.quiver(
            _clim.longitude.values[::4],
            _clim.z_t.values[::2],
            _sel(_clim.u).values[::2, ::4],
            _sel(_clim.w).values[::2, ::4] * _aspect,
            pivot="middle",
            alpha=0.7,
            scale=3e7,
        )

    ## temperature
    _cp_diff = axs_21[2].contourf(
        _clim["T"].longitude,
        _clim["T"].z_t,
        _sel(diff_3["T"]),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(5, 0.5),
        extend="both",
    )

    ## vertical velocity
    ## u and w
    axs_21[2].quiver(
        diff_3.longitude.values[::4],
        diff_3.z_t.values[::2],
        _sel(diff_3.u).values[::2, ::4],
        _sel(diff_3.w).values[::2, ::4] * _aspect,
        pivot="middle",
        alpha=0.7,
        scale=1.5e7,
    )

    ## plot MLD/thermocline
    _kwargs = dict(c="w", alpha=0.8)

    ## plot MLD
    _lon = clim_early.longitude
    axs_21[0].plot(_lon, _sel(clim_early["mld"]), **_kwargs)
    axs_21[1].plot(_lon, _sel(clim_late["mld"]), ls="--", **_kwargs)
    axs_21[2].plot(_lon, _sel(clim_early["mld"]), **_kwargs)
    axs_21[2].plot(_lon, _sel(clim_late["mld"]), ls="--", **_kwargs)

    ## plot thermocline
    sel_H = lambda x: _sel(src.utils.get_H_int(x["T"], thresh=0.04))
    axs_21[0].plot(_lon, sel_H(clim_early), **_kwargs)
    axs_21[1].plot(_lon, sel_H(clim_late), ls="--", **_kwargs)
    axs_21[2].plot(_lon, sel_H(clim_early), **_kwargs)
    axs_21[2].plot(_lon, sel_H(clim_late), ls="--", **_kwargs)

    ## plot 20deg isotherm
    axs_21[0].contour(_lon, clim_early.z_t, _sel(clim_early["T"]), levels=[20], colors="red")
    axs_21[1].contour(_lon, clim_early.z_t, _sel(clim_late["T"]), levels=[20], colors="red")

    ## label/format
    fig_26.colorbar(_cp, ax=axs_21[1], ticks=[10, 32], label=r"$^{\circ}$C")
    fig_26.colorbar(_cp_diff, ax=axs_21[2], ticks=[-5, 0, 5], label=r"$^{\circ}$C")
    format_subsurf_axs(axs_21)
    for _ax in axs_21:
        _ax.set_xlim([145, 278])
        _ax.set_xticks([145, 190, 240, 275])


    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hovmoller
    """)
    return


@app.cell
def _(clim_early, clim_late, get_diags):
    ## get diagnostics
    diags_early = get_diags(clim_early)
    diags_late = get_diags(clim_late)
    diags_diff = diags_late - diags_early
    return diags_diff, diags_early, diags_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Stratification
    """)
    return


@app.cell
def _(diags_diff, diags_early, diags_late, np, plt, src):
    ## make hövmöllers
    fig_27, axs_22 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.amp", levels=np.arange(10, 19, 1), extend="both")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_22[0], diags_early["dTdz"], **_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_22[1], diags_late["dTdz"], **_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_22[2],
        diags_diff["dTdz"],
        cmap="cmo.amp",
        levels=np.arange(0, 6.6, 0.6),
        extend="both",
    )

    ## label
    axs_22[0].set_title("Early")
    axs_22[1].set_title("Late")
    axs_22[2].set_title("Difference")
    axs_22[-1].set_xlabel("Longitude")
    axs_22[-1].set_xticks([140, 190, 240])

    _cb0 = fig_27.colorbar(_cp0, ax=axs_22[0], ticks=[10, 14, 18], label=r"$^{\circ}C$")
    _cb1 = fig_27.colorbar(_cp1, ax=axs_22[1], ticks=[10, 14, 18], label=r"$^{\circ}C$")
    _cb2 = fig_27.colorbar(_cp2, ax=axs_22[2], ticks=[0, 3, 6], label=r"$^{\circ}C$")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Upwelling
    """)
    return


@app.cell
def _(diags_diff, diags_early, diags_late, plt, src):
    ## make hövmöllers
    fig_28, axs_23 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.balance", levels=src.utils.make_cb_range(40, 4), extend="both")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_23[0], diags_early["w_int"], **_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_23[1], diags_late["w_int"], **_kwargs)

    # ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_23[2],
        diags_diff["w_int"],
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(20, 4),
        extend="both",
    )

    ## label
    axs_23[0].set_title("Early")
    axs_23[1].set_title("Late")
    axs_23[2].set_title("Difference")
    axs_23[-1].set_xlabel("Longitude")
    axs_23[-1].set_xticks([140, 190, 240])

    _cb0 = fig_28.colorbar(_cp0, ax=axs_23[0], ticks=[-40, 0, 40], label=r"$m/mo$")
    _cb1 = fig_28.colorbar(_cp1, ax=axs_23[1], ticks=[-40, 0, 40], label=r"$m/mo$")
    _cb2 = fig_28.colorbar(_cp2, ax=axs_23[2], ticks=[-20, 0, 20], label=r"$m/mo$")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### vertical temperature flux
    \begin{align}
        \overline{w}/\overline{H}
    \end{align}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Functions to use in computation
    """)
    return


@app.cell
def _(pd, xr):
    def get_w_sub(data, mld=None):
        """get velocity at base of mixed layer"""

        ## set mld
        if mld is None:
            mld = data["mld"]

        ## get w along section
        w_sub = []
        for i, lon in enumerate(data.longitude):
            w_ = data["w"].sel(longitude=lon)
            mld_ = mld.sel(longitude=lon)
            w_sub.append(w_.interp(z_t=mld_))

        return xr.concat(w_sub, dim=pd.Index(data.longitude.values, name="longitude"))


    def get_v_timescale(data, mld=None):
        """get inverse timescale for mean upwelling (w/H)"""

        ## get w at base of ML
        w_sub = get_w_sub(data, mld)

        ## set negative values to zero
        w_sub = w_sub.where(w_sub > 0, other=0)

        ## compute
        return w_sub / data["mld"]

    return get_v_timescale, get_w_sub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Compute
    """)
    return


@app.cell
def _(clim_early, clim_late, get_v_timescale, get_w_sub):
    ## specify MLD
    # mld = 70 * xr.ones_like(clim_early["mld"])
    mld = None

    ## get vertical velocities
    w_sub_early = get_w_sub(clim_early, mld)
    w_sub_late = get_w_sub(clim_late, mld)

    ## get (inverse) upwelling timescale
    t_inv_early = get_v_timescale(clim_early, mld)
    t_inv_late = get_v_timescale(clim_late, mld)

    ## get pct_change
    t_inv_diff = t_inv_late - t_inv_early
    t_inv_diff_pct = 100 * t_inv_diff / t_inv_early
    return t_inv_diff, t_inv_early, t_inv_late, w_sub_early, w_sub_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Plot for single month
    """)
    return


@app.cell
def _(
    clim_early,
    clim_late,
    plt,
    t_inv_early,
    t_inv_late,
    w_sub_early,
    w_sub_late,
):
    ## select
    _sel = lambda x: x.mean("month")

    fig_29, axs_24 = plt.subplots(1, 3, figsize=(5, 1.5), layout="constrained")

    ## get longitude for plotting
    _lon = clim_early.longitude

    ## plot data
    axs_24[0].plot(_lon, _sel(clim_early["mld"]))
    axs_24[0].plot(_lon, _sel(clim_late["mld"]))

    axs_24[1].plot(_lon, _sel(w_sub_early))
    axs_24[1].plot(_lon, _sel(w_sub_late))

    axs_24[2].plot(_lon, _sel(t_inv_early))
    axs_24[2].plot(_lon, _sel(t_inv_late))

    axs_24[0].set_ylim(axs_24[0].get_ylim()[::-1])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hovmoller version of plot
    """)
    return


@app.cell
def _(np, plt, src, t_inv_diff, t_inv_early, t_inv_late):
    ## make hövmöllers
    fig_30, axs_25 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.amp", levels=np.arange(0, 1.65, 0.15), extend="max")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_25[0], t_inv_early, **_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_25[1], t_inv_late, **_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_25[2],
        t_inv_diff,
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(0.75, 0.075),
        extend="both",
    )

    ## label
    axs_25[0].set_title("Early")
    axs_25[1].set_title("Late")
    axs_25[2].set_title("Difference")
    axs_25[-1].set_xlabel("Longitude")
    axs_25[-1].set_xticks([140, 190, 240])

    _cb0 = fig_30.colorbar(_cp0, ax=axs_25[0], ticks=[0, 1.5], label=r"month$^{-1}$")
    _cb1 = fig_30.colorbar(_cp1, ax=axs_25[1], ticks=[0, 1.5], label=r"month$^{-1}$")
    _cb2 = fig_30.colorbar(_cp2, ax=axs_25[2], ticks=[-0.75, 0.75], label=r"month$^{-1}$")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at temperature gradient more carefully
    """)
    return


@app.cell
def _(clim_early, clim_late, get_dTdz_sub, xr):
    ## should we use fixed MLD?
    USE_FIXED_MLD = True

    if USE_FIXED_MLD:
        H0 = 50 * xr.ones_like(clim_early["mld"])
        dTdz_early = get_dTdz_sub(Tsub=clim_early["T"], mld=H0)
        dTdz_late = get_dTdz_sub(Tsub=clim_late["T"], mld=H0)

    else:
        dTdz_early = get_dTdz_sub(Tsub=clim_early["T"], mld=clim_early["mld"])
        dTdz_late = get_dTdz_sub(Tsub=clim_late["T"], mld=clim_late["mld"])
    return dTdz_early, dTdz_late


@app.cell
def _(dTdz_early, dTdz_late, np, plt, src):
    ## make hövmöllers
    fig_31, axs_26 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.amp", levels=np.arange(0, 0.15, 0.015), extend="max")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_26[0], dTdz_early, **_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_26[1], dTdz_late, **_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_26[2],
        dTdz_late - dTdz_early,
        cmap="cmo.balance",
        # levels=src.utils.make_cb_range(0.075, 0.0075),
        levels=src.utils.make_cb_range(0.01, 0.001),
        extend="both",
    )

    ## label
    axs_26[0].set_title("Early")
    axs_26[1].set_title("Late")
    axs_26[2].set_title("Difference")
    axs_26[-1].set_xlabel("Longitude")
    axs_26[-1].set_xticks([140, 190, 240])

    _cb0 = fig_31.colorbar(_cp0, ax=axs_26[0], ticks=[0, 0.3], label=r"$K~m^{-1}$")
    _cb1 = fig_31.colorbar(_cp1, ax=axs_26[1], ticks=[0, 0.3], label=r"$K~m^{-1}$")
    _cb2 = fig_31.colorbar(_cp2, ax=axs_26[2], ticks=[-0.15, 0.15], label=r"$K~m^{-1}$")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Change over time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute indices
    """)
    return


@app.cell
def _(forced, get_dTdz_idx, get_w_int_idx, reconstruct_fn, src, xr):
    ## compute indices
    w_idx = reconstruct_fn(forced["w_comp"], forced["w"], get_w_int_idx)
    dTdz_idx = reconstruct_fn(forced["T_comp"], forced["T"], get_dTdz_idx)

    ## merge
    idxs = xr.merge([w_idx.rename("w"), dTdz_idx.rename("dTdz")])

    ## unstack month and year, and get percent change
    idxs = src.utils.unstack_month_and_year(idxs)
    idxs = src.utils.get_rolling_avg(idxs, n=15, dim="year")
    delta_idxs = idxs - idxs.isel(year=0)
    return (delta_idxs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hovmoller indices
    """)
    return


@app.cell
def _(delta_idxs, plot_hov2, plt):
    ## setup plot
    fig_32, axs_27 = plt.subplots(1, 2, figsize=(2, 3), layout="constrained")

    ## plot mean
    plot_hov2(axs_27[0], delta_idxs["dTdz"].T, amp=4, fig=fig_32, axs=axs_27)
    plot_hov2(axs_27[1], delta_idxs["w"].T, amp=12, fig=fig_32, axs=axs_27)

    ## label
    _kwargs = dict(size=12)
    axs_27[0].set_title(r"$\Delta \frac{\partial T}{\partial z}$", **_kwargs)
    axs_27[1].set_title(r"$\Delta w$", **_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Change in variance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Equatorial region
    """)
    return


@app.cell
def _(anom, eq_avg, get_var_for_periods, np):
    ## get variance by period
    # period_centers = np.array([1868, 2082])
    period_centers = np.array([1868, 1990, 2082])

    ## Get equatorial strip variance
    _kwargs = dict(
        periods=period_centers,
        n=15,
        fn=eq_avg,
    )
    var_by_period_eq = get_var_for_periods(
        data=anom[["sst", "sst_comp", "z20", "z20_comp"]],
        **_kwargs,
    )

    ## get full variance
    _kwargs = dict(_kwargs, fn=None, periods=period_centers, by_month=True)
    var_by_period = get_var_for_periods(
        data=anom[["sst", "sst_comp", "z20", "z20_comp", "T", "T_comp"]], **_kwargs
    )
    return var_by_period, var_by_period_eq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Spatial (Hovmoller)
    """)
    return


@app.cell
def _(np, plt, src, var_by_period_eq):
    ## get plot data
    _baseline = var_by_period_eq["sst"].isel(period=1)
    _future = var_by_period_eq["sst"].isel(period=2)
    change = _future - _baseline

    ## shared args for plotting
    _plot_kwargs = dict(cmap="cmo.amp", extend="max")

    ## Set up plot
    fig_33, axs_28 = plt.subplots(3, 1, figsize=(4, 6), layout="constrained")

    ## make hövmöllers
    _cp0 = src.utils.plot_cycle_hov(
        axs_28[0], _baseline, levels=np.arange(0, 3.3, 0.3), **_plot_kwargs
    )
    _cp1 = src.utils.plot_cycle_hov(
        axs_28[1], _future, levels=np.arange(0, 3.3, 0.3), **_plot_kwargs
    )
    _cp2 = src.utils.plot_cycle_hov(
        axs_28[2],
        change,
        levels=src.utils.make_cb_range(1.5, 0.15),
        cmap="cmo.balance",
        extend="both",
    )

    ## label
    axs_28[0].set_title("Baseline")
    axs_28[1].set_title("Future")
    axs_28[2].set_title("Difference")
    axs_28[-1].set_xlabel("Longitude")
    axs_28[-1].set_xticks([140, 190, 240])

    ## add colorbars
    _kwargs = dict(ticks=[0, 1, 2, 3], label=r"$^{\circ}\text{C}^2$")
    _cb0 = fig_33.colorbar(_cp0, ax=axs_28[0], **_kwargs)
    _cb1 = fig_33.colorbar(_cp1, ax=axs_28[1], **_kwargs)
    _cb2 = fig_33.colorbar(_cp2, ax=axs_28[2], **dict(_kwargs, ticks=[-1.5, 0, 1.5]))

    plt.gca()
    return (change,)


@app.cell
def _(np, plt, src, var_by_period_eq):
    ## get plot data
    _baseline = var_by_period_eq["z20"].isel(period=0)
    _future = var_by_period_eq["z20"].isel(period=1)
    change_z20 = _future - _baseline

    ## shared args for plotting
    _plot_kwargs = dict(cmap="cmo.amp", extend="max")

    ## Set up plot
    fig_34, axs_29 = plt.subplots(3, 1, figsize=(4, 6), layout="constrained")

    ## make hövmöllers
    _cp0 = src.utils.plot_cycle_hov(
        axs_29[0], _baseline, levels=np.arange(0, 500, 50), **_plot_kwargs
    )
    _cp1 = src.utils.plot_cycle_hov(
        axs_29[1], _future, levels=np.arange(0, 500, 50), **_plot_kwargs
    )
    _cp2 = src.utils.plot_cycle_hov(
        axs_29[2],
        change_z20,
        levels=src.utils.make_cb_range(250, 25),
        cmap="cmo.balance",
        extend="both",
    )

    ## label
    axs_29[0].set_title("Baseline")
    axs_29[1].set_title("Future")
    axs_29[2].set_title("Difference")
    axs_29[-1].set_xlabel("Longitude")
    axs_29[-1].set_xticks([140, 190, 240])

    ## add colorbars
    _kwargs = dict(ticks=[0, 1, 2, 3], label=r"$^{\circ}\text{C}^2$")
    _cb0 = fig_34.colorbar(_cp0, ax=axs_29[0], **_kwargs)
    _cb1 = fig_34.colorbar(_cp1, ax=axs_29[1], **_kwargs)
    _cb2 = fig_34.colorbar(_cp2, ax=axs_29[2], **dict(_kwargs, ticks=[-1.5, 0, 1.5]))

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Subsurface variance
    """)
    return


@app.cell
def _(format_subsurf_axs, np, plt, src, var_by_period):
    ## specify which period/month to plot
    # sel = lambda x: x.mean("month")
    _sel = lambda x: np.sqrt(x["T"].sel(month=6))

    fig_35, axs_30 = plt.subplots(1, 3, figsize=(8, 2.5), layout="constrained")

    ## specify periods
    periods = var_by_period.period.values[[1, 2]]

    for _ax, p in zip(axs_30[:2], periods):

        ## temperature
        _cp = _ax.contourf(
            var_by_period.longitude,
            var_by_period.z_t,
            _sel(var_by_period.sel(period=p)),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(3, 0.3),
            extend="both",
        )

    # ## difference
    axs_30[2].contourf(
        var_by_period.longitude,
        var_by_period.z_t,
        _sel(var_by_period.sel(period=periods[1]))
        - _sel(var_by_period.sel(period=periods[0])),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(1, 0.1),
        extend="both",
    )
    ## plot MLD
    # plot_mlds(axs=axs, sel=sel)

    ## label
    _cb = fig_35.colorbar(
        _cp, ax=axs_30[2], ticks=[-3, 0, 3], label=r"$K~\left(\text{Niño 3.4}\right)^{-1}$"
    )
    format_subsurf_axs(axs_30)
    for _ax in axs_30:
        _ax.set_ylim([200, 5])
        _ax.axvline(190, ls="--", c="w", lw=0.8)
        _ax.axvline(240, ls="--", c="w", lw=0.8)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Skew
    """)
    return


@app.cell
def _(scipy, src, xr):
    def skew_helper(data):
        """get skew for data"""

        ## specify kwargs
        kwargs = dict(input_core_dims=[["sample"]], kwargs=dict(axis=-1))

        return xr.apply_ufunc(
            scipy.stats.skew, data.stack(sample=["time", "member"]), **kwargs
        )


    def _merimean(x, lat_bound=2):
        """get meridional mean"""

        ## check dimensions (`"latitude" in x` checks a DataArray's values)
        if "latitude" in x.dims:

            ## get bounds for latitude averaging
            coords = dict(
                latitude=slice(-lat_bound, lat_bound),
            )

            return x.sel(coords).mean("latitude")

        else:
            return x


    def skew_by_month(data):
        """compute skewness for equatorial strip by month"""

        ## get data on equator
        data_eq = src.utils.reconstruct_wrapper(
            data=data,
            fn=_merimean,
        )
        ## compute skewness
        return data_eq.groupby("time.month").map(skew_helper)

    return (skew_by_month,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute
    """)
    return


@app.cell
def _(anom_early, anom_late, skew_by_month):
    ## compute skew
    SKEW_VARNAMES = [
        "sst",
        "sst_comp",
        # "ssh",
        # "ssh_comp",
        # "z20",
        # "z20_comp",
    ]  # , "T", "T_comp"]
    skew_early = skew_by_month(data=anom_early[SKEW_VARNAMES])
    skew_late = skew_by_month(data=anom_late[SKEW_VARNAMES])
    return skew_early, skew_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot
    """)
    return


@app.cell
def _(change, plt, skew_early, skew_late, src):
    ## shared args for plotting
    _plot_kwargs = dict(cmap="cmo.balance", extend="both")

    ## Set up plot
    fig_36, axs_31 = plt.subplots(3, 3, figsize=(9, 6), layout="constrained")

    ## make hövmöllers
    for j, _n in enumerate(["sst", "ssh", "z20"]):
        _cp0 = src.utils.plot_cycle_hov(
            axs_31[0, j],
            skew_early[_n],
            levels=src.utils.make_cb_range(1.5, 0.15),
            **_plot_kwargs,
        )
        _cp1 = src.utils.plot_cycle_hov(
            axs_31[1, j],
            skew_late[_n],
            levels=src.utils.make_cb_range(1.5, 0.15),
            **_plot_kwargs,
        )
        _cp2 = src.utils.plot_cycle_hov(
            axs_31[2, j],
            (skew_late - skew_early)[_n],
            levels=src.utils.make_cb_range(1.5, 0.15),
            **_plot_kwargs,
        )

        ## label
        axs_31[0, j].set_title("Baseline")
        axs_31[1, j].set_title("Future")
        axs_31[2, j].set_title("Difference")
        axs_31[-1, j].set_xlabel("Longitude")
        axs_31[-1, j].set_xticks([140, 190, 240])

    ## add colorbars
    _kwargs = dict(ticks=[-1.5, 0, 1.5])
    _cb0 = fig_36.colorbar(_cp0, ax=axs_31[0], label=r"skew", **_kwargs)
    _cb1 = fig_36.colorbar(_cp1, ax=axs_31[1], label=r"skew", **_kwargs)
    _cb2 = fig_36.colorbar(_cp2, ax=axs_31[2], label=r"$\Delta$ skew", **_kwargs)

    for _ax in axs_31[:, 1:].flatten():
        _ax.set_yticks([])
        _ax.set_ylabel(None)

    ## super-impose plot variance change
    axs_31[2, 0].contour(
        change.longitude,
        change.month,
        change,
        colors="k",
        levels=src.utils.make_cb_range(3, 0.3),
        linewidths=1,
    )

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Spatial
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute
    """)
    return


@app.cell
def _(plt, src, var_by_period):
    ## specify month
    _month = 7

    ## set up paneled subplot
    fig_37 = plt.figure(figsize=(6, 5), layout="constrained")
    axs_32 = src.utils.subplots_with_proj(
        fig_37, nrows=3, ncols=1, format_func=src.utils.plot_setup_pac
    )

    ## plot data
    _kwargs = dict(
        var0=var_by_period["sst"].isel(period=0).sel(month=_month),
        var1=var_by_period["sst"].isel(period=1).sel(month=_month),
        amp=3.5,
        amp_diff=1,
        show_colorbars=True,
        cbar_label=r"$^{\circ}$C$^2$",
    )
    fig_37, axs_32 = src.utils.make_variance_subplots(fig_37, axs_32, **_kwargs)
    axs_32[0, 0].set_title("Early (1853 – 1883)")
    axs_32[1, 0].set_title("Late (2067 – 2097)")
    axs_32[2, 0].set_title("Difference")

    for _ax in axs_32.flatten():
        src.utils.plot_nino34_box(_ax, c="w")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Look at change in thermocline depth over time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Change in Bjerknes coupling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hövmöller
    """)
    return


@app.cell
def _(np, src, xr):
    def make_scatter(ax, data, x_var, y_var, fn_x, fn_y, scale=1):
        """scatter plot data on axis"""

        ## evaluate functions
        if "mode" in data.dims:
            fn_x_eval = src.utils.reconstruct_fn(
                scores=data[x_var], components=data[f"{x_var}_comp"], fn=fn_x
            )
            fn_y_eval = src.utils.reconstruct_fn(
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
        slope = src.utils.regress_core(X=fn_x_eval, Y=scale * fn_y_eval, dim=dim)

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


    def get_alpha(data, dim):
        """compute mu: linear dependence of nhf on sst"""

        return src.utils.regress_core(Y=data["nhf"], X=data["sst"], dim=dim)


    def get_mu(data, dim):
        """compute mu: linear dependence of taux on sst"""

        return src.utils.regress_core(Y=data["taux"], X=data["sst"], dim=dim)


    def get_beta(data, dim):
        """compute mu: linear dependence of ssh on taux"""

        return src.utils.regress_core(Y=data["ssh"], X=data["taux"], dim=dim)


    def get_xi(data, dim):
        """compute mu: linear dependence of sst on ssh"""

        return src.utils.regress_core(Y=data["sst"], X=data["ssh"], dim=dim)


    def get_params(data, dim="time"):
        """compute all parameters"""
        params = xr.merge(
            [
                get_alpha(data, dim=dim).rename("alpha"),
                get_mu(data, dim=dim).rename("mu"),
                get_beta(data, dim=dim).rename("beta"),
                get_xi(data, dim=dim).rename("xi"),
            ]
        )

        params["coupling"] = params["mu"] * params["beta"] * params["xi"]

        return params


    def get_rolling_params(data, n=10, reduce_ensemble=True):
        """get rolling parameters"""

        ## get rolling data
        idx_rolling = data.rolling({"time": 2 * n + 1}, center=True)

        ## expand rolling object along "window" dimension
        idx_rolling = idx_rolling.construct("window")

        ## stack member/window if desired
        if reduce_ensemble:
            idx_rolling = idx_rolling.stack(sample=["member", "window"])

        else:
            idx_rolling = idx_rolling.rename({"window": "sample"})

        return get_params(idx_rolling, dim="sample")


    def get_rolling_params_bymonth(data, **kwargs):
        """get rolling parameters for each month separately..."""

        return data.groupby("time.month").map(get_rolling_params, **kwargs)


    def get_fractional_change(data, dim="year"):
        clim = data.isel({dim: slice(None, 30)}).mean(dim)

        return (data - clim) / clim

    return (get_rolling_params_bymonth,)


@app.cell
def _(anom, src):
    components, _ = src.utils.split_components(anom)
    return (components,)


@app.cell
def _(anom, components, src, xr):
    ## compute indices
    varnames = ["sst", "nhf", "ssh", "taux"]
    _kwargs = dict(scores=anom[varnames], components=components[varnames])
    nino34 = src.utils.reconstruct_fn(fn=src.utils.get_nino34, **_kwargs)
    nino4 = src.utils.reconstruct_fn(fn=src.utils.get_nino4, **_kwargs)

    ## subset for relevant indices
    idxs_1 = xr.merge([nino34[["sst", "nhf", "ssh"]], nino4["taux"]])
    return (idxs_1,)


@app.cell
def _(get_rolling_params_bymonth, idxs_1, src):
    ## get params over time
    params = get_rolling_params_bymonth(idxs_1, n=16)

    ## unstack month and year to separate dims
    params = src.utils.unstack_month_and_year(params)

    ## subtract off climatology
    delta_params = params - params.isel(year=slice(None, 30)).mean("year")
    return (delta_params,)


@app.cell
def _(delta_params, np, plot_hov2, plt):
    ## setup plot
    fig_38, axs_33 = plt.subplots(1, 5, figsize=(9, 4), layout="constrained")

    ## plot T data
    plot_hov2(axs_33[0], delta_params["alpha"].T, amp=20, label=r"$\Delta~ \alpha$", fig=fig_38, axs=axs_33)
    plot_hov2(axs_33[1], delta_params["mu"].T, amp=0.01, label=r"$\tau_x-\text{SST}$", fig=fig_38, axs=axs_33)
    plot_hov2(axs_33[2], delta_params["beta"].T, amp=200, label=r"$\text{SSH}-\tau_x$", fig=fig_38, axs=axs_33)
    plot_hov2(axs_33[3], delta_params["xi"].T, amp=7e-2, label=r"$\text{SST}-\text{SSH}$", fig=fig_38, axs=axs_33)
    plot_hov2(axs_33[4], delta_params["coupling"].T, amp=0.5, label=r"coupling", fig=fig_38, axs=axs_33)

    for _ax in axs_33:
        _ax.axvline(7, c="k", lw=1, ls="--")

    ## label
    axs_33[0].set_yticks(np.linspace(1870, 2082, 5))
    axs_33[0].set_ylabel("Year")
    axs_33[1].set_ylim(axs_33[0].get_ylim())
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### preprocessing
    """)
    return


@app.cell
def _(src):
    def _merimean(x):
        return x.sel(longitude=slice(140, 285), latitude=slice(-5, 5)).mean("latitude")


    def plot_cycle_hov(ax, data, amp, is_filled=True, xticks=[190, 240]):
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
            plot_data = _merimean(data)
        else:
            plot_data = data

        ## do the plotting
        cp = plot_fn(
            plot_data.longitude,
            plot_data.month,
            plot_data,
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

    return (plot_cycle_hov,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add EOF info to dataset
    """)
    return


@app.cell
def _(anom, components, src):
    for _v in list(components):
        if f"{_v}_comp" not in list(anom):
            anom[f"{_v}_comp"] = components[_v]

    ## add T and h indices to data
    names = ["nino3", "nino34", "nino4"]
    fns = [src.utils.get_nino3, src.utils.get_nino34, src.utils.get_nino4]
    for _n, _fn in zip(names, fns):
        if _n not in list(anom):
            anom[_n] = src.utils.reconstruct_fn(
                scores=anom["sst"], components=anom["sst_comp"], fn=_fn
            )

    names = ["h_w", "h"]
    fns = [src.utils.get_RO_hw, src.utils.get_RO_h]
    for _n, _fn in zip(names, fns):
        if _n not in list(anom):
            anom[_n] = src.utils.reconstruct_fn(
                scores=anom["ssh"], components=anom["ssh_comp"], fn=_fn
            )
    return


@app.cell
def _():
    ### To-do: compute Tsub...
    # get_dT_sub(Tsub =
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Partition into early/late
    """)
    return


@app.cell
def _(anom):
    anom_early_1 = anom.isel(time=slice(12, 372))
    anom_late_1 = anom.isel(time=slice(-372, -12))
    return anom_early_1, anom_late_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Remove linear dependence on SST and get tendencies
    """)
    return


@app.cell
def _(src):
    def prep(data):
        """remove sst dependence and compute tendencies"""

        ## remove SST dependence from SSH field
        if "ssh" in list(data):
            data["ssh_hat"] = src.utils.remove_sst_dependence_v2(
                data, h_var="ssh", T_var="nino34"
            )
            data["ssh_hat_comp"] = data["ssh_comp"]

        ## remove from h indices
        for h_idx in ["h_w", "h"]:
            data[f"{h_idx}_hat"] = src.utils.remove_sst_dependence_v2(
                data, h_var=h_idx, T_var="nino34"
            )

        ## compute tendencies
        data = src.utils.get_ddt(data)

        return data

    return (prep,)


@app.cell
def _(anom_early_1, anom_late_1, prep):
    anom_early_2 = prep(anom_early_1)
    anom_late_2 = prep(anom_late_1)
    return anom_early_2, anom_late_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $R$
    $\frac{d T}{dt}$ vs $T_{34}$
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared args
    _kwargs = dict(x_vars=["nino34", "h_w_hat"], y_var="ddt_sst")

    m_early = src.utils.multi_regress_bymonth(anom_early_2, **_kwargs)["nino34"]
    m_late = src.utils.multi_regress_bymonth(anom_late_2, **_kwargs)["nino34"]
    return m_early, m_late


@app.cell
def _(m_early, m_late, plt, src):
    ## select month
    sel_mon_3 = lambda x: x.sel(month=7)

    ## set up plot
    fig_39 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_34 = src.utils.subplots_with_proj(fig_39, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=5, sel=sel_mon_3)
    _cp0 = src.utils.make_contour_plot(axs_34[0, 0], m_early, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_34[1, 0], m_late, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_34[2, 0], (m_late - m_early), **dict(_contour_kwargs, amp=5)
    )

    ## colorbar
    _lab = r"$K ~\text{yr}^{-1}~ \left(T_{34}\right)^{-1}$"
    _cb0 = fig_39.colorbar(_cp0, ax=axs_34, ticks=[-5, 0, 5], label=_lab)

    ## Niño 3 box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_34.flatten():
        src.utils.plot_nino4_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Scatter plot of Niño 3.4 vs. $\frac{d}{dt}\left(\text{Niño 3.4}\right)$ in July
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, plt):
    ## get data for july
    def get_month(data, month):
        month = data.sel(time=anom_late_2.time.dt.month == month)
        return month.stack(sample=["member", "time"])


    jul_early = get_month(anom_early_2, 7)
    jul_late = get_month(anom_late_2, 7)


    ## make plot
    fig_40, axs_35 = plt.subplots(1, 2, figsize=(5, 2.4))

    axs_35[0].scatter(
        jul_early["nino4"],
        jul_early["ddt_nino4"],
        s=2,
        alpha=0.5,
    )

    axs_35[1].scatter(
        jul_late["nino4"],
        jul_late["ddt_nino4"],
        s=2,
        alpha=0.5,
    )

    for _ax in axs_35:
        _ax.set_xlim([-3, 3])
        _ax.set_ylim([-25, 15])
        _ax.axhline(0, ls="--", c="k", lw=1)
        _ax.axvline(0, ls="--", c="k", lw=1)

    axs_35[1].set_yticks([])
    axs_35[0].set_title("Early")
    axs_35[1].set_title("Late")

    plt.gca()
    return (get_month,)


@app.cell
def _(m_early, m_late, plt, src):
    _merimean = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")

    ## make hövmöllers
    fig_41, axs_36 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(
        cmap="cmo.balance", levels=src.utils.make_cb_range(7.5, 0.75), extend="both"
    )
    _cb_kwargs = dict(ticks=[-5, 0, 5], label=r"$m~yr^{-1}~K^{-1}$")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_36[0], _merimean(m_early), **_kwargs)
    _cb0 = fig_41.colorbar(_cp0, ax=axs_36[0], **_cb_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_36[1], _merimean(m_late), **_kwargs)
    _cb1 = fig_41.colorbar(_cp1, ax=axs_36[1], **_cb_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_36[2],
        _merimean(m_late - m_early),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(7.5, 0.75),
        extend="both",
    )

    _cb2 = fig_41.colorbar(_cp2, ax=axs_36[2], ticks=[-7.5, 0, 7.5], label=r"$m~yr^{-1}~K^{-1}$")

    ## label
    axs_36[0].set_title("Early")
    axs_36[1].set_title("Late")
    axs_36[2].set_title("Difference")
    axs_36[-1].set_xlabel("Longitude")
    axs_36[-1].set_xticks([140, 190, 240])
    for _ax in axs_36:
        _kwargs = dict(ls="--", c="k", lw=1, alpha=0.5)
        _ax.axhline(6, **_kwargs)
        _ax.axhline(7, **_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $F_2$
    $\frac{d h}{dt}$ vs $T_{34}$
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared args
    _kwargs = dict(x_vars=["nino34", "h_w_hat"], y_var="ddt_ssh_hat")

    m_early_1 = src.utils.multi_regress_bymonth(anom_early_2, **_kwargs)["nino34"]
    m_late_1 = src.utils.multi_regress_bymonth(anom_late_2, **_kwargs)["nino34"]
    return m_early_1, m_late_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Spatial
    """)
    return


@app.cell
def _(m_early_1, m_late_1, plt, src):
    ## select month
    sel_mon_4 = lambda x: x.sel(month=slice(5, 5)).mean("month")

    ## set up plot
    fig_42 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_37 = src.utils.subplots_with_proj(fig_42, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=50, sel=sel_mon_4)
    _cp0 = src.utils.make_contour_plot(axs_37[0, 0], m_early_1, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_37[1, 0], m_late_1, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_37[2, 0], m_late_1 - m_early_1, **dict(_contour_kwargs, amp=50)
    )

    ## colorbar
    _lab = r"$m ~\text{yr}^{-1}~ \left(T_{34}\right)^{-1}$"
    _cb0 = fig_42.colorbar(_cp0, ax=axs_37, ticks=[-50, 0, 50], label=_lab)

    ## h_w box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_37.flatten():
        src.utils.plot_box(_ax, lons=[120, 210], lats=[-5, 5], **_box_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hovmoller
    """)
    return


@app.cell
def _(m_early_1, m_late_1, plt, src):
    _merimean = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")

    ## make hövmöllers
    fig_43, axs_38 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.balance", levels=src.utils.make_cb_range(40, 4), extend="both")
    _cb_kwargs = dict(ticks=[-40, 0, 40], label=r"$m~yr^{-1}~K^{-1}$")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_38[0], _merimean(m_early_1), **_kwargs)
    _cb0 = fig_43.colorbar(_cp0, ax=axs_38[0], **_cb_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_38[1], _merimean(m_late_1), **_kwargs)
    _cb1 = fig_43.colorbar(_cp1, ax=axs_38[1], **_cb_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_38[2],
        _merimean(m_late_1 - m_early_1),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(32, 3.2),
        extend="both",
    )

    # ## contour MLD changes
    # axs[2].contour(
    #     mld_clim_diff_pct2.longitude,
    #     mld_clim_diff_pct2.month,
    #     mld_clim_diff_pct2.T,
    #     levels=src.utils.make_cb_range(100,25),
    #     colors="k",
    #     linewidths=.5,
    #     alpha=.5,
    # )

    _cb2 = fig_43.colorbar(_cp2, ax=axs_38[2], ticks=[-32, 0, 32], label=r"$m~yr^{-1}~K^{-1}$")

    ## label
    axs_38[0].set_title("Early")
    axs_38[1].set_title("Late")
    axs_38[2].set_title("Difference")
    axs_38[-1].set_xlabel("Longitude")
    axs_38[-1].set_xticks([140, 190, 240])

    plt.gca()
    return


@app.cell
def _(np, src):
    def scatter_data(ax, data):
        """add scatter data to plot"""

        ## get data for plotting
        x = data["nino34"]
        y = data["ddt_h_w_hat"]

        ## compute regression coefficient
        m = src.utils.regress_core(Y=y, X=x, dim="sample").values.item()

        ## plot data
        ax.scatter(x, y, s=2, alpha=0.5)

        ## plot best fit
        z = np.linspace(x.min().values.item(), x.max().values.item())
        ax.plot(z, m * z, c="k", lw=1, label=f"{m:.1f}" + r" $m~yr^{-1}~K^{-1}$")

        ## formatting
        ax.set_xlim([-3, 3])
        ax.set_ylim([-80, 80])
        ax.axhline(0, ls="--", c="k", lw=1)
        ax.axvline(0, ls="--", c="k", lw=1)
        ax.set_xlabel("Niño 3.4")
        ax.set_yticks([])
        ax.legend(prop=dict(size=8))

        return

    return (scatter_data,)


@app.cell
def _(anom_early_2, anom_late_2, get_month, plt, scatter_data):
    ## select month to plot
    _month = 5

    ## extract data for month
    may_early = get_month(anom_early_2, _month)
    may_late = get_month(anom_late_2, _month)

    ## make plot
    fig_44, axs_39 = plt.subplots(1, 2, figsize=(5, 2.4), layout="constrained")

    scatter_data(axs_39[0], may_early)
    scatter_data(axs_39[1], may_late)

    ## formatting
    axs_39[0].set_yticks([-60, 0, 60])
    axs_39[0].set_title("Early")
    axs_39[1].set_title("Late")
    axs_39[0].set_ylabel("Recharge rate ($m~yr^{-1}$)", size=10)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### dynamical damping: $\overline{\mathbf{u}}\cdot\nabla T'$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get temperature coefficients
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared args
    _T_var = "nino34"
    _kwargs = dict(x_vars=[_T_var, "h_w_hat"], y_var="sst")

    m_early_2 = src.utils.multi_regress_bymonth(anom_early_2, **_kwargs)[_T_var]
    m_late_2 = src.utils.multi_regress_bymonth(anom_late_2, **_kwargs)[_T_var]
    return m_early_2, m_late_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get velocity clim
    """)
    return


@app.cell
def _(forced, src, vel_comps):
    _kwargs = dict(
        components=vel_comps.rename({"uvel_comp": "uvel", "vvel_comp": "vvel"}),
        fn=lambda x: x,
    )

    vel_clim = lambda t_idx: src.utils.reconstruct_fn(
        scores=forced[["uvel", "vvel"]].isel(t_idx).groupby("time.month").mean(),
        **_kwargs,
    )


    vel_early = vel_clim(t_idx=dict(time=slice(12, 372)))
    vel_late = vel_clim(t_idx=dict(time=slice(-373, -13)))
    return vel_early, vel_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute tendency from advection
    """)
    return


@app.cell
def _(get_adv, m_early_2, m_late_2, vel_early, vel_late):
    uvT_early_ubar = get_adv(uv=vel_early, T=m_early_2)
    uvT_late_ubar = get_adv(uv=vel_late, T=m_late_2)
    return uvT_early_ubar, uvT_late_ubar


@app.cell
def _(plt, src, uvT_early_ubar, uvT_late_ubar):
    ## select month
    sel_mon_5 = lambda x: x.sel(month=9)

    ## set up plot
    fig_45 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_40 = src.utils.subplots_with_proj(fig_45, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=2, sel=sel_mon_5)
    _cp0 = src.utils.make_contour_plot(axs_40[0, 0], uvT_early_ubar, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_40[1, 0], uvT_late_ubar, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_40[2, 0], uvT_late_ubar - uvT_early_ubar, **dict(_contour_kwargs, amp=2)
    )

    ## colorbar
    _lab = r"$cm ~\text{s}^{-1}~ \left(T_{34}\right)^{-1}$"
    _cb0 = fig_45.colorbar(_cp0, ax=axs_40, ticks=[-2, 0, 2], label=_lab)

    ## h_w box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_40.flatten():
        src.utils.plot_nino34_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell
def _(plt, src, uvT_early_ubar, uvT_late_ubar):
    _merimean = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")

    ## make hövmöllers
    fig_46, axs_41 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.balance", levels=src.utils.make_cb_range(2, 0.2), extend="both")
    _cb_kwargs = dict(ticks=[-2, 0, 2], label=r"$K~yr^{-1}$")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_41[0], _merimean(uvT_early_ubar), **_kwargs)
    _cb0 = fig_46.colorbar(_cp0, ax=axs_41[0], **_cb_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_41[1], _merimean(uvT_late_ubar), **_kwargs)
    _cb1 = fig_46.colorbar(_cp1, ax=axs_41[1], **_cb_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_41[2],
        _merimean(uvT_late_ubar - uvT_early_ubar),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(2, 0.2),
        extend="both",
    )

    _cb2 = fig_46.colorbar(_cp2, ax=axs_41[2], ticks=[-2, 0, 2], label=r"$K~yr^{-1}$")

    ## label
    axs_41[0].set_title("Early")
    axs_41[1].set_title("Late")
    axs_41[2].set_title("Difference")
    axs_41[-1].set_xlabel("Longitude")
    axs_41[-1].set_xticks([140, 190, 240])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Horizontal advection: $\mathbf{u}'\cdot\nabla \overline{T}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get climatological SST
    """)
    return


@app.cell
def _(components, forced, src):
    _kwargs = dict(
        components=components["sst"],
        fn=lambda x: x,
    )

    sst_clim = lambda t_idx: src.utils.reconstruct_fn(
        scores=forced["sst"].isel(t_idx).groupby("time.month").mean(),
        **_kwargs,
    )


    sst_early = sst_clim(t_idx=dict(time=slice(12, 372)))
    sst_late = sst_clim(t_idx=dict(time=slice(-373, -13)))
    return sst_early, sst_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get horizontal vel anoms.
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src, xr):
    ## shared args
    _T_var = "nino34"
    _kwargs = dict(x_vars=[_T_var, "h_w_hat"])

    m_early_u = src.utils.multi_regress_bymonth(anom_early_2, y_var="uvel", **_kwargs)[_T_var]
    m_late_u = src.utils.multi_regress_bymonth(anom_late_2, y_var="uvel", **_kwargs)[_T_var]

    m_early_v = src.utils.multi_regress_bymonth(anom_early_2, y_var="vvel", **_kwargs)[_T_var]
    m_late_v = src.utils.multi_regress_bymonth(anom_late_2, y_var="vvel", **_kwargs)[_T_var]

    ## merge results
    merge = lambda u, v: xr.merge([u.rename("uvel"), v.rename("vvel")])
    m_early_3 = merge(m_early_u, m_early_v)
    m_late_3 = merge(m_late_u, m_late_v)
    return m_early_3, m_late_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute advection
    """)
    return


@app.cell
def _(get_adv, m_early_3, m_late_3, sst_early, sst_late):
    uvT_early_Tbar = get_adv(uv=m_early_3, T=sst_early)
    uvT_late_Tbar = get_adv(uv=m_late_3, T=sst_late)
    return uvT_early_Tbar, uvT_late_Tbar


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot
    """)
    return


@app.cell
def _(plt, src, uvT_early_Tbar, uvT_late_Tbar):
    ## select month
    sel_mon_6 = lambda x: x.sel(month=7)

    ## set up plot
    fig_47 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_42 = src.utils.subplots_with_proj(fig_47, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=2, sel=sel_mon_6)
    _cp0 = src.utils.make_contour_plot(axs_42[0, 0], uvT_early_Tbar, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_42[1, 0], uvT_late_Tbar, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_42[2, 0], uvT_late_Tbar - uvT_early_Tbar, **dict(_contour_kwargs, amp=2)
    )

    ## colorbar
    _lab = r"$K ~\text{yr}^{-1}$"
    _cb0 = fig_47.colorbar(_cp0, ax=axs_42, ticks=[-2, 0, 2], label=_lab)

    ## h_w box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_42.flatten():
        src.utils.plot_nino34_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell
def _(plt, src, uvT_early_Tbar, uvT_late_Tbar):
    _merimean = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")

    ## make hövmöllers
    fig_48, axs_43 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.balance", levels=src.utils.make_cb_range(2, 0.2), extend="both")
    _cb_kwargs = dict(ticks=[-2, 0, 2], label=r"$K~yr^{-1}$")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_43[0], _merimean(uvT_early_Tbar), **_kwargs)
    _cb0 = fig_48.colorbar(_cp0, ax=axs_43[0], **_cb_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_43[1], _merimean(uvT_late_Tbar), **_kwargs)
    _cb1 = fig_48.colorbar(_cp1, ax=axs_43[1], **_cb_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_43[2],
        _merimean(uvT_late_Tbar - uvT_early_Tbar),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(2, 0.2),
        extend="both",
    )

    _cb2 = fig_48.colorbar(_cp2, ax=axs_43[2], ticks=[-2, 0, 2], label=r"$K~yr^{-1}$")

    ## label
    axs_43[0].set_title("Early")
    axs_43[1].set_title("Late")
    axs_43[2].set_title("Difference")
    axs_43[-1].set_xlabel("Longitude")
    axs_43[-1].set_xticks([140, 190, 240])
    for _ax in axs_43:
        _ax.set_xlim([150, 270])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Total advection
    """)
    return


@app.cell
def _(uvT_early_Tbar, uvT_early_ubar, uvT_late_Tbar, uvT_late_ubar):
    uvT_early = uvT_early_ubar + uvT_early_Tbar
    uvT_late = uvT_late_ubar + uvT_late_Tbar
    return uvT_early, uvT_late


@app.cell
def _(plt, src, uvT_early, uvT_late):
    _merimean = lambda x: x.sel(latitude=slice(-5, 5)).mean("latitude")

    ## make hövmöllers
    fig_49, axs_44 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.balance", levels=src.utils.make_cb_range(4, 0.4), extend="both")
    _cb_kwargs = dict(ticks=[-4, 0, 4], label=r"$K~yr^{-1}$")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_44[0], _merimean(uvT_early), **_kwargs)
    _cb0 = fig_49.colorbar(_cp0, ax=axs_44[0], **_cb_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_44[1], _merimean(uvT_late), **_kwargs)
    _cb1 = fig_49.colorbar(_cp1, ax=axs_44[1], **_cb_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_44[2],
        _merimean(uvT_late - uvT_early),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(4, 0.4),
        extend="both",
    )

    _cb2 = fig_49.colorbar(_cp2, ax=axs_44[2], ticks=[-4, 0, 4], label=r"$K~yr^{-1}$")

    ## label
    axs_44[0].set_title("Early")
    axs_44[1].set_title("Late")
    axs_44[2].set_title("Difference")
    axs_44[-1].set_xlabel("Longitude")
    axs_44[-1].set_xticks([140, 190, 240])
    for _ax in axs_44:
        _ax.set_xlim([150, 270])
        for _t in [6, 7]:
            _ax.axhline(_t, ls="--", c="k", lw=0.8, alpha=0.5)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Scatter
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### precip
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared args
    _kwargs = dict(x_vars=["nino34"], y_var="pr")

    m_early_4 = src.utils.multi_regress_bymonth(anom_early_2, **_kwargs)["nino34"]
    m_late_4 = src.utils.multi_regress_bymonth(anom_late_2, **_kwargs)["nino34"]
    return m_early_4, m_late_4


@app.cell
def _(m_early_4, m_late_4, plt, src):
    ## select month
    sel_mon_7 = lambda x: x.sel(month=slice(7, 7)).mean("month")

    ## set up plot
    fig_50 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_45 = src.utils.subplots_with_proj(fig_50, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=8e-5, sel=sel_mon_7)
    _cp0 = src.utils.make_contour_plot(axs_45[0, 0], m_early_4, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_45[1, 0], m_late_4, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_45[2, 0], m_late_4 - m_early_4, **dict(_contour_kwargs, amp=8e-5)
    )

    ## colorbar
    _lab = r"$cm ~\text{s}^{-1}~ \left(T_{34}\right)^{-1}$"
    _cb0 = fig_50.colorbar(_cp0, ax=axs_45, ticks=[-3, 0, 3], label=_lab)

    ## h_w box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_45.flatten():
        src.utils.plot_nino34_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    def get_data(data, y_var, x_var="nino34", month=7):

        ## indexer for jul
        is_mo = dict(time=data.time.dt.month == month)

        y_data = src.utils.reconstruct_fn(
            scores=data[y_var].isel(is_mo),
            components=data[f"{y_var}_comp"],
            fn=src.utils.get_nino34,
        )

        x_data = data[x_var].isel(is_mo)

        ## stack
        y_data = y_data.stack(sample=["time", "member"])
        x_data = x_data.stack(sample=["time", "member"])

        return x_data, y_data


    _x_data_early, _y_data_early = get_data(anom_early_2, y_var="pr")
    _x_data_late, _y_data_late = get_data(anom_late_2, y_var="pr")
    return (get_data,)


@app.cell
def _(anom_early_2, anom_late_2, get_data, plt, set_xlims, set_ylims):
    ## compute
    _x_data_early, _y_data_early = get_data(anom_early_2, y_var="pr")
    _x_data_late, _y_data_late = get_data(anom_late_2, y_var="pr")


    ## plot
    fig_51, axs_46 = plt.subplots(1, 2, figsize=(7, 3.2))

    axs_46[0].scatter(_x_data_early, _y_data_early, s=3)
    axs_46[1].scatter(_x_data_late, _y_data_late, s=3)

    set_ylims(axs_46)
    set_xlims(axs_46)
    axs_46[1].set_yticks([])
    for _ax in axs_46:
        _kwargs = dict(ls="--", c="k", lw=0.8)
        _ax.axhline(0, **_kwargs)
        _ax.axvline(0, **_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### advection
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src, xr):
    ## shared args
    _kwargs = dict(x_vars=["nino34"])

    ## compute coefs
    uvel_early = src.utils.multi_regress_bymonth(anom_early_2, y_var="uvel", **_kwargs)
    uvel_late = src.utils.multi_regress_bymonth(anom_late_2, y_var="uvel", **_kwargs)

    vvel_early = src.utils.multi_regress_bymonth(anom_early_2, y_var="vvel", **_kwargs)
    vvel_late = src.utils.multi_regress_bymonth(anom_late_2, y_var="vvel", **_kwargs)

    ## merge
    vel_early_1 = xr.merge(
        [uvel_early.rename({"nino34": "uvel"}), vvel_early.rename({"nino34": "vvel"})]
    )

    vel_late_1 = xr.merge(
        [uvel_late.rename({"nino34": "uvel"}), vvel_late.rename({"nino34": "vvel"})]
    )
    return vel_early_1, vel_late_1


@app.cell
def _(components, forced, src):
    _kwargs = dict(components=components["sst"], fn=lambda x: x)

    tos_early = src.utils.reconstruct_fn(
        scores=forced.isel(time=slice(None, 360))["sst"].mean("time"),
        **_kwargs,
    )

    tos_late = src.utils.reconstruct_fn(
        scores=forced.isel(time=slice(-360, None))["sst"].mean("time"),
        **_kwargs,
    )
    return tos_early, tos_late


@app.cell
def _(ccrs, np, plt, src, tos_early, tos_late, vel_early_1, vel_late_1):


    ## select month
    sel_mon_8 = lambda x: x.sel(month=slice(7, 7)).mean("month")

    ## set up plot
    fig_52 = plt.figure(figsize=(10.5, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_47 = src.utils.subplots_with_proj(fig_52, nrows=2, ncols=1, format_func=_format_func)

    ## plot SST in background
    _cp = axs_47[0, 0].contourf(
        tos_early.longitude,
        tos_early.latitude,
        tos_early,
        levels=np.arange(22, 30),
        cmap="cmo.thermal",
        transform=ccrs.PlateCarree(),
        extend="both",
    )

    _cp = axs_47[1, 0].contourf(
        tos_early.longitude,
        tos_early.latitude,
        tos_late,
        levels=np.arange(25, 33),
        cmap="cmo.thermal",
        transform=ccrs.PlateCarree(),
        extend="both",
    )

    ## plot velocity on top
    _kwargs = dict(
        transform=ccrs.PlateCarree(),
        scale=3e2,
        headwidth=4,
        width=2e-3,
    )
    _n = 2
    axs_47[0, 0].quiver(
        vel_early_1.longitude.values[::_n],
        vel_early_1.latitude.values[::_n],
        sel_mon_8(vel_early_1["uvel"]).values[::_n, ::_n],
        sel_mon_8(vel_early_1["vvel"]).values[::_n, ::_n],
        **_kwargs
    )

    axs_47[1, 0].quiver(
        vel_early_1.longitude.values[::_n],
        vel_early_1.latitude.values[::_n],
        sel_mon_8(vel_late_1["uvel"]).values[::_n, ::_n],
        sel_mon_8(vel_late_1["vvel"]).values[::_n, ::_n],
        **_kwargs
    )

    # ## colorbar
    # lab = r"$cm ~\text{s}^{-1}~ \left(T_{34}\right)^{-1}$"
    # cb0 = fig.colorbar(cp0, ax=axs, ticks=[-3, 0, 3], label=lab)

    ## h_w box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_47.flatten():
        # src.utils.plot_box(ax, lons=[120, 210], lats=[-5, 5], **box_kwargs)
        src.utils.plot_nino34_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### zonal
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared args
    _kwargs = dict(x_vars=["nino34"], y_var="uvel")

    ## compute coefs
    m_early_5 = src.utils.multi_regress_bymonth(anom_early_2, **_kwargs)["nino34"]
    m_late_5 = src.utils.multi_regress_bymonth(anom_late_2, **_kwargs)["nino34"]
    return m_early_5, m_late_5


@app.cell
def _(m_early_5, m_late_5, plt, src):
    ## select month
    sel_mon_9 = lambda x: x.sel(month=7)

    ## set up plot
    fig_53 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_48 = src.utils.subplots_with_proj(fig_53, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=30, sel=sel_mon_9)
    _cp0 = src.utils.make_contour_plot(axs_48[0, 0], m_early_5, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_48[1, 0], m_late_5, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_48[2, 0], m_late_5 - m_early_5, **dict(_contour_kwargs, amp=30)
    )

    ## colorbar
    _lab = r"$cm ~\text{s}^{-1}~ \left(T_{34}\right)^{-1}$"
    _cb0 = fig_53.colorbar(_cp0, ax=axs_48, ticks=[-3, 0, 3], label=_lab)

    ## h_w box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_48.flatten():
        # src.utils.plot_box(ax, lons=[120, 210], lats=[-5, 5], **box_kwargs)
        src.utils.plot_nino34_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### meridional
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared args
    _kwargs = dict(x_vars=["nino34"], y_var="vvel")

    ## compute coefs
    m_early_6 = src.utils.multi_regress_bymonth(anom_early_2, **_kwargs)["nino34"]
    m_late_6 = src.utils.multi_regress_bymonth(anom_late_2, **_kwargs)["nino34"]
    return m_early_6, m_late_6


@app.cell
def _(m_early_6, m_late_6, plt, src):
    ## select month
    sel_mon_10 = lambda x: x.sel(month=7)

    ## set up plot
    fig_54 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_49 = src.utils.subplots_with_proj(fig_54, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=3, sel=sel_mon_10)
    _cp0 = src.utils.make_contour_plot(axs_49[0, 0], m_early_6, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_49[1, 0], m_late_6, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_49[2, 0], m_late_6 - m_early_6, **dict(_contour_kwargs, amp=3)
    )

    ## colorbar
    _lab = r"$cm ~\text{s}^{-1}~ \left(T_{34}\right)^{-1}$"
    _cb0 = fig_54.colorbar(_cp0, ax=axs_49, ticks=[-3, 0, 3], label=_lab)

    ## h_w box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_49.flatten():
        # src.utils.plot_box(ax, lons=[120, 210], lats=[-5, 5], **box_kwargs)
        src.utils.plot_nino34_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SST-SST
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared args
    _T_var = "nino3"
    _kwargs = dict(x_vars=[_T_var, "h_w_hat"], y_var="sst")

    ## compute coefs
    m_early_7 = src.utils.multi_regress_bymonth(anom_early_2, **_kwargs)[_T_var]
    m_late_7 = src.utils.multi_regress_bymonth(anom_late_2, **_kwargs)[_T_var]
    return m_early_7, m_late_7


@app.cell
def _(m_early_7, m_late_7, plt, src):
    ## select month
    sel_mon_11 = lambda x: x.sel(month=2)

    ## set up plot
    fig_55 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_50 = src.utils.subplots_with_proj(fig_55, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=1.5, sel=sel_mon_11)
    _cp0 = src.utils.make_contour_plot(axs_50[0, 0], m_early_7, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_50[1, 0], m_late_7, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_50[2, 0], m_late_7 - m_early_7, **dict(_contour_kwargs, amp=0.75)
    )

    ## colorbar
    _lab = r"$Pa~K^{-1}$"
    # cb0 = fig.colorbar(cp0, ax=axs[:2], ticks=[-3e-2, 0, 3e-2], label=lab)
    # cb2 = fig.colorbar(cp2, ax=axs[-1], ticks=[-1.5e-2, 0, 1.5e-2], label=lab)

    ## Niño 3 box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_50.flatten():
        src.utils.plot_nino34_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell
def _(m_early_7, m_late_7, plot_cycle_hov, plt):
    fig_56, axs_51 = plt.subplots(1, 2, figsize=(4, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_51[0], data=m_early_7, amp=1.5)
    plot_cycle_hov(axs_51[0], data=m_late_7, amp=1.5, is_filled=False)

    ## plot difference
    _cp2 = plot_cycle_hov(axs_51[1], data=m_late_7 - m_early_7, amp=0.75)

    ## make it look nicer
    axs_51[1].set_yticks([])
    axs_51[0].set_yticks([1, 5, 9, 12], labels=["Jan", "May", "Sep", "Dec"])
    axs_51[0].set_ylabel("Month")
    axs_51[0].set_title(r"$\tau_x$-SST coupling")
    axs_51[1].set_title("Change")

    # for ax in axs:
    #     ax.set_xlim([150,280])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $\tau_x$-SST
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute slope
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared args
    _T_var = "nino3"
    _kwargs = dict(x_vars=["nino34"], y_var="taux")

    ## compute coefs
    m_early_8 = src.utils.multi_regress_bymonth(anom_early_2, **_kwargs).to_dataarray().squeeze()
    m_late_8 = src.utils.multi_regress_bymonth(anom_late_2, **_kwargs).to_dataarray().squeeze()
    return m_early_8, m_late_8


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot hovmoller
    """)
    return


@app.cell
def _(m_early_8, m_late_8, plot_cycle_hov, plt):
    fig_57, axs_52 = plt.subplots(1, 2, figsize=(4, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_52[0], data=m_early_8, amp=0.015)
    plot_cycle_hov(axs_52[0], data=m_late_8, amp=0.015, is_filled=False)

    ## plot difference
    _cp2 = plot_cycle_hov(axs_52[1], data=m_late_8 - m_early_8, amp=0.0075)

    ## make it look nicer
    axs_52[1].set_yticks([])
    axs_52[0].set_yticks([1, 5, 9, 12], labels=["Jan", "May", "Sep", "Dec"])
    axs_52[0].set_ylabel("Month")
    axs_52[0].set_title(r"$\tau_x$-SST coupling")
    axs_52[1].set_title("Change")

    # for ax in axs:
    #     ax.set_xlim([150,280])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Spatial plot
    """)
    return


@app.cell
def _(m_early_8, m_late_8, plt, src):
    ## select month
    sel_mon_12 = lambda x: x.sel(month=6)
    # sel_mon = lambda x: x.mean("month")

    ## set up plot
    fig_58 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_53 = src.utils.subplots_with_proj(fig_58, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=3e-2, sel=sel_mon_12)
    _cp0 = src.utils.make_contour_plot(axs_53[0, 0], m_early_8, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_53[1, 0], m_late_8, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_53[2, 0], m_late_8 - m_early_8, **dict(_contour_kwargs, amp=1.5e-2)
    )

    ## colorbar
    _lab = r"$Pa~K^{-1}$"
    _cb0 = fig_58.colorbar(_cp0, ax=axs_53[:2], ticks=[-3e-2, 0, 3e-2], label=_lab)
    _cb2 = fig_58.colorbar(_cp2, ax=axs_53[-1], ticks=[-1.5e-2, 0, 1.5e-2], label=_lab)

    ## Niño 3 box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_53.flatten():
        src.utils.plot_nino34_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell
def _(m_early_8, m_late_8, plt):
    avg = lambda x: x.sel(month=6).sel(latitude=slice(-5, 5)).mean("latitude")
    fig_59, _ax = plt.subplots(figsize=(3, 2.5))
    _ax.plot(m_early_8.longitude, avg(m_early_8))
    _ax.plot(m_early_8.longitude, avg(m_late_8))
    _ax.set_ylim([-0.008, 0.018])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Explained variance
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    def regress_helper(data):
        """helper function for regression"""

        ## stack data
        data_stack = data.stack(sample=["member", "time"])

        ## kwargs for regression
        kwargs = dict(X=data_stack["nino3"], Y=data_stack["taux"], dim="sample")

        return src.utils.regress_core(**kwargs)


    def get_recon_skill(data):
        """get reconstruction error"""

        ## get coefficients
        m = regress_helper(data)

        ## get reconstruction
        recon = data["nino34"] * m

        ## reconstruct correlation
        cov = src.utils.reconstruct_cov_da(
            V_x=recon,
            V_y=data["taux"],
            U_x=data["taux_comp"],
            U_y=data["taux_comp"],
        )
        var = src.utils.reconstruct_var(
            scores=data["taux"],
            components=data["taux_comp"],
        )

        return cov / var


    ## function to get correlation
    get_skill = lambda x: x.groupby("time.month").map(get_recon_skill)

    ## then, reconstruct regression coefficient
    r_early = get_skill(anom_early_2)
    r_late = get_skill(anom_late_2)
    return r_early, r_late


@app.cell
def _(plot_cycle_hov, plt, r_early, r_late):
    fig_60, axs_54 = plt.subplots(1, 2, figsize=(4, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_54[0], data=r_early, amp=0.8)
    plot_cycle_hov(axs_54[0], data=r_late, amp=0.8, is_filled=False)

    ## plot difference
    _cp2 = plot_cycle_hov(axs_54[1], data=r_late - r_early, amp=0.4)

    ## make it look nicer
    axs_54[1].set_yticks([])
    axs_54[0].set_yticks([1, 5, 9, 12], labels=["Jan", "May", "Sep", "Dec"])
    axs_54[0].set_ylabel("Month")
    axs_54[0].set_title(r"Correlation")
    axs_54[1].set_title("Change")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SSH - $\tau_x$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared kwargs
    _kwargs = dict(x_var="taux", y_var="ssh", fn_x=src.utils.get_nino4)

    ## function to get slope
    _get_slope = lambda x: x.groupby("time.month").map(src.utils.regress_proj, **_kwargs)

    ## then, reconstruct regression coefficient
    m_early_9 = _get_slope(anom_early_2)
    m_late_9 = _get_slope(anom_late_2)
    return m_early_9, m_late_9


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot
    """)
    return


@app.cell
def _(m_early_9, m_late_9, plot_cycle_hov, plt):
    fig_61, axs_55 = plt.subplots(1, 2, figsize=(4, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_55[0], data=m_early_9, amp=8e2)
    plot_cycle_hov(axs_55[0], data=m_late_9, amp=8e2, is_filled=False)

    ## plot difference
    _cp2 = plot_cycle_hov(axs_55[1], data=m_late_9 - m_early_9, amp=4e2)

    ## make it look nicer
    axs_55[1].set_yticks([])
    axs_55[0].set_yticks([1, 5, 9, 12], labels=["Jan", "May", "Sep", "Dec"])
    axs_55[0].set_ylabel("Month")
    axs_55[0].set_title(r"SSH-$\tau_x$ coupling")
    axs_55[1].set_title("Change")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Scatter plot
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, avg_mon_range, components, forced, src):
    ## specify month range
    m_range = [1, 3]
    sel_fn = lambda x: avg_mon_range(x, *m_range)

    get_ssh_ = lambda x: sel_fn(
        src.utils.reconstruct_fn(
            scores=x["ssh"],
            components=components["ssh"],
            fn=src.utils.get_nino3,
        )
    )

    get_taux_ = lambda x: sel_fn(
        src.utils.reconstruct_fn(
            scores=x["taux"],
            components=components["taux"],
            fn=src.utils.get_nino4,
        )
    )

    ## get early/late NHF
    ssh_early_1 = get_ssh_(anom_early_2)
    ssh_late_1 = get_ssh_(anom_late_2)

    ## get early/late taux
    taux_early = get_taux_(anom_early_2)
    taux_late = get_taux_(anom_late_2)

    ## compute slope
    m_early_ = src.utils.regress_core(
        X=taux_early.stack(sample=["time", "member"]),
        Y=ssh_early_1.stack(sample=["time", "member"]),
        dim="sample",
    ).values.item()

    m_late_ = src.utils.regress_core(
        X=taux_late.stack(sample=["time", "member"]),
        Y=ssh_late_1.stack(sample=["time", "member"]),
        dim="sample",
    ).values.item()

    ## Get Niño 3.4 mixed layer depth
    H = src.utils.reconstruct_fn(
        scores=forced["mld"],
        components=components["mld"],
        fn=src.utils.get_nino34,
    )

    ## get early/late
    H_early = H.isel(time=slice(None, 360)).groupby("time.month").mean()
    H_late = H.isel(time=slice(-361, -1)).groupby("time.month").mean()

    ## subset for month
    H_early = H_early.sel(month=slice(*m_range)).mean("month")
    H_late = H_late.sel(month=slice(*m_range)).mean("month")


    ## get "predicted" m_late
    m_late_hat = m_early_ * (H_early / H_late).values.item()
    return (
        m_early_,
        m_late_,
        m_late_hat,
        ssh_early_1,
        ssh_late_1,
        taux_early,
        taux_late,
    )


@app.cell
def _(
    m_early_,
    m_late_,
    m_late_hat,
    np,
    plt,
    ssh_early_1,
    ssh_late_1,
    taux_early,
    taux_late,
):
    ## plot
    fig_62, axs_56 = plt.subplots(2, 1, figsize=(2, 4), layout="constrained")

    ## early
    axs_56[0].scatter(
        taux_early.stack(sample=["time", "member"]).values,
        ssh_early_1.stack(sample=["time", "member"]).values,
        s=1,
    )

    ## late
    axs_56[1].scatter(
        taux_late.stack(sample=["time", "member"]).values,
        ssh_late_1.stack(sample=["time", "member"]).values,
        s=1,
    )

    ## plot slope
    z = np.linspace(-0.035, 0.035)
    axs_56[0].plot(z, m_early_ * z, c="k", lw=1)
    axs_56[1].plot(z, m_late_ * z, c="k", lw=2)
    axs_56[1].plot(z, m_early_ * z, c="k", lw=2, ls="--")
    axs_56[1].plot(z, m_late_hat * z, c="r", lw=2, ls="--")

    ## set limits
    for _ax in axs_56:
        _ax.set_xlim([-0.04, 0.04])
        _ax.set_ylim([-20, 20])
        _ax.axhline(0, ls="--", lw=0.8, c="k")
        _ax.axvline(0, ls="--", lw=0.8, c="k")
        _ax.set_ylabel(r"Niño 3.4 SSH (cm)", size=8)
        _ax.set_yticks([-10, 0, 20])

    # ## format
    axs_56[0].set_xticks([])
    axs_56[1].set_xticks([-0.04, 0.04])
    axs_56[1].set_xlabel(r"$\tau_x$ (Niño 3.4)", size=8)


    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    spatial pattern
    """)
    return


@app.cell
def _(m_early_9, m_late_9, plt, src):
    sel_mon_13 = lambda x: x.sel(month=slice(1, 3)).mean("month")

    ## set up plot
    fig_63 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_57 = src.utils.subplots_with_proj(fig_63, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=6e2, sel=sel_mon_13)
    _cp0 = src.utils.make_contour_plot(axs_57[0, 0], m_early_9, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_57[1, 0], m_late_9, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_57[2, 0], m_late_9 - m_early_9, **dict(_contour_kwargs, amp=3e2)
    )

    # ## colorbar
    _cb_kwargs = dict(label=r"$cm~Pa^{-1}$")
    _cb0 = fig_63.colorbar(_cp0, ax=axs_57[:2], ticks=[-400, 0, 400], **_cb_kwargs)
    _cb2 = fig_63.colorbar(_cp2, ax=axs_57[-1], ticks=[-200, 0, 200], **_cb_kwargs)

    ## Niño 4 box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_57.flatten():
        src.utils.plot_nino4_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Heat flux
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared kwargs
    _kwargs = dict(x_var="sst", y_var="nhf", fn_x=src.utils.get_nino34)
    # kwargs = dict(x_var="sst", y_var="nhf")

    ## function to get slope
    _get_slope = lambda x: x.groupby("time.month").map(src.utils.regress_proj, **_kwargs)

    ## then, reconstruct regression coefficient
    m_early_10 = _get_slope(anom_early_2)
    m_late_10 = _get_slope(anom_late_2)
    return m_early_10, m_late_10


@app.cell
def _():
    # ## shared args
    # kwargs = dict(x_vars=["nino34"], y_var="vvel")

    # ## compute coefs
    # m_early = src.utils.multi_regress_bymonth(anom_early, **kwargs)["nino34"]
    # m_late = src.utils.multi_regress_bymonth(anom_late, **kwargs)["nino34"]

    # ## rename coords for plotting
    # coord_dict = dict({"lon": "longitude", "lat": "latitude"})
    # m_early = m_early.rename(coord_dict)
    # m_late = m_late.rename(coord_dict)
    return


@app.cell
def _(m_early_10, m_late_10, plot_cycle_hov, plt):
    fig_64, axs_58 = plt.subplots(1, 2, figsize=(4, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_58[0], data=m_early_10, amp=40)
    plot_cycle_hov(axs_58[0], data=m_late_10, amp=40, is_filled=False)

    ## plot difference
    _cp2 = plot_cycle_hov(axs_58[1], data=m_late_10 - m_early_10, amp=20)

    ## make it look nicer
    axs_58[1].set_yticks([])
    axs_58[0].set_yticks([1, 5, 9, 12], labels=["Jan", "May", "Sep", "Dec"])
    axs_58[0].set_ylabel("Month")
    axs_58[0].set_title(r"NHF-SST coupling")
    axs_58[1].set_title("Change")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### spatial plot
    """)
    return


@app.cell
def _(m_early_10, m_late_10, plt, src):
    # sel_mon = lambda x: x.mean("month")
    # sel_mon = lambda x: x.sel(month=slice(3, 5)).mean("month")
    sel_mon_14 = lambda x: x.sel(month=7)

    ## set up plot
    fig_65 = plt.figure(figsize=(7, 3.9), layout="constrained")
    _format_func = lambda ax,: src.utils.plot_setup_pac(ax, max_lat=20)
    axs_59 = src.utils.subplots_with_proj(fig_65, nrows=3, ncols=1, format_func=_format_func)

    _contour_kwargs = dict(amp=40, sel=sel_mon_14)
    _cp0 = src.utils.make_contour_plot(axs_59[0, 0], m_early_10, **_contour_kwargs)
    _cp1 = src.utils.make_contour_plot(axs_59[1, 0], m_late_10, **_contour_kwargs)
    _cp2 = src.utils.make_contour_plot(
        axs_59[2, 0], m_late_10 - m_early_10, **(dict(_contour_kwargs, amp=30))
    )

    ## colorbar
    _cb_kwargs = dict(label=r"$W~m^{-2}~K^{-1}$")
    _cb0 = fig_65.colorbar(_cp0, ax=axs_59[:2], ticks=[-40, 0, 40], **_cb_kwargs)
    _cb2 = fig_65.colorbar(_cp2, ax=axs_59[-1], ticks=[-30, 0, 30], **_cb_kwargs)

    ## Niño 3 box
    _box_kwargs = dict(c="k", linewidth=0.9, alpha=0.5)
    for _ax in axs_59.flatten():
        src.utils.plot_nino3_box(_ax, **_box_kwargs)
        src.utils.plot_nino34_box(_ax, **_box_kwargs)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PDF of tropical SST
    """)
    return


@app.cell
def _(Th, get_mam, np, plt, scipy, src):
    ## get data for histogram
    y = get_mam(Th["T_3_rel"])

    ## specify month idx
    m_idx = 8
    _edges = np.arange(-3, 3.25, 0.25)
    _tick_kwargs = dict(ticks=[-3, 0, 3])


    ## get data
    _y0 = y.isel(time=slice(None, 30)).values.flatten()
    _y1 = y.isel(time=slice(-30, None)).values.flatten()

    ## compute pdfs
    _pdf0, _ = src.utils.get_empirical_pdf(
        _y0,
        edges=_edges,
    )
    _pdf1, _ = src.utils.get_empirical_pdf(
        _y1,
        edges=_edges,
    )

    ## compute skewness
    s0_1 = scipy.stats.skew(_y0)
    s1_1 = scipy.stats.skew(_y1)

    fig_66, _ax = plt.subplots(figsize=(4, 3))
    _ax.stairs(_pdf0, _edges, label=f"early: skew = {s0_1:.2f}")
    _ax.stairs(_pdf1, _edges, label=f"late: skew = {s1_1:.2f}")
    _ax.axvline(0, c="k", lw=0.8, ls="--")
    _ax.legend(prop=dict(size=8))
    _ax.set_xticks(**_tick_kwargs)
    _ax.set_xlabel(r"relative SST")
    _ax.set_xlim([None, 1.5])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Re-do scatter plots
    """)
    return


@app.cell
def _(Th, anom_early_2, anom_late_2, components, get_mam, src):
    get_nhf_ = lambda x: get_mam(
        src.utils.reconstruct_fn(
            scores=x["nhf"],
            components=components["nhf"],
            fn=src.utils.get_nino34,
        )
    )

    ## get early/late NHF
    nhf_early = get_nhf_(anom_early_2)
    nhf_late = get_nhf_(anom_late_2)

    ## get early/late Th
    Th_early = get_mam(Th.sel(time=anom_early_2.time))
    Th_late = get_mam(Th.sel(time=anom_late_2.time))
    return Th_early, Th_late, nhf_early, nhf_late


@app.cell
def _(Th_early, Th_late, nhf_early, nhf_late, plt):
    ## specify font size
    fs = 8

    ## plot
    fig_67, axs_60 = plt.subplots(2, 1, figsize=(2, 4), layout="constrained")

    ## early
    axs_60[0].scatter(
        Th_early["T_34_rel"].stack(sample=["time", "member"]).values,
        nhf_early.stack(sample=["time", "member"]).values,
        s=1,
    )

    ## late
    axs_60[1].scatter(
        Th_late["T_34_rel"].stack(sample=["time", "member"]).values,
        nhf_late.stack(sample=["time", "member"]).values,
        s=1,
    )

    ## set limits
    for _ax in axs_60:
        _ax.set_xlim([-4, 1])
        _ax.set_ylim([-100, 80])
        _ax.axhline(0, ls="--", lw=0.8, c="k")
        _ax.axvline(0, ls="--", lw=0.8, c="k")
        _ax.set_ylabel(r"Niño 3.4 NHF ($W~m^{-2}$)", size=fs)

    # ## format
    axs_60[0].set_xticks([])
    axs_60[1].set_xticks([-3, 0])
    axs_60[1].set_xlabel(r"Niño 3.4 relative SST ($K$)", size=fs)


    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SST-SSH
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, src):
    ## shared kwargs
    _kwargs = dict(x_var="ssh", y_var="sst")

    ## function to get slope
    _get_slope = lambda x: x.groupby("time.month").map(src.utils.regress_proj, **_kwargs)

    ## then, reconstruct regression coefficient
    m_early_11 = _get_slope(anom_early_2)
    m_late_11 = _get_slope(anom_late_2)
    return m_early_11, m_late_11


@app.cell
def _(m_early_11, m_late_11, plot_cycle_hov, plt):
    fig_68, axs_61 = plt.subplots(1, 2, figsize=(4, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_61[0], data=m_early_11, amp=0.3)
    plot_cycle_hov(axs_61[0], data=m_late_11, amp=0.3, is_filled=False)

    ## plot difference
    _cp2 = plot_cycle_hov(axs_61[1], data=m_late_11 - m_early_11, amp=0.15)

    ## make it look nicer
    axs_61[1].set_yticks([])
    axs_61[0].set_yticks([1, 5, 9, 12], labels=["Jan", "May", "Sep", "Dec"])
    axs_61[0].set_ylabel("Month")
    axs_61[0].set_title(r"SST-SSH coupling")
    axs_61[1].set_title("Change")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $\tau_x$-SST asymmetry
    """)
    return


@app.cell
def _(src, xr):
    def regress_relu(data, is_pos=True):
        """regress on function for positive values of nino3"""

        ## stack data
        data = data.stack(sample=["member", "time"])

        ## find which indices to keep
        if is_pos:
            idx = data["nino3"] > 0
        else:
            idx = data["nino3"] < 0

        ## select valid values of idx
        data_idx = data.where(idx, other=0)

        ## do regression
        return src.utils.regress_core(Y=data_idx["taux"], X=data_idx["nino3"], dim="sample")


    def regress_relu_wrapper(data, eofs):
        """wrapper function"""

        ## get nino3
        nino3 = src.utils.reconstruct_fn(
            components=eofs["sst"].components(), scores=data["sst"], fn=src.utils.get_nino3
        )

        ## new array
        taux = xr.merge([data["taux"], nino3.rename("nino3")])

        ## compute regression coefs
        m_pos = eofs["taux"].inverse_transform(
            taux.groupby("time.month").map(regress_relu, is_pos=True)
        )

        m_neg = eofs["taux"].inverse_transform(
            taux.groupby("time.month").map(regress_relu, is_pos=False)
        )

        return m_pos - m_neg

    return (regress_relu_wrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, eofs, regress_relu_wrapper):
    asym_early = regress_relu_wrapper(anom_early_2, eofs)
    asym_late = regress_relu_wrapper(anom_late_2, eofs)
    return asym_early, asym_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot
    """)
    return


@app.cell
def _(asym_early, asym_late, plot_cycle_hov, plt):
    fig_69, axs_62 = plt.subplots(1, 2, figsize=(4, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_62[0], data=asym_early, amp=0.0075)
    # plot_cycle_hov(axs[0], data=asym_late, amp=0.0075, is_filled=False)
    _cp2 = plot_cycle_hov(axs_62[1], data=asym_late, amp=0.0075)

    ## make it look nicer
    axs_62[1].set_yticks([])
    axs_62[0].set_yticks([1, 5, 9, 12], labels=["Jan", "May", "Sep", "Dec"])
    axs_62[0].set_ylabel("Month")

    axs_62[0].set_title("Early asym.")
    axs_62[1].set_title("Late asym.")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Mixed layer
    """)
    return


@app.cell
def _(anom_early_2, anom_late_2, eq_mld_early, eq_mld_late, get_eq_mld, src):
    ## shared kwargs
    _kwargs = dict(
        x_var="sst",
        y_var="mld",
        fn_x=src.utils.get_nino34,
        fn_y=get_eq_mld,
    )

    ## function to get slope
    _get_slope = lambda x: x.groupby("time.month").map(src.utils.regress_proj, **_kwargs)

    ## then, reconstruct regression coefficient
    m_early_12 = _get_slope(anom_early_2)
    m_late_12 = _get_slope(anom_late_2)

    ## get MLD during Niño events for convenience
    mld_nino_early = eq_mld_early + m_early_12
    mld_nino_late = eq_mld_late + m_late_12
    return m_early_12, m_late_12, mld_nino_early, mld_nino_late


@app.cell
def _(eq_mld_early, eq_mld_late, m_early_12, m_late_12, plot_mld_bounds, plt):
    _sel = lambda x: x.sel(month=3)

    fig_70, axs_63 = plt.subplots(1, 2, figsize=(5, 2.3), layout="constrained")

    plot_mld_bounds(axs_63[0], clim=_sel(eq_mld_early), m=_sel(m_early_12))
    plot_mld_bounds(axs_63[1], clim=_sel(eq_mld_late), m=_sel(m_late_12))

    ## add guidelines and label
    for _ax in axs_63:
        _ax.set_ylim([75, 10])
        _ax.set_xlabel("Lon")
        for tick in [190, 240]:
            _ax.axvline(tick, c="k", lw=0.8, ls="--")

    axs_63[0].set_ylabel("Depth (m)")
    axs_63[1].set_yticks([])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Subsurface stuff
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Should we plot MLD during neutral or Niño?
    """)
    return


@app.cell
def _(eq_mld_early, eq_mld_late, mld_nino_early, mld_nino_late):
    SHOW_NINO_MLD = False

    if SHOW_NINO_MLD:
        mld_early = mld_nino_early
        mld_late = mld_nino_late
    else:
        mld_early = eq_mld_early
        mld_late = eq_mld_late


    ## function to plot MLDs
    def plot_mlds(axs, sel):
        axs[0].plot(mld_early.longitude, sel(mld_early), c="k")
        axs[1].plot(mld_late.longitude, sel(mld_late), c="k", ls="--")
        axs[2].plot(mld_early.longitude, sel(mld_early), c="k")
        axs[2].plot(mld_late.longitude, sel(mld_late), c="k", ls="--")

        return

    return mld_early, mld_late, plot_mlds


@app.cell
def _(anom, anom_sub, components_sub, prep):
    ## add components to dataset
    for _v in list(components_sub):
        if f"{_v}_comp" not in list(anom_sub):
            anom_sub[f"{_v}_comp"] = components_sub[_v]

    ## add sst to dataset
    if "sst" not in list(anom_sub):
        anom_sub["sst"] = anom["sst"].isel(time=slice(1, None))
        anom_sub["sst_comp"] = anom["sst_comp"]

    ## add indices to dataset
    if "nino34" not in list(anom_sub):
        for _n in ["nino3", "nino34", "h_w", "h"]:
            anom_sub[_n] = anom[_n]

    ## split into early/late periods
    anom_sub_early = anom_sub.isel(time=slice(None, 360))
    anom_sub_late = anom_sub.isel(time=slice(-372, -12))

    ## compute time derivatives
    anom_sub_early = prep(anom_sub_early)
    anom_sub_late = prep(anom_sub_late)
    return anom_sub_early, anom_sub_late


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### $\frac{d T}{dt}$
    """)
    return


@app.cell
def _(anom_sub_early, anom_sub_late, src):
    _kwargs = dict(x_vars=["nino34", "h_w_hat"], y_var="ddt_T")

    m_early_13 = src.utils.multi_regress_bymonth(anom_sub_early, **_kwargs)["nino34"]
    m_late_13 = src.utils.multi_regress_bymonth(anom_sub_late, **_kwargs)["nino34"]
    return m_early_13, m_late_13


@app.cell
def _(format_subsurf_axs, m_early_13, m_late_13, plot_mlds, plt, src):
    ## specify which period/month to plot
    # sel = lambda x: x.mean("month")
    _sel = lambda x: x.sel(month=7)

    fig_71, axs_64 = plt.subplots(1, 3, figsize=(8, 2.5), layout="constrained")

    for _ax, _m in zip(axs_64[:2], [m_early_13, m_late_13]):

        ## temperature
        _cp = _ax.contourf(
            _m.lon,
            _m.z_t,
            _sel(_m),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(10, 1),
            extend="both",
        )

    ## difference
    axs_64[2].contourf(
        _m.lon,
        _m.z_t,
        _sel(m_late_13 - m_early_13),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(10, 1),
        extend="both",
    )
    ## plot MLD
    plot_mlds(axs=axs_64, sel=_sel)

    ## label
    _cb = fig_71.colorbar(_cp, ax=axs_64[2], ticks=[-10, 0, 10], label=r"$K~\text{yr}^{-1}$")
    format_subsurf_axs(axs_64)
    for _ax in axs_64:
        _ax.set_ylim([100, 5])
        _ax.axvline(190, ls="--", c="w", lw=0.8)
        _ax.axvline(240, ls="--", c="w", lw=0.8)

    plt.gca()
    return


@app.cell
def _(
    eq_mld_early,
    eq_mld_late,
    format_hov_axs,
    get_ml_avg_wrapper,
    m_early_13,
    m_late_13,
    plot_cycle_hov,
    plt,
):
    ## compute
    _ml_avg_early = get_ml_avg_wrapper(m_early_13, eq_mld_early)
    _ml_avg_late = get_ml_avg_wrapper(m_late_13, eq_mld_late)

    fig_72, axs_65 = plt.subplots(1, 3, figsize=(6, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_65[0], data=_ml_avg_early, amp=7)
    _cp1 = plot_cycle_hov(axs_65[1], data=_ml_avg_late, amp=7)
    _cp2 = plot_cycle_hov(axs_65[2], data=_ml_avg_late - _ml_avg_early, amp=7)

    ## make it look nicer
    _cb = fig_72.colorbar(
        _cp0, ax=axs_65[2], ticks=[-7, 0, 7], label=r"$K~\left(\text{month}\right)^{-1}$"
    )
    format_hov_axs(axs_65)
    for _ax in axs_65:
        _ax.axhline(7, c="k", ls="--", lw=1)

    plt.gca()
    return


@app.cell
def _(
    eq_mld_early,
    eq_mld_late,
    get_ml_avg_wrapper,
    m_early_13,
    m_late_13,
    plt,
    src,
):
    ## compute
    _ml_avg_early = get_ml_avg_wrapper(m_early_13, eq_mld_early)
    _ml_avg_late = get_ml_avg_wrapper(m_late_13, eq_mld_late)
    # ml_avg_early = get_ml_avg_wrapper(m_early, 50 * xr.ones_like(eq_mld_early))
    # ml_avg_late = get_ml_avg_wrapper(m_late, 50 * xr.ones_like(eq_mld_early))

    ## make hövmöllers
    fig_73, axs_66 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(cmap="cmo.balance", levels=src.utils.make_cb_range(7, 0.7), extend="both")
    _cb_kwargs = dict(ticks=[-7, 0, 7], label=r"$m~yr^{-1}~K^{-1}$")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_66[0], _ml_avg_early, **_kwargs)
    _cb0 = fig_73.colorbar(_cp0, ax=axs_66[0], **_cb_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_66[1], _ml_avg_late, **_kwargs)
    _cb1 = fig_73.colorbar(_cp1, ax=axs_66[1], **_cb_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_66[2],
        _ml_avg_late - _ml_avg_early,
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(7, 0.7),
        extend="both",
    )

    _cb2 = fig_73.colorbar(_cp2, ax=axs_66[2], ticks=[-7, 0, 7], label=r"$K~yr^{-1}$")

    ## label
    axs_66[0].set_title("Early")
    axs_66[1].set_title("Late")
    axs_66[2].set_title("Difference")
    axs_66[-1].set_xlabel("Longitude")
    axs_66[-1].set_xticks([140, 190, 240])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### First, look at $T_{sub}'$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot

    Another idea: plot temperature advection.
    Note: upper ocean response (above ML) is actually weaker in late period: cf. contours at surface b/n early and late periods.
    """)
    return


@app.cell
def _(anom_sub_early, anom_sub_late, src):
    _kwargs = dict(x_vars=["nino34", "h_w_hat"], y_var="T")
    # kwargs = dict(x_vars=["nino34"],  y_var="T")

    m_early_14 = src.utils.multi_regress_bymonth(anom_sub_early, **_kwargs)["nino34"]
    m_late_14 = src.utils.multi_regress_bymonth(anom_sub_late, **_kwargs)["nino34"]
    return m_early_14, m_late_14


@app.cell
def _(format_subsurf_axs, m_early_14, m_late_14, plot_mlds, plt, src):
    ## specify which period/month to plot
    # sel = lambda x: x.mean("month")
    _sel = lambda x: x.sel(month=7)

    fig_74, axs_67 = plt.subplots(1, 3, figsize=(8, 2.5), layout="constrained")

    for _ax, _m in zip(axs_67[:2], [m_early_14, m_late_14]):

        ## temperature
        _cp = _ax.contourf(
            _m.lon,
            _m.z_t,
            _sel(_m),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(3, 0.3),
            extend="both",
        )

    ## difference
    axs_67[2].contourf(
        _m.lon,
        _m.z_t,
        _sel(m_late_14 - m_early_14),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(1.5, 0.15),
        extend="both",
    )
    ## plot MLD
    plot_mlds(axs=axs_67, sel=_sel)

    ## label
    _cb = fig_74.colorbar(
        _cp, ax=axs_67[2], ticks=[-3, 0, 3], label=r"$K~\left(\text{Niño 3.4}\right)^{-1}$"
    )
    format_subsurf_axs(axs_67)
    for _ax in axs_67:
        _ax.set_ylim([100, 5])
        _ax.axvline(190, ls="--", c="w", lw=0.8)
        _ax.axvline(240, ls="--", c="w", lw=0.8)

    plt.gca()
    return


@app.cell
def _(get_dT_sub, m_early_14, m_late_14, mld_early, mld_late):
    ## specify max depth for entrainment zone (relative to base of ML)
    ez_depth = 20

    ## get vertical grad
    dT_early = -get_dT_sub(Tsub=m_early_14, mld=mld_early, delta=ez_depth)
    dT_late = -get_dT_sub(Tsub=m_late_14, mld=mld_late, delta=ez_depth)
    return dT_early, dT_late


@app.cell
def _(dT_early, dT_late, plt, src):
    ## make hövmöllers
    fig_75, axs_68 = plt.subplots(3, 1, figsize=(3.5, 5), layout="constrained")

    ## kwargs
    _kwargs = dict(
        cmap="cmo.balance", levels=src.utils.make_cb_range(1e0, 1e-1), extend="both"
    )
    _cb_kwargs = dict(ticks=[-1, 0, 1], label=r"$K~/~K$")

    ## plot early
    _cp0 = src.utils.plot_cycle_hov(axs_68[0], dT_early, **_kwargs)
    _cb0 = fig_75.colorbar(_cp0, ax=axs_68[0], **_cb_kwargs)

    # ## plot late
    _cp1 = src.utils.plot_cycle_hov(axs_68[1], dT_late, **_kwargs)
    _cb1 = fig_75.colorbar(_cp1, ax=axs_68[1], **_cb_kwargs)

    ## plot difference
    _cp2 = src.utils.plot_cycle_hov(
        axs_68[2],
        dT_late - dT_early,
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(0.5e0, 0.5e-1),
        extend="both",
    )

    _cb2 = fig_75.colorbar(_cp2, ax=axs_68[2], ticks=[-0.5, 0, 0.5], label=r"$K~/~K$")

    ## label
    axs_68[0].set_title("Early")
    axs_68[1].set_title("Late")
    axs_68[2].set_title("Difference")
    axs_68[-1].set_xlabel("Longitude")
    axs_68[-1].set_xticks([140, 190, 240])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, plot $\overline{w}~\frac{\partial T'}{\partial z}$
    """)
    return


@app.cell
def _(clim_sub_early, clim_sub_late, get_wdTdz, m_early_14, m_late_14):
    adv_early = get_wdTdz(w=clim_sub_early["w"], T=m_early_14)
    adv_late = get_wdTdz(w=clim_sub_late["w"], T=m_late_14)
    return adv_early, adv_late


@app.cell
def _(adv_early, adv_late, format_subsurf_axs, plot_mlds, plt, src):
    ## specify which period/month to plot
    _sel = lambda x: x.sel(month=8)

    fig_76, axs_69 = plt.subplots(1, 3, figsize=(8, 2.5), layout="constrained")

    for _ax, adv in zip(axs_69[:2], [adv_early, adv_late]):

        ## temperature
        _cp = _ax.contourf(
            adv.lon,
            adv.z_t,
            _sel(adv),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(2, 0.2),
            extend="both",
        )

    ## difference
    axs_69[2].contourf(
        adv.lon,
        adv.z_t,
        _sel(adv_late - adv_early),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(1, 0.1),
        extend="both",
    )

    ## plot MLD
    plot_mlds(axs_69, sel=_sel)

    ## set ax limit and plot Niño 3.4 bounds
    _cb = fig_76.colorbar(
        _cp, ax=axs_69[2], ticks=[-2, 0, 2], label=r"$K~\left(\text{month}\right)^{-1}$"
    )
    format_subsurf_axs(axs_69)
    for _ax in axs_69:
        _ax.set_ylim([100, 5])
        _ax.axvline(190, ls="--", c="w", lw=0.8)
        _ax.axvline(240, ls="--", c="w", lw=0.8)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Integrate over mixed layer
    """)
    return


@app.cell
def _(
    adv_early,
    adv_late,
    eq_mld_early,
    eq_mld_late,
    format_hov_axs,
    get_ml_avg_wrapper,
    plot_cycle_hov,
    plt,
):
    ## compute
    _ml_avg_early = get_ml_avg_wrapper(adv_early, eq_mld_early, delta=10)
    _ml_avg_late = get_ml_avg_wrapper(adv_late, eq_mld_late, delta=10)
    # ml_avg_early = get_ml_avg_wrapper(adv_early, 70 * xr.ones_like(eq_mld_early))
    # ml_avg_late = get_ml_avg_wrapper(adv_late, 70 * xr.ones_like(eq_mld_early))

    fig_77, axs_70 = plt.subplots(1, 3, figsize=(6, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_70[0], data=_ml_avg_early, amp=0.5)
    _cp1 = plot_cycle_hov(axs_70[1], data=_ml_avg_late, amp=0.5)
    _cp2 = plot_cycle_hov(axs_70[2], data=_ml_avg_late - _ml_avg_early, amp=0.25)

    ## make it look nicer
    _cb = fig_77.colorbar(
        _cp0, ax=axs_70[2], ticks=[-0.4, 0, 0.4], label=r"$K~\left(\text{month}\right)^{-1}$"
    )
    format_hov_axs(axs_70)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Next, look at $w'$
    """)
    return


@app.cell
def _(
    anom_sub_early,
    anom_sub_late,
    clim_sub_early,
    clim_sub_late,
    get_wdTdz,
    src,
):
    _kwargs = dict(x_vars=["nino34", "h_w_hat"], y_var="w")
    # kwargs = dict(x_vars=["nino34"],  y_var="w")

    m_early_15 = src.utils.multi_regress_bymonth(anom_sub_early, **_kwargs)["nino34"]
    m_late_15 = src.utils.multi_regress_bymonth(anom_sub_late, **_kwargs)["nino34"]

    ## get advection term
    # wT_early = get_wT(w=clim_sub_early["w"], T=m_early)
    # wT_late = get_wT(w=clim_sub_late["w"], T=m_late)
    adv_early_1 = get_wdTdz(T=clim_sub_early["T"], w=m_early_15)
    adv_late_1 = get_wdTdz(T=clim_sub_late["T"], w=m_late_15)
    return adv_early_1, adv_late_1, m_early_15, m_late_15


@app.cell
def _(format_subsurf_axs, m_early_15, m_late_15, plot_mlds, plt, src):
    ## specify which period/month to plot
    # sel = lambda x: x.mean("month")
    _sel = lambda x: x.sel(month=7)

    fig_78, axs_71 = plt.subplots(1, 3, figsize=(10, 2.5), layout="constrained")

    for _ax, _m in zip(axs_71[:2], [m_early_15, m_late_15]):

        ## temperature
        _cp = _ax.contourf(
            _m.lon,
            _m.z_w_top,
            _sel(_m),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(15, 1.5),
            extend="both",
        )

    ## difference
    axs_71[2].contourf(
        _m.lon,
        _m.z_w_top,
        _sel(m_late_15 - m_early_15),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(15, 1.5),
        extend="both",
    )

    ## plot MLD
    plot_mlds(axs=axs_71, sel=_sel)

    ## set ax limit and plot Niño 3.4 bounds
    _label = r"$m~\left(\text{mo}\right)^{-1}~\left(\text{Niño 3.4}\right)^{-1}$"
    _cb = fig_78.colorbar(_cp, ax=axs_71[2], ticks=[-15, 0, 15], label=_label)
    format_subsurf_axs(axs_71)
    for _ax in axs_71:
        _ax.set_ylim([150, 5])
        _ax.axvline(190, ls="--", c="w", lw=0.8)
        _ax.axvline(240, ls="--", c="w", lw=0.8)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot advection, $w'\frac{\partial \overline{T}}{\partial z}$
    """)
    return


@app.cell
def _(adv_early_1, adv_late_1, format_subsurf_axs, plot_mlds, plt, src):
    ## specify which period/month to plot
    _sel = lambda x: x.sel(month=3)

    fig_79, axs_72 = plt.subplots(1, 3, figsize=(10, 2.5), layout="constrained")

    for _ax, adv_1 in zip(axs_72[:2], [adv_early_1, adv_late_1]):

        ## temperature
        _cp = _ax.contourf(
            adv_1.lon,
            adv_1.z_t,
            _sel(adv_1),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(2, 0.2),
            extend="both",
        )

    ## difference
    axs_72[2].contourf(
        adv_1.lon,
        adv_1.z_t,
        _sel(adv_late_1 - adv_early_1),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(1, 0.1),
        extend="both",
    )

    ## plot MLD
    plot_mlds(axs=axs_72, sel=_sel)

    ## set ax limit and plot Niño 3.4 bounds
    _cb = fig_79.colorbar(
        _cp, ax=axs_72[2], ticks=[-2, 0, 2], label=r"$K~\left(\text{month}\right)^{-1}$"
    )
    format_subsurf_axs(axs_72)
    for _ax in axs_72:
        _ax.set_ylim([150, 5])
        _ax.axvline(190, ls="--", c="w", lw=0.8)
        _ax.axvline(240, ls="--", c="w", lw=0.8)

    plt.gca()
    return (adv_1,)


@app.cell
def _(
    adv_early_1,
    adv_late_1,
    eq_mld_early,
    eq_mld_late,
    format_hov_axs,
    get_ml_avg_wrapper,
    plot_cycle_hov,
    plt,
):
    ## compute
    _ml_avg_early = get_ml_avg_wrapper(adv_early_1, eq_mld_early, delta=10)
    _ml_avg_late = get_ml_avg_wrapper(adv_late_1, eq_mld_late, delta=10)
    # ml_avg_early = get_ml_avg_wrapper(adv_early, 40 * xr.ones_like(eq_mld_early))
    # ml_avg_late = get_ml_avg_wrapper(adv_late, 40 * xr.ones_like(eq_mld_early))

    ## plot
    fig_80, axs_73 = plt.subplots(1, 3, figsize=(6, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_73[0], data=_ml_avg_early, amp=0.5)
    _cp1 = plot_cycle_hov(axs_73[1], data=_ml_avg_late, amp=0.5)
    _cp2 = plot_cycle_hov(axs_73[2], data=_ml_avg_late - _ml_avg_early, amp=0.25)

    ## label
    format_hov_axs(axs_73)
    _cb = fig_80.colorbar(
        _cp0, ax=axs_73[2], ticks=[-0.4, 0, 0.4], label=r"$K~\left(\text{month}\right)^{-1}$"
    )

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Subsurface zonal advection
    """)
    return


@app.cell
def _(
    anom_sub_early,
    anom_sub_late,
    clim_sub_early,
    clim_sub_late,
    get_u_adv,
    src,
    xr,
):
    _kwargs = dict(x_vars=["nino34", "h_w_hat"])

    ## compute regression coeff.
    u_early = src.utils.multi_regress_bymonth(anom_sub_early, y_var="u", **_kwargs)["nino34"]
    u_late = src.utils.multi_regress_bymonth(anom_sub_late, y_var="u", **_kwargs)["nino34"]

    ## w
    w_early = src.utils.multi_regress_bymonth(anom_sub_early, y_var="w", **_kwargs)["nino34"]
    w_late = src.utils.multi_regress_bymonth(anom_sub_late, y_var="w", **_kwargs)["nino34"]

    ## T
    T_early = src.utils.multi_regress_bymonth(anom_sub_early, y_var="T", **_kwargs)["nino34"]
    T_late = src.utils.multi_regress_bymonth(anom_sub_late, y_var="T", **_kwargs)["nino34"]


    ## put results in xarray
    res_early = xr.merge(
        [
            clim_sub_early["T"].rename("Tbar"),
            clim_sub_early["u"].rename("ubar"),
            clim_sub_early["w"].rename("wbar"),
            u_early.rename("uprime"),
            T_early.rename("Tprime"),
            w_early.rename("wprime"),
        ]
    )
    res_late = xr.merge(
        [
            clim_sub_late["T"].rename("Tbar"),
            clim_sub_late["u"].rename("ubar"),
            clim_sub_late["w"].rename("wbar"),
            T_late.rename("Tprime"),
            u_late.rename("uprime"),
            w_late.rename("wprime"),
        ]
    )

    ## compute advection terms
    res_early["adv_uprime"] = get_u_adv(T=res_early["Tbar"], u=res_early["uprime"])
    res_early["adv_Tprime"] = get_u_adv(T=res_early["Tprime"], u=res_early["ubar"])
    res_early["adv_total"] = res_early["adv_uprime"] + res_early["adv_Tprime"]
    res_late["adv_uprime"] = get_u_adv(T=res_late["Tbar"], u=res_late["uprime"])
    res_late["adv_Tprime"] = get_u_adv(T=res_late["Tprime"], u=res_late["ubar"])
    res_late["adv_total"] = res_late["adv_uprime"] + res_late["adv_Tprime"]
    return res_early, res_late


@app.cell
def _(
    format_subsurf_axs,
    get_dx,
    np,
    plot_mlds,
    plt,
    res_early,
    res_late,
    src,
):
    ## specify which period/month to plot
    _sel = lambda x: x.sel(month=8)

    ## get aspect ratio (for scaling arrows)
    _dz = 300  # units: m
    _dx = get_dx(lat_deg=0, dlon_deg=150)
    _aspect = _dx / _dz

    fig_81, axs_74 = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")

    for _ax, _res in zip(axs_74[:2], [res_early, res_late]):

        ## temperature
        _ax.contourf(
            _res.lon,
            _res.z_t,
            _sel(_res["Tbar"]),
            cmap="cmo.thermal",
            levels=np.arange(10, 34, 2),
            extend="both",
        )

        ## u and w
        _ax.quiver(
            _res.lon.values[::4],
            _res.z_t.values[::2],
            _sel(_res["uprime"]).values[::2, ::4],
            _sel(_res["wprime"]).values[::2, ::4] * _aspect,
            pivot="middle",
            alpha=0.7,
            scale=1.5e7,
        )


    ## plot difference
    diff_4 = res_late - res_early

    ## temperature
    axs_74[2].contourf(
        diff_4.lon,
        diff_4.z_t,
        _sel(diff_4["Tbar"]),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(5, 0.5),
        extend="both",
    )

    ## vertical velocity
    ## u and w
    axs_74[2].quiver(
        diff_4.lon.values[::4],
        diff_4.z_t.values[::2],
        _sel(diff_4["uprime"]).values[::2, ::4],
        _sel(diff_4["wprime"]).values[::2, ::4] * _aspect,
        pivot="middle",
        alpha=0.7,
        scale=1.5e7,
    )

    ## label
    plot_mlds(axs=axs_74, sel=_sel)
    format_subsurf_axs(axs_74)
    for _ax in axs_74:
        _ax.set_ylim([305, 10])


    plt.gca()
    return


@app.cell
def _(format_subsurf_axs, get_dx, plot_mlds, plt, res_early, res_late, src):
    ## specify which period/month to plot
    _sel = lambda x: x.sel(month=7)

    ## get aspect ratio (for scaling arrows)
    _dz = 300  # units: m
    _dx = get_dx(lat_deg=0, dlon_deg=150)
    _aspect = _dx / _dz

    fig_82, axs_75 = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")

    for _ax, _res in zip(axs_75[:2], [res_early, res_late]):

        ## temperature
        _ax.contourf(
            _res.lon,
            _res.z_t,
            _sel(_res["Tprime"]),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(3, 0.3),
            extend="both",
        )

        ## u and w
        _ax.quiver(
            _res.lon.values[::4],
            _res.z_t.values[::2],
            _sel(_res["ubar"]).values[::2, ::4],
            _sel(_res["wbar"]).values[::2, ::4] * _aspect,
            pivot="middle",
            alpha=0.7,
            scale=4e7,
        )


    ## plot difference
    diff_5 = res_late - res_early

    ## temperature
    axs_75[2].contourf(
        diff_5.lon,
        diff_5.z_t,
        _sel(diff_5["Tprime"]),
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(3, 0.3),
        extend="both",
    )

    ## vertical velocity
    ## u and w
    axs_75[2].quiver(
        diff_5.lon.values[::4],
        diff_5.z_t.values[::2],
        _sel(diff_5["ubar"]).values[::2, ::4],
        _sel(diff_5["wbar"]).values[::2, ::4] * _aspect,
        pivot="middle",
        alpha=0.7,
        scale=2e7,
    )

    ## label
    plot_mlds(axs=axs_75, sel=_sel)
    format_subsurf_axs(axs_75)
    for _ax in axs_75:
        _ax.set_ylim([305, 10])


    plt.gca()
    return


@app.cell
def _(adv_1, format_subsurf_axs, plot_mlds, plt, res_early, res_late, src):
    ## specify which variable to plot
    _plot_var = "adv_Tprime"

    ## specify which period/month to plot
    _sel = lambda x: x.sel(month=7)

    fig_83, axs_76 = plt.subplots(1, 3, figsize=(10, 2.5), layout="constrained")

    for _ax, _res in zip(axs_76[:2], [res_early, res_late]):

        ## temperature
        _cp = _ax.contourf(
            adv_1.lon,
            adv_1.z_t,
            _sel(_res[_plot_var]),
            cmap="cmo.balance",
            levels=src.utils.make_cb_range(5, 0.5),
            extend="both",
        )

    ## difference
    axs_76[2].contourf(
        adv_1.lon,
        adv_1.z_t,
        _sel(res_late - res_early)[_plot_var],
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(2, 0.2),
        extend="both",
    )

    ## plot MLD
    plot_mlds(axs=axs_76, sel=_sel)

    ## set ax limit and plot Niño 3.4 bounds
    _cb = fig_83.colorbar(
        _cp, ax=axs_76[2], ticks=[-5, 0, 5], label=r"$K~\left(\text{yr}\right)^{-1}$"
    )
    format_subsurf_axs(axs_76)
    for _ax in axs_76:
        _ax.set_ylim([150, 10])
        _ax.axvline(190, ls="--", c="w", lw=0.8)
        _ax.axvline(240, ls="--", c="w", lw=0.8)

    plt.gca()
    return


@app.cell
def _(
    eq_mld_early,
    eq_mld_late,
    format_hov_axs,
    get_ml_avg_wrapper,
    plot_cycle_hov,
    plt,
    res_early,
    res_late,
):
    ## specify plot variable
    _plot_var = "adv_total"

    ## compute
    _ml_avg_early = get_ml_avg_wrapper(res_early[_plot_var], eq_mld_early)
    _ml_avg_late = get_ml_avg_wrapper(res_late[_plot_var], eq_mld_late)
    # ml_avg_early = get_ml_avg_wrapper(res_early[plot_var], 30.5 * xr.ones_like(eq_mld_early))
    # ml_avg_late = get_ml_avg_wrapper(res_late[plot_var], 5.5 * xr.ones_like(eq_mld_late))

    ## plot
    fig_84, axs_77 = plt.subplots(1, 3, figsize=(6, 2.5), layout="constrained")

    ## plot data
    _cp0 = plot_cycle_hov(axs_77[0], data=_ml_avg_early, amp=2)
    _cp1 = plot_cycle_hov(axs_77[1], data=_ml_avg_late, amp=2)
    _cp2 = plot_cycle_hov(axs_77[2], data=_ml_avg_late - _ml_avg_early, amp=2)

    ## label
    format_hov_axs(axs_77)
    _cb = fig_84.colorbar(
        _cp0, ax=axs_77[2], ticks=[-3, 0, 3], label=r"$K~\left(\text{yr}\right)^{-1}$"
    )


    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Change in zonal gradient
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute zonal gradient
    """)
    return


@app.cell
def _(src):
    def get_zonal_grad(x):
        """
        Function to compute zonal gradient. Ref: Fig 7 in Maher et al, 2023.
        """

        ## outlines for boxes
        ebox_coords = dict(longitude=slice(210, 270), latitude=slice(-5, 5))
        wbox_coords = dict(longitude=slice(120, 180), latitude=slice(-5, 5))

        ## compute box averages
        ebox_avg = src.utils.spatial_avg(x.sel(ebox_coords))
        wbox_avg = src.utils.spatial_avg(x.sel(wbox_coords))

        return ebox_avg - wbox_avg

    return (get_zonal_grad,)


@app.cell
def _(components, forced, get_zonal_grad, src):
    ## compute zonal gradient
    zonal_grad_forced = src.utils.reconstruct_fn(components, forced, fn=get_zonal_grad)

    ## sep. time into year and month
    zonal_grad_forced_bymonth = src.utils.unstack_month_and_year(zonal_grad_forced)

    ## get change from initial climatology
    _clim = zonal_grad_forced_bymonth.isel(year=slice(None, 30)).mean("year")
    zonal_grad_change = zonal_grad_forced_bymonth - _clim
    return (zonal_grad_change,)


@app.cell
def _(plt, src, zonal_grad_change):
    ## setup plot
    fig_85, _ax = plt.subplots(figsize=(2, 4), layout="constrained")

    ## plot data
    _kwargs = dict(levels=src.utils.make_cb_range(2, 0.2), cmap="cmo.balance")
    plot_data = _ax.contourf(
        zonal_grad_change.month, zonal_grad_change.year, zonal_grad_change["sst"], **_kwargs
    )

    ## label
    _ax.set_ylabel("Year")
    _ax.set_xlabel("Month")
    _ax.set_xticks([1, 6, 12])
    _ax.set_title(r"$\Delta \left(\partial_x T\right)$")
    _ax.set_ylim([1950, None])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Look at mean state-dependence
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compare Bjerknes growth rate to zonal gradient
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load $T,h$ data
    """)
    return


@app.cell
def _(DATA_FP, pathlib, xr):
    ## MPI data
    mpi_load_fp = pathlib.Path(DATA_FP, "mpi_Th", "Th.nc")
    Th_1 = xr.open_dataset(mpi_load_fp)
    return (Th_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Fit RO to $T,h$ data
    """)
    return


@app.cell
def _(Th_1, XRO, src):
    ## get subset of data
    Th_sub = Th_1.sel(time=slice("1979", "2024"))

    ## initialize model
    model = XRO(ncycle=12, ac_order=3, is_forward=True)

    ## fit to individual ensemble members
    _kwargs = dict(model=model, T_var="T_3", h_var="h_w", verbose=True)
    _, fits = src.utils.get_RO_ensemble(Th_sub, **_kwargs)

    ## extract parameters
    params_1 = model.get_RO_parameters(fits)
    return (params_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Look at intra-ensemble spread
    $\partial_x T$ vs. BJ index
    Compare to Maher et al (2023)
    """)
    return


@app.cell
def _(anom, components, get_zonal_grad, params_1, plt, src):
    ## get subset of data to look at
    anom_ = anom.sel(time=slice("1979", "2024"))

    ## compute zonal gradient at every time step
    zonal_grad = src.utils.reconstruct_fn(components, anom_, fn=get_zonal_grad)

    ## get monthly avg
    zonal_grad_by_month = zonal_grad.groupby("time.month").mean()

    fig_86, _ax = plt.subplots(figsize=(2, 2))
    _ax.scatter(zonal_grad_by_month["sst"].mean("month"), params_1["R"].mean("cycle"), s=10)
    _ax.set_xlabel("Zonal SST gradient")
    _ax.set_ylabel("Bjerknes feedback")
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
