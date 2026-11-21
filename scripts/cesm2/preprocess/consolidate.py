import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Consolidate
    Merge EOF data for faster loading
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
    import xarray as xr
    import pathlib
    import src.utils
    import os
    import numpy as np

    return np, os, pathlib, src, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Functions
    """)
    return


@app.cell
def _(np, os, pathlib, src, xr):
    def fix_lon_coord(data_sub):
        """fix longitude coordinate on subsurface data"""

        data_sub = data_sub.assign_coords({"nlon": data_sub.lon.isel(z_t=0).values})
        data_sub = data_sub.drop_vars("lon").rename({"nlon": "lon"})

        return data_sub


    def convert_cm_to_m_helper(data, z_coord_name):
        """convert z-coord from cm to m"""
        return data.assign_coords({z_coord_name: data[z_coord_name].values / 100})


    def convert_cm_to_m(data):
        """convert all z-coords from cm to m"""

        for z_coord in ["z_t", "z_w_top"]:
            if z_coord in data.coords:
                data = convert_cm_to_m_helper(data, z_coord_name=z_coord)

        return data


    def load_sub():
        """load subsurface data"""

        ## path to EOF data
        eofs_fp = pathlib.Path(os.environ["DATA_FP"], "cesm")

        ## variables to load (and how to rename them)
        names = [
            "temp",
            "wvel",
            "uvel_sub",
        ]
        newnames = ["T", "w", "u"]

        ## load the EOFs
        load_var = lambda x: src.utils.load_eofs(pathlib.Path(eofs_fp, f"eofs_{x}.nc"))
        eofs_sub = {y: load_var(x) for (y, x) in zip(newnames, names)}

        ## for convenience, put spatial patterns / components in single dataset
        components_sub = xr.merge(
            [eofs_.components().rename(y) for (y, eofs_) in eofs_sub.items()]
        )

        ## fix longitude coord
        components_sub = fix_lon_coord(components_sub)

        ## rename w grid
        w_comp = components_sub["w"].rename({"z_w_top": "z_t"})
        w_comp = w_comp.assign_coords({"z_t": components_sub.z_t})

        ## update in xarray
        components_sub = components_sub.drop_vars("w")
        components_sub["w"] = w_comp

        # reset member dimension so they all match (NHF labeled differently...)
        member_coord = dict(member_id=np.arange(100))
        get_scores = lambda x, n: x.scores().assign_coords(member_coord).rename(n)
        scores_sub = xr.merge([get_scores(eofs_, n) for (n, eofs_) in eofs_sub.items()])

        ## convert z coords from cm to m
        components_sub = convert_cm_to_m(components_sub)

        ## convert u and w from cm/s to m/month

        # conversion factors
        m_per_cm = 1 / 100
        s_per_day = 86400
        s_per_month = s_per_day * 30

        # do conversion
        scores_sub["w"] = scores_sub["w"] * m_per_cm * s_per_month
        scores_sub["u"] = scores_sub["u"] * m_per_cm * s_per_month

        return components_sub, scores_sub


    def load_surf():
        """Load surface -- or near-surface -- data"""

        ## path to EOF data
        eofs_fp = pathlib.Path(os.environ["DATA_FP"], "cesm")

        ## variables to load (and how to rename them)
        names = ["tos", "zos", "tauu", "tauv", "nhf", "mlotst", "pr", "z20"]
        newnames = ["sst", "ssh", "taux", "tauy", "nhf", "mld", "pr", "z20"]

        # ## load the EOFs
        load_var = lambda x: src.utils.load_eofs(pathlib.Path(eofs_fp, f"eofs_{x}.nc"))
        eofs = {y: load_var(x) for (y, x) in zip(newnames, names)}

        ## for convenience, put spatial patterns / components in single dataset
        components = xr.merge([eofs_.components().rename(y) for (y, eofs_) in eofs.items()])

        # reset member dimension so they all match (NHF labeled differently...)
        member_coord = dict(member=np.arange(100))
        get_scores = lambda x, n: x.scores().assign_coords(member_coord).rename(n)
        scores = xr.merge([get_scores(eofs_, n) for (n, eofs_) in eofs.items()])

        ## convert ssh from m to cm
        scores["ssh"].values *= 100

        ## convert from stress on atm to stress on ocn
        scores["taux"].values *= -1

        ## convert MLD from cm to m
        scores["mld"] = scores["mld"] / 100

        return components, scores


    def load_vel():
        """Load (ocean) surface velocity data"""

        ## path to EOF data
        eofs_fp = pathlib.Path(os.environ["DATA_FP"], "cesm")

        ## load advection data
        uvel_eofs = xr.open_dataset(eofs_fp / "eofs_uvel.nc")
        vvel_eofs = xr.open_dataset(eofs_fp / "eofs_vvel.nc")

        ## func to merge u and v data
        merge = lambda u, v: xr.merge(
            [
                x.rename(n).drop_vars(["variable", "z_t"])
                for x, n in zip([u, v], ["uvel", "vvel"])
            ]
        )

        ## merge component data
        vel_comps = merge(uvel_eofs.components, vvel_eofs.components)

        ## merge scores
        vel_scores = merge(uvel_eofs.scores, vvel_eofs.scores)
        vel_scores = vel_scores.rename({"member_id": "member"})
        vel_scores = vel_scores.assign_coords(dict(member=np.arange(100)))

        # conversion factors
        m_per_cm = 1 / 100
        s_per_day = 86400
        s_per_month = s_per_day * 30

        # do conversion
        vel_scores["uvel"] = vel_scores["uvel"] * m_per_cm * s_per_month
        vel_scores["vvel"] = vel_scores["vvel"] * m_per_cm * s_per_month

        return vel_comps, vel_scores


    def regrid_pop(pop_data, target_grid):
        """regrid POP data to target grid"""

        ## get names of old/new coords
        if "lat" in pop_data:
            rename_dict = dict(lat="latitude", lon="longitude")
        else:
            rename_dict = dict(lon="longitude")

        ## rename coords
        pop_data = pop_data.rename(rename_dict)

        ## regrid
        return pop_data.interp_like(target_grid)


    def load_and_merge():
        """load data components, merge, and save to specified filepath"""

        ## load data from different datasets
        components_sub, scores_sub = load_sub()
        components, scores = load_surf()
        components_vel, scores_vel = load_vel()

        ## regrid so everything matches MMLEA
        components_vel = regrid_pop(components_vel, components[["latitude", "longitude"]])
        components_sub = regrid_pop(components_sub, components["longitude"])

        ## rename subsurface scores to match other data
        scores_sub = scores_sub.rename({"member_id": "member"})

        ## merge variables
        components_all = xr.merge([components, components_vel, components_sub])
        scores_all = xr.merge([scores, scores_vel, scores_sub])

        return components_all, scores_all


    def compute_forced_anom(save_dir):
        """Compute forced and anomaly components, and save to file"""

        ## get filepaths
        forced_fp = save_dir / "forced.nc"
        anom_fp = save_dir / "anom.nc"

        ## try loading pre-computed data
        try:
            forced = xr.open_dataset(forced_fp)
            anom = xr.open_dataset(anom_fp)

        except:

            ## load data
            components, scores = load_and_merge()

            ## separate scores into forced and anomalies
            forced, anom = src.utils.separate_forced(scores)

            ## new names for components to avoid merge conflict
            old_names = list(components)
            rename_dict = {n: f"{n}_comp" for n in old_names}

            ## add component information to scores
            forced = xr.merge([forced, components.rename(rename_dict)])
            anom = xr.merge([anom, components.rename(rename_dict)])

            ## drop attrs (to avoid issues saving)
            forced = forced.drop_attrs()
            anom = anom.drop_attrs()

            ## save to file
            forced.to_netcdf(forced_fp)
            anom.to_netcdf(anom_fp)

        return forced, anom


    def compute_budget_forced_anom(save_dir, target_grid=None):
        """Compute forced and anomaly components, and save to file"""

        ## get filepaths
        forced_fp = save_dir / "budg_forced.nc"
        anom_fp = save_dir / "budg_anom.nc"

        ## try loading pre-computed data
        try:
            forced = xr.open_dataset(forced_fp)
            anom = xr.open_dataset(anom_fp)

        except:

            ## load to memory
            forced, anom = load_budget_data(target_grid)

            ## save to file
            forced.to_netcdf(forced_fp)
            anom.to_netcdf(anom_fp)

        return forced, anom


    def load_budget_data(target_grid):
        """load ground truth heat budget data"""

        def load_var(varname):
            """load variable from prepped folder"""

            ## get directory
            budg_dir = pathlib.Path(os.environ["DATA_FP"], "cesm", f"{varname}_temp")

            ## open data
            data = xr.open_mfdataset(
                sorted(list(budg_dir.glob("*.nc"))),
                concat_dim="member",
                combine="nested",
                parallel=True,
            )

            return data.assign_coords({"member": np.arange(100)})

        ## load data
        budget_data = xr.merge([load_var(v) for v in ["adv", "ddt_T"]])

        ## align times (off-by-one in POP)
        budget_data = src.utils.align_pop_times(budget_data, pop_vars=list(budget_data))

        ## get difference
        budget_data["diff"] = budget_data["TEND_TEMP"] - budget_data["ADV_3D_TEMP"]

        ## convert z coord from cm to m, and unit from K/s to K/mo
        M_PER_CM = 1e-2
        SEC_PER_MO = 8.64e4 * 30

        ## convert from (i) cm to m and (ii) K/s to K/mo
        budget_data = budget_data.assign_coords({"z_t": budget_data.z_t * M_PER_CM})
        budget_data = budget_data * SEC_PER_MO

        ## fix longitude coordinate
        budget_data = budget_data.assign_coords({"nlon": budget_data.lon.values})
        budget_data = budget_data.drop_vars("lon").rename({"nlon": "longitude"})

        ## interpolate
        budget_data = budget_data.interp_like(target_grid)

        # ## separate forced/anomalies
        forced_bud, anom_bud = src.utils.separate_forced(budget_data)

        return forced_bud, anom_bud

    return (
        compute_budget_forced_anom,
        compute_forced_anom,
        convert_cm_to_m,
        regrid_pop,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute
    """)
    return


@app.cell
def _(compute_budget_forced_anom, compute_forced_anom, os, pathlib):
    ## save filepath
    _SAVE_DIR = pathlib.Path(os.environ['DATA_FP'], 'cesm', 'consolidated')
    _forced, _anom = compute_forced_anom(save_dir=_SAVE_DIR)
    ## non-budget data
    kwargs = dict(target_grid=_anom[['longitude', 'z_t']])
    ## budget data
    budg_forced, budg_anom = compute_budget_forced_anom(save_dir=_SAVE_DIR, **kwargs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## "Wide" ocean data
    """)
    return


@app.cell
def _(convert_cm_to_m, np, os, pathlib, regrid_pop, src, xr):
    def fix_lon_coord_05(data_sub):
        """fix longitude coordinate on subsurface data"""

        data_sub = data_sub.assign_coords(
            {"nlon": data_sub.lon.isel(z_t=0, z_w_top=0).values}
        )
        data_sub = data_sub.drop_vars("lon").rename({"nlon": "lon"})

        return data_sub


    def load_var(eofs_fp, name, newname, is_comp=False):

        ## load data
        eofs_ = xr.open_dataset(pathlib.Path(eofs_fp, f"eofs_{name}.nc"))
        eofs_ = eofs_.drop_vars("variable")

        ## get components or scores
        if is_comp:
            return eofs_["components"].rename(newname)
        else:
            return eofs_["scores"].rename(newname)


    def load_vars(eofs_fp, names, newnames, is_comp=False):

        eofs_ = []
        for name, newname in zip(names, newnames):
            eofs_.append(load_var(eofs_fp, name, newname, is_comp))

        return xr.merge(eofs_)


    def load_sub_05_helper(target_lon):
        """load subsurface data"""

        ## path to EOF data
        eofs_fp = pathlib.Path(os.environ["DATA_FP"], "cesm")

        ## variables to load (and how to rename them)
        names = ["temp_05", "wvel_05", "uvel_05"]
        newnames = ["T", "w", "u"]

        ## load data
        kwargs = dict(eofs_fp=eofs_fp, names=names, newnames=newnames)
        components = load_vars(**kwargs, is_comp=True)
        scores = load_vars(**kwargs, is_comp=False)

        ## rename w grid
        w_comp = components["w"].rename({"z_w_top": "z_t"})
        w_comp = w_comp.assign_coords({"z_t": components.z_t})

        ## replace in components
        components = components.drop_vars(["w", "z_w_top"])
        components["w"] = w_comp

        ## fix longitude coord
        components = fix_lon_coord_05(components)

        ## rename scores to match other data
        scores = scores.rename({"member_id": "member"})

        ## fix lon coord
        components = regrid_pop(components, target_lon)

        # reset member dimension so they all match (NHF labeled differently...)
        scores = scores.assign_coords(dict(member=np.arange(100)))

        ## convert z coords from cm to m
        components = convert_cm_to_m(components)

        ## convert u and w from cm/s to m/month

        # conversion factors
        m_per_cm = 1 / 100
        s_per_day = 86400
        s_per_month = s_per_day * 30

        # do conversion
        scores["w"] = scores["w"] * m_per_cm * s_per_month
        scores["u"] = scores["u"] * m_per_cm * s_per_month

        return components, scores


    def load_sub_05(save_dir, target_lon):
        """load/save subsurface data"""

        ## specify save filepaths
        forced_fp = save_dir / "forced.nc"
        anom_fp = save_dir / "anom.nc"

        if forced_fp.is_file():
            forced = xr.open_dataset(forced_fp)
            anom = xr.open_dataset(forced_fp)

        else:

            ## load data
            components, scores = load_sub_05_helper(target_lon)

            ## separate scores into forced and anomalies
            forced, anom = src.utils.separate_forced(scores)

            ## new names for components to avoid merge conflict
            old_names = list(components)
            rename_dict = {n: f"{n}_comp" for n in old_names}

            ## add component information to scores
            forced = xr.merge([forced, components.rename(rename_dict)])
            anom = xr.merge([anom, components.rename(rename_dict)])

            ## drop attrs (to avoid issues saving)
            forced = forced.drop_attrs()
            anom = anom.drop_attrs()

            ## save to file
            forced.to_netcdf(forced_fp)
            anom.to_netcdf(anom_fp)

        return forced, anom

    return (load_sub_05,)


@app.cell
def _(load_sub_05, os, pathlib, src):
    save_dir = pathlib.Path(os.environ['DATA_FP'], 'cesm', 'consolidated_05')
    _forced, _ = src.utils.load_consolidated()
    _forced, _anom = load_sub_05(target_lon=_forced.longitude, save_dir=save_dir)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flux data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Funcs
    """)
    return


@app.cell
def _(np, os, pathlib, src, xr):
    def load_flux():
        """Load components of surface heat flux"""

        ## path to EOF data
        eofs_fp = pathlib.Path(os.environ["DATA_FP"], "cesm")

        ## variables to load
        names = ["fsns", "flns", "lhflx", "fsnsc"]

        # ## load the EOFs
        load_var = lambda x: src.utils.load_eofs(pathlib.Path(eofs_fp, f"eofs_{x}.nc"))
        eofs = {x: load_var(x) for x in names}

        ## for convenience, put spatial patterns / components in single dataset
        components = xr.merge([eofs_.components().rename(y) for (y, eofs_) in eofs.items()])

        ## reset member dimension so they all match (NHF labeled differently...)
        member_coord = dict(member=np.arange(100))
        get_scores = lambda x, n: x.scores().assign_coords(member_coord).rename(n)
        scores = xr.merge([get_scores(eofs_, n) for (n, eofs_) in eofs.items()])

        ## fix off-by-one in time label
        scores = src.utils.align_pop_times(scores, pop_vars=list(scores))

        ## separate scores into forced and anomalies
        forced, anom = src.utils.separate_forced(scores)

        ## new names for components to avoid merge conflict
        old_names = list(components)
        rename_dict = {n: f"{n}_comp" for n in old_names}

        ## add component information to scores
        forced = xr.merge([forced, components.rename(rename_dict)])
        anom = xr.merge([anom, components.rename(rename_dict)])

        ## drop attrs (to avoid issues saving)
        forced = forced.drop_attrs()
        anom = anom.drop_attrs()

        return forced, anom


    def compute_flux_forced_anom(save_dir):
        """Compute forced and anomaly components, and save to file"""

        ## get filepaths
        forced_fp = save_dir / "flux_forced_with_clearsky.nc"
        anom_fp = save_dir / "flux_anom_with_clearsky.nc"

        ## try loading pre-computed data
        try:
            forced = xr.open_dataset(forced_fp)
            anom = xr.open_dataset(anom_fp)

        except:

            ## load to memory
            forced, anom = load_flux()

            ## save to file
            forced.to_netcdf(forced_fp)
            anom.to_netcdf(anom_fp)

        return forced, anom

    return (compute_flux_forced_anom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute
    """)
    return


@app.cell
def _(compute_flux_forced_anom, os, pathlib):
    ## save filepath
    _SAVE_DIR = pathlib.Path(os.environ['DATA_FP'], 'cesm', 'consolidated')
    ## non-budget data
    _forced, _anom = compute_flux_forced_anom(save_dir=_SAVE_DIR)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cloud fraction
    """)
    return


@app.cell
def _(np, os, pathlib, src, xr):
    def load_frac():
        """Load components of surface heat flux"""

        ## path to EOF data
        eofs_fp = pathlib.Path(os.environ["DATA_FP"], "cesm")

        ## variables to load
        names = ["cldlow", "cldhgh", "cldmed"]

        # ## load the EOFs
        load_var = lambda x: src.utils.load_eofs(pathlib.Path(eofs_fp, f"eofs_{x}.nc"))
        eofs = {x: load_var(x) for x in names}

        ## for convenience, put spatial patterns / components in single dataset
        components = xr.merge([eofs_.components().rename(y) for (y, eofs_) in eofs.items()])

        ## reset member dimension so they all match (NHF labeled differently...)
        member_coord = dict(member=np.arange(100))
        get_scores = lambda x, n: x.scores().assign_coords(member_coord).rename(n)
        scores = xr.merge([get_scores(eofs_, n) for (n, eofs_) in eofs.items()])

        ## fix off-by-one in time label
        scores = src.utils.align_pop_times(scores, pop_vars=list(scores))

        ## separate scores into forced and anomalies
        forced, anom = src.utils.separate_forced(scores)

        ## new names for components to avoid merge conflict
        old_names = list(components)
        rename_dict = {n: f"{n}_comp" for n in old_names}

        ## add component information to scores
        forced = xr.merge([forced, components.rename(rename_dict)])
        anom = xr.merge([anom, components.rename(rename_dict)])

        ## drop attrs (to avoid issues saving)
        forced = forced.drop_attrs()
        anom = anom.drop_attrs()

        return forced, anom


    def compute_frac_forced_anom(save_dir):
        """Compute forced and anomaly components, and save to file"""

        ## get filepaths
        forced_fp = save_dir / "cldfrac_forced.nc"
        anom_fp = save_dir / "cldfrac_anom.nc"

        ## try loading pre-computed data
        try:
            forced = xr.open_dataset(forced_fp)
            anom = xr.open_dataset(anom_fp)

        except:

            ## load to memory
            forced, anom = load_frac()

            ## save to file
            forced.to_netcdf(forced_fp)
            anom.to_netcdf(anom_fp)

        return forced, anom

    return (compute_frac_forced_anom,)


@app.cell
def _(compute_frac_forced_anom, os, pathlib):
    ## save filepath
    _SAVE_DIR = pathlib.Path(os.environ['DATA_FP'], 'cesm', 'consolidated')
    ## non-budget data
    _forced, _anom = compute_frac_forced_anom(save_dir=_SAVE_DIR)
    return


if __name__ == "__main__":
    app.run()
