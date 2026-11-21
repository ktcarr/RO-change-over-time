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
    ## Imports
    """)
    return


@app.cell
def _():
    import xarray as xr
    import pathlib
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import xeofs as xe
    import xesmf
    import time
    import src.utils
    import copy

    ## specify filepath for data
    DATA_FP = pathlib.Path(os.environ["DATA_FP"])

    ## set plotting specs
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False})

    ## bump up DPI for presentation
    mpl.rcParams["figure.dpi"] = 100
    return DATA_FP, np, pathlib, pd, plt, src, time, tqdm, xe, xesmf, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Functions
    """)
    return


@app.cell
def _(np, pathlib, pd, src, tqdm, xr):
    def get_files(varname):
        """get files for given variable name"""

        ## path to cesm2 data in MMLEA archive
        cesm2_fp = pathlib.Path("/glade/campaign/collections/rda/data/d651039/cesm2_lens")

        ## check if ocean or atmosphere
        is_oc = varname in ["mlotst", "sos", "tos", "z20", "zos"]

        if is_oc:
            data_fp = cesm2_fp / pathlib.Path("Omon", varname)

        else:
            data_fp = cesm2_fp / pathlib.Path("Amon", varname)

        return sorted(data_fp.glob("*.nc"))


    def check_member_file(i):
        """check zos and tos files match for given member idx"""
        return str(get_files("tos")[i])[-71:] == str(get_files("zos")[i])[-71:]


    def check_member_files():
        """check all files match"""

        checks = np.array([check_member_file(i) for i in range(100)])
        return np.all(checks)


    def load_member_varname(member_idx, varname):
        """load ensemble member. Args:
        - member_idx: integer in [0,99]
        """

        ## Get list of files
        files = get_files(varname)

        ## open data
        data = xr.open_dataset(files[member_idx])

        ## remove un-needed coords
        data = data[varname].squeeze(drop=True)

        ## rename lon/lat
        data = data.rename({"lat": "latitude", "lon": "longitude"})

        return data


    def load_member_Th(member_idx):
        """Load T and h data for given member index"""

        ## compute indices
        T_idxs = src.utils.get_RO_T_indices(load_member_varname(member_idx, "tos"))
        h_idxs = src.utils.get_RO_h_indices(load_member_varname(member_idx, "zos"))

        ## compute indices
        return xr.merge([T_idxs, h_idxs])


    def load_ensemble_Th(save_fp):
        """Load all ensemble members"""

        ## check if file exists
        if save_fp.is_file():

            data = xr.open_dataset(save_fp)

        else:

            ## new dimension: ensemble member
            member_dim = pd.Index(np.arange(100), name="member")

            ## do computation
            data = xr.concat(
                [load_member_Th(i) for i in tqdm.tqdm(member_dim)],
                dim=member_dim,
            )

            ## save to file
            data.to_netcdf(save_fp)

        return data


    def preprocess_Th(Th, save_dir):
        """pre-process Th data (compute ensemble mean and anomalies"""

        ## define filepaths for saving
        save_fp_emean = pathlib.Path(save_dir, "Th_emean.nc")
        save_fp_anom = pathlib.Path(save_dir, "Th_anom.nc")

        ## compute ensemble mean and anomalies
        Th_emean = Th.mean("member")
        Th_anom = Th - Th_emean

        ## save to file if not already
        if not save_fp_emean.is_file():
            Th_emean.to_netcdf(save_fp_emean)

        if not save_fp_anom.is_file():
            Th_anom.to_netcdf(save_fp_anom)

        return Th_emean, Th_anom

    return get_files, load_ensemble_Th, load_member_varname, preprocess_Th


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $T$, $h$
    """)
    return


@app.cell
def _(DATA_FP, load_ensemble_Th, preprocess_Th):
    ## specify save file paths
    _save_dir = DATA_FP / 'cesm'
    Th = load_ensemble_Th(_save_dir / 'Th.nc')
    ## load data
    ## compute ensemble stats/anomalies
    Th_emean, Th_anom = preprocess_Th(Th, save_dir=_save_dir)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## tropical SST
    """)
    return


@app.cell
def _(load_member_varname, np, pd, tqdm, xr):
    def load_member_trop_sst(member_idx):
        """Load T and h data for given member index"""

        ## load sst data
        sst = load_member_varname(member_idx, "tos")

        ## compute sst averaged over various lat bands
        avgs = []
        bands = np.arange(5, 35, 5)
        for b in bands:
            avg = sst.sel(latitude=slice(-b, b)).mean(["latitude", "longitude"])
            avgs.append(avg.rename(f"trop_sst_{b:02d}"))

        return xr.merge(avgs)


    def load_ensemble_trop_sst(save_fp):
        """Load all ensemble members"""

        ## check if file exists
        if save_fp.is_file():

            data = xr.open_dataset(save_fp)

        else:

            ## new dimension: ensemble member
            member_dim = pd.Index(np.arange(100), name="member")

            ## do computation
            data = xr.concat(
                [load_member_trop_sst(i) for i in tqdm.tqdm(member_dim)],
                dim=member_dim,
            )

            ## save to file
            data.to_netcdf(save_fp)

        return data

    return (load_ensemble_trop_sst,)


@app.cell
def _(DATA_FP, load_ensemble_trop_sst, pathlib):
    ## specify save file paths
    _save_fp = pathlib.Path(DATA_FP, 'cesm', 'trop_sst.nc')
    ## load data
    trop_sst = load_ensemble_trop_sst(_save_fp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ELI index
    """)
    return


@app.cell
def _(load_member_varname, np, pd, tqdm, xr):
    def get_eli(sst_trop):
        """compute ELI from tropical SST data"""

        ## get relative SST
        rsst = sst_trop - sst_trop.mean(["latitude", "longitude"])

        ## get SST in tropical Pac
        rsst_pac = rsst.sel(longitude=slice(140, 285))

        ## get boolean array where SST exceeds thresh
        exceeds_thresh = rsst_pac >= 0

        ## sum and count longitudes exceeding thresh
        longitude_sum = (exceeds_thresh * rsst_pac["longitude"]).sum(
            ["longitude", "latitude"]
        )
        longitude_count = exceeds_thresh.sum(["longitude", "latitude"])

        ## eli is average longitude
        eli = longitude_sum / longitude_count

        return eli


    def load_member_eli(member_idx):
        """Load eli data for given member index"""

        ## load sst data
        sst = load_member_varname(member_idx, "tos")

        ## compute sst averaged over various lat bands
        eli = []
        bands = np.arange(5, 35, 5)
        for b in bands:

            ## get sst subset
            sst_band = sst.sel(latitude=slice(-b, b))

            ## get ELI
            eli_ = get_eli(sst_band)
            eli.append(eli_.rename(f"eli_{b:02d}"))

        return xr.merge(eli)


    def load_ensemble_eli(save_fp):
        """Load all ensemble members"""

        ## check if file exists
        if save_fp.is_file():

            data = xr.open_dataset(save_fp)

        else:

            ## new dimension: ensemble member
            member_dim = pd.Index(np.arange(100), name="member")

            ## do computation
            data = xr.concat(
                [load_member_eli(i) for i in tqdm.tqdm(member_dim)],
                dim=member_dim,
            )

            ## save to file
            data.to_netcdf(save_fp)

        return data

    return (load_ensemble_eli,)


@app.cell
def _(DATA_FP, load_ensemble_eli, pathlib):
    ## specify save file paths
    _save_fp = pathlib.Path(DATA_FP, 'cesm', 'eli.nc')
    ## load data
    eli = load_ensemble_eli(_save_fp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EOFs
    """)
    return


@app.cell
def _(DATA_FP, load_member_varname, np, pathlib, pd, src, tqdm, xe, xr):
    def trim_to_eq_pac(data):
        """trim data to eq. Pac"""

        ## specfy
        lonlat_idx = dict(longitude=slice(100, 300), latitude=slice(-30, 30))

        return data.sel(lonlat_idx)


    def load_ensemble(varname, trim_fn=None):
        """load spatial data for given variable"""

        ## specify loading function
        if trim_fn is None:
            load = lambda i: load_member_varname(i, varname)

        else:
            load = lambda i: trim_fn(load_member_varname(i, varname))

        ## new dimension: ensemble member
        member_dim = pd.Index(np.arange(100), name="member")

        ## load data
        data = xr.concat([load(i) for i in tqdm.tqdm(member_dim)], dim=member_dim)

        return data


    def compute_eofs(varname):
        """compute/load eofs for given variable"""

        ## get filename
        filename = DATA_FP / pathlib.Path(f"cesm/eofs_{varname}.nc")

        ## try to load pre-computed EOFs
        if filename.is_file():
            eofs = src.utils.load_eofs(filename)

        ## if not pre-computed, do the computation here...
        else:
            data = load_ensemble(varname, trim_fn=trim_to_eq_pac)

            ## specs for EOFs
            eofs_kwargs = dict(
                n_modes=300, standardize=False, use_coslat=True, center=False
            )

            ## initialize EOF model
            eofs = xe.single.EOF(**eofs_kwargs)

            ## compute
            eofs.fit(data, dim=["time", "member"])

            ## save to file
            eofs.save(filename, engine="netcdf4")

        return eofs

    return (compute_eofs,)


@app.cell
def _(compute_eofs):
    print(f"\nloading TOS EOFs...")
    eofs_tos = compute_eofs("tos")

    print(f"\nloading ZOS EOFs...")
    eofs_tos = compute_eofs("zos")

    print(f"\nloading tauu EOFs...")
    eofs_tauu = compute_eofs("tauu")

    print(f"\nloading pr EOFs...")
    eofs_tauu = compute_eofs("pr")

    print(f"\nloading mixed layer EOFs...")
    eofs_mlotst = compute_eofs("mlotst")

    print(f"\nloading tauv EOFs...")
    eofs_tauv = compute_eofs("tauv")

    print(f"\nloading z20 EOFs...")
    eofs_z20 = compute_eofs("z20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## surface heat flux data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### First, compute for each ensemble member
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Link to heat flux calculation](https://bb.cgd.ucar.edu/cesm/threads/sign-definition-about-shflx-lhflx.8301/)
    "FSNS (net shortwave) is defined as positive into the surface
    FLNS (net longwave), LHFLX (latent heat), and SHFLX (sensible heat) are defined as positive into the atmosphere":
    ```Fnet_sfc = FSNS - FLNS - LHFLX - SHFLX```
    """)
    return


@app.cell
def _(DATA_FP, pathlib, src, tqdm, xr):
    def get_ensemble_ids():
        """get files for given variable name"""

        ## path to cesm2 lens data
        cesm2_fp = pathlib.Path(
            "/glade/campaign/collections/rda/data/d651056/CESM2-LE/atm/proc/tseries/month_1"
        )

        ## path to FSNS (arbitrary, just want the ids)
        data_fp = cesm2_fp / "FSNS"

        ## get list of ensemble ids
        ensemble_ids = []
        for f in data_fp.glob("*.nc"):
            ensemble_ids.append(str(f)[-54:-29])

        ## get unique values and sort
        ensemble_ids = sorted(list(set(ensemble_ids)))

        return ensemble_ids


    def trim_to_eq_pac_CAM(data):
        """trim data to eq. Pac"""

        ## specfy
        lonlat_idx = dict(lon=slice(100, 300), lat=slice(-30, 30))

        return data.sel(lonlat_idx)


    def load_var(varname, ensemble_id):
        """Load variable for given ensemble ID"""

        ## get path to data
        cesm2_fp = pathlib.Path(
            "/glade/campaign/collections/rda/data/d651056/CESM2-LE/atm/proc/tseries/month_1"
        )

        ## path to FSNS (arbitrary, just want the ids)
        data_fp = cesm2_fp / varname

        ## open data for ensemble member
        data = xr.open_mfdataset(data_fp.glob(f"*{ensemble_id}*.nc"))

        ## trim to eq Pac
        data = trim_to_eq_pac_CAM(data).compute()

        ## rename coords (to match reference grid)
        data = data[varname].rename({"lat": "latitude", "lon": "longitude"})

        ## open reference grid
        ref_fp = pathlib.Path(DATA_FP, "cesm", "eofs_tos.nc")
        ref = src.utils.load_eofs(ref_fp).components().isel(mode=0)

        ## interpolate to reference grid
        data = data.interp_like(ref)

        return data


    def compute_nhf(ensemble_id):
        """Load net heat flux for given ensemble id"""

        ## Load individual components
        FSNS = load_var("FSNS", ensemble_id)
        FLNS = load_var("FLNS", ensemble_id)
        LHFLX = load_var("LHFLX", ensemble_id)
        SHFLX = load_var("SHFLX", ensemble_id)

        return FSNS - FLNS - LHFLX - SHFLX


    def compute_nhf_ensemble(temp_dir):
        """compute net heat flux for full ensemble. Save to temp directory"""

        ## get ensemble ids
        ensemble_ids = get_ensemble_ids()

        ## loop through members
        for i in tqdm.tqdm(ensemble_ids):

            ## save filepath
            save_fp = pathlib.Path(temp_dir, f"nhf_{i}.nc")

            if save_fp.is_file():
                pass

            else:
                nhf = compute_nhf(i)
                nhf.to_netcdf(save_fp)

        return


    def compute_flux_ensemble(varname, temp_dir):
        """compute net heat flux for full ensemble. Save to temp directory"""

        ## get ensemble ids
        ensemble_ids = get_ensemble_ids()

        ## loop through members
        for i in tqdm.tqdm(ensemble_ids):

            ## save filepath
            save_fp = pathlib.Path(temp_dir, f"{varname}_{i}.nc")

            if save_fp.is_file():
                pass

            else:
                flux = load_var(varname=varname, ensemble_id=i)
                flux.to_netcdf(save_fp)

        return

    return compute_flux_ensemble, get_ensemble_ids


@app.cell
def _():
    from dask.distributed import LocalCluster, Client

    cluster = LocalCluster(n_workers=12)
    client = Client(cluster)
    client
    return


@app.cell
def _(DATA_FP, compute_flux_ensemble, pathlib):
    ## compute nhf for each file
    # compute_nhf_ensemble(pathlib.Path(DATA_FP, "cesm", "nhf_temp"))
    CESM_FP = pathlib.Path(DATA_FP, 'cesm')
    # ## compute component fluxes
    # CESM_FP = pathlib.Path(DATA_FP, "cesm")
    # for varname in ["FSNS", "FLNS", "LHFLX", "SHFLX"]:
    #     compute_flux_ensemble(varname=varname, temp_dir=CESM_FP / f"{varname.lower()}_temp")
    for _varname in ['CLDLOW', 'CLDHGH', 'CLDMED']:
    # ## clear-sky shortwave radiation
    # compute_flux_ensemble(varname="FSNSC", temp_dir=CESM_FP / f"fsnsc_temp")
    ## compute cloud fractions
        compute_flux_ensemble(varname=_varname, temp_dir=CESM_FP / f'{_varname.lower()}_temp')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Then, compute EOFs
    """)
    return


@app.cell
def _(get_ensemble_ids, get_files):
    checks = []
    for i, (f0, f1) in enumerate(zip(get_files("tos"), get_ensemble_ids())):

        ## get names for each
        n0 = f"{str(f0)[-59:-55]} {str(f0)[-36:-28]}"
        n1 = f"{f1[:4]} {f1[-8:]}"

        ## check they match
        checks.append(n0 == n1)

    print(all(checks))
    return


@app.cell
def _(DATA_FP, get_ensemble_ids, pathlib, pd, src, tqdm, xe, xr):
    def load_nhf_ensemble():
        """load spatial data for NHF"""

        ## new dimension: ensemble member
        member_dim = pd.Index(get_ensemble_ids(), name="member")

        ## load data
        load_fp = pathlib.Path(DATA_FP, "cesm", "nhf_temp")
        data = [xr.open_dataarray(load_fp / f"nhf_{i}.nc") for i in tqdm.tqdm(member_dim)]
        data = xr.concat(data, dim=member_dim)

        return data


    def compute_nhf_eofs():
        """compute/load eofs for given variable"""

        ## get filename
        filename = DATA_FP / pathlib.Path(f"cesm/eofs_nhf.nc")

        ## try to load pre-computed EOFs
        if filename.is_file():
            eofs = src.utils.load_eofs(filename)

        ## if not pre-computed, do the computation here...
        else:
            data = load_nhf_ensemble()

            ## specs for EOFs
            eofs_kwargs = dict(
                n_modes=300, standardize=False, use_coslat=True, center=False
            )

            ## initialize EOF model
            eofs = xe.single.EOF(**eofs_kwargs)

            ## compute
            eofs.fit(data, dim=["time", "member"])

            ## save to file
            eofs.save(filename, engine="netcdf4")

        return eofs


    def load_flux_ensemble(varname):
        """load spatial data for flux variable"""

        ## new dimension: ensemble member
        member_dim = pd.Index(get_ensemble_ids(), name="member")

        ## load data
        load_fp = pathlib.Path(DATA_FP, "cesm", f"{varname.lower()}_temp")
        data = [
            xr.open_dataarray(load_fp / f"{varname}_{i}.nc") for i in tqdm.tqdm(member_dim)
        ]
        data = xr.concat(data, dim=member_dim)

        return data


    def compute_flux_eofs(varname):
        """compute/load eofs for given variable"""

        ## get filename
        filename = DATA_FP / pathlib.Path(f"cesm/eofs_{varname.lower()}.nc")

        ## try to load pre-computed EOFs
        if filename.is_file():
            eofs = src.utils.load_eofs(filename)

        ## if not pre-computed, do the computation here...
        else:
            data = load_flux_ensemble(varname)

            ## specs for EOFs
            eofs_kwargs = dict(
                n_modes=300, standardize=False, use_coslat=True, center=False
            )

            ## initialize EOF model
            eofs = xe.single.EOF(**eofs_kwargs)

            ## compute
            eofs.fit(data, dim=["time", "member"])

            ## save to file
            eofs.save(filename, engine="netcdf4")

        return eofs

    return compute_flux_eofs, compute_nhf_eofs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Net heat flux
    """)
    return


@app.cell
def _(compute_nhf_eofs, time):
    print(f'\nloading NHF EOFs...')
    _t0 = time.time()
    eofs_nhf = compute_nhf_eofs()
    _t1 = time.time()
    print(f'Elapsed time: {_t1 - _t0:.1f} seconds')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Components
    """)
    return


@app.cell
def _(compute_flux_eofs):
    for _varname in ['FSNS', 'FLNS', 'LHFLX', 'SHFLX']:
        compute_flux_eofs(varname=_varname)
    compute_flux_eofs(varname='FSNSC')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cloud fraction
    """)
    return


@app.cell
def _(compute_flux_eofs):
    for _varname in ['CLDLOW', 'CLDHGH', 'CLDMED']:
        compute_flux_eofs(varname=_varname)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ocean data (subsurface)
    """)
    return


@app.cell
def _(DATA_FP, pathlib, pd, src, xe, xr):
    def load_subsurf_ensemble(varname):
        """load subsurface data"""

        ## Get list of files and member idx
        # load_fp = pathlib.Path(DATA_FP, "cesm", f"{varname}_temp_v2")
        load_fp = pathlib.Path(DATA_FP, "cesm", f"{varname}_temp")
        # load_fp = pathlib.Path(DATA_FP, "cesm", f"{varname}_05")
        files = list(sorted(load_fp.glob("*.nc")))
        member_id = pd.Index([str(f)[-28:-3] for f in files], name="member_id")

        # ## load data (loop method)
        # data = [xr.open_dataarray(f) for f in files]
        # data = xr.concat(data, dim=member_id)

        ## load data (dask method?)
        data = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="member_id",
            chunks={"time": 720},
        )
        data = data.assign_coords({"member_id": member_id})

        return data


    def compute_subsurf_eofs(varname):
        """compute/load eofs for given variable"""

        ## get filename
        # filename = DATA_FP / pathlib.Path(f"cesm/eofs_{varname}.nc")
        filename = DATA_FP / pathlib.Path(f"cesm/eofs_{varname}_1200.nc")

        ## try to load pre-computed EOFs
        if filename.is_file():
            eofs = src.utils.load_eofs(filename)

        ## if not pre-computed, do the computation here...
        else:
            data = load_subsurf_ensemble(varname)

            ## specs for EOFs
            eofs_kwargs = dict(n_modes=1200, standardize=False, center=False)

            ## initialize EOF model
            eofs = xe.single.EOF(**eofs_kwargs)

            ## compute
            eofs.fit(data, dim=["time", "member_id"])

            ## save to file
            try:
                eofs.save(filename, engine="netcdf4")

            except:

                ## extract information (avoids saving issue)
                components = eofs.components().to_dataarray()
                components = components.squeeze().rename("components")
                scores = eofs.scores()
                del scores.attrs["solver_kwargs"]

                ## netcdf with data
                eof_data = xr.merge([scores, components])

                ## save to file
                eof_data.to_netcdf(filename)

        return eofs

    return (compute_subsurf_eofs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute
    """)
    return


@app.cell
def _(compute_subsurf_eofs, time):
    # print(f"\nloading subsurf. temp EOFs...")
    # t0 = time.time()
    # eofs_temp = compute_subsurf_eofs("temp")
    # t1 = time.time()
    # print(f"Elapsed time: {t1-t0:.1f} seconds")
    print(f'\nloading 3d TEMP EOFs...')
    # print(f"\nloading subsurf. wvel EOFs...")
    _t0 = time.time()
    # eofs_wvel = compute_subsurf_eofs("wvel")
    eofs_temp_3d = compute_subsurf_eofs('temp_3d')
    _t1 = time.time()
    print(f'Elapsed time: {_t1 - _t0:.1f} seconds')
    # print(f"\nloading subsurf. WTT EOFs...")
    print(f'\nloading 3d WVEL EOFs...')
    # eofs_wvel = compute_subsurf_eofs("wtt")
    _t0 = time.time()
    eofs_wvel_3d = compute_subsurf_eofs('wvel_3d')
    _t1 = time.time()
    # print(f"\nloading subsurf. UET EOFs...")
    print(f'Elapsed time: {_t1 - _t0:.1f} seconds')
    # eofs_wvel = compute_subsurf_eofs("uet")
    print(f'\nloading 3d VVEL EOFs...')
    _t0 = time.time()
    eofs_vvel_3d = compute_subsurf_eofs('vvel_3d')
    # print(f"\nloading subsurf. UVEL EOFs...")
    _t1 = time.time()
    # eofs_wvel = compute_subsurf_eofs("uvel")
    print(f'Elapsed time: {_t1 - _t0:.1f} seconds')
    print(f'\nloading 3d UVEL EOFs...')
    _t0 = time.time()
    eofs_uvel_3d = compute_subsurf_eofs('uvel_3d')
    _t1 = time.time()
    # print(f"\nloading 3d VNT EOFs...")
    # eofs_vnt = compute_subsurf_eofs("vnt")
    print(f'Elapsed time: {_t1 - _t0:.1f} seconds')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demo for NaN filling / lazy evaluation
    See: https://xeofs.readthedocs.io/en/develop/content/user_guide/core_functionalities/dask_support.html
    """)
    return


@app.cell
def _():
    # ## Load test data
    # d = load_subsurf_ensemble("vvel_3d")
    # e = d.isel(member_id=slice(0,2))
    # e.load();

    # ## find NaN values
    # nan_idx = np.isnan(e.isel(member_id=0)).all("time").drop_vars("member_id")

    # ## fill with zero(?)
    # e.where(~nan_idx, other=0.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Look at data
    """)
    return


@app.cell
def _(DATA_FP, pathlib, xr):
    temp_load_fp = pathlib.Path(DATA_FP, "cesm", "temp_temp_v2")
    temp = xr.open_dataarray(sorted(list(temp_load_fp.glob("*.nc")))[0])

    wvel_load_fp = pathlib.Path(DATA_FP, "cesm", "wvel_temp_v2")
    wvel = xr.open_dataarray(sorted(list(wvel_load_fp.glob("*.nc")))[0])

    data = xr.merge([temp.rename("T"), wvel.rename("w")])

    data.load()
    d = data.mean("time")
    return (d,)


@app.cell
def _(d, np, plt, src):
    import cmocean

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.contourf(
        d["T"].lon,
        d["T"].z_t / 100,
        d["T"],
        cmap="cmo.thermal",
        levels=np.arange(12, 32, 2),
        extend="both",
    )

    cp = ax.contour(
        d["w"].lon,
        d["w"].z_w_top / 100,
        d["w"],
        colors="k",
        levels=src.utils.make_cb_range(0.0016, 0.00032),
        extend="both",
        linewidths=1,
    )
    # cb = fig.colorbar(cp)


    ax.set_ylim(ax.get_ylim()[::-1])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ocean data (surface)
    """)
    return


@app.cell
def _(DATA_FP, np, pathlib, pd, src, xe, xesmf, xr):
    def load_grid(lon_range, lat_range):
        """Create mask from OISST data on cloud"""

        ## load sst data
        sst = xr.open_dataset(
            r"http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2/new/sst.oisst.mon.ltm.1991-2020.nc",
            decode_times=False,
        )
        sst = sst["sst"].isel(time=0).drop_vars("time")

        ## convert to lsm (fill ones over ocean)
        lsm = sst.where(np.isnan(sst), other=1.0)

        ## sel lon/lat range
        lsm = lsm.sel(lon=slice(*lon_range), lat=slice(*lat_range))

        # ## add binary mask for regridding
        lsm["mask"] = ~np.isnan(lsm)

        return lsm


    def load_ocn_surf_ensemble(varname):
        """load subsurface data"""

        ## Get list of files and member idx
        load_fp = pathlib.Path(DATA_FP, "cesm", f"{varname}_temp")
        files = list(sorted(load_fp.glob("*.nc")))
        member_id = pd.Index([str(f)[-28:-3] for f in files], name="member_id")

        ## load data
        data = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="member_id",
        )
        data = data.assign_coords({"member_id": member_id})

        ## load into memory
        data.load()

        ## rename coords for regridding
        data = data.rename({"ULONG": "lon", "ULAT": "lat"})

        ## regrid
        grid = load_grid(lon_range=[120, 300], lat_range=[-15, 15])
        regridder = xesmf.Regridder(data, grid, "bilinear")
        data_regrid = regridder(data)

        return data_regrid.drop_vars("mask")


    def compute_ocn_surf_eofs(varname):
        """compute/load eofs for given variable"""

        ## get filename
        filename = DATA_FP / pathlib.Path(f"cesm/eofs_{varname}.nc")

        ## try to load pre-computed EOFs
        if filename.is_file():
            eofs = src.utils.load_eofs(filename)

        ## if not pre-computed, do the computation here...
        else:
            data = load_ocn_surf_ensemble(varname).compute()

            ## specs for EOFs
            eofs_kwargs = dict(
                n_modes=300, standardize=False, use_coslat=True, center=False
            )

            ## initialize EOF model
            eofs = xe.single.EOF(**eofs_kwargs)

            ## compute
            eofs.fit(data, dim=["time", "member_id"])

            ## extract information (avoids saving issue
            components = eofs.components().to_dataarray()
            components = components.squeeze().rename("components")
            scores = eofs.scores()
            del scores.attrs["solver_kwargs"]

            eof_data = xr.merge([scores, components])

            ## save to file
            eof_data.to_netcdf(filename)

        return eof_data

    return (compute_ocn_surf_eofs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute
    """)
    return


@app.cell
def _(compute_ocn_surf_eofs, time):
    print(f'\nloading uvel EOFs...')
    _t0 = time.time()
    eofs_uvel = compute_ocn_surf_eofs('uvel')
    _t1 = time.time()
    print(f'Elapsed time: {_t1 - _t0:.1f} seconds')
    print(f'\nloading vvel EOFs...')
    _t0 = time.time()
    eofs_vvel = compute_ocn_surf_eofs('vvel')
    _t1 = time.time()
    print(f'Elapsed time: {_t1 - _t0:.1f} seconds')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CVDP indices
    """)
    return


@app.cell
def _(DATA_FP, get_files, np, pathlib, pd, tqdm, xr):
    def get_cvdp_file(member_id):
        """Get filename corresponding to given ensemble id"""

        ## Get filename for corresponding spatial data
        orig_filename = str(get_files("tos")[member_id])

        ## get year initialization and member idx
        year_init = orig_filename[-36:-32]
        idx = orig_filename[-31:-28]

        ## get updated filename
        filename = f"CESM2-LENS_{year_init}.{idx}.cvdp_data.1850-2100.nc"

        ## get path to data
        cvdp_fp = DATA_FP / pathlib.Path("cesm/cvdp_output")

        return cvdp_fp / filename


    def load_member_cvdp_idxs(member_id):
        """Load CVDP indices for given member"""

        ## get filename
        filename = get_cvdp_file(member_id)

        ## open data
        data = xr.open_dataset(filename, decode_times=False)

        ## extract given variable names
        names = [
            "indian_ocean_dipole",
            "nino34",
            "north_pacific_meridional_mode",
            "south_pacific_meridional_mode",
            "tropical_indian_ocean",
            "north_tropical_atlantic",
            "atlantic_nino",
        ]

        return data[names]


    def load_cvdp_idxs():
        """Load CVDP data for all members"""

        ## ensemble member index
        member_idx = pd.Index(np.arange(100), name="member")

        ## load indices and concatenate
        data = xr.concat(
            [load_member_cvdp_idxs(i) for i in tqdm.tqdm(member_idx)], dim=member_idx
        )

        return data

    return (load_cvdp_idxs,)


@app.cell
def _(DATA_FP, load_cvdp_idxs):
    ## specify save file paths
    _save_dir = DATA_FP / 'cesm'
    cvdp_total = load_cvdp_idxs()
    ## load data
    cvdp_anom = cvdp_total - cvdp_total.mean('member')
    ## compute anomalies
    ## save to file
    cvdp_anom.to_netcdf(_save_dir / 'cvdp_anom.nc')
    return


if __name__ == "__main__":
    app.run()
