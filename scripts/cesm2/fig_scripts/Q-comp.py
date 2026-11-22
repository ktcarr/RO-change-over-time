import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Surface heat flux feedback in observations and CESM2

    Paper figure `fig:Q-comp` (`paper-outline.tex` in the ch3 Overleaf project). Scatter
    plots of Sep-Oct-Nov (SON) mean surface heat flux against SST, both averaged over the
    Niño 3 region, for observations (top: ERSSTv5 SST and ERA5 fluxes, 1950-2022) and
    CESM2 (bottom: all 100 members, 1950-1980). Columns are the net flux and its
    shortwave, latent, and longwave components (positive into the ocean). SST is in
    standard deviations; both variables have their mean removed.

    Curves are cubic least-squares fits, drawn over the range of the data: solid for the
    observations (repeated on the CESM2 panels for comparison), dashed for CESM2.

    **Inputs:** `DATA_FP/era5_nhf_glade.nc` (ERA5 fluxes),
    `ersstv5.185401-202412.nc` in CVDP-OBS (`src.data.CVDP_OBS_FP`), and under
    `DATA_FP/cesm/consolidated`: `anom.nc` (SST, net flux) and
    `flux_anom_with_clearsky.nc` (flux components).

    **Output:** `SAVE_FP/figures/Q-comp.pdf`

    Before this script, the figure came from the "Make plot" cell of `nhf_validation.py`
    (saved as `Q-validation`). Changes:

    - CESM2's longwave flux is now positive into the ocean, like ERA5's and the other
      fluxes; before, it kept CESM's positive-upward sign (only the latent flux was
      flipped), so the CESM2 longwave panel was upside down.
    - The season is set by `SEASON_START` (SON, as in the figure in the paper draft). The
      notebook's committed code used Nov-Dec-Jan, and the draft's caption said
      Oct-Nov-Dec; SON reproduces the draft's figure.
    - The x-axis is labeled "Niño 3 SST ($\sigma$)" (it was "sst").
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Imports
    """
    )
    return


@app.cell
def _():
    ## plotting and plotting style
    import matplotlib.pyplot as plt
    import seaborn as sns

    ## numerics and labeled arrays
    import numpy as np
    import xarray as xr

    ## this project's modules: loading data, plotting
    import src.data
    import src.plot

    ## older general utilities (to be split into the modules above)
    import src.utils

    ## plotting style used for all paper figures
    sns.set(rc={"axes.facecolor": "white", "axes.grid": False}, palette="colorblind")
    return np, plt, src, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Settings
    """
    )
    return


@app.cell
def _(np):
    ## first month of the season: 9 = Sep-Oct-Nov (SON), as in the paper draft
    SEASON_START = 9

    ## CESM2 years (a period overlapping the observations)
    CESM_YEARS = slice("1950", "1980")

    ## fluxes (ERA5 name, CESM2 name) and panel titles
    FLUXES = dict(
        Net=("nhf", "nhf"),
        Shortwave=("sw", "fsns"),
        Latent=("lhf", "lhflx"),
        Longwave=("lw", "flns"),
    )

    ## SST values (standard deviations) for evaluating the fits
    SIGMA = np.arange(-3.5, 3.6, 0.1)
    return CESM_YEARS, FLUXES, SEASON_START, SIGMA


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load Niño 3 averages

    Monthly values, then seasonal means (full seasons only).
    """
    )
    return


@app.cell
def _(CESM_YEARS, FLUXES, SEASON_START, src, xr):
    ## observations: ERA5 fluxes and ERSST anomalies over Niño 3, dims (time,)
    _flux_era5 = src.data.load_era5_flux()
    obs = xr.Dataset()
    for _title, (_v_era5, _) in FLUXES.items():
        obs[_title] = src.utils.get_nino3(_flux_era5[_v_era5])

    ## ERSST over the ERA5 period
    obs["sst"] = src.utils.get_nino3(src.data.load_ersst_anom()).sel(time=obs.time)

    ## CESM2: anomalies over Niño 3 (positive into the ocean), dims (member, time)
    cesm = xr.Dataset()
    for _title, (_, _v_cesm) in FLUXES.items():
        cesm[_title] = src.data.load_F(_v_cesm, region="3")
    cesm["sst"] = src.data.load_region_avg("sst", region="3")

    ## CESM2 period
    cesm = cesm.sel(time=CESM_YEARS)

    ## seasonal means (full seasons only)
    obs_season = src.data.get_season(obs, SEASON_START).compute()
    cesm_season = src.data.get_season(cesm, SEASON_START).compute()
    return cesm_season, obs_season


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Standardize and fit

    SST in standard deviations; every variable minus its mean (over time, and members for
    CESM2). Then a cubic fit of each flux to SST.
    """
    )
    return


@app.cell
def _(FLUXES, SIGMA, cesm_season, np, obs_season, xr):
    ## standardized, mean-removed data; dims (s,) where s is the sample
    data = dict()
    for _name, _ds in [("obs", obs_season), ("cesm", cesm_season)]:

        ## pool members and times (CESM2) into one sample dim
        if "member" in _ds.dims:
            _ds = _ds.stack(s=["time", "member"])
        else:
            _ds = _ds.rename(time="s")

        ## SST in standard deviations
        _ds["sst"] = _ds["sst"] / _ds["sst"].std()

        ## remove each variable's mean
        data[_name] = _ds - _ds.mean("s")

    ## cubic fit of each flux to SST, evaluated over the range of the data
    fits = dict()
    for _name, _ds in data.items():
        _x = _ds["sst"].values

        ## SST values inside the range of the data
        _sigma = SIGMA[(SIGMA >= _x.min()) & (SIGMA <= _x.max())]

        ## fit and evaluate each flux
        fits[_name] = dict()
        for _title in FLUXES:
            _coefs = np.polyfit(_x, _ds[_title].values, deg=3)
            _curve = np.polyval(_coefs, _sigma)
            fits[_name][_title] = xr.DataArray(_curve, coords=dict(sigma=_sigma))
    return data, fits


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Figure
    """
    )
    return


@app.cell
def _(FLUXES, data, fits, plt, src):
    ## rows: observations, CESM2; columns: fluxes; shared axes
    fig, axs = plt.subplots(
        2, 4, figsize=(9, 4.5), layout="constrained", sharex=True, sharey=True
    )

    ## row settings: data, marker size, opacity, label
    _rows = [("obs", 20, 1, "ERA5"), ("cesm", 1, 0.5, "CESM2")]

    ## guidelines at zero
    _line_kwargs = dict(c="k", ls="--", lw=0.8, zorder=0.5)

    for _i, (_name, _size, _alpha, _label) in enumerate(_rows):
        for _j, _title in enumerate(FLUXES):
            _ax = axs[_i, _j]

            ## scatter of the data
            _x = data[_name]["sst"]
            _ax.scatter(_x, data[_name][_title], s=_size, alpha=_alpha)

            ## observed fit (solid) on both rows; CESM2 fit (dashed) on its row
            _fit = fits["obs"][_title]
            _ax.plot(_fit.sigma, _fit, c="k", ls="-")
            if _name == "cesm":
                _fit = fits["cesm"][_title]
                _ax.plot(_fit.sigma, _fit, c="k", ls="--")
            else:
                _ax.set_title(_title)

            ## guidelines
            _ax.axvline(0, **_line_kwargs)
            _ax.axhline(0, **_line_kwargs)

        ## row label to the right
        axs[_i, -1].text(s=_label, x=1.1, y=0.5, transform=axs[_i, -1].transAxes)

    ## axis labels and ticks
    for _ax in axs[:, 0]:
        _ax.set_ylabel(r"W m$^{-2}$")
    for _ax in axs[-1]:
        _ax.set_xlabel(r"Niño 3 SST ($\sigma$)")
    axs[0, 0].set_xticks([-2, 0, 2])

    ## save to SAVE_FP/figures
    src.plot.save_fig(fig, "Q-comp")

    fig
    return


if __name__ == "__main__":
    app.run()
