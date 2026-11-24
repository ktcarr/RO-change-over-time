"""
Diagnostics of the RO sensitivity tests, from the cache written by
simulate_case.py and roundtrip.py (nothing is refitted or resimulated, so this
runs in about a minute on the current node).

Each comparison group of cases.GROUPS (e.g. round1, round2) gets its own
outputs, with the group's name as a suffix (cases that haven't finished are
left out). Outputs, in SAVE_FP/ro_sensitivity/:
    - compare_<group>.png/pdf: changes over time in CESM2 and the RO, one row per case:
      std and skewness of DJF T, and the contributions of warm and cold
      anomalies to its variance (src.ro.get_contributions)
    - experiments_<group>.png/pdf: each experiment's effect on the warm and cold
      contributions (control minus experiment), one row per case
    - baseline_<group>.png/pdf: the first window (1850-1889): std and skewness of T by
      calendar month, std of h_hat by month, PSD of T, lagged correlation of T
      and h_hat, autocorrelation of T
    - params_<group>.png/pdf: seasonal cycle of every RO parameter, the noise
      and the net heat flux feedback alpha in the first window (1850-1889)
    - tables_<group>.txt: fit quality (RMSE against CESM2 over windows),
      parameters, the months with a positive T^3 coefficient, the first harmonic
      of the seasonal skewness, experiment effects in 2080, the experiments'
      T^3 coefficient, the T^3 bound's effect on R_o, and the round-trip test
      (annual means, and R_2 and R_3 by month)

Usage (from scripts/cesm2, with SAVE_FP set):
    python ro_sensitivity/diagnostics.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.stats
import xarray as xr

import src.data
import src.ro

## the case table (cases.py, next to this script; Python puts the script's
## folder on the import path)
import cases


## where the cache is, and where the outputs go
OUT_DIR = f"{os.environ['SAVE_FP']}/ro_sensitivity"

## experiments, in the order shown, with short labels
EXPERIMENTS = dict(
    noise="noise",
    R="R",
    F1="F1",
    F2_eps="F2, eps",
    Ro="R_o",
    alpha_nhf="alpha (net)",
    alpha_fsns="alpha (SW)",
    alpha_lhflx="alpha (LH)",
    alpha_flns="alpha (LW)",
)

## the stats over time, with titles and units
STATS = dict(
    sigma=r"$\sigma(T)$ DJF (K)",
    skew="skew(T) DJF",
    warm=r"warm contribution (K$^2$)",
    cold=r"cold contribution (K$^2$)",
)

## lags (months) for the lagged correlations
LAGS = np.arange(-24, 25)


def load_case(tag):
    """
    Read one case's cache.

    Args:
        - tag (str): case tag (a key of cases.CASES)

    Returns:
        - case (dict): "config" (dict), "data" (CESM2 T_3, h_hat), "T_scale"
          (float), "fits", "sims" (dict of xr.Dataset: "ctrl" and each
          experiment), "alpha" (xr.Dataset, one variable per flux),
          "experiments_R3" (xr.DataArray, or None if not saved) and
          "roundtrip" (xr.Dataset, or None if not run)
    """

    ## settings, CESM2 data and fits
    case_dir = f"{OUT_DIR}/{tag}"
    with open(f"{case_dir}/config.json") as f:
        config = json.load(f)

    ## the CESM2 data the RO was fitted to, and the fitted parameters
    data = xr.open_dataset(f"{case_dir}/data.nc")
    fits = xr.open_dataset(f"{case_dir}/fits.nc")

    ## control and experiment simulations
    sims = dict(ctrl=xr.open_dataset(f"{case_dir}/sims_ctrl.nc"))
    for name in EXPERIMENTS:
        sims[name] = xr.open_dataset(f"{case_dir}/sims_{name}.nc")

    ## heat flux feedbacks, and the experiments' T^3 coefficient (only saved
    ## in the later cases)
    alpha = xr.open_dataset(f"{case_dir}/alpha.nc")
    R3_path = f"{case_dir}/experiments_R3.nc"
    if os.path.exists(R3_path):
        experiments_R3 = xr.open_dataarray(R3_path)
    else:
        experiments_R3 = None

    ## round-trip test, if it has run
    roundtrip_path = f"{case_dir}/roundtrip.nc"
    if os.path.exists(roundtrip_path):
        roundtrip = xr.open_dataset(roundtrip_path)
    else:
        roundtrip = None

    case = dict(
        config=config,
        data=data,
        T_scale=data.attrs["T_scale"],
        fits=fits,
        sims=sims,
        alpha=alpha,
        experiments_R3=experiments_R3,
        roundtrip=roundtrip,
    )
    return case


def get_stats(x, T_scale):
    """
    Statistics of DJF T in each window: std, skewness and the warm and cold
    contributions to the variance, each the median over members (or copies).

    Args:
        - x (xr.Dataset): CESM2 data or RO simulations with T_3, dims (member,
          year, time) in any order
        - T_scale (float): standard deviation of T_3 (K)

    Returns:
        - stats (xr.Dataset): "sigma" (K), "skew", "warm" and "cold" (K^2),
          dims (year,)
    """

    ## DJF means in K
    T_djf = T_scale * src.data.get_djf(x["T_3"])

    ## std and skewness over time, for each member
    stats = xr.Dataset()
    stats["sigma"] = T_djf.std("time")
    stats["skew"] = xr.apply_ufunc(
        scipy.stats.skew, T_djf, input_core_dims=[["time"]], kwargs={"axis": -1}
    )

    ## contributions of warm and cold anomalies (about each window's median)
    contrib = src.ro.get_contributions(x, T_scale)
    stats["warm"] = contrib["warm"]
    stats["cold"] = contrib["cold"]

    ## median over members
    stats = stats.median("member")

    return stats


def rmse(a, b):
    """
    Root-mean-square difference over windows.

    Args:
        - a, b (xr.DataArray): values with dims (year,)

    Returns:
        - rmse (float): root-mean-square of a - b
    """

    ## mean over windows of the squared difference
    return float(np.sqrt(((a - b) ** 2).mean()))


def monthly_stats(T, months):
    """
    Std and skewness for each calendar month, pooling members and times.

    Args:
        - T (np.ndarray): values, shape (member, time)
        - months (np.ndarray): calendar month of each time, shape (time,)

    Returns:
        - sd, sk (np.ndarray): std and skewness for months 1-12, shape (12,)
    """

    ## each calendar month in turn
    sd = []
    sk = []
    for month in range(1, 13):
        x = T[:, months == month].ravel()
        sd.append(x.std())
        sk.append(scipy.stats.skew(x))

    return np.array(sd), np.array(sk)


def lag_corr(a, b, lags):
    """
    Correlation of a(t) with b(t + lag), pooling members.

    Args:
        - a, b (np.ndarray): values, shape (member, time)
        - lags (np.ndarray): lags (time steps)

    Returns:
        - corr (np.ndarray): one correlation per lag
    """

    ## line up a(t) and b(t + lag) for each lag
    corr = []
    for lag in lags:
        if lag >= 0:
            x, y = a[:, : a.shape[1] - lag], b[:, lag:]
        else:
            x, y = a[:, -lag:], b[:, : b.shape[1] + lag]
        corr.append(np.corrcoef(x.ravel(), y.ravel())[0, 1])

    return np.array(corr)


def baseline_diagnostics(x, T_scale):
    """
    Diagnostics of the first window: std and skewness of T by calendar month,
    std of h_hat by month, PSD of T, corr(T(t), h_hat(t + lag)), and the
    autocorrelation of T.

    Args:
        - x (xr.Dataset): CESM2 data or RO simulations (T_3, h_hat) for one
          window, dims (member, time)
        - T_scale (float): standard deviation of T_3 (K)

    Returns:
        - diag (dict of np.ndarray): "sdT", "skT", "sdh", "f" (frequency,
          yr^-1), "pT", "cTh", "aT"
    """

    ## arrays of shape (member, time); T in K
    T = T_scale * x["T_3"].transpose("member", "time").values
    h = x["h_hat"].transpose("member", "time").values
    months = x.time.dt.month.values

    ## seasonal cycle of std and skewness
    sdT, skT = monthly_stats(T, months)
    sdh, _ = monthly_stats(h, months)

    ## Welch PSD of T (20-year segments), averaged over members; drop f = 0
    f, pT = scipy.signal.welch(T, fs=12, nperseg=240, axis=-1)

    diag = dict(
        sdT=sdT,
        skT=skT,
        sdh=sdh,
        f=f[1:],
        pT=pT.mean(0)[1:],
        cTh=lag_corr(T, h, LAGS),
        aT=lag_corr(T, T, LAGS),
    )
    return diag


def plot_over_time(all_stats, stats_cesm, tags):
    """
    compare.png: CESM2 and RO stats over time, one row per case.

    Args:
        - all_stats (dict of xr.Dataset): RO stats (from get_stats) by tag
        - stats_cesm (xr.Dataset): CESM2 stats
        - tags (list of str): cases, in row order

    Returns:
        - fig (matplotlib Figure)
    """

    ## rows: cases; columns: stats; the same y-axis down each column
    fig, axs = plt.subplots(
        len(tags),
        len(STATS),
        figsize=(13, 2 * len(tags)),
        layout="constrained",
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    for row, tag in zip(axs, tags):
        for ax, (name, title) in zip(row, STATS.items()):
            ## CESM2 in black, the RO in color
            ax.plot(stats_cesm.year, stats_cesm[name], c="k", lw=2, label="CESM2")
            ax.plot(
                all_stats[tag].year, all_stats[tag][name], c="tab:orange", label="RO"
            )

            ## zero line for skewness
            if name == "skew":
                ax.axhline(0, c="gray", lw=0.5)
            else:
                pass

        ## case label on the left
        row[0].set_ylabel(tag)

    ## stat names on the top row, and the legend
    for ax, title in zip(axs[0], STATS.values()):
        ax.set_title(title, loc="left")
    axs[0, 0].legend(fontsize=8)

    return fig


def plot_experiments(effects, tags):
    """
    experiments.png: each experiment's effect on the warm and cold
    contributions, one row per case.

    Args:
        - effects (dict of dict of xr.Dataset): effect (control minus
          experiment) by tag, then by experiment, with "warm" and "cold"
        - tags (list of str): cases, in row order

    Returns:
        - fig (matplotlib Figure)
    """

    ## rows: cases; columns: experiments; one y-axis for all
    fig, axs = plt.subplots(
        len(tags),
        len(EXPERIMENTS),
        figsize=(18, 1.8 * len(tags)),
        layout="constrained",
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, tag in zip(axs, tags):
        ## one panel per experiment
        for ax, (name, label) in zip(row, EXPERIMENTS.items()):
            ## El Niño (warm) in red, La Niña (cold) in blue
            effect = effects[tag][name]
            ax.plot(effect.year, effect["warm"], c="r", label="El Niño")
            ax.plot(effect.year, effect["cold"], c="b", label="La Niña")

            ## zero line: no effect
            ax.axhline(0, c="gray", lw=0.5)

        ## case label on the left
        row[0].set_ylabel(tag)

    ## experiment names on the top row
    for ax, label in zip(axs[0], EXPERIMENTS.values()):
        ax.set_title(f"effect of {label}", loc="left", fontsize=9)
    axs[0, 0].legend(fontsize=8)

    return fig


def plot_baseline(diags, diags_cesm, tags):
    """
    baseline.png: the first window's diagnostics, all cases in each panel.

    Args:
        - diags (dict of dict): RO diagnostics (from baseline_diagnostics) by tag
        - diags_cesm (dict of dict): CESM2 diagnostics, by h_hat method
          ("hmonth", "hconst")
        - tags (list of str): cases

    Returns:
        - fig (matplotlib Figure)
    """

    ## line for each dataset: CESM2 in black (dashed: one h_hat regression),
    ## cases in the default color cycle
    lines = []
    for method, diag in diags_cesm.items():
        ls = "-" if method == "hmonth" else "--"
        lines.append((f"CESM2 ({method})", diag, dict(c="k", lw=2.5, ls=ls)))
    for i, tag in enumerate(tags):
        ls = "--" if tag.endswith("hconst") else "-"
        lines.append((tag, diags[tag], dict(c=f"C{i}", ls=ls)))

    ## panels
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained")
    months = np.arange(1, 13)
    for label, diag, kw in lines:
        ## seasonal cycles (top row)
        axs[0, 0].plot(months, diag["sdT"], label=label, **kw)
        axs[0, 1].plot(months, diag["sdh"], **kw)
        axs[0, 2].plot(months, diag["skT"], **kw)

        ## spectrum and lagged correlations (bottom row)
        axs[1, 0].loglog(1 / diag["f"], diag["pT"], **kw)
        axs[1, 1].plot(LAGS, diag["cTh"], **kw)
        axs[1, 2].plot(LAGS, diag["aT"], **kw)

    ## titles and axes
    axs[0, 0].set_title("a) std(T) by month (K)", loc="left")
    axs[0, 1].set_title(r"b) std($\hat h$) by month", loc="left")
    axs[0, 2].set_title("c) skew(T) by month", loc="left")

    ## bottom row titles and axes
    axs[1, 0].set_title(r"d) PSD of T (K$^2$ yr)", loc="left")
    axs[1, 0].set_xlabel("period (yr)")

    ## lagged correlations
    axs[1, 1].set_title(r"e) corr(T(t), $\hat h$(t+lag))", loc="left")
    axs[1, 2].set_title("f) autocorrelation of T", loc="left")

    ## lag axes, with zero lines
    for ax in axs[1, 1:]:
        ax.set_xlabel("lag (months)")
        ax.axhline(0, c="gray", lw=0.5)
        ax.axvline(0, c="gray", lw=0.5)

    ## month labels on the top row, and the legend
    for ax in axs[0]:
        ax.set_xticks(months, list("JFMAMJJASOND"))
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("First window (1850-1889)")

    return fig


def save(fig, name):
    """
    Save a figure as png and pdf in OUT_DIR.

    Args:
        - fig (matplotlib Figure): figure to save
        - name (str): file name without extension

    Returns:
        - None
    """

    ## both formats
    for ext in ["png", "pdf"]:
        fig.savefig(f"{OUT_DIR}/{name}.{ext}", dpi=150)
    plt.close(fig)


## the panels of params_<group>: the parameters (src.ro.get_params), the noise
## and the net flux's alpha, with titles
PARAM_PANELS = dict(
    R1="$R_1$",
    F1="$F_1$",
    F2="$F_2$",
    eps=r"$\varepsilon$",
    R2="$R_2$",
    R3="$R_3$",
    bh="$b_h$",
    cT="$c_T$",
    ch="$c_h$",
    xi_T=r"noise $\sigma_T$",
    xi_h=r"noise $\sigma_h$",
    alpha_T1=r"$\alpha_1$ (net)",
    alpha_T2=r"$\alpha_2$ (net)",
    alpha_T3=r"$\alpha_3$ (net)",
    alpha_h=r"$\alpha_h$ (net)",
)


def get_param_cycles(case, year):
    """
    Seasonal cycle of every quantity in PARAM_PANELS for one window.

    Args:
        - case (dict): from load_case
        - year (int): the window

    Returns:
        - cycles (dict of np.ndarray): values for months 1-12 by PARAM_PANELS key
          (missing keys: not in this case, e.g. no constant or no alpha_h)
    """

    ## the RO's parameters
    fits = case["fits"].sel(year=year)
    params = src.ro.get_params(fits)

    ## as arrays, by name
    cycles = {}
    for name in params:
        cycles[name] = params[name].values

    ## the noise standard deviation of each equation
    cycles["xi_T"] = fits["xi_stdac"].isel(ranky=0).values
    cycles["xi_h"] = fits["xi_stdac"].isel(ranky=1).values

    ## the net flux's feedback, each coefficient it has
    alpha = case["alpha"]["nhf"].sel(year=year)
    for param in alpha.param.values:
        cycles[f"alpha_{param}"] = alpha.sel(param=param).values

    return cycles


def plot_params(all_cases, tags, year=1870):
    """
    params_<group>.png: the seasonal cycle of each parameter in one window, all
    cases in each panel.

    Args:
        - all_cases (dict): cases (from load_case) by tag
        - tags (list of str): cases
        - year (int): the window

    Returns:
        - fig (matplotlib Figure)
    """

    ## one panel per quantity
    fig, axs = plt.subplots(3, 5, figsize=(16, 8.5), layout="constrained")
    months = np.arange(1, 13)
    for i, tag in enumerate(tags):
        cycles = get_param_cycles(all_cases[tag], year)

        ## each quantity the case has
        for ax, name in zip(axs.flat, PARAM_PANELS):
            if name in cycles:
                ax.plot(months, cycles[name], c=f"C{i}", label=tag)
            else:
                pass

    ## titles, zero lines and month labels
    for ax, title in zip(axs.flat, PARAM_PANELS.values()):
        ax.set_title(title, loc="left")
        ax.axhline(0, c="gray", lw=0.5)
        ax.set_xticks(months, list("JFMAMJJASOND"))

    ## legend once
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"Seasonal cycle of the parameters, window centered on {year}")

    return fig


def get_first_harmonic(values):
    """
    Amplitude and month of maximum of the first harmonic of a seasonal cycle.

    Args:
        - values (np.ndarray): one value per calendar month (1-12)

    Returns:
        - amplitude (float): A in A cos(2 pi (m - m_max) / 12)
        - m_max (float): month of the maximum, in (0, 12]
    """

    ## the Fourier coefficient of the annual harmonic
    omega = 2 * np.pi / 12
    months = np.arange(1, 13)
    c = (values * np.exp(-1j * omega * months)).sum()

    ## A cos(omega (m - m_max)) gives c = 6 A exp(-i omega m_max)
    amplitude = 2 * np.abs(c) / 12
    m_max = (-np.angle(c) / omega) % 12
    if m_max == 0:
        m_max = 12.0
    else:
        pass

    return amplitude, m_max


def format_row(label, values, fmt="{:7.2f}"):
    """
    One table row: a label and formatted values.

    Args:
        - label (str): first column
        - values (iterable of float): the other columns
        - fmt (str): format of each value

    Returns:
        - row (str)
    """

    ## the label, then each value
    row = f"{label:16s}"
    for value in values:
        row += fmt.format(value)

    return row


def get_Ro(case):
    """
    The ocean part of R, R_o = R + alpha (net flux), as in simulate_case.py.

    Args:
        - case (dict): from load_case

    Returns:
        - Ro (xr.DataArray): dims (year, cycle, param), param = src.ro.R_PARAMS
    """

    ## R's coefficients plus the net flux's feedback on them
    R = src.ro.get_R_coefs(case["fits"])
    alpha = case["alpha"]["nhf"].sel(param=src.ro.R_PARAMS)

    return R + alpha


def make_tables(all_cases, tags, all_stats, stats_cesm, effects, diags):
    """
    The tables of one comparison group.

    Args:
        - all_cases (dict): cases (from load_case) by tag
        - tags (list of str): the group's cases
        - all_stats (dict of xr.Dataset): RO stats (from get_stats) by tag
        - stats_cesm (xr.Dataset): CESM2 stats
        - effects (dict of dict of xr.Dataset): experiment effects by tag
        - diags (dict): seasonal std and skewness ("sdT", "skT") of CESM2
          ("CESM2") and each case, by window, from monthly_stats

    Returns:
        - lines (list of str): the tables' lines
    """

    lines = []

    ## fit quality: RMSE against CESM2 over windows
    lines.append("RMSE against CESM2 over windows, and the number of non-finite copies")
    lines.append(
        f"{'case':16s}" + "".join(f"{name:>8s}" for name in STATS) + "     bad"
    )
    for tag in tags:
        s = all_stats[tag]
        row = f"{tag:16s}"
        for name in STATS:
            row += f"{rmse(s[name], stats_cesm[name]):8.3f}"

        ## copies that blew up
        T = all_cases[tag]["sims"]["ctrl"]["T_3"]
        n_bad = int((~np.isfinite(T)).any("time").sum())
        lines.append(row + f"{n_bad:8d}")

    ## annual-mean parameters in the first and last windows
    for year in [1870, 2080]:
        lines.append(f"\nannual-mean parameters, {year}")
        params = {}
        for tag in tags:
            fits = all_cases[tag]["fits"].sel(year=year)
            params[tag] = src.ro.get_params(fits).mean("cycle")

        ## one row per case, one column per parameter
        names = list(params[tags[-1]])
        lines.append(f"{'case':16s}" + "".join(f"{n:>7s}" for n in names))
        for tag in tags:
            values = []
            for n in names:
                values.append(float(params[tag][n]) if n in params[tag] else np.nan)
            lines.append(format_row(tag, values))

    ## months with a positive T^3 coefficient, in each window
    years = all_cases[tags[0]]["fits"].year.values
    lines.append("\nmonths with R_3 > 0 in each window (1870, 1880, ..., 2080)")
    for tag in tags:
        R3 = src.ro.get_params(all_cases[tag]["fits"])["R3"]
        lines.append(format_row(tag, (R3 > 0).sum("cycle").values, "{:3d}"))

    ## the windows' centers (decade of the century), for reading the columns
    lines.append(format_row("(window)", (years % 1000) // 10, "{:3d}"))

    ## first harmonic of the seasonal skewness: amplitude, month of maximum
    lines.append(
        "\nfirst harmonic of skew(T) by month: amplitude, month of maximum (1-12)"
    )
    lines.append(f"{'':16s}{'1870':>16s}{'2080':>16s}")
    for label in ["CESM2"] + tags:
        row = f"{label:16s}"
        for year in [1870, 2080]:
            amplitude, m_max = get_first_harmonic(diags[label][year]["skT"])
            row += f"{amplitude:8.2f}{m_max:8.1f}"
        lines.append(row)

    ## experiment effects in 2080 (warm / cold, K^2)
    lines.append("\neffect on the warm / cold contributions in 2080 (K^2)")
    lines.append(f"{'case':16s}" + "".join(f"{n:>13s}" for n in EXPERIMENTS))
    for tag in tags:
        row = f"{tag:16s}"
        for name in EXPERIMENTS:
            e = effects[tag][name].sel(year=2080)
            row += f"  {float(e['warm']):5.2f}/{float(e['cold']):5.2f}"
        lines.append(row)

    ## change in the contributions from 1870 to 2080, for scale
    lines.append("\nchange in the warm / cold contributions, 1870 to 2080 (K^2)")
    change = stats_cesm.sel(year=2080) - stats_cesm.sel(year=1870)

    ## CESM2 first, then each case
    lines.append(
        f"{'CESM2':16s}  {float(change['warm']):5.2f} / {float(change['cold']):5.2f}"
    )
    for tag in tags:
        s = all_stats[tag]
        change = s.sel(year=2080) - s.sel(year=1870)
        lines.append(
            f"{tag:16s}  {float(change['warm']):5.2f} / {float(change['cold']):5.2f}"
        )

    ## the experiments' T^3 coefficient: (window, month) pairs with R_3 > 0, and
    ## the largest value
    lines.append(
        "\nexperiments' R_3: number of (window, month) with R_3 > 0 / largest R_3"
    )
    lines.append(f"{'case':16s}" + "".join(f"{n:>13s}" for n in EXPERIMENTS))
    for tag in tags:
        R3 = all_cases[tag]["experiments_R3"]
        if R3 is None:
            lines.append(f"{tag:16s}  (not saved)")
        else:
            row = f"{tag:16s}"
            for name in EXPERIMENTS:
                x = R3.sel(experiment=name)
                row += f"  {int((x > 0).sum()):5d}/{float(x.max()):5.2f}"
            lines.append(row)

    ## the T^3 bound's effect on R_o: bounded minus unbounded case
    lines.append(
        "\nT^3 bound's effect on R_o = R + alpha (bounded minus unbounded case):"
        " annual mean and DJF of R_o1, R_o2, R_o3"
    )
    lines.append(
        f"{'':16s}{'year':>6s}" + "".join(f"{n:>14s}" for n in src.ro.R_PARAMS)
    )
    for tag in tags:
        partner = tag.removesuffix("_neg")
        if tag.endswith("_neg") and partner in all_cases:
            dRo = get_Ro(all_cases[tag]) - get_Ro(all_cases[partner])
            for year in [1870, 2080]:
                d = dRo.sel(year=year)
                row = f"{tag:16s}{year:6d}"

                ## each coefficient's annual mean and DJF mean (Dec, Jan, Feb)
                for param in src.ro.R_PARAMS:
                    annual = float(d.sel(param=param).mean("cycle"))
                    djf = float(d.sel(param=param).isel(cycle=[11, 0, 1]).mean("cycle"))
                    row += f"  {annual:5.2f}/{djf:6.2f}"
                lines.append(row)
        else:
            pass

    ## round-trip test: true vs refitted annual-mean parameters
    for tag in tags:
        lines += roundtrip_tables(tag, all_cases[tag]["roundtrip"])

    return lines


def roundtrip_tables(tag, rt):
    """
    Round-trip tables of one case: true vs refitted annual-mean parameters
    (mean +- std over seeds), and R_2 and R_3 by month (seasonal recovery).

    Args:
        - tag (str): the case
        - rt (xr.Dataset or None): from load_case

    Returns:
        - lines (list of str)
    """

    lines = []
    if rt is None:
        lines.append(f"\nround trip, {tag}: not run")
    else:
        ## the refitted parameters' names (the true ones have the prefix "true_")
        names = []
        for n in rt:
            if not n.startswith("true_"):
                names.append(n)
        rt_mean = rt.mean("cycle")

        ## annual means, each window
        for year in rt.year.values:
            lines.append(
                f"\nround trip, {tag}, {year}: annual means (refit: mean +- std over seeds)"
            )
            lines.append(f"{'':8s}" + "".join(f"{n:>13s}" for n in names))

            ## the true values
            row = f"{'true':8s}"
            for n in names:
                row += f"{float(rt_mean[f'true_{n}'].sel(year=year)):13.2f}"
            lines.append(row)

            ## each h_hat method
            for method in rt.method.values:
                x = rt_mean.sel(dict(year=year, method=method))

                ## mean and spread over seeds of each parameter
                row = f"{method:8s}"
                for n in names:
                    row += f"{float(x[n].mean('seed')):8.2f}+-{float(x[n].std('seed')):3.2f}"
                lines.append(row)

            ## R_2 and R_3 by month: true, and refit mean (std) without h_hat changes
            for n in ["R2", "R3"]:
                lines.append(f"  {n} by month, J to D (refit: 'none', mean and std)")
                true = rt[f"true_{n}"].sel(year=year).values
                refit = rt[n].sel(dict(year=year, method="none"))

                ## one row each: true, refit mean, refit spread
                lines.append(format_row("  true", true))
                lines.append(format_row("  refit", refit.mean("seed").values))
                lines.append(format_row("  std", refit.std("seed").values))

    return lines


## every case whose simulations have finished
all_cases = {}
for tag in cases.CASES:
    if os.path.exists(f"{OUT_DIR}/{tag}/sims_{list(EXPERIMENTS)[-1]}.nc"):
        all_cases[tag] = load_case(tag)
    else:
        print(f"skipping {tag} (not finished)")

## CESM2 stats: T doesn't depend on the case, and the stats don't depend on the
## centering, so any case's data will do
first = next(iter(all_cases.values()))
stats_cesm = get_stats(first["data"], first["T_scale"])

## RO stats and experiment effects for each case
all_stats = {}
effects = {}
for tag, case in all_cases.items():
    all_stats[tag] = get_stats(case["sims"]["ctrl"], case["T_scale"])

    ## effect of each experiment: control minus experiment (median over copies)
    contrib_ctrl = src.ro.get_contributions(case["sims"]["ctrl"], case["T_scale"])
    effects[tag] = {}
    for name in EXPERIMENTS:
        contrib = src.ro.get_contributions(case["sims"][name], case["T_scale"])
        effects[tag][name] = (contrib_ctrl - contrib).median("member")

## baseline diagnostics: CESM2 with each h_hat method, and every case
diags_cesm = {}
diags = {}
for tag, case in all_cases.items():
    data0 = case["data"].isel(year=0)
    method = "hmonth" if case["config"]["h_by_month"] else "hconst"
    diags_cesm[method] = baseline_diagnostics(data0, case["T_scale"])
    diags[tag] = baseline_diagnostics(
        case["sims"]["ctrl"].isel(year=0), case["T_scale"]
    )

## seasonal std and skewness in the first and last windows, for the tables
seasonal = {}
for label, case in [("CESM2", first)] + list(all_cases.items()):
    if label == "CESM2":
        x = case["data"]
    else:
        x = case["sims"]["ctrl"]
    seasonal[label] = {}

    ## T in K, shape (member, time), for each window
    for year in [1870, 2080]:
        x_year = x.sel(year=year)
        T = case["T_scale"] * x_year["T_3"].transpose("member", "time").values
        sdT, skT = monthly_stats(T, x_year.time.dt.month.values)
        seasonal[label][year] = dict(sdT=sdT, skT=skT)

## each comparison group: its finished cases, figures and tables
for group, group_tags in cases.GROUPS.items():
    tags = []
    for tag in group_tags:
        if tag in all_cases:
            tags.append(tag)
        else:
            pass
    if len(tags) == 0:
        print(f"skipping group {group} (no finished cases)")
    else:
        ## figures
        save(plot_over_time(all_stats, stats_cesm, tags), f"compare_{group}")
        save(plot_experiments(effects, tags), f"experiments_{group}")
        save(plot_baseline(diags, diags_cesm, tags), f"baseline_{group}")
        save(plot_params(all_cases, tags), f"params_{group}")

        ## tables, printed and saved
        lines = make_tables(all_cases, tags, all_stats, stats_cesm, effects, seasonal)
        text = "\n".join(lines)
        print(f"\n===== {group} =====\n" + text)
        with open(f"{OUT_DIR}/tables_{group}.txt", "w") as f:
            f.write(text + "\n")
