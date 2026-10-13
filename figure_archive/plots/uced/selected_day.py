"""Selected 2050, day-index 7, 12% error, member 6 trajectories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.uced.style import FONT
from plots.uced.style import register_font as register_arial

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
REPLAY = ROOT / "data/supplementary/fig4_8_national.csv"
REGIONAL = ROOT / "data/supplementary/fig8_regional_transfer.csv"
TARGET = None
register_arial()
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "axes.labelsize": 12,
        "axes.labelweight": "normal",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "font.weight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.formatter.use_mathtext": True,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "mathtext.cal": f"{FONT}:italic",
        "mathtext.sf": FONT,
        "mathtext.tt": FONT,
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.9,
        "lines.antialiased": True,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",
        "lines.dash_joinstyle": "round",
        "patch.antialiased": True,
    }
)

REG_EN = {
    "华北": "North China",
    "华东": "East China",
    "华中": "Central China",
    "西北": "Northwest China",
    "西南": "Southwest China",
    "东北": "Northeast China",
    "南方": "South China",
}


COL = {
    "Coal": "#4e79a7",
    "Oil/Gas": "#9c755f",
    "Nuclear": "#bab0ab",
    "Bioenergy": "#59a14f",
    "Coal-CCS": "#76b7b2",
    "Gas-CCS": "#edc948",
    "Bio-CCS": "#8cd17d",
    "Hydro": "#4e9fd1",
    "CSP": "#f28e2b",
    "Wind": "#00a6b2",
    "PV": "#e5c849",
    "Storage discharge": "#b07aa1",
    "Import": "#79706e",
    "Qflex": "#d62728",
}


def restyle_axes(fig):
    for ax in fig.axes:
        right_axis = ax.yaxis.get_label_position() == "right"
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(right_axis)
        ax.spines["left"].set_visible(not right_axis)
        ax.spines["bottom"].set_visible(not right_axis)
        ax.tick_params(direction="out", length=3, width=0.8, labelsize=11)


def save_figure(fig, name):
    if TARGET is not None and name != TARGET:
        plt.close(fig)
        return
    restyle_axes(fig)
    fig.savefig(OUT / (name + ".png"), dpi=600, facecolor="white")
    plt.close(fig)


def flat(a):
    return np.asarray(a, dtype=float).reshape(-1)


def pchip_hourly_display(values, points_per_hour=16):
    """PCHIP interpolation of 24 hourly means for display."""
    y = np.asarray(values, dtype=float).reshape(24)
    x = np.arange(24, dtype=float) + 0.5
    h = np.diff(x)
    d = np.diff(y)
    m = np.zeros_like(y)
    for k in range(1, 23):
        if d[k - 1] * d[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def endpoint(d0, d1, h0, h1):
        slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if slope * d0 <= 0:
            return 0.0
        return 3 * d0 if d0 * d1 < 0 and abs(slope) > abs(3 * d0) else slope

    m[0] = endpoint(d[0], d[1], h[0], h[1])
    m[-1] = endpoint(d[-1], d[-2], h[-1], h[-2])
    xd = np.linspace(x[0], x[-1], 23 * points_per_hour + 1)
    idx = np.clip(np.searchsorted(x, xd, side="right") - 1, 0, 22)
    z = (xd - x[idx]) / h[idx]
    yd = (
        (2 * z**3 - 3 * z**2 + 1) * y[idx]
        + (z**3 - 2 * z**2 + z) * h[idx] * m[idx]
        + (-2 * z**3 + 3 * z**2) * y[idx + 1]
        + (z**3 - z**2) * h[idx] * m[idx + 1]
    )
    return xd, yd


def replay_national():
    """Load the archived selected-day national trajectory from a regular CSV."""
    frame = pd.read_csv(REPLAY).sort_values("point_index")
    if len(frame) != 97 or not np.array_equal(frame.point_index, np.arange(97)):
        raise ValueError("Selected-day table must contain points 0 through 96")
    out = {"input": {}}
    for key in [
        "load_mw",
        "wind_available_mw",
        "pv_available_mw",
        "forecast_wind_mw",
        "forecast_pv_mw",
        "fixed_import_mw",
        "fixed_export_mw",
        "fixed_transmission_loss_mw",
    ]:
        out["input"][key] = frame[f"input_{key}"].to_numpy(float)[:96]
    keys = [
        "wind_used",
        "pv_used",
        "hydro",
        "csp_power",
        "qflex_up",
        "storage_charge",
        "storage_discharge",
        "fixed_import_mw",
        "fixed_export_mw",
        "fixed_transmission_loss_mw",
    ]
    for state in ("uc", "ed"):
        out[state] = {}
        for key in ("p_net", "p_gross", "online"):
            out[state][key] = np.stack(
                [frame[f"{state}_{key}_{i}"].to_numpy(float)[:96] for i in range(7)]
            )
        for key in keys:
            out[state][key] = frame[f"{state}_{key}"].to_numpy(float)[:96]
        out[state]["storage_soc"] = frame[f"{state}_storage_soc"].to_numpy(float)
    out["techs"] = [
        "Coal",
        "Oil/Gas",
        "Nuclear",
        "Bioenergy",
        "Coal-CCS",
        "Gas-CCS",
        "Bio-CCS",
    ]
    return out


def make_figures(n):
    """Selected-day dispatch and flexibility plots."""
    t = np.arange(96) / 4
    inp = n["input"]
    actual = inp["wind_available_mw"] + inp["pv_available_mw"]
    fc = inp["forecast_wind_mw"] + inp["forecast_pv_mw"]
    # 01: raw 15-min forecast input, intentionally with no smoothing.
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax[0].plot(t, inp["load_mw"], label="Load", color="black", lw=2.0)
    ax[0].plot(t, inp["wind_available_mw"], label="Actual Wind", color=COL["Wind"])
    ax[0].plot(t, inp["pv_available_mw"], label="Actual PV", color=COL["PV"])
    ax[0].plot(
        t, inp["forecast_wind_mw"], label="Forecast Wind", color=COL["Wind"], ls="--"
    )
    ax[0].plot(t, inp["forecast_pv_mw"], label="Forecast PV", color=COL["PV"], ls="--")
    ax[0].legend(ncol=3)
    ax[0].set(title="30 May 2050 — M006 forecast input", ylabel="MW")
    err = fc - actual
    ax[1].fill_between(
        t,
        0,
        err,
        where=err >= 0,
        color="#d95f02",
        alpha=0.7,
        label="Overforecast (> 0)",
        antialiased=True,
    )
    ax[1].fill_between(
        t,
        0,
        err,
        where=err < 0,
        color="#1b9e77",
        alpha=0.7,
        label="Underforecast (< 0)",
        antialiased=True,
    )
    ax[1].plot(t, err, color="#444", lw=2.0, label="Forecast VRE − Actual VRE")
    ax[1].axhline(0, color="black", lw=1.1)
    ax[1].legend(ncol=3)
    ax[1].set(xlabel="Hour", ylabel="MW", xlim=(0, 24))
    save_figure(fig, "Supplementary_Fig4")
    # 02: exact four-interval arithmetic means (energy-preserving 1-h averages).
    th = np.arange(25)
    hmean = lambda x: np.asarray(x, dtype=float).reshape(24, 4).mean(axis=1)
    hext = lambda x: np.r_[x, x[-1]]
    order = [
        "Coal",
        "Oil/Gas",
        "Nuclear",
        "Bioenergy",
        "Coal-CCS",
        "Gas-CCS",
        "Bio-CCS",
        "Hydro",
        "CSP",
        "Wind",
        "PV",
    ]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.16, wspace=0.06)
    ymins = []
    ymaxs = []
    for ax, st, title in zip(axs, ["uc", "ed"], ["UC Dispatch", "ED Dispatch"]):
        comp = [hmean(n[st]["p_net"][i]) for i in range(7)] + [
            hmean(n[st]["hydro"]),
            hmean(n[st]["csp_power"]),
            hmean(n[st]["wind_used"]),
            hmean(n[st]["pv_used"]),
        ]
        base = np.sum(comp, axis=0)
        snet = hmean(n[st]["storage_discharge"] - n[st]["storage_charge"])
        ax.stackplot(
            th,
            *[hext(x) for x in comp],
            labels=order,
            colors=[COL[x] for x in order],
            alpha=0.86,
            antialiased=True,
            linewidth=0,
        )
        # Storage_net is a signed stack: discharge is above the positive stack;
        # charging is below the zero line, rather than hidden inside the stack.
        ax.fill_between(
            th,
            hext(base),
            hext(base + np.maximum(snet, 0)),
            color="#b07aa1",
            alpha=0.86,
            label="Storage_net",
            antialiased=True,
        )
        ax.fill_between(
            th,
            0,
            hext(np.minimum(snet, 0)),
            color="#b07aa1",
            alpha=0.86,
            antialiased=True,
        )
        ax.plot(th, hext(hmean(inp["load_mw"])), color="black", lw=2.0, label="Load")
        ax.plot(
            th,
            hext(hmean(n[st]["qflex_up"])),
            color=COL["Qflex"],
            lw=2.0,
            label="Qflex",
        )
        ax.axhline(0, color="black", lw=0.9)
        ax.set(title=title, xlabel="Hour", ylabel="MW", xlim=(0, 24))
        ymins.append(min(0, float(np.min(snet))))
        ymaxs.append(float(np.max(base + np.maximum(snet, 0))))
    ymin = min(ymins)
    ymax = max(ymaxs)
    for ax in axs:
        ax.set_ylim(min(-0.05 * ymax, ymin * 1.05), 1.05 * ymax)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=7,
        fontsize=11,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.7,
    )
    fig.subplots_adjust(top=0.84)
    fig.text(
        0.5,
        0.035,
        "The UC–ED model is solved at 15-min resolution; hourly averages of the four constituent 15-min intervals are shown for visualization.",
        ha="center",
        fontsize=11,
    )
    save_figure(fig, "Supplementary_Fig5")
    # 03--05: values are first exact 4x15-min hourly means; PCHIP below is
    # display-only, shape-preserving, and passes through every hourly point.
    hm = lambda a: np.asarray(a, dtype=float).reshape(24, 4).mean(axis=1)
    smooth = lambda a: pchip_hourly_display(hm(a))
    # 03
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    ed = n["ed"]
    therm = ed["p_gross"].sum(0)
    online = n["uc"]["online"].sum(0)
    head = np.maximum(online - therm, 0)
    x, y = smooth(ed["storage_charge"])
    ax[0].plot(x, y, label="Storage charge", color="#8c564b")
    x, y = smooth(ed["storage_discharge"])
    ax[0].plot(x, y, label="Storage discharge", color=COL["Storage discharge"])
    a = ax[0].twinx()
    x, y = smooth(ed["storage_soc"][:-1])
    a.plot(x, y, color="#111", label="SOC")
    ax[0].set_ylabel("MW")
    a.set_ylabel("MWh")
    ax[0].legend(loc="upper left")
    a.legend(loc="upper right")
    ax[0].set_title("Storage flexibility")
    for v, label, color in [
        (online, "Online thermal capacity", "#4e79a7"),
        (therm, "Thermal generation", "#e15759"),
        (head, "Physical headroom", "#59a14f"),
    ]:
        x, y = smooth(v)
        ax[1].plot(x, y, label=label, color=color)
    ax[1].legend()
    ax[1].set_ylabel("MW")
    for v, label, color in [
        (ed["hydro"], "Hydro", COL["Hydro"]),
        (ed["csp_power"], "CSP", COL["CSP"]),
    ]:
        x, y = smooth(v)
        ax[2].plot(x, y, label=label, color=color)
    ax[2].legend()
    ax[2].set(xlabel="Hour", ylabel="MW", xlim=(0, 24))
    save_figure(fig, "Supplementary_Fig6")
    # 04
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    uc = n["uc"]
    curt_uc = fc - uc["wind_used"] - uc["pv_used"]
    curt_ed = actual - ed["wind_used"] - ed["pv_used"]
    x, y = smooth(fc - actual)
    ax[0].plot(x, y, color="#555")
    ax[0].axhline(0, color="k", lw=0.9)
    ax[0].set(title="Forecast error (Forecast − Actual)", ylabel="MW")
    for v, label, color, lw in [
        (curt_uc, "UC curtailment", "#e15759", 2.0),
        (curt_ed, "ED curtailment", "#4e79a7", 2.0),
        (curt_uc - curt_ed, "CurtUC − CurtED", "#7a3e9d", 2.0),
    ]:
        x, y = smooth(v)
        ax[1].plot(x, y, label=label, color=color, lw=lw)
    ax[1].legend()
    ax[1].set_ylabel("MW")
    x, y = smooth(ed["qflex_up"])
    ax[2].plot(x, y, color=COL["Qflex"], lw=2.0, label="Qflex")
    ax[2].legend()
    ax[2].set(xlabel="Hour", ylabel="MW", xlim=(0, 24))
    save_figure(fig, "Supplementary_Fig7")
    # 05 regional transfer using the exact replay input boundary.
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    regions = pd.read_csv(REGIONAL)
    for name, group in regions.groupby("region", sort=False):
        values = group.sort_values("point_index").net_import_mw.to_numpy(float)
        x, y = smooth(values)
        ax.plot(x, y, label=REG_EN[str(name)], lw=2.0)
    ax.axhline(0, color="k", lw=0.9)
    ax.set(
        title="Regional transfer: positive = net import",
        xlabel="Hour",
        ylabel="MW",
        xlim=(0, 24),
    )
    ax.legend(ncol=4, fontsize=11)
    save_figure(fig, "Supplementary_Fig8")
