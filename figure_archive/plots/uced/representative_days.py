"""Twelve shared representative days: load and available wind/PV."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.uced.style import FONT
from plots.uced.style import register_font as register_arial

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "data/supplementary/fig3_typical_days.csv"
OUTPUT_DIR = ROOT / "figures"


REGION_EN = {
    "华北": "North China",
    "华东": "East China",
    "华中": "Central China",
    "西北": "Northwest China",
    "西南": "Southwest China",
    "东北": "Northeast China",
    "南方": "South China",
}


BLUE = "#1f77b4"


RED = "#d62728"


def configure_style() -> None:
    register_arial()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "axes.linewidth": 0.9,
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
        }
    )


def resolve_region(requested: str, regions: list[str]) -> tuple[str, np.ndarray]:
    value = requested.strip()
    if value.casefold() == "national":
        return "National", np.arange(len(regions), dtype=int)
    aliases = {region.casefold(): region for region in regions}
    aliases.update(
        {english.casefold(): chinese for chinese, english in REGION_EN.items()}
    )
    selected = aliases.get(value.casefold())
    if selected is None or selected not in regions:
        choices = ", ".join(["national", *regions, *REGION_EN.values()])
        raise ValueError(f"Unknown region {requested!r}. Choose one of: {choices}")
    return REGION_EN.get(selected, selected), np.array(
        [regions.index(selected)], dtype=int
    )


def expand_limits(
    values: np.ndarray, include_zero: bool = False
) -> tuple[float, float]:
    low = (
        min(0.0, float(np.nanmin(values))) if include_zero else float(np.nanmin(values))
    )
    high = max(0.0, float(np.nanmax(values)))
    span = max(high - low, max(abs(low), abs(high), 1.0) * 0.02)
    return low - 0.04 * span, high + 0.06 * span


def load_series(
    year: int, region_request: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(INPUT_CSV).sort_values(["day_index", "point_index_15min"])
    frame = frame.loc[frame.year.eq(year) & frame.region.eq("National")]
    if region_request.lower() != "national" or len(frame) != 12 * 96:
        raise ValueError("This archive contains the 2050 national 12-day figure only")
    load = frame.load_mw.to_numpy(float).reshape(12, 96)
    renewable = frame.wind_pv_actual_mw.to_numpy(float).reshape(12, 96)
    context = {"year": year, "region": "National", "load": load, "renewable": renewable}
    return frame, context


def render(frame: pd.DataFrame, context: dict[str, object], output_stem: str) -> Path:
    load = np.asarray(context["load"], dtype=float)
    renewable = np.asarray(context["renewable"], dtype=float)
    common_limits = expand_limits(np.concatenate([load, renewable]), include_zero=True)
    # All intervals last 0.25 h, so the duration cancels in the energy ratio.
    daily_load_energy = load.sum(axis=1) * 0.25
    penetration_pct = 100.0 * (renewable.sum(axis=1) * 0.25) / daily_load_energy
    fig, axes = plt.subplots(4, 3, figsize=(15, 18), sharex=True)
    fig.subplots_adjust(
        left=0.095, right=0.905, top=0.955, bottom=0.070, hspace=0.36, wspace=0.34
    )
    hours = np.arange(96, dtype=float) * 0.25
    ticks = np.arange(0, 25, 3)
    tick_labels = [f"{int(hour):02d}:00" for hour in ticks]
    for day, axis in enumerate(axes.flat):
        day_frame = frame.loc[frame["day_index"].eq(day)]
        axis.plot(hours, day_frame["load_mw"], color=BLUE, linestyle="-", linewidth=2.0)
        axis.plot(
            hours,
            day_frame["wind_pv_actual_mw"],
            color=RED,
            linestyle="--",
            linewidth=2.0,
        )
        axis.set(
            xlim=(0, 24),
            ylim=common_limits,
            title=f"Typical Day {day + 1} ({penetration_pct[day]:.2f}%)",
        )
        axis.set_ylabel("Power (MW)")
        axis.set_xticks(ticks, tick_labels, rotation=45, ha="right")
        axis.set_xlabel("Time (h)")
        for spine in axis.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.9)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(
            axis="both",
            direction="out",
            length=3,
            width=0.8,
            labelsize=11,
            colors="black",
        )
        axis.tick_params(axis="x", labelbottom=True)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, linewidth=2.0, label="Load"),
            Line2D(
                [0],
                [0],
                color=RED,
                linestyle="--",
                linewidth=2.0,
                label="Wind + PV actual",
            ),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.987),
        frameon=False,
    )
    png = OUTPUT_DIR / f"{output_stem}.png"
    fig.savefig(png, dpi=600, facecolor="white")
    plt.close(fig)
    return png
