"""Build the confirmed E3 Supplementary Fig. 2 with an EMD result strip.

The two panels reproduce the supported-bin median relationships from the
confirmed H candidate.  The compact table underneath reports the primary
equal-bin Earth mover's distance summaries.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CURVE_CSV = ROOT / "data/supplementary/weather_power_curves_centered_bins.csv"
EMD_CSV = ROOT / "data/supplementary/conditional_emd_summary.csv"
OUT_DIR = ROOT / "figures"
OUT_STEM = OUT_DIR / "Supplementary_Fig2"

SERIES = ["real", "level1", "level2", "level3"]
LABELS = {
    "real": "Real-site data",
    "level1": "Level 1",
    "level2": "Level 2",
    "level3": "Level 3",
}
COLORS = {
    "real": "#171717",
    "level1": "#A6CEE3",
    "level2": "#4A98C9",
    "level3": "#08519C",
}
MARKERS = {"real": "o", "level1": "s", "level2": "D", "level3": "^"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_curve_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with CURVE_CSV.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "technology": raw["technology"],
                    "series": raw["series"],
                    "bin_center": float(raw["bin_center"]),
                    "median": float(raw["median_site_scaled_power"]),
                    "site_count": int(raw["site_count"]),
                }
            )
    return rows


def load_emd_rows() -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    with EMD_CSV.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["aggregation"] != "equal_bin":
                continue
            values[(raw["technology"], raw["surrogate_level"])] = float(
                raw["earth_movers_distance"]
            )
    expected = {
        (technology, level)
        for technology in ("wind", "solar")
        for level in ("level1", "level2", "level3")
    }
    if set(values) != expected:
        raise RuntimeError(f"Unexpected EMD result keys: {sorted(values)}")
    return values


def selected_rows(
    rows: list[dict[str, object]], technology: str, series: str
) -> list[dict[str, object]]:
    minimum = 1.0 if technology == "wind" else 100.0
    maximum = 15.0 if technology == "wind" else 950.0
    values = [
        row
        for row in rows
        if row["technology"] == technology
        and row["series"] == series
        and minimum <= float(row["bin_center"]) <= maximum
        and int(row["site_count"]) >= 100
    ]
    return sorted(values, key=lambda row: float(row["bin_center"]))


def format_emd(value: float) -> str:
    return f"{value:.3f}"


def build() -> None:
    configure_style()
    curves = load_curve_rows()
    emd = load_emd_rows()

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.95), sharey=True)
    settings = (
        ("wind", "Wind", "ERA5 wind speed (m/s)"),
        ("solar", "Solar", "ERA5 surface solar irradiance (W/m²)"),
    )

    for panel_index, (ax, (technology, title, xlabel)) in enumerate(
        zip(axes, settings)
    ):
        for series in SERIES:
            values = selected_rows(curves, technology, series)
            x = np.array([float(row["bin_center"]) for row in values])
            y = np.array([float(row["median"]) for row in values])
            ax.plot(
                x,
                y,
                color=COLORS[series],
                linewidth=1.8 if series == "real" else 1.35,
                marker=MARKERS[series],
                markersize=3.0 if series == "real" else 2.7,
                markerfacecolor=COLORS[series] if series == "real" else "white",
                markeredgecolor=COLORS[series],
                markeredgewidth=0.65,
                label=LABELS[series],
                zorder=4 if series == "real" else 3,
            )

        if technology == "wind":
            ax.set_xlim(0.82, 15.18)
            ax.set_xticks([1, 4, 7, 10, 13, 15])
        else:
            ax.set_xlim(90, 960)
            ax.set_xticks([100, 300, 500, 700, 900])
        ax.set_ylim(-0.015, 1.13)
        ax.set_yticks(np.arange(0.0, 1.11, 0.2))
        ax.set_xlabel(xlabel, labelpad=4)
        ax.set_title(title, loc="left", pad=6, fontweight="bold")
        ax.text(
            -0.12,
            1.04,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Median site-normalized power", labelpad=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.91),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.25,
        handletextpad=0.55,
    )

    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.38, top=0.79, wspace=0.22)

    fig.text(
        0.5,
        0.195,
        "Mean EMD across weather ranges",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
    )
    table_ax = fig.add_axes([0.105, 0.012, 0.79, 0.145])
    table_ax.axis("off")
    table_data = [
        [
            "Wind",
            format_emd(emd[("wind", "level1")]),
            format_emd(emd[("wind", "level2")]),
            format_emd(emd[("wind", "level3")]),
        ],
        [
            "Solar",
            format_emd(emd[("solar", "level1")]),
            format_emd(emd[("solar", "level2")]),
            format_emd(emd[("solar", "level3")]),
        ],
    ]
    table = table_ax.table(
        cellText=table_data,
        colLabels=["", "Level 1", "Level 2", "Level 3"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.18, 0.273, 0.273, 0.273],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.15)
    last_row = len(table_data)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.65)
        if row == 0:
            cell.visible_edges = "TB"
            cell.get_text().set_fontweight("bold")
        elif row == last_row:
            cell.visible_edges = "B"
        else:
            cell.visible_edges = ""
        if column == 0 and row > 0:
            cell.get_text().set_fontweight("bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_STEM.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def plot(output: Path) -> None:
    global OUT_DIR, OUT_STEM
    OUT_DIR = output
    OUT_STEM = output / "Supplementary_Fig2"
    build()


if __name__ == "__main__":
    plot(OUT_DIR)
