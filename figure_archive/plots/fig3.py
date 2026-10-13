"""Draw the six independent sensitivity panels of manuscript Fig. 3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fig3"


def plot(panel: str, output: Path) -> None:
    if panel not in "abcdef":
        raise ValueError(f"Unknown Fig. 3 panel: {panel}")
    settings = pd.read_csv(DATA / "fig3_panels.csv").set_index("panel").loc[panel]
    data = pd.read_csv(DATA / "fig3_curves.csv")
    data = data.loc[data.panel.eq(panel)]
    if data.empty:
        raise ValueError(f"Fig. 3{panel} has no data")

    style = {
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(7.5, 3.5), layout="constrained")
        for (_, name), frame in data.groupby(["series_order", "series"], sort=True):
            frame = frame.sort_values("point")
            first = frame.iloc[0]
            axis.plot(
                frame.x,
                frame.loss_plot,
                label=first.label,
                color=first.color,
                linestyle=first.line_style,
                linewidth=2,
            )
        axis.set_xlabel(settings.x_label)
        axis.set_ylabel(settings.y_label)
        if np.isfinite(settings.y_min) and np.isfinite(settings.y_max):
            axis.set_ylim(settings.y_min, settings.y_max)
        else:
            values = data.loss_plot.to_numpy(float)
            margin = 0.03 * (values.max() - values.min())
            axis.set_ylim(values.min() - margin, values.max() + margin)
            axis.set_xticks(np.arange(1, int(data.x.max()) + 1, 3))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_formatter(
            FormatStrFormatter("%.3f" if panel == "f" else "%.2f")
        )
        axis.spines[["top", "right"]].set_visible(False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("0.25")
        axis.tick_params(direction="out", length=3, width=0.8, colors="0.25")
        axis.yaxis.grid(True, linestyle="--", linewidth=0.8, color="0.85", alpha=0.35)
        axis.legend(loc="upper right", frameon=False, handlelength=2.2)
        axis.text(
            -0.10,
            1.02,
            panel,
            transform=axis.transAxes,
            fontsize=20,
            fontweight="bold",
            va="top",
        )
        output.mkdir(parents=True, exist_ok=True)
        figure.savefig(output / f"Fig3{panel}.png", dpi=600, facecolor="white")
        plt.close(figure)
