"""Plot Extended Data Figures 1 and 3 from their evaluation records."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/extended_data"


def _save(figure: plt.Figure, name: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.png", dpi=600, facecolor="white")
    plt.close(figure)


def fig1(output: Path) -> None:
    """Plot the first five fine-tuning epochs for four initializations."""
    series = (
        ("valid_loss_headonly_pretrain.csv", "Head (pretrain)", "#8B0000", "-", 2.1),
        (
            "valid_loss_headonly_pretrain_10dpst.csv",
            "Head (pretrain-10d)",
            "0.35",
            "--",
            1.8,
        ),
        (
            "valid_loss_headonly_pretrain_large.csv",
            "Head (pretrain-Large)",
            "0.55",
            "-.",
            1.8,
        ),
        (
            "valid_loss_headonly_pretrain_20000.csv",
            "Head (pretrain-20k)",
            "0.00",
            ":",
            1.8,
        ),
    )
    with plt.rc_context(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(6.4, 4.2), dpi=600)
        all_y = []
        for filename, label, color, line_style, width in series:
            data = pd.read_csv(DATA / filename).sort_values("epoch")
            data = data.loc[data.epoch.le(5)]
            x = data.epoch.to_numpy(dtype=float)
            y = data.global_eval_loss_weighted.to_numpy(dtype=float)
            all_y.extend(y)
            axis.plot(
                x,
                y,
                label=label,
                color=color,
                linestyle=line_style,
                linewidth=width,
                marker="o",
                markersize=3.2,
                zorder=3,
            )
        axis.set_xlabel("Epoch", fontsize=12, color="0.15")
        axis.set_ylabel("Held-out real-site MSE", fontsize=12, color="0.15")
        axis.set_xlim(1, 5)
        axis.set_xticks([1, 2, 3, 4, 5])
        spread = max(all_y) - min(all_y)
        axis.set_ylim(min(all_y) - 0.08 * spread, max(all_y) + 0.08 * spread)
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.tick_params(axis="both", labelsize=11, colors="0.25")
        axis.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_linewidth(1)
            axis.spines[side].set_color("0.25")
        axis.legend(
            loc="upper right",
            ncol=1,
            frameon=False,
            fontsize=11,
            handlelength=2.2,
            labelspacing=0.35,
            handletextpad=0.6,
            borderaxespad=0.3,
        )
        figure.subplots_adjust(left=0.16, right=0.97, bottom=0.14, top=0.96)
        output.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output / "ExtendedDataFig1.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def fig3(output: Path) -> None:
    """Plot the three archived ablation curves; CSV retains raw and plotted loss."""
    data = pd.read_csv(DATA / "fig3_ablation.csv")
    style = (
        (
            "no_temporal_no_attention",
            "w/o Temporal Experts & Cross-Attention",
            "#3b0f70",
        ),
        ("no_temporal", "w/o Temporal Experts", "#8c2981"),
        ("base", "base", "#de4968"),
    )
    with plt.rc_context(
        {"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}
    ):
        figure, axis = plt.subplots(figsize=(7.4, 3.4), dpi=600)
        for key, label, color in style:
            frame = data.loc[data.series.eq(key)].sort_values("point")
            if len(frame) != 55:
                raise ValueError(f"Extended Data Fig. 3: {key} requires 55 points")
            axis.plot(
                frame.samples_m,
                frame.eval_loss_smoothed,
                color=color,
                linewidth=2,
                label=label,
            )
        axis.set_xlabel("Samples (M)")
        axis.set_ylabel("Eval loss")
        axis.set_xlim(0, 2.15)
        axis.set_ylim(0.36, 0.96)
        axis.set_xticks(np.arange(0.25, 2.01, 0.25))
        axis.set_yticks([0.45, 0.60, 0.75, 0.90])
        axis.grid(axis="y", color="0.85", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper right", frameon=False)
        figure.subplots_adjust(left=0.094, right=0.986, bottom=0.16, top=0.97)
        _save(figure, "ExtendedDataFig3", output)
