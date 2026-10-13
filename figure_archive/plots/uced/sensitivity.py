"""Reserve and flexible-resource penalty sensitivity curves."""

import matplotlib.pyplot as plt
import pandas as pd

from plots.uced.style import FONT


def render(summary_path, output_dir):
    summary = pd.read_csv(summary_path)
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 11,
            "legend.frameon": False,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "Reserve 0.8x": "#4C78A8",
        "Baseline": "#333333",
        "Reserve 1.2x": "#E45756",
        "Penalty 2000": "#4C78A8",
        "Penalty 5000": "#E45756",
    }
    panels = [
        ("reserve", 2030, "qflex_load_pct"),
        ("reserve", 2050, "qflex_load_pct"),
        ("penalty", 2030, "unit_cost_cny_kwh"),
        ("penalty", 2050, "unit_cost_cny_kwh"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.10, top=0.92, wspace=0.26, hspace=0.32
    )
    for ax, (family, year, metric) in zip(axes.flat, panels):
        names = (
            ["Reserve 0.8x", "Baseline", "Reserve 1.2x"]
            if family == "reserve"
            else ["Penalty 2000", "Baseline", "Penalty 5000"]
        )
        for name in names:
            rows = summary[
                (summary.Year == year)
                & (summary.metric_key == metric)
                & (summary.Setting == name)
            ].sort_values("NRMSE_pct")
            ax.fill_between(
                rows.NRMSE_pct,
                rows.Bootstrap_P5,
                rows.Bootstrap_P95,
                color=colors[name],
                alpha=0.14,
            )
            ax.plot(
                rows.NRMSE_pct,
                rows.Mean,
                "o-",
                ms=4,
                lw=2,
                color=colors[name],
                label=name,
            )
        title = (
            "Reserve multiplier: Qflex / load"
            if family == "reserve"
            else "Qflex penalty: unit operating cost"
        )
        ax.set(title=f"{title}, {year}", xlabel="NRMSE (%)", xlim=(3.3, 20.7))
        ax.set_xticks([4, 12, 20])
        ax.set_ylabel(
            "Qflex / load (%)"
            if family == "reserve"
            else "Unit operating cost (CNY/kWh)"
        )
        ax.grid(axis="y", alpha=0.28, lw=0.6)
        ax.legend(loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "Supplementary_Fig9.png", dpi=600, facecolor="white")
    plt.close(fig)
