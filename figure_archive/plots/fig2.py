"""Draw the four independent panels of manuscript Fig. 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("MAE", "RMSE", "R2", "Corr")
GROUPS = (("wind", "us"), ("wind", "sm"), ("solar", "us"), ("solar", "sm"))


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


RED, BLACK = "#8B0000", "#222222"
GRAY = ("#5b6067", "#777d84", "#92979d", "#adb1b6", "#c8cbd0")
FIG2C_MIN_STATIONS = 5


def save(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=600)
    plt.close(fig)


def radar_label_positions(
    angle: np.ndarray, ours: np.ndarray, sota: np.ndarray
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    placed: list[np.ndarray] = []
    ours_positions = []
    sota_positions = []
    for index, theta in enumerate(angle):
        for positions, radius, outward in (
            (ours_positions, min(float(ours[index]) + 0.065, 1.15), True),
            (sota_positions, max(float(sota[index]) - 0.055, 0.02), False),
        ):
            if index == 8 and not outward:
                radius += 0.08
                theta += np.deg2rad(-1.5)
            trial_theta = float(theta)
            for attempt in range(40):
                point = np.array(
                    [radius * np.cos(trial_theta), radius * np.sin(trial_theta)]
                )
                if all(
                    np.linalg.norm(point - previous) >= 0.082 for previous in placed
                ):
                    break
                radius = (
                    min(radius + 0.018, 1.15) if outward else max(radius - 0.018, 0.02)
                )
                trial_theta += (1 if attempt % 2 == 0 else -1) * np.deg2rad(1)
            placed.append(point)
            positions.append((trial_theta, radius))
    return ours_positions, sota_positions


def panel_a(
    metrics: Path, figures: Path, radial_range: tuple[float, float] | None = None
) -> None:
    data = pd.read_csv(metrics / "fig2a_values.csv")
    ordered = data.set_index(["type", "group", "metric"])
    keys = [(kind, group, metric) for kind, group in GROUPS for metric in METRICS]
    if len(data) != 16 or not set(keys).issubset(ordered.index):
        raise ValueError("Fig. 2a needs all 16 task-metric results")
    rows = ordered.loc[keys]
    ours = rows.IresFM_SFT_improvement_percent.to_numpy(float)
    sota = rows.SOTA_improvement_percent.to_numpy(float)
    if not np.isfinite(ours).all() or not np.isfinite(sota).all():
        raise ValueError("Fig. 2a contains non-finite improvements")
    angle = 2 * np.pi * np.arange(16) / 16
    minimum = (
        radial_range[0]
        if radial_range is not None
        else min(float(np.min(ours)), float(np.min(sota)), 0.0)
    )
    maximum = (
        radial_range[1]
        if radial_range is not None
        else max(float(np.max(ours)), float(np.max(sota)), 1.0)
    )
    if min(ours.min(), sota.min()) < minimum or max(ours.max(), sota.max()) > maximum:
        raise ValueError("Fig. 2a values exceed the requested common radial range")
    span = maximum - minimum
    ours_r = (ours - minimum) / span
    sota_r = (sota - minimum) / span
    edge = {
        ("wind", "us"): "#1f77b4",
        ("wind", "sm"): "#ff7f0e",
        ("solar", "us"): "#2ca02c",
        ("solar", "sm"): "#9467bd",
    }
    red, blue = "#B22222", "#1F3A5F"
    fig, ax = plt.subplots(figsize=(7.2, 6), dpi=600, subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for theta in angle:
        ax.plot((theta, theta), (0, 1), color=".78", linewidth=1, zorder=1)
    closed = np.r_[angle, angle[0]]
    if minimum < 0:
        zero_r = -minimum / span
        ax.plot(
            closed,
            np.full(len(closed), zero_r),
            color=".45",
            linewidth=1,
            linestyle=":",
            zorder=2,
        )
    ax.plot(
        closed,
        np.r_[ours_r, ours_r[0]],
        color=red,
        linewidth=3.2,
        alpha=0.95,
        zorder=10,
    )
    ax.plot(
        closed, np.r_[sota_r, sota_r[0]], color=blue, linewidth=2, alpha=0.85, zorder=9
    )
    for index, ((kind, group, metric), theta) in enumerate(zip(keys, angle)):
        ax.text(
            theta,
            1.2,
            metric,
            ha="center",
            va="center",
            fontsize=12,
            bbox={
                "boxstyle": "square,pad=.34",
                "facecolor": "white",
                "edgecolor": edge[(kind, group)],
                "linewidth": 2,
            },
            clip_on=False,
            zorder=60,
        )
    ours_positions, sota_positions = radar_label_positions(angle, ours_r, sota_r)
    for values, positions, color in (
        (ours, ours_positions, red),
        (sota, sota_positions, blue),
    ):
        for value, (theta, radius) in zip(values, positions):
            ax.text(
                theta,
                radius,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
                zorder=50,
            )
    task_legend = ax.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor=edge[(kind, group)],
                linewidth=2,
                label=f"{kind.title()} ({group.upper()})",
            )
            for kind, group in GROUPS
        ],
        loc="lower left",
        bbox_to_anchor=(-0.28, -0.12),
        frameon=False,
        fontsize=12,
        handlelength=1,
        handletextpad=0.6,
        labelspacing=0.35,
    )
    ax.add_artist(task_legend)
    value_handles = [
        Line2D([0], [0], color=red, linewidth=3.2, label="IresFM-SFT"),
        Line2D([0], [0], color=blue, linewidth=2, label="SOTA (per-metric)"),
    ]
    if minimum < 0:
        value_handles.append(
            Line2D(
                [0],
                [0],
                color=".45",
                linewidth=1,
                linestyle=":",
                label="0% improvement",
            )
        )
    ax.legend(
        handles=value_handles,
        loc="lower right",
        bbox_to_anchor=(1.28, -0.12),
        frameon=False,
        fontsize=12,
        handlelength=2.2,
        borderaxespad=0,
    )
    fig.suptitle("Relative improvement over Persistence (%)", y=0.99, fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.12, top=0.86)
    save(fig, figures, "Fig2a")


def panel_b(
    metrics: Path,
    figures: Path,
    x_limits: dict[tuple[str, str], float] | None = None,
    allow_incomplete: bool = False,
) -> None:
    data = pd.read_csv(metrics / "fig2b_values.csv", dtype={"station_id": str})
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=600)
    gray = ("0.70", "0.62", "0.54", "0.46", "0.38")
    styles = ("-", "--", "-.", ":", (0, (5, 2)))
    plot_order = {
        ("wind", "us"): ("HBOLA", "DMOM", "LSSVM+RBFNN", "TFT", "SC-VAR"),
        ("wind", "sm"): ("HEFTCom24", "GEFCom12", "FFNN", "HPE", "CNN-RBFNN"),
        ("solar", "us"): ("RFs-ALO", "WPD-LSTM", "ATCN", "NARX-GA", "LSTM-GCN-MLP"),
        ("solar", "sm"): ("CNN-LSTM", "CrossViViT", "PVTransNet", "ANN", "FusionSF"),
    }
    for ax, (kind, group) in zip(axes.flat, GROUPS):
        subset = data[data.type.eq(kind) & data.group.eq(group)]
        site_count = 633 if kind == "wind" else 616
        methods = (*plot_order[(kind, group)], "persistence", "iresfm-sft")
        for index, method in enumerate(methods):
            series = subset[subset.method.eq(method)]
            if not allow_incomplete and series.station_id.nunique() != site_count:
                raise ValueError(
                    f"Fig. 2b {kind}/{group}/{method}: "
                    f"expected {site_count} sites, got {series.station_id.nunique()}"
                )
            values = series.RMSE.to_numpy(float)
            if allow_incomplete:
                values = values[np.isfinite(values)]
                if len(values) < 600:
                    raise ValueError(
                        f"Fig. 2b {kind}/{group}/{method}: too few finite scores"
                    )
            elif not np.isfinite(values).all():
                raise ValueError("Fig. 2b contains non-finite site RMSE")
            # Match the archived Fig. 2b: remove values above each curve's
            # 95th percentile before constructing its ECDF.
            x = np.sort(values[values <= np.quantile(values, 0.95)])
            y = np.arange(1, len(x) + 1) / len(x)
            line_x = np.repeat(x, 2)
            line_y = np.r_[0.0, np.repeat(y[:-1], 2), y[-1]]
            color = (
                RED
                if method == "iresfm-sft"
                else "0.00" if method == "persistence" else gray[index]
            )
            label = (
                "IresFM-SFT"
                if method == "iresfm-sft"
                else (
                    "Persistence"
                    if method == "persistence"
                    else "LSSVM-RBFNN" if method == "LSSVM+RBFNN" else method
                )
            )
            linestyle = (
                "--"
                if method == "persistence"
                else "-" if method == "iresfm-sft" else styles[index]
            )
            ax.plot(
                line_x,
                line_y,
                color=color,
                label=label,
                linewidth=(
                    2.0
                    if method == "iresfm-sft"
                    else 1.8 if method == "persistence" else 1.3
                ),
                linestyle=linestyle,
                zorder=(
                    4 if method == "iresfm-sft" else 3 if method == "persistence" else 2
                ),
            )
        ax.text(
            0.5,
            1.02,
            f"{kind.title()} ({group.upper()})",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            clip_on=False,
        )
        ax.set_xlabel("RMSE")
        ax.set_ylabel("ECDF")
        ax.set_ylim(0, 1)
        if x_limits is not None:
            ax.set_xlim(0, x_limits[(kind, group)])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.8)
        ax.margins(x=0.02)
        ax.tick_params(direction="out", length=3, width=0.8, colors="0.15")
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index("IresFM-SFT"), labels.index("Persistence")]
        order.extend(
            i
            for i, label in sorted(enumerate(labels), key=lambda item: item[1].lower())
            if label not in ("IresFM-SFT", "Persistence")
        )
        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            frameon=False,
            loc="lower right",
            handlelength=2.2,
            labelspacing=0.25,
            borderpad=0.15,
            columnspacing=0.8,
        )
    fig.suptitle("Site-wise RMSE distribution (ECDF)", y=0.99, fontsize=12)
    fig.subplots_adjust(
        left=0.1, right=0.95, bottom=0.1, top=0.9, hspace=0.35, wspace=0.2
    )
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "Fig2b.png", dpi=600)
    plt.close(fig)


def panel_c(
    metrics: Path,
    figures: Path,
    scenario_order: list[str] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    data = pd.read_csv(metrics / "fig2c_site_scenarios.csv", dtype={"station_id": str})
    required = {
        "scenario",
        "station_id",
        "climate_major",
        "season",
        "weather",
        "delta_rmse",
        "ours_sum_rmse",
        "best_sum_rmse",
    }
    if required - set(data) or data.empty:
        raise ValueError("Fig. 2c site-scenario metrics are missing")
    data = data[np.isfinite(data.delta_rmse)].copy()
    # Keep every finite site-scenario value; the minimum station count applies to all groups.
    if data.empty:
        raise ValueError("No finite Fig. 2c site-scenario values remain")
    season_abbr = {"Spring": "Sp", "Summer": "Su", "Autumn": "Au", "Winter": "Wi"}
    weather_abbr = {
        "HighWind": "HW",
        "LowWind": "LW",
        "HighVar": "HV",
        "TurningPoint": "TP",
        "LowTemp": "LT",
        "ColdWave": "CW",
        "SolarNoon": "SN",
    }
    climate_order = ("A", "B", "C", "D")
    groups = []
    for scenario, frame in data.groupby("scenario"):
        climate, season, weather = (
            str(frame.iloc[0][column])
            for column in ("climate_major", "season", "weather")
        )
        if climate not in climate_order:
            raise ValueError(f"Fig. 2c has an unsupported climate group: {climate}")
        if frame.station_id.nunique() < FIG2C_MIN_STATIONS:
            continue
        values = frame.delta_rmse.to_numpy(float)
        groups.append((climate, scenario, season, weather, values))
    groups.sort(
        key=lambda item: (
            climate_order.index(item[0]),
            -float(np.median(item[4])),
            item[1],
        )
    )
    if scenario_order is not None:
        by_name = {item[1]: item for item in groups}
        if set(by_name) != set(scenario_order):
            raise ValueError("Fig. 2c scenario sets differ between the two datasets")
        groups = [by_name[name] for name in scenario_order]
    if not groups:
        raise ValueError("No Fig. 2c scenarios meet the common sample threshold")
    positions = []
    block_centers = {}
    dividers = []
    x = 1.0
    for index, climate in enumerate(climate_order):
        count = sum(item[0] == climate for item in groups)
        if not count:
            continue
        start, end = x, x + count - 1
        positions.extend(np.arange(start, end + 1).tolist())
        block_centers[climate] = (start + end) / 2
        if index < len(climate_order) - 1:
            dividers.append(end + 1.0)
        x = end + 2.0
    labels = [
        f"{season_abbr.get(season, season)}-" f"{weather_abbr.get(weather, weather)}"
        for _, _, season, weather, _ in groups
    ]
    fig, ax = plt.subplots(figsize=(14, 3.6))
    boxes = ax.boxplot(
        [values for _, _, _, _, values in groups],
        positions=positions,
        tick_labels=labels,
        widths=0.55,
        showfliers=False,
        whis=(5, 95),
        patch_artist=True,
        boxprops={
            "facecolor": "#1F78B4",
            "edgecolor": "#0B2A3F",
            "linewidth": 1,
            "alpha": 0.85,
        },
        medianprops={"color": "#FF7F00", "linewidth": 1.6},
        whiskerprops={"color": "#0B2A3F", "linewidth": 1.05},
        capprops={"color": "#0B2A3F", "linewidth": 1.05},
    )
    gray = {"A": 0.80, "B": 0.65, "C": 0.50, "D": 0.35}
    base = np.array([31, 120, 180], dtype=float)
    for box, (climate, *_rest) in zip(boxes["boxes"], groups):
        shade = np.round(0.65 * base + 0.35 * 255 * gray[climate]) / 255
        box.set_facecolor(shade)
    for divider in dividers:
        ax.axvline(divider, color="#444444", linewidth=1, alpha=0.85, linestyle="--")
    ax.axhline(0, color="red", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_ylabel(r"$\Delta$RMSE (MW; SOTA $-$ IresFM-SFT)")
    ax.set_title(r"Scenario-wise $\Delta$RMSE distribution", y=1.06)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    lower, upper = ax.get_ylim()
    label_y = upper + 0.01 * (upper - lower)
    for climate, center in block_centers.items():
        ax.text(
            center,
            label_y,
            climate,
            ha="center",
            va="bottom",
            clip_on=False,
            fontsize=12,
        )
    ax.set_xticklabels(labels, rotation=90, ha="right")
    ax.tick_params(axis="x", pad=2)
    ax.yaxis.grid(True, color="#E8ECEF", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.9)
        ax.spines[side].set_color("#222222")
    ax.tick_params(axis="both", direction="out", length=3, width=0.8, colors="#111111")
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.18, top=0.88)
    save(fig, figures, "Fig2c")


def panel_d(
    metrics: Path,
    figures: Path,
    y_limits: dict[str, tuple[float, float]] | None = None,
    reference_method: str = "ERA5-informed reference",
) -> None:
    data = pd.read_csv(metrics / "fig2d_months.csv")
    required = {"panel", "training_month", "method", "rmse"}
    if required - set(data):
        raise ValueError("Fig. 2d monthly metrics are missing")
    months = (0, 1, 3, 6, 9, 12)
    methods = ("IresFM", reference_method, "Persistence")
    expected = {
        (f"{kind.title()}-{group.upper()}", month, method)
        for kind, group in GROUPS
        for method in methods
        for month in (months[1:] if method == methods[1] else months)
    }
    actual = set(zip(data.panel, data.training_month, data.method))
    if not expected.issubset(actual):
        raise ValueError(
            f"Fig. 2d is incomplete: {len(expected - actual)} monthly points absent"
        )
    if data.duplicated(["panel", "training_month", "method"]).any():
        raise ValueError("Fig. 2d contains duplicate monthly points")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    positions = np.arange(len(months))
    for ax, (kind, group) in zip(axes, GROUPS):
        panel = f"{kind.title()}-{group.upper()}"
        for method, color, style in (
            (methods[0], RED, "-"),
            (methods[1], "0.55", "-."),
            (methods[2], BLACK, "--"),
        ):
            subset = data[data.panel.eq(panel) & data.method.eq(method)].set_index(
                "training_month"
            )
            series = subset.rmse.reindex(months)
            y = series.to_numpy(float)
            if not np.isfinite(y[1:] if method == methods[1] else y).all():
                raise ValueError(
                    f"Fig. 2d {panel}/{method}: missing or non-finite RMSE"
                )
            ax.plot(
                positions,
                y,
                marker="o" if method != "Persistence" else None,
                markersize=3.2 if method == methods[0] else 2.8,
                color=color,
                linestyle=style,
                label=method,
                linewidth=2 if method == methods[0] else 1.6,
            )
        ax.set_title(f"{kind.title()} ({group.upper()})")
        ax.set_xticks(positions, [str(month) for month in months])
        ax.set_xlim(-0.25, len(months) - 1 + 0.05)
        ax.set_xlabel("Training Period (month)")
        ax.set_ylabel("RMSE")
        finite = np.concatenate(
            [
                data[data.panel.eq(panel) & data.method.eq(method)].rmse.to_numpy(float)
                for method in methods
            ]
        )
        finite = finite[np.isfinite(finite)]
        ymin, ymax = float(finite.min()), float(finite.max())
        pad = 0.07 * (ymax - ymin + 1e-12)
        lower, upper = ymin - pad, ymax + pad
        if np.isclose(lower, upper):
            lower -= 0.5
            upper += 0.5
        if y_limits is not None:
            lower, upper = y_limits[panel]
        ax.set_ylim(lower, upper)
        ax.set_yticks(np.linspace(lower, upper, 4))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.9)
            ax.spines[side].set_color("0.25")
        ax.tick_params(axis="both", direction="out", length=3, width=0.8, colors="0.25")
    axes[0].legend(
        frameon=False,
        fontsize=11,
        loc="upper left",
        handlelength=2.2,
        borderaxespad=2,
        labelspacing=0.3,
    )
    fig.suptitle("Training-period-wise RMSE comparison", y=0.97)
    fig.subplots_adjust(top=0.85, left=0.055, right=0.995, bottom=0.15, wspace=0.32)
    save(fig, figures, "Fig2d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("a", "b", "c", "d"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    globals()[f"panel_{args.panel}"](ROOT / "data/fig2", args.output)
    print(args.output / f"Fig2{args.panel}.png")


if __name__ == "__main__":
    main()
