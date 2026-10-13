"""Figure 4 plotting functions used in the final manuscript."""

from itertools import pairwise

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import ScalarFormatter

from plots.uced.style import FONT
from plots.uced.style import register_font as register_arial

PERIODS = ((2025, 2030, 5), (2030, 2040, 10), (2040, 2050, 10), (2050, 2060, 10))
YEARS = (2025, 2030, 2040, 2050)


LEVEL_PCT = (4, 8, 12, 16, 20)


LEVELS = np.asarray(LEVEL_PCT, dtype=float) / 100.0


REGION_EN = {
    "north_china": "North China",
    "east_china": "East China",
    "central_china": "Central China",
    "northwest_china": "Northwest China",
    "southwest_china": "Southwest China",
    "northeast_china": "Northeast China",
    "south_china": "South China",
}


YEAR_COLORS = {
    2025: "#79bb56",
    2030: "#00aae3",
    2040: "#1f4e9f",
    2050: "#ec347c",
}


YEAR_MARKERS = {2025: "h", 2030: "d", 2040: "s", 2050: "v"}


PERIOD_COLORS = {
    "2025–2030": "#92d050",
    "2030–2040": "#00b0f0",
    "2040–2050": "#4472c4",
    "2050–2060": "#e867a7",
}


STAR_COLOR = "#d7191c"


TRIANGLE_COLOR = "#159414"


PF_COLOR = "#f6c445"


PF_LINE_COLOR = "#7f7f7f"


CURTAIL_LOWER_MAX = 18.0


CURTAIL_UPPER_MIN = 55.0


CURTAIL_UPPER_MAX = 460.0


CURTAIL_LOWER_HEIGHT = 0.43


CURTAIL_GAP_HEIGHT = 0.14


CURTAIL_UPPER_HEIGHT = 0.43


METRICS = {
    "load_shedding": {
        "column": "qflex_mwh",
        "top_title": "National Load Shedding",
        "bottom_title": "Regional Load Shedding Reduction",
        "top_unit": "MWh/day",
        "bottom_unit": "TWh",
        "top_scale": 1.0 / 365.0,
        "bottom_scale": 1.0e-6,
        "left_label": "Load Shedding Reduction by ResFM (TWh)",
        "right_label": "Load Shedding Reduction by Perfect Forecast (TWh)",
    },
    "curtailment": {
        "column": "wind_pv_curtailment_mwh",
        "top_title": "National Curtailment",
        "bottom_title": "Regional Curtailment Reduction",
        "top_unit": "TWh",
        "bottom_unit": "TWh",
        # The national panel is plotted as annual absolute curtailment, rather
        # than the PF-relative daily increment used by the other top panels.
        "top_scale": 1.0e-6,
        "bottom_scale": 1.0e-6,
        "left_label": "Curtailment Reduction by ResFM (TWh)",
        "right_label": "Curtailment Reduction by Perfect Forecast (TWh)",
    },
    "cost": {
        "column": "comparison_operating_cost_10k_cny",
        "top_title": "National Cost",
        "bottom_title": "Regional Cost Reduction",
        "top_unit": "million yuan/day",
        "bottom_unit": "billion yuan",
        "top_scale": 1.0 / (365.0 * 100.0),
        "bottom_scale": 1.0e-5,
        "left_label": "Cost Reduction by ResFM (billion yuan)",
        "right_label": "Cost Reduction by Perfect Forecast (billion yuan)",
    },
    "carbon": {
        "column": "accounted_system_emissions_t",
        "top_title": "National Carbon Emission Factor",
        "bottom_title": "Regional Emission Reduction",
        "top_unit": "g/kWh",
        "bottom_unit": "Mt",
        "bottom_scale": 1.0e-6,
        "left_label": "Emission Reduction by ResFM (Mt)",
        "right_label": "Emission Reduction by Perfect Forecast (Mt)",
    },
}


def compact(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f}m"
    if magnitude >= 10_000:
        return f"{value / 1_000:.1f}k"
    if magnitude >= 1_000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def style_axes(
    ax: plt.Axes, font_size: float = 11.0, *, secondary: bool = False
) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(secondary)
    ax.spines["left"].set_visible(not secondary)
    ax.spines["bottom"].set_visible(not secondary)
    ax.tick_params(
        axis="both", labelsize=font_size, width=0.8, length=3.0, direction="out"
    )
    ax.grid(False)
    ax.set_facecolor("white")


def add_side_decorations(ax: plt.Axes, *, bottom: bool = False) -> None:
    if bottom:
        ax.scatter(
            [-0.11],
            [0.91],
            transform=ax.transAxes,
            marker="*",
            s=95,
            color=STAR_COLOR,
            edgecolor="none",
            clip_on=False,
            zorder=20,
        )
        ax.text(
            -0.11,
            0.865,
            "-",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            clip_on=False,
        )
        ax.scatter(
            [-0.11],
            [0.82],
            transform=ax.transAxes,
            marker="<",
            s=55,
            color=TRIANGLE_COLOR,
            edgecolor="none",
            clip_on=False,
            zorder=20,
        )
        ax.scatter(
            [1.17],
            [0.95],
            transform=ax.transAxes,
            marker="o",
            s=80,
            color=PF_COLOR,
            edgecolor="none",
            clip_on=False,
            zorder=20,
        )
        ax.text(
            1.17,
            0.90,
            "-",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            clip_on=False,
        )
        ax.scatter(
            [1.17],
            [0.85],
            transform=ax.transAxes,
            marker="<",
            s=55,
            color=TRIANGLE_COLOR,
            edgecolor="none",
            clip_on=False,
            zorder=20,
        )
        return

    vertices = [
        (1.018, 0.34),
        (1.055, 0.34),
        (1.055, 0.27),
        (1.018, 0.22),
        (1.055, 0.17),
        (1.055, 0.06),
        (1.018, 0.06),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 6
    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="#666666",
            linewidth=0.8,
            clip_on=False,
            zorder=18,
        )
    )
    ax.scatter(
        [1.09],
        [0.24],
        transform=ax.transAxes,
        marker="o",
        s=95,
        color=PF_COLOR,
        edgecolor="none",
        clip_on=False,
        zorder=20,
    )
    ax.text(
        1.09,
        0.18,
        "-",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="black",
        clip_on=False,
    )
    ax.scatter(
        [1.09],
        [0.11],
        transform=ax.transAxes,
        marker="<",
        s=60,
        color=TRIANGLE_COLOR,
        edgecolor="none",
        clip_on=False,
        zorder=20,
    )


def apply_curtailment_broken_axis(ax: plt.Axes) -> None:
    """Apply the one-panel 0–18 / 55–460 TWh broken linear y-axis."""
    lower_max, upper_min, upper_max = (
        CURTAIL_LOWER_MAX,
        CURTAIL_UPPER_MIN,
        CURTAIL_UPPER_MAX,
    )
    lower_height, gap_height, upper_height = (
        CURTAIL_LOWER_HEIGHT,
        CURTAIL_GAP_HEIGHT,
        CURTAIL_UPPER_HEIGHT,
    )

    def forward(values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.where(
            values <= lower_max,
            values / lower_max * lower_height,
            lower_height
            + gap_height
            + (values - upper_min) / (upper_max - upper_min) * upper_height,
        )

    def inverse(values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.where(
            values <= lower_height,
            values / lower_height * lower_max,
            upper_min
            + (values - lower_height - gap_height)
            / upper_height
            * (upper_max - upper_min),
        )

    ax.set_yscale("function", functions=(forward, inverse))
    ax.set_ylim(0.0, upper_max)
    ax.set_yticks((0, 6, 12, 18, 55, 150, 250, 350, 460))
    ax.set_yticklabels(("0", "6", "12", "18", "55", "150", "250", "350", "460"))
    for y in (lower_height, lower_height + gap_height):
        ax.plot(
            (-0.014, 0.014),
            (y - 0.012, y + 0.012),
            transform=ax.transAxes,
            color="black",
            linewidth=0.85,
            clip_on=False,
            zorder=30,
        )


def plot_top(
    ax: plt.Axes,
    top: pd.DataFrame,
    metric: str,
    penetration: dict[int, float],
    positions: dict[str, float],
) -> list[dict[str, object]]:
    spec = METRICS[metric]
    cell = top.loc[top["metric"].eq(metric)]
    handles: list[Line2D] = []
    for year in YEARS:
        curve = cell.loc[cell["year"].eq(year)].sort_values("rmse")
        x = curve["rmse"].to_numpy(float)
        mean = curve["mean"].to_numpy(float)
        lower = curve["bootstrap_p05"].to_numpy(float)
        upper = curve["bootstrap_p95"].to_numpy(float)
        color = YEAR_COLORS[year]
        if metric == "curtailment":
            # Draw retained ranges separately so no line or band artist enters
            # the intentionally omitted 18–55 TWh interval.
            lower_mask = lower < CURTAIL_LOWER_MAX
            if np.any(lower_mask):
                ax.fill_between(
                    x,
                    np.minimum(lower, CURTAIL_LOWER_MAX),
                    np.minimum(upper, CURTAIL_LOWER_MAX),
                    where=lower_mask,
                    color=color,
                    alpha=0.15,
                    edgecolor="none",
                    linewidth=0,
                    zorder=1,
                )
            upper_mask = upper > CURTAIL_UPPER_MIN
            if np.any(upper_mask):
                ax.fill_between(
                    x,
                    np.minimum(np.maximum(lower, CURTAIL_UPPER_MIN), CURTAIL_UPPER_MAX),
                    np.minimum(np.maximum(upper, CURTAIL_UPPER_MIN), CURTAIL_UPPER_MAX),
                    where=upper_mask,
                    color=color,
                    alpha=0.15,
                    edgecolor="none",
                    linewidth=0,
                    zorder=1,
                )
            overflow_mask = upper > CURTAIL_UPPER_MAX
            if np.any(overflow_mask):
                for overflow_x, overflow_y in zip(
                    x[overflow_mask], upper[overflow_mask]
                ):
                    ax.scatter(
                        [overflow_x],
                        [CURTAIL_UPPER_MAX],
                        marker="^",
                        s=38,
                        facecolors="white",
                        edgecolors=color,
                        linewidths=0.9,
                        clip_on=False,
                        zorder=12,
                    )
                    ax.annotate(
                        f"P95 {overflow_y:.2f}",
                        (overflow_x, CURTAIL_UPPER_MAX),
                        xytext=(-3, 6),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        fontsize=10.0,
                        color=color,
                        clip_on=False,
                        zorder=12,
                    )
            for mask in (mean <= CURTAIL_LOWER_MAX, mean >= CURTAIL_UPPER_MIN):
                if np.any(mask):
                    ax.plot(
                        x[mask],
                        mean[mask],
                        color=color,
                        marker=YEAR_MARKERS[year],
                        markersize=4.1,
                        linewidth=2.0,
                        zorder=4,
                    )
        else:
            ax.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.15,
                edgecolor="none",
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                x,
                mean,
                color=color,
                marker=YEAR_MARKERS[year],
                markersize=4.1,
                linewidth=2.0,
                zorder=4,
            )
        label = (
            f"{year} (Penetration rate: {penetration[year]:.3f})"
            if year == YEARS[0]
            else f"{year} ({penetration[year]:.3f})"
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=YEAR_MARKERS[year],
                markersize=4.1,
                linewidth=2.0,
                label=label,
            )
        )

    if metric == "curtailment":
        apply_curtailment_broken_axis(ax)
    else:
        y_min = min(0.0, float(cell["bootstrap_p05"].min()))
        y_max = max(0.0, float(cell["bootstrap_p95"].max()))
        span = max(y_max - y_min, 1.0)
        ax.set_ylim(y_min - 0.06 * span, y_max + 0.10 * span)
    ax.set_xlim(-0.006, 0.206)
    threshold_rows: list[dict[str, object]] = []
    marker_specs = (
        ("IresFM", positions["IresFM"], STAR_COLOR, "*"),
        ("Second-best", positions["Second_best"], TRIANGLE_COLOR, "^"),
    )
    for index, (name, x_value, color, marker) in enumerate(marker_specs):
        choices = []
        for year in YEARS:
            curve = cell.loc[cell["year"].eq(year)].sort_values("rmse")
            y_value = float(np.interp(x_value, curve["rmse"], curve["mean"]))
            choices.append((y_value, year))
        y_value, source_year = next(pair for pair in choices if pair[1] == 2050)
        if metric == "curtailment" and y_value >= CURTAIL_UPPER_MIN:
            # Do not draw through the omitted 18–55 TWh interval.
            ax.vlines(
                x_value,
                0.0,
                CURTAIL_LOWER_MAX,
                colors=color,
                linestyles="--",
                linewidth=0.85,
                zorder=2,
            )
            ax.vlines(
                x_value,
                CURTAIL_UPPER_MIN,
                y_value,
                colors=color,
                linestyles="--",
                linewidth=0.85,
                zorder=2,
            )
        else:
            ax.vlines(
                x_value,
                ax.get_ylim()[0],
                y_value,
                colors=color,
                linestyles="--",
                linewidth=0.85,
                zorder=2,
            )
        ax.hlines(
            y_value,
            ax.get_xlim()[0],
            x_value,
            colors=color,
            linestyles="--",
            linewidth=0.85,
            zorder=2,
        )
        ax.scatter(
            [x_value],
            [y_value],
            marker=marker,
            s=85 if marker == "*" else 65,
            color=color,
            edgecolor="none",
            zorder=10,
        )
        dx, dy, horizontal, vertical = (
            (-22, 18, "right", "bottom"),
            (-3, 36, "right", "bottom"),
        )[index]
        offset = (dx, dy)
        ax.annotate(
            f"({x_value:.2f}, {compact(y_value)})",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=10.0,
            color="black",
            ha=horizontal,
            va=vertical,
            zorder=20,
        )
        threshold_rows.append(
            {
                "metric": metric,
                "marker": "red star" if index == 0 else "green triangle",
                "method": name,
                "rmse": x_value,
                "curve_year": source_year,
                "display_value": y_value,
                "unit": spec["top_unit"],
                "selection_rule": "2050 mean curve at the fixed capacity-weighted benchmark NRMSE",
            }
        )

    pf_y = float(cell.loc[cell["rmse"].eq(0.0), "mean"].iloc[0])
    ax.scatter(
        [0], [pf_y], marker="o", s=90, color=PF_COLOR, edgecolor="none", zorder=9
    )
    ax.axhline(pf_y, color=PF_COLOR, linestyle="--", linewidth=0.75, zorder=2)
    if metric == "curtailment":
        endpoint_labels = {
            2025: "+1.23",
            2030: "+14.97",
            2040: "+134.88",
            2050: "+95.48",
        }
        for year in YEARS:
            curve = cell.loc[cell["year"].eq(year)].sort_values("rmse")
            endpoint_y = float(curve.loc[curve["rmse"].eq(0.20), "mean"].iloc[0])
            ax.annotate(
                f"{endpoint_labels[year]} TWh",
                (0.20, endpoint_y),
                xytext=(-4, -10) if year == 2050 else (-4, 0),
                textcoords="offset points",
                ha="right",
                va="top" if year == 2050 else "center",
                fontsize=10.0,
                color=YEAR_COLORS[year],
                zorder=20,
            )
    ax.set_xticks(np.r_[0.0, LEVELS])
    ax.set_xticklabels([f"{value:.2f}" for value in np.r_[0.0, LEVELS]])
    ax.set_xlabel("RMSE", fontsize=12.0)
    ax.set_ylabel(str(spec["top_unit"]), fontsize=12.0, loc="top")
    if metric == "load_shedding":
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((6, 6))
        ax.yaxis.set_major_formatter(formatter)
    ax.set_title(str(spec["top_title"]), fontsize=12.0, pad=3)
    style_axes(ax, 11.0)
    return threshold_rows


def plot_unit_cost_top(
    ax: plt.Axes, unit_cost: pd.DataFrame, positions: dict[str, float]
) -> list[dict[str, object]]:
    """Unit operating cost and bootstrap intervals."""
    for year in YEARS:
        curve = unit_cost.loc[unit_cost["Year"].eq(year)].sort_values("NRMSE_pct")
        x = curve["NRMSE_pct"].to_numpy(float)
        y = curve["Unit_operating_cost_CNY_per_kWh"].to_numpy(float)
        color = YEAR_COLORS[year]
        ax.fill_between(
            x,
            curve["Bootstrap_P5"],
            curve["Bootstrap_P95"],
            color=color,
            alpha=0.15,
            edgecolor="none",
            linewidth=0,
            zorder=1,
        )
        ax.plot(x, y, color=color, linewidth=2.0, zorder=3)
        ax.scatter(
            x,
            y,
            marker=YEAR_MARKERS[year],
            s=27,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )

    y_min = float(
        min(
            unit_cost["Bootstrap_P5"].min(),
            unit_cost["Unit_operating_cost_CNY_per_kWh"].min(),
        )
    )
    y_max = float(
        max(
            unit_cost["Bootstrap_P95"].max(),
            unit_cost["Unit_operating_cost_CNY_per_kWh"].max(),
        )
    )
    y_pad = max(0.08 * (y_max - y_min), 0.002)
    ax.set(xlim=(-1.1, 21.1), ylim=(y_min - y_pad, y_max + y_pad))
    threshold_rows: list[dict[str, object]] = []
    curve_2050 = unit_cost.loc[unit_cost["Year"].eq(2050)].sort_values("NRMSE_pct")
    for index, (name, x_value, color, marker) in enumerate(
        (
            ("IresFM", positions["IresFM"], STAR_COLOR, "*"),
            ("Second-best", positions["Second_best"], TRIANGLE_COLOR, "^"),
        )
    ):
        x_percent = 100.0 * x_value
        y_value = float(
            np.interp(
                x_percent,
                curve_2050["NRMSE_pct"],
                curve_2050["Unit_operating_cost_CNY_per_kWh"],
            )
        )
        ax.vlines(
            x_percent,
            ax.get_ylim()[0],
            y_value,
            colors=color,
            linestyles="--",
            linewidth=0.85,
            zorder=2,
        )
        ax.hlines(
            y_value,
            ax.get_xlim()[0],
            x_percent,
            colors=color,
            linestyles="--",
            linewidth=0.85,
            zorder=2,
        )
        ax.scatter(
            [x_percent],
            [y_value],
            marker=marker,
            s=85 if marker == "*" else 65,
            color=color,
            edgecolor="none",
            zorder=10,
        )
        ax.annotate(
            f"({x_value:.2f}, {y_value:.3f})",
            (x_percent, y_value),
            xytext=((-14, -24), (14, 18))[index],
            textcoords="offset points",
            fontsize=10.0,
            color="black",
            ha=("right", "left")[index],
            va=("top", "bottom")[index],
            zorder=20,
        )
        threshold_rows.append(
            {
                "metric": "cost",
                "marker": "red star" if index == 0 else "green triangle",
                "method": name,
                "rmse": x_value,
                "curve_year": 2050,
                "display_value": y_value,
                "unit": "CNY/kWh",
                "selection_rule": "2050 unit-operating-cost curve at the fixed capacity-weighted benchmark NRMSE",
            }
        )
    ax.set_xticks((0, *LEVEL_PCT))
    ax.set_xlabel("NRMSE (%)", fontsize=12.0)
    ax.set_ylabel("CNY/kWh", fontsize=12.0, loc="top")
    ax.set_title("Unit Operating Cost", fontsize=12.0, pad=3)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.45, alpha=0.36)
    ax.grid(axis="x", visible=False)
    style_axes(ax, 11.0)
    return threshold_rows


def plot_curtailment_stacked_axes(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    top: pd.DataFrame,
    penetration: dict[int, float],
    positions: dict[str, float],
) -> list[dict[str, object]]:
    """National Curtailment as three linear, shared-x stacked axis segments."""
    # Cached curtailment is annual TWh; display the equivalent daily MWh.
    daily_scale = 1.0e6 / 365.0
    ranges = tuple(
        (lo * daily_scale, hi * daily_scale)
        for lo, hi in ((350.0, 470.0), (55.0, 220.0), (0.0, 18.0))
    )
    cell = top.loc[top["metric"].eq("curtailment")].copy()
    columns = ["mean", "bootstrap_p05", "bootstrap_p95"]
    cell[columns] = cell[columns] * daily_scale
    cell["unit"] = "MWh/day"
    for axis, ticks in zip(
        axes,
        (
            np.arange(1.0e6, 1.250001e6, 100000),
            [200000, 300000, 400000, 500000, 600000],
            [0, 25000],
        ),
    ):
        axis.set_yticks(ticks)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((6, 6))
        formatter.set_useOffset(False)
        axis.yaxis.set_major_formatter(formatter)
    for axis in axes[1:]:
        axis.yaxis.get_offset_text().set_visible(False)
    handles: list[Line2D] = []
    for axis, (lo, hi) in zip(axes, ranges):
        axis.set_ylim(lo, hi)
        axis.set_xlim(-0.006, 0.206)
        for year in YEARS:
            curve = cell.loc[cell["year"].eq(year)].sort_values("rmse")
            x = curve["rmse"].to_numpy(float)
            mean = curve["mean"].to_numpy(float)
            p05 = curve["bootstrap_p05"].to_numpy(float)
            p95 = curve["bootstrap_p95"].to_numpy(float)
            color = YEAR_COLORS[year]
            visible = (p95 >= lo) & (p05 <= hi)
            if np.any(visible):
                axis.fill_between(
                    x,
                    np.clip(p05, lo, hi),
                    np.clip(p95, lo, hi),
                    where=visible,
                    color=color,
                    alpha=0.15,
                    edgecolor="none",
                    linewidth=0,
                    zorder=1,
                )
            points = (mean >= lo) & (mean <= hi)
            if np.any(points):
                axis.plot(
                    x[points],
                    mean[points],
                    color=color,
                    marker=YEAR_MARKERS[year],
                    markersize=5.0 if lo == 0.0 else 4.1,
                    linewidth=2.0,
                    zorder=4,
                )
        style_axes(axis, 11.0)

    for year in YEARS:
        label = (
            f"{year} (Penetration rate: {penetration[year]:.3f})"
            if year == YEARS[0]
            else f"{year} ({penetration[year]:.3f})"
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=YEAR_COLORS[year],
                marker=YEAR_MARKERS[year],
                markersize=4.1,
                linewidth=2.0,
                label=label,
            )
        )
    # Draw aligned vertical guides through every retained segment.  Each guide
    # stops at its benchmark point instead of extending to the panel top.
    threshold_rows: list[dict[str, object]] = []
    for index, (name, x_value, color, marker) in enumerate(
        (
            ("IresFM", positions["IresFM"], STAR_COLOR, "*"),
            ("Second-best", positions["Second_best"], TRIANGLE_COLOR, "^"),
        )
    ):
        choices = []
        for year in YEARS:
            curve = cell.loc[cell["year"].eq(year)].sort_values("rmse")
            choices.append(
                (float(np.interp(x_value, curve["rmse"], curve["mean"])), year)
            )
        y_value, source_year = next(pair for pair in choices if pair[1] == 2050)
        for axis, (lo, hi) in zip(axes, ranges):
            guide_top = min(y_value, hi)
            if guide_top > lo:
                axis.vlines(
                    x_value,
                    lo,
                    guide_top,
                    colors=color,
                    linestyles="--",
                    linewidth=0.85,
                    zorder=2,
                )
            if lo <= y_value <= hi:
                axis.hlines(
                    y_value,
                    axis.get_xlim()[0],
                    x_value,
                    colors=color,
                    linestyles="--",
                    linewidth=0.85,
                    zorder=2,
                )
                axis.scatter(
                    [x_value],
                    [y_value],
                    marker=marker,
                    s=85 if marker == "*" else 65,
                    color=color,
                    edgecolor="none",
                    zorder=10,
                )
                axis.annotate(
                    f"({x_value:.2f}, {compact(y_value)})",
                    (x_value, y_value),
                    xytext=((-14, -24), (26, 30))[index],
                    textcoords="offset points",
                    fontsize=10.0,
                    ha=("right", "left")[index],
                    va=("top", "bottom")[index],
                    zorder=20,
                )
        threshold_rows.append(
            {
                "metric": "curtailment",
                "marker": "red star" if index == 0 else "green triangle",
                "method": name,
                "rmse": x_value,
                "curve_year": source_year,
                "display_value": y_value,
                "unit": "MWh/day",
                "selection_rule": "2050 mean curve at the fixed capacity-weighted benchmark NRMSE",
            }
        )

    # The P95 continuation is still disclosed when it exceeds the visible cap.
    curve = cell.loc[cell["year"].eq(2050)].sort_values("rmse")
    p95 = float(curve.loc[curve["rmse"].eq(0.20), "bootstrap_p95"].iloc[0])
    if p95 > ranges[0][1]:
        axes[0].scatter(
            [0.20],
            [ranges[0][1]],
            marker="^",
            s=38,
            facecolors="white",
            edgecolors=YEAR_COLORS[2050],
            linewidths=0.9,
            clip_on=False,
            zorder=12,
        )
        axes[0].annotate(
            f"P95 {p95:.2f}",
            (0.20, ranges[0][1]),
            xytext=(-3, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=10.0,
            color=YEAR_COLORS[2050],
            clip_on=False,
            zorder=12,
        )
    axes[0].set_title("National Curtailment", fontsize=12.0, pad=3)
    axes[0].set_ylabel("MWh/day", fontsize=12.0, loc="top")
    axes[-1].set_xticks(np.r_[0.0, LEVELS])
    axes[-1].set_xticklabels([f"{value:.2f}" for value in np.r_[0.0, LEVELS]])
    axes[-1].set_xlabel("RMSE", fontsize=12.0)
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
        axis.spines["bottom"].set_visible(False)
    for axis in axes[1:]:
        axis.spines["top"].set_visible(False)
    # Compact diagonal marks at both omitted y-intervals.  Their matching
    # endpoints also visually bind the three axes into one broken-axis panel.
    break_size = 0.010
    break_style = {"color": "black", "linewidth": 0.75, "clip_on": False, "zorder": 30}
    for upper_axis, lower_axis in pairwise(axes):
        upper_axis.plot(
            (-break_size, break_size),
            (-break_size, break_size),
            transform=upper_axis.transAxes,
            **break_style,
        )
        lower_axis.plot(
            (-break_size, break_size),
            (1 - break_size, 1 + break_size),
            transform=lower_axis.transAxes,
            **break_style,
        )
    return threshold_rows


def layered_left_limits(
    positive_top: np.ndarray, negative_bottom: np.ndarray
) -> tuple[float, float]:
    """Keep the stacked bars in the lower half of the shared plotting area."""
    top = max(float(np.nanmax(positive_top)), 1.0e-12)
    bottom = min(float(np.nanmin(negative_bottom)), 0.0)
    lower = min(1.15 * bottom, -0.04 * top)
    upper = lower + (top - lower) / 0.52
    return lower, upper


def layered_right_limits(values: np.ndarray) -> tuple[float, float]:
    """Map the complete PF line into the upper 60%–90% axes-height band."""
    lo_value = float(np.nanmin(values))
    hi_value = float(np.nanmax(values))
    if np.isclose(lo_value, hi_value):
        span = max(abs(lo_value), 1.0)
        return lo_value - 0.75 * span, lo_value + 0.25 * span
    span = (hi_value - lo_value) / (0.90 - 0.60)
    lower = lo_value - 0.60 * span
    return lower, lower + span


def plot_bottom(
    ax: plt.Axes,
    stacked: pd.DataFrame,
    pf_line: pd.DataFrame,
    metric: str,
) -> dict[str, object]:
    spec = METRICS[metric]
    cell = stacked.loc[stacked["metric"].eq(metric)]
    line_cell = pf_line.loc[pf_line["metric"].eq(metric)]
    totals = cell.groupby("region")["stacked_value"].sum()
    tie_break = {region: index for index, region in enumerate(REGION_EN)}
    regions = sorted(
        totals.index, key=lambda region: (float(totals[region]), tie_break[region])
    )
    x = np.arange(len(regions), dtype=float)
    positive = np.zeros(len(regions))
    negative = np.zeros(len(regions))
    patches: list[Patch] = []
    for period in [f"{start}–{end}" for start, end, _ in PERIODS]:
        period_cell = cell.loc[cell["period"].eq(period)].set_index("region")
        values = np.asarray(
            [period_cell.loc[region, "stacked_value"] for region in regions], float
        )
        bottoms = np.where(values >= 0.0, positive, negative)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.78,
            color=PERIOD_COLORS[period],
            edgecolor="none",
            zorder=3,
        )
        positive += np.where(values >= 0.0, values, 0.0)
        negative += np.where(values < 0.0, values, 0.0)
        patches.append(
            Patch(facecolor=PERIOD_COLORS[period], edgecolor="none", label=period)
        )

    total_values = np.asarray([totals[region] for region in regions], float)
    line_indexed = line_cell.set_index("region")
    line_values = np.asarray(
        [line_indexed.loc[region, "perfect_forecast_reduction"] for region in regions],
        float,
    )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        line_values,
        color=PF_LINE_COLOR,
        marker="o",
        markersize=3.5,
        linewidth=2.0,
        zorder=6,
    )
    ax.set_ylim(*layered_left_limits(positive, negative))
    ax2.set_ylim(*layered_right_limits(line_values))
    left_span = np.ptp(ax.get_ylim())
    for xpos, value, visual_top, visual_bottom in zip(
        x, total_values, positive, negative
    ):
        if visual_top > 0.0:
            anchor = visual_top + 0.018 * left_span
            vertical_alignment = "bottom"
        else:
            anchor = visual_bottom - 0.018 * left_span
            vertical_alignment = "top"
        ax.text(
            xpos,
            anchor,
            compact(value),
            ha="center",
            va=vertical_alignment,
            fontsize=10.0,
            zorder=20,
        )
    line_label_placements = {
        "load_shedding": (
            (-6, 12, "right", "bottom"),
            (6, -14, "left", "top"),
            (-6, 12, "right", "bottom"),
            (6, 12, "left", "bottom"),
            (-6, -14, "right", "top"),
            (-6, 12, "right", "bottom"),
            (-6, 12, "right", "bottom"),
        ),
        "curtailment": (
            (-6, -14, "right", "top"),
            (6, -14, "left", "top"),
            (-6, 12, "right", "bottom"),
            (6, -14, "left", "top"),
            (-6, 12, "right", "bottom"),
            (-6, 12, "right", "bottom"),
            (-6, -14, "right", "top"),
        ),
        "cost": (
            (-6, 12, "right", "bottom"),
            (6, -14, "left", "top"),
            (-6, 12, "right", "bottom"),
            (6, 12, "left", "bottom"),
            (-6, -14, "right", "top"),
            (-6, 12, "right", "bottom"),
            (-6, 12, "right", "bottom"),
        ),
    }[metric]
    for xpos, value, (dx, dy, horizontal, vertical) in zip(
        x, line_values, line_label_placements
    ):
        ax2.annotate(
            compact(value),
            (xpos, value),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10.0,
            color="#444444",
            ha=horizontal,
            va=vertical,
            zorder=20,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.65, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [REGION_EN[region] for region in regions],
        rotation=45,
        ha="right",
        fontsize=11.0,
    )
    ax.set_xlabel("Region", fontsize=12.0)
    ax.set_ylabel(str(spec["left_label"]), fontsize=12.0)
    ax2.set_ylabel(str(spec["right_label"]), fontsize=12.0)
    ax.set_title(str(spec["bottom_title"]), fontsize=12.0, pad=3)
    line_handle = Line2D(
        [0],
        [0],
        color=PF_LINE_COLOR,
        marker="o",
        markersize=3.5,
        linewidth=2.0,
        label="Reduction\nby Perfect Forecast",
    )
    # Matplotlib fills columns first: keep the four years in two rows,
    # followed immediately by the forecast entry in the same legend.
    if metric == "load_shedding":
        ax2.legend(
            handles=[patches[0], patches[1], line_handle, patches[2], patches[3]],
            loc="upper left",
            fontsize=11.0,
            frameon=False,
            ncol=2,
            columnspacing=0.8,
            labelspacing=0.25,
            handlelength=1.4,
        )
    style_axes(ax, 11.0)
    style_axes(ax2, 11.0, secondary=True)
    ax2.spines["left"].set_visible(False)
    left_lo, left_hi = ax.get_ylim()
    right_lo, right_hi = ax2.get_ylim()
    bar_top_fraction = float(np.max((positive - left_lo) / (left_hi - left_lo)))
    line_fractions = (line_values - right_lo) / (right_hi - right_lo)
    return {
        "metric": metric,
        "bar_visual_top_max_axes_fraction": bar_top_fraction,
        "pf_line_min_axes_fraction": float(np.min(line_fractions)),
        "pf_line_max_axes_fraction": float(np.max(line_fractions)),
        "minimum_vertical_separation_fraction": float(
            np.min(line_fractions) - bar_top_fraction
        ),
        "left_ylim_lower": float(left_lo),
        "left_ylim_upper": float(left_hi),
        "right_ylim_lower": float(right_lo),
        "right_ylim_upper": float(right_hi),
    }


def render(
    top: pd.DataFrame,
    stacked: pd.DataFrame,
    pf_line: pd.DataFrame,
    penetration: dict[int, float],
    positions: dict[str, float],
    unit_cost: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    register_arial()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT,
            "mathtext.it": f"{FONT}:italic",
            "mathtext.bf": f"{FONT}:bold",
            "mathtext.cal": f"{FONT}:italic",
            "mathtext.sf": FONT,
            "mathtext.tt": FONT,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(20.4, 12.6), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.80, 1.02),
        left=0.060,
        right=0.810,
        top=0.950,
        bottom=0.100,
        hspace=0.16,
    )
    top_grid = outer[0].subgridspec(1, 3, wspace=0.26)
    bottom_grid = outer[1].subgridspec(1, 3, wspace=0.38)
    threshold_rows: list[dict[str, object]] = []
    layout_rows: list[dict[str, object]] = []
    for column, metric in enumerate(("load_shedding", "curtailment", "cost")):
        bottom_ax = fig.add_subplot(bottom_grid[0, column])
        if metric == "curtailment":
            curtail_grid = top_grid[0, column].subgridspec(
                3, 1, height_ratios=(120, 165, 18), hspace=0.0
            )
            curtail_axes = tuple(
                fig.add_subplot(curtail_grid[row, 0]) for row in range(3)
            )
            curtail_axes[1].sharex(curtail_axes[0])
            curtail_axes[2].sharex(curtail_axes[0])
            threshold_rows.extend(
                plot_curtailment_stacked_axes(curtail_axes, top, penetration, positions)
            )
        else:
            top_ax = fig.add_subplot(top_grid[0, column])
            if metric == "cost":
                threshold_rows.extend(plot_unit_cost_top(top_ax, unit_cost, positions))
            else:
                first_top_ax = top_ax
                threshold_rows.extend(
                    plot_top(top_ax, top, metric, penetration, positions)
                )
        layout_rows.append(plot_bottom(bottom_ax, stacked, pf_line, metric))

    shared_year_handles = [
        Line2D(
            [0],
            [0],
            color=YEAR_COLORS[year],
            marker=YEAR_MARKERS[year],
            markersize=4.6,
            linewidth=2.0,
            label=f"{year} ({penetration[year]:.3f})",
        )
        for year in YEARS
    ]
    first_top_ax.legend(
        handles=shared_year_handles,
        title="Year (penetration rate)",
        loc="upper left",
        fontsize=11.0,
        title_fontsize=11.0,
        frameon=False,
        labelspacing=0.50,
        handlelength=2.4,
        borderaxespad=0.6,
    )

    # Draw data labels above the curves without a background box.
    from matplotlib.text import Annotation

    for axis in fig.axes:
        for label in axis.texts:
            if isinstance(label, Annotation):
                label.set_zorder(40)

    return fig, pd.DataFrame(threshold_rows), pd.DataFrame(layout_rows)
