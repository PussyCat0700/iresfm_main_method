"""Print-size layout for Figure 4."""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Annotation, Text
from matplotlib.ticker import FuncFormatter

from plots.uced.style import FONT

WIDTH, HEIGHT = 180.0, 132.0
LEFTS, PANEL_WIDTH = (12.0, 71.0, 130.0), 40.0


def box(x, y, w, h):
    return (x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT)


def clamp_labels(fig, axes):
    """Move complete label boxes inside their axes, rather than clipping text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    for ax in axes:
        boundary = ax.get_window_extent(renderer).padded(-1.8 * px_per_pt)
        for label in ax.texts:
            if not label.get_visible():
                continue
            bounds = label.get_window_extent(renderer)
            dx = max(boundary.x0 - bounds.x0, 0) + min(boundary.x1 - bounds.x1, 0)
            dy = max(boundary.y0 - bounds.y0, 0) + min(boundary.y1 - bounds.y1, 0)
            if isinstance(label, Annotation):
                x, y = label.get_position()
                label.set_position((x + dx / px_per_pt, y + dy / px_per_pt))
            elif dx or dy:
                x, y = label.get_transform().transform(label.get_position())
                label.set_position(
                    label.get_transform().inverted().transform((x + dx, y + dy))
                )
    fig.canvas.draw()


def prepare(fig, module, penetration):
    positions = {ax: ax.get_position().frozen() for ax in fig.axes}
    data_axes = [ax for ax in fig.axes if ax.axison]
    fig.set_size_inches(WIDTH / 25.4, HEIGHT / 25.4, forward=True)
    fig.set_dpi(180)

    lower = [ax for ax in data_axes if positions[ax].y0 < 0.2]
    lower_x = sorted({round(positions[ax].x0, 9) for ax in lower})
    upper = [ax for ax in data_axes if ax not in lower]
    upper_x = sorted({round(positions[ax].x0, 9) for ax in upper})
    # Size strokes and symbols in points for the actual print-size canvas.
    for ax in data_axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for line in ax.lines:
            line.set_linewidth(max(0.45, line.get_linewidth() * 0.57))
            line.set_markersize(line.get_markersize() * 0.57)
        for collection in ax.collections:
            collection.set_linewidths(
                np.maximum(0.35, collection.get_linewidths() * 0.57)
            )
            if hasattr(collection, "get_sizes"):
                collection.set_sizes(collection.get_sizes() * 0.27)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("#333333")
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=6.5, length=2.0, width=0.5, pad=2)
        ax.xaxis.label.set_fontsize(7)
        ax.yaxis.label.set_fontsize(7)
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3
        ax.title.set_text("")
        for label in ax.texts:
            label.set_fontsize(6.3)
            if isinstance(label, Annotation):
                x, y = label.get_position()
                label.set_position((0.5 * x, 0.5 * y))

    for ax in upper:
        column = upper_x.index(round(positions[ax].x0, 9))
        if column == 1:
            segments = sorted(
                [a for a in upper if round(positions[a].x0, 9) == upper_x[1]],
                key=lambda a: positions[a].y0,
            )
            segment = segments.index(ax)
            y, h = ((72.0, 5.0), (77.5, 19.5), (97.5, 14.5))[segment]
            ax.set_position(box(LEFTS[column], y, PANEL_WIDTH, h))
            if segment > 0:
                ax.tick_params(axis="x", bottom=False)
        else:
            ax.set_position(box(LEFTS[column], 72, PANEL_WIDTH, 40))
        ax.yaxis.get_offset_text().set_visible(False)
        if ax.get_ylabel() == "MWh/day":
            ax.set_ylabel(r"$10^6$ MWh/day", fontsize=7, loc="center", labelpad=3)
        elif column == 2:
            ax.set_ylabel("CNY/kWh", fontsize=7, loc="center", labelpad=3)
        for label in ax.texts:
            if not isinstance(label, Annotation):
                continue
            second = label.get_text().startswith("(0.14,")
            offsets = {
                0: ((-5, 10), (-3, 22)),
                1: ((-5, -15), (-3, 8)),
                2: ((-5, -16), (5, 9)),
            }
            label.set_position(offsets[column][int(second)])
            label.set_ha("right" if column in (0, 1) or not second else "left")
            label.set_va(
                "bottom"
                if (column == 1 and second) or column == 0 or (column == 2 and second)
                else "top"
            )

    for ax in lower:
        column = lower_x.index(round(positions[ax].x0, 9))
        secondary = ax.yaxis.get_label_position() == "right"
        ax.set_position(box(LEFTS[column], 17, PANEL_WIDTH, 35))
        unit = "billion yuan" if column == 2 else "TWh"
        axis_label = "SOTA − PF" if secondary else "SOTA − IresFM-SFT"
        ax.set_ylabel(f"{axis_label} ({unit})", fontsize=7, labelpad=1)
        if not secondary:
            regions = [
                label.get_text().replace(" China", "") for label in ax.get_xticklabels()
            ]
            ax.set_xticks(
                ax.get_xticks(), regions, rotation=40, ha="right", fontsize=6.5
            )
            ax.set_xlabel("Region (China)", fontsize=7)
            for i, label in enumerate(ax.texts):
                label.set_fontsize(6.3)
                # Separate adjacent low bars whose four-digit labels nearly touch.
                if column == 1 and i == 1:
                    x, y = label.get_position()
                    label.set_position((x, y + np.ptp(ax.get_ylim()) * 0.034))
        else:
            # A common multiplier shortens repeated four-/five-digit ticks.
            # Axis limits and every plotted value stay in the original units.
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value / 1000:g}")
            )
            fig.text(
                (LEFTS[column] + PANEL_WIDTH + 1.0) / WIDTH,
                52.8 / HEIGHT,
                r"$\times10^3$",
                fontsize=6.2,
                color="#555555",
                va="bottom",
            )
            ax.tick_params(axis="y", colors="#555555")
            ax.yaxis.label.set_color("#555555")
            ax.spines["right"].set_color("#777777")
            for label in ax.texts:
                x, _ = label.xy
                label.set_position((0, 4.0))
                label.set_ha("center")
                label.set_va("bottom")
                if int(x) in (1, 4):
                    label.set_position((0, -5.0))
                    label.set_va("top")

    # Shared legends free the plotting area and remove the right-hand gutter.
    method_handles = [
        Line2D(
            [],
            [],
            color=module.STAR_COLOR,
            marker="*",
            linestyle="None",
            markersize=5.0,
            label="IresFM-SFT",
        ),
        Line2D(
            [],
            [],
            color=module.TRIANGLE_COLOR,
            marker="^",
            linestyle="None",
            markersize=3.7,
            label="SOTA reference",
        ),
    ]
    year_handles = [
        Line2D(
            [],
            [],
            color=module.YEAR_COLORS[year],
            marker=module.YEAR_MARKERS[year],
            lw=1.15,
            markersize=2.8,
            label=f"{year} ({penetration[year]:.3f})",
        )
        for year in module.YEARS
    ]
    # Keep every top-row key on one shared, full-width legend line.
    # Bring the shared key closer to the upper-panel titles while retaining a
    # clear separation from them.
    fig.text(
        12 / WIDTH, 121.0 / HEIGHT, "Year (penetration rate)", fontsize=6.5, va="center"
    )
    fig.legend(
        handles=[*year_handles, *method_handles],
        ncol=6,
        loc="center",
        bbox_to_anchor=(0.57, 121.0 / HEIGHT),
        fontsize=6.5,
        frameon=False,
        handlelength=1.35,
        handletextpad=0.4,
        columnspacing=1.45,
        borderaxespad=0,
    )

    period_handles = [
        Patch(facecolor=module.PERIOD_COLORS[p], label=p) for p in module.PERIOD_COLORS
    ]
    period_handles.append(
        Line2D(
            [],
            [],
            color=module.PF_LINE_COLOR,
            marker="o",
            lw=1.15,
            markersize=2.2,
            label="SOTA − PF (right axis)",
        )
    )
    fig.legend(
        handles=period_handles,
        ncol=5,
        loc="center",
        bbox_to_anchor=(0.51, 63 / HEIGHT),
        fontsize=6.5,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.8,
    )

    titles = (
        "National flexible-resource\nenergy requirement",
        "National curtailment",
        "Unit operating cost",
        "Regional reduction in flexible-resource\nenergy requirement",
        "Regional curtailment reduction",
        "Regional cost reduction",
    )
    for i, title in enumerate(titles):
        column = i % 3
        y = 114 if i < 3 else 55
        fig.text(
            (LEFTS[column] - 4.0) / WIDTH,
            y / HEIGHT,
            "abcdef"[i],
            fontsize=8,
            weight="bold",
            va="bottom",
        )
        display_title = title
        fig.text(
            LEFTS[column] / WIDTH,
            y / HEIGHT,
            display_title,
            fontsize=7,
            va="bottom",
            ha="left",
        )
    for text in fig.findobj(match=Text):
        text.set_fontfamily(FONT)

    # Match all national-panel normalized-error axes to the percentage display.
    for ax in upper:
        column = upper_x.index(round(positions[ax].x0, 9))
        if ax.get_xlabel():
            ax.set_xlabel("nRMSE (%)", fontsize=7)
        if column in (0, 1):
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{100 * value:g}")
            )
            for label in ax.texts:
                if isinstance(label, Annotation) and label.get_text().startswith("("):
                    x, _ = label.xy
                    old_text = label.get_text()
                    label.set_text(f"({100 * x:.2f}{old_text[old_text.find(chr(44)):]}")
        if column == 2:
            for label in ax.texts:
                if isinstance(label, Annotation) and label.get_text().startswith("("):
                    x, y = label.xy
                    label.set_text(f"({x:.2f}, {y:.3f})")
    clamp_labels(fig, data_axes)
