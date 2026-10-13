"""Render the final compact manuscript Fig. 4 from its six metric tables."""

from io import BytesIO
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plots.uced import fig4 as chart
from plots.uced import fig4_layout

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fig4"


def plot(output: Path) -> None:
    files = {
        key: DATA / f"{key}.csv"
        for key in (
            "national_curves",
            "regional_reductions",
            "regional_pf_reference",
            "penetration",
            "benchmark_positions",
            "unit_cost_curves",
        )
    }
    tables = {key: pd.read_csv(path) for key, path in files.items()}
    penetration = dict(
        zip(tables["penetration"].year, tables["penetration"].renewable_penetration)
    )
    positions = dict(
        zip(tables["benchmark_positions"].method, tables["benchmark_positions"].rmse)
    )
    positions["Second_best"] = positions["SOTA"]
    figure, _, _ = chart.render(
        tables["national_curves"],
        tables["regional_reductions"],
        tables["regional_pf_reference"],
        penetration,
        positions,
        tables["unit_cost_curves"],
    )
    fig4_layout.prepare(figure, chart, penetration)

    raw = BytesIO()
    figure.savefig(raw, format="pdf", facecolor="white")
    plt.close(figure)
    raw.seek(0)
    document = fitz.open(stream=raw.read(), filetype="pdf")
    page = document[0]
    # The final manuscript uses the archived 24 pt top and 7 pt bottom crop.
    page.set_cropbox(fitz.Rect(0, 24, page.rect.width, page.rect.height - 7))
    output.mkdir(parents=True, exist_ok=True)
    # Match the final archived PNG canvas, 4251 × 2859 pixels at 600 dpi.
    clip = fitz.Rect(0, 0, page.rect.width - 0.12, page.rect.height - 0.12)
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(600 / 72, 600 / 72), clip=clip, alpha=False
    )
    pixmap.save(output / "Fig4.png")
    document.close()
