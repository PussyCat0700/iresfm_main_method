"""Draw Supplementary Figures 2–9 from the archived plotting tables."""

from pathlib import Path

from plots import supplementary2
from plots.uced import representative_days, selected_day, sensitivity

ROOT = Path(__file__).resolve().parents[1]


def plot(number: int, output: Path) -> None:
    if number not in range(2, 10):
        raise ValueError("Available supplementary figures are 2 through 9")
    output.mkdir(parents=True, exist_ok=True)
    if number == 2:
        supplementary2.plot(output)
    elif number == 3:
        representative_days.OUTPUT_DIR = output
        representative_days.configure_style()
        frame, context = representative_days.load_series(2050, "national")
        representative_days.render(frame, context, "Supplementary_Fig3")
    elif number in range(4, 9):
        selected_day.OUT = output
        selected_day.TARGET = f"Supplementary_Fig{number}"
        try:
            selected_day.make_figures(selected_day.replay_national())
        finally:
            selected_day.TARGET = None
    else:
        sensitivity.render(ROOT / "data/supplementary/fig9_sensitivity.csv", output)
