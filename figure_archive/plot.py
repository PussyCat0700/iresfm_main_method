"""Generate individual manuscript, extended-data, and supplementary figures."""

import argparse
from pathlib import Path

from plots import extended_data, fig2, fig3, fig4, supplementary

ROOT = Path(__file__).resolve().parent
FIGURES = (
    *(f"fig2{panel}" for panel in "abcd"),
    *(f"fig3{panel}" for panel in "abcdef"),
    "fig4",
    "ed1",
    "ed3",
    *(f"si{number}" for number in range(2, 10)),
)


def draw(name: str, output: Path) -> None:
    if name.startswith("fig2"):
        getattr(fig2, f"panel_{name[-1]}")(ROOT / "data/fig2", output)
    elif name.startswith("fig3"):
        fig3.plot(name[-1], output)
    elif name == "fig4":
        fig4.plot(output)
    elif name == "ed1":
        extended_data.fig1(output)
    elif name == "ed3":
        extended_data.fig3(output)
    elif name.startswith("si"):
        supplementary.plot(int(name[2:]), output)
    else:
        raise ValueError(f"Unknown figure: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=FIGURES,
        action="append",
        help="Select one or more figures; omit to draw all",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    for name in args.figure or FIGURES:
        draw(name, args.output)
        print(f"Done: {name}")


if __name__ == "__main__":
    main()
