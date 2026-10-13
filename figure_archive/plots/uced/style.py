"""Use installed Arial when available; otherwise use Matplotlib's bundled font."""

from pathlib import Path

from matplotlib import font_manager


def register_font():
    directory = Path.home() / ".local/share/fonts/arial"
    for filename in ("Arial.TTF", "Arialbd.TTF", "Ariali.TTF", "Arialbi.TTF"):
        path = directory / filename
        if path.is_file():
            font_manager.fontManager.addfont(path)


register_font()
try:
    font_manager.findfont("Arial", fallback_to_default=False)
    FONT = "Arial"
except ValueError:
    FONT = "DejaVu Sans"
