"""Shared settings for the studies: output folder, mesh sequence, plot style, table writer."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                 # write files only, never open windows
import matplotlib.pyplot as plt       # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Nested mesh sequence: each mesh halves the element size of the previous one
# (element aspect ratio a/b = (L/nx)/(D/ny) = 4.44 throughout). the validation table in README.md
# goes to 192 x 64; 384 x 128 is one extra level for the convergence study.
MESHES = [(3, 1), (6, 2), (12, 4), (24, 8), (48, 16), (96, 32), (192, 64), (384, 128)]

# Colours: validated categorical slots 1-3 (blue, orange, aqua) and neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def setup_style():
    plt.rcParams.update({
        "figure.figsize": (6.4, 4.0), "figure.dpi": 100, "savefig.dpi": 200,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0, "lines.markersize": 7,
        "legend.frameon": False, "legend.labelcolor": INK2,
    })


def save_figure(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def write_table(name, lines):
    """Print the table and write it to figures/<name>.txt (ASCII, so any console can print it)."""
    FIGURES.mkdir(exist_ok=True)
    text = "\n".join(lines) + "\n"
    (FIGURES / f"{name}.txt").write_text(text, encoding="utf-8")
    print(text)
