"""Shared figure style: Times-family serif text, STIX math, embedded TrueType fonts.

Every figure script imports this module first. Sizes are in points and match the
CVPR two-column layout (column width 3.25 in, text width 6.875 in).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from pathlib import Path  # noqa: E402

# TeX Gyre Termes (a Times clone) converted to TrueType in scripts/fonts/, so PDFs embed clean
# Type 42 fonts. The system OpenType copies are dropped so matplotlib cannot pick them.
_FONTS = Path(__file__).resolve().parent / "fonts"
font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist
                                    if not (f.name == "TeX Gyre Termes" and f.fname.endswith(".otf"))]
for _f in sorted(_FONTS.glob("*.ttf")):
    font_manager.fontManager.addfont(str(_f))

COL = 3.25     # single column width (in)
TEXT = 6.875   # full text width (in)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
})

# One colour per visibility condition, used in every figure and in the text.
C = {
    "ball": "#2f5d8c",      # blue
    "band": "#c4511c",      # burnt orange
    "full": "#555555",      # grey
    "sources": "#1b8a7a",   # teal
    "blind": "#7a4fa3",     # attention heads share purple; rows are labelled
    "spatial": "#7a4fa3",   # purple
    "compact": "#3d3d3d",
    "never": "#4d4d4d",
    "overlap": "#c4511c",
    "null": "#9a9a9a",
}
LIGHT = {"ball": "#cfdcec", "band": "#f3d3bf", "full": "#e6e6e6", "sources": "#bfe3dc"}
SEED_MARK = {"3691": ("o", True), "9143": ("s", False)}   # marker, filled
SEED_NAME = {"3691": "Fit 1", "9143": "Fit 2"}


def save(fig, path):
    fig.savefig(path, metadata={"Creator": "scripts/ (aggregate renderer)", "CreationDate": None,
                                "ModDate": None})
    plt.close(fig)


def ci_bar(ax, y, est, lo, hi, color, marker="o", filled=True, ms=4.2, lw=1.3, cap=0.13, z=3):
    """Horizontal point estimate with a 95% interval and clearly visible end caps."""
    ax.hlines(y, lo, hi, color=color, lw=lw, zorder=z - 1)
    ax.vlines([lo, hi], y - cap, y + cap, color=color, lw=lw * 0.85, zorder=z - 1)
    ax.plot(est, y, marker=marker, ms=ms, mec=color, mfc=color if filled else "white",
            mew=1.0, ls="", zorder=z)


def zero_line(ax, vertical=True):
    if vertical:
        ax.axvline(0, color="#777777", lw=0.7, ls=(0, (3, 2)), zorder=1)
    else:
        ax.axhline(0, color="#777777", lw=0.7, ls=(0, (3, 2)), zorder=1)


def panel(ax, letter, title=None, x=-0.02, y=1.04):
    txt = f"({letter})" + (f" {title}" if title else "")
    ax.text(x, y, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, fontweight="bold")
