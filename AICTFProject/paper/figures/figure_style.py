"""Shared publication style for every CCP-S2 paper figure.

Single source of truth for typography, column widths, color/marker encoding, and export
format, so every figure in paper/generated/ shares one visual language and none of it needs
touching by hand afterward. Every build_fig*.py script must call apply_style() before creating
any matplotlib Figure, and save_figure() to export it.

Design targets (IEEE conference graphics guidance):
  - Times New Roman throughout (fails hard, not a silent fallback to DejaVu Sans, if the font
    is not actually installed -- a plot that "looks right" on this machine but silently
    degrades on the machine that builds the camera-ready PDF is worse than a loud crash now).
  - Text sized for FINAL printed size, not shrunk after the fact: ~9-9.5pt body, 8pt floor.
  - Canonical column widths: one-column 3.5in, two-column 7.16in.
  - Vector PDF is the citable artifact; PNG @ 600dpi is a convenience preview only.
  - Color is never the only encoding -- every semantic color also gets a distinct marker
    and/or linestyle, so the figure survives grayscale printing and CVD readers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated"

# ----------------------------------------------------------------------- column widths (in)
ONE_COLUMN = 3.5
TWO_COLUMN = 7.16

# ----------------------------------------------------------------------------- type sizes (pt)
FONT_SIZES = {
    "axis_label": 9.5,
    "tick_label": 9.0,
    "legend": 9.0,
    "annotation": 9.0,
    "panel_label": 9.0,          # bold, e.g. "(a)"
    "floor": 8.0,                 # nothing in a figure may go below this
}

# --------------------------------------------------------------- semantic colors (CVD-safe)
# Wong (2011) colorblind-safe palette. Meaning is fixed across every figure in the paper --
# "A" and "B" must never mean something different from one figure to the next.
COLORS = {
    "A": "#0072B2",         # blue    -- Pole A / specialist A / pi_A
    "B": "#D55E00",         # vermillion -- Pole B / specialist B / pi_B
    "control": "#666666",   # neutral gray -- incumbent / control arm
    "zero": "#000000",      # reference line at Delta = 0
}
# redundant, non-color encoding for the same semantics -- required so the figure still reads
# correctly in grayscale or for a colorblind reader relying on shape/line alone
MARKERS = {"A": "o", "B": "s", "control": "D"}
LINESTYLES = {"A": "-", "B": "--", "control": ":"}


class FigureStyleError(RuntimeError):
    pass


def _require_times_new_roman() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" not in names:
        raise FigureStyleError(
            "REFUSING: Times New Roman is not installed/discoverable by matplotlib on this "
            "machine. Silently falling back to DejaVu Sans would produce a figure that looks "
            "fine here and wrong on the machine that builds the camera-ready PDF. Install the "
            "font (or run this on a machine that has it) before generating any figure.")


def apply_style() -> None:
    """Call once, before creating any Figure. Idempotent."""
    _require_times_new_roman()
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": FONT_SIZES["tick_label"],
        "axes.labelsize": FONT_SIZES["axis_label"],
        "axes.titlesize": FONT_SIZES["axis_label"],
        "xtick.labelsize": FONT_SIZES["tick_label"],
        "ytick.labelsize": FONT_SIZES["tick_label"],
        "legend.fontsize": FONT_SIZES["legend"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,        # embed as real, editable/selectable text -- not curves
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,      # native mathtext, so pdf.fonttype=42 embedding still applies
    })


def save_figure(fig, name: str) -> dict:
    """Write name.pdf (canonical) and name.png (600dpi preview) to paper/generated/.

    Returns the paths written, for the caller to fold into a provenance manifest.
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = GENERATED_DIR / f"{name}.pdf"
    png_path = GENERATED_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    return {"pdf": str(pdf_path), "png": str(png_path)}
