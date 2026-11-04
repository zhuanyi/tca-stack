"""The repository-wide figure style: palette, layout, 1600 x 900 px PNG output."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.figure import Figure

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

DPI = 150
WIDTH_PX = 1600
HEIGHT_PX = 900
_GREY = "#595959"
_LIGHT_GREY = "#D9D9D9"
_LEFT = 0.08

_state = {"applied": False, "base": 11.0}


def figures_dir() -> Path:
    """Output folder: ``$TCA_FIGURES_DIR`` if set, else ``docs/figures`` at the repository root."""
    env = os.environ.get("TCA_FIGURES_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "docs" / "figures"


def apply(hero: bool = False) -> None:
    """Apply the repository style to matplotlib; ``hero`` raises the base font to 14 pt."""
    base = 14.0 if hero else 11.0
    matplotlib.use("Agg", force=True)
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.figsize": (WIDTH_PX / DPI, HEIGHT_PX / DPI),
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": base,
            "axes.labelsize": base,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "legend.fontsize": base - 1,
            "legend.frameon": False,
            "axes.prop_cycle": cycler(color=PALETTE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": _GREY,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": _LIGHT_GREY,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.0,
            "image.cmap": "viridis",
        }
    )
    _state.update(applied=True, base=base)


def figure(
    title: str, subtitle: str, source: str, nrows: int = 1, ncols: int = 1, **kw: Any
) -> tuple[Figure, Any]:
    """Create a 10.67 x 6 inch figure with title, grey subtitle and source line in place.

    Returns the figure and the axes (an array when there is more than one panel).
    """
    if not _state["applied"]:
        apply()
    base = _state["base"]
    fig, ax = plt.subplots(nrows, ncols, **kw)
    fig.subplots_adjust(left=_LEFT, right=0.97, top=0.80, bottom=0.15)
    fig.text(_LEFT, 0.945, title, fontsize=base + 5, fontweight="bold", ha="left", va="center")
    fig.text(_LEFT, 0.885, subtitle, fontsize=base, color=_GREY, ha="left", va="center")
    fig.text(_LEFT, 0.03, source, fontsize=base - 2, color=_GREY, ha="left", va="center")
    return fig, ax


def save(fig: Figure, name: str) -> Path:
    """Save ``fig`` as ``<figures_dir>/<name>.png`` at exactly 1600 x 900 px and close it.

    The PNG carries no software or time metadata; other keys may still be written by
    matplotlib depending on version, so byte-identical output is not guaranteed across
    environments.
    """
    path = figures_dir() / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
    fig.savefig(path, format="png", dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return path
