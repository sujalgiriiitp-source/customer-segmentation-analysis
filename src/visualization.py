"""
visualization.py
================
Reusable plotting helpers: consistent styling across all charts so every
visualization looks like part of one cohesive executive report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#6A994E"]
VIZ_DIR = Path("visualizations")


def set_style() -> None:
    """Apply a clean, executive-friendly style globally."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })
    sns.set_palette(PALETTE)


def save_fig(name: str, viz_dir: Path = VIZ_DIR) -> Path:
    """Save the current matplotlib figure to /visualizations and close it."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    path = viz_dir / f"{name}.png"
    plt.savefig(path)
    plt.close()
    return path
