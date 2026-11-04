"""Chart helpers shared by every module: boxplots with counts, binned means, fits, heatmaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca.core.stats import bootstrap_ci, fit_power_law
from tca.plotting.style import PALETTE


def boxplot_with_n(ax, groups: dict[str, np.ndarray], annotate_median: bool = True) -> None:
    """Draw one box per group with ``n`` under its label and optionally the median above the box."""
    labels = list(groups)
    data = [np.asarray(groups[k], dtype=float) for k in labels]
    boxes = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for i, patch in enumerate(boxes["boxes"]):
        patch.set(facecolor=PALETTE[i % len(PALETTE)], alpha=0.55, edgecolor="#595959")
    ax.set_xticks(
        range(1, len(labels) + 1), [f"{k}\nn={len(v):,}" for k, v in zip(labels, data, strict=True)]
    )
    if annotate_median:
        for i, values in enumerate(data, start=1):
            top = boxes["whiskers"][2 * (i - 1) + 1].get_ydata()[1]
            ax.annotate(
                f"{np.median(values):.2f}",
                (i, top),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )


def binned_means(x, y, bins, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Mean of ``y`` in bins of ``x`` with 95% bootstrap intervals.

    ``bins`` is a bin count (equal-count quantile bins) or an array of edges. Columns: ``bin``,
    ``x_mid`` (median of ``x`` in the bin), ``mean``, ``lo``, ``hi``, ``n``.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
    frame = frame.dropna(subset=["x", "y"])
    if np.ndim(bins) == 0:
        frame["bin"] = pd.qcut(frame["x"], int(bins), labels=False, duplicates="drop")
    else:
        frame["bin"] = pd.cut(frame["x"], np.asarray(bins), labels=False, include_lowest=True)
    rows = []
    for b, g in frame.dropna(subset=["bin"]).groupby("bin", sort=True):
        lo, hi = bootstrap_ci(g["y"].to_numpy(), rng=rng) if len(g) > 1 else (g["y"].iloc[0],) * 2
        rows.append((int(b), g["x"].median(), g["y"].mean(), lo, hi, len(g)))
    return pd.DataFrame(rows, columns=["bin", "x_mid", "mean", "lo", "hi", "n"])


def cost_vs_size(
    ax, size, cost, n_bins: int = 10, rng: np.random.Generator | None = None
) -> tuple[float, float, float]:
    """Binned mean cost against size on a log x-axis with the fitted ``a + b * size**c`` overlaid.

    Draws means with 95% intervals, prints the fitted exponent ``c`` on the axes and returns
    ``(a, b, c)``.
    """
    size = np.asarray(size, dtype=float)
    cost = np.asarray(cost, dtype=float)
    table = binned_means(size, cost, n_bins, rng)
    ax.errorbar(
        table["x_mid"],
        table["mean"],
        yerr=[table["mean"] - table["lo"], table["hi"] - table["mean"]],
        fmt="o",
        color=PALETTE[0],
        capsize=3,
        label="Bin mean, 95% CI",
    )
    a, b, c = fit_power_law(size, cost)
    lo, hi = float(size.min()), float(size.max())
    if lo <= 0 or hi <= 0 or lo >= hi:
        raise ValueError(
            f"cost_vs_size requires strictly positive sizes with min < max; got [{lo}, {hi}]"
        )
    grid = np.geomspace(lo, hi, 200)
    ax.plot(grid, a + b * grid**c, color=PALETTE[3], label=f"Fit a + b·size^c, c = {c:.2f}")
    ax.set_xscale("log")
    ax.text(0.03, 0.95, f"exponent c = {c:.2f}", transform=ax.transAxes, va="top")
    ax.legend(loc="lower right")
    return a, b, c


def markout_curve(ax, horizons_s, mean, lo, hi, label: str, color: str) -> None:
    """Plot a mean markout against horizon with a shaded interval and a zero reference line."""
    ax.axhline(0.0, color="#595959", linewidth=0.8)
    ax.fill_between(horizons_s, lo, hi, color=color, alpha=0.2, linewidth=0)
    ax.plot(horizons_s, mean, color=color, marker="o", label=label)


def heatmap_with_values(
    ax, frame: pd.DataFrame, standardise_columns: bool = True, fmt: str = ".2f"
) -> None:
    """Heatmap of ``frame`` annotated with its raw values.

    With ``standardise_columns`` each column is z-scored for the colour so columns in different
    units are comparable; the printed numbers are always the raw values.
    """
    values = frame.to_numpy(dtype=float)
    shade = values
    if standardise_columns:
        sd = values.std(axis=0)
        shade = (values - values.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    ax.grid(False)
    ax.imshow(shade, aspect="auto", cmap="viridis")
    ax.set_xticks(range(frame.shape[1]), [str(c) for c in frame.columns])
    ax.set_yticks(range(frame.shape[0]), [str(i) for i in frame.index])
    if np.all(np.isnan(shade)):
        lo = hi = 0.0
    else:
        lo, hi = np.nanmin(shade), np.nanmax(shade)
    for (i, j), v in np.ndenumerate(values):
        dark = hi > lo and (shade[i, j] - lo) / (hi - lo) < 0.5
        ax.text(j, i, format(v, fmt), ha="center", va="center", color="white" if dark else "black")
