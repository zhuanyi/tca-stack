"""The Module 1 figures, drawn from the TCA tables in ``data/derived``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from tca.core import storage
from tca.core.stats import bootstrap_ci
from tca.plotting import common, style
from tca.ust import costs, regime, scorecard
from tca.ust.config import SimConfig
from tca.ust.summary import SIZE_FIT_BINS

SOURCE = "Simulated RFQ data calibrated to NY Fed liquidity statistics, Mar\u2013May 2025"
_PERIOD = "Mar 3 to May 30 2025, includes the April sell-off"
_OFF_THE_RUN = ("OLD", "DOUBLE_OLD")
_HERO_TENORS = (2, 5, 10, 30)
_N_LABELS = ["1", "2", "3", "4-5", "6+"]
_MIN_BUCKET_FILLS = 20
_MIN_GROUP_FILLS = 30
_POOL_TENORS = (2, 5, 10)


def _fills(derived: Path) -> pd.DataFrame:
    fills = storage.read_table("ust_tca_fill", derived)
    return fills.assign(off_the_run=fills["run_status"].isin(_OFF_THE_RUN))


def _off_run_ratios(fills: pd.DataFrame) -> list[float]:
    """Median cost of the off-the-runs over the on-the-run median, for each hero tenor."""
    med = fills.groupby(["tenor", "off_the_run"])["cost_32nds"].median()
    return [float(med[(t, True)] / med[(t, False)]) for t in _HERO_TENORS]


def cost_by_tenor(derived: Path) -> Path:
    """``ust_01``: cost against the arrival mid by tenor, on-the-run beside off-the-run."""
    fills = _fills(derived)
    ratios = _off_run_ratios(fills)
    style.apply(hero=True)
    fig, ax = style.figure(
        f"Off-the-runs cost {min(ratios):.1f} to {max(ratios):.1f}x on-the-runs to trade by RFQ",
        "Cost vs arrival mid, 32nds; whiskers 1.5 x IQR. Mar 3 to May 30 2025, incl. April",
        SOURCE,
    )
    groups, colours = {}, []
    for tenor in _HERO_TENORS:
        for off, name, colour in (
            (False, "OTR", style.PALETTE[0]),
            (True, "Off-run", style.PALETTE[1]),
        ):
            sel = fills[(fills["tenor"] == tenor) & (fills["off_the_run"] == off)]
            groups[f"{tenor}y {name}"] = sel["cost_32nds"].to_numpy()
            colours.append(colour)
    common.boxplot_with_n(ax, groups)
    for patch, colour in zip(ax.patches, colours, strict=True):
        patch.set_facecolor(colour)
    ax.set_ylabel("Cost vs arrival mid (32nds)")
    ax.legend(
        handles=[
            Patch(color=style.PALETTE[0], alpha=0.55, label="On-the-run"),
            Patch(color=style.PALETTE[1], alpha=0.55, label="Off-the-run (old, double-old)"),
        ],
        loc="upper left",
    )
    return style.save(fig, "ust_01_cost_by_tenor")


def weekly_10y(cfg: SimConfig, derived: Path) -> pd.DataFrame:
    """Weekly distributions of 10y on-the-run fill cost and parent IS (see ``regime.weekly``)."""
    return regime.weekly_otr10(
        cfg,
        storage.read_table("ust_tca_fill", derived),
        storage.read_table("ust_daily_anchor", derived),
        storage.read_table("ust_tca_parent", derived),
    )


def regime_figure(cfg: SimConfig, derived: Path) -> Path:
    """``ust_04``: weekly 10y on-the-run cost range and the simulator regime ratio."""
    table = weekly_10y(cfg, derived)
    style.apply()
    fig, ax = style.figure(
        "April widens 10y on-the-run cost outcomes; the median moves little",
        "10y on-the-run RFQ fills, weeks starting Monday: median cost, p10 to p90 range, "
        "and the volatility ratio r",
        SOURCE,
    )
    x = np.arange(len(table))
    median = table["cost_median"].to_numpy()
    ax.fill_between(
        x,
        table["cost_p10"],
        table["cost_p90"],
        color=style.PALETTE[1],
        alpha=0.3,
        linewidth=0,
        label="Cost p10 to p90 range",
    )
    ax.bar(x, median, color=style.PALETTE[2], alpha=0.6, width=0.7, label="Median cost, 95% CI")
    ax.errorbar(
        x,
        median,
        yerr=[median - table["cost_median_lo"], table["cost_median_hi"] - median],
        fmt="none",
        ecolor="#595959",
        capsize=3,
    )
    ax.axhline(0.0, color="#595959", linewidth=0.8)
    ax.set_ylabel("Cost vs arrival mid (32nds)")
    ax.set_xticks(x, [f"{w:%b %d}" for w in table["week"]], rotation=45, ha="right")
    twin = ax.twinx()
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    twin.plot(
        x,
        table["mean_r"],
        color=style.PALETTE[0],
        marker="o",
        label="Volatility ratio r (right axis)",
    )
    twin.set_ylabel("r = 5-day realised variance / period median")
    twin.set_ylim(0, None)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper left")
    fig.subplots_adjust(right=0.91)
    return style.save(fig, "ust_04_regime")


def cost_vs_size(derived: Path) -> Path:
    """``ust_02``: 10y cost against size, on-the-run beside old, with the fitted curve."""
    fills = storage.read_table("ust_tca_fill", derived)
    fills = fills[fills["tenor"] == 10]
    style.apply()
    fig, axes = style.figure(
        "10y cost rises with size, but the fitted exponent is poorly pinned down",
        f"Filled RFQs, cost vs arrival mid in 32nds, size in $mm face. {_PERIOD}",
        SOURCE,
        ncols=2,
        sharey=True,
    )
    for ax, status, title in zip(
        axes, ("OTR", "OLD"), ("On-the-run", "First off-the-run"), strict=True
    ):
        sel = fills[fills["run_status"] == status]
        ax.scatter(sel["qty_mm"], sel["cost_32nds"], s=8, color="#BFBFBF", alpha=0.6, linewidths=0)
        common.cost_vs_size(ax, sel["qty_mm"], sel["cost_32nds"], n_bins=SIZE_FIT_BINS)
        ax.set_title(f"{title} (n={len(sel):,})", loc="left", fontsize=11)
        ax.set_xlabel("Size ($mm face, log scale)")
    axes[0].set_ylabel("Cost vs arrival mid (32nds)")
    return style.save(fig, "ust_02_cost_vs_size")


def intraday_profile(derived: Path) -> Path:
    """``ust_03``: median cost and median composite half-spread by 15-minute bucket of NY time."""
    fills = storage.read_table("ust_tca_fill", derived)
    by = fills.groupby("bucket_15m").agg(
        n=("cost_32nds", "size"),
        cost=("cost_32nds", "median"),
        half_spread=("hs_send_32nds", "median"),
    )
    by = by[by["n"] >= _MIN_BUCKET_FILLS]
    style.apply()
    fig, ax = style.figure(
        "Cost runs above the composite half-spread through the day",
        f"Median over filled RFQs by 15-minute bucket of New York time. {_PERIOD}",
        SOURCE,
    )
    for start, label in (
        (8 * 60 + 30, "08:30 data"),
        (13 * 60, "13:00 auctions"),
        (15 * 60, "15:00 close"),
    ):
        ax.axvspan(start, start + 15, color="#D9D9D9", alpha=0.7, linewidth=0)
        ax.text(
            start + 7.5,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="top",
            fontsize=9,
            color="#595959",
        )
    ax.plot(by.index + 7.5, by["cost"], color=style.PALETTE[0], marker="o", label="Median cost")
    ax.plot(
        by.index + 7.5,
        by["half_spread"],
        color=style.PALETTE[1],
        marker="s",
        label="Median composite half-spread",
    )
    ticks = np.arange(7 * 60, 17 * 60 + 1, 60)
    ax.set_xticks(ticks, [f"{t // 60:02d}:00" for t in ticks])
    ax.set_xlabel("New York time (start of 15-minute bucket)")
    ax.set_ylabel("32nds")
    ax.legend(loc="lower left")
    return style.save(fig, "ust_03_intraday_profile")


def dealer_scorecard(cfg: SimConfig, derived: Path) -> Path:
    """``ust_05``: dealers by response, hit rate, rank, margin and 5-minute markout on wins."""
    card = scorecard.dealer_overall(storage.read_table("ust_dealer_scorecard", derived))
    card.columns = [
        "Response\nrate",
        "Hit\nrate",
        "Avg\nrank",
        "Margin vs\ncover (32nds)",
        "Markout 5m\non wins (32nds)",
    ]
    style.apply()
    fig, ax = style.figure(
        "Dealers differ in how often they respond and how often they win",
        f"Colour is the column z-score, cells show raw values. 12 dealers. {_PERIOD}",
        SOURCE,
    )
    common.heatmap_with_values(ax, card, fmt=".2f")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    return style.save(fig, "ust_05_dealer_scorecard")


def markout_by_n(cfg: SimConfig, derived: Path) -> Path:
    """``ust_06``: markout curves by number of dealers, 10y on-the-run tickets of $10mm to $50mm."""
    fills = storage.read_table("ust_tca_fill", derived)
    sel = fills[
        (fills["tenor"] == 10)
        & (fills["run_status"] == "OTR")
        & (fills["qty_mm"] >= 10)
        & (fills["qty_mm"] <= 50)
    ].copy()
    sel["n_group"] = pd.cut(sel["n_dealers"], [0, 1, 2, 3, 5, 99], labels=_N_LABELS)
    labels = [costs.horizon_label(h) for h in cfg.tca.markout_horizons_s]
    rng = np.random.default_rng(cfg.seed)
    style.apply()
    fig, ax = style.figure(
        "Markouts after the fill, by number of dealers asked",
        "10y on-the-run, \\$10mm to \\$50mm. Positive markout = market moved in the trader's "
        "favour after the fill",
        SOURCE,
    )
    x = np.arange(len(labels))
    shown = []
    for colour, (name, g) in zip(
        style.PALETTE, sel.groupby("n_group", observed=True), strict=False
    ):
        if len(g) < _MIN_GROUP_FILLS:
            continue
        mean, lo, hi = [], [], []
        for h in labels:
            v = g[f"markout_32nds_{h}"].dropna().to_numpy()
            ci = bootstrap_ci(v, rng=rng)
            mean.append(v.mean())
            lo.append(ci[0])
            hi.append(ci[1])
        common.markout_curve(ax, x, mean, lo, hi, f"N = {name} (n={len(g):,})", colour)
        shown.append(str(name))
    ax.set_xticks(x, labels)
    ax.set_xlabel("Horizon after fill")
    ax.set_ylabel("Mean markout (32nds)")
    ax.legend(loc="best")
    ax.text(
        0.99,
        0.02,
        f"Plotted: N in {', '.join(shown)}; other N have too few fills",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color="#595959",
    )
    return style.save(fig, "ust_06_markout_by_n_dealers")


def units(derived: Path) -> Path:
    """``ust_07``: the same on-the-run fills as price bp, 32nds, yield bp and DV01 dollars."""
    fills = storage.read_table("ust_tca_fill", derived)
    fills = fills[fills["run_status"] == "OTR"]
    panels = (
        ("cost_px_bp", "Price (bp of price)"),
        ("cost_32nds", "32nds of a point"),
        ("cost_yld_bp", "Yield (bp)"),
        ("cost_dv01_usd", "DV01 dollars per fill"),
    )
    style.apply()
    fig, axes = style.figure(
        "One set of fills, four ways to quote the same cost",
        f"On-the-run RFQ fills by tenor, cost vs arrival mid. {_PERIOD}",
        SOURCE,
        nrows=2,
        ncols=2,
    )
    for ax, (column, title) in zip(axes.ravel(), panels, strict=True):
        groups = {f"{t}y": fills.loc[fills["tenor"] == t, column].to_numpy() for t in _HERO_TENORS}
        common.boxplot_with_n(ax, groups, annotate_median=False)
        ax.set_title(title, loc="left", fontsize=11)
    fig.subplots_adjust(hspace=0.75, wspace=0.25)
    return style.save(fig, "ust_07_units")


def cost_vs_size_regime(cfg: SimConfig, derived: Path) -> Path:
    """``ust_08``: cost per composite half-spread against size, calm weeks beside high-vol weeks.

    Pools the on-the-run 2y, 5y and 10y because the 10y alone has too few fills in the
    high-volatility weeks.
    """
    weekly = weekly_10y(cfg, derived)
    hot = regime.high_vol_weeks(weekly)
    pool = regime.on_the_run_pool(storage.read_table("ust_tca_fill", derived), _POOL_TENORS)
    is_hot = pool["week"].isin(hot)
    names = ", ".join(f"{w:%b} {w.day}" for w in hot)
    style.apply()
    fig, axes = style.figure(
        "The high-volatility cost gap does not widen systematically with size",
        "On-the-run 2y, 5y and 10y RFQ fills, cost in composite half-spreads at send, "
        "size in \\$mm face. Mar 3 to May 30 2025",
        SOURCE,
        ncols=2,
        sharey=True,
    )
    for ax, sel, title in (
        (axes[0], pool[~is_hot], "Calm weeks"),
        (axes[1], pool[is_hot], f"High-volatility weeks starting {names}"),
    ):
        ax.scatter(sel["qty_mm"], sel["cost_per_hs"], s=8, color="#BFBFBF", alpha=0.6, linewidths=0)
        common.cost_vs_size(ax, sel["qty_mm"], sel["cost_per_hs"], n_bins=6)
        ax.set_title(f"{title} (n={len(sel):,})", loc="left", fontsize=10)
        ax.set_xlabel("Size ($mm face, log scale)")
    axes[0].set_ylabel("Cost / composite half-spread")
    return style.save(fig, "ust_08_cost_vs_size_regime")


def draw_all(cfg: SimConfig, derived: Path) -> list[Path]:
    """Draw every Module 1 figure and return their paths."""
    return [
        cost_by_tenor(derived),
        cost_vs_size(derived),
        intraday_profile(derived),
        regime_figure(cfg, derived),
        dealer_scorecard(cfg, derived),
        markout_by_n(cfg, derived),
        units(derived),
        cost_vs_size_regime(cfg, derived),
    ]
