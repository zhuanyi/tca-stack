"""The seven figures of the FX module, drawn from the stored tables of the active profile."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from tca.core.profile import is_ci
from tca.core.stats import bootstrap_ci
from tca.fx import analysis, paths
from tca.fx.config import FxConfig
from tca.fx.liquidity import ALL_WEEKDAYS
from tca.plotting import common, style

STYLES = ["TWAP", "POV", "IS", "RISK_TRANSFER"]
STYLE_LABELS = {
    "TWAP": "TWAP",
    "POV": "POV",
    "IS": "IS",
    "RISK_TRANSFER": "Risk transfer",
    "FIX_ORDER": "Fix order",
}
SIZE_EDGES_USD = [0.0, 10e6, 50e6, 150e6, np.inf]
SIZE_LABELS = ["<10mm", "10-50mm", "50-150mm", ">150mm"]
SESSIONS_UTC = {"Tokyo": (0, 9), "London": (7, 16), "New York": (12, 21)}
MARKOUT_SIGN = "Markouts are positive when the market moved in the trader's favour after the fill."
HORIZON_LABELS = {0: "0", 1: "1s", 5: "5s", 30: "30s", 60: "1m", 300: "5m", 900: "15m"}
HORIZON_LABELS |= {1800: "30m", 3600: "60m"}


def _subtitle(cfg: FxConfig, what: str, pairs: bool = False) -> str:
    """One-line subtitle: ``what`` followed by the period (and the pairs when ``pairs``)."""
    if is_ci():
        return f"{what}. Synthetic ticks, ci smoke test"
    start, end = cfg.period.start, cfg.period.end
    period = f"{start.day} {start:%b} to {end.day} {end:%b %Y}"
    return f"{what}. {', '.join(cfg.pairs.order) + ', ' if pairs else ''}{period}"


def _sign_note(fig) -> None:
    """The markout sign convention, in grey between the subtitle and the plot."""
    fig.text(0.08, 0.835, MARKOUT_SIGN, fontsize=9, color="#595959", ha="left", va="center")


def _source() -> str:
    if is_ci():
        return "Synthetic ticks and simulated orders; not market data. Smoke test only."
    return (
        "Dukascopy tick data (single venue), simulated orders and replay engine, BIS volume proxy."
    )


def _mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    lo, hi = bootstrap_ci(values, rng=rng) if len(values) > 1 else (values[0], values[0])
    return float(values.mean()), float(lo), float(hi)


def size_bucket(notional_usd: pd.Series) -> pd.Series:
    """Size bucket label of each notional (USD)."""
    return pd.cut(notional_usd, SIZE_EDGES_USD, labels=SIZE_LABELS, right=False)


def fig_is_by_style_size(cfg: FxConfig, parents: pd.DataFrame) -> None:
    """Figure 1: mean implementation shortfall by style and size bucket, 95% CI."""
    style.apply(hero=True)
    rng = np.random.default_rng(1)
    fig, ax = style.figure(
        "Implementation shortfall by execution style and order size",
        _subtitle(cfg, "Mean IS in bp, 95% CI; positive is worse", pairs=True),
        _source(),
    )
    data = parents[parents["style"].isin(STYLES)].assign(
        bucket=lambda d: size_bucket(d["notional_usd"])
    )
    width = 0.8 / len(SIZE_LABELS)
    ceiling = 0.0
    for k, bucket in enumerate(SIZE_LABELS):
        stats = [
            _mean_ci(
                data.loc[(data["style"] == s) & (data["bucket"] == bucket), "is_bp"].to_numpy(), rng
            )
            for s in STYLES
        ]
        mean, lo, hi = (np.array(col) for col in zip(*stats, strict=True))
        ceiling = max(ceiling, float(np.nanmax(hi)))
        x = np.arange(len(STYLES)) + (k - (len(SIZE_LABELS) - 1) / 2) * width
        ax.bar(x, mean, width, color=style.PALETTE[k], label=bucket)
        ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="black", capsize=2)
    ax.set_xticks(range(len(STYLES)), [STYLE_LABELS[s] for s in STYLES])
    ax.axhline(0.0, color="#595959", linewidth=0.8)
    ax.set_ylabel("IS (bp)")
    ax.set_ylim(top=ceiling * 1.3)
    ax.legend(title="Order size", ncols=4, loc="upper left")
    style.save(fig, "fx_01_is_by_style_size")


def _hour_spread(profile: pd.DataFrame) -> pd.DataFrame:
    """Median over the twelve buckets of each UTC hour of the median spread in bp, pair x hour."""
    overall = profile[profile["dow"] == ALL_WEEKDAYS].copy()
    overall["hour"] = overall["bucket"] // 12
    return overall.groupby(["pair", "hour"])["spread_median_bp"].median().unstack("hour")


def fig_liquidity_heatmap(cfg: FxConfig, profile: pd.DataFrame) -> None:
    """Figure 2: median quoted spread by pair and UTC hour with the sessions and the 4pm fix."""
    style.apply()
    fig, ax = style.figure(
        "Quoted spread by pair and hour of day",
        _subtitle(cfg, "Median quoted spread in bp of mid by UTC hour"),
        _source(),
    )
    fig.subplots_adjust(top=0.74)
    table = _hour_spread(profile).reindex(cfg.pairs.order)
    ax.grid(False)
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(24), [f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(len(table)), list(table.index))
    ax.set_xlabel("UTC hour; orange box: 4pm London fix window")
    for (i, j), v in np.ndenumerate(table.to_numpy()):
        dark = (v - np.nanmin(table.to_numpy())) / (np.ptp(table.to_numpy()) or 1.0) < 0.5
        ax.text(
            j,
            i,
            f"{v:.1f}",
            ha="center",
            va="center",
            fontsize=6.5,
            color="white" if dark else "black",
        )
    wmr = profile[(profile["dow"] == ALL_WEEKDAYS) & profile["wmr_window"]]["bucket"] // 12
    if len(wmr):
        ax.add_patch(
            Rectangle(
                (wmr.min() - 0.5, -0.5),
                wmr.max() - wmr.min() + 1,
                len(table),
                fill=False,
                edgecolor="#D55E00",
                linewidth=2.5,
            )
        )
    for k, (name, (start, end)) in enumerate(SESSIONS_UTC.items()):
        y = -0.62 - 0.17 * k
        ax.plot(
            [start - 0.5, end - 0.5], [y, y], color=style.PALETTE[k], linewidth=3, clip_on=False
        )
        ax.text(start - 0.4, y - 0.03, name, color=style.PALETTE[k], fontsize=8, va="bottom")
    fig.colorbar(image, ax=ax, label="spread (bp)", pad=0.01)
    style.save(fig, "fx_02_liquidity_heatmap")


def fig_is_decomposition(cfg: FxConfig, parents: pd.DataFrame) -> None:
    """Figure 3: stacked mean contributions to ``is_bp`` by style; they sum to the black marker."""
    style.apply()
    fig, ax = style.figure(
        "Where the shortfall comes from",
        _subtitle(cfg, "Mean contribution to IS in bp; filled terms times filled share"),
        _source(),
    )
    x_share = parents["filled_frac"]
    terms = {
        "Delay": x_share * parents["delay_bp"],
        "Spread": x_share * parents["spread_bp"],
        "Impact": x_share * parents["impact_bp"],
        "Timing": x_share * parents["timing_bp"],
        "Opportunity": parents["opportunity_bp"],
    }
    frame = pd.DataFrame(terms).assign(style=parents["style"].to_numpy())
    mean = frame.groupby("style").mean().reindex([*STYLES, "FIX_ORDER"]).dropna()
    positive, negative = np.zeros(len(mean)), np.zeros(len(mean))
    for k, term in enumerate(mean.columns):
        values = mean[term].to_numpy()
        base = np.where(values >= 0, positive, negative)
        ax.bar(range(len(mean)), values, bottom=base, color=style.PALETTE[k], label=term, width=0.6)
        positive += np.where(values >= 0, values, 0.0)
        negative += np.where(values < 0, values, 0.0)
    total = frame.groupby("style")[list(terms)].sum().sum(axis=1).reindex(mean.index)
    total = total / frame.groupby("style").size().reindex(mean.index)
    ax.scatter(range(len(mean)), total, color="black", zorder=5, label="Total IS")
    ax.set_xticks(range(len(mean)), [STYLE_LABELS[s] for s in mean.index])
    ax.axhline(0.0, color="#595959", linewidth=0.8)
    ax.set_ylabel("bp")
    low, high = min(negative.min(), total.min(), 0.0), max(positive.max(), total.max())
    ax.set_ylim(low - 0.05 * (high - low), high + 0.35 * (high - low))
    ax.legend(ncols=5, loc="upper left")
    style.save(fig, "fx_03_is_decomposition")


def fig_cost_vs_size(cfg: FxConfig, parents: pd.DataFrame) -> None:
    """Figure 4: IS against notional on a log axis, one panel per pair, square-root fit overlaid.

    FIX_ORDER is excluded from the fit: its IS is dominated by market drift to the fix window
    and is not comparable to the other styles (docs/fx.md).
    """
    style.apply()
    fig, axes = style.figure(
        "Shortfall against order size",
        _subtitle(cfg, "Binned mean IS in bp, 95% CI, fit a + b*size^c; FIX_ORDER excluded"),
        _source(),
        nrows=2,
        ncols=2,
        sharex=True,
    )
    non_fix = parents[parents["style"] != "FIX_ORDER"]
    for ax, pair in zip(axes.ravel(), cfg.pairs.order, strict=True):
        sub = non_fix[(non_fix["pair"] == pair) & (non_fix["filled_frac"] > 0)]
        ax.set_title(pair, fontsize=10, loc="left")
        if len(sub) >= 20:
            common.cost_vs_size(
                ax, sub["notional_usd"] / 1e6, sub["is_bp"], n_bins=6, seed=20250303
            )
            ax.legend().remove()
    for ax in axes[1]:
        ax.set_xlabel("Notional (USD mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("IS (bp)")
    style.save(fig, "fx_04_cost_vs_size")


def _pooled_scorecard(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return analysis.lp_scorecard(
        tables["fx_orders"],
        tables["fx_children"],
        tables["fx_fill_attempts"],
        tables["fx_fills"],
        tables["fx_tca_fill"],
        by=("lp_id",),
    ).set_index("lp_id")


def fig_lp_scorecard(cfg: FxConfig, tables: dict[str, pd.DataFrame]) -> None:
    """Figure 5: one row per LP with fill, reject, hold, markout and rejection-cost columns."""
    style.apply()
    fig, ax = style.figure(
        "Liquidity-provider scorecard",
        _subtitle(cfg, "Aggressive children, pooled; markout and rejection cost in pips"),
        _source(),
    )
    _sign_note(fig)
    card = _pooled_scorecard(tables)
    frame = pd.DataFrame(
        {
            "Fill ratio": card["fill_ratio"],
            "Reject ratio": card["reject_ratio"],
            "Asym. reject": card["asym_reject"],
            "Median hold (ms)": card["median_hold_ms"],
            "Markout 1s (pips)": card["markout_eff_pips_1"],
            "Cost of rejection (pips)": card["cost_of_rejection_pips"],
        }
    )
    frame.index = [f"LP{i}" for i in frame.index]
    common.heatmap_with_values(ax, frame, fmt=".3f")
    ax.tick_params(axis="x", labelsize=8)
    style.save(fig, "fx_05_lp_scorecard")


def fig_pretrade_calibration(cfg: FxConfig, parents: pd.DataFrame) -> None:
    """Figure 6: realised IS against the pre-trade estimate, binned, with bias and RMSE by style."""
    style.apply()
    fig, ax = style.figure(
        "Pre-trade estimate against realised shortfall",
        _subtitle(cfg, "Binned mean realised IS in bp, 95% CI; estimate not fitted"),
        _source(),
    )
    filled = parents[parents["filled_frac"] > 0]
    table = common.binned_means(filled["pretrade_est_bp"], filled["is_bp"], 8)
    ax.errorbar(
        table["x_mid"],
        table["mean"],
        yerr=[table["mean"] - table["lo"], table["hi"] - table["mean"]],
        fmt="o",
        color=style.PALETTE[0],
        capsize=3,
        label="Realised, bin mean",
    )
    top = float(max(table["x_mid"].max(), table["mean"].max()))
    ax.plot([0, top], [0, top], color=style.PALETTE[3], label="45-degree line")
    errors = analysis.pretrade_errors(parents)
    lines = [
        f"{r.style}: bias {r.bias_bp:+.2f}, RMSE {r.rmse_bp:.2f} (n={r.n})"
        for r in errors.itertuples()
    ]
    fig.subplots_adjust(right=0.70)
    fig.text(0.72, 0.5, "\n".join(lines), ha="left", va="center", fontsize=8)
    ax.set_xlabel("Pre-trade estimate (bp)")
    ax.set_ylabel("Realised IS (bp)")
    ax.legend(loc="lower left")
    style.save(fig, "fx_06_pretrade_calibration")


def fig_markouts_by_lp(cfg: FxConfig, tables: dict[str, pd.DataFrame]) -> None:
    """Figure 7: effective markout curves by LP, aggressive fills only, 95% CI."""
    style.apply()
    fig, ax = style.figure(
        "Markouts after the fill, by liquidity provider",
        _subtitle(cfg, "Mean effective markout in bp, 95% CI, aggressive fills"),
        _source(),
    )
    _sign_note(fig)
    tfill = tables["fx_tca_fill"]
    fills = tfill[(tfill["intent"] == "AGGRESSIVE") & (tfill["lp_id"] > 0)]
    horizons = analysis.horizons_from(tfill)
    rng = np.random.default_rng(7)
    for k, (lp_id, group) in enumerate(fills.groupby("lp_id", sort=True)):
        stats = [_mean_ci(group[f"markout_eff_bp_{h}"].to_numpy(), rng) for h in horizons]
        mean, lo, hi = (np.array(col) for col in zip(*stats, strict=True))
        common.markout_curve(
            ax, range(len(horizons)), mean, lo, hi, f"LP{lp_id}", style.PALETTE[k % 8]
        )
    ax.set_xticks(range(len(horizons)), [HORIZON_LABELS.get(h, f"{h}s") for h in horizons])
    ax.set_xlabel("Horizon after fill")
    ax.set_ylabel("Markout (bp)")
    ax.legend(ncols=5, loc="upper left")
    style.save(fig, "fx_07_markouts_by_lp")


def make_all(cfg: FxConfig, tables: dict[str, pd.DataFrame]) -> list[str]:
    """Draw the seven figures into :func:`tca.fx.paths.figures_dir`; return their names."""
    with paths.figures_env() as target:
        parents = tables["fx_tca_parent"]
        fig_is_by_style_size(cfg, parents)
        fig_liquidity_heatmap(cfg, tables["fx_liquidity_profile"])
        fig_is_decomposition(cfg, parents)
        fig_cost_vs_size(cfg, parents)
        fig_lp_scorecard(cfg, tables)
        fig_pretrade_calibration(cfg, parents)
        fig_markouts_by_lp(cfg, tables)
        return sorted(p.name for p in target.glob("fx_*.png"))
