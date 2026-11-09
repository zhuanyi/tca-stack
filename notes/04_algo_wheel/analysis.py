"""Note 1 - ranking brokers, algos and dealers honestly: the algo-wheel framework.

The note asks whether a desk can rank its execution brokers from realised flow. The flow is the
Module 2 FX simulation, so the broker effects are planted (config/brokers.yaml) and the question
is whether the methodology recovers them. This script reproduces every number and figure in
NOTE.md; the prose quotes values from results.json, which it writes here.

Run from the repository root::

    uv run python notes/04_algo_wheel/analysis.py

It reads the committed Module 2 tables (fx_tca_parent, fx_orders, fx_liquidity_profile) and the
Module 1 dealer tables (ust_rfq, ust_rfq_quotes, ust_orders). Nothing here writes to data/; the
outputs are figures/*.png, results.json and the printed tables.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tca.core import stats
from tca.fx.config import BrokerCfg, FxConfig, hhmm_to_seconds, load_brokers, load_config
from tca.fx.paths import derived_dir
from tca.plotting import style

NOTE_DIR = Path(__file__).resolve().parent
FIG_DIR = NOTE_DIR / "figures"
RESULTS_PATH = NOTE_DIR / "results.json"

STYLES = ("TWAP", "POV", "IS")
SIZE_EDGES = [0.0, 10e6, 50e6, np.inf]
SIZE_LABELS = ["<10mm", "10-50mm", ">50mm"]
DAYS_PER_YEAR = 252
HOUR_EDGES_LATE = 16
N_BOOT = 2000
SEED = 20250303
DELTAS_BP = (0.1, 0.25, 0.5, 1.0)
LEARNING_ORDERS = 2000
LEARNING_WARMUP = 200
LEARNING_BATCH = 50
LEARNING_SEEDS = 200
PRIOR_VAR_BP2 = 1.0
DEFAULT_WINDOW = (7, 12)
# UST flow is quoted in US hours; the same four time-of-day buckets are used as a control, so
# the window only sets the bucket boundaries and is not a session definition.
WINDOW_DEALER = DEFAULT_WINDOW

MAIN_FORMULA = (
    "is_bp ~ log_size_rel + sigma_d_bp + hs_bp_at_arrival + C(hour_bucket) + C(pair) "
    "+ C(style) + C(broker)"
)
# The interaction fit replaces the four hour-bucket dummies with a quadratic hour trend: the
# session dummy is a function of the hour, so it is not identified next to its own bucket dummy.
INTERACTION_FORMULA = (
    "is_bp ~ log_size_rel + sigma_d_bp + hs_bp_at_arrival + hour + I(hour**2) + C(pair) "
    "+ C(style) + C(broker) + C(broker):log_size_rel + C(broker):london_morning"
)
DEALER_HIT_FORMULA = (
    "win ~ log_qty + C(tenor) + C(run_status) + n_dealers + C(hour_bucket) + C(dealer_id)"
)
DEALER_MARGIN_FORMULA = (
    "margin_32nds ~ log_qty + C(tenor) + C(run_status) + n_dealers + C(hour_bucket) + C(dealer_id)"
)


# ------------------------------------------------------------------------------------------
# Configuration and data
# ------------------------------------------------------------------------------------------


def london_window(book: dict[str, BrokerCfg]) -> tuple[int, int]:
    """Start and end hour (UTC) of the planted time-of-day markup window, read from config.

    config/brokers.yaml gives one broker a window; no hour boundary is hard-coded here.
    """
    for broker in book.values():
        for window in broker.markup_windows:
            start = hhmm_to_seconds(window.start) // 3600
            end = hhmm_to_seconds(window.end) // 3600
            return start, end
    return DEFAULT_WINDOW


def _hour_buckets(hour: pd.Series, window: tuple[int, int]) -> pd.Series:
    """Four UTC hour buckets: before the window, the window, the overlap, the rest."""
    start, end = window
    if not 0 < start < end < HOUR_EDGES_LATE <= 24:
        raise ValueError(f"unexpected markup window {window}; expected 0 < start < end < 16")
    edges = [0, start, end, HOUR_EDGES_LATE, 24]
    labels = ["pre", "london", "overlap", "late"]
    return pd.cut(hour, edges, right=False, labels=labels).astype(str)


def _liquidity_lookup(liquidity: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """All-day rows of the liquidity profile, indexed by pair and five-minute bucket."""
    all_day = liquidity[liquidity["dow"] == -1]
    if all_day.empty:
        # A profile without the all-day sentinel falls back to the per-weekday median.
        overall = liquidity.groupby(["pair", "bucket"])[
            ["spread_median_bp", "realised_vol_ann"]
        ].median()
    else:
        overall = all_day.set_index(["pair", "bucket"])
    return overall, overall["spread_median_bp"] / 2.0


def add_features(
    parents: pd.DataFrame,
    orders: pd.DataFrame,
    liquidity: pd.DataFrame,
    cfg: FxConfig,
    book: dict[str, BrokerCfg],
) -> pd.DataFrame:
    """The analysis panel: one row per TWAP, POV or IS parent with its difficulty controls.

    Adds log_size_rel (log notional over the pair's daily volume proxy), sigma_d_bp (daily
    volatility from the liquidity profile), hs_bp_at_arrival (half the median spread in the
    arrival bucket), hour_bucket, london_morning, size_bucket and day.
    """
    cols = ["order_id", "style", "assignment", "notional_usd", "ts_arrival_utc", "broker", "pair"]
    merged = parents.merge(
        orders[cols], on="order_id", suffixes=("", "_order"), validate="one_to_one"
    )
    for col in cols[1:]:
        left, right = merged[col], merged[f"{col}_order"]
        same = (left == right) | (left.isna() & right.isna())
        if not bool(np.asarray(same).all()):
            raise ValueError(f"fx_orders and fx_tca_parent disagree on {col!r}")
    df = merged.drop(columns=[f"{c}_order" for c in cols[1:]])
    df = df[df["style"].isin(STYLES)].copy()

    df["V_pair"] = df["pair"].map({p: cfg.daily_volume_usd(p) for p in df["pair"].unique()})
    df["log_size_rel"] = np.log(df["notional_usd"] / df["V_pair"])

    ts = df["ts_arrival_utc"]
    df["day"] = ts.dt.date
    df["dow"] = ts.dt.day_of_week
    df["bucket"] = (ts.dt.hour * 60 + ts.dt.minute) // 5
    df["hour"] = ts.dt.hour
    window = london_window(book)
    df["hour_bucket"] = _hour_buckets(df["hour"], window)
    df["london_morning"] = ((df["hour"] >= window[0]) & (df["hour"] < window[1])).astype(int)
    df["size_bucket"] = pd.cut(df["notional_usd"], SIZE_EDGES, labels=SIZE_LABELS)

    overall, hs = _liquidity_lookup(liquidity)
    vol = liquidity.set_index(["pair", "dow", "bucket"])["realised_vol_ann"]
    key = pd.MultiIndex.from_arrays([df["pair"], df["dow"], df["bucket"]])
    daily_vol = vol.reindex(key).to_numpy(dtype=float)
    fallback = df["pair"].map(overall.groupby(level="pair")["realised_vol_ann"].median())
    daily_vol = np.where(np.isfinite(daily_vol), daily_vol, fallback.to_numpy(dtype=float))
    df["sigma_d_bp"] = daily_vol / np.sqrt(DAYS_PER_YEAR) * 1e4

    df["hs_bp_at_arrival"] = [
        hs.get((p, b), np.nan) for p, b in zip(df["pair"], df["bucket"], strict=True)
    ]
    required = ["is_bp", "log_size_rel", "sigma_d_bp", "hs_bp_at_arrival", "hour_bucket"]
    return df.dropna(subset=required).reset_index(drop=True)


def load_panel() -> pd.DataFrame:
    """The committed Module 2 tables as one analysis panel."""
    derived = derived_dir()
    parents = pd.read_parquet(derived / "fx_tca_parent.parquet")
    orders = pd.read_parquet(derived / "fx_orders.parquet")
    liquidity = pd.read_parquet(derived / "fx_liquidity_profile.parquet")
    return add_features(parents, orders, liquidity, load_config(), load_brokers())


# ------------------------------------------------------------------------------------------
# 3.1 Naive ranking
# ------------------------------------------------------------------------------------------


def naive_ranking(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mean implementation shortfall by broker, with a 95% bootstrap interval.

    Two scopes: every order (all) and RANDOM assignment only. Returns one row per broker and
    scope with mean, lo, hi and n.
    """
    rows = []
    for scope, subset in (("all", df), ("random", df[df["assignment"] == "RANDOM"])):
        for broker, group in subset.groupby("broker", sort=True):
            values = group["is_bp"].to_numpy(dtype=float)
            lo, hi = stats.bootstrap_ci(values, rng=rng) if len(values) > 1 else (np.nan, np.nan)
            rows.append((scope, broker, float(values.mean()), lo, hi, len(values)))
    return pd.DataFrame(rows, columns=["scope", "broker", "mean", "lo", "hi", "n"])


def markup_at_arrival(df: pd.DataFrame, book: dict[str, BrokerCfg]) -> pd.DataFrame:
    """The configured broker markup on every row at its arrival hour, one column per broker.

    The time-of-day window of config/brokers.yaml replaces the base markup inside the window.
    Broker markup is charged on aggressive fills only, so the realised average is scaled by the
    aggressive share in planted_markers.
    """
    minute = df["ts_arrival_utc"].dt.hour * 60 + df["ts_arrival_utc"].dt.minute
    columns = {}
    for broker in sorted(book):
        cfg = book[broker]
        values = np.full(len(df), cfg.markup_bp, dtype=float)
        for window in cfg.markup_windows:
            start = hhmm_to_seconds(window.start) // 60
            end = hhmm_to_seconds(window.end) // 60
            values = np.where((minute >= start) & (minute < end), window.markup_bp, values)
        columns[broker] = values
    return pd.DataFrame(columns, index=df.index)


def aggressive_share(df: pd.DataFrame) -> pd.Series:
    """Share of each broker's fills that are aggressive, and so carry its markup."""
    share = 1.0 - df.groupby("broker")["pct_passive_filled"].mean() / 100.0
    return share.clip(lower=0.0)


def _panel_mean(ranking: pd.DataFrame, scope: str, broker: str) -> float:
    """Mean IS of one broker in one panel, NaN when that broker has no orders there."""
    rows = ranking[(ranking["scope"] == scope) & (ranking["broker"] == broker)]
    return float(rows["mean"].iloc[0]) if len(rows) else float("nan")


def planted_markers(
    df: pd.DataFrame, book: dict[str, BrokerCfg], centre: float
) -> dict[str, float]:
    """Planted markup as it reaches the flow, relative to the cross-broker mean.

    The configured markup at each order's arrival hour, averaged over the order mix, times the
    broker's aggressive share (passive fills pay no broker markup), centred on the observed
    panel mean. Implementation shortfall also contains spread, delay and impact that are not
    planted, so only the differences between brokers are comparable.
    """
    markup = markup_at_arrival(df, book).mean()
    realised = markup * aggressive_share(df)
    return {b: float(centre + realised[b] - realised.mean()) for b in realised.index}


# ------------------------------------------------------------------------------------------
# 3.2 Difficulty adjustment
# ------------------------------------------------------------------------------------------


@dataclass
class Adjustment:
    """The two difficulty-adjusted fits and their broker effects."""

    main: object
    interaction: object
    table: pd.DataFrame
    effects: pd.DataFrame
    random_effects: pd.DataFrame


def _broker_effects(model, brokers: list[str]) -> pd.DataFrame:
    """Broker coefficients relative to the base broker, with 95% confidence intervals."""
    rows = []
    for broker in brokers:
        name = f"C(broker)[T.{broker}]"
        coef = float(model.params.get(name, 0.0))
        se = float(model.bse.get(name, np.nan))
        lo = coef - 1.96 * se if np.isfinite(se) else np.nan
        hi = coef + 1.96 * se if np.isfinite(se) else np.nan
        rows.append((broker, coef, se, lo, hi))
    return pd.DataFrame(rows, columns=["broker", "coef", "se", "lo", "hi"])


def _regression_table(model) -> pd.DataFrame:
    """Coefficient, clustered standard error, p-value and 95% interval for every term."""
    conf = model.conf_int()
    rows = []
    for name in model.params.index:
        rows.append(
            (
                name,
                float(model.params[name]),
                float(model.bse[name]),
                float(model.pvalues[name]),
                float(conf.loc[name, 0]),
                float(conf.loc[name, 1]),
            )
        )
    return pd.DataFrame(rows, columns=["term", "coef", "se", "p", "lo", "hi"])


def difficulty_adjustment(df: pd.DataFrame, brokers: list[str]) -> Adjustment:
    """OLS of implementation shortfall on difficulty controls and broker dummies.

    Standard errors are clustered by day. The interaction fit adds broker-by-size slopes and
    broker-by-session shifts, which is where A's large-order penalty and C's window show up.
    """
    main = stats.ols_cluster(df, MAIN_FORMULA, "day")
    interaction = stats.ols_cluster(df, INTERACTION_FORMULA, "day")
    random_only = stats.ols_cluster(df[df["assignment"] == "RANDOM"], MAIN_FORMULA, "day")
    return Adjustment(
        main=main,
        interaction=interaction,
        table=_regression_table(main),
        effects=_broker_effects(main, brokers),
        random_effects=_broker_effects(random_only, brokers),
    )


def _contrast(model, name: str, base: str) -> tuple[float, float]:
    """Coefficient difference and its standard error from the model covariance matrix."""
    params = model.params
    cov = model.cov_params()
    diff = float(params[name] - params[base])
    var = float(cov.loc[name, name] + cov.loc[base, base] - 2.0 * cov.loc[name, base])
    return diff, float(np.sqrt(max(var, 0.0)))


def _interaction_summary(model, brokers: list[str]) -> dict[str, object]:
    """The interaction terms the note quotes, with the base-broker contrasts made explicit.

    The base broker is read from the model, never assumed: patsy uses the alphabetically first
    broker as the reference level. Every contrast key carries the base broker's name, so the
    keys stay self-describing if the broker set changes.
    """
    params, bse = model.params, model.bse
    base = next(b for b in brokers if f"C(broker)[T.{b}]" not in params)
    out: dict[str, object] = {
        "base_broker": base,
        "size_slope_base": round(float(params["log_size_rel"]), 4),
    }
    session_base = f"C(broker)[{base}]:london_morning"
    out["session_base"] = round(float(params[session_base]), 4) if session_base in params else 0.0
    out["session_base_se"] = round(float(bse[session_base]), 4) if session_base in params else 0.0
    for broker in brokers:
        if broker == base:
            continue
        size_name = f"C(broker)[T.{broker}]:log_size_rel"
        out[f"size_slope_{broker}_minus_{base}"] = round(float(params[size_name]), 4)
        out[f"size_slope_{broker}_se"] = round(float(bse[size_name]), 4)
        session_name = f"C(broker)[{broker}]:london_morning"
        if session_name in params and session_base in params:
            session, session_se = _contrast(model, session_name, session_base)
        else:
            session = float(params.get(session_name, np.nan))
            session_se = float(bse.get(session_name, np.nan))
        out[f"session_{broker}_minus_{base}"] = round(session, 4)
        out[f"session_{broker}_se"] = round(session_se, 4)
    out["session_C_coef"] = round(float(params.get("C(broker)[C]:london_morning", np.nan)), 4)
    out["session_C_coef_se"] = round(float(bse.get("C(broker)[C]:london_morning", np.nan)), 4)
    return out


def broker_effect_series(model, brokers: list[str]) -> pd.Series:
    """Broker dummy coefficient per broker, with the base broker at zero."""
    base = next(b for b in brokers if f"C(broker)[T.{b}]" not in model.params)
    values = {b: (0.0 if b == base else float(model.params[f"C(broker)[T.{b}]"])) for b in brokers}
    return pd.Series(values)


def adjusted_values(df: pd.DataFrame, model, broker_effect: pd.Series) -> np.ndarray:
    """Implementation shortfall with the non-broker part of the model removed.

    is_bp - (fitted - broker_dummy) strips size, volatility, spread, hour, pair and style.
    What remains is the row's own cost after the difficulty adjustment.
    """
    fitted = np.asarray(model.fittedvalues, dtype=float)
    effects = broker_effect.reindex(df["broker"]).to_numpy(dtype=float)
    if len(fitted) != len(df):
        raise ValueError("adjusted_values needs a model fitted on every row of df")
    return df["is_bp"].to_numpy(dtype=float) - (fitted - effects)


# ------------------------------------------------------------------------------------------
# 3.3 Shrinkage
# ------------------------------------------------------------------------------------------


def shrinkage_cells(df: pd.DataFrame, adjustment: Adjustment, brokers: list[str]) -> pd.DataFrame:
    """Empirical-Bayes shrunk cell means of the adjusted cost, 48 cells.

    Cells are broker x pair x size bucket. Each cell is shrunk toward its broker's overall
    adjusted mean with a normal-normal model; tau2 is the method-of-moments between-cell
    variance across all cells. Columns: n, raw (adjusted cell mean), s2 (within-cell variance),
    mu (broker mean), shrunk and move (shrunk minus raw).
    """
    effects = broker_effect_series(adjustment.main, brokers)
    df = df.assign(adjusted=adjusted_values(df, adjustment.main, effects))
    grouped = df.groupby(["broker", "pair", "size_bucket"], observed=True)["adjusted"]
    cells = pd.DataFrame({"n": grouped.size(), "raw": grouped.mean(), "s2": grouped.var()})
    cells["s2"] = cells["s2"].fillna(0.0)
    tau2 = stats.tau2_mom(cells["raw"], cells["n"], cells["s2"])
    broker_mean = df.groupby("broker")["adjusted"].mean()
    cells["mu"] = cells.index.get_level_values("broker").map(broker_mean)
    cells["tau2"] = tau2
    cells["shrunk"] = stats.eb_shrink(cells["raw"], cells["n"], cells["s2"], cells["mu"], tau2)
    cells["move"] = cells["shrunk"] - cells["raw"]
    return cells.reset_index()


def shrinkage_summary(cells: pd.DataFrame) -> dict[str, float]:
    """Mean absolute move of the smallest and largest third of cells, and the rank correlation."""
    order = cells.sort_values("n")
    third = max(1, len(cells) // 3)
    small = float(order.head(third)["move"].abs().mean())
    large = float(order.tail(third)["move"].abs().mean())
    rho = float(order["move"].abs().corr(order["n"], method="spearman"))
    return {"small_n_move": small, "large_n_move": large, "move_vs_n_spearman": rho}


# ------------------------------------------------------------------------------------------
# 3.4 Power
# ------------------------------------------------------------------------------------------


def power_table(
    raw_sd: float, resid_sd: float, n_per_broker: int, orders_per_month: float
) -> pd.DataFrame:
    """Orders per broker needed to detect each effect size, and the months at this flow.

    stats.required_n is the two-sample normal approximation at 80% power and 5% two-sided
    significance. months divides the required orders by the desk's orders per broker per month.
    """
    rows = []
    for delta in DELTAS_BP:
        n_raw = stats.required_n(delta, raw_sd)
        n_adj = stats.required_n(delta, resid_sd)
        rows.append((delta, n_raw, n_adj, n_adj / orders_per_month if orders_per_month else np.nan))
    return pd.DataFrame(rows, columns=["delta_bp", "n_raw", "n_adjusted", "months"])


# ------------------------------------------------------------------------------------------
# 3.5 Learning while allocating
# ------------------------------------------------------------------------------------------


def planted_components(df: pd.DataFrame, book: dict[str, BrokerCfg]) -> np.ndarray:
    """The planted cost of every broker on every row: markup plus impact effects.

    Columns follow sorted(book). Markup uses the time-of-day window of config/brokers.yaml at
    the row's arrival hour; impact_multiplier scales the observed impact and
    post_fill_impact_bp adds leakage per fill.
    """
    brokers = sorted(book)
    markup_part = markup_at_arrival(df, book)[brokers].to_numpy(dtype=float)
    impact = df["impact_bp"].to_numpy(dtype=float)[:, None]
    multiplier = np.array([book[b].impact_multiplier - 1.0 for b in brokers])
    fills = df["n_fills"].to_numpy(dtype=float)[:, None]
    leakage = np.array([book[b].post_fill_impact_bp for b in brokers])
    return markup_part + impact * multiplier + fills * leakage


def counterfactual_costs(df: pd.DataFrame, book: dict[str, BrokerCfg]) -> np.ndarray:
    """Cost every broker would have charged on every row, at its planted effect.

    The row's own broker cost is differenced out (is_bp - planted) and each broker's planted
    cost added back, so the matrix shares the row's market noise and difficulty.
    """
    brokers = sorted(book)
    planted = planted_components(df, book)
    index = {b: i for i, b in enumerate(brokers)}
    own = np.array([planted[i, index[b]] for i, b in enumerate(df["broker"])])
    return df["is_bp"].to_numpy(dtype=float)[:, None] - own[:, None] + planted


@dataclass
class LearningResult:
    """Cumulative regret paths and the summary the note quotes."""

    curve: pd.DataFrame
    summary: dict[str, object]


def _design(df: pd.DataFrame) -> np.ndarray:
    """Control matrix for the adjustment in the learning simulation (no broker column)."""
    dummies = [
        pd.get_dummies(df["pair"].astype(str)).to_numpy(dtype=float),
        pd.get_dummies(df["style"].astype(str)).to_numpy(dtype=float),
        pd.get_dummies(df["hour_bucket"].astype(str)).to_numpy(dtype=float),
    ]
    return np.column_stack([np.ones(len(df)), df["log_size_rel"].to_numpy(dtype=float), *dummies])


def _wta_pick(observed_broker: np.ndarray, observed_cost: np.ndarray, n_brokers: int) -> int:
    """Broker with the lowest naive mean over the warm-up window (infinite when unseen)."""
    sums = np.bincount(observed_broker, weights=observed_cost, minlength=n_brokers)
    counts = np.bincount(observed_broker, minlength=n_brokers)
    means = np.divide(sums, counts, out=np.full(n_brokers, np.inf), where=counts > 0)
    return int(np.argmin(means))


def learning_experiment(
    df: pd.DataFrame,
    book: dict[str, BrokerCfg],
    n_orders: int = LEARNING_ORDERS,
    warmup: int = LEARNING_WARMUP,
    batch: int = LEARNING_BATCH,
    seeds: int = LEARNING_SEEDS,
    seed: int = SEED,
) -> LearningResult:
    """Cumulative regret of three allocation policies over a simulated sequence of orders.

    The sequence is n_orders rows drawn with replacement from the order population; every
    broker's cost on every order is known from the planted effects. Policies: (a) fixed equal
    split, (b) winner takes all on the naive mean after the warm-up orders, (c) Thompson sampling
    on the difficulty-adjusted residual, refitted every batch orders. Regret is against always
    picking the cheapest broker for that order, in bp.
    """
    brokers = sorted(book)
    n_brokers = len(brokers)
    if n_orders <= warmup:
        raise ValueError("n_orders must exceed the warm-up window")
    rng = np.random.default_rng(seed)
    index = {b: i for i, b in enumerate(brokers)}

    paths = {
        "equal_split": np.zeros((seeds, n_orders)),
        "wta": np.zeros((seeds, n_orders)),
        "ts": np.zeros((seeds, n_orders)),
    }
    picks = np.zeros(seeds, dtype=int)
    ts_share = np.zeros((seeds, n_brokers))
    mean_cost = np.zeros((seeds, n_brokers))

    for s in range(seeds):
        # Each seed draws its own order sequence, so the warm-up window differs across seeds and
        # winner-takes-all can lock in a wrong broker on some of them.
        policy_rng = np.random.default_rng(seed + s)
        population = df.iloc[rng.integers(0, len(df), n_orders)].reset_index(drop=True)
        costs = counterfactual_costs(population, book)
        best = costs.min(axis=1)
        observed = population["broker"].map(index).to_numpy()
        design = _design(population)
        warm_cost = costs[np.arange(warmup), observed[:warmup]]
        mean_cost[s] = costs.mean(axis=0)
        paths["equal_split"][s] = np.cumsum(costs.mean(axis=1) - best)

        pick = _wta_pick(observed[:warmup], warm_cost, n_brokers)
        picks[s] = pick
        chosen = np.full(n_orders, pick)
        chosen[:warmup] = observed[:warmup]
        paths["wta"][s] = np.cumsum(costs[np.arange(n_orders), chosen] - best)

        rows = np.arange(warmup)
        brokers_seen = list(observed[:warmup])
        costs_seen = list(warm_cost)
        regret = np.zeros(n_orders)
        for start in range(warmup, n_orders, batch):
            x = design[rows]
            y = np.asarray(costs_seen, dtype=float)
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            residual = y - x @ beta
            sd = float(residual.std(ddof=x.shape[1]))
            seen = np.asarray(brokers_seen)
            mean = np.array(
                [residual[seen == k].mean() if (seen == k).any() else 0.0 for k in range(n_brokers)]
            )
            count = np.bincount(seen, minlength=n_brokers)
            posterior_var = 1.0 / (1.0 / PRIOR_VAR_BP2 + count / sd**2)
            posterior_mean = mean * count / sd**2 * posterior_var
            pick = int(policy_rng.normal(posterior_mean, np.sqrt(posterior_var)).argmin())
            end = min(start + batch, n_orders)
            regret[start:end] = costs[start:end, pick] - best[start:end]
            rows = np.concatenate([rows, np.arange(start, end)])
            brokers_seen += [pick] * (end - start)
            costs_seen += list(costs[start:end, pick])
        paths["ts"][s] = np.cumsum(regret)
        ts_share[s] = np.bincount(np.asarray(brokers_seen), minlength=n_brokers)

    population_mean = mean_cost.mean(axis=0)
    true_best = int(np.argmin(population_mean))
    curve = pd.concat(
        [
            pd.DataFrame(
                {
                    "policy": policy,
                    "order": np.arange(1, n_orders + 1),
                    "mean": array.mean(axis=0),
                    "lo": np.quantile(array, 0.1, axis=0),
                    "hi": np.quantile(array, 0.9, axis=0),
                }
            )
            for policy, array in paths.items()
        ],
        ignore_index=True,
    )
    summary = {
        "n_orders": n_orders,
        "warmup": warmup,
        "batch": batch,
        "seeds": seeds,
        "true_best": brokers[true_best],
        "true_mean_cost": {b: float(population_mean[i]) for i, b in enumerate(brokers)},
        "equal_split_bp": float(paths["equal_split"][:, -1].mean() / n_orders),
        "wta_bp": float(paths["wta"][:, -1].mean() / n_orders),
        "ts_bp": float(paths["ts"][:, -1].mean() / n_orders),
        "wta_wrong_lock_fraction": float(np.mean(picks != true_best)),
        "ts_below_wta_fraction": float(np.mean(paths["ts"][:, -1] < paths["wta"][:, -1])),
        "ts_share": {b: float(ts_share[:, i].mean() / n_orders) for i, b in enumerate(brokers)},
    }
    return LearningResult(curve, summary)


# ------------------------------------------------------------------------------------------
# 3.6 The dealer version
# ------------------------------------------------------------------------------------------


def dealer_table(
    quotes: pd.DataFrame, rfq: pd.DataFrame, orders: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Dealer fixed effects on RFQ hit probability and on the winner's margin.

    The hit model is a logit of winning on size, tenor, run status, the number of dealers and
    hour, with dealer dummies, clustered by parent order. The margin model is the same OLS on
    winners, in 32nds of a point. Both use only dealers that responded; the base dealer is the
    alphabetically first.
    """
    cols = ["rfq_id", "order_id", "qty_mm", "n_dealers", "side", "cover_px", "ts_send_utc"]
    q = quotes.merge(rfq[cols], on="rfq_id", how="left")
    q = q.merge(orders[["order_id", "tenor", "run_status"]], on="order_id", how="left")
    q = q[q["responded"]].copy()
    q["win"] = q["is_winner"].astype(int)
    q["log_qty"] = np.log(q["qty_mm"])
    q["margin_32nds"] = q["side"] * (q["cover_px"] - q["quote_px"]) * 32.0
    # ts_send_utc is stored tz-aware UTC; convert explicitly so the bucket labels stay UTC.
    q["hour_bucket"] = _hour_buckets(q["ts_send_utc"].dt.tz_convert("UTC").dt.hour, WINDOW_DEALER)

    hit = stats.logit_cluster(q, DEALER_HIT_FORMULA, "order_id")
    wins = q[q["win"] == 1]
    margin = stats.ols_cluster(wins, DEALER_MARGIN_FORMULA, "order_id")

    dealers = sorted(q["dealer_id"].unique())
    base = next(d for d in dealers if f"C(dealer_id)[T.{d}]" not in hit.params)
    rows = []
    for dealer in dealers:
        name = f"C(dealer_id)[T.{dealer}]"
        rows.append(
            (
                dealer,
                int(q.loc[q["dealer_id"] == dealer, "win"].sum()),
                int((q["dealer_id"] == dealer).sum()),
                float(q.loc[q["dealer_id"] == dealer, "win"].mean()),
                float(hit.params.get(name, 0.0)),
                float(hit.bse.get(name, np.nan)),
                0.0 if dealer == base else float(margin.params.get(name, np.nan)),
                0.0 if dealer == base else float(margin.bse.get(name, np.nan)),
                len(wins[wins["dealer_id"] == dealer]),
            )
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "dealer_id",
            "wins",
            "responses",
            "hit_rate",
            "hit_effect",
            "hit_se",
            "margin_effect_32nds",
            "margin_se",
            "margin_wins",
        ],
    )
    meta = {
        "base_dealer": base,
        "n_responses": len(q),
        "hit_n": int(hit.nobs),
        "margin_n": int(margin.nobs),
    }
    return table, meta


# ------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------


@contextmanager
def figures_env() -> Iterator[Path]:
    """Point tca.plotting at notes/04_algo_wheel/figures for the duration of the block."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("TCA_FIGURES_DIR")
    os.environ["TCA_FIGURES_DIR"] = str(FIG_DIR)
    try:
        yield FIG_DIR
    finally:
        if previous is None:
            os.environ.pop("TCA_FIGURES_DIR", None)
        else:
            os.environ["TCA_FIGURES_DIR"] = previous


def _source() -> str:
    return (
        "Simulated spot FX flow and replay engine (Module 2); simulated UST RFQ flow "
        "(Module 1). Planted broker effects in config/brokers.yaml."
    )


def figure_naive(ranking: pd.DataFrame, df: pd.DataFrame, book: dict[str, BrokerCfg]) -> Path:
    """Figure 1: mean IS by broker, all orders and random only, with planted markup markers."""
    fig, axes = style.figure(
        "Figure 1 - The naive ranking reads the routing, not the broker",
        "Mean implementation shortfall (bp), 95% bootstrap intervals; diamonds: planted markup "
        "on the aggressive share",
        _source(),
        ncols=2,
    )
    for ax, scope, label in zip(
        axes, ("all", "random"), ("All orders", "Random only"), strict=True
    ):
        part = ranking[ranking["scope"] == scope].sort_values("broker")
        x = np.arange(len(part))
        ax.errorbar(
            x,
            part["mean"],
            yerr=[part["mean"] - part["lo"], part["hi"] - part["mean"]],
            fmt="o",
            capsize=4,
            color=style.PALETTE[0],
            label="Observed mean, 95% CI",
        )
        subset = df if scope == "all" else df[df["assignment"] == "RANDOM"]
        markers = planted_markers(subset, book, float(part["mean"].mean()))
        ax.scatter(
            x,
            [markers[b] for b in part["broker"]],
            marker="D",
            s=55,
            color=style.PALETTE[3],
            zorder=3,
            label="Planted markup effect",
        )
        ax.set_xticks(x, [f"{b}\nn={n:,}" for b, n in zip(part["broker"], part["n"], strict=True)])
        ax.set_ylabel("Implementation shortfall (bp)")
        ax.set_title(label, loc="left")
        ax.legend(loc="upper left", fontsize=9)
    return style.save(fig, "01_naive_ranking")


def figure_effects(
    effects_all: pd.DataFrame,
    effects_random: pd.DataFrame,
    df: pd.DataFrame,
    book: dict[str, BrokerCfg],
) -> Path:
    """Figure 2: broker coefficients with CIs against the planted markup differences."""
    fig, axes = style.figure(
        "Figure 2 - Adjusting for difficulty moves the ranking toward the truth",
        "Broker coefficient vs base (bp), clustered by day; diamonds: planted markup vs base",
        _source(),
        ncols=2,
    )
    for ax, effects, label in zip(
        axes, (effects_all, effects_random), ("All orders", "Random only"), strict=True
    ):
        subset = df if label == "All orders" else df[df["assignment"] == "RANDOM"]
        markers = planted_markers(subset, book, 0.0)
        part = effects.sort_values("broker")
        x = np.arange(len(part))
        ax.errorbar(
            x,
            part["coef"],
            yerr=[part["coef"] - part["lo"], part["hi"] - part["coef"]],
            fmt="o",
            capsize=4,
            color=style.PALETTE[0],
            label="Adjusted coefficient, 95% CI",
        )
        base = part["broker"].iloc[0]
        ax.scatter(
            x,
            [float(markers[b] - markers[base]) for b in part["broker"]],
            marker="D",
            s=55,
            color=style.PALETTE[3],
            zorder=3,
            label="Planted markup vs base",
        )
        ax.axhline(0.0, color="#595959", linewidth=0.8)
        ax.set_xticks(x, part["broker"])
        ax.set_ylabel("Coefficient vs base broker (bp)")
        ax.set_title(label, loc="left")
        ax.legend(loc="upper left", fontsize=9)
    return style.save(fig, "02_difficulty_adjustment")


def figure_shrinkage(cells: pd.DataFrame) -> Path:
    """Figure 3: raw against shrunk cell means, as a dumbbell chart ordered by cell size."""
    fig, axes = style.figure(
        "Figure 3 - Small cells move a lot, large cells barely move",
        "Adjusted mean implementation shortfall by broker, pair and size bucket (bp); each row "
        "is one cell, ordered by its order count",
        _source(),
        ncols=2,
    )
    ordered = cells.sort_values("n").reset_index(drop=True)
    half = len(ordered) // 2
    for ax, part in zip(axes, (ordered.iloc[:half], ordered.iloc[half:]), strict=True):
        y = np.arange(len(part))
        for yi, (_, row) in zip(y, part.iterrows(), strict=True):
            ax.plot([row["raw"], row["shrunk"]], [yi, yi], color="#D9D9D9", linewidth=1.5, zorder=1)
        ax.scatter(part["raw"], y, s=24, color=style.PALETTE[0], label="Raw cell mean", zorder=2)
        ax.scatter(part["shrunk"], y, s=24, color=style.PALETTE[3], label="Shrunk mean", zorder=3)
        labels = [
            f"{b} {p} {s} (n={n})"
            for b, p, s, n in zip(
                part["broker"], part["pair"], part["size_bucket"], part["n"], strict=True
            )
        ]
        ax.set_yticks(y, labels, fontsize=5.5)
        ax.set_xlabel("Adjusted implementation shortfall (bp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.97, 0.862), fontsize=9)
    fig.subplots_adjust(left=0.18, right=0.99)
    return style.save(fig, "03_shrinkage")


def figure_power(power: pd.DataFrame, n_per_broker: int) -> Path:
    """Figure 4: orders per broker needed to detect an effect, raw versus adjusted."""
    fig, ax = style.figure(
        "Figure 4 - How many orders before you know",
        "Orders per broker for 80% power at 5% two-sided significance, log scale",
        _source(),
    )
    ax.plot(
        power["delta_bp"],
        power["n_raw"],
        marker="o",
        linestyle="--",
        color=style.PALETTE[2],
        label="Unadjusted (raw sd)",
    )
    ax.plot(
        power["delta_bp"],
        power["n_adjusted"],
        marker="o",
        color=style.PALETTE[0],
        label="Difficulty-adjusted (residual sd)",
    )
    ax.axhline(n_per_broker, color=style.PALETTE[3], linestyle="--", label="Orders this sample")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True difference between two brokers (bp)")
    ax.set_ylabel("Orders per broker")
    ax.set_xticks(list(power["delta_bp"]), [f"{d:g}" for d in power["delta_bp"]])
    ax.legend(loc="upper right", fontsize=9)
    return style.save(fig, "04_power")


def figure_learning(curve: pd.DataFrame) -> Path:
    """Figure 5: cumulative regret of the three allocation policies, mean with 10-90 band."""
    fig, ax = style.figure(
        "Figure 5 - Allocating while learning",
        "Cumulative regret against always choosing the cheapest broker for the order (bp), "
        "mean over seeds with a 10-90% band",
        _source(),
    )
    labels = {
        "equal_split": "Fixed equal split",
        "wta": "Winner takes all after warm-up",
        "ts": "Thompson sampling, difficulty-adjusted",
    }
    for i, (policy, part) in enumerate(curve.groupby("policy", sort=False)):
        colour = style.PALETTE[i]
        ax.fill_between(
            part["order"], part["lo"], part["hi"], color=colour, alpha=0.18, linewidth=0
        )
        ax.plot(part["order"], part["mean"], color=colour, label=labels[policy])
    ax.set_xlabel("Order in the sequence")
    ax.set_ylabel("Cumulative regret (bp)")
    ax.legend(loc="upper left", fontsize=9)
    return style.save(fig, "05_learning")


# ------------------------------------------------------------------------------------------
# Assemble, print and write
# ------------------------------------------------------------------------------------------


def _table1(book: dict[str, BrokerCfg]) -> pd.DataFrame:
    """What the simulator planted, read from config/brokers.yaml."""
    rows = []
    for broker in sorted(book):
        cfg = book[broker]
        windows = "; ".join(f"{w.start}-{w.end}: {w.markup_bp:+.2f}" for w in cfg.markup_windows)
        rows.append(
            (
                broker,
                cfg.markup_bp,
                windows or "-",
                cfg.impact_multiplier,
                cfg.post_fill_impact_bp,
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "broker",
            "markup_bp",
            "markup_window",
            "impact_multiplier",
            "post_fill_impact_bp",
        ],
    )


def _round(value, digits: int = 6):
    """Round floats (and float containers) for JSON; non-finite values become null.

    Python writes NaN and Infinity bare, which is not valid JSON and breaks strict parsers, so
    anything non-finite is mapped to None here.
    """
    if isinstance(value, float):
        return round(value, digits) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    return value


def load_dealer_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The committed Module 1 tables used by the dealer version."""
    derived = derived_dir()
    return (
        pd.read_parquet(derived / "ust_rfq_quotes.parquet"),
        pd.read_parquet(derived / "ust_rfq.parquet"),
        pd.read_parquet(derived / "ust_orders.parquet"),
    )


def collect_results(
    panel: pd.DataFrame,
    book: dict[str, BrokerCfg],
    dealer: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """Run every analysis and return the JSON payload and the tables to print."""
    brokers = sorted(book)
    table1 = _table1(book)
    rng = np.random.default_rng(SEED)
    ranking = naive_ranking(panel, rng)
    adjustment = difficulty_adjustment(panel, brokers)
    cells = shrinkage_cells(panel, adjustment, brokers)
    shrink = shrinkage_summary(cells)

    months = (panel["day"].max() - panel["day"].min()).days / 30.44
    n_per_broker = len(panel) / len(brokers)
    orders_per_month = n_per_broker / months
    resid_sd = float(np.std(adjustment.main.resid, ddof=1))
    power = power_table(
        float(panel["is_bp"].std(ddof=1)), resid_sd, int(n_per_broker), orders_per_month
    )
    learning = learning_experiment(panel, book)

    interaction = _regression_table(adjustment.interaction)
    interaction_summary = _interaction_summary(adjustment.interaction, brokers)
    markup_mean = markup_at_arrival(panel, book).mean()
    share = aggressive_share(panel)
    results: dict[str, object] = {
        "meta": {
            "orders": len(panel),
            "orders_per_broker": int(n_per_broker),
            "months": round(float(months), 4),
            "orders_per_broker_month": round(float(orders_per_month), 4),
            "period_start": str(panel["day"].min()),
            "period_end": str(panel["day"].max()),
            "seed": SEED,
            "styles": list(STYLES),
            "trading_days": int(panel["day"].nunique()),
            "mean_fills": round(float(panel["n_fills"].mean()), 3),
            "passive_share": round(float(panel["pct_passive_filled"].mean() / 100.0), 4),
        },
        "table1": table1.to_dict(orient="records"),
        "table1_by_broker": {
            row["broker"]: {
                "markup_bp": float(row["markup_bp"]),
                "markup_window": str(row["markup_window"]),
                "impact_multiplier": float(row["impact_multiplier"]),
                "post_fill_impact_bp": float(row["post_fill_impact_bp"]),
            }
            for row in table1.to_dict(orient="records")
        },
        "naive_width": {
            scope: {
                row["broker"]: round(float(row["hi"] - row["lo"]), 4)
                for _, row in ranking[ranking["scope"] == scope].iterrows()
            }
            for scope in ("all", "random")
        },
        "naive_gaps": {
            "all_B_minus_A": round(
                _panel_mean(ranking, "all", "B") - _panel_mean(ranking, "all", "A"), 4
            ),
            "all_A_minus_random_A": round(
                _panel_mean(ranking, "all", "A") - _panel_mean(ranking, "random", "A"), 4
            ),
            "random_B_minus_A": round(
                _panel_mean(ranking, "random", "B") - _panel_mean(ranking, "random", "A"), 4
            ),
        },
        "window_spread": {
            f"{broker}_{side}": round(
                float(
                    panel[(panel["broker"] == broker) & (panel["london_morning"] == inside)][
                        "spread_bp"
                    ].mean()
                ),
                4,
            )
            for broker in ("A", "C")
            for side, inside in (("inside", 1), ("outside", 0))
        }
        | {
            "C_drop": round(
                float(
                    panel[(panel["broker"] == "C") & (panel["london_morning"] == 0)][
                        "spread_bp"
                    ].mean()
                    - panel[(panel["broker"] == "C") & (panel["london_morning"] == 1)][
                        "spread_bp"
                    ].mean()
                ),
                4,
            )
        },
        "planted_realised": {
            broker: {
                "markup_at_arrival": round(float(markup_mean[broker]), 4),
                "aggressive_share": round(float(share[broker]), 4),
                "realised_markup_bp": round(float(markup_mean[broker] * share[broker]), 4),
            }
            for broker in sorted(book)
        },
        "naive": {
            scope: {
                row["broker"]: {
                    "mean": round(float(row["mean"]), 4),
                    "lo": round(float(row["lo"]), 4),
                    "hi": round(float(row["hi"]), 4),
                    "n": int(row["n"]),
                }
                for _, row in ranking[ranking["scope"] == scope].iterrows()
            }
            for scope in ("all", "random")
        },
        "regression": {
            "n": int(adjustment.main.nobs),
            "r2": round(float(adjustment.main.rsquared), 5),
            "resid_sd": round(resid_sd, 4),
            "raw_sd": round(float(panel["is_bp"].std(ddof=1)), 4),
            "terms": {
                row["term"]: {
                    "coef": round(float(row["coef"]), 4),
                    "se": round(float(row["se"]), 4),
                    "p": round(float(row["p"]), 5),
                }
                for _, row in adjustment.table.iterrows()
            },
            "brokers": {
                row["broker"]: {
                    "coef": round(float(row["coef"]), 4),
                    "se": round(float(row["se"]), 4),
                    "lo": round(float(row["lo"]), 4),
                    "hi": round(float(row["hi"]), 4),
                }
                for _, row in adjustment.effects.iterrows()
            },
            "brokers_random": {
                row["broker"]: {
                    "coef": round(float(row["coef"]), 4),
                    "se": round(float(row["se"]), 4),
                    "lo": round(float(row["lo"]), 4),
                    "hi": round(float(row["hi"]), 4),
                }
                for _, row in adjustment.random_effects.iterrows()
            },
        },
        "interaction_summary": interaction_summary,
        "interaction": {
            "terms": {
                row["term"]: {
                    "coef": round(float(row["coef"]), 4),
                    "se": round(float(row["se"]), 4),
                    "p": round(float(row["p"]), 5),
                }
                for _, row in interaction.iterrows()
            },
            "r2": round(float(adjustment.interaction.rsquared), 5),
        },
        "shrinkage": {
            "tau2": round(float(cells["tau2"].iloc[0]), 5),
            "cells": len(cells),
            "smallest_n": int(cells["n"].min()),
            "smallest_se": round(
                float(
                    np.sqrt(
                        cells.sort_values("n")["s2"].iloc[0] / cells.sort_values("n")["n"].iloc[0]
                    )
                )
                if cells["n"].min() > 1
                else float("nan"),
                3,
            ),
            "largest_n": int(cells["n"].max()),
            "extremes": [
                {
                    "broker": row["broker"],
                    "pair": row["pair"],
                    "size_bucket": row["size_bucket"],
                    "n": int(row["n"]),
                    "raw": round(float(row["raw"]), 3),
                    "shrunk": round(float(row["shrunk"]), 3),
                }
                for _, row in cells.reindex(cells["move"].abs().sort_values(ascending=False).index)
                .head(3)
                .iterrows()
            ],
            **{k: round(v, 5) for k, v in shrink.items()},
        },
        "power_gain_0_1": round(
            float(
                1.0
                - power.loc[np.isclose(power["delta_bp"], 0.1), "n_adjusted"].iloc[0]
                / power.loc[np.isclose(power["delta_bp"], 0.1), "n_raw"].iloc[0]
            ),
            3,
        ),
        "power_summary": {
            f"n_{str(row['delta_bp']).replace('.', '_')}": int(row["n_adjusted"])
            for _, row in power.iterrows()
        }
        | {
            f"months_{str(row['delta_bp']).replace('.', '_')}": round(float(row["months"]), 3)
            for _, row in power.iterrows()
        },
        "power": {
            "rows": [
                {
                    "delta_bp": float(row["delta_bp"]),
                    "n_raw": int(row["n_raw"]),
                    "n_adjusted": int(row["n_adjusted"]),
                    "months": round(float(row["months"]), 2),
                }
                for _, row in power.iterrows()
            ]
        },
        "learning": {
            **learning.summary,
            "gap_C_minus_A": round(
                float(learning.summary["true_mean_cost"]["C"])
                - float(learning.summary["true_mean_cost"]["A"]),
                4,
            ),
        },
    }
    tables = {
        "table1": table1,
        "ranking": ranking,
        "regression": adjustment.table,
        "interaction": interaction,
        "cells": cells,
        "power": power,
        "effects": adjustment.effects,
        "effects_random": adjustment.random_effects,
        "learning_curve": learning.curve,
    }
    if dealer is not None:
        dealer_rows, dealer_meta = dealer_table(*dealer)
        best_hit = dealer_rows.loc[dealer_rows["hit_effect"].idxmax()]
        worst_hit = dealer_rows.loc[dealer_rows["hit_effect"].idxmin()]
        results["dealers"] = {
            **dealer_meta,
            "n_wins": int(dealer_rows["wins"].sum()),
            "max_hit_effect": round(float(best_hit["hit_effect"]), 4),
            "max_hit_dealer": str(best_hit["dealer_id"]),
            "min_hit_effect": round(float(worst_hit["hit_effect"]), 4),
            "min_hit_dealer": str(worst_hit["dealer_id"]),
            "max_abs_margin_32nds": round(float(dealer_rows["margin_effect_32nds"].abs().max()), 4),
            "rows": dealer_rows.to_dict(orient="records"),
        }
        tables["dealers"] = dealer_rows
    return results, tables


def _print_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Print every table of the note to stdout, in the order the note uses them."""
    print("\nTable 1 - what the simulator planted (config/brokers.yaml)")
    print(tables["table1"].to_string(index=False))
    print("\nNaive ranking, mean IS (bp) with 95% bootstrap CI")
    print(tables["ranking"].to_string(index=False, float_format="%.3f"))
    print("\nTable 2 - main difficulty-adjusted regression (clustered by day)")
    print(tables["regression"].to_string(index=False, float_format="%.4f"))
    print("\nBroker-by-size and broker-by-session interactions")
    print(tables["interaction"].to_string(index=False, float_format="%.4f"))
    print("\nPower: orders per broker for 80% power")
    print(tables["power"].to_string(index=False))
    if "dealers" in tables:
        print("\nTable 3 - dealer fixed effects (logit hit rate, OLS margin)")
        print(tables["dealers"].to_string(index=False, float_format="%.4f"))


def main() -> None:
    """Build every figure and table of the note and write results.json."""
    book = load_brokers()
    panel = load_panel()
    results, tables = collect_results(panel, book, load_dealer_inputs())

    with figures_env():
        figure_naive(tables["ranking"], panel, book)
        figure_effects(tables["effects"], tables["effects_random"], panel, book)
        figure_shrinkage(tables["cells"])
        figure_power(tables["power"], int(results["meta"]["orders_per_broker"]))
        figure_learning(tables["learning_curve"])

    RESULTS_PATH.write_text(json.dumps(_round(results), indent=2, allow_nan=False) + "\n")
    _print_tables(tables)
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Wrote five figures under {FIG_DIR}")


if __name__ == "__main__":
    main()
