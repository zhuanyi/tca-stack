#!/usr/bin/env python3
"""Replicate the FX fill-rate claims of docs/fx.md over several master seeds.

Usage:
    TCA_DATA_RAW=/path/to/data/raw nice -n 10 python scripts/fx_fill_rate_seeds.py \\
        --seeds 20250303,20250310,20250317 --out docs/tables/fx_fill_rate_seeds.csv

For every seed the full Dukascopy sample is re-simulated three ways, and one CSV row is written:

    baseline      the calibrated passive hazard, the run the committed tables come from
    crossing      lambda_fill = 0 for every pair, so only crossings and timeouts fill
    haircut       crossings dropped at a probability solved per seed to put the pooled
                  crossing-only fill rate at 40%, plus the paired TWAP implementation
                  shortfall change against the baseline of the same seed. The solver stops
                  at a 0.05 point tolerance or after a fixed number of runs, and the
                  achieved rate is what haircut_pct records, so it can sit a few tenths of
                  a point off 40.0

The sample needs the cached Dukascopy day files under $TCA_DATA_RAW/fx and the committed
fx_bars_1m table (read-only). Nothing is written to the repository except --out: the simulated
tables stay in memory and TCA_DATA_DIR points at a scratch folder, so no committed table can be
overwritten. Runs are deterministic, so re-running with the same seeds reproduces the CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRS = ("EURUSD", "USDJPY", "GBPUSD", "AUDUSD")
TARGET_PCT = 40.0
TOLERANCE_PCT = 0.05
MAX_HAIRCUT_RUNS = 6


def _configure(data_dir: Path, raw_dir: Path) -> None:
    """Point the package at the scratch data folder and the read-only raw ticks."""
    os.environ["TCA_PROFILE"] = "full"
    os.environ["TCA_DATA_DIR"] = str(data_dir)
    os.environ["TCA_DATA_RAW"] = str(raw_dir)
    sys.path.insert(0, str(REPO_ROOT))


def _passive_metrics(result) -> dict:
    """Pooled and per-pair passive fill percentage of one run."""
    tables = result.tables()
    pair_of = tables["fx_orders"].set_index("order_id")["pair"]
    passive = tables["fx_children"]
    passive = passive[passive["intent"] == "PASSIVE"]
    filled = passive["child_id"].isin(tables["fx_fills"]["child_id"])
    pairs = passive["order_id"].map(pair_of)
    out = {
        "pct": 100.0 * float(filled.sum()) / max(1, len(passive)),
        "n_children": len(passive),
    }
    for pair in PAIRS:
        mask = pairs == pair
        n = int(mask.sum())
        out[pair] = 100.0 * float(filled[mask].sum()) / n if n else float("nan")
    return out


def _tca_tables(result) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill-level and parent-level TCA of a run."""
    from tca.fx import analysis

    tables = result.tables()
    tfill = analysis.tca_fill(tables["fx_orders"], tables["fx_fills"], tables["fx_markouts"])
    parent = analysis.tca_parent(
        tables["fx_orders"],
        tables["fx_fills"],
        tables["fx_fill_attempts"],
        tables["fx_children"],
        tfill,
    )
    return tfill, parent


def _twap_summary(parent: pd.DataFrame) -> tuple[int, float]:
    """Row count and mean implementation shortfall of the TWAP orders."""
    twap = parent[parent["style"] == "TWAP"]
    return len(twap), float(twap["is_bp"].mean())


@contextmanager
def _crossing_haircut(seed: int, pi: float):
    """Keep a crossing fill with probability pi for the duration of the block.

    Only the replay's scan is patched, so the block must be used with an overrides object that
    fixes lambda_fill: the passive calibration in tca.fx.simulate bound the original scan at import
    time and would not see the patch.

    A dropped crossing falls back to a passive scan over the whole posting window with the cross
    test removed, which is the counterfactual "that crossing did not happen": the child keeps
    resting (and with lambda_fill = 0 never fills passively, so it times out and goes aggressive).
    The at_limit test is the original's, same tolerance and same abs() form, and this copy has to
    track tca/fx/passive.scan_passive if that function changes. The draw is keyed on
    the seed, the posting start and the side, so a run is reproducible and a sweep is monotone in
    pi; postings of one seed share a draw only when their start microsecond and side coincide,
    which the continuous arrival times make vanishingly unlikely.
    """
    from tca.fx import passive as passive_mod
    from tca.fx import replay
    from tca.fx.ticks import US_PER_S

    original = replay.scan_passive

    def scan_no_cross(day, side, limit, start_us, end_us, impact_at, tol):
        first = int(day.index_at(start_us))
        stop = max(int(np.searchsorted(day.t_us, end_us, side="right")), first + 1)
        idx = np.arange(first, stop)
        knots = np.concatenate(([start_us], day.t_us[first + 1 : stop]))
        ends = np.concatenate((knots[1:], [end_us]))
        impact = impact_at(knots)
        own = (day.bid[idx] if side > 0 else day.ask[idx]) + impact
        at_limit = np.abs(own - limit) < tol
        return passive_mod.PassiveScan(knots, (ends - knots) / US_PER_S, at_limit, idx, None, None)

    def patched(day, side, limit, start_us, end_us, impact_at, tol):
        scan = original(day, side, limit, start_us, end_us, impact_at, tol)
        if scan.cross_us is None:
            return scan
        draw = np.random.default_rng([seed, 99, int(start_us) % (2**31), int(side) + 2]).random()
        if draw <= pi:
            return scan
        return scan_no_cross(day, side, limit, start_us, end_us, impact_at, tol)

    replay.scan_passive = patched
    try:
        yield
    finally:
        replay.scan_passive = original


def _solve_haircut(runner, target_pct: float = TARGET_PCT) -> tuple[float, pd.Series]:
    """Probability of keeping a crossing that puts the pooled crossing-only rate at the target.

    The runner maps a kept-crossing probability to the passive metrics of that run. The rate rises
    monotonically with the probability, so the search probes the crossing-only run first and then
    closes in with secant steps inside the bracket it has established. Returns the probability and
    the metrics it produced.
    """
    cache: dict[float, pd.Series] = {}

    def run(pi: float) -> pd.Series:
        key = round(pi, 6)
        if key not in cache:
            cache[key] = runner(key)
        return cache[key]

    high = run(1.0)
    if high["pct"] <= target_pct:
        return 1.0, high
    low, low_pi = None, 0.0
    high_pi = 1.0
    pi = min(0.999, target_pct / high["pct"])
    for _ in range(MAX_HAIRCUT_RUNS):
        current = run(pi)
        if abs(current["pct"] - target_pct) <= TOLERANCE_PCT:
            return pi, current
        if current["pct"] > target_pct:
            high, high_pi = current, pi
        else:
            low, low_pi = current, pi
        if low is not None and high["pct"] != low["pct"]:
            slope = (high["pct"] - low["pct"]) / (high_pi - low_pi)
            nxt = pi + (target_pct - current["pct"]) / slope
        else:
            nxt = target_pct / high["pct"]
        step = 0.5 * (high_pi - low_pi) if low is not None else 0.2
        nxt = float(np.clip(nxt, pi - step, pi + step))
        nxt = float(np.clip(nxt, 1e-4, 0.9999))
        if abs(nxt - pi) < 1e-4:
            return pi, current
        pi = nxt
    return pi, run(pi)


def _seed_row(cfg, seed: int, source, market) -> dict:
    """One CSV row: baseline, crossing-only and 40%-haircut numbers for one master seed."""
    from tca.fx.simulate import Overrides, run

    zeros = Overrides(lambda_fill={p: 0.0 for p in cfg.pairs.order})
    baseline = run(cfg, seed, source, market)
    base_passive = _passive_metrics(baseline)
    _, base_parent = _tca_tables(baseline)
    twap_n, twap_is = _twap_summary(base_parent)

    crossing = run(cfg, seed, source, market, zeros)
    cross_passive = _passive_metrics(crossing)
    _, cross_parent = _tca_tables(crossing)

    def haircut(pi: float) -> pd.Series:
        with _crossing_haircut(seed, pi):
            return pd.Series(_passive_metrics(run(cfg, seed, source, market, zeros)))

    pi, cut_passive = _solve_haircut(haircut)
    with _crossing_haircut(seed, pi):
        cut = run(cfg, seed, source, market, zeros)
    _, cut_parent = _tca_tables(cut)
    cut_twap_n, cut_twap_is = _twap_summary(cut_parent)

    joined = base_parent[base_parent["style"] == "TWAP"][["order_id", "is_bp"]].merge(
        cut_parent[cut_parent["style"] == "TWAP"][["order_id", "is_bp"]],
        on="order_id",
        suffixes=("_base", "_cut"),
    )
    delta = joined["is_bp_cut"] - joined["is_bp_base"]
    if len(delta) < 2:
        print(f"seed {seed}: {len(delta)} paired TWAP orders, delta left empty", file=sys.stderr)
        delta_mean, delta_se = float("nan"), float("nan")
    else:
        delta_mean = round(float(delta.mean()), 4)
        delta_se = round(float(delta.std(ddof=1) / np.sqrt(len(delta))), 4)
    row = {
        "seed": seed,
        "n_orders": len(baseline.orders),
        "n_passive_children": base_passive["n_children"],
        "passive_pct": round(base_passive["pct"], 3),
        "twap_n": twap_n,
        "twap_is_bp": round(twap_is, 4),
        "crossing_pct": round(cross_passive["pct"], 3),
        "crossing_twap_is_bp": round(_twap_summary(cross_parent)[1], 4),
        "haircut_pi": round(pi, 4),
        "haircut_pct": round(float(cut_passive["pct"]), 3),
        "haircut_twap_n": cut_twap_n,
        "haircut_twap_is_bp": round(cut_twap_is, 4),
        "haircut_delta_bp": delta_mean,
        "haircut_delta_se": delta_se,
    }
    for pair in PAIRS:
        row[f"passive_{pair}"] = round(base_passive[pair], 3)
        row[f"crossing_{pair}"] = round(cross_passive[pair], 3)
        row[f"haircut_{pair}"] = round(float(cut_passive[pair]), 3)
    row["lambda_audusd"] = round(float(baseline.lambda_fill.get("AUDUSD", 0.0)), 6)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seeds",
        default="20250303,20250310,20250317,20250324,20250331,20250407",
        help="Comma-separated master seeds.",
    )
    parser.add_argument("--out", default="docs/tables/fx_fill_rate_seeds.csv", help="CSV to write.")
    parser.add_argument("--bars", default="data/derived", help="Folder holding fx_bars_1m.")
    parser.add_argument(
        "--raw", default=os.environ.get("TCA_DATA_RAW", "data/raw"), help="Folder holding fx/."
    )
    parser.add_argument(
        "--scratch", default="", help="Scratch TCA_DATA_DIR; default a new temp folder."
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        parser.error("--seeds is empty")
    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="tca-fx-seeds-"))
    scratch.mkdir(parents=True, exist_ok=True)
    _configure(scratch, Path(args.raw).resolve())

    from tca.core import storage
    from tca.fx import paths, pipeline
    from tca.fx.config import load_config
    from tca.fx.ticks import RawTickSource

    repo_data = (REPO_ROOT / "data").resolve()
    derived = paths.derived_dir().resolve()
    if scratch.resolve() == repo_data or repo_data in scratch.resolve().parents:
        print("refusing to run inside the repository data folder", file=sys.stderr)
        return 2
    cfg = load_config(None)
    bars = storage.read_table("fx_bars_1m", Path(args.bars).resolve())
    market, _ = pipeline.build_market(cfg, bars)
    source = RawTickSource()
    print(
        f"{len(bars):,} bars, {len(source.days(cfg.pairs.order[0]))} days, "
        f"{len(seeds)} seeds, scratch {derived}",
        flush=True,
    )

    rows = []
    for seed in seeds:
        started = time.time()
        row = _seed_row(cfg, seed, source, market)
        rows.append(row)
        print(json.dumps({**row, "seconds": round(time.time() - started, 1)}), flush=True)

    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, float_format="%.4f")
    print(f"\n{len(frame)} seeds -> {out}", flush=True)
    for column in ("passive_pct", "crossing_pct", "haircut_delta_bp"):
        values = frame[column].dropna()
        print(
            f"{column}: mean {values.mean():.3f}, min {values.min():.3f}, max {values.max():.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
