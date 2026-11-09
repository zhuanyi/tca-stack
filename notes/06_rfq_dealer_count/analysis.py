"""Note 3: how many dealers to put in competition, in the Module 1 UST RFQ simulator.

`tca.ust.simulate.run` is used as a library: every grid cell is one simulator run with an
explicit RFQ list (so the size, tenor and run status of each ticket are fixed, not drawn) and
the dealer-policy override `n_dealers`. Each programme sends two RFQs of the same size, side
and security 20 minutes apart; the first RFQ's cost shows the competition benefit, the second
RFQ's arrival shift shows the leakage cost, and `programme_cost` combines them.

Usage (from the repository root)::

    uv run python notes/06_rfq_dealer_count/analysis.py                       # figures + tables
    uv run python notes/06_rfq_dealer_count/analysis.py --run --part 0 --parts 2
    uv run python notes/06_rfq_dealer_count/analysis.py --run --collect
    uv run python notes/06_rfq_dealer_count/analysis.py --reduced             # CI-sized grid

`--run --part k --parts m` writes `tables/parts/part-k.csv` (one row per cell and seed) so
the grid can be split over processes; `--collect` merges the parts, draws the figures and
writes `tables/`. Without `--run` the committed `tables/grid_runs.csv` is read, so the
figures regenerate in seconds without the 20-minute grid. `--reduced` runs the CI-sized grid
(N in {1, 3, 7}, 10y on-the-run, one leakage pair, one seed) in memory and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import integrate, stats

from tca.core.calendar import NY, local_to_utc
from tca.plotting import style
from tca.ust import simulate
from tca.ust.config import SimConfig, load_config
from tca.ust.inputs import Inputs, load_inputs

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TCA_FIGURES_DIR", str(HERE / "figures"))

NOTE = "notes/06_rfq_dealer_count"
TABLES = HERE / "tables"
PARTS = TABLES / "parts"
GRID_RUNS = TABLES / "grid_runs.csv"

N_GRID = (1, 2, 3, 4, 5, 7, 10)
SIZES = (5, 25, 50, 100, 250)
STATUSES = ("OTR", "OLD")
LEAKAGES = (0.0, 0.5, 1.0, 2.0)
TENORS = (2, 10, 30)
SEEDS = (20250303, 20250304, 20250305, 20250306, 20250307, 20250308, 20250309, 20250310)
DEFAULT_LEAKAGE = 1.0

PROGRAMMES = 400
DAYS_SAMPLED = 10
SECOND_RFQ_MINUTES = 20.0
SCHEDULE_SEED = 7

#: First RFQ send window, seconds after the 07:00 session start (09:30 to 14:30 New York), so
#: the second RFQ lands before the 15:00 close marks.
_SEND_FROM = 2.5 * 3600.0
_SEND_SPAN = 5.0 * 3600.0

_FIGURE_SOURCE = (
    "Simulated UST RFQs, Module 1 simulator; 400 programmes per cell, 10 of 63 business days"
)


def order_statistic_max(n: int) -> float:
    """E[max of `n` iid standard normals], from the tail integral over `[0, inf)`."""
    def integrand(x: float) -> float:
        tail = 1.0 - stats.norm.cdf(x)
        return 1.0 - stats.norm.cdf(x) ** n - tail**n

    value, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
    return float(value)


def sample_days(cfg: SimConfig, inputs: Inputs, n: int = DAYS_SAMPLED) -> list:
    """`n` business days spaced evenly over the simulated period."""
    days = inputs.business_days(cfg.period.start, cfg.period.end)
    picked = sorted({int(i) for i in np.linspace(0, len(days) - 1, n).round()})
    return [days[i] for i in picked]


def schedule(
    inputs: Inputs, days, tenor: int, status: str, size: float, is_axe: bool = False
) -> pd.DataFrame:
    """The RFQ list of one cell: `PROGRAMMES` pairs, spread over the sampled days.

    Each programme is a first RFQ and a second one `SECOND_RFQ_MINUTES` later in the same
    security and direction at the same size. The schedule depends only on the cell's tenor,
    status and size, so every cell of a size sees the same tickets (common random numbers).
    """
    rng = np.random.default_rng(
        [SCHEDULE_SEED, tenor, 1 if status == "OTR" else 2, int(size), int(is_axe)]
    )
    uni = inputs.universe
    per_day = PROGRAMMES // len(days)
    rows = []
    pid = 0
    for d in days:
        sel = uni[(uni["date"] == d) & (uni["tenor"] == tenor) & (uni["run_status"] == status)]
        if len(sel) == 0:
            continue
        cusip = str(sel["cusip"].iloc[0])
        for _ in range(per_day):
            ts = local_to_utc(d, dtime(9, 30), NY) + pd.Timedelta(
                seconds=float(rng.uniform(0.0, _SEND_SPAN))
            )
            side = int(rng.choice([-1, 1]))
            for which in (1, 2):
                rows.append(
                    (
                        pid,
                        which,
                        ts if which == 1 else ts + pd.Timedelta(minutes=SECOND_RFQ_MINUTES),
                        cusip,
                        side,
                        float(size),
                    )
                )
            pid += 1
    frame = pd.DataFrame(
        rows, columns=["pair_id", "which", "ts_send_utc", "cusip", "side", "qty_mm"]
    )
    frame["ts_key"] = pd.to_datetime(frame["ts_send_utc"], utc=True)
    return frame.sort_values(["ts_key", "pair_id", "which"], kind="stable").reset_index(drop=True)


def _pairs(tables: dict, sched: pd.DataFrame) -> pd.DataFrame:
    """Simulator RFQ rows matched to the schedule, with per-RFQ cost and markout columns."""
    rfq = tables["ust_rfq"].copy()
    rfq["ts_key"] = pd.to_datetime(rfq["ts_send_utc"], utc=True)
    cols = ["ts_key", "cusip", "side", "qty_mm"]
    merged = rfq.merge(
        sched.rename(columns={"pair_id": "pair", "which": "leg"})[cols + ["pair", "leg"]],
        on=cols,
        how="inner",
        validate="one_to_one",
    )
    marks = tables["ust_rfq_markouts"][["rfq_id", "mid_px_0s", "mid_px_5m", "mid_px_30m"]]
    merged = merged.merge(marks, on="rfq_id", how="left")
    merged["cost"] = merged["side"] * (merged["best_px"] - merged["mid_send"]) * 32.0
    merged["done"] = merged["status"].eq("DONE")
    merged["exec_cost"] = merged["side"] * (merged["exec_px"] - merged["mid_send"]) * 32.0
    merged["markout_5m"] = np.where(
        merged["done"], merged["side"] * (merged["mid_px_5m"] - merged["exec_px"]) * 32.0, np.nan
    )
    merged["markout_30m"] = np.where(
        merged["done"], merged["side"] * (merged["mid_px_30m"] - merged["exec_px"]) * 32.0, np.nan
    )
    merged["drift_30m"] = merged["side"] * (merged["mid_px_30m"] - merged["mid_px_0s"]) * 32.0
    merged["margin"] = np.where(
        merged["done"] & merged["cover_px"].notna(),
        merged["side"] * (merged["cover_px"] - merged["exec_px"]) * 32.0,
        np.nan,
    )
    return merged


def cell_metrics(tables: dict, sched: pd.DataFrame, axe_only: bool = False) -> dict:
    """Metrics of one simulator run: means over its programmes, with pair-level standard errors.

    With `axe_only` the metrics cover only the programmes whose first RFQ found a favourable
    axe (the single-dealer axe trades), so Table 1 compares axed trading with competition.
    """
    m = _pairs(tables, sched)
    axe_share = float(m.loc[m["leg"] == 1, "is_axe"].mean()) if "is_axe" in m else 0.0
    if axe_only:
        m = m[m["is_axe"].astype(bool)]
    first = m[m["leg"] == 1].set_index("pair")
    second = m[m["leg"] == 2].set_index("pair")
    both = first.index.intersection(second.index)
    shift = (
        first.loc[both, "side"] * (second.loc[both, "mid_send"] - first.loc[both, "mid_send"]) * 32.0
    )
    programme = (first.loc[both, "cost"] + shift + second.loc[both, "cost"]) / 2.0
    cost1 = first["cost"]
    out = {
        "exec_cost_1": float(cost1.mean()),
        "exec_cost_1_done": float(first.loc[first["done"], "exec_cost"].mean()),
        "exec_cost_2": float(second["cost"].mean()),
        "arrival_shift_2": float(shift.mean()),
        "programme_cost": float(programme.mean()),
        "markout_5m_1": float(first["markout_5m"].mean()),
        "markout_30m_1": float(first["markout_30m"].mean()),
        "drift_30m_1": float(first["drift_30m"].mean()),
        "winner_margin": float(first["margin"].mean()),
        "response_rate": float((m["n_responses"] / m["n_dealers"]).mean()),
        "fill_rate": float(m["done"].mean()),
        "pass_rate": float(m["status"].eq("PASSED").mean()),
        "no_response_rate": float(m["status"].eq("NO_RESPONSE").mean()),
        "n_pairs": int(len(both)),
        "n_quoted_1": int(first["best_px"].notna().sum()),
        "n_done_1": int(first["done"].sum()),
        "axe_share": axe_share,
    }
    for name, series in (
        ("exec_cost_1", cost1),
        ("arrival_shift_2", shift),
        ("programme_cost", programme),
    ):
        sd = float(series.std(ddof=1))
        out[f"sd_{name}"] = sd
        out[f"se_{name}"] = sd / np.sqrt(len(series))
    return out


def run_cell(
    cfg: SimConfig,
    inputs: Inputs,
    days,
    sched: pd.DataFrame,
    seed: int,
    n_dealers: int | None,
    leakage_scale: float,
    force_axe_selection: bool = False,
    widening_scale: float | None = None,
) -> dict:
    """One simulator run of one cell and its metrics."""
    frame = sched[["ts_key", "cusip", "side", "qty_mm"]].rename(columns={"ts_key": "ts_send_utc"})
    frame["n_dealers"] = n_dealers if n_dealers is not None else 1
    frame["is_axe"] = force_axe_selection
    overrides = simulate.Overrides(
        rfqs=frame,
        days=days,
        leakage_scale=leakage_scale,
        competition_widening_scale=widening_scale,
        n_dealers=None if force_axe_selection else n_dealers,
    )
    tables = simulate.run(cfg, seed, overrides, inputs)
    return cell_metrics(tables, sched, axe_only=force_axe_selection)


def cell_list(seeds=SEEDS, kinds=("grid",)):
    """Every (cell, seed) run of the grid, as dicts."""
    if "grid" not in kinds:
        yield from _axe_cells(seeds)
        return
    for tenor in TENORS:
        for status in STATUSES:
            for size in SIZES:
                for n in N_GRID:
                    for leak in LEAKAGES:
                        for seed in seeds:
                            yield dict(
                                kind="grid",
                                tenor=tenor,
                                status=status,
                                size=float(size),
                                n_dealers=n,
                                leakage_scale=float(leak),
                                seed=int(seed),
                            )
    if "axe" in kinds:
        yield from _axe_cells(seeds)


def _axe_cells(seeds=SEEDS):
    """The single-dealer axe runs and their three-dealer comparison."""
    for size in SIZES:
            for n, forced in ((1, True), (3, False)):
                for seed in seeds:
                    yield dict(
                        kind="axe",
                        tenor=10,
                        status="OTR",
                        size=float(size),
                        n_dealers=n,
                        leakage_scale=DEFAULT_LEAKAGE,
                        seed=int(seed),
                        force_axe=forced,
                    )


def reduced_cells(seed: int = SEEDS[0]):
    """The CI-sized grid: N in {1, 3, 7}, 10y on-the-run, two leakage levels, one seed.

    The zero-leakage cells switch the dealer widening off: the invariant they check is the
    competition benefit alone, and with the configured widening (0.03 32nds per dealer beyond
    three) the N = 7 cell costs about 0.02 more than N = 3, which is the flattening Figure 1
    panel (b) reports, not noise.
    """
    for n in (1, 3, 7):
        for leak, widening in ((0.0, 0.0), (2.0, None)):
            yield dict(
                kind="grid",
                tenor=10,
                status="OTR",
                size=50.0,
                n_dealers=n,
                leakage_scale=leak,
                seed=seed,
                widening_scale=widening,
            )


def run_reduced(cfg: SimConfig, inputs: Inputs, seed: int = SEEDS[0]) -> pd.DataFrame:
    """Run the CI-sized grid in memory and return one row per run (used by the tests)."""
    days = sample_days(cfg, inputs)
    out = []
    for cell in reduced_cells(seed):
        sched = schedule(inputs, days, cell["tenor"], cell["status"], cell["size"])
        metrics = run_cell(
            cfg,
            inputs,
            days,
            sched,
            cell["seed"],
            cell["n_dealers"],
            cell["leakage_scale"],
            widening_scale=cell.get("widening_scale"),
        )
        out.append({**cell, **metrics, "days": len(days), "programmes": PROGRAMMES})
    return pd.DataFrame(out)


_KEYS = ("kind", "tenor", "status", "size", "n_dealers", "leakage_scale", "seed")


def _dump_part(path: Path, existing: pd.DataFrame | None, rows: list) -> pd.DataFrame:
    """Write `rows` on top of what the part file already holds, one row per run."""
    frame = pd.DataFrame(rows)
    if existing is not None and not existing.empty:
        frame = pd.concat([existing, frame], ignore_index=True)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=_KEYS, keep="last")
    frame.to_csv(path, index=False)
    return frame


def _key(row: dict) -> tuple:
    """Identity of a run: the cell and its seed."""
    return tuple(
        round(float(row[k]), 6) if isinstance(row[k], float) else row[k]
        for k in _KEYS
        if k in row
    )


def run_part(cfg: SimConfig, inputs: Inputs, part: int, parts: int) -> Path:
    """Run the grid cells whose index is `part` modulo `parts` and write them to a CSV."""
    PARTS.mkdir(parents=True, exist_ok=True)
    path = PARTS / f"part-{part}-of-{parts}.csv"
    done = set()
    existing = pd.read_csv(path) if path.exists() else None
    if GRID_RUNS.exists() or any(PARTS.glob("part-*-of-*.csv")):
        done = {_key(row) for row in load_runs().to_dict("records")}
    days_by_key: dict = {}
    started = time.time()
    rows = []
    for index, cell in enumerate(cell_list(kinds=("grid", "axe"))):
        if index % parts != part:
            continue
        if _key(cell) in done:
            continue
        key = (cell["tenor"], cell["status"], cell["size"], bool(cell.get("force_axe", False)))
        if key not in days_by_key:
            days_by_key[key] = sample_days(cfg, inputs)
        days = days_by_key[key]
        sched = schedule(inputs, days, cell["tenor"], cell["status"], cell["size"], key[3])
        metrics = run_cell(
            cfg,
            inputs,
            days,
            sched,
            cell["seed"],
            cell["n_dealers"],
            cell["leakage_scale"],
            force_axe_selection=key[3],
        )
        rows.append(
            {
                **{k: v for k, v in cell.items() if k != "force_axe"},
                "days": len(days),
                "programmes": PROGRAMMES,
                **metrics,
            }
        )
        if len(rows) % 100 == 0:
            _dump_part(path, existing, rows)
            print(f"part {part}: {len(rows)} runs, {time.time() - started:.0f}s", flush=True)
    frame = _dump_part(path, existing, rows)
    meta = {
        "part": part,
        "parts": parts,
        "runs_this_run": len(rows),
        "runs_in_part": len(frame),
        "seconds": round(time.time() - started, 1),
    }
    (PARTS / f"part-{part}-of-{parts}.json").write_text(json.dumps(meta, indent=1) + chr(10))
    print(json.dumps(meta))
    return path


def collect(cfg: SimConfig, inputs: Inputs) -> pd.DataFrame:
    """Merge the part files into `tables/grid_runs.csv` and write the run-time record."""
    files = sorted(PARTS.glob("part-*-of-*.csv"))
    if not files:
        raise SystemExit(f"no part files in {PARTS}; run with --run --part k --parts m first")
    frames = [pd.read_csv(GRID_RUNS)] if GRID_RUNS.exists() else []
    frames += [pd.read_csv(f) for f in files]
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset=_KEYS, keep="last")
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(GRID_RUNS, index=False)
    metas = [json.loads(p.read_text()) for p in sorted(PARTS.glob("part-*-of-*.json"))]
    seconds = max((m["seconds"] for m in metas), default=0.0)
    history = []
    if (TABLES / "runtime.json").exists():
        history = json.loads((TABLES / "runtime.json").read_text()).get("history", [])
    entry = {
        "seconds": seconds,
        "runs_this_batch": [m.get("runs_this_run", m.get("runs")) for m in metas],
        "seeds": sorted({int(s) for f in files for s in pd.read_csv(f, usecols=["seed"])["seed"]}),
    }
    if not any(
        h.get("seeds") == entry["seeds"] and h.get("runs_this_batch") == entry["runs_this_batch"]
        for h in history
    ):
        history.append(entry)
    (TABLES / "runtime.json").write_text(
        json.dumps(
            {
                "history": history,
                "seconds_this_batch": seconds,
                "parts": len(metas),
                "runs": int(len(frame)),
                "cells": int(
                    frame[frame["kind"] == "grid"]
                    .groupby(["tenor", "status", "size", "n_dealers", "leakage_scale"])
                    .ngroups
                ),
                "axe_cells": int(
                    frame[frame["kind"] == "axe"]
                    .groupby(["tenor", "status", "size", "n_dealers", "leakage_scale"])
                    .ngroups
                ),
                "runs_this_batch": [m.get("runs_this_run", m.get("runs")) for m in metas],
                "runs_in_part": [m.get("runs_in_part") for m in metas],
                "seeds": sorted(int(s) for s in frame["seed"].unique()),
                "programmes_per_cell": PROGRAMMES,
                "days_sampled": DAYS_SAMPLED,
            },
            indent=1,
        )
        + "\n"
    )
    print(
        f"collected {len(frame)} runs from {len(files)} parts; this batch took {seconds:.0f}s "
        "wall clock on the slowest part"
    )
    return frame


def load_runs() -> pd.DataFrame:
    """The committed per-run grid, or the part files when the merge has not run yet."""
    if GRID_RUNS.exists():
        return pd.read_csv(GRID_RUNS)
    files = sorted(PARTS.glob("part-*-of-*.csv"))
    if not files:
        raise SystemExit(f"{GRID_RUNS} is missing; run with --run --collect first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def cells(runs: pd.DataFrame, kind: str = "grid") -> pd.DataFrame:
    """Mean over seeds of every cell, with the across-seed spread."""
    grid = runs[runs["kind"] == kind]
    keys = ["tenor", "status", "size", "n_dealers", "leakage_scale"]
    metrics = [
        "exec_cost_1",
        "exec_cost_2",
        "arrival_shift_2",
        "programme_cost",
        "markout_5m_1",
        "markout_30m_1",
        "drift_30m_1",
        "winner_margin",
        "response_rate",
        "fill_rate",
        "pass_rate",
        "no_response_rate",
        "axe_share",
    ]
    mean = grid.groupby(keys)[metrics].mean()
    spread = grid.groupby(keys)[metrics].std(ddof=1)
    count = grid.groupby(keys)["seed"].nunique()
    out = mean.add_suffix("_mean").join(spread.add_suffix("_sd")).reset_index()
    out["n_seeds"] = count.to_numpy()
    out["se_seeds_programme_cost"] = out["programme_cost_sd"] / np.sqrt(out["n_seeds"])
    return out.sort_values(keys, kind="stable").reset_index(drop=True)


def _focus(cells_frame: pd.DataFrame, **kw) -> pd.DataFrame:
    sel = cells_frame
    for key, value in kw.items():
        if isinstance(value, float):
            sel = sel[np.isclose(sel[key], value)]
        else:
            sel = sel[sel[key] == value]
    return sel.sort_values("n_dealers", kind="stable")


def order_statistics() -> pd.DataFrame:
    return pd.DataFrame(
        {"n_dealers": list(N_GRID), "emax": [order_statistic_max(n) for n in N_GRID]}
    )


def fitted_dispersion(cells_frame: pd.DataFrame) -> pd.DataFrame:
    """`s` per size from the N = 1 and N = 2 cells of 10y on-the-run at zero leakage."""
    zero = _focus(cells_frame, tenor=10, status="OTR", leakage_scale=0.0)
    emax = {row.n_dealers: row.emax for row in order_statistics().itertuples()}
    rows = []
    for size, group in zero.groupby("size"):
        one = group[group["n_dealers"] == 1]["exec_cost_1_mean"].iloc[0]
        two = group[group["n_dealers"] == 2]["exec_cost_1_mean"].iloc[0]
        rows.append(dict(size=float(size), s=(one - two) / emax[2], cost_n1=one))
    return pd.DataFrame(rows)


def optimal_n(runs: pd.DataFrame, leakage: float, tenor: int, status: str) -> pd.DataFrame:
    """Argmin `programme_cost` over N for every size, at one leakage level and segment.

    The tie flag compares the best and the next-best N on the paired per-seed differences
    (the cells share their RFQ schedule and market draws), at 5% two-sided.
    """
    sel = runs[
        (runs["kind"] == "grid")
        & (runs["tenor"] == tenor)
        & (runs["status"] == status)
        & np.isclose(runs["leakage_scale"], leakage)
    ]
    rows = []
    for size, group in sel.groupby("size"):
        per_seed = group.pivot(index="seed", columns="n_dealers", values="programme_cost")
        means = per_seed.mean().sort_values()
        best, second = int(means.index[0]), int(means.index[1])
        paired = per_seed[best] - per_seed[second]
        se = float(paired.std(ddof=1) / np.sqrt(len(paired)))
        critical = float(stats.t.ppf(0.975, max(len(paired) - 1, 1)))
        rows.append(
            dict(
                size=float(size),
                optimal_n=best,
                cost=float(means.iloc[0]),
                second_n=second,
                second_cost=float(means.iloc[1]),
                se=se,
                tie=bool(abs(paired.mean()) < critical * se),
            )
        )
    return pd.DataFrame(
        rows,
        columns=["size", "optimal_n", "cost", "second_n", "second_cost", "se", "tie"],
    )


def figure_competition(cells_frame: pd.DataFrame) -> Path:
    """Figure 1: cost falls with N at zero leakage; the order-statistics fit and the residual."""
    zero = _focus(cells_frame, tenor=10, status="OTR", leakage_scale=0.0)
    disp = fitted_dispersion(cells_frame).set_index("size")
    emax = order_statistics().set_index("n_dealers")["emax"]
    style.apply(hero=True)
    fig, axes = style.figure(
        "Competition alone: more dealers, tighter quotes, then widening",
        f"10y on-the-run, leakage_scale = 0, {len(SEEDS)} seeds; cost vs arrival mid, 32nds",
        _FIGURE_SOURCE,
        nrows=1,
        ncols=2,
    )
    fig.subplots_adjust(wspace=0.34)
    ax = axes[0]
    sizes = sorted(zero["size"].unique())
    handles = []
    for i, size in enumerate(sizes):
        group = zero[np.isclose(zero["size"], size)].sort_values("n_dealers")
        colour = style.PALETTE[i % len(style.PALETTE)]
        s = float(disp.loc[size, "s"])
        (line,) = ax.plot(
            group["n_dealers"],
            group["exec_cost_1_mean"],
            marker="o",
            color=colour,
            label=f"${int(size)}mm (s={s:.2f})",
        )
        ax.errorbar(
            group["n_dealers"],
            group["exec_cost_1_mean"],
            yerr=1.96 * group["exec_cost_1_sd"] / np.sqrt(len(SEEDS) * PROGRAMMES),
            fmt="none",
            ecolor=colour,
            capsize=3,
        )
        handles.append(line)
        grid_n = np.array(N_GRID, dtype=float)
        predicted = float(disp.loc[size, "cost_n1"]) - s * emax.loc[grid_n].to_numpy()
        ax.plot(grid_n, predicted, linestyle="--", linewidth=1.2, color=colour, alpha=0.8)
    ax.set_xlabel("Dealers asked per RFQ")
    ax.set_ylabel("Cost vs arrival mid (32nds)")
    ax.set_xticks(list(N_GRID))
    ax.text(
        0.03,
        0.05,
        "dashed: order statistics, s fitted to the N=1 and N=2 cells",
        transform=ax.transAxes,
        fontsize=9,
        color="#595959",
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.862),
        ncol=len(handles),
        fontsize=9,
        title=None,
    )

    ax = axes[1]
    for i, size in enumerate(sizes):
        group = zero[np.isclose(zero["size"], size)].sort_values("n_dealers")
        s = float(disp.loc[size, "s"])
        base = float(disp.loc[size, "cost_n1"])
        ns = group["n_dealers"].to_numpy(dtype=float)
        residual = group["exec_cost_1_mean"].to_numpy() - (base - s * emax.loc[ns].to_numpy())
        ax.plot(
            ns,
            residual,
            marker="o",
            color=style.PALETTE[i % len(style.PALETTE)],
            label=f"${int(size)}mm",
        )
    ax.axhline(0.0, color="#595959", linewidth=0.8)
    ax.set_xlabel("Dealers asked per RFQ")
    ax.set_ylabel("Observed minus fit (32nds)")
    ax.set_xticks(list(N_GRID))
    ax.text(
        0.03,
        0.05,
        "the drift past N=5 is dealers widening quotes",
        transform=ax.transAxes,
        fontsize=9,
        color="#595959",
    )
    return style.save(fig, "note06_01_competition_benefit")


def figure_leakage(cells_frame: pd.DataFrame) -> Path:
    """Figure 2: markout at 30 minutes and the next RFQ's arrival shift against N, by leakage."""
    sel = _focus(cells_frame, tenor=10, status="OTR", size=50.0)
    style.apply(hero=True)
    fig, axes = style.figure(
        "Leakage alone: drift and the next arrival grow with N",
        f"10y on-the-run, $50mm, {len(SEEDS)} seeds; 32nds, positive is drift in the trade's direction",
        _FIGURE_SOURCE,
        nrows=1,
        ncols=2,
    )
    for ax, column, title in (
        (axes[0], "markout_30m_1_mean", "Markout at 30 minutes (32nds)"),
        (axes[1], "arrival_shift_2_mean", "Second arrival minus first (32nds)"),
    ):
        for i, leak in enumerate(sorted(sel["leakage_scale"].unique())):
            group = sel[np.isclose(sel["leakage_scale"], leak)].sort_values("n_dealers")
            ax.plot(
                group["n_dealers"],
                group[column],
                marker="o",
                color=style.PALETTE[i % len(style.PALETTE)],
                label=f"leakage_scale={leak:g}",
            )
        ax.set_xlabel("Dealers asked per RFQ")
        ax.set_ylabel(title)
        ax.set_xticks(list(N_GRID))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.862), ncol=4, fontsize=9)
    fig.subplots_adjust(wspace=0.34)
    return style.save(fig, "note06_02_leakage")


def figure_programme(cells_frame: pd.DataFrame) -> Path:
    """Figure 3: programme cost over N and size, one panel per leakage level, argmin marked."""
    leaks = list(LEAKAGES)
    style.apply(hero=True)
    fig, axes = style.figure(
        "Programme cost: competition pays for small tickets only",
        "10y on-the-run; programme cost = (first + shift + second) / 2, 32nds; red box = best N",
        _FIGURE_SOURCE,
        nrows=1,
        ncols=len(leaks),
    )
    fig.subplots_adjust(wspace=0.16)
    frames = {}
    lo, hi = np.inf, -np.inf
    for leak in leaks:
        sel = _focus(cells_frame, tenor=10, status="OTR", leakage_scale=leak)
        table = sel.pivot(
            index="size", columns="n_dealers", values="programme_cost_mean"
        ).reindex(index=list(SIZES), columns=list(N_GRID))
        frames[leak] = table
        lo = min(lo, float(np.nanmin(table.to_numpy())))
        hi = max(hi, float(np.nanmax(table.to_numpy())))
    image = None
    for panel, (ax, leak) in enumerate(zip(axes, leaks, strict=True)):
        table = frames[leak]
        values = table.to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto", cmap="viridis_r", vmin=lo, vmax=hi)
        ax.grid(False)
        ax.set_xticks(range(len(N_GRID)), [str(n) for n in N_GRID])
        ax.set_yticks(range(len(SIZES)), [f"${int(s)}mm" for s in SIZES] if panel == 0 else [])
        ax.set_xlabel("Dealers asked")
        ax.set_title(f"leakage_scale = {leak:g}", fontsize=12)
        shade = (hi - values) / (hi - lo) if hi > lo else np.zeros_like(values)
        for i in range(values.shape[0]):
            j = int(np.nanargmin(values[i]))
            for k in range(values.shape[1]):
                ax.text(
                    k,
                    i,
                    f"{values[i, k]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if shade[i, k] > 0.45 else "black",
                )
            ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, edgecolor="#D55E00", linewidth=2.0))
    if image is not None:
        fig.colorbar(image, ax=list(axes), shrink=0.8, label="Programme cost (32nds)")
    focus = _focus(cells_frame, tenor=10, status="OTR", leakage_scale=DEFAULT_LEAKAGE)
    fills = {
        int(n): float(focus[(np.isclose(focus["size"], 250.0)) & (focus["n_dealers"] == n)]["fill_rate_mean"].iloc[0])
        for n in (1, 5)
    }
    fig.text(
        0.08,
        0.852,
        f"Fill rate on the $250mm row: {fills[1]:.0%} at N=1 against {fills[5]:.0%} at N=5; "
        "prices are for programmes that got a quote",
        fontsize=10,
        color="#595959",
    )
    return style.save(fig, "note06_03_programme_cost")


def figure_tiers(runs: pd.DataFrame) -> Path:
    """Figure 4: optimal N by size across run status and tenor at the default leakage."""
    style.apply(hero=True)
    fig, ax = style.figure(
        "Where the optimal N falls by size, run status and tenor",
        f"leakage_scale = {DEFAULT_LEAKAGE:g}; optimal N minimises programme cost, {len(SEEDS)} seeds",
        _FIGURE_SOURCE,
    )
    segments = ((2, "OTR"), (10, "OTR"), (30, "OTR"), (10, "OLD"))
    top = 0
    for i, (tenor, status) in enumerate(segments):
        frame = optimal_n(runs, DEFAULT_LEAKAGE, tenor, status)
        colour = style.PALETTE[i]
        ax.plot(
            frame["size"],
            frame["optimal_n"],
            color=colour,
            label=f"{tenor}y {'on-the-run' if status == 'OTR' else 'old'}",
        )
        tie = frame["tie"].to_numpy(dtype=bool)
        ax.scatter(frame.loc[~tie, "size"], frame.loc[~tie, "optimal_n"], color=colour, s=55, zorder=3)
        ax.scatter(
            frame.loc[tie, "size"],
            frame.loc[tie, "optimal_n"],
            facecolors="white",
            edgecolors=colour,
            s=55,
            zorder=3,
        )
        top = max(top, int(frame["optimal_n"].max()))
    ax.set_xscale("log")
    ax.set_xlabel("Ticket size ($mm, log scale)")
    ax.set_ylabel("Optimal number of dealers")
    ax.set_xticks(list(SIZES), [str(int(s)) for s in SIZES])
    ax.set_yticks([n for n in N_GRID if n <= max(top, 3)])
    ax.set_ylim(0.5, max(top, 3) + 0.4)
    ax.legend(loc="upper right")
    ax.text(
        0.03,
        0.05,
        "hollow marker: the best N is not separated from the next best at 5% (paired, by seed)",
        transform=ax.transAxes,
        fontsize=9,
        color="#595959",
    )
    return style.save(fig, "note06_04_optimal_n_by_tier")


def table_axe(runs: pd.DataFrame) -> pd.DataFrame:
    """Table 1: an axed single dealer against one random dealer and three unrestricted dealers.

    The axed columns cover only the programmes whose first RFQ found a favourable axe; the random
    single-dealer column is the N = 1 grid cell (every programme, no axe restriction), which is
    what a desk without an axe gets for the same ticket.
    """
    axe = cells(runs, "axe")
    grid = cells(runs)
    rows = []
    for size in SIZES:
        sel = axe[np.isclose(axe["size"], size)]
        axed = sel[sel["n_dealers"] == 1]
        comp = sel[sel["n_dealers"] == 3]
        plain = _focus(grid, tenor=10, status="OTR", size=size, leakage_scale=DEFAULT_LEAKAGE)
        one = plain[plain["n_dealers"] == 1].iloc[0]
        rows.append(
            dict(
                size=float(size),
                axe_share=float(axed["axe_share_mean"].iloc[0]),
                axe_exec_cost=float(axed["exec_cost_1_mean"].iloc[0]),
                axe_markout_30m=float(axed["markout_30m_1_mean"].iloc[0]),
                axe_programme_cost=float(axed["programme_cost_mean"].iloc[0]),
                one_exec_cost=float(one["exec_cost_1_mean"]),
                one_programme_cost=float(one["programme_cost_mean"]),
                comp_exec_cost=float(comp["exec_cost_1_mean"].iloc[0]),
                comp_markout_30m=float(comp["markout_30m_1_mean"].iloc[0]),
                comp_programme_cost=float(comp["programme_cost_mean"].iloc[0]),
            )
        )
    return pd.DataFrame(rows)


def table_calibrate(runs: pd.DataFrame) -> pd.DataFrame:
    """Table 2: the field to measure and the sample size per cell to detect an effect.

    Two-sample power at 5% two-sided and 80%: `n = 2 * (1.96 + 0.84)**2 * sd**2 / delta**2`.
    `sd` is the pair-level residual of the quantity in the N = 3, 10y on-the-run, $50mm,
    leakage_scale = 1 cells (averaged over seeds): the spread a desk sees between programmes.
    `delta` is the effect it wants to detect, in 32nds (1 32nd is 3.1 bp of price).
    """
    sel = runs[
        (runs["kind"] == "grid")
        & (runs["tenor"] == 10)
        & (runs["status"] == "OTR")
        & np.isclose(runs["size"], 50.0)
        & (runs["n_dealers"] == 3)
        & np.isclose(runs["leakage_scale"], DEFAULT_LEAKAGE)
    ]
    rows = []
    for quantity, field, note, sd_key in (
        (
            "Competition benefit",
            "N asked, number of responses, best quote at each N",
            "one row per RFQ; compare holding size and tenor fixed",
            "sd_exec_cost_1",
        ),
        (
            "Post-trade drift",
            "arrival mid, mid at 5 and 30 minutes after the award",
            "markouts by N, with dealer identity on the losing side",
            "sd_arrival_shift_2",
        ),
        (
            "Arrival shift of repeat RFQs",
            "send timestamp, arrival mid, side and size of every repeat RFQ",
            "pair repeat RFQs by security, side and size within the session",
            "sd_arrival_shift_2",
        ),
        (
            "Programme cost",
            "all of the above, plus a programme identifier",
            "the desk's KPI: cost per programme, not per ticket",
            "sd_programme_cost",
        ),
    ):
        sd = float(sel[sd_key].mean())
        row = dict(quantity=quantity, field_needed=field, note=note, residual_sd=round(sd, 2))
        for delta in (0.10, 0.25, 0.50):
            n = 2.0 * (1.96 + 0.84) ** 2 * sd**2 / delta**2
            row[f"n_at_{delta:.2f}"] = int(np.ceil(n))
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown(frame: pd.DataFrame) -> str:
    """A small markdown table for NOTE.md, from a DataFrame."""
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(runs: pd.DataFrame) -> dict:
    """Figures, tables and the summary JSON; returns the summary."""
    TABLES.mkdir(parents=True, exist_ok=True)
    grid = cells(runs)
    paths = {
        "fig1": figure_competition(grid),
        "fig2": figure_leakage(grid),
        "fig3": figure_programme(grid),
        "fig4": figure_tiers(runs),
    }
    one = table_axe(runs)
    one.to_csv(TABLES / "table1_axe.csv", index=False)
    calib = table_calibrate(runs)
    calib.to_csv(TABLES / "table2_calibration.csv", index=False)
    (TABLES / "table1_axe.md").write_text(_markdown(one))
    (TABLES / "table2_calibration.md").write_text(_markdown(calib))

    def cost_at(tenor, status, size, n, leak=DEFAULT_LEAKAGE):
        sel = _focus(grid, tenor=tenor, status=status, size=size, leakage_scale=leak)
        return float(sel.loc[sel["n_dealers"] == n, "programme_cost_mean"].iloc[0])

    opt_10y = optimal_n(runs, DEFAULT_LEAKAGE, 10, "OTR")
    by_size = {row.size: row.optimal_n for row in opt_10y.itertuples()}
    drop_size = min((s for s, n in by_size.items() if n < 3), default=None)
    penalty_5 = cost_at(10, "OTR", 250.0, 5) - cost_at(10, "OTR", 250.0, by_size[250.0])
    disp = fitted_dispersion(grid)
    runtime = {}
    if (TABLES / "runtime.json").exists():
        runtime = json.loads((TABLES / "runtime.json").read_text())
    emax = order_statistics()
    check = {}
    for leak in (0.0, 2.0):
        sel = _focus(grid, tenor=10, status="OTR", size=50.0, leakage_scale=leak)
        check[f"cost_by_n_leak{leak:g}"] = {
            int(row.n_dealers): round(float(row.exec_cost_1_mean), 4) for row in sel.itertuples()
        }
        check[f"shift_by_n_leak{leak:g}"] = {
            int(row.n_dealers): round(float(row.arrival_shift_2_mean), 4)
            for row in sel.itertuples()
        }
    summary = {
        "generated_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "note": NOTE,
        "grid": {
            "n_dealers": list(N_GRID),
            "sizes": list(SIZES),
            "statuses": list(STATUSES),
            "leakage_scales": list(LEAKAGES),
            "tenors": list(TENORS),
            "seeds": list(SEEDS),
            "programmes_per_cell": PROGRAMMES,
            "days_sampled": DAYS_SAMPLED,
            "cells": int(len(grid)),
            "runs": int(len(runs)),
            "runtime": runtime,
        },
        "order_statistics": {int(r.n_dealers): round(float(r.emax), 4) for r in emax.itertuples()},
        "dispersion_fit": {
            str(int(r.size)): dict(s=round(float(r.s), 4), cost_n1=round(float(r.cost_n1), 4))
            for r in disp.itertuples()
        },
        "optimal_n_10y_otr": {int(r.size): int(r.optimal_n) for r in opt_10y.itertuples()},
        "cost_at_optimal_10y_otr": {
            int(r.size): round(float(r.cost), 4) for r in opt_10y.itertuples()
        },
        "size_where_optimal_drops_below_3": drop_size,
        "penalty_of_5_dealers_at_250mm": round(float(penalty_5), 4),
        "optimal_n_by_segment": {
            f"{tenor}y_{status}": {
                int(r.size): int(r.optimal_n)
                for r in optimal_n(runs, DEFAULT_LEAKAGE, tenor, status).itertuples()
            }
            for tenor in TENORS
            for status in STATUSES
        },
        "invariants": check,
        "figures": {k: str(v.relative_to(HERE.parent.parent)) for k, v in paths.items()},
        "table1_axe": one.to_dict(orient="records"),
        "table2_calibration": calib.to_dict(orient="records"),
    }
    (TABLES / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Note 3: dealer count.")
    parser.add_argument("--run", action="store_true", help="run simulator cells for this part")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--collect", action="store_true", help="merge part files and write outputs")
    parser.add_argument("--reduced", action="store_true", help="run the CI-sized grid only")
    args = parser.parse_args()
    cfg = load_config()
    inputs = load_inputs()
    if args.reduced:
        frame = run_reduced(cfg, inputs)
        zero = frame[frame["leakage_scale"] == 0.0].sort_values("n_dealers")
        heavy = frame[frame["leakage_scale"] == 2.0].sort_values("n_dealers")
        print("zero-leakage cells with the dealers' competition widening switched off:")
        print(zero[["n_dealers", "exec_cost_1", "programme_cost"]].to_string(index=False))
        print("leakage_scale = 2 cells with the configured widening:")
        print(heavy[["n_dealers", "arrival_shift_2", "programme_cost"]].to_string(index=False))
        return
    if args.run:
        path = run_part(cfg, inputs, args.part, args.parts)
        print(f"wrote {path}")
        if not args.collect:
            return
    if args.collect:
        runs = collect(cfg, inputs)
    else:
        runs = load_runs()
    summary = write_outputs(runs)
    grid = summary["grid"]
    print(
        f"{grid['cells']} cells, {grid['runs']} runs; "
        f"optimal N 10y OTR {summary['optimal_n_10y_otr']}; "
        f"penalty of 5 dealers at 250mm "
        f"{summary['penalty_of_5_dealers_at_250mm']:.3f} 32nds"
    )


if __name__ == "__main__":
    main()
