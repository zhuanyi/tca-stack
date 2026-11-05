"""Public inputs: build the committed input tables from ``<raw>/ust`` and load them back.

``prepare`` is the only function that reads raw downloads. Its four output tables
(``ust_securities``, ``ust_universe``, ``ust_daily_anchor``, ``ust_events``) are committed, so
the simulator, the tests and the figures run from a fresh clone without the raw files.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tca.core import calendar, storage
from tca.ust.config import REPO_ROOT, TENORS, SimConfig

TREASURY_DIRECT_URL = "https://www.treasurydirect.gov/TA_WS/securities/auctioned"
_PAR_COLUMNS = {f"{t} Yr": t for t in TENORS}
_OFF_THE_RUN_TENORS = (2, 5, 10, 30)
_AUCTION_EVENT = "AUCTION_RESULT"


def _anchors(par: pd.DataFrame, window_days: int, period_start: date, period_end: date):
    """Daily par yields to the anchor table with 5-day realised variance and regime ratio.

    ``realised_vol_5d`` is the root of the trailing mean of squared daily changes (bp per day)
    over ``window_days`` ending on the day itself. ``regime_mult`` divides the same variance by
    its median over the period.
    """
    wide = par.rename(columns=_PAR_COLUMNS).set_index("date")[list(TENORS)] / 100.0
    wide.index = [date.fromisoformat(d) for d in wide.index]
    change_bp = wide.diff() * 1e4
    rv = (change_bp**2).rolling(window_days).mean()
    in_period = [period_start <= d <= period_end for d in wide.index]
    median = rv[in_period].median()
    frames = []
    for tenor in TENORS:
        frames.append(
            pd.DataFrame(
                {
                    "date": wide.index,
                    "tenor": tenor,
                    "close_yield": wide[tenor].to_numpy(),
                    "change_bp": change_bp[tenor].to_numpy(),
                    "realised_vol_5d": np.sqrt(rv[tenor]).to_numpy(),
                    "regime_mult": (rv[tenor] / median[tenor]).to_numpy(),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True).sort_values(["date", "tenor"], kind="stable")
    return out.reset_index(drop=True)


def _select_universe(days: pd.DataFrame) -> pd.DataFrame:
    """The 15 daily securities on business days: every on-the-run, plus OLD and DOUBLE_OLD for
    2y, 5y, 10y and 30y. The source file lists weekday holidays too; they are dropped."""
    keep = (days["run_status"] == "OTR") | days["tenor_y"].isin(_OFF_THE_RUN_TENORS)
    out = days.loc[keep, ["date", "cusip", "tenor_y", "run_status"]]
    out = out.rename(columns={"tenor_y": "tenor"})
    out["date"] = [date.fromisoformat(d) for d in out["date"]]
    out = out[[calendar.is_business_day(d) for d in out["date"]]]
    return out.sort_values(["date", "tenor", "run_status"], kind="stable").reset_index(drop=True)


def _auction_events(auctions: pd.DataFrame, offset_s: int) -> pd.DataFrame:
    """One result event per auction, ``offset_s`` after the competitive bidding close."""
    rows = []
    for r in auctions.itertuples():
        close = datetime.strptime(r.auction_close_time_et, "%I:%M %p").time()
        d = date.fromisoformat(r.auction_date)
        ts = calendar.local_to_utc(d, close, calendar.NY) + timedelta(seconds=offset_s)
        url = f"{TREASURY_DIRECT_URL}?format=json&type={r.security_type}"
        rows.append((ts, _AUCTION_EVENT, int(r.tenor_y), url))
    return pd.DataFrame(rows, columns=["ts_utc", "type", "tenor", "source"])


def prepare(raw: Path, cfg: SimConfig) -> dict[str, pd.DataFrame]:
    """Build the four input tables from the raw downloads in ``<raw>/ust``."""
    ust = Path(raw) / "ust"
    start, end = cfg.period.start, cfg.period.end
    days = pd.read_csv(ust / "universe_by_day.csv")
    days = days[(days["date"] >= start.isoformat()) & (days["date"] <= end.isoformat())]
    universe = _select_universe(days)

    auctioned = pd.read_csv(ust / "securities_auctioned.csv")
    used = auctioned[auctioned["cusip"].isin(universe["cusip"].unique())]
    used = used[used["auction_date"] <= end.isoformat()]
    used = used.sort_values(["maturity_date", "auction_date"]).drop_duplicates("cusip", keep="last")
    securities = pd.DataFrame(
        {
            "cusip": used["cusip"],
            "tenor": used["tenor_y"].astype(int),
            "coupon": used["coupon_pct"] / 100.0,
            "issue_date": [date.fromisoformat(d) for d in used["issue_date"]],
            "maturity_date": [date.fromisoformat(d) for d in used["maturity_date"]],
            "auction_date": [date.fromisoformat(d) for d in used["auction_date"]],
            "auction_high_yield": used["high_yield_pct"] / 100.0,
        }
    )
    securities = securities.sort_values(["tenor", "maturity_date"]).reset_index(drop=True)

    in_window = auctioned[(auctioned["auction_date"] >= start.isoformat())]
    in_window = in_window[in_window["auction_date"] <= end.isoformat()]
    offset = cfg.yield_process.variance_weight.bump_auction_auctioned_tenor.offset_s_after_close
    macro = calendar.load_events(_events_path(cfg))
    events = pd.concat([macro, _auction_events(in_window, offset or 0)], ignore_index=True)
    events["tenor"] = events["tenor"].astype("Int64")
    events = events.sort_values(["ts_utc", "type"], kind="stable").reset_index(drop=True)

    par = pd.read_csv(ust / "par_yield_window.csv")
    anchors = _anchors(par, cfg.yield_process.window_days, start, end)
    return {
        "ust_securities": securities,
        "ust_universe": universe,
        "ust_daily_anchor": anchors,
        "ust_events": events,
    }


def _events_path(cfg: SimConfig) -> Path:
    return REPO_ROOT / cfg.events_file


@dataclass(frozen=True)
class Inputs:
    """The committed input tables, indexed for the simulator."""

    securities: pd.DataFrame
    universe: pd.DataFrame
    anchors: pd.DataFrame
    events: pd.DataFrame

    def business_days(self, start: date, end: date) -> list[date]:
        """Business days in the universe table between ``start`` and ``end``, inclusive."""
        days = sorted(set(self.universe["date"]))
        return [d for d in days if start <= d <= end]


def load_inputs(derived: Path | None = None) -> Inputs:
    """Read the four input tables from ``derived`` (default ``<data>/derived``)."""
    names = ("ust_securities", "ust_universe", "ust_daily_anchor", "ust_events")
    tables = [storage.read_table(n, derived) for n in names]
    return Inputs(*tables)
