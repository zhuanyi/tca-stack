"""Plain data types shared by order generation, strategies and the replay engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from tca.fx import ticks

PASSIVE = "PASSIVE"
AGGRESSIVE = "AGGRESSIVE"
RISK_TRANSFER_INTENT = "RT"
FIX_INTENT = "FIX"

URGENCIES = ("LOW", "MEDIUM", "HIGH", "FIX")
ORDER_COLUMNS = [
    "order_id",
    "pair",
    "side",
    "notional_usd",
    "ts_arrival_utc",
    "urgency",
    "style",
    "broker",
    "assignment",
    "trader_id",
    "planned_duration_s",
]


@dataclass(frozen=True)
class Order:
    """A parent order. ``broker`` and ``assignment`` are ``None`` for risk transfer and fix orders.

    ``ts_arrival_us`` is epoch microseconds, ``notional_usd`` is in USD of the base currency
    and ``planned_duration_s`` in seconds.
    """

    order_id: int
    day: date
    pair: str
    side: int
    notional_usd: float
    ts_arrival_us: int
    urgency: str
    style: str
    broker: str | None
    assignment: str | None
    trader_id: int
    planned_duration_s: float


@dataclass(frozen=True)
class ChildPlan:
    """One scheduled child: seconds after arrival, USD quantity and intent."""

    offset_s: float
    qty_usd: float
    intent: str


def orders_from_frame(frame: pd.DataFrame) -> list[Order]:
    """Parent orders from the ``fx_orders`` columns, in frame order."""
    arrival = ticks.to_us(frame["ts_arrival_utc"])
    out = []
    for row, us in zip(frame.itertuples(index=False), arrival, strict=True):
        out.append(
            Order(
                order_id=int(row.order_id),
                day=pd.Timestamp(us, unit="us").date(),
                pair=str(row.pair),
                side=int(row.side),
                notional_usd=float(row.notional_usd),
                ts_arrival_us=int(us),
                urgency=str(row.urgency),
                style=str(row.style),
                broker=None if pd.isna(row.broker) else str(row.broker),
                assignment=None if pd.isna(row.assignment) else str(row.assignment),
                trader_id=int(row.trader_id),
                planned_duration_s=float(row.planned_duration_s),
            )
        )
    return out
