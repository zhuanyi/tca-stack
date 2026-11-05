#!/usr/bin/env python
"""Fetch the public inputs for the UST simulator (Module 1) into <raw>/ust/.

Sources (all public, no account needed):
  * US Treasury daily par yield curve CSV (fallback: FRED DGS series)
  * TreasuryDirect auctioned securities JSON (type=Note and type=Bond)
  * FINRA TRACE Treasury daily aggregates (xlsx files on cdn.finra.org)

Usage:
    python scripts/fetch_ust_anchors.py --start 2025-03-03 --end 2025-05-30

The raw directory is $TCA_DATA_RAW (default data/raw). Everything is cached; a
re-run only downloads what is missing (use --refresh to force). Requests are
sequential, spaced by --delay seconds, and use a plain User-Agent. On HTTP 429
or 403 the request backs off (60 s, doubling, at most 4 tries) and is then
abandoned and recorded; no workaround is attempted.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

USER_AGENT = "tca-stack-research/0.1"
TREASURY_CSV = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
TD_URL = "https://www.treasurydirect.gov/TA_WS/securities/auctioned?format=json&type={type}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_SERIES = {
    "2y": "DGS2",
    "3y": "DGS3",
    "5y": "DGS5",
    "7y": "DGS7",
    "10y": "DGS10",
    "20y": "DGS20",
    "30y": "DGS30",
}
FINRA_DAILY = "https://cdn.finra.org/trace/treasury-aggregates/daily/ts-daily-aggregates-{day}.xlsx"

TENORS = (2, 3, 5, 7, 10, 20, 30)
# FINRA bucket label -> tenor whose on-the-run security sits in that remaining-maturity bucket
FINRA_BUCKET_TENOR = {
    "<= 2 years": 2,
    "> 2 years and <= 3 years": 3,
    "> 3 years and <= 5 years": 5,
    "> 5 years and <= 7 years": 7,
    "> 7 years and <= 10 years": 10,
    "> 10 years and <= 20 years": 20,
    "> 20 years": 30,
}
HISTORY_DAYS = 548  # auction history kept before the start so OLD / DOUBLE_OLD exist on day one
YIELD_LOOKBACK_DAYS = 30


class Fetcher:
    """Sequential, polite HTTP client with 429/403 backoff and a record of every request."""

    def __init__(self, delay: float) -> None:
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60
        )
        self.delay = delay
        self.log: list[dict] = []
        self._last = 0.0

    def get(self, url: str, max_tries: int = 4) -> tuple[int, bytes]:
        wait = 60.0
        for attempt in range(1, max_tries + 1):
            gap = self.delay - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            try:
                r = self.client.get(url)
                status, body = r.status_code, r.content
            except httpx.HTTPError as exc:  # network error: record and treat as failure
                status, body = 0, str(exc).encode()
            self._last = time.monotonic()
            self.log.append({"url": url, "status": status, "bytes": len(body), "attempt": attempt})
            if status in (429, 403) and attempt < max_tries:
                print(f"  HTTP {status} for {url}; backing off {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                wait *= 2
                continue
            return status, body
        return status, body


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# --------------------------------------------------------------------------- par yield curve
def fetch_par_yields(f: Fetcher, out: Path, start: date, end: date, refresh: bool) -> dict:
    lo = start - timedelta(days=YIELD_LOOKBACK_DAYS)
    rows: list[dict] = []
    source = "treasury"
    for year in range(lo.year, end.year + 1):
        path = out / f"par_yield_curve_{year}.csv"
        if refresh or not path.exists():
            status, body = f.get(TREASURY_CSV.format(year=year))
            if status != 200 or not body.startswith(b"Date"):
                print(
                    f"Treasury CSV {year} failed (HTTP {status}); falling back to FRED",
                    file=sys.stderr,
                )
                return fetch_par_yields_fred(f, out, lo, end, refresh)
            path.write_bytes(body)
        for rec in csv.DictReader(io.StringIO(path.read_text())):
            d = datetime.strptime(rec["Date"], "%m/%d/%Y").date()
            if lo <= d <= end:
                rows.append(
                    {"date": d.isoformat(), **{k: v for k, v in rec.items() if k != "Date"}}
                )
    rows.sort(key=lambda r: r["date"])
    return write_yield_window(out, rows, source)


def fetch_par_yields_fred(f: Fetcher, out: Path, lo: date, end: date, refresh: bool) -> dict:
    series: dict[str, dict[str, str]] = {}
    for tenor, sid in FRED_SERIES.items():
        path = out / f"fred_{sid}.csv"
        if refresh or not path.exists():
            status, body = f.get(FRED_URL.format(series=sid))
            if status != 200:
                raise RuntimeError(f"FRED {sid} failed with HTTP {status}")
            path.write_bytes(body)
        for rec in csv.DictReader(io.StringIO(path.read_text())):
            val = rec[sid]
            if val not in ("", "."):
                series.setdefault(
                    rec["observation_date"] if "observation_date" in rec else rec["DATE"], {}
                )[f"{tenor[:-1]} Yr"] = val
    rows = [
        {"date": d, **v}
        for d, v in sorted(series.items())
        if lo <= parse_date(d) <= end and len(v) == len(FRED_SERIES)
    ]
    return write_yield_window(out, rows, "fred")


def write_yield_window(out: Path, rows: list[dict], source: str) -> dict:
    cols = ["date"] + [c for c in rows[0] if c != "date"]
    with (out / "par_yield_window.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return {"rows": len(rows), "first": rows[0]["date"], "last": rows[-1]["date"], "source": source}


# ------------------------------------------------------------------- TreasuryDirect auctions
def term_to_tenor(term: str) -> int | None:
    """Map a securityTerm such as '9-Year 10-Month' (reopening) to the benchmark tenor 10."""
    m = re.fullmatch(r"(\d+)-Year(?: (\d+)-Month)?", term.strip())
    if not m:
        return None
    years = int(m.group(1)) + (int(m.group(2)) / 12 if m.group(2) else 0)
    best = min(TENORS, key=lambda t: abs(t - years))
    return best if abs(best - years) <= 1.0 else None


def fetch_auctions(f: Fetcher, out: Path, start: date, end: date, refresh: bool) -> dict:
    raw: list[dict] = []
    for typ in ("Note", "Bond"):
        path = out / f"td_auctioned_{typ.lower()}.json"
        if refresh or not path.exists():
            status, body = f.get(TD_URL.format(type=typ))
            if status != 200:
                raise RuntimeError(f"TreasuryDirect type={typ} failed with HTTP {status}")
            path.write_bytes(body)
        raw.extend(json.loads(path.read_text()))
    cutoff = start - timedelta(days=HISTORY_DAYS)
    kept, excluded = (
        [],
        {"tips": 0, "frn": 0, "other_type": 0, "unmapped_term": 0, "out_of_window": 0},
    )
    for x in raw:
        if x.get("inflationProtected") == "Yes" or x.get("securityType") == "TIPS":
            excluded["tips"] += 1
            continue
        if x.get("floatingRate") == "Yes" or x.get("securityType") == "FRN":
            excluded["frn"] += 1
            continue
        if x.get("securityType") not in ("Note", "Bond"):
            excluded["other_type"] += 1
            continue
        tenor = term_to_tenor(x["securityTerm"])
        if tenor is None:
            excluded["unmapped_term"] += 1
            continue
        adate = parse_date(x["competitiveAuctionCloseDate"][:10])
        if not (cutoff <= adate <= end):
            excluded["out_of_window"] += 1
            continue
        kept.append(
            {
                "cusip": x["cusip"],
                "tenor_y": tenor,
                "security_type": x["securityType"],
                "security_term": x["securityTerm"],
                "original_security_term": x["originalSecurityTerm"],
                "is_reopening": x["reopening"],
                "auction_date": adate.isoformat(),
                "auction_close_time_et": x["competitiveAuctionCloseTime"],
                "issue_date": x["issueDate"][:10],
                "maturity_date": x["maturityDate"][:10],
                "coupon_pct": x["interestRate"],
                "high_yield_pct": x["highYield"],
                "high_price": x["highPrice"],
                "offering_amount": x["offeringAmount"],
            }
        )
    kept.sort(key=lambda r: (r["auction_date"], r["tenor_y"], r["cusip"]))
    cols = list(kept[0])
    with (out / "securities_auctioned.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(kept)
    n_universe, thin = build_universe(out, kept, start - timedelta(days=YIELD_LOOKBACK_DAYS), end)
    return {
        "auction_rows_raw": len(raw),
        "auction_rows_kept": len(kept),
        "distinct_cusips": len({r["cusip"] for r in kept}),
        "excluded": excluded,
        "tenor_days_with_fewer_than_3_ranks": thin,
        "universe_rows": n_universe,
    }


def build_universe(out: Path, secs: list[dict], lo: date, hi: date) -> tuple[int, int]:
    """Daily OTR / OLD / DOUBLE_OLD mapping per tenor.

    Rule: for a tenor, order distinct CUSIPs by the date of their most recent auction on or before
    the day (a reopening keeps the CUSIP, so it refreshes that CUSIP and does not create a new
    one); rank 0 = OTR, 1 = OLD, 2 = DOUBLE_OLD. A security is OTR from its auction date, so
    `when_issued` is true between auction and issue date.
    """
    status = ("OTR", "OLD", "DOUBLE_OLD")
    by_tenor: dict[int, list[dict]] = {}
    for s in secs:
        by_tenor.setdefault(s["tenor_y"], []).append(s)
    rows, thin = [], 0
    for day in weekdays(lo, hi):
        iso = day.isoformat()
        for tenor in TENORS:
            last: dict[str, dict] = {}
            for s in by_tenor.get(tenor, []):
                if s["auction_date"] <= iso:
                    last[s["cusip"]] = s  # list is date sorted, later auction overwrites
            ranked = sorted(
                last.values(), key=lambda s: (s["auction_date"], s["issue_date"]), reverse=True
            )
            thin += len(ranked) < 3
            for i, s in enumerate(ranked[:3]):
                rows.append(
                    {
                        "date": iso,
                        "tenor_y": tenor,
                        "run_status": status[i],
                        "cusip": s["cusip"],
                        "coupon_pct": s["coupon_pct"],
                        "maturity_date": s["maturity_date"],
                        "issue_date": s["issue_date"],
                        "last_auction_date": s["auction_date"],
                        "when_issued": str(s["issue_date"] > iso),
                    }
                )
    with (out / "universe_by_day.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows), thin


# ------------------------------------------------------------------- FINRA TRACE daily aggregates
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx_rows(data: bytes) -> list[dict[str, object]]:
    """Minimal xlsx reader (sheet1 only) so that no extra dependency is needed."""
    z = zipfile.ZipFile(io.BytesIO(data))
    shared = [
        "".join(t.text or "" for t in si.iter(_NS + "t"))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(_NS + "si")
    ]
    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(_NS + "row"):
        cells: dict[str, object] = {}
        for c in r.findall(_NS + "c"):
            v = c.find(_NS + "v")
            if v is None:
                continue
            col = re.match(r"[A-Z]+", c.get("r")).group()
            cells[col] = shared[int(v.text)] if c.get("t") == "s" else float(v.text)
        rows.append(cells)
    return rows


def parse_trace(day: str, data: bytes) -> list[dict]:
    """Flatten one daily file into rows (section, bucket, run). Par values in $ billions."""
    out, section, bucket = [], None, None
    for cells in read_xlsx_rows(data):
        label = cells.get("A")
        if not isinstance(label, str) or "B" not in cells:
            continue
        if label in ("Bills", "FRNs", "Nominal Coupons", "TIPS", "Total"):
            section, bucket, run = label, "ALL", "ALL"
        elif label.startswith(("<=", ">")):
            bucket, run = label, "ALL"
        elif label in ("On-the-run", "Off-the-run"):
            run = "OTR" if label == "On-the-run" else "OFF"
        else:
            continue
        out.append(
            {
                "date": day,
                "section": section,
                "bucket": bucket,
                "run": run,
                "ats_trades": cells.get("B"),
                "ats_par_bn": cells.get("C"),
                "d2c_trades": cells.get("D"),
                "d2c_par_bn": cells.get("E"),
                "total_trades": cells.get("F"),
                "total_par_bn": cells.get("G"),
                "vwap": cells.get("H"),
            }
        )
    return out


def fetch_trace(f: Fetcher, out: Path, start: date, end: date, refresh: bool) -> dict:
    cache = out / "finra_daily"
    cache.mkdir(exist_ok=True)
    rows, downloaded = [], 0
    miss_path = out / "finra_missing_days.json"
    missing_prev = {} if refresh or not miss_path.exists() else json.loads(miss_path.read_text())
    missing: list[dict] = []
    for day in weekdays(start, end):
        iso = day.isoformat()
        path = cache / f"ts-daily-aggregates-{iso}.xlsx"
        if iso in missing_prev and not path.exists():
            missing.append({"date": iso, "status": missing_prev[iso]})  # known gap, not asked again
            continue
        if refresh or not path.exists():
            # The CDN answers 403 for a date without a file (holidays), so one retry is enough.
            status, body = f.get(FINRA_DAILY.format(day=iso), max_tries=2)
            if status != 200:
                missing.append({"date": iso, "status": status})
                continue
            path.write_bytes(body)
            downloaded += 1
        rows.extend(parse_trace(iso, path.read_bytes()))
    miss_path.write_text(json.dumps({m["date"]: m["status"] for m in missing}))
    if rows:
        with (out / "trace_daily_buckets.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    shares = trace_tenor_shares(rows)
    (out / "trace_tenor_shares.json").write_text(json.dumps(shares, indent=2))
    days = sorted({r["date"] for r in rows})
    return {
        "files_days": len(days),
        "rows": len(rows),
        "downloaded_this_run": downloaded,
        "missing_days": missing,
        "tenor_shares": shares,
    }


def trace_tenor_shares(rows: list[dict]) -> dict:
    """Period totals ($bn par) of nominal-coupon D2C volume by tenor bucket, OTR / off-the-run."""
    acc: dict[int, dict[str, float]] = {
        t: {"d2c_otr": 0.0, "d2c_off": 0.0, "total_otr": 0.0, "total_off": 0.0} for t in TENORS
    }
    for r in rows:
        if (
            r["section"] != "Nominal Coupons"
            or r["bucket"] not in FINRA_BUCKET_TENOR
            or r["run"] == "ALL"
        ):
            continue
        tenor = FINRA_BUCKET_TENOR[r["bucket"]]
        k = "otr" if r["run"] == "OTR" else "off"
        acc[tenor][f"d2c_{k}"] += r["d2c_par_bn"] or 0.0
        acc[tenor][f"total_{k}"] += r["total_par_bn"] or 0.0
    d2c_all = sum(v["d2c_otr"] + v["d2c_off"] for v in acc.values()) or 1.0
    ndays = len({r["date"] for r in rows}) or 1
    return {
        str(t): {
            **{k: round(v, 3) for k, v in acc[t].items()},
            "d2c_share": round((acc[t]["d2c_otr"] + acc[t]["d2c_off"]) / d2c_all, 5),
            "d2c_otr_fraction": round(
                acc[t]["d2c_otr"] / ((acc[t]["d2c_otr"] + acc[t]["d2c_off"]) or 1.0), 4
            ),
            "d2c_adv_bn_per_day": round((acc[t]["d2c_otr"] + acc[t]["d2c_off"]) / ndays, 3),
        }
        for t in TENORS
    }


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", default="2025-03-03")
    ap.add_argument("--end", default="2025-05-30")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--refresh", action="store_true", help="re-download cached files")
    args = ap.parse_args()
    start, end = parse_date(args.start), parse_date(args.end)

    raw = Path(os.environ.get("TCA_DATA_RAW", "data/raw")) / "ust"
    raw.mkdir(parents=True, exist_ok=True)
    f = Fetcher(args.delay)

    summary: dict[str, object] = {"start": args.start, "end": args.end, "raw_dir": str(raw)}
    steps = (
        ("par_yield", lambda: fetch_par_yields(f, raw, start, end, args.refresh)),
        ("treasurydirect_auctions", lambda: fetch_auctions(f, raw, start, end, args.refresh)),
        ("finra_trace_daily", lambda: fetch_trace(f, raw, start, end, args.refresh)),
    )
    failed = False
    for name, step in steps:
        try:
            summary[name] = step()
        except Exception as exc:  # keep going so one unreachable source does not hide the others
            failed = True
            summary[name] = {"error": f"{type(exc).__name__}: {exc}"}
    summary["requests"] = f.log
    (raw / "fetch_manifest.json").write_text(json.dumps(summary, indent=2, default=str))

    print("== fetch_ust_anchors run summary ==")
    print(f"period {args.start} .. {args.end}; raw dir {raw}")
    print(f"HTTP requests this run: {len(f.log)}")
    for name in ("par_yield", "treasurydirect_auctions", "finra_trace_daily"):
        s = dict(summary[name])  # type: ignore[arg-type]
        s.pop("tenor_shares", None)
        print(f"{name}: {json.dumps(s, default=str)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
