# Module 2 calibration register (spot FX)

Every number Module 2 uses is listed here with the place it came from, the day it was read, and a
status: **READ** (taken from the named publication), **DERIVED** (computed from readings; the
derivation is written out), or **ASSUMPTION** (a modelling choice with no public source). Nothing in
this module is presented as a reading when it is not one. The machine-readable copy of these values
is `config/fx_sim.yaml`.

## Data source and licence

Tick data: **Dukascopy Bank SA**, historical tick feed at
`https://datafeed.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5`, where `MM` is the
zero-based month. Files are LZMA-compressed 20-byte big-endian records
`(ms_offset, ask, bid, ask_vol, bid_vol)` with prices scaled by `1e5` (`1e3` for JPY-quoted pairs).

Dukascopy data is used here for personal, non-commercial research only. **Raw tick files are not
committed to this repository and are not redistributed in any form.** Only derived aggregates
(liquidity profiles) and simulated orders and costs in basis points are published, with attribution to
Dukascopy. Anyone re-running the download accepts Dukascopy's terms themselves.

Access policy: a single honest `User-Agent` (`tca-stack-research/0.1`), one request at a time, at
least one second between requests, no proxies, no IP rotation, no parallelism. HTTP 403/429, and a
sustained 5xx (the feed returns 503 when no backend is available), stop the run and keep what is
cached.

### Parser verification against live files (read 2026-11-03)

| URL | Records | First record `(ms, ask, bid, ask_vol, bid_vol)` | Check |
|---|---|---|---|
| `/EURUSD/2025/02/03/10h_ticks.bi5` (3 Mar 2025, 10:00 UTC) | 6,049 | `(12, 104274, 104270, 0.9, 0.9)` | ask 1.04274, bid 1.04270, ask > bid |
| `/EURUSD/2025/03/03/10h_ticks.bi5` (3 Apr 2025, 10:00 UTC) | 17,025 | `(2212, 110311, 110307, 0.9, 3.6)` | ask 1.10311, bid 1.10307 |

The 17,025-record reference quoted in the build brief matches the second URL: with the zero-based
month, `2025/03/03` is 3 April 2025, not 3 March 2025. The March file is a different, quieter hour.
Both confirm the field order (ask before bid) and the 1e5 scale. The parser test uses synthetic bytes
and needs no network.

## BIS volume proxy

FX has no consolidated volume, so the simulator participates against a daily **spot-turnover proxy**
`V`. It is allocated across the day by the tick-rate profile per pair. The README and `docs/fx.md`
must say this is a modelling convenience.

Source: **BIS Triennial Central Bank Survey 2025** (14th survey, turnover in April 2025). Preliminary
results published 30 September 2025; final turnover results released June 2026.

- Survey page: <https://www.bis.org/publications/triennial-central-bank-survey-foreign-exchange-and-over-the-counter-otc-derivatives-markets-2025>
- Tables file: <https://www.bis.org/publications/triennial-survey/rpfx25-fx-tables.xlsx>

| Parameter | Value | Source | What was read | Date read | Status |
|---|---|---|---|---|---|
| Global FX turnover, all instruments | 9,595.5 USD bn/day | BIS 2025 tables, sheet "OTC foreign exchange turnover by instrument", 2025 column | Daily average turnover in April 2025, net-net basis | 2026-11-03 | READ |
| Global **spot** turnover | 2,957.1 USD bn/day | Same table, row "Spot transactions", 2025 column | Spot is 30.82% of the 9,595.5 total | 2026-11-03 | READ |
| EURUSD turnover, all instruments | 2,033.3 USD bn/day | BIS 2025 tables, sheet "OTC foreign exchange turnover by currency pair", 2025 column (labelled `USD/EUR`) | Pair line for the euro/dollar pair | 2026-11-03 | READ |
| USDJPY turnover, all instruments | 1,372.0 USD bn/day | Same sheet (labelled `USD/JPY`) | Pair line | 2026-11-03 | READ |
| GBPUSD turnover, all instruments | 781.3 USD bn/day | Same sheet (labelled `USD/GBP`) | Pair line | 2026-11-03 | READ |
| AUDUSD turnover, all instruments | 731.1 USD bn/day | Same sheet (labelled `USD/AUD`) | Pair line | 2026-11-03 | READ |
| Pair share of all instruments | EURUSD 21.190%, USDJPY 14.298%, GBPUSD 8.143%, AUDUSD 7.619% | Computed from the rows above | Pair turnover / 9,595.5 | 2026-11-03 | DERIVED |
| Pair **spot** turnover (the volume proxy `V`) | EURUSD 626.6, USDJPY 422.8, GBPUSD 240.8, AUDUSD 225.3 USD bn/day | Computed | 2,957.1 x pair share of all instruments | 2026-11-03 | DERIVED |

BIS publishes turnover by instrument and by currency pair separately; it does not publish spot
turnover by pair. The spot split above therefore assumes each pair's spot share equals its share of
all instruments. That is one step removed from a reading and is labelled DERIVED everywhere.

## Flow parameters (simulation design)

These are the client-flow assumptions of the simulator. They are choices, not measurements, so the
status is ASSUMPTION; the source is the simulation design.

| Parameter | Value | Source | What was read | Date read | Status |
|---|---|---|---|---|---|
| Parents per weekday | Poisson, mean 70 | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| Pair weights | 0.45 / 0.25 / 0.18 / 0.12 (EURUSD, USDJPY, GBPUSD, AUDUSD) | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| Arrival windows (UTC) | 50% 07–11, 35% 12–16, 10% 16–21, 5% 00–07 | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| Notional | lognormal, median 25mm USD, sigma 1.0, clip [1mm, 750mm] | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| Round-number snap | 30% to {5, 10, 25, 50, 100, 250, 500} mm | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| Side | 50/50 plus daily bias `b_d ~ N(0, 0.25)`, P(buy) clipped to [0.2, 0.8] | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| Urgency mix | LOW 0.35, MEDIUM 0.35, HIGH 0.20, FIX 0.10 | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| TWAP (LOW) | duration `clip(notional_mm/5, 10, 120)` min; slices every 30–90 s; passive first for 20 s | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| POV (MEDIUM) | 20% of the volume proxy, duration cap 90 min, aggressive | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| IS (HIGH) | Almgren–Chriss, `clip(notional_mm/15, 3, 30)` min | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| Almgren–Chriss risk aversion | 1.0e-7 per USD | config | No value given; chosen for mild front-loading | 2026-11-03 | ASSUMPTION |
| RISK_TRANSFER | 30% of HIGH; `mid ± (hs x 1.2 + k_rt x sqrt(Q/V))` | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| `k_rt` | 0.003 (price units per sqrt of Q/V) | config | No value given; about 1 bp at Q/V = 0.1% | 2026-11-03 | ASSUMPTION |
| FIX | WM/R 16:00 London benchmark, TWAP over the 5-minute window, aggressive child every 5 s | simulation design | Simulation design (benchmark is a 1-second-mid proxy) | 2026-11-03 | ASSUMPTION |
| Broker assignment | 70% random, 30% discretionary (>100mm to broker A) | simulation design | Simulation design; planted effects live in `config/brokers.yaml` | 2026-11-03 | ASSUMPTION |
| Decision latency | 20 ms | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |

## Impact model (simulation design)

| Parameter | Value | Source | What was read | Date read | Status |
|---|---|---|---|---|---|
| Square-root coefficient `Y` | 0.7 | Tóth et al. 2011, arXiv:1105.1694; Bucci et al. 2019, arXiv:1811.05230 | Equity-market estimate, borrowed for FX | 2026-11-03 | ASSUMPTION |
| Permanent fraction | 0.5 | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| Transient half-life | 15 min | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| `sigma_d` | realised daily vol of 1-minute mid returns, trailing 20 days | simulation design | Computed from the downloaded tape | 2026-11-03 | DERIVED |
| Passive impact multiple | 0.5 of the aggressive coefficient | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |

## Liquidity providers and last look (simulation design)

| Parameter | Value | Source | What was read | Date read | Status |
|---|---|---|---|---|---|
| Number of LPs | 5 | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| `markup_pips` | uniform −0.05 to +0.15 vs the tape touch | simulation design | Range given, shape assumed uniform | 2026-11-03 | ASSUMPTION |
| `reject_base` | uniform 0.01 to 0.08 | simulation design | Range given | 2026-11-03 | ASSUMPTION |
| `reject_asym` | uniform 0 to 0.25 | simulation design | Range given | 2026-11-03 | ASSUMPTION |
| `hold_ms` | lognormal, median uniform 30 to 250 ms, sigma 0.5 | simulation design (median range); sigma from config | Median range given; lognormal sigma not given | 2026-11-03 | ASSUMPTION |
| `max_clip_mm` | 50/40/30/20/10 mm for the five LPs | config | Attribute given without a range | 2026-11-03 | ASSUMPTION |
| Quote noise | 0.05 pips SD | config | Quotes carry noise, no scale given | 2026-11-03 | ASSUMPTION |
| Rejections before ECN | 3, then cross at `ask_eff + 0.1` pip | simulation design | Simulation design | 2026-11-03 | ASSUMPTION |
| Passive fill target | about 40% of passive children fill | simulation design | Calibration target for `lambda_fill`; realised rate is 49.8% because crossing alone fills about 46% on the full sample (realised rate 48.2% to 50.1%, crossing alone 45.0% to 46.9% over six seeds; the model has no queue, so a resting order fills at first touch); the ~40% target is not reachable through `lambda_fill` on EURUSD, USDJPY and most GBPUSD runs | 2026-11-03 | ASSUMPTION |

## Notes

- Dates and hours are UTC throughout. The download skips Saturday, Friday after 22:00 UTC and Sunday
  before 21:00 UTC, which is an approximation of the weekly market close; it is not a holiday
  calendar, so a handful of holiday hours download empty and are recorded as empty.
- Dukascopy volumes are single-venue and indicative. They are stored on the ticks but are never used
  as ADV or as the participation denominator; the BIS proxy above is.
- Spot turnover by pair is DERIVED from the all-instrument share. If a future BIS release publishes
  spot by pair, replace the derived block in `config/fx_sim.yaml` with the reading and mark it READ.
- **Passive fill-rate robustness (2026-11-08).** `scripts/fx_fill_rate_seeds.py` reruns the full
  sample on six master seeds and writes `docs/tables/fx_fill_rate_seeds.csv`. The realised passive
  fill rate ranges 48.2% to 50.1%, crossing alone 45.0% to 46.9%, and a crossing haircut solved to a
  40% pooled rate raises TWAP implementation shortfall by 0.054 to 0.081 bp in every seed
  (mean 0.061, paired standard error 0.006 or less). The shorter-timeout lever of `docs/fx.md`,
  which lowers TWAP IS by 0.03 to 0.07 bp, was measured on the configured seed only.
