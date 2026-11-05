# UST simulator calibration: public inputs and what was read from them

This page records, for Module 1, where every calibrated number in `config/ust_sim.yaml` and
`config/events_2025.yaml` comes from. All dates read are **2026-11-03**. Status values:

- **READ**: a number read off a public source (for chart values the reading tolerance is given).
- **DERIVED**: computed from READ numbers or from the downloaded public data, with the formula stated.
- **ASSUMPTION**: a modelling choice with no public number behind it. It is never presented as a reading.

Period: 2025-03-03 to 2025-05-30, 65 weekdays of which 63 are business days (Good Friday 2025-04-18 and
Memorial Day 2025-05-26 are excluded). The window contains the April 2025 tariff sell-off.

Everything below is public. Nothing comes from a private source.

## 1. Parameter groups

| Parameter group (config key) | Value used | Source URL | What was read | Date read | Status |
|---|---|---|---|---|---|
| Yield anchors (`yield_process.anchor_reference`) | daily change sd 5.9 to 7.0 bp by tenor; PCA explained variance 0.840 / 0.143 / 0.009 (cumulative 0.991); 5-day 10y realised variance peaks 3.4x its period median on 2025-04-09 | https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2025/all?type=daily_treasury_yield_curve&field_tdr_date_value=2025&page&_format=csv | Daily par yields 2y..30y, 2025-02-03 to 2025-05-30 (82 rows); 63 daily changes inside the period | 2026-11-03 | DERIVED |
| Security universe (`universe_by_day.csv`, not in the yaml) | 21 securities per day (7 tenors x OTR, OLD, DOUBLE_OLD), 36 distinct CUSIPs in the period | https://www.treasurydirect.gov/TA_WS/securities/auctioned?format=json&type=Note and `...&type=Bond` | CUSIP, securityTerm, auction date and time, issue and maturity dates, coupon, high yield, reopening flag | 2026-11-03 | DERIVED |
| Auction event times | per auction, from `competitiveAuctionCloseTime` (1:00 PM ET for every auction in the period) | same TreasuryDirect JSON | `securities_auctioned.csv` columns `auction_date`, `auction_close_time_et` | 2026-11-03 | READ |
| Macro event calendar (`events_2025.yaml`) | NFP 03-07, 04-04, 05-02 at 08:30; CPI 03-12, 04-10, 05-13 at 08:30; FOMC statements 03-19 and 05-07 at 14:00 (all ET) | https://www.bls.gov/schedule/2025/03_sched.htm, `04_sched.htm`, `05_sched.htm`; https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm; https://www.federalreserve.gov/newsevents/pressreleases/monetary20250319a.htm and `monetary20250507a.htm` | Release dates and times; "For release at 2:00 p.m. EDT" on both statements | 2026-11-03 | READ |
| Composite half-spread, 2y, 5y, 10y (`composite_quote.hs_base_32nds`) | 0.07, 0.135, 0.27 (32nds) | https://libertystreeteconomics.newyorkfed.org/2025/11/how-has-treasury-market-liquidity-fared-in-2025/ and https://libertystreeteconomics.newyorkfed.org/2024/09/has-treasury-market-liquidity-improved-in-2024/ | Chart "Bid-Ask Spreads": full spread about 0.14 / 0.27 / 0.54 in Mar 2025 and in mid-2024 (10y on the right axis); half-spread = full / 2 | 2026-11-03 | DERIVED (from READ, tolerance +-10%) |
| Composite half-spread, 3y and 7y | 0.092 and 0.189 | not in the posts | linear interpolation in tenor between 2y/5y and 5y/10y | 2026-11-03 | DERIVED |
| Composite half-spread, 20y and 30y | 0.40 and 0.50 | not in the posts | extrapolated at a constant yield half-spread of about 0.1 bp, which is what the 2y/5y/10y readings imply | 2026-11-03 | ASSUMPTION |
| Off-the-run multipliers (`run_mult`) and yield spreads | run_mult 1.0 / 2.5 / 4.0; yield spread +1.5 bp OLD, +3 bp DOUBLE_OLD | no numeric public source found; the NY Fed speech https://www.newyorkfed.org/newsevents/speeches/2025/per250509 says off-the-run spreads roughly doubled in April 2025 but gives no level | placeholders kept | 2026-11-03 | ASSUMPTION |
| Time-of-day multipliers, variance weights, event bumps, OU noise, vol_mult formula | from the simulation design | none (no public intraday source) | placeholders kept | 2026-11-03 | ASSUMPTION |
| Depth at the touch, 2y, 5y, 10y (`composite_quote.depth_mm`) | 30, 30, 30 ($mm per side) | same two Fleming posts | Chart "Order Book Depth" (log axis, gridlines at 10 and 100), all three series about 30 in Mar 2025 and mid-2024. Text cross-check, 5y average bid depth $36.4mm and offer $36.6mm for 2019-2023: https://fraser.stlouisfed.org/files/docs/publications/frbnylse/frbny_libertystreet_20240212.pdf | 2026-11-03 | READ (tolerance +-30%) |
| Depth, 3y and 7y | 30, 30 | not in the posts | flat interpolation between the readings | 2026-11-03 | DERIVED |
| Depth, 20y and 30y | 15, 12 | not in the posts | 0.5x and 0.4x of the 10y level (0.4 is the ratio of the earlier 30y to 10y placeholders) | 2026-11-03 | ASSUMPTION |
| Price impact per $100mm, 2y, 5y, 10y (`impact.price_impact_32nds_per_100mm`) | 0.15, 0.30, 0.80 (32nds) | same two Fleming posts | Chart "Price Impact" (slope of one-minute price change on one-minute net order flow), Mar 2025 and mid-2024 level; 10y on the right axis (0 to 3) | 2026-11-03 | READ (tolerance +-30%, noisy daily series) |
| Price impact, 3y and 7y | 0.20, 0.50 | not in the posts | interpolation between 2y/5y and 5y/10y | 2026-11-03 | DERIVED |
| Price impact, 20y and 30y | 1.10, 1.40 | not in the posts | extrapolated proportionally to DV01 (about 8.5 32nds per $100mm per unit of DV01 per 100 face) | 2026-11-03 | ASSUMPTION |
| Impact shape and decay (`size_exponent`, `perm_frac`, `decay_half_life_min`) | 0.5, 0.4, 10 min | none | Fleming's measure is linear in net flow; the square-root form equals his slope at Q = 100mm and is a modelling choice | 2026-11-03 | ASSUMPTION |
| Leakage (`leakage.*`) | p_hedge_lose 0.35, phi_lose 0.25, windows 1-5 min and 5-30 min, `leakage_scale` 1.0 | none | placeholders kept | 2026-11-03 | ASSUMPTION |
| Dealer attribute ranges (`dealers.*`) | 12 dealers, tiers 4/5/3, ranges from the simulation design; `competition_widening_range` [0, 0.015]; quote floor 0.5 x hs x spread_mult; `competition_widening_scale` 1.0 | none | placeholders. Unspecified items (tier counts, selection weights, latency sigma, no-quote size range, inventory OU parameters) are my choices | 2026-11-03 | ASSUMPTION |
| Client tenor weights (`order_flow.tenor_weights`) | 2y 0.197, 3y 0.130, 5y 0.262, 7y 0.093, 10y 0.170, 20y 0.053, 30y 0.095 | https://cdn.finra.org/trace/treasury-aggregates/daily/ts-daily-aggregates-YYYY-MM-DD.xlsx (63 files) | Dealer-to-customer par value of nominal coupons by remaining-maturity bucket, summed over the period, OTR plus off-the-run rows | 2026-11-03 | DERIVED |
| TRACE reference statistics (`trace_reference.*`) | OTR share of D2C par 0.42 to 0.83; D2C average per day $22bn (20y) to $107bn (5y); mean D2C trade size $3.3mm to $6.3mm (all runs), $4.6mm to $8.3mm (OTR) | same FINRA files | same buckets; for docs and tests only | 2026-11-03 | DERIVED |
| Order sizes, parents per day, arrival mixture, execution policy, N-dealer policy, RFQ life cycle, traders | from the simulation design | none | placeholders kept. TRACE mean trade sizes are far below the desk tickets assumed here because TRACE includes small tickets; the desk sizes are not a TRACE reading | 2026-11-03 | ASSUMPTION |
| April 2025 episode (`composite_quote.april_2025_reference`) | spread peak 0.16 / 0.32 / 0.62; depth trough 12 / 9 / 12; price impact peak 0.40 / 0.80 / 1.55 | same two posts, 2025 charts (five-day averages around 2025-04-09) | peak or trough of each series, 2y / 5y / 10y; NY Fed speech: 10y depth fell to about one quarter of recent levels | 2026-11-03 | READ (tolerance +-10% to +-35%, see the yaml) |

Counts of tagged entries in `config/ust_sim.yaml` (one tag per parameter line or block header):
READ 9, DERIVED 28, ASSUMPTION 82.

## 2. Fleming chart readings

The posts publish charts only, with no data table. The images were downloaded and viewed; values are
five-day moving averages of BrokerTec interdealer data for the on-the-run notes.

| Measure (unit) | Tenor | Normal level (Mar 2025, mid-2024) | April 2025 peak or trough | Axis note |
|---|---|---|---|---|
| Bid-ask spread, full (32nds) | 2y | 0.14 | 0.16 | left axis |
| | 5y | 0.27 | 0.32 | left axis |
| | 10y | 0.54 | 0.62 | right axis, 0 to 1 |
| Depth, inside tier, per side ($mm) | 2y | 30 | 12 | log axis |
| | 5y | 30 | 9 | log axis |
| | 10y | 30 | 12 | log axis |
| Price impact (32nds per $100mm) | 2y | 0.15 | 0.40 | left axis |
| | 5y | 0.30 | 0.80 | left axis |
| | 10y | 0.80 | 1.55 | right axis, 0 to 3 |

Sanity check on the spread readings: the full spreads are about 1.0 to 1.1 BrokerTec ticks (2y 1/256 = 0.125,
5y 1/128 = 0.25, 10y 1/64 = 0.5 in 32nds). The 10y series sits on the right axis, which matters for impact
as well: reading the 10y price impact off the left axis gives 0.4, the earlier placeholder, a factor of two too low.

## 3. Where the readings differ from the placeholder values

1. **Half-spread.** The design placeholders list `hs_base` as 0.12 / 0.19 / 0.25 for 2y / 5y / 10y. Fleming's *full* spreads
   are 0.14 / 0.27 / 0.54, so the half-spreads used here are 0.07 / 0.135 / 0.27. The 2y placeholder was close
   to the full spread, not the half. `test_calibration_bounds.py` should compare against the config values.
2. **Depth.** Placeholders 150 / 80 / 50 are 2 to 5 times larger than the inside-tier depth he reports
   (about 30 in all three). Depth is per side at the best price and no multi-level scaling is applied.
3. **Price impact.** 10y 0.80 against the placeholder 0.40 (right axis, see above); 2y 0.15 and 5y 0.30
   against 0.10 and 0.25.
4. **April 2025 shape versus the single-factor `vol_mult`.** On the 2025-04-09 anchors the formula gives
   `vol_mult` = 1.84. Fleming's charts show the spread about 1.15 to 1.2 times normal, depth about 0.3 to 0.4
   times normal, price impact about 2 to 2.7 times normal. The formula therefore over-widens the spread and
   under-reduces depth relative to the public episode. This is reported for the Module 1 owner; the config keeps
   the design formula, and no parameter was tuned to reach a desired conclusion.

## 4. TreasuryDirect handling notes

- Tenor comes from `securityTerm`, not from the original term: a 10y reopening shows `9-Year 10-Month`
  and maps to 10y. The reopened 10y, 20y and 30y keep their CUSIP and stay one security.
- On 2025-02-25 the "5-Year" auction was a reopening of CUSIP 91282CGQ8 (maturity 2030-02-28, originally a 7y
  note issued in 2023). By the rule used it is the 5y on-the-run on 2025-03-03, with the January 2025 5y as OLD.
  This follows the data.
- A security is OTR from its auction date. The `when_issued` column in
  `universe_by_day.csv` is true between auction and issue date; on 2025-05-30 the 2y, 5y, 7y and 20y OTRs
  are when-issued (issue date 2025-06-02).
- The Note and Bond endpoints return 250 rows each and no TIPS or FRN rows, so the TIPS/FRN exclusion
  filter in the script removed nothing. Coverage check: 0 tenor-days with fewer than 3 ranked securities.
- The universe holds all seven tenors at three ranks; the 15 per day of the earlier design is the subset OTR for all tenors
  plus OLD and DOUBLE_OLD for 2y, 5y, 10y and 30y, which Module 1 selects.

## 5. What was unreachable or only partly reachable

- **FINRA Query API**: not used (it needs an account). Not needed: the daily aggregate files are public.
- **FINRA daily files**: the catalogue pages
  https://www.finra.org/finra-data/browse-catalog/about-treasury/daily-file and
  https://www.finra.org/finra-data/browse-catalog/about-treasury/daily-data link only the latest files, but
  the link pattern `https://cdn.finra.org/trace/treasury-aggregates/daily/ts-daily-aggregates-YYYY-MM-DD.xlsx`
  answers HTTP 200 for historical dates without a login. 63 of 65 weekdays were fetched. The two gaps,
  2025-04-18 (Good Friday) and 2025-05-26 (Memorial Day), return HTTP 403 from the CDN, which is what a
  date without a file returns; they are recorded in `finra_missing_days.json` and not retried.
  One catalogue page (the monthly aggregates) answered HTTP 429 during discovery and was not retried.
- **BLS schedule pages** (bls.gov): a scripted request with the plain research User-Agent returned HTTP 403. I did
  not work around it. The page text was read through a third-party page reader instead; the pages are the
  BLS ones.
- **Fleming posts**: no data table, only chart images; the 3y, 7y, 20y and 30y series are not shown, so
  those values are interpolated or extrapolated and labelled. A third post (Dyer and Fleming, 2026-04-02,
  https://libertystreeteconomics.newyorkfed.org/2026/04/treasury-market-liquidity-since-april-2025/) covers the same episode and
  was not read, since the 2025 post already gives the levels.
- **Off-the-run spreads, intraday spread or variance profiles, dealer behaviour, leakage parameters**:
  no public numeric source was found; those parameters stay ASSUMPTION.
- **FRED fallback** for the par curve is implemented in the script but was not needed.

## 6. Reproducing the raw data

```
TCA_DATA_RAW=data/raw python scripts/fetch_ust_anchors.py --start 2025-03-03 --end 2025-05-30
```

Files written to `data/raw/ust/` (gitignored): `par_yield_curve_2025.csv`, `par_yield_window.csv`,
`td_auctioned_note.json`, `td_auctioned_bond.json`, `securities_auctioned.csv`, `universe_by_day.csv`,
`finra_daily/*.xlsx`, `trace_daily_buckets.csv`, `trace_tenor_shares.json`, `finra_missing_days.json`,
`fetch_manifest.json`. Requests are sequential, 1.5 s apart, with the User-Agent `tca-stack-research/0.1`.
