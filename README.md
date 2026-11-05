# tca-stack

Transaction cost analysis for US Treasuries and spot FX, run on simulated order flow.

The repository holds a shared core library and two modules built on it. Neither module uses a
proprietary data feed. The Treasury market is simulated, calibrated to public numbers. The FX module replays
simulated orders against historical Dukascopy quotes that the author downloaded once for personal research.

## Status

| module | contents | state |
|---|---|---|
| 00 core | sign conventions, bond maths, calendar, storage, impact state, statistics, plotting, CLI | released |
| 01 UST | dealer-to-client RFQ simulator and TCA, figures, `docs/ust.md` | released, raw simulated level columns not committed (see Data) |
| 02 FX | tick replay engine, last look, passive fills, algo wheel, TCA, figures, `docs/fx.md` | released, without price-level tables (see Data) |
| 03 and research notes | not part of this release | private |

## Layout

| path | contents |
|---|---|
| `tca/core/` | bond maths, trading calendar, conventions, storage, impact state, statistics |
| `tca/plotting/` | figure style and shared plots |
| `tca/ust/` | Module 01 simulator, TCA and figures |
| `tca/fx/` | Module 02 tick source, replay engine, TCA and figures |
| `tca/cli.py` | command line entry point, `tca ust ...` and `tca fx ...` |
| `config/` | simulator parameters, each with its source or an `ASSUMPTION` label |
| `docs/ust.md`, `docs/fx.md` | findings, method and schema for each module |
| `docs/ust_calibration.md`, `docs/fx_calibration.md` | where each calibrated number comes from |
| `docs/figures/` | the committed figures |
| `data/derived/` | committed simulated tables |
| `scripts/` | fetch scripts for public anchors and Dukascopy ticks |
| `tests/` | unit and integration tests |

## Running it

```
uv sync --locked
uv run ruff check .
uv run pytest -q
```

CI sets `TCA_PROFILE=ci`, which runs the pipelines on small synthetic fixtures. `make test` and `make lint`
run the same checks locally.

Some Makefile targets need inputs that this release does not contain. `make data`, `make simulate`,
`make tca` and `make figures` start from raw inputs under `data/raw/` (git-ignored): run
`scripts/fetch_ust_anchors.py` for the Treasury inputs and `tca fx fetch-ticks` for FX ticks first. The
post-trade stages `tca ust tca`, `tca ust summary` and `tca fx tca`, `figures`, `summary` read price and
yield-level columns that are not committed. `tca ust simulate` rebuilds them from the published inputs; run it first, then
`tca ust tca` and `tca ust summary`. For FX run `tca fx fetch-ticks`, `tca fx simulate`, then `tca fx tca`. On the published
tables alone only `tca ust figures` runs for UST, and none of the FX post-trade commands do. `make notes` and
`make reproduce` call research-note scripts under `notes/`, which are not released, so they fail here.

## Data

No tick or intraday market data is committed, and none of the data comes from an employer. The only
market values committed directly are the public daily Treasury `close_yield` anchors.

- UST: the intraday market is simulated; the daily yield anchors come from public sources named in
  `docs/ust_calibration.md`. The UST data is simulated. The committed UST tables hold simulated orders, RFQs and costs in
  basis points, 32nds and dollars, plus the security reference table and the daily Treasury `close_yield` anchors. The raw
  simulated level columns (`ust_bars_1m`, every price column, `exec_yld`, `quote_yld`, `mid_yld_*`) are not committed;
  `tca ust simulate` rebuilds them. UST levels can also be derived from the published cost columns and the public
  security terms, and nothing here tries to prevent that.
- FX: raw Dukascopy ticks stay local and git-ignored and are never redistributed. The committed FX tables
  hold simulated orders and costs in basis points and pips. One-minute bars and the columns that hold a price
  level directly are not committed. FX price levels can still be derived from the published cost columns, for
  example the pips and bp pair in `fx_tca_fill`, so the omission limits convenience and does not hide the levels.

Dukascopy data is used for personal, non-commercial research. Anyone who repeats the download accepts
Dukascopy's terms themselves. The BIS Triennial Survey 2025 supplies the FX volume proxy; see
`docs/fx_calibration.md`.

## Licence

MIT.
