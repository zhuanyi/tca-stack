# tca-stack

Transaction cost analysis for US Treasuries and spot FX.

This repository holds the shared pieces the TCA work sits on: sign conventions, bond maths, a trading
calendar, parquet and DuckDB storage, a market-impact state, statistics, figure style and a small CLI.

**Status: module 00 only.** The Treasury RFQ module and the FX replay module are still being built and
stay in a private repository until they are finished and reviewed.

## Layout

| path | contents |
|---|---|
| `tca/core/bondmath.py` | price, yield, accrued, 32nds formatting for USTs |
| `tca/core/conventions.py` | sign conventions and markout definitions used throughout |
| `tca/core/calendar.py` | SIFMA trading calendar and settlement dates |
| `tca/core/storage.py` | parquet and DuckDB writers with size-aware part files |
| `tca/core/impact.py` | decaying permanent and transient impact state |
| `tca/core/stats.py` | bootstrap intervals, shrinkage, chunked aggregation |
| `tca/plotting/` | figure style and helper plots |
| `tca/cli.py` | command line entry points |
| `tests/` | unit tests for all of the above |

## Running it

```
uv sync
uv run pytest -q
```

`make reproduce` runs the same sequence the CI does: sync, lint, format check, tests.

## Data

No market data is committed to this repository, and none of it comes from an employer.

The FX work uses a one-off historical tick download kept local and git-ignored; only derived figures
and aggregated statistics are published. Raw ticks are never redistributed. Market data attribution
appears in the FX module's docs when that module is released.

## Licence

MIT.
