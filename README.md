# tca-stack

Transaction cost analysis and execution research for US Treasuries and spot FX, built the way a buy-side trading desk would build it.

**Status: in progress.** This first release is the shared library: sign conventions, bond maths, trading calendar, parquet and DuckDB storage, market impact state, statistics and figure style. The Treasury RFQ module, the FX replay module and the research notes follow.

Nothing here comes from any employer. All data will be public or simulated by code in this repository.

## Run it

```
uv sync
uv run pytest -q
```

## Licence

MIT
