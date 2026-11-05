from tca.ust import simulate


def _bytes(tables, folder) -> dict[str, bytes]:
    simulate.write(tables, folder)
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*.parquet"))
    }


def test_same_seed_gives_identical_parquet_and_other_seeds_differ(
    cfg, inputs, three_days, tmp_path
):
    over = simulate.Overrides(days=three_days[:2])
    a = _bytes(simulate.run(cfg, 11, over, inputs), tmp_path / "a")
    b = _bytes(simulate.run(cfg, 11, over, inputs), tmp_path / "b")
    c = _bytes(simulate.run(cfg, 12, over, inputs), tmp_path / "c")
    assert a == b
    assert a["ust_rfq.parquet"] != c["ust_rfq.parquet"]
