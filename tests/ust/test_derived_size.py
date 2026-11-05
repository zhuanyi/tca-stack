from tca.core import storage

MAX_FILE = 25 * 1024 * 1024
MAX_TOTAL = 150 * 1024 * 1024


def test_committed_tables_respect_the_size_limits():
    derived = storage.data_dir() / "derived"
    files = [p for p in derived.rglob("*.parquet")]
    assert files
    assert max(p.stat().st_size for p in files) <= MAX_FILE
    assert sum(p.stat().st_size for p in files) <= MAX_TOTAL
