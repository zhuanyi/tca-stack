import sys
import types

import pandas as pd
import typer
from typer.testing import CliRunner

from tca import cli
from tca.core import storage

runner = CliRunner()


def test_db_build_creates_tables(tmp_path):
    derived = tmp_path / "derived"
    storage.write_table(
        pd.DataFrame({"ts_utc": pd.date_range("2025-03-03", periods=3, tz="UTC"), "v": [1, 2, 3]}),
        "alpha",
        derived,
    )
    db = tmp_path / "out" / "tca.duckdb"
    result = runner.invoke(
        cli.app, ["db", "build", "--derived", str(derived), "--db-path", str(db)]
    )
    assert result.exit_code == 0, result.output
    assert "alpha: 3 rows" in result.output
    assert db.exists()


def test_module_sub_apps_are_discovered(monkeypatch):
    sub = typer.Typer()

    @sub.command("hello")
    def hello() -> None:
        typer.echo("hello from fx")

    package = types.ModuleType("tca.fx")
    package.__path__ = []
    module = types.ModuleType("tca.fx.cli")
    module.app = sub
    monkeypatch.setitem(sys.modules, "tca.fx", package)
    monkeypatch.setitem(sys.modules, "tca.fx.cli", module)

    result = runner.invoke(cli.build_app(), ["fx", "hello"])
    assert result.exit_code == 0, result.output
    assert "hello from fx" in result.output


def test_root_app_registers_db_group():
    names = [g.name for g in cli.build_app().registered_groups]
    assert "db" in names
