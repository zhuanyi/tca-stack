"""``tca ust simulate --profile`` selects the run profile."""

from __future__ import annotations

from typer.testing import CliRunner

from tca.core.profile import is_ci
from tca.ust import cli, simulate

runner = CliRunner()


def test_profile_ci_is_applied_before_the_run(monkeypatch, tmp_path):
    monkeypatch.setenv("TCA_PROFILE", "full")
    seen = {}

    def fake_run(cfg, seed):
        seen["ci"] = is_ci()
        return {}

    monkeypatch.setattr(simulate, "run", fake_run)
    monkeypatch.setattr(simulate, "write", lambda tables, derived: None)
    monkeypatch.setattr(cli, "_derived", lambda: tmp_path)
    result = runner.invoke(cli.app, ["simulate", "--profile", "ci"])
    assert result.exit_code == 0, result.output
    assert seen["ci"] is True


def test_unknown_profile_is_rejected(monkeypatch):
    monkeypatch.setenv("TCA_PROFILE", "full")
    result = runner.invoke(cli.app, ["simulate", "--profile", "bogus"])
    assert result.exit_code != 0
