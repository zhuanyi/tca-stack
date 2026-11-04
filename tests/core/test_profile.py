import pytest

from tca.core import profile as pf


def test_default_is_full(monkeypatch):
    monkeypatch.delenv("TCA_PROFILE", raising=False)
    assert pf.profile() == "full"
    assert not pf.is_ci()


def test_ci_profile(monkeypatch):
    monkeypatch.setenv("TCA_PROFILE", "ci")
    assert pf.profile() == "ci"
    assert pf.is_ci()


def test_unknown_profile_raises(monkeypatch):
    monkeypatch.setenv("TCA_PROFILE", "fast")
    with pytest.raises(ValueError, match="TCA_PROFILE"):
        pf.profile()
