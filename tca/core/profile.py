"""Run profile: ``full`` for the published period, ``ci`` for a short smoke-test run."""

from __future__ import annotations

import os

_PROFILES = ("full", "ci")


def profile() -> str:
    """Profile name from ``$TCA_PROFILE``: ``"full"`` (default) or ``"ci"``."""
    name = os.environ.get("TCA_PROFILE", "full")
    if name not in _PROFILES:
        raise ValueError(f"TCA_PROFILE must be one of {_PROFILES}, got {name!r}")
    return name


def is_ci() -> bool:
    """True when running the short ``ci`` profile."""
    return profile() == "ci"
