"""Where the FX module reads and writes, for the ``full`` and ``ci`` profiles.

The ``full`` profile reads raw ticks from ``<raw>/fx`` and writes to ``<data>/derived`` and
``docs/figures``. The ``ci`` profile reads the synthetic fixture and writes under ``<data>/ci`` so a
smoke run never touches committed outputs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tca.core import storage
from tca.core.profile import is_ci

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fx_ticks_synthetic_2d.parquet"


def fixture_path() -> Path:
    """The synthetic tick fixture: ``$TCA_FX_FIXTURE`` if set, else the committed file."""
    env = os.environ.get("TCA_FX_FIXTURE")
    return Path(env) if env else FIXTURE


def raw_fx_dir() -> Path:
    """Folder of cached Dukascopy ticks, ``<raw>/fx``."""
    return storage.raw_dir() / "fx"


def derived_dir() -> Path:
    """Folder the parquet tables go to for the active profile."""
    base = storage.data_dir()
    return base / "ci" / "derived" if is_ci() else base / "derived"


def figures_dir() -> Path:
    """Folder the figures go to: ``$TCA_FIGURES_DIR``, else by profile."""
    env = os.environ.get("TCA_FIGURES_DIR")
    if env:
        return Path(env)
    if is_ci():
        return storage.data_dir() / "ci" / "figures"
    return REPO_ROOT / "docs" / "figures"


@contextmanager
def figures_env() -> Iterator[Path]:
    """Point ``tca.plotting`` at :func:`figures_dir` for the duration of the block."""
    target = figures_dir()
    previous = os.environ.get("TCA_FIGURES_DIR")
    os.environ["TCA_FIGURES_DIR"] = str(target)
    try:
        yield target
    finally:
        if previous is None:
            os.environ.pop("TCA_FIGURES_DIR", None)
        else:
            os.environ["TCA_FIGURES_DIR"] = previous
