from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent
_BASETEMP = _PROJECT_ROOT / ".pytest_tmp"


def pytest_configure(config: pytest.Config) -> None:
    _BASETEMP.mkdir(exist_ok=True)
    config.option.basetemp = str(_BASETEMP)
