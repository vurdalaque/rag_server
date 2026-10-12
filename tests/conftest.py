"""Pytest configuration for tests/ (mirrors package-root conftest)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from radius_rag._legacy_bootstrap import install_legacy_imports

install_legacy_imports()

os.environ.setdefault("MCP_JSON_RESPONSE", "true")

_BASETEMP = Path(tempfile.gettempdir()) / "radius_rag_server_pytest"


def pytest_configure(config: pytest.Config) -> None:
    _BASETEMP.mkdir(exist_ok=True)
    config.option.basetemp = str(_BASETEMP)
