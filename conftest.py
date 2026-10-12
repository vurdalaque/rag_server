"""Pytest hooks for rag_server."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Совпадает с production-дефолтом: Streamable HTTP без сессий + JSON-тело на POST.
os.environ.setdefault("MCP_JSON_RESPONSE", "true")

_PROJECT_ROOT = Path(__file__).resolve().parent
_BASETEMP = Path(tempfile.gettempdir()) / "radius_rag_server_pytest"


def pytest_configure(config: pytest.Config) -> None:
    _BASETEMP.mkdir(exist_ok=True)
    config.option.basetemp = str(_BASETEMP)
