"""Pytest hooks for rag_server."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Совпадает с production-дефолтом: Streamable HTTP без сессий + JSON-тело на POST.
os.environ.setdefault("MCP_JSON_RESPONSE", "true")

_PROJECT_ROOT = Path(__file__).resolve().parent
_BASETEMP = _PROJECT_ROOT / ".pytest_tmp"


def pytest_configure(config: pytest.Config) -> None:
    _BASETEMP.mkdir(exist_ok=True)
    config.option.basetemp = str(_BASETEMP)
