"""Pytest hooks for rag_server."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Stateful MCP (default) is required for long tools/call with json_response=true.
# Must be set before rag_server is first imported.
os.environ.setdefault("MCP_STATELESS_HTTP", "false")

_PROJECT_ROOT = Path(__file__).resolve().parent
_BASETEMP = _PROJECT_ROOT / ".pytest_tmp"


def pytest_configure(config: pytest.Config) -> None:
    _BASETEMP.mkdir(exist_ok=True)
    config.option.basetemp = str(_BASETEMP)
