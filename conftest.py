"""Pytest: ensure src/ is on sys.path when collecting from repo root."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from radius_rag._legacy_bootstrap import install_legacy_imports

install_legacy_imports()
