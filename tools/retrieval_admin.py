#!/usr/bin/env python3
"""Repo-local launcher: python tools/retrieval_admin.py ops"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from radius_rag.observability.retrieval_admin_cli import main

if __name__ == "__main__":
    main()
