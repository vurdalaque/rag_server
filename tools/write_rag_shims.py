#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SHIMS = {
    "rag_metrics.py": "radius_rag.rag.metrics",
    "rag_bundle.py": "radius_rag.rag.bundle",
    "rag_service.py": "radius_rag.rag.service",
    "rag_search.py": "radius_rag.rag.search",
    "rag_proxy.py": "radius_rag.rag.proxy",
}

HEADER = '"""Backward-compatible shim; see radius_rag.rag."""\n\n'

for name, mod in SHIMS.items():
    (ROOT / name).write_text(
        HEADER + f"from {mod} import *  # noqa: F403\n",
        encoding="utf-8",
    )

print("wrote", len(SHIMS), "rag shims")
