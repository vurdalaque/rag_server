#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SHIMS = {
    "llm_params.py": "radius_rag.llm.params",
    "llm_client.py": "radius_rag.llm.client",
}

HEADER = '"""Backward-compatible shim; see radius_rag.llm."""\n\n'

for name, mod in SHIMS.items():
    (ROOT / name).write_text(
        HEADER + f"from {mod} import *  # noqa: F403\n",
        encoding="utf-8",
    )

print("wrote", len(SHIMS), "llm shims")
