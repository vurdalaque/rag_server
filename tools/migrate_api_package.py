#!/usr/bin/env python3
"""Copy rag_server.py into src/radius_rag/api/server.py."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src" / "radius_rag" / "api"
SERVER = DEST / "server.py"
SHIM = ROOT / "rag_server.py"

HEADER = '''"""Backward-compatible shim; uvicorn rag_server:app."""
from radius_rag.api.server import *  # noqa: F403
'''


def main() -> None:
    src = ROOT / "rag_server.py"
    text = src.read_text(encoding="utf-8")
    if "radius_rag.api.server" in text and len(text) < 300:
        raise SystemExit("rag_server.py is already a shim; restore server.py first")

    DEST.mkdir(parents=True, exist_ok=True)
    init = DEST / "__init__.py"
    if not init.exists():
        init.write_text('"""FastAPI application and MCP wiring."""\n', encoding="utf-8")

    SERVER.write_text(text, encoding="utf-8")
    SHIM.write_text(HEADER, encoding="utf-8")
    print(f"wrote {SERVER} and shim {SHIM}")


if __name__ == "__main__":
    main()
