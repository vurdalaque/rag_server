#!/usr/bin/env python3
"""Delete root *.py shims replaced by radius_rag._legacy_bootstrap."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEEP = frozenset(
    {
        "rag_server.py",
        "retrieval_evaluator.py",
        "conftest.py",
    }
)

def main() -> None:
    from radius_rag._legacy_bootstrap import LEGACY_MODULE_TARGETS

    removed = 0
    for name in sorted(LEGACY_MODULE_TARGETS):
        if name == "retrieval_evaluator":
            continue
        path = ROOT / f"{name}.py"
        if path.exists():
            path.unlink()
            removed += 1
            print("removed", path.name)
    for path in ROOT.glob("*.py"):
        if path.name in KEEP:
            continue
        if path.name.startswith("test_"):
            continue
        # Unknown root module — leave for manual review
    print(f"removed {removed} legacy shim files")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    main()
