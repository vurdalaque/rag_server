"""Lazy asyncpg import so observability code can load without the driver installed."""

from __future__ import annotations

from typing import Any


def load_asyncpg() -> Any:
    try:
        import asyncpg
    except ImportError as exc:
        raise RuntimeError(
            "asyncpg is required for retrieval observability storage. "
            "Install the radius-rag-server package or run: pip install asyncpg"
        ) from exc
    return asyncpg
