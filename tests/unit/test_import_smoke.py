"""Import smoke tests for packaged layout."""

from __future__ import annotations

import pytest


def test_radius_rag_observability_import() -> None:
    import radius_rag.observability  # noqa: F401
    from radius_rag.observability.config import load_retrieval_config

    assert callable(load_retrieval_config)


def test_radius_rag_rag_import() -> None:
    from radius_rag.rag.service import RagService
    from radius_rag.rag.bundle import BundleStore

    assert RagService is not None
    assert BundleStore is not None


def test_legacy_retrieval_shims() -> None:
    pytest.importorskip("asyncpg")
    from retrieval_db import insert_events, run_migrations
    from retrieval_event import maybe_emit_retrieval_event

    assert callable(insert_events)
    assert callable(run_migrations)
    assert callable(maybe_emit_retrieval_event)
