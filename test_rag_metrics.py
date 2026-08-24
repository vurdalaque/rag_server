from __future__ import annotations

import httpx
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

import rag_metrics
import rag_server


def test_metrics_endpoint_returns_prometheus_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Path(__file__).resolve().parent / ".test_metrics_run"
    workspace.mkdir(exist_ok=True)
    monkeypatch.setenv("RAG_METRICS_ENABLED", "true")
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "test-token")
    monkeypatch.setenv("RAG_STAGING_DIR", str(workspace / "staging"))
    monkeypatch.setenv("RAG_BUNDLE_STATE_DIR", str(workspace / "state"))
    monkeypatch.setenv("RAG_INDEX_FILE", str(workspace / "missing.faiss"))
    monkeypatch.setenv("RAG_METADATA_FILE", str(workspace / "missing.jsonl"))
    monkeypatch.setenv("RAG_METRICS_PROBE_ENABLED", "false")
    rag_server.rag_service.clear()

    client = TestClient(rag_server.app)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "rag_index_loaded" in response.text
    assert "# HELP rag_index_loaded" in response.text


def test_classify_httpx_connect_error() -> None:
    error = httpx.ConnectError("connection failed")
    assert rag_metrics.classify_error(error) == "connect"


def test_classify_httpx_timeout_error() -> None:
    error = httpx.ReadTimeout("timed out")
    assert rag_metrics.classify_error(error) == "timeout"


def test_update_index_state_sets_gauges() -> None:
    service = rag_server.rag_service
    service.clear()
    rag_metrics.update_index_state(service)

    payload = rag_metrics.metrics_payload().decode("utf-8")
    assert "rag_index_loaded 0.0" in payload
