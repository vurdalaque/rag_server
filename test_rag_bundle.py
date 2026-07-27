from __future__ import annotations

import asyncio

import json
import zipfile
from pathlib import Path

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

import rag_service as rag_service_module
from rag_service import RagService

import rag_server
from rag_bundle import BUNDLE_FILES, BundleValidationError, file_sha256, stage_bundle


def make_bundle(tmp_path: Path, corrupt_checksum: bool = False) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    index_path = source_dir / "rag_index.faiss"
    metadata_path = source_dir / "rag_metadata.jsonl"
    manifest_path = source_dir / "rag_manifest.json"

    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0]], dtype=np.float32))
    faiss.write_index(index, str(index_path))
    metadata_path.write_text('{"file": "example.py"}\n', encoding="utf-8")

    checksums = {
        index_path.name: file_sha256(index_path),
        metadata_path.name: file_sha256(metadata_path),
    }

    if corrupt_checksum:
        checksums[metadata_path.name] = "0" * 64

    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "index_file": index_path.name,
                "metadata_file": metadata_path.name,
                "checksums": checksums,
                "document_count": 1,
                "dimensions": 2,
                "embedding_model": "test-model",
                "built_at": "2026-07-26T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    archive_path = tmp_path / "bundle.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in (index_path, metadata_path, manifest_path):
            archive.write(path, arcname=path.name)

    return archive_path


def test_stage_bundle_accepts_valid_bundle(tmp_path: Path) -> None:
    bundle_dir, manifest = stage_bundle(
        make_bundle(tmp_path),
        tmp_path / "staging",
        1024 * 1024,
    )

    assert {path.name for path in bundle_dir.iterdir()} == BUNDLE_FILES
    assert manifest["document_count"] == 1
    assert manifest["dimensions"] == 2


def test_stage_bundle_removes_invalid_bundle(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"

    with pytest.raises(BundleValidationError, match="Checksum mismatch"):
        stage_bundle(make_bundle(tmp_path, corrupt_checksum=True), staging_dir, 1024 * 1024)

    assert list(staging_dir.iterdir()) == []


def test_upload_requires_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "test-token")
    response = TestClient(rag_server.app).post(
        "/admin/index/upload",
        content=b"",
        headers={"content-type": "application/zip"},
    )

    assert response.status_code == 401


def test_upload_stages_valid_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging_dir = tmp_path / "staging"
    active_index = tmp_path / "active.faiss"
    active_index.write_bytes(b"active")
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "test-token")
    monkeypatch.setenv("RAG_STAGING_DIR", str(staging_dir))

    response = TestClient(rag_server.app).post(
        "/admin/index/upload",
        content=make_bundle(tmp_path).read_bytes(),
        headers={
            "authorization": "Bearer test-token",
            "content-type": "application/zip",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "staged"
    assert {path.name for path in (staging_dir / payload["staging_id"]).iterdir()} == BUNDLE_FILES
    assert active_index.read_bytes() == b"active"


def test_admin_activate_reload_and_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging_dir = tmp_path / "staging"
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "test-token")
    monkeypatch.setenv("RAG_STAGING_DIR", str(staging_dir))
    monkeypatch.setenv("RAG_BUNDLE_STATE_DIR", str(tmp_path / "state"))
    headers = {"authorization": "Bearer test-token", "content-type": "application/zip"}
    client = TestClient(rag_server.app)
    bundle = make_bundle(tmp_path).read_bytes()

    first = client.post("/admin/index/upload", content=bundle, headers=headers).json()
    activated = client.post(f"/admin/index/activate/{first['staging_id']}", headers=headers)

    assert activated.status_code == 200
    assert activated.json()["status"] == "active"
    assert client.post("/admin/index/reload", headers=headers).json()["status"] == "reloaded"

    second = client.post("/admin/index/upload", content=bundle, headers=headers).json()
    assert client.post(f"/admin/index/activate/{second['staging_id']}", headers=headers).status_code == 200

    status = client.get("/admin/index/status", headers={"authorization": "Bearer test-token"})
    assert status.status_code == 200
    assert status.json()["active_staging_id"] == second["staging_id"]
    assert status.json()["previous_staging_id"] == first["staging_id"]

    rolled_back = client.post("/admin/index/rollback", headers=headers)
    assert rolled_back.status_code == 200
    assert rolled_back.json()["staging_id"] == first["staging_id"]


def test_admin_status_requires_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "test-token")

    assert TestClient(rag_server.app).get("/admin/index/status").status_code == 401


def make_rag_service(metadata: list[dict[str, object]], vectors: list[list[float]]) -> RagService:
    service = RagService()
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray(vectors, dtype=np.float32))
    service.index = index
    service.metadata = metadata
    service._build_bm25_index()
    return service


def metadata_item(
    file_name: str,
    source_type: str,
    language: str,
    code: str,
) -> dict[str, object]:
    return {
        "file": file_name,
        "source_type": source_type,
        "language": language,
        "full_name": file_name,
        "detail": "",
        "start_line": 1,
        "end_line": 2,
        "code": code,
    }


def test_retrieve_applies_source_type_language_and_path_prefix_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_rag_service(
        [
            metadata_item("src/main.py", "code", "python", "alpha"),
            metadata_item("docs/guide.md", "documentation", "markdown", "alpha"),
            metadata_item("src/test_main.py", "code", "python", "alpha"),
        ],
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    )

    async def embedding(*_: object) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(service, "get_embedding", embedding)
    monkeypatch.setattr(rag_service_module, "RERANK_ENABLED", False)

    docs = asyncio.run(service.retrieve("alpha", source_type="documentation"))
    python = asyncio.run(service.retrieve("alpha", language="python"))
    source = asyncio.run(service.retrieve("alpha", path_prefix="src\\"))

    assert [result["file"] for result in docs] == ["docs/guide.md"]
    assert {result["file"] for result in python} == {"src/main.py", "src/test_main.py"}
    assert {result["file"] for result in source} == {"src/main.py", "src/test_main.py"}


def test_retrieve_applies_source_type_boost_to_hybrid_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_rag_service(
        [
            metadata_item("src/main.py", "code", "python", "other"),
            metadata_item("docs/guide.md", "documentation", "markdown", "needle"),
        ],
        [[1.0, 0.0], [0.9, 0.1]],
    )

    async def embedding(*_: object) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(service, "get_embedding", embedding)
    monkeypatch.setattr(rag_service_module, "RERANK_ENABLED", False)
    monkeypatch.setattr(rag_service_module, "SOURCE_TYPE_BOOSTS", {"documentation": 3.0})

    results = asyncio.run(service.retrieve("needle", top_k=2))

    assert results[0]["file"] == "docs/guide.md"
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]


def test_retrieve_combines_bm25_and_faiss_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_rag_service(
        [
            metadata_item("src/main.py", "code", "python", "other"),
            metadata_item("src/near.py", "code", "python", "other"),
            metadata_item("docs/needle.md", "documentation", "markdown", "needle needle"),
        ],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )

    async def embedding(*_: object) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(service, "get_embedding", embedding)
    monkeypatch.setattr(rag_service_module, "RERANK_ENABLED", False)

    results = asyncio.run(service.retrieve("needle", top_k=2))
    bm25_only = next(result for result in results if result["file"] == "docs/needle.md")

    assert bm25_only["faiss_score"] is None
    assert bm25_only["bm25_score"] > 0
    assert bm25_only["score"] == bm25_only["hybrid_score"]


def test_build_context_includes_scores_and_source_metadata() -> None:
    context, sources = RagService.build_context(
        [
            {
                "score": 0.9,
                "rerank_score": 0.8,
                "faiss_score": 0.7,
                "bm25_score": 1.2,
                "source_type": "documentation",
                "language": "markdown",
                "file": "docs/guide.md",
                "symbol": "guide",
                "signature": "",
                "start_line": 1,
                "end_line": 2,
                "code": "# Guide",
            }
        ]
    )

    assert "Rerank relevance: 0.800000" in context
    assert "FAISS similarity: 0.700000" in context
    assert "BM25 score: 1.200000" in context
    assert "Source type: documentation" in context
    assert sources == [
        {
            "score": 0.9,
            "rerank_score": 0.8,
            "faiss_score": 0.7,
            "file": "docs/guide.md",
            "source_type": "documentation",
            "symbol": "guide",
            "start_line": 1,
            "end_line": 2,
        }
    ]


def test_public_search_entry_points_forward_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieve_calls: list[dict[str, object]] = []
    ask_calls: list[dict[str, object]] = []
    prepare_calls: list[dict[str, object]] = []

    async def retrieve(**kwargs: object) -> list[dict[str, object]]:
        retrieve_calls.append(kwargs)
        return []

    async def ask(**kwargs: object) -> dict[str, object]:
        ask_calls.append(kwargs)
        return {}

    async def prepare_messages(**kwargs: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        prepare_calls.append(kwargs)
        return [], []

    class FakeRequest:
        async def json(self) -> dict[str, object]:
            return {
                "messages": [{"role": "user", "content": "question"}],
                "stream": True,
                "rag_source_type": "documentation",
                "rag_language": "markdown",
                "rag_path_prefix": "docs/",
            }

    monkeypatch.setattr(rag_server.rag_service, "retrieve", retrieve)
    monkeypatch.setattr(rag_server.rag_service, "ask", ask)
    monkeypatch.setattr(rag_server.rag_service, "prepare_messages", prepare_messages)

    asyncio.run(rag_server.search_project("query", source_type="documentation", language="markdown", path_prefix="docs/"))
    asyncio.run(rag_server.ask_project("question", source_type="documentation", language="markdown", path_prefix="docs/"))
    asyncio.run(rag_server.debug_search("query", source_type="documentation", language="markdown", path_prefix="docs/"))
    asyncio.run(rag_server.chat_completions(FakeRequest()))

    for call in retrieve_calls:
        assert call["source_type"] == "documentation"
        assert call["language"] == "markdown"
        assert call["path_prefix"] == "docs/"
    assert ask_calls[0]["source_type"] == "documentation"
    assert ask_calls[0]["language"] == "markdown"
    assert ask_calls[0]["path_prefix"] == "docs/"
    assert prepare_calls[0]["source_type"] == "documentation"
    assert prepare_calls[0]["language"] == "markdown"
    assert prepare_calls[0]["path_prefix"] == "docs/"



