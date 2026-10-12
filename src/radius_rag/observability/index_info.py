"""Active index metadata for retrieval telemetry."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radius_rag.rag.bundle import BundleStore

RRF_CONSTANT = 60


@dataclass(frozen=True)
class ActiveIndexInfo:
    staging_id: str | None
    index_fingerprint: str | None
    embedding_model: str | None
    document_count: int | None
    dimensions: int | None
    manifest: dict[str, Any] | None
    source: str


def _bundle_store() -> BundleStore:
    return BundleStore(
        Path(os.getenv("RAG_STAGING_DIR", "rag_staging")),
        Path(os.getenv("RAG_BUNDLE_STATE_DIR", "rag_bundle_state")),
    )


def compute_index_fingerprint(manifest: dict[str, Any], staging_id: str) -> str:
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict):
        checksums = {}
    payload = {
        "staging_id": staging_id,
        "built_at": manifest.get("built_at"),
        "document_count": manifest.get("document_count"),
        "dimensions": manifest.get("dimensions"),
        "embedding_model": manifest.get("embedding_model"),
        "checksums": checksums,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def chunk_code_hash(metadata_item: dict[str, Any]) -> str:
    raw = metadata_item.get("code_hash")
    if isinstance(raw, str) and len(raw) == 64:
        return raw
    code = metadata_item.get("code", "")
    if not isinstance(code, str):
        code = str(code)
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def chunk_stable_id(
    metadata_item: dict[str, Any],
    metadata_index: int,
    index_fingerprint: str,
) -> str:
    payload = {
        "index_fingerprint": index_fingerprint,
        "metadata_index": metadata_index,
        "file": metadata_item.get("file", ""),
        "repo": metadata_item.get("repo", ""),
        "start_line": metadata_item.get("start_line"),
        "end_line": metadata_item.get("end_line"),
        "code_hash": chunk_code_hash(metadata_item),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_active_index_info(rag_service: Any | None = None) -> ActiveIndexInfo:
    active = _bundle_store().active_bundle()
    if active is not None:
        path, manifest = active
        staging_id = path.name
        fingerprint = compute_index_fingerprint(manifest, staging_id)
        return ActiveIndexInfo(
            staging_id=staging_id,
            index_fingerprint=fingerprint,
            embedding_model=str(manifest.get("embedding_model", "")),
            document_count=int(manifest.get("document_count", 0))
            if manifest.get("document_count") is not None
            else None,
            dimensions=int(manifest.get("dimensions", 0))
            if manifest.get("dimensions") is not None
            else None,
            manifest=manifest,
            source="bundle",
        )

    if rag_service is not None:
        index = getattr(rag_service, "index", None)
        metadata = getattr(rag_service, "metadata", None)
        if index is not None and isinstance(metadata, list) and metadata:
            embedding_model = os.getenv("EMBEDDING_MODEL", "")
            pseudo_manifest = {
                "document_count": int(index.ntotal),
                "dimensions": int(index.d),
                "embedding_model": embedding_model,
                "built_at": "filesystem",
                "checksums": {},
            }
            staging_id = Path(
                os.getenv("RAG_INDEX_FILE", "rag_index.faiss"),
            ).resolve().as_posix()
            fingerprint = compute_index_fingerprint(pseudo_manifest, staging_id)
            return ActiveIndexInfo(
                staging_id=staging_id,
                index_fingerprint=fingerprint,
                embedding_model=embedding_model or None,
                document_count=int(index.ntotal),
                dimensions=int(index.d),
                manifest=pseudo_manifest,
                source="filesystem",
            )

    return ActiveIndexInfo(
        staging_id=None,
        index_fingerprint=None,
        embedding_model=None,
        document_count=None,
        dimensions=None,
        manifest=None,
        source="none",
    )
