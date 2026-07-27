"""Проверка, staging и активация загружаемых RAG bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import faiss


BUNDLE_FILES = {"rag_index.faiss", "rag_metadata.jsonl", "rag_manifest.json"}


class BundleValidationError(ValueError):
    """Bundle не соответствует контракту доставки."""


class BundleStore:
    """Хранит атомарно переключаемые ссылки на staged bundle."""

    def __init__(self, staging_dir: Path, state_dir: Path) -> None:
        self.staging_dir = staging_dir
        self.state_dir = state_dir

    def _marker_path(self, name: str) -> Path:
        return self.state_dir / f"{name}.json"

    def _read_marker(self, name: str) -> str | None:
        try:
            value = json.loads(self._marker_path(name).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            raise BundleValidationError(f"Invalid {name} bundle marker: {error}") from error

        if not isinstance(value, dict) or not isinstance(value.get("staging_id"), str):
            raise BundleValidationError(f"Invalid {name} bundle marker")

        return value["staging_id"]

    def _write_marker(self, name: str, staging_id: str | None) -> None:
        marker = self._marker_path(name)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        if staging_id is None:
            marker.unlink(missing_ok=True)
            return

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=f".{name}-",
            dir=self.state_dir,
            delete=False,
        ) as file:
            json.dump({"staging_id": staging_id}, file)
            file.flush()
            os.fsync(file.fileno())
            temporary_path = Path(file.name)

        os.replace(temporary_path, marker)

    def _bundle_path(self, staging_id: str) -> Path:
        candidate = self.staging_dir / staging_id

        if candidate.parent != self.staging_dir or not candidate.is_dir():
            raise BundleValidationError("Staged bundle was not found")

        return candidate

    def active_bundle(self) -> tuple[Path, dict[str, Any]] | None:
        staging_id = self._read_marker("active")

        if staging_id is None:
            return None

        path = self._bundle_path(staging_id)
        return path, validate_staged_bundle(path)

    def status(self) -> dict[str, Any]:
        active_id = self._read_marker("active")
        previous_id = self._read_marker("previous")
        active = self.active_bundle()

        return {
            "active_staging_id": active_id,
            "previous_staging_id": previous_id,
            "active_manifest": active[1] if active else None,
        }

    def activate(self, staging_id: str) -> tuple[Path, dict[str, Any]]:
        path = self._bundle_path(staging_id)
        manifest = validate_staged_bundle(path)
        active_id = self._read_marker("active")

        if active_id != staging_id:
            self._write_marker("previous", active_id)
            self._write_marker("active", staging_id)

        return path, manifest

    def rollback(self) -> tuple[Path, dict[str, Any]]:
        active_id = self._read_marker("active")
        previous_id = self._read_marker("previous")

        if active_id is None or previous_id is None:
            raise BundleValidationError("No previous bundle is available for rollback")

        path = self._bundle_path(previous_id)
        manifest = validate_staged_bundle(path)
        self._write_marker("previous", active_id)
        self._write_marker("active", previous_id)
        return path, manifest


def file_sha256(path: Path) -> str:
    """Возвращает SHA-256 содержимого файла."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def load_metadata(path: Path) -> list[dict[str, Any]]:
    """Загружает и проверяет JSONL metadata."""
    metadata: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise BundleValidationError(
                    f"Invalid metadata JSON at line {line_number}: {error}"
                ) from error

            if not isinstance(item, dict):
                raise BundleValidationError(
                    f"Invalid metadata object at line {line_number}"
                )

            metadata.append(item)

    return metadata


def _require_int(manifest: dict[str, Any], name: str, minimum: int) -> int:
    value = manifest.get(name)

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BundleValidationError(f"Manifest field '{name}' is invalid")

    return value


def validate_staged_bundle(path: Path) -> dict[str, Any]:
    """Проверяет распакованный bundle и возвращает manifest."""
    manifest_path = path / "rag_manifest.json"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError(f"Invalid manifest: {error}") from error

    if not isinstance(manifest, dict):
        raise BundleValidationError("Manifest must be a JSON object")

    if manifest.get("format_version") != 1:
        raise BundleValidationError("Unsupported manifest format version")

    if manifest.get("index_file") != "rag_index.faiss":
        raise BundleValidationError("Manifest index_file is invalid")

    if manifest.get("metadata_file") != "rag_metadata.jsonl":
        raise BundleValidationError("Manifest metadata_file is invalid")

    if not isinstance(manifest.get("embedding_model"), str):
        raise BundleValidationError("Manifest embedding_model is invalid")

    if not isinstance(manifest.get("built_at"), str) or not manifest["built_at"]:
        raise BundleValidationError("Manifest built_at is invalid")

    document_count = _require_int(manifest, "document_count", 0)
    dimensions = _require_int(manifest, "dimensions", 1)
    checksums = manifest.get("checksums")

    if not isinstance(checksums, dict) or set(checksums) != {
        "rag_index.faiss",
        "rag_metadata.jsonl",
    }:
        raise BundleValidationError("Manifest checksums are invalid")

    for file_name in checksums:
        checksum = checksums[file_name]

        if not isinstance(checksum, str) or len(checksum) != 64:
            raise BundleValidationError(f"Invalid checksum for {file_name}")

        if file_sha256(path / file_name) != checksum:
            raise BundleValidationError(f"Checksum mismatch for {file_name}")

    metadata = load_metadata(path / "rag_metadata.jsonl")

    try:
        index = faiss.read_index(str(path / "rag_index.faiss"))
    except Exception as error:
        raise BundleValidationError(f"Invalid FAISS index: {error}") from error

    if index.ntotal != len(metadata) or index.ntotal != document_count:
        raise BundleValidationError("FAISS document count does not match metadata")

    if index.d != dimensions:
        raise BundleValidationError("FAISS dimensions do not match manifest")

    return manifest


def stage_bundle(archive_path: Path, staging_dir: Path, max_size: int) -> tuple[Path, dict[str, Any]]:
    """Проверяет ZIP и сохраняет валидный bundle в staging directory."""
    if max_size <= 0:
        raise BundleValidationError("Bundle size limit must be positive")

    if archive_path.stat().st_size > max_size:
        raise BundleValidationError("Bundle exceeds the configured size limit")

    staging_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = Path(tempfile.mkdtemp(prefix="upload-", dir=staging_dir))

    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]

            if len(names) != len(set(names)) or set(names) != BUNDLE_FILES:
                raise BundleValidationError("Bundle must contain exactly the required files")

            if any(member.is_dir() for member in members):
                raise BundleValidationError("Bundle must not contain directories")

            if sum(member.file_size for member in members) > max_size:
                raise BundleValidationError("Extracted bundle exceeds the configured size limit")

            for member in members:
                with archive.open(member) as source, (bundle_dir / member.filename).open(
                    "wb"
                ) as target:
                    shutil.copyfileobj(source, target)

        return bundle_dir, validate_staged_bundle(bundle_dir)

    except (OSError, zipfile.BadZipFile, BundleValidationError):
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise


