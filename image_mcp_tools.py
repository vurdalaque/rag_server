"""Conditional MCP tools for image generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

from comfy_progress import reset_wait_progress, set_wait_progress
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field
from mcp.types import CallToolResult, TextContent

from image_generation import GenerateImageRequest
from image_mcp_backends import ImageMcpBackends
from image_mcp_capabilities import build_platform_capabilities_payload
from image_mcp_ops import register_image_ops_tools
from image_timings import attach_total, elapsed_ms
from image_generation_errors import (
    ImageGenerationError,
    InternalImageGenerationError,
    InvalidMaskImageError,
    InvalidReferenceImageError,
    InvalidRequestError,
    InvalidSketchImageError,
    public_error_code,
)
from image_reference import (
    decode_and_validate_mask_image,
    decode_and_validate_reference_image,
    decode_and_validate_sketch_image,
)
from image_mcp_schemas import (
    GENERATE_IMAGE_TOOL_DESCRIPTION,
    GenerateImageInput,
    GenerateImageJobOutput,
    GenerationJobOutput,
    GeneratedArtifactReference,
    GenerateImageStructuredOutput,
    ImageGenerationCapabilitiesOutput,
    _MASK_FIELD_DESCRIPTION,
    _REFERENCE_IMAGES_FIELD_DESCRIPTION,
    _SKETCH_FIELD_DESCRIPTION,
    capabilities_from_backend,
    generate_image_input_json_schema,
)
from rag_metrics import track_mcp_tool

logger = logging.getLogger(__name__)

CORE_MCP_TOOL_NAMES: tuple[str, ...] = (
    "search_project",
    "web_search",
    "ask_project",
    "ping",
)

IMAGE_MCP_TOOL_NAMES: tuple[str, ...] = (
    "generate_image",
    "get_generation_job",
    "confirm_generation_delivery",
    "image_generation_capabilities",
    "analyze_image",
    "segment_image",
    "upscale_image",
)

GENERATION_MCP_TOOL_NAMES: tuple[str, ...] = (
    "generate_image",
    "get_generation_job",
    "confirm_generation_delivery",
    "image_generation_capabilities",
)

IMAGE_OPS_TOOL_NAMES: tuple[str, ...] = (
    "analyze_image",
    "segment_image",
    "upscale_image",
)

_backends: ImageMcpBackends | None = None
_registered_tool_names: list[str] = []
_registered = False
_ARTIFACT_ID_PATTERN = re.compile(r"(?P<job>[A-Za-z0-9-]{1,128})_(?P<index>\d{1,3})\Z")
_ARTIFACT_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
_generation_tasks: dict[str, asyncio.Task] = {}


@contextmanager
def _jobs_db():
    root = _artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "generation-jobs.sqlite3", timeout=30)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS generation_jobs (request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL, prompt_ids TEXT NOT NULL DEFAULT '[]', result TEXT, error TEXT, updated REAL NOT NULL, expected INTEGER NOT NULL DEFAULT 1)")
        db.execute("CREATE TABLE IF NOT EXISTS delivered_artifacts (artifact_id TEXT PRIMARY KEY, delivered REAL NOT NULL)")
        if not any(row[1] == "expected" for row in db.execute("PRAGMA table_info(generation_jobs)")):
            db.execute("ALTER TABLE generation_jobs ADD COLUMN expected INTEGER NOT NULL DEFAULT 1")
        with db:
            yield db
    finally:
        db.close()


def _job_read(request_id: str) -> dict | None:
    with _jobs_db() as db:
        row = db.execute("SELECT fingerprint, status, prompt_ids, result, error, updated, expected FROM generation_jobs WHERE request_id=?", (request_id,)).fetchone()
    if row is None:
        return None
    return {"request_id": request_id, "fingerprint": row[0], "status": row[1], "prompt_ids": json.loads(row[2]), "result": json.loads(row[3]) if row[3] else None, "error": row[4], "updated": row[5], "expected": row[6]}


def _job_update(request_id: str, status: str, *, prompt_ids=None, result=None, error=None) -> None:
    with _jobs_db() as db:
        db.execute("UPDATE generation_jobs SET status=?, prompt_ids=COALESCE(?,prompt_ids), result=COALESCE(?,result), error=?, updated=? WHERE request_id=?", (status, json.dumps(prompt_ids) if prompt_ids is not None else None, json.dumps(result) if result is not None else None, error, time.time(), request_id))


def _job_register(request_id: str, fingerprint: str, expected: int) -> tuple[dict, bool]:
    with _jobs_db() as db:
        cursor = db.execute("INSERT OR IGNORE INTO generation_jobs(request_id,fingerprint,status,expected,updated) VALUES(?,?,?,?,?)", (request_id, fingerprint, "running", expected, time.time()))
        created = cursor.rowcount == 1
    return _job_read(request_id), created


def artifact_delivery_confirmed(artifact_id: str) -> bool:
    if parse_image_artifact_id(artifact_id) is None:
        return False
    with _jobs_db() as db:
        return db.execute("SELECT 1 FROM delivered_artifacts WHERE artifact_id=?", (artifact_id,)).fetchone() is not None


def confirm_generated_image_delivery(request_id: str, artifact_ids: list[str]) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
        raise InvalidRequestError("invalid request_id")
    if not isinstance(artifact_ids, list) or not artifact_ids or any(not isinstance(item, str) for item in artifact_ids):
        raise InvalidRequestError("artifact_ids must contain all generated artifact IDs")
    with _jobs_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT status, result FROM generation_jobs WHERE request_id=?", (request_id,)).fetchone()
        if not row or row[0] not in ("completed", "delivered") or not row[1]:
            raise InvalidRequestError("generation job is not completed")
        expected = [item["artifact_id"] for item in json.loads(row[1]).get("artifacts", [])]
        if len(expected) != len(artifact_ids) or set(expected) != set(artifact_ids):
            raise InvalidRequestError("artifact_ids do not match generated job artifacts")
        db.executemany("INSERT OR IGNORE INTO delivered_artifacts(artifact_id,delivered) VALUES(?,?)", [(item, time.time()) for item in expected])
        db.execute("UPDATE generation_jobs SET status='delivered', updated=? WHERE request_id=?", (time.time(), request_id))
    for artifact_id in expected:
        for extension in _ARTIFACT_MIME_EXTENSIONS.values():
            path = _artifact_root() / f"{artifact_id}{extension}"
            try:
                if not path.is_symlink():
                    path.unlink(missing_ok=True)
            except OSError:
                logger.warning("MCP ARTIFACT DELIVERY CLEANUP FAILED artifact_id=%s", artifact_id, exc_info=True)
    logger.info("MCP ARTIFACT DELIVERY CONFIRMED request_id=%s count=%s", request_id, len(expected))
    return _job_read(request_id)


def _artifact_root() -> Path:
    return Path(os.getenv("MCP_ARTIFACTS_DIR", "data/mcp-artifacts")).resolve()


def _artifact_ttl_seconds() -> float:
    try:
        return max(60.0, float(os.getenv("MCP_ARTIFACT_TTL_SECONDS", "604800")))
    except ValueError:
        return 604800.0


def cleanup_expired_image_artifacts(now: float | None = None) -> int:
    root = _artifact_root()
    if not root.is_dir():
        return 0
    current = time.time() if now is None else now
    cutoff = current - _artifact_ttl_seconds()
    removed = 0
    for path in root.iterdir():
        if parse_image_artifact_id(path.stem) is None or path.suffix not in _ARTIFACT_MIME_EXTENSIONS.values():
            continue
        try:
            if path.is_file() and not path.is_symlink() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            logger.warning("MCP ARTIFACT CLEANUP FAILED filename=%s", path.name)
    return removed


def _safe_artifact_job_id(prompt_id: str | None) -> str:
    value = str(prompt_id or uuid.uuid4().hex)
    return value if re.fullmatch(r"[A-Za-z0-9-]{1,128}", value) else uuid.uuid4().hex


def store_generated_image_artifact(
    prompt_id: str | None,
    image_index: int,
    image,
) -> GeneratedArtifactReference:
    if image_index < 0 or image_index > 999:
        raise InvalidRequestError("generated artifact index is out of range")
    job_id = _safe_artifact_job_id(prompt_id)
    mime_type = image.mime_type.lower()
    extension = _ARTIFACT_MIME_EXTENSIONS.get(mime_type)
    if extension is None:
        raise InvalidRequestError("unsupported generated artifact MIME type", mime_type=mime_type)
    root = _artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    artifact_id = f"{job_id}_{image_index}"
    target = root / f"{artifact_id}{extension}"
    temporary = root / f".{artifact_id}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(image.data)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    filename = f"image_{image_index + 1}{extension}"
    logger.info(
        "MCP ARTIFACT STORED prompt_id=%s artifact_id=%s bytes=%s mime_type=%s",
        prompt_id or job_id,
        artifact_id,
        len(image.data),
        mime_type,
    )
    return GeneratedArtifactReference(
        artifact_id=artifact_id,
        mime_type=mime_type,
        size_bytes=len(image.data),
        filename=filename,
    )


def store_generated_image_artifacts(result) -> list[GeneratedArtifactReference]:
    """Persist image bytes before returning compact metadata to the caller."""
    cleanup_expired_image_artifacts()
    indices: dict[str, int] = {}
    references: list[GeneratedArtifactReference] = []
    for image in result.images:
        job_id = _safe_artifact_job_id(image.prompt_id or result.prompt_id)
        index = indices.get(job_id, 0)
        references.append(store_generated_image_artifact(job_id, index, image))
        indices[job_id] = index + 1
    return references


def parse_image_artifact_id(artifact_id: str) -> tuple[str, int] | None:
    match = _ARTIFACT_ID_PATTERN.fullmatch(artifact_id)
    if match is None:
        return None
    return match.group("job"), int(match.group("index"))


def resolve_generated_image_artifact(artifact_id: str) -> tuple[Path, str, str] | None:
    parsed = parse_image_artifact_id(artifact_id)
    if parsed is None:
        return None
    job_id, index = parsed
    root = _artifact_root()
    cutoff = time.time() - _artifact_ttl_seconds()
    for mime_type, extension in _ARTIFACT_MIME_EXTENSIONS.items():
        path = root / f"{job_id}_{index}{extension}"
        try:
            if path.is_symlink() or not path.is_file():
                continue
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                return None
        except OSError:
            continue
        return path, mime_type, f"image_{index + 1}{extension}"
    return None


def image_tools_registered() -> bool:
    return _registered


def get_image_backend():
    if _backends is None:
        return None
    return _backends.generation


def get_image_mcp_backends() -> ImageMcpBackends | None:
    return _backends


def list_mcp_tool_names(
    *,
    include_image_tools: bool | None = None,
) -> list[str]:
    if include_image_tools is None:
        include_image_tools = _registered

    names = list(CORE_MCP_TOOL_NAMES)

    if include_image_tools:
        if _registered_tool_names:
            names.extend(_registered_tool_names)
        elif _registered:
            names.extend(IMAGE_MCP_TOOL_NAMES)

    return names


def _coerce_image_base64_payload(value: Any, *, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        data = value.get("data")
        if isinstance(data, str) and data.strip():
            return data
    raise InvalidRequestError(
        f"{field} must be a base64-encoded image",
        field=field,
    )


def _error_tool_result(error: ImageGenerationError) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": {
                            "code": public_error_code(error),
                            "message": error.message,
                            "details": error.details or None,
                        }
                    },
                    ensure_ascii=False,
                ),
            )
        ],
        is_error=True,
    )


def _job_tool_result(job: dict) -> CallToolResult:
    public = {key: job[key] for key in ("request_id", "status", "prompt_ids", "result", "error")}
    if job["status"] == "delivered":
        public["result"] = None
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(public))], structured_content=public)


async def _refresh_generation_job(request_id: str) -> dict:
    job = _job_read(request_id)
    if job is None or job["status"] != "running":
        return job
    if request_id in _generation_tasks:
        return job
    prompts = job["prompt_ids"]
    if not prompts and time.time() - job["updated"] < float(os.getenv("IMAGE_GENERATION_TIMEOUT", "3600")) + 60:
        return job
    if not prompts:
        _job_update(request_id, "unknown", error="Submission result is unknown; generation was not retried")
        return _job_read(request_id)
    expected = job["expected"]
    if len(prompts) < expected:
        if time.time() - job["updated"] < float(os.getenv("IMAGE_GENERATION_TIMEOUT", "3600")) + 60:
            return job
        _job_update(request_id, "unknown", error="Not all ComfyUI prompt submissions are known; generation was not retried")
        return _job_read(request_id)
    backend = _backends.generation if _backends is not None else None
    if backend is None or not callable(getattr(backend, "recover_generated_image", None)):
        _job_update(request_id, "unknown", error="Backend cannot reconcile submitted prompts")
        return _job_read(request_id)
    try:
        recovered = []
        for prompt_id in prompts:
            image, state = await backend.recover_generated_image(prompt_id, 0)
            if state == "pending" or (image is None and state not in ("error", "output_missing")):
                return job
            if image is None:
                _job_update(request_id, "failed", error=f"ComfyUI output missing: {prompt_id}")
                return _job_read(request_id)
            recovered.append(store_generated_image_artifact(prompt_id, 0, image))
        result = {"artifacts": [item.model_dump(mode="json") for item in recovered], "prompt_id": prompts[-1], "image_count": len(recovered)}
        _job_update(request_id, "completed", result=result)
    except Exception as exc:
        logger.warning("GENERATION RECOVERY request_id=%s error=%s", request_id, exc)
        return job
    return _job_read(request_id)


def list_ping_tool_names() -> list[str]:
    """Tool names advertised by ping (excludes ping itself)."""
    return [name for name in list_mcp_tool_names() if name != "ping"]


def _remove_image_tools_from_server(mcp: MCPServer) -> None:
    for name in IMAGE_MCP_TOOL_NAMES:
        try:
            mcp.remove_tool(name)
        except ToolError:
            pass


def register_image_mcp_backends(
    mcp: MCPServer,
    backends: ImageMcpBackends,
) -> None:
    """Register all available image MCP tools from ``backends``."""
    register_image_tools(mcp, backends.generation, backends=backends)


def register_image_tools(
    mcp: MCPServer,
    backend,
    *,
    backends: ImageMcpBackends | None = None,
) -> None:
    global _backends, _registered, _registered_tool_names

    if backends is None:
        backends = ImageMcpBackends(generation=backend)
    elif backend is not None and backends.generation is None:
        backends = ImageMcpBackends(
            generation=backend,
            analyzer=backends.analyzer,
            segmenter=backends.segmenter,
            upscaler=backends.upscaler,
        )

    _backends = backends
    _registered_tool_names = []
    _remove_image_tools_from_server(mcp)

    async def run_generation_job(request_id: str, request: GenerateImageRequest) -> None:
        def submitted(prompt_id: str) -> None:
            job = _job_read(request_id)
            _job_update(request_id, "running", prompt_ids=job["prompt_ids"] + [prompt_id])

        try:
            backend = _backends.generation
            if callable(getattr(backend, "recover_generated_image", None)):
                result = await backend.generate(request, on_submitted=submitted)
            else:
                result = await backend.generate(request)
            artifacts = await asyncio.to_thread(store_generated_image_artifacts, result)
            output = GenerateImageStructuredOutput.from_generation(
                seed=result.seed, prompt_id=result.prompt_id, image_count=len(result.images),
                artifacts=artifacts, timings=result.timings,
            ).model_dump(mode="json")
            _job_update(request_id, "completed", result=output)
        except asyncio.CancelledError:
            logger.warning("GENERATION INTERRUPTED request_id=%s", request_id)
            raise
        except Exception as exc:
            job = _job_read(request_id)
            # После сбоя отправки состояние запуска в ComfyUI неизвестно.
            status = "unknown" if not job["prompt_ids"] else "running"
            if isinstance(exc, ImageGenerationError):
                if not job["prompt_ids"] and exc.code != "comfyui_unavailable":
                    status = "failed"
                elif job["prompt_ids"] and exc.code in ("execution_failed", "generation_failed", "output_too_large"):
                    status = "failed"
            _job_update(request_id, status, error=f"{type(exc).__name__}: {exc}")
            logger.exception("GENERATION JOB ERROR request_id=%s status=%s", request_id, status)
        finally:
            _generation_tasks.pop(request_id, None)

    @track_mcp_tool("generate_image")
    async def generate_image(
        prompt: Annotated[str, Field(description=GenerateImageInput.model_fields["prompt"].description)],
        negative_prompt: Annotated[
            str | None,
            Field(description=GenerateImageInput.model_fields["negative_prompt"].description),
        ] = None,
        width: Annotated[
            int | None,
            Field(description=GenerateImageInput.model_fields["width"].description),
        ] = None,
        height: Annotated[
            int | None,
            Field(description=GenerateImageInput.model_fields["height"].description),
        ] = None,
        steps: Annotated[
            int | None,
            Field(description=GenerateImageInput.model_fields["steps"].description),
        ] = None,
        seed: Annotated[
            int | None,
            Field(description=GenerateImageInput.model_fields["seed"].description),
        ] = None,
        cfg: Annotated[
            float | None,
            Field(description=GenerateImageInput.model_fields["cfg"].description),
        ] = None,
        sampler: Annotated[
            str | None,
            Field(description=GenerateImageInput.model_fields["sampler"].description),
        ] = None,
        scheduler: Annotated[
            str | None,
            Field(description=GenerateImageInput.model_fields["scheduler"].description),
        ] = None,
        image_count: Annotated[
            int,
            Field(description=GenerateImageInput.model_fields["image_count"].description),
        ] = 1,
        reference_images: Annotated[
            list[str] | None,
            Field(description=_REFERENCE_IMAGES_FIELD_DESCRIPTION),
        ] = None,
        mask: Annotated[
            str | None,
            Field(description=_MASK_FIELD_DESCRIPTION),
        ] = None,
        sketch: Annotated[
            str | None,
            Field(description=_SKETCH_FIELD_DESCRIPTION),
        ] = None,
        request_id: Annotated[
            str | None,
            Field(description="Stable idempotency key for a generation job."),
        ] = None,
        *,
        ctx: Context,
    ) -> Annotated[CallToolResult, GenerateImageJobOutput]:
        """Registered via ``add_tool``; see ``GENERATE_IMAGE_TOOL_DESCRIPTION``."""
        tool_started = time.monotonic()
        if _backends is None or _backends.generation is None:
            logger.warning("generate_image rejected: image backend not registered")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": {
                                    "code": "backend_unavailable",
                                    "message": (
                                        "image generation is not available "
                                        "on this server"
                                    ),
                                }
                            },
                            ensure_ascii=False,
                        ),
                    )
                ],
                is_error=True,
            )

        ref_count = len(reference_images or [])
        logger.info(
            "MCP generate_image start prompt_chars=%s reference_images=%s "
            "has_mask=%s has_sketch=%s image_count=%s",
            len(prompt),
            ref_count,
            mask is not None,
            sketch is not None,
            image_count,
        )

        decoded_refs: list[bytes] = []
        decoded_mask: bytes | None = None
        decoded_sketch: bytes | None = None

        try:
            if reference_images:
                for index, encoded in enumerate(reference_images):
                    payload = _coerce_image_base64_payload(
                        encoded,
                        field="reference_images",
                    )
                    decoded_refs.append(
                        decode_and_validate_reference_image(payload, index=index),
                    )

            if mask is not None:
                decoded_mask = decode_and_validate_mask_image(
                    _coerce_image_base64_payload(mask, field="mask"),
                )

            if sketch is not None:
                decoded_sketch = decode_and_validate_sketch_image(
                    _coerce_image_base64_payload(sketch, field="sketch"),
                )
        except (
            InvalidReferenceImageError,
            InvalidMaskImageError,
            InvalidSketchImageError,
            InvalidRequestError,
        ) as error:
            return _error_tool_result(error)

        request = GenerateImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            image_count=image_count,
            reference_images=tuple(decoded_refs),
            mask=decoded_mask,
            sketch=decoded_sketch,
        )

        if request_id is not None:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
                return _error_tool_result(InvalidRequestError("invalid request_id"))
            signature = json.dumps({
                "prompt": prompt, "negative_prompt": negative_prompt,
                "width": width, "height": height, "steps": steps, "seed": seed,
                "cfg": cfg, "sampler": sampler, "scheduler": scheduler,
                "image_count": image_count,
                "references": [hashlib.sha256(x).hexdigest() for x in decoded_refs],
                "mask": hashlib.sha256(decoded_mask).hexdigest() if decoded_mask else None,
                "sketch": hashlib.sha256(decoded_sketch).hexdigest() if decoded_sketch else None,
            }, sort_keys=True)
            fingerprint = hashlib.sha256(signature.encode()).hexdigest()
            job, created = _job_register(request_id, fingerprint, image_count)
            if job["fingerprint"] != fingerprint:
                return _error_tool_result(InvalidRequestError("request_id already used with different parameters"))
            if created:
                task = asyncio.create_task(run_generation_job(request_id, request), name=f"generation-{request_id}")
                _generation_tasks[request_id] = task
            else:
                job = await _refresh_generation_job(request_id)
            return _job_tool_result(job)

        async def _comfy_wait_progress(elapsed: float, state: str) -> None:
            # Доходит до клиента только если тот прислал params._meta.progressToken
            # и включён SSE-режим: в JSON-режиме SDK отбрасывает такие сообщения.
            try:
                await ctx.report_progress(
                    min(99.0, elapsed),
                    600.0,
                    f"comfyui:{state}",
                )
            except Exception:
                logger.debug("MCP progress notification failed", exc_info=True)

        json_response = os.getenv("MCP_JSON_RESPONSE", "true").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        # JSON responses cannot carry request-scoped progress notifications.
        # Emitting one can terminate a stateful session while the tool is running.
        progress_token = set_wait_progress(None if json_response else _comfy_wait_progress)
        try:
            try:
                result = await _backends.generation.generate(request)
            except asyncio.CancelledError:
                logger.warning(
                    "MCP generate_image cancelled (client likely disconnected)",
                )
                raise
            except ImageGenerationError as error:
                logger.warning(
                    "MCP generate_image failed code=%s message=%s details=%s",
                    public_error_code(error),
                    error.message,
                    error.details or {},
                )
                return _error_tool_result(error)
            except Exception as error:
                logger.exception("MCP generate_image unexpected error")
                return _error_tool_result(
                    InternalImageGenerationError(
                        "image generation failed unexpectedly",
                        reason=str(error),
                    ),
                )
        finally:
            reset_wait_progress(progress_token)

        try:
            artifact_refs = await asyncio.to_thread(store_generated_image_artifacts, result)
        except Exception as error:
            logger.exception(
                "MCP artifact persistence failed prompt_id=%s",
                result.prompt_id,
            )
            return _error_tool_result(
                InternalImageGenerationError(
                    "generated image could not be persisted for download",
                    prompt_id=result.prompt_id,
                    reason=str(error),
                ),
            )

        structured = GenerateImageStructuredOutput.from_generation(
            seed=result.seed,
            prompt_id=result.prompt_id,
            image_count=len(result.images),
            artifacts=artifact_refs,
            timings=attach_total(result.timings, elapsed_ms(tool_started)),
        )

        tool_result = CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="Image generation finished. Download the returned artifact IDs.",
                )
            ],
            structured_content=structured.model_dump(mode="json"),
        )
        logger.info(
            "MCP RETURN generate_image prompt_id=%s artifacts=%s seed=%s",
            result.prompt_id,
            len(artifact_refs),
            structured.seed,
        )
        logger.info(
            "MCP TOOL FINISHED generate_image prompt_id=%s",
            result.prompt_id,
        )
        return tool_result

    @track_mcp_tool("image_generation_capabilities")
    async def image_generation_capabilities() -> (
        Annotated[CallToolResult, ImageGenerationCapabilitiesOutput]
    ):
        """
        Report server image-generation limits, defaults, and sampler metadata.
        """
        payload = await build_platform_capabilities_payload(_backends or ImageMcpBackends())
        structured = capabilities_from_backend(payload)
        return CallToolResult(
            content=[],
            structured_content=structured.model_dump(mode="json"),
        )

    @track_mcp_tool("get_generation_job")
    async def get_generation_job(request_id: str) -> Annotated[CallToolResult, GenerationJobOutput]:
        """Get a durable image generation job without starting another generation."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
            return _error_tool_result(InvalidRequestError("invalid request_id"))
        job = await _refresh_generation_job(request_id)
        if job is None:
            return _error_tool_result(InvalidRequestError("generation job not found"))
        return _job_tool_result(job)

    @track_mcp_tool("confirm_generation_delivery")
    async def confirm_generation_delivery(request_id: str, artifact_ids: list[str]) -> Annotated[CallToolResult, GenerationJobOutput]:
        """Acknowledge that all images were uploaded to the final destination."""
        try:
            job = await asyncio.to_thread(confirm_generated_image_delivery, request_id, artifact_ids)
        except InvalidRequestError as error:
            return _error_tool_result(error)
        return _job_tool_result(job)

    if backends.generation is not None:
        mcp.add_tool(
            generate_image,
            description=GENERATE_IMAGE_TOOL_DESCRIPTION,
        )
        _registered_tool_names.append("generate_image")
        mcp.add_tool(get_generation_job)
        _registered_tool_names.append("get_generation_job")
        mcp.add_tool(confirm_generation_delivery)
        _registered_tool_names.append("confirm_generation_delivery")

    register_image_ops_tools(
        mcp,
        backends,
        error_result=_error_tool_result,
        coerce_image=_coerce_image_base64_payload,
        registered_names=_registered_tool_names,
    )

    mcp.add_tool(image_generation_capabilities)
    _registered_tool_names.append("image_generation_capabilities")

    _registered = bool(_registered_tool_names)
    logger.info(
        "Registered MCP image tools: %s",
        ", ".join(_registered_tool_names),
    )


def reset_image_mcp_registration(mcp: MCPServer | None = None) -> None:
    """Test helper: clear module registration state and remove tools from ``mcp``."""
    global _backends, _registered, _registered_tool_names
    if mcp is not None:
        _remove_image_tools_from_server(mcp)
    _backends = None
    _registered_tool_names = []
    _registered = False


def published_generate_image_input_schema() -> dict[str, Any]:
    """Input schema exposed on ``tools/list`` for ``generate_image``."""
    return generate_image_input_json_schema()
