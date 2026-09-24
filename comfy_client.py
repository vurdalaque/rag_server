"""Async ComfyUI HTTP + WebSocket client."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import websockets

from comfy_workflow import resolve_comfy_output_path
from image_generation_config import ImageGenerationConfig, load_image_generation_config
from image_generation_errors import (
    CancelledError,
    ComfyUIUnavailableError,
    ComfyUIWorkflowRejectedError,
    ExecutionFailedError,
    ImageGenerationTimeoutError,
    InternalImageGenerationError,
    OutputMissingError,
)

logger = logging.getLogger(__name__)

# WebSocket is a fast path only; history polling is the source of terminal truth.
_HISTORY_POLL_INTERVAL_SECONDS = 1.0

REQUIRED_NODE_CLASSES = frozenset(
    {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "TextEncodeQwenImage21",
        "KSampler",
        "VAEDecode",
        "SaveImage",
        "LoadImage",
    }
)


@dataclass(frozen=True)
class ComfyUploadedImage:
    """ComfyUI input folder name returned from ``/upload/image``."""

    name: str
    subfolder: str = ""
    type: str = "input"


@dataclass(frozen=True)
class ComfyOutputImage:
    filename: str
    subfolder: str = ""
    type: str = "output"


@dataclass
class ComfyPromptOutputs:
    prompt_id: str
    client_id: str
    seed_used: int
    images: list[ComfyOutputImage] = field(default_factory=list)


PromptHistoryStatus = Literal["pending", "running", "success", "error"]


@dataclass(frozen=True)
class PromptHistoryState:
    status: PromptHistoryStatus
    output_count: int = 0
    error_detail: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("success", "error")


def _decode_ws_message(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        if len(raw) > 8:
            raw = raw[8:]
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw
    data = json.loads(text)
    if not isinstance(data, dict):
        raise InternalImageGenerationError("Unexpected WebSocket payload type")
    return data


class ComfyUIClient:
    """Upload images, submit workflows, and wait for completion (history + WebSocket)."""

    def __init__(
        self,
        config: ImageGenerationConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._config = config or load_image_generation_config()
        self._base_url = (base_url or self._config.comfy_base_url).rstrip("/")
        self._timeout_override = timeout
        self._http = http_client
        self._owned_http = http_client is None

    async def aclose(self) -> None:
        if self._owned_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    def _http_timeout(self) -> float:
        if self._timeout_override is not None:
            return self._timeout_override
        return self._config.timeout_seconds

    def _ws_timeout(self) -> float:
        if self._timeout_override is not None:
            return self._timeout_override
        return self._config.ws_timeout_seconds

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._http_timeout()),
            )
        return self._http

    async def fetch_object_info(self, timeout: float | None = None) -> dict[str, Any]:
        client = await self._client()
        try:
            response = await client.get(
                "/object_info",
                timeout=timeout or self._config.probe_timeout_seconds,
            )
            response.raise_for_status()
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            raise ComfyUIUnavailableError(
                "ComfyUI object_info probe failed",
                reason=str(exc),
            ) from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise ComfyUIUnavailableError("Invalid object_info response")
        return payload

    async def probe(self) -> dict[str, Any]:
        """Cheap capability check: reachability and required node classes."""
        client = await self._client()
        try:
            response = await client.get(
                "/object_info",
                timeout=self._config.probe_timeout_seconds,
            )
            response.raise_for_status()
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            raise ComfyUIUnavailableError(
                "ComfyUI object_info probe failed",
                reason=str(exc),
            ) from exc

        payload = response.json()
        if not isinstance(payload, dict):
            raise ComfyUIUnavailableError("Invalid object_info response")

        missing = sorted(REQUIRED_NODE_CLASSES - set(payload.keys()))
        if missing:
            raise ComfyUIUnavailableError(
                "ComfyUI is missing required node classes",
                missing_nodes=missing,
            )
        return payload

    async def upload_image(
        self,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> ComfyUploadedImage:
        client = await self._client()
        files = {"image": (filename, content, content_type)}
        data = {"overwrite": "true"}
        try:
            response = await client.post(
                "/upload/image",
                files=files,
                data=data,
                timeout=self._config.upload_timeout_seconds,
            )
            response.raise_for_status()
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            raise ComfyUIUnavailableError(
                "Image upload to ComfyUI failed",
                reason=str(exc),
            ) from exc

        body = response.json()
        name = body.get("name")
        if not name:
            raise InternalImageGenerationError("ComfyUI upload returned no name")
        uploaded = ComfyUploadedImage(
            name=str(name),
            subfolder=str(body.get("subfolder") or ""),
            type=str(body.get("type") or "input"),
        )
        return uploaded

    async def fetch_view_bytes(self, image: ComfyUploadedImage) -> bytes:
        client = await self._client()
        params = {
            "filename": image.name,
            "subfolder": image.subfolder,
            "type": image.type,
        }
        try:
            response = await client.get(
                "/view",
                params=params,
                timeout=self._config.upload_timeout_seconds,
            )
            response.raise_for_status()
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            raise ComfyUIUnavailableError(
                "ComfyUI view fetch failed",
                reason=str(exc),
                filename=image.name,
            ) from exc
        return response.content

    async def verify_uploaded_image_bytes(
        self,
        uploaded: ComfyUploadedImage,
        expected: bytes,
        *,
        sha256_before: str | None = None,
    ) -> None:
        downloaded = await self.fetch_view_bytes(uploaded)
        digest_after = hashlib.sha256(downloaded).hexdigest()
        digest_before = sha256_before or hashlib.sha256(expected).hexdigest()
        if digest_before != digest_after:
            logger.error(
                "ComfyUI upload bytes mismatch name=%s before_sha256=%s after_sha256=%s "
                "before_len=%s after_len=%s",
                uploaded.name,
                digest_before,
                digest_after,
                len(expected),
                len(downloaded),
            )
            raise InternalImageGenerationError(
                "Uploaded reference image bytes do not match ComfyUI stored file",
                filename=uploaded.name,
                size_before=len(expected),
                size_after=len(downloaded),
                sha256_before=digest_before,
                sha256_after=digest_after,
            )
        if downloaded != expected:
            raise InternalImageGenerationError(
                "Uploaded reference image content mismatch after hash collision check",
                filename=uploaded.name,
            )

    async def submit_prompt(
        self,
        workflow: dict[str, Any],
        client_id: str,
    ) -> str:
        client = await self._client()
        body = {"prompt": workflow, "client_id": client_id}
        try:
            response = await client.post(
                "/prompt",
                json=body,
                timeout=self._config.timeout_seconds,
            )
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            raise ComfyUIUnavailableError(
                "ComfyUI prompt submission failed",
                reason=str(exc),
            ) from exc

        if response.status_code >= 400:
            raise ComfyUIWorkflowRejectedError(
                "ComfyUI rejected workflow",
                status_code=response.status_code,
                body=response.text[:500],
            )

        payload = response.json()
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise InternalImageGenerationError("ComfyUI prompt response missing prompt_id")
        logger.info(
            "COMFY SUBMIT prompt_id=%s client_id=%s",
            prompt_id,
            client_id,
        )
        logger.info("COMFY QUEUED prompt_id=%s", prompt_id)
        return str(prompt_id)

    async def fetch_history(self, prompt_id: str) -> dict[str, Any]:
        client = await self._client()
        try:
            response = await client.get(
                f"/history/{prompt_id}",
                timeout=self._config.output_read_timeout_seconds,
            )
            response.raise_for_status()
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            raise ComfyUIUnavailableError(
                "ComfyUI history fetch failed",
                reason=str(exc),
            ) from exc

        payload = response.json()
        if not isinstance(payload, dict):
            raise InternalImageGenerationError("Invalid history response")
        return payload

    @staticmethod
    def extract_output_images(history: dict[str, Any], prompt_id: str) -> list[ComfyOutputImage]:
        entry = history.get(prompt_id) or {}
        outputs = entry.get("outputs") or {}
        images: list[ComfyOutputImage] = []
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for item in node_output.get("images") or []:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename")
                if not filename:
                    continue
                images.append(
                    ComfyOutputImage(
                        filename=str(filename),
                        subfolder=str(item.get("subfolder") or ""),
                        type=str(item.get("type") or "output"),
                    ),
                )
        return images

    @staticmethod
    def history_entry_for_prompt(
        history: dict[str, Any],
        prompt_id: str,
    ) -> dict[str, Any] | None:
        entry = history.get(prompt_id)
        if isinstance(entry, dict):
            return entry
        if len(history) == 1:
            only = next(iter(history.values()))
            if isinstance(only, dict) and (
                "outputs" in only or "status" in only
            ):
                return only
        return None

    @staticmethod
    def classify_history_entry(entry: dict[str, Any] | None) -> PromptHistoryState:
        if entry is None:
            return PromptHistoryState("pending", 0)

        outputs = entry.get("outputs") or {}
        output_count = 0
        if isinstance(outputs, dict):
            for node_output in outputs.values():
                if not isinstance(node_output, dict):
                    continue
                images = node_output.get("images") or []
                if isinstance(images, list):
                    output_count += sum(
                        1 for item in images if isinstance(item, dict) and item.get("filename")
                    )

        error_detail = ComfyUIClient._history_execution_error(entry)
        if error_detail:
            return PromptHistoryState("error", output_count, error_detail)

        status = entry.get("status") or {}
        status_str = ""
        if isinstance(status, dict):
            status_str = str(status.get("status_str") or "").lower()

        if status_str in {"error", "failed"}:
            return PromptHistoryState(
                "error",
                output_count,
                error_detail or status_str,
            )

        if output_count > 0:
            return PromptHistoryState("success", output_count)

        if ComfyUIClient._history_terminal_success(entry):
            return PromptHistoryState("success", output_count)

        outputs = entry.get("outputs")
        if isinstance(outputs, dict) and outputs:
            # ComfyUI only commits history at task_done; non-empty outputs without
            # images usually mean a finished graph where SaveImage metadata is delayed
            # or missing — do not treat as infinite "running".
            if isinstance(status, dict) and status_str == "running":
                return PromptHistoryState("running", output_count)
            return PromptHistoryState("pending", output_count)

        return PromptHistoryState("pending", output_count)

    @staticmethod
    def _history_terminal_success(entry: dict[str, Any]) -> bool:
        status = entry.get("status")
        if not isinstance(status, dict):
            return False
        status_str = str(status.get("status_str") or "").lower()
        if status_str == "success" or status.get("completed") is True:
            return True
        messages = status.get("messages") or []
        if not isinstance(messages, list):
            return False
        for message in messages:
            if not isinstance(message, (list, tuple)) or not message:
                continue
            if message[0] in ("execution_success", "execution_cached"):
                return True
        return False

    @staticmethod
    def _history_execution_error(entry: dict[str, Any]) -> str | None:
        status = entry.get("status")
        if not isinstance(status, dict):
            return None
        messages = status.get("messages") or []
        if not isinstance(messages, list):
            return None
        for message in messages:
            if not isinstance(message, (list, tuple)) or not message:
                continue
            kind = message[0]
            if kind != "execution_error":
                continue
            payload = message[1] if len(message) > 1 else {}
            if isinstance(payload, dict):
                return str(
                    payload.get("exception_message")
                    or payload.get("exception_type")
                    or payload,
                )
            return str(payload)
        return None

    async def inspect_history_state(self, prompt_id: str) -> PromptHistoryState:
        history = await self.fetch_history(prompt_id)
        entry = self.history_entry_for_prompt(history, prompt_id)
        return self.classify_history_entry(entry)

    def _raise_if_history_error(
        self,
        state: PromptHistoryState,
        prompt_id: str,
    ) -> None:
        if state.status != "error":
            return
        raise ExecutionFailedError(
            "ComfyUI execution failed",
            prompt_id=prompt_id,
            detail=state.error_detail or "history reported error",
        )

    async def wait_for_prompt_terminal(
        self,
        prompt_id: str,
        client_id: str,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        """Wait until ComfyUI reports a terminal state for ``prompt_id``.

        Uses periodic ``/history/{prompt_id}`` polling as the source of truth.
        WebSocket events are an optional fast path and may be missed (e.g. if the
        workflow finishes before the socket subscribes).
        """
        started = time.monotonic()
        deadline = started + self._ws_timeout()

        logger.info(
            "COMFY WAITER REGISTER prompt_id=%s client_id=%s",
            prompt_id,
            client_id,
        )
        ws_task = asyncio.create_task(
            self.wait_for_prompt_ws(
                prompt_id,
                client_id,
                cancel_event=cancel_event,
            ),
        )
        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    raise CancelledError("ComfyUI wait cancelled")

                state = await self.inspect_history_state(prompt_id)
                logger.info(
                    "COMFY HISTORY CHECK prompt_id=%s",
                    prompt_id,
                )
                logger.info(
                    "COMFY HISTORY STATE prompt_id=%s state=%s outputs=%s",
                    prompt_id,
                    state.status,
                    state.output_count,
                )
                if state.is_terminal:
                    self._raise_if_history_error(state, prompt_id)
                    logger.info(
                        "COMFY TERMINAL prompt_id=%s state=%s elapsed=%.3fs",
                        prompt_id,
                        state.status,
                        time.monotonic() - started,
                    )
                    return

                if ws_task.done():
                    try:
                        ws_task.result()
                    except ExecutionFailedError:
                        state = await self.inspect_history_state(prompt_id)
                        self._raise_if_history_error(state, prompt_id)
                        raise
                    except Exception:
                        logger.warning(
                            "COMFY WS ended without terminal history prompt_id=%s",
                            prompt_id,
                        )
                    else:
                        state = await self.inspect_history_state(prompt_id)
                        if state.is_terminal:
                            self._raise_if_history_error(state, prompt_id)
                            logger.info(
                                "COMFY TERMINAL prompt_id=%s state=%s "
                                "elapsed=%.3fs source=ws_reconcile",
                                prompt_id,
                                state.status,
                                time.monotonic() - started,
                            )
                            return

                if time.monotonic() >= deadline:
                    raise ImageGenerationTimeoutError(
                        "Timed out waiting for ComfyUI execution",
                        prompt_id=prompt_id,
                    )

                remaining = deadline - time.monotonic()
                sleep_for = min(_HISTORY_POLL_INTERVAL_SECONDS, remaining)
                sleep_task = asyncio.create_task(asyncio.sleep(sleep_for))
                done, pending = await asyncio.wait(
                    {ws_task, sleep_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    if task is sleep_task:
                        continue
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if exc is not None:
                        if isinstance(exc, ExecutionFailedError):
                            state = await self.inspect_history_state(prompt_id)
                            self._raise_if_history_error(state, prompt_id)
                        raise exc
        finally:
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
            except ExecutionFailedError:
                state = await self.inspect_history_state(prompt_id)
                self._raise_if_history_error(state, prompt_id)
                raise

    async def wait_for_prompt_ws(
        self,
        prompt_id: str,
        client_id: str,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        ws_url = self._config.comfy_ws_url(client_id)
        deadline = asyncio.get_running_loop().time() + self._ws_timeout()

        try:
            async with websockets.connect(ws_url) as ws:
                while True:
                    if cancel_event and cancel_event.is_set():
                        raise CancelledError("ComfyUI wait cancelled")

                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise ImageGenerationTimeoutError(
                            "Timed out waiting for ComfyUI execution",
                            prompt_id=prompt_id,
                        )

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        raise ImageGenerationTimeoutError(
                            "Timed out waiting for ComfyUI execution",
                            prompt_id=prompt_id,
                        ) from None

                    message = _decode_ws_message(raw)
                    msg_type = message.get("type")
                    data = message.get("data") or {}
                    event_prompt_id = (
                        data.get("prompt_id")
                        if isinstance(data, dict)
                        else None
                    )
                    logger.info(
                        "COMFY WS EVENT prompt_id=%s type=%s event_prompt_id=%s",
                        prompt_id,
                        msg_type,
                        event_prompt_id,
                    )
                    if not _message_matches_prompt(message, prompt_id):
                        continue

                    if msg_type == "execution_error":
                        raise ExecutionFailedError(
                            "ComfyUI execution failed",
                            prompt_id=prompt_id,
                            detail=str(data.get("exception_message") or data),
                        )

                    if msg_type == "execution_success":
                        return

                    if msg_type == "executing" and data.get("node") is None:
                        return
        except websockets.exceptions.WebSocketException as exc:
            raise ComfyUIUnavailableError(
                "ComfyUI WebSocket connection failed",
                reason=str(exc),
            ) from exc

    async def run_prompt(
        self,
        workflow: dict[str, Any],
        *,
        seed_used: int,
        client_id: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ComfyPromptOutputs:
        run_client_id = client_id or str(uuid.uuid4())
        prompt_id = await self.submit_prompt(workflow, run_client_id)
        await self.wait_for_prompt_terminal(
            prompt_id,
            run_client_id,
            cancel_event=cancel_event,
        )
        history = await self.fetch_history(prompt_id)
        images = self.extract_output_images(history, prompt_id)
        if not images:
            logger.warning(
                "ComfyUI history has no images prompt_id=%s history_keys=%s",
                prompt_id,
                sorted(history.keys())[:5],
            )
            raise OutputMissingError(
                "ComfyUI completed without output images",
                prompt_id=prompt_id,
            )
        logger.info(
            "COMFY OUTPUTS prompt_id=%s count=%s",
            prompt_id,
            len(images),
        )
        for image in images:
            logger.info(
                "COMFY OUTPUT FETCH prompt_id=%s filename=%s subfolder=%s",
                prompt_id,
                image.filename,
                image.subfolder,
            )
        logger.info(
            "ComfyUI history images prompt_id=%s count=%s names=%s",
            prompt_id,
            len(images),
            [img.filename for img in images],
        )
        return ComfyPromptOutputs(
            prompt_id=prompt_id,
            client_id=run_client_id,
            seed_used=seed_used,
            images=images,
        )

    def read_output_bytes(self, image: ComfyOutputImage) -> bytes:
        path = resolve_comfy_output_path(
            self._config.comfyui_output_root,
            image.filename,
            image.subfolder,
            image.type,
        )
        if not path.is_file():
            logger.warning(
                "ComfyUI output file missing path=%s output_root=%s filename=%s subfolder=%s",
                path,
                self._config.comfyui_output_root,
                image.filename,
                image.subfolder,
            )
            raise OutputMissingError(
                "Generated output file not found",
                filename=image.filename,
                subfolder=image.subfolder,
                resolved_path=str(path),
                output_root=str(self._config.comfyui_output_root.resolve()),
            )
        size = path.stat().st_size
        logger.info(
            "COMFY RESULT READY filename=%s bytes=%s path=%s",
            image.filename,
            size,
            path,
        )
        return path.read_bytes()

    async def run_workflow_to_history(
        self,
        workflow: dict[str, Any],
        *,
        request_id: str,
        cancel_event: asyncio.Event | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Submit workflow and return ``(prompt_id, history)`` without requiring SaveImage."""
        prompt_id = await self.submit_prompt(workflow, request_id)
        await self.wait_for_prompt_terminal(
            prompt_id,
            request_id,
            cancel_event=cancel_event,
        )
        history = await self.fetch_history(prompt_id)
        return prompt_id, history

    @staticmethod
    def extract_node_outputs(
        history: dict[str, Any],
        prompt_id: str,
        node_id: str,
    ) -> dict[str, Any]:
        entry = ComfyUIClient.history_entry_for_prompt(history, prompt_id)
        if entry is None:
            return {}
        outputs = entry.get("outputs") or {}
        if not isinstance(outputs, dict):
            return {}
        node_output = outputs.get(node_id)
        if isinstance(node_output, dict):
            return node_output
        return {}

    async def run_workflow(
        self,
        workflow: dict[str, Any],
        *,
        request_id: str,
        timeout: float | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> list[dict[str, Any]]:
        previous_timeout = self._timeout_override
        if timeout is not None:
            self._timeout_override = timeout
        try:
            seed_used = int(workflow.get("5", {}).get("inputs", {}).get("seed", 0))
            outputs = await self.run_prompt(
                workflow,
                seed_used=seed_used,
                client_id=request_id,
                cancel_event=cancel_event,
            )
            results: list[dict[str, Any]] = []
            for image in outputs.images:
                data = self.read_output_bytes(image)
                results.append(
                    {
                        "prompt_id": outputs.prompt_id,
                        "filename": image.filename,
                        "subfolder": image.subfolder,
                        "type": image.type,
                        "data": data,
                        "mime_type": "image/png",
                    },
                )
            return results
        finally:
            self._timeout_override = previous_timeout


def _message_matches_prompt(message: dict[str, Any], prompt_id: str) -> bool:
    data = message.get("data")
    if not isinstance(data, dict):
        return False
    return data.get("prompt_id") == prompt_id
