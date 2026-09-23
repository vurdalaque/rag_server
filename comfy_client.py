"""Async ComfyUI HTTP + WebSocket client."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """Upload images, submit workflows, and wait for completion via WebSocket."""

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
        return ComfyUploadedImage(
            name=str(name),
            subfolder=str(body.get("subfolder") or ""),
            type=str(body.get("type") or "input"),
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
        logger.info("ComfyUI prompt submitted prompt_id=%s", prompt_id)
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
                    if not _message_matches_prompt(message, prompt_id):
                        continue

                    msg_type = message.get("type")
                    data = message.get("data") or {}

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
        await self.wait_for_prompt_ws(
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
        logger.info(
            "ComfyUI output read path=%s bytes=%s",
            path,
            path.stat().st_size,
        )
        return path.read_bytes()

    async def run_workflow(
        self,
        workflow: dict[str, Any],
        *,
        request_id: str,
        timeout: float | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> list[dict[str, Any]]:
        if timeout is not None:
            self._timeout_override = timeout
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
                    "filename": image.filename,
                    "subfolder": image.subfolder,
                    "type": image.type,
                    "data": data,
                    "mime_type": "image/png",
                },
            )
        return results


def _message_matches_prompt(message: dict[str, Any], prompt_id: str) -> bool:
    data = message.get("data")
    if not isinstance(data, dict):
        return False
    return data.get("prompt_id") == prompt_id
