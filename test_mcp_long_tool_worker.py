"""Worker: production defaults (stateless Streamable HTTP + JSON on POST) over a real socket.

Reproduces the bare ``tools/call`` POST the Realm bot and ``tests/mcp_smoke.sh``
use: no ``initialize``, no ``Mcp-Session-Id``, ``MCP-Protocol-Version: 2024-11-05``.
A real socket is required: ``ASGITransport`` buffers until the response completes
and ignores timeouts, so it cannot observe an early drop.
"""

from __future__ import annotations

import asyncio
import os
import socket
import time

os.environ["RAG_ADMIN_TOKEN"] = "test-token"
os.environ["IMAGE_GENERATION_ENABLED"] = "true"
os.environ["IMAGE_ANALYSIS_ENABLED"] = "false"
os.environ["IMAGE_SEGMENTATION_ENABLED"] = "false"
os.environ["IMAGE_UPSCALE_ENABLED"] = "false"
os.environ["RAG_METRICS_PROBE_ENABLED"] = "false"

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import uvicorn

import rag_server
from comfy_progress import notify_wait_progress
from image_generation import GeneratedImage, ImageGenerationResult

SLOW_SECONDS = 6.0
PROGRESS_TICK_SECONDS = 2.0

CALL_BODY = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {"name": "generate_image", "arguments": {"prompt": "red cube"}},
}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _slow_mock_backend(delay_seconds: float) -> MagicMock:
    backend = MagicMock()
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"x" * 32

    async def _capabilities() -> dict[str, Any]:
        return {
            "backend": "comfyui",
            "max_images": 1,
            "max_output_bytes": 20971520,
            "max_reference_images": 0,
            "max_reference_bytes": 0,
            "inputs": {},
            "samplers": {"sampler_name": ["euler"], "scheduler": ["simple"]},
            "safety_validation_enabled": False,
        }

    async def _generate(_request: Any) -> ImageGenerationResult:
        # Повторяет цикл ожидания Comfy: прогресс-тики во время долгой работы.
        started = time.monotonic()
        while time.monotonic() - started < delay_seconds:
            await asyncio.sleep(PROGRESS_TICK_SECONDS)
            await notify_wait_progress(time.monotonic() - started, "running")
        return ImageGenerationResult(
            images=[
                GeneratedImage(
                    data=png_bytes,
                    mime_type="image/png",
                    filename="out.png",
                )
            ],
            seed=1,
            prompt_id="test-prompt",
            timings={"comfy_execute": delay_seconds},
        )

    backend.capabilities = AsyncMock(side_effect=_capabilities)
    backend.generate = AsyncMock(side_effect=_generate)
    backend.probe = AsyncMock(return_value=True)
    backend.aclose = AsyncMock()
    return backend


@pytest.fixture
def tmp_rag_paths(tmp_path: pytest.TempPath) -> None:
    os.environ["RAG_STAGING_DIR"] = str(tmp_path / "staging")
    os.environ["RAG_BUNDLE_STATE_DIR"] = str(tmp_path / "state")
    os.environ["RAG_INDEX_FILE"] = str(tmp_path / "missing.faiss")
    os.environ["RAG_METADATA_FILE"] = str(tmp_path / "missing.jsonl")
    rag_server.rag_service.clear()


def test_mcp_bare_post_waits_for_slow_backend(tmp_rag_paths: None) -> None:
    slow = _slow_mock_backend(SLOW_SECONDS)

    async def _create_backend() -> MagicMock:
        return slow

    import image_generation

    image_generation.create_comfy_backend_if_ready = _create_backend  # type: ignore[method-assign]

    port = _free_port()

    async def _wait_until_healthy(client: httpx.AsyncClient) -> None:
        for _ in range(120):
            try:
                health = await client.get(f"http://127.0.0.1:{port}/health")
            except httpx.RequestError:
                await asyncio.sleep(0.25)
                continue
            if health.status_code == 200:
                return
            await asyncio.sleep(0.25)
        raise AssertionError("server did not become healthy")

    async def _run() -> None:
        config = uvicorn.Config(
            rag_server.app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        try:
            timeout = httpx.Timeout(30.0, read=SLOW_SECONDS + 60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                await _wait_until_healthy(client)

                started = time.monotonic()
                response = await client.post(
                    f"http://127.0.0.1:{port}/mcp/",
                    json=CALL_BODY,
                    headers={
                        "Accept": "application/json, text/event-stream",
                        "Content-Type": "application/json",
                        "MCP-Protocol-Version": "2024-11-05",
                    },
                )
                elapsed = time.monotonic() - started
        finally:
            server.should_exit = True
            await server_task

        assert response.status_code == 200, response.text
        # Stateless Streamable HTTP: никаких сессий и никакого initialize.
        assert "mcp-session-id" not in response.headers
        content_type = response.headers.get("content-type", "")
        assert content_type.startswith("application/json"), content_type
        # Соединение дожило до конца долгого вызова, а не оборвалось на первом тике.
        assert elapsed >= SLOW_SECONDS, elapsed

        payload = response.json()
        assert "error" not in payload, payload
        content = payload["result"]["content"]
        assert [block for block in content if block.get("type") == "image"], content

    asyncio.run(_run())
