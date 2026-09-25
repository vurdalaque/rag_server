"""Worker: initialized MCP client waits for a long tool over a real socket.

This follows the radius-bot path: ``streamable_http_client`` plus ``ClientSession``,
``initialize()``, then a long ``tools/call`` on the same stateful transport.
A real socket is required to observe an early transport close.
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
import httpx2
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import rag_server
from comfy_progress import notify_wait_progress
from image_generation import GeneratedImage, ImageGenerationResult

SLOW_SECONDS = 6.0
PROGRESS_TICK_SECONDS = 2.0

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


def test_mcp_initialized_client_waits_for_slow_backend(tmp_rag_paths: None) -> None:
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
            health_timeout = httpx.Timeout(30.0, read=SLOW_SECONDS + 60.0)
            async with httpx.AsyncClient(timeout=health_timeout) as health_client:
                await _wait_until_healthy(health_client)

            mcp_timeout = httpx2.Timeout(30.0, read=SLOW_SECONDS + 60.0)
            async with httpx2.AsyncClient(timeout=mcp_timeout) as mcp_client:
                async with streamable_http_client(
                    f"http://127.0.0.1:{port}/mcp/",
                    http_client=mcp_client,
                ) as transport:
                    async with ClientSession(transport[0], transport[1]) as session:
                        await session.initialize()
                        started = time.monotonic()
                        result = await session.call_tool(
                            "generate_image",
                            {"prompt": "red cube"},
                        )
                        elapsed = time.monotonic() - started
        finally:
            server.should_exit = True
            await server_task

        # Сессия и HTTP-соединение дожили до конца долгого вызова.
        assert elapsed >= SLOW_SECONDS, elapsed
        assert result.is_error is not True
        assert any(getattr(block, "type", None) == "image" for block in result.content)

    asyncio.run(_run())
