"""Worker module: run alone (subprocess) — MCP session_manager is single-use per process."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import rag_server
from image_generation import GeneratedImage, ImageGenerationResult
from mcp_test_helpers import mcp_initialize


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
        await asyncio.sleep(delay_seconds)
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
def mcp_image_client(tmp_path: pytest.TempPath) -> TestClient:
    os.environ["RAG_ADMIN_TOKEN"] = "test-token"
    os.environ["IMAGE_GENERATION_ENABLED"] = "true"
    os.environ["IMAGE_ANALYSIS_ENABLED"] = "false"
    os.environ["IMAGE_SEGMENTATION_ENABLED"] = "false"
    os.environ["IMAGE_UPSCALE_ENABLED"] = "false"
    os.environ["MCP_STATELESS_HTTP"] = "false"
    os.environ["RAG_STAGING_DIR"] = str(tmp_path / "staging")
    os.environ["RAG_BUNDLE_STATE_DIR"] = str(tmp_path / "state")
    os.environ["RAG_INDEX_FILE"] = str(tmp_path / "missing.faiss")
    os.environ["RAG_METADATA_FILE"] = str(tmp_path / "missing.jsonl")
    os.environ["RAG_METRICS_PROBE_ENABLED"] = "false"
    rag_server.rag_service.clear()

    slow = _slow_mock_backend(3.0)

    async def _create_backend() -> MagicMock:
        return slow

    import image_generation

    image_generation.create_comfy_backend_if_ready = _create_backend  # type: ignore[method-assign]

    client = TestClient(rag_server.app, base_url="http://localhost:8000")
    client.__enter__()
    yield client
    client.__exit__(None, None, None)


def test_mcp_generate_image_waits_for_slow_backend(
    mcp_image_client: TestClient,
) -> None:
    session_headers, _init = mcp_initialize(mcp_image_client)

    response = mcp_image_client.post(
        "/mcp/",
        json={
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "generate_image",
                "arguments": {"prompt": "red cube"},
            },
        },
        headers=session_headers,
        timeout=30.0,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("id") == 10
    error = payload.get("error")
    assert error is None, error
    result = payload["result"]
    assert result.get("isError") is not True
    content = result.get("content") or []
    assert any(block.get("type") == "image" for block in content)
