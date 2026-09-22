from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.mcpserver import MCPServer

import image_mcp_tools
from image_mcp_tools import (
    IMAGE_MCP_TOOL_NAMES,
    list_mcp_tool_names,
    list_ping_tool_names,
    register_image_tools,
    reset_image_mcp_registration,
)


@pytest.fixture(autouse=True)
def _reset_image_tools_state() -> None:
    reset_image_mcp_registration()
    yield
    reset_image_mcp_registration()


def test_list_mcp_tool_names_without_image_tools() -> None:
    assert list_mcp_tool_names(include_image_tools=False) == [
        "search_project",
        "web_search",
        "ask_project",
        "ping",
    ]
    assert list_ping_tool_names() == [
        "search_project",
        "web_search",
        "ask_project",
    ]


def test_list_mcp_tool_names_with_image_tools() -> None:
    names = list_mcp_tool_names(include_image_tools=True)
    assert names[-2:] == list(IMAGE_MCP_TOOL_NAMES)


def test_register_image_tools_adds_tools_when_backend_available() -> None:
    server = MCPServer("image-test")
    backend = MagicMock()
    backend.capabilities = AsyncMock(
        return_value={
            "backend": "comfyui",
            "max_images": 4,
        }
    )

    register_image_tools(server, backend)
    assert image_mcp_tools.image_tools_registered()

    tools = asyncio.run(server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert tool_names == set(IMAGE_MCP_TOOL_NAMES)


def test_register_image_tools_is_idempotent() -> None:
    server = MCPServer("image-test-idempotent")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value={})

    register_image_tools(server, backend)
    register_image_tools(server, backend)

    tools = asyncio.run(server.list_tools())
    assert len(tools) == len(IMAGE_MCP_TOOL_NAMES)


def test_generate_image_tool_returns_error_without_backend() -> None:
    server = MCPServer("image-test-missing-backend")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value={})
    register_image_tools(server, backend)

    reset_image_mcp_registration()
    image_mcp_tools._registered = True
    image_mcp_tools._backend = None

    result = asyncio.run(server.call_tool("generate_image", {"prompt": "a red cube"}))
    assert result.is_error is True
    text = next(block for block in result.content if block.type == "text")
    assert json.loads(text.text)["error"]["code"] == "backend_unavailable"


def test_capabilities_tool_delegates_to_backend() -> None:
    server = MCPServer("image-test-capabilities")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value={"max_images": 2})
    register_image_tools(server, backend)

    result = asyncio.run(server.call_tool("image_generation_capabilities", {}))
    assert result.structured_content == {"max_images": 2}
    backend.capabilities.assert_awaited_once()


def test_generate_image_tool_schema_is_model_agnostic() -> None:
    server = MCPServer("image-test-schema")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value={})
    register_image_tools(server, backend)

    tools = asyncio.run(server.list_tools())
    generate_tool = next(tool for tool in tools if tool.name == "generate_image")
    schema: dict[str, Any] = generate_tool.input_schema
    properties = schema.get("properties", {})
    assert "prompt" in properties
    assert "sampler" in properties
    assert "scheduler" in properties
    assert "unet_name" not in properties


def test_probe_skipped_when_disabled() -> None:
    from image_generation import ComfyUIBackend
    from image_generation_config import ImageGenerationConfig

    config = ImageGenerationConfig(
        enabled=False,
        comfy_base_url="http://127.0.0.1:8188",
        comfyui_output_root=__import__("pathlib").Path("."),
        workflow_template_path=__import__("pathlib").Path("template.json"),
        probe_timeout=1.0,
        generate_timeout=1.0,
        upload_timeout_seconds=1.0,
        output_read_timeout_seconds=1.0,
        max_images=1,
        max_output_bytes=1024,
        max_reference_images=0,
        max_reference_bytes=1024,
        safety_enabled=False,
        safety_fail_closed=True,
        default_resolution=512,
        min_resolution=256,
        max_resolution=2048,
        default_steps=10,
        min_steps=1,
        max_steps=100,
        default_cfg=1.0,
        min_cfg=0.0,
        max_cfg=30.0,
        default_sampler="euler",
        default_scheduler="simple",
        default_denoise=1.0,
        min_denoise=0.0,
        max_denoise=1.0,
        allowed_input_mime_types=frozenset({"image/png"}),
    )
    backend = ComfyUIBackend(config)

    assert asyncio.run(backend.probe()) is False
