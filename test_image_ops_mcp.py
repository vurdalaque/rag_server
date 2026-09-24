"""Contract tests for analyze / segment / upscale MCP tools (tools/list schemas)."""

from __future__ import annotations

import asyncio
import base64
import inspect
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.mcpserver import MCPServer

import image_mcp_tools
from image_analysis import AnalysisResult
from image_mcp_tools import reset_image_mcp_registration
from image_segmentation import SegmentRequest
from image_upscale import UpscaleResult
from test_image_contract import _png_b64
from test_image_mcp_image import _sample_capabilities_payload

IMAGE_OPS_TOOL_NAMES: tuple[str, ...] = (
    "analyze_image",
    "segment_image",
    "upscale_image",
)

try:
    from image_mcp_tools import ImageMcpBackends as _ImageMcpBackends
except ImportError:

    @dataclass(frozen=True)
    class _ImageMcpBackends:
        generation: Any | None = None
        analyzer: Any | None = None
        segmenter: Any | None = None
        upscaler: Any | None = None


ImageMcpBackends = _ImageMcpBackends


def _mock_generation_backend() -> MagicMock:
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value=_sample_capabilities_payload())
    return backend


def _mock_analyzer_backend() -> MagicMock:
    backend = MagicMock()
    backend.probe = AsyncMock(return_value=True)
    backend.analyze = AsyncMock(
        return_value=AnalysisResult(text="deterministic analysis", timings={"llm": 0.01}),
    )
    return backend


def _mock_segmenter_backend() -> MagicMock:
    mask_png = base64.b64decode(_png_b64((8, 8), color=(255, 255, 255)))
    backend = MagicMock()
    backend.probe = AsyncMock(return_value=True)
    backend.segment = AsyncMock(return_value=mask_png)
    backend.capabilities = AsyncMock(
        return_value={
            "backend": "comfyui",
            "modes": {
                "text": True,
                "points": True,
                "box": True,
                "mask_refinement": True,
            },
        },
    )
    backend.last_timings = {"total_ms": 1.0}
    return backend


def _mock_upscaler_backend() -> MagicMock:
    png = base64.b64decode(_png_b64((16, 16)))
    backend = MagicMock()
    backend.probe = AsyncMock(return_value=True)
    backend.capabilities = AsyncMock(return_value={"default_scale": 2.0, "max_scale": 4.0})
    backend.upscale = AsyncMock(
        return_value=UpscaleResult(
            image=png,
            width=32,
            height=32,
            timings={"upscale": 0.02},
        ),
    )
    return backend


def _default_backends() -> ImageMcpBackends:
    return ImageMcpBackends(
        generation=_mock_generation_backend(),
        analyzer=_mock_analyzer_backend(),
        segmenter=_mock_segmenter_backend(),
        upscaler=_mock_upscaler_backend(),
    )


def _ops_tools_present(server: MCPServer) -> bool:
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    return all(name in names for name in IMAGE_OPS_TOOL_NAMES)


def register_image_ops_for_tests(
    server: MCPServer,
    backends: ImageMcpBackends | None = None,
) -> bool:
    """Register generation + ops tools when integration exposes them."""
    backends = backends or _default_backends()
    register_mcp = getattr(image_mcp_tools, "register_image_mcp_backends", None)
    if register_mcp is not None:
        register_mcp(server, backends)
        return _ops_tools_present(server)

    register_gen = image_mcp_tools.register_image_tools
    sig = inspect.signature(register_gen)
    if "backends" in sig.parameters:
        generation = backends.generation or _mock_generation_backend()
        register_gen(server, generation, backends=backends)
        return _ops_tools_present(server)

    if backends.generation is not None:
        register_gen(server, backends.generation)
    return _ops_tools_present(server)


def _tool_map(server: MCPServer) -> dict[str, Any]:
    tools = asyncio.run(server.list_tools())
    return {tool.name: tool for tool in tools}


def _field_description(schema: dict[str, Any], field: str) -> str:
    properties = schema.get("properties") or {}
    block = properties.get(field) or {}
    return str(block.get("description") or "")


@pytest.fixture(autouse=True)
def _reset_image_tools_state() -> None:
    reset_image_mcp_registration()
    yield
    reset_image_mcp_registration()


@pytest.fixture
def ops_mcp_server() -> tuple[MCPServer, ImageMcpBackends]:
    server = MCPServer("image-ops-contract")
    backends = _default_backends()
    if not register_image_ops_for_tests(server, backends):
        pytest.skip("analyze/segment/upscale MCP tools are not registered yet")
    return server, backends


def test_image_ops_tools_listed_when_registered(
    ops_mcp_server: tuple[MCPServer, ImageMcpBackends],
) -> None:
    server, _ = ops_mcp_server
    names = set(_tool_map(server))
    for tool_name in IMAGE_OPS_TOOL_NAMES:
        assert tool_name in names


def test_analyze_image_input_schema_required_and_optional(
    ops_mcp_server: tuple[MCPServer, ImageMcpBackends],
) -> None:
    server, _ = ops_mcp_server
    tool = _tool_map(server)["analyze_image"]
    schema: dict[str, Any] = tool.input_schema
    properties = schema.get("properties") or {}

    assert schema.get("required") == ["images"]
    assert "images" in properties
    assert "instruction" in properties
    assert "instruction" not in (schema.get("required") or [])
    assert "sketch" not in properties
    assert "mask" not in properties

    description = tool.description.lower()
    assert any(
        token in description
        for token in ("analy", "describe", "visual", "multimodal")
    )


def test_segment_image_input_schema_required_and_optional(
    ops_mcp_server: tuple[MCPServer, ImageMcpBackends],
) -> None:
    server, _ = ops_mcp_server
    tool = _tool_map(server)["segment_image"]
    schema: dict[str, Any] = tool.input_schema
    properties = schema.get("properties") or {}

    assert schema.get("required") == ["image"]
    assert "image" in properties
    for optional in ("prompt", "points", "box", "mask"):
        assert optional in properties
        assert optional not in (schema.get("required") or [])
    assert "sketch" not in properties

    mask_desc = _field_description(schema, "mask").lower()
    assert mask_desc
    assert "sketch" not in mask_desc
    assert any(
        token in mask_desc
        for token in ("region", "refin", "where", "existing", "mask")
    )

    tool_desc = tool.description.lower()
    assert "segment" in tool_desc or "mask" in tool_desc


def test_upscale_image_input_schema_required_and_optional(
    ops_mcp_server: tuple[MCPServer, ImageMcpBackends],
) -> None:
    server, _ = ops_mcp_server
    tool = _tool_map(server)["upscale_image"]
    schema: dict[str, Any] = tool.input_schema
    properties = schema.get("properties") or {}

    assert schema.get("required") == ["image"]
    assert "image" in properties
    for optional in ("scale",):
        assert optional in properties
        assert optional not in (schema.get("required") or [])
    assert "sketch" not in properties
    assert "mask" not in properties

    description = tool.description.lower()
    assert any(token in description for token in ("upscale", "scale", "resolution"))


def test_segment_mask_not_confused_with_generate_sketch(
    ops_mcp_server: tuple[MCPServer, ImageMcpBackends],
) -> None:
    server, _ = ops_mcp_server
    by_name = _tool_map(server)
    segment_schema = by_name["segment_image"].input_schema
    segment_mask = _field_description(segment_schema, "mask").lower()

    assert "sketch" not in (segment_schema.get("properties") or {})
    assert "composition" not in segment_mask or "how" not in segment_mask

    if "generate_image" in by_name:
        gen_schema = by_name["generate_image"].input_schema
        gen_mask = _field_description(gen_schema, "mask").lower()
        gen_sketch = _field_description(gen_schema, "sketch").lower()
        assert gen_mask != gen_sketch
        assert any(token in gen_sketch for token in ("how", "composition", "layout"))
        assert any(token in gen_mask for token in ("where", "region", "spatial"))


def test_image_ops_tools_publish_object_output_schema(
    ops_mcp_server: tuple[MCPServer, ImageMcpBackends],
) -> None:
    server, _ = ops_mcp_server
    for name in IMAGE_OPS_TOOL_NAMES:
        schema = _tool_map(server)[name].output_schema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_analyze_image_delegates_to_analyzer_backend(
    ops_mcp_server: tuple[MCPServer, ImageMcpBackends],
) -> None:
    server, backends = ops_mcp_server
    encoded = _png_b64()
    result = asyncio.run(
        server.call_tool(
            "analyze_image",
            {"images": [encoded], "instruction": "What color is dominant?"},
        ),
    )
    assert result.is_error is not True
    backends.analyzer.analyze.assert_awaited_once()


def test_segment_image_delegates_to_segmenter_backend(
    ops_mcp_server: tuple[MCPServer, ImageMcpBackends],
) -> None:
    server, backends = ops_mcp_server
    encoded = _png_b64()
    result = asyncio.run(
        server.call_tool(
            "segment_image",
            {"image": encoded, "prompt": "center object"},
        ),
    )
    assert result.is_error is not True
    backends.segmenter.segment.assert_awaited_once()
    request = backends.segmenter.segment.await_args.args[0]
    assert isinstance(request, SegmentRequest)


def test_upscale_image_delegates_to_upscaler_backend(
    ops_mcp_server: tuple[MCPServer, ImageMcpBackends],
) -> None:
    server, backends = ops_mcp_server
    encoded = _png_b64()
    result = asyncio.run(
        server.call_tool(
            "upscale_image",
            {"image": encoded, "scale": 2.0},
        ),
    )
    assert result.is_error is not True
    backends.upscaler.upscale.assert_awaited_once()
