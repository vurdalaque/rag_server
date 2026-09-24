"""Frozen MCP image-generation v1 contract tests (mocked ComfyUI)."""

from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent

from image_generation import (
    ComfyUIBackend,
    GeneratedImage,
    ImageGenerationResult,
)
from image_generation_config import ImageGenerationConfig
from image_generation_errors import ComfyUIUnavailableError, UnsupportedParameterError
from image_mcp_tools import register_image_tools, reset_image_mcp_registration
from test_image_mcp_image import _sample_capabilities_payload

import comfy_workflow
from comfy_workflow import load_workflow_template
from image_generation_config import _DEFAULT_TEMPLATE


@pytest.fixture
def config(tmp_path: Any) -> ImageGenerationConfig:
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(load_workflow_template(_DEFAULT_TEMPLATE)),
        encoding="utf-8",
    )
    return ImageGenerationConfig(
        enabled=True,
        comfy_base_url="http://127.0.0.1:8188",
        comfyui_output_root=tmp_path / "output",
        workflow_template_path=template,
        probe_timeout=5.0,
        generate_timeout=30.0,
        upload_timeout_seconds=5.0,
        output_read_timeout_seconds=5.0,
        max_images=4,
        max_output_bytes=20 * 1024 * 1024,
        max_reference_images=10,
        max_reference_bytes=1024 * 1024,
        safety_enabled=False,
        safety_fail_closed=True,
        default_resolution=1024,
        min_resolution=256,
        max_resolution=2048,
        default_steps=25,
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


def _png_b64(size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (1, 2, 3)) -> str:
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _truncated_png_b64() -> str:
    data = base64.b64decode(_png_b64())
    return base64.b64encode(data[:-40]).decode("ascii")


@pytest.fixture(autouse=True)
def _reset_tools() -> None:
    reset_image_mcp_registration()
    yield
    reset_image_mcp_registration()


@pytest.fixture
def mcp_server() -> tuple[MCPServer, MagicMock]:
    server = MCPServer("contract-test")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value=_sample_capabilities_payload())
    backend.generate = AsyncMock(
        return_value=ImageGenerationResult(
            images=[GeneratedImage(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png")],
            seed=1,
            prompt_id="pid",
            timings={},
        ),
    )
    register_image_tools(server, backend)
    return server, backend


def test_capabilities_schema_and_aliases(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, _ = mcp_server
    result = asyncio.run(server.call_tool("image_generation_capabilities", {}))
    body = result.structured_content
    assert body is not None
    assert body["backend"] == "comfyui"
    assert body["inputs"]["sketch"]["semantics"] == "composition_guidance"
    assert body["resolution"] == {"min": 256, "max": 2048}
    assert body["safety_validation_enabled"] is False
    assert body["safety_enabled"] is False
    assert body["max_reference_images"] == 10


def test_prompt_only_generate(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, backend = mcp_server
    result = asyncio.run(server.call_tool("generate_image", {"prompt": "a red ball"}))
    assert result.is_error is not True
    backend.generate.assert_awaited_once()
    request = backend.generate.await_args.args[0]
    assert request.reference_images == ()
    assert request.mask is None
    assert request.sketch is None


def test_one_reference_image(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, backend = mcp_server
    encoded = _png_b64()
    asyncio.run(
        server.call_tool(
            "generate_image",
            {"prompt": "match style", "reference_images": [encoded]},
        ),
    )
    request = backend.generate.await_args.args[0]
    assert len(request.reference_images) == 1


def test_multiple_reference_images(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, backend = mcp_server
    refs = [_png_b64(color=(i, i, i)) for i in range(3)]
    asyncio.run(
        server.call_tool(
            "generate_image",
            {"prompt": "scene", "reference_images": refs},
        ),
    )
    request = backend.generate.await_args.args[0]
    assert len(request.reference_images) == 3


def test_valid_sketch_and_mask(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, backend = mcp_server
    asyncio.run(
        server.call_tool(
            "generate_image",
            {
                "prompt": "edit",
                "sketch": _png_b64(color=(10, 20, 30)),
                "mask": _png_b64(color=(200, 200, 200)),
            },
        ),
    )
    request = backend.generate.await_args.args[0]
    assert request.sketch is not None
    assert request.mask is not None


def test_reference_sketch_mask_together(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, backend = mcp_server
    asyncio.run(
        server.call_tool(
            "generate_image",
            {
                "prompt": "compose",
                "reference_images": [_png_b64()],
                "sketch": _png_b64(color=(5, 5, 5)),
                "mask": _png_b64(color=(9, 9, 9)),
            },
        ),
    )
    request = backend.generate.await_args.args[0]
    assert len(request.reference_images) == 1
    assert request.sketch and request.mask


def test_image_count_two_and_four(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, backend = mcp_server
    backend.generate = AsyncMock(
        return_value=ImageGenerationResult(
            images=[
                GeneratedImage(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png"),
                GeneratedImage(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png"),
            ],
            seed=1,
            prompt_id="pid",
            timings={},
        ),
    )
    result = asyncio.run(
        server.call_tool("generate_image", {"prompt": "x", "image_count": 2}),
    )
    images = [b for b in result.content if isinstance(b, ImageContent)]
    assert len(images) == 2
    assert result.structured_content["image_count"] == 2

    backend.generate = AsyncMock(
        return_value=ImageGenerationResult(
            images=[
                GeneratedImage(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png")
                for _ in range(4)
            ],
            seed=1,
            prompt_id="pid",
            timings={},
        ),
    )
    result = asyncio.run(
        server.call_tool("generate_image", {"prompt": "x", "image_count": 4}),
    )
    assert result.structured_content["image_count"] == 4


def test_unequal_width_height_rejected(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, backend = mcp_server
    backend.generate = AsyncMock(
        side_effect=UnsupportedParameterError(
            "width and height must be equal for the current backend",
            width=768,
            height=1024,
            constraint="width_must_equal_height",
        ),
    )
    result = asyncio.run(
        server.call_tool(
            "generate_image",
            {"prompt": "x", "width": 768, "height": 1024},
        ),
    )
    assert result.is_error is True
    body = json.loads(next(b for b in result.content if b.type == "text").text)
    assert body["error"]["code"] == "unsupported_parameter"
    assert body["error"]["details"]["constraint"] == "width_must_equal_height"


def test_corrupt_reference_mask_sketch_codes(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, _ = mcp_server
    bad = _truncated_png_b64()

    for field, args in (
        ("invalid_reference_image", {"reference_images": [bad]}),
        ("invalid_mask", {"mask": bad}),
        ("invalid_sketch", {"sketch": bad}),
    ):
        result = asyncio.run(server.call_tool("generate_image", {"prompt": "x", **args}))
        assert result.is_error is True
        body = json.loads(next(b for b in result.content if b.type == "text").text)
        assert body["error"]["code"] == field


def test_more_than_ten_references_rejected(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, backend = mcp_server
    from image_generation_errors import TooManyImagesError

    backend.generate = AsyncMock(
        side_effect=TooManyImagesError("too many", count=11, max_images=10),
    )
    refs = [_png_b64(color=(1, 1, 1)) for _ in range(11)]
    result = asyncio.run(
        server.call_tool(
            "generate_image",
            {"prompt": "x", "reference_images": refs},
        ),
    )
    body = json.loads(next(b for b in result.content if b.type == "text").text)
    assert body["error"]["code"] == "invalid_request"


def test_backend_unavailable_mapped(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, backend = mcp_server
    backend.generate = AsyncMock(
        side_effect=ComfyUIUnavailableError("down", reason="connection refused"),
    )
    result = asyncio.run(
        server.call_tool("generate_image", {"prompt": "x"}),
    )
    body = json.loads(next(b for b in result.content if b.type == "text").text)
    assert body["error"]["code"] == "backend_unavailable"


def test_no_internal_path_leak_in_success(mcp_server: tuple[MCPServer, MagicMock]) -> None:
    server, backend = mcp_server
    backend.generate = AsyncMock(
        return_value=ImageGenerationResult(
            images=[
                GeneratedImage(
                    data=b"\x89PNG\r\n\x1a\n",
                    mime_type="image/png",
                    filename="mcp/secret/result.png",
                ),
            ],
            seed=1,
            prompt_id="internal-pid",
            timings={},
        ),
    )
    result = asyncio.run(server.call_tool("generate_image", {"prompt": "x"}))
    dumped = json.dumps(result.structured_content)
    assert "mcp/" not in dumped
    assert "result.png" not in dumped


def test_backend_capabilities_payload_shape() -> None:
    config = ImageGenerationConfig(
        enabled=True,
        comfy_base_url="http://127.0.0.1:8188",
        comfyui_output_root=__import__("pathlib").Path("."),
        workflow_template_path=__import__("pathlib").Path("t.json"),
        probe_timeout=1.0,
        generate_timeout=1.0,
        upload_timeout_seconds=1.0,
        output_read_timeout_seconds=1.0,
        max_images=4,
        max_output_bytes=20 * 1024 * 1024,
        max_reference_images=10,
        max_reference_bytes=10 * 1024 * 1024,
        safety_enabled=True,
        safety_fail_closed=True,
        default_resolution=1024,
        min_resolution=256,
        max_resolution=2048,
        default_steps=25,
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
    payload = asyncio.run(backend.capabilities())
    assert payload["inputs"]["mask"]["semantics"] == "soft_region_guidance"
    assert payload["safety_validation_enabled"] is True
    assert payload["safety_enabled"] is True


def test_workflow_sketch_mask_augment_prompt(config: ImageGenerationConfig) -> None:
    from test_image_generation_comfy import SAMPLERS, SCHEDULERS

    cfg = config
    params = comfy_workflow.WorkflowBuildParams(
        request_id="req-1",
        prompt="hello",
        image_inputs=(
            comfy_workflow.WorkflowImageInput("ref.png", "reference"),
            comfy_workflow.WorkflowImageInput("sketch.png", "sketch"),
            comfy_workflow.WorkflowImageInput("mask.png", "mask"),
        ),
    )
    result = comfy_workflow.build_image_workflow(
        params,
        cfg,
        allowed_samplers=SAMPLERS,
        allowed_schedulers=SCHEDULERS,
    )
    prompt = result.workflow["4"]["inputs"]["prompt"]
    assert "sketch" in prompt.lower() or "composition" in prompt.lower()
    assert "mask" in prompt.lower() or "region" in prompt.lower()


def test_resolve_square_resolution_mismatch(config: ImageGenerationConfig) -> None:
    with pytest.raises(UnsupportedParameterError) as exc:
        comfy_workflow.resolve_square_resolution(
            config,
            width=768,
            height=1024,
        )
    assert exc.value.details["constraint"] == "width_must_equal_height"
