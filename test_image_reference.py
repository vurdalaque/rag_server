from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from PIL import Image

from comfy_client import ComfyUIClient, ComfyUploadedImage
from image_generation import ComfyUIBackend, GenerateImageRequest, ImageGenerationResult
from image_generation_config import ImageGenerationConfig
from image_reference import (
    canonicalize_reference_png_for_comfy,
    decode_and_validate_reference_image,
    decode_reference_image_base64,
    validate_reference_image_bytes,
)
from image_generation_errors import (
    InvalidMaskImageError,
    InvalidReferenceImageError,
    InvalidSketchImageError,
)
from image_reference import decode_and_validate_mask_image, decode_and_validate_sketch_image
from image_mcp_tools import register_image_tools, reset_image_mcp_registration
from mcp.server.mcpserver import MCPServer


def _rgb_png_256() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (256, 256), color=(40, 80, 120)).save(buffer, format="PNG")
    data = buffer.getvalue()
    validate_reference_image_bytes(data, index=0)
    return data


def test_canonicalize_produces_fresh_decodable_png() -> None:
    data = _rgb_png_256()
    out = canonicalize_reference_png_for_comfy(data, index=0)
    assert out.startswith(b"\x89PNG\r\n\x1a\n")
    validate_reference_image_bytes(out, index=0)
    with Image.open(BytesIO(out)) as image:
        image.load()
        assert image.size == (256, 256)


def test_valid_png_passes_full_decode() -> None:
    data = _rgb_png_256()
    encoded = base64.b64encode(data).decode("ascii")
    decoded = decode_and_validate_reference_image(encoded, index=0)
    assert decoded == data


def test_data_url_and_whitespace_base64() -> None:
    data = _rgb_png_256()
    raw = base64.b64encode(data).decode("ascii")
    wrapped = "data:image/png;base64," + "\n".join(
        raw[i : i + 40] for i in range(0, len(raw), 40)
    )
    decoded = decode_and_validate_reference_image(wrapped, index=0)
    assert decoded == data


def test_truncated_png_rejected() -> None:
    data = _rgb_png_256()[:-80]
    encoded = base64.b64encode(data).decode("ascii")
    with pytest.raises(InvalidReferenceImageError) as exc:
        decode_and_validate_reference_image(encoded, index=2)
    assert exc.value.code == "invalid_reference_image"
    assert exc.value.details["index"] == 2


def test_truncated_mask_rejected() -> None:
    data = _rgb_png_256()[:-80]
    encoded = base64.b64encode(data).decode("ascii")
    with pytest.raises(InvalidMaskImageError) as exc:
        decode_and_validate_mask_image(encoded)
    assert exc.value.code == "invalid_mask"


def test_truncated_sketch_rejected() -> None:
    data = _rgb_png_256()[:-80]
    encoded = base64.b64encode(data).decode("ascii")
    with pytest.raises(InvalidSketchImageError) as exc:
        decode_and_validate_sketch_image(encoded)
    assert exc.value.code == "invalid_sketch"


def test_upload_verify_detects_corruption_on_wire() -> None:
    original = _rgb_png_256()
    uploaded_meta = ComfyUploadedImage(name="ref_0.png", subfolder="", type="input")
    truncated = original[:-64]

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/upload/image":
            return httpx.Response(200, json={"name": "ref_0.png", "subfolder": "", "type": "input"})
        if request.url.path == "/view":
            return httpx.Response(200, content=truncated)
        return httpx.Response(404)

    client = ComfyUIClient(base_url="http://127.0.0.1:8188")
    client._http = httpx.AsyncClient(
        base_url="http://127.0.0.1:8188",
        transport=httpx.MockTransport(handler),
    )

    async def run() -> None:
        uploaded = await client.upload_image("ref_0.png", original, "image/png")
        with pytest.raises(Exception) as exc:
            await client.verify_uploaded_image_bytes(uploaded, original)
        assert "mismatch" in str(exc.value).lower() or "do not match" in str(exc.value).lower()

    asyncio.run(run())


def test_mcp_generate_image_accepts_valid_reference() -> None:
    reset_image_mcp_registration()
    server = MCPServer("ref-test")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value={})
    png = _rgb_png_256()
    backend.generate = AsyncMock(
        return_value=ImageGenerationResult(
            images=[],
            seed=1,
            prompt_id="p",
            timings={},
        )
    )
    register_image_tools(server, backend)

    encoded = base64.b64encode(png).decode("ascii")
    asyncio.run(
        server.call_tool(
            "generate_image",
            {"prompt": "test", "reference_images": [encoded]},
        )
    )
    backend.generate.assert_awaited_once()
    request: GenerateImageRequest = backend.generate.await_args.args[0]
    assert request.reference_images[0].startswith(b"\x89PNG")
    validate_reference_image_bytes(request.reference_images[0], index=0)


def test_upload_references_calls_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    png = _rgb_png_256()
    config = ImageGenerationConfig(
        enabled=True,
        comfy_base_url="http://127.0.0.1:8188",
        comfyui_output_root=__import__("pathlib").Path("."),
        workflow_template_path=__import__("pathlib").Path("t.json"),
        probe_timeout=5.0,
        generate_timeout=30.0,
        upload_timeout_seconds=5.0,
        output_read_timeout_seconds=5.0,
        max_images=1,
        max_output_bytes=1024 * 1024,
        max_reference_images=1,
        max_reference_bytes=1024 * 1024,
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
    mock_client = MagicMock()
    mock_client.upload_image = AsyncMock(
        return_value=ComfyUploadedImage(name="ref_0.png", subfolder="", type="input"),
    )
    mock_client.verify_uploaded_image_bytes = AsyncMock()
    backend._client_instance = lambda: mock_client  # type: ignore[method-assign]

    request = GenerateImageRequest(prompt="x", reference_images=(png,))
    asyncio.run(backend._build_workflow_image_inputs(request))
    mock_client.upload_image.assert_awaited_once()
    mock_client.verify_uploaded_image_bytes.assert_awaited_once()
