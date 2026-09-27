"""Unit tests for image upscale validation and Comfy SR workflow."""

from __future__ import annotations

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from comfy_client import ComfyUploadedImage
from image_generation_errors import (
    InputTooLargeError,
    OutputTooLargeError,
    UnsupportedParameterError,
)
from image_upscale import (
    ComfyUpscaleBackend,
    UpscaleRequest,
    build_comfy_upscale_workflow,
    _pick_sr_model_name,
    _resolve_upscale_params,
)
from image_upscale_config import ImageUpscaleConfig


def _png(w: int, h: int) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (w, h), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _config(**overrides: object) -> ImageUpscaleConfig:
    base = dict(
        enabled=True,
        comfy_base_url="http://127.0.0.1:8188",
        comfyui_output_root=__import__("pathlib").Path("."),
        probe_timeout=5.0,
        upscale_timeout=60.0,
        upload_timeout_seconds=30.0,
        output_read_timeout_seconds=30.0,
        default_scale=4.0,
        max_scale=4.0,
        min_scale=4.0,
        max_input_bytes=1024 * 1024,
        max_output_bytes=4 * 1024 * 1024,
        max_input_dimension=512,
        max_output_dimension=2048,
        allowed_input_mime_types=frozenset({"image/png"}),
        comfy_upscale_model="RealESRGAN_x4plus.pth",
        supported_sr_scales=(4.0,),
    )
    base.update(overrides)
    return ImageUpscaleConfig(**base)


def test_default_scale_is_four() -> None:
    data = _png(64, 64)
    resolved = _resolve_upscale_params(UpscaleRequest(image=data), _config())
    assert resolved.scale == 4.0


def test_input_too_large_bytes() -> None:
    data = _png(64, 64)
    with pytest.raises(InputTooLargeError):
        _resolve_upscale_params(
            UpscaleRequest(image=data, scale=4.0),
            _config(max_input_bytes=100),
        )


def test_scale_without_configured_sr_model_is_rejected() -> None:
    with pytest.raises(UnsupportedParameterError):
        _resolve_upscale_params(
            UpscaleRequest(image=_png(64, 64), scale=2.0),
            _config(min_scale=1.0, max_scale=4.0),
        )


def test_output_dimension_limit() -> None:
    data = _png(400, 400)
    with pytest.raises(OutputTooLargeError):
        _resolve_upscale_params(
            UpscaleRequest(image=data, scale=4.0),
            _config(max_output_dimension=1000),
        )


def test_build_workflow_uses_image_upscale_with_model() -> None:
    workflow = build_comfy_upscale_workflow(
        request_id="req-1",
        uploaded_filename="up.png",
        model_name="RealESRGAN_x4plus.pth",
    )
    assert workflow["2"]["class_type"] == "UpscaleModelLoader"
    assert workflow["3"]["class_type"] == "ImageUpscaleWithModel"


def test_pick_sr_model_for_scale_four() -> None:
    data = _png(32, 32)
    resolved = _resolve_upscale_params(
        UpscaleRequest(image=data, scale=4.0),
        _config(),
    )
    name, factor = _pick_sr_model_name(
        resolved,
        _config(),
        ["RealESRGAN_x4plus.pth"],
    )
    assert name == "RealESRGAN_x4plus.pth"
    assert factor == 4.0


def test_target_dimensions_must_match_exact_x4() -> None:
    data = _png(200, 200)
    resolved = _resolve_upscale_params(
        UpscaleRequest(image=data, target_width=800, target_height=800),
        _config(),
    )
    assert resolved.scale == 4.0

    with pytest.raises(UnsupportedParameterError):
        _resolve_upscale_params(
            UpscaleRequest(image=data, target_width=799, target_height=800),
            _config(),
        )


def test_comfy_realesrgan_x4_200_to_800() -> None:
    config = _config(max_input_dimension=512, max_output_dimension=2048)
    client = MagicMock()
    client.upload_image = AsyncMock(return_value=ComfyUploadedImage(name="uploaded.png"))
    client.run_workflow = AsyncMock(
        return_value=[{"data": _png(800, 800), "filename": "upscaled.png"}]
    )
    backend = ComfyUpscaleBackend(config, client=client)
    backend._available_models = ["RealESRGAN_x4plus.pth"]

    result = asyncio.run(
        backend.upscale(
            UpscaleRequest(
                image=_png(200, 200),
                scale=4,
                target_width=800,
                target_height=800,
            )
        )
    )

    assert (result.width, result.height) == (800, 800)
    assert result.timings["total_ms"] >= 0
    assert result.timings["upscale_ms"] >= 0
    workflow = client.run_workflow.await_args.args[0]
    assert workflow["2"]["inputs"]["model_name"] == "RealESRGAN_x4plus.pth"
    client.upload_image.assert_awaited_once()
    client.run_workflow.assert_awaited_once()
