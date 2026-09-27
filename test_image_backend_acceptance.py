"""Live ComfyUI integration checks (run on Spark with COMFYUI_URL)."""

from __future__ import annotations

import asyncio
import base64
import os
import time
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from image_segmentation import create_comfy_segmenter_if_ready
from image_generation_errors import NotFoundError
from image_segmentation_types import SegmentBox, SegmentRequest
from image_upscale import create_upscale_backend_if_ready, UpscaleRequest
from test_image_contract import _png_b64

COMFY_URL = os.getenv("COMFYUI_URL", "").strip()


def _sample_png(size: int = 128) -> bytes:
    return base64.b64decode(_png_b64((size, size)))


def _schema_png(size: int = 512) -> bytes:
    image = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 60, 210, 180), outline="black", width=6)
    draw.rectangle((280, 300, 450, 430), outline="black", width=6)
    draw.line((210, 120, 280, 365), fill="black", width=5)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.acceptance
@pytest.mark.skipif(not COMFY_URL, reason="COMFYUI_URL not set")
def test_live_comfy_segment_box() -> None:
    segmenter = asyncio.run(create_comfy_segmenter_if_ready())
    assert segmenter is not None

    started = time.monotonic()
    mask_bytes = asyncio.run(
        segmenter.segment(
            SegmentRequest(
                image=_sample_png(128),
                box=SegmentBox(x1=0.25, y1=0.25, x2=0.75, y2=0.75),
            ),
        ),
    )
    elapsed = time.monotonic() - started

    with Image.open(BytesIO(mask_bytes)) as mask:
        assert mask.size == (128, 128)
    assert segmenter.last_timings["total_ms"] >= 0
    print(f"segment timings={getattr(segmenter, 'last_timings', {})} elapsed_s={elapsed:.2f}")


@pytest.mark.acceptance
@pytest.mark.skipif(not COMFY_URL, reason="COMFYUI_URL not set")
def test_live_comfy_text_schema_found_and_person_not_found() -> None:
    segmenter = asyncio.run(create_comfy_segmenter_if_ready())
    assert segmenter is not None
    capabilities = asyncio.run(segmenter.capabilities())
    assert capabilities["modes"]["text"] is True
    schema_image = _schema_png()

    mask = asyncio.run(
        segmenter.segment(SegmentRequest(image=schema_image, prompt="schema"))
    )
    with Image.open(BytesIO(mask)) as output_mask:
        assert output_mask.size == (512, 512)

    with pytest.raises(NotFoundError):
        asyncio.run(
            segmenter.segment(SegmentRequest(image=schema_image, prompt="person"))
        )


@pytest.mark.acceptance
@pytest.mark.skipif(not COMFY_URL, reason="COMFYUI_URL not set")
def test_live_comfy_upscale_x4() -> None:
    upscaler = asyncio.run(create_upscale_backend_if_ready())
    assert upscaler is not None

    source = _sample_png(200)
    started = time.monotonic()
    result = asyncio.run(
        upscaler.upscale(
            UpscaleRequest(
                image=source,
                scale=4.0,
                target_width=800,
                target_height=800,
            )
        )
    )
    elapsed = time.monotonic() - started

    assert result.width == 800
    assert result.height == 800
    assert result.timings["total_ms"] >= 0
    assert result.timings["upscale_ms"] >= 0
    print(f"upscale timings={result.timings} elapsed_s={elapsed:.2f}")
