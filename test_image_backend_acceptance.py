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
from image_generation_errors import ImageGenerationError, NotFoundError
from image_segmentation_types import SegmentBox, SegmentRequest
from image_upscale import create_upscale_backend_if_ready, UpscaleRequest
from test_image_contract import _png_b64

COMFY_URL = os.getenv("COMFYUI_URL", "").strip()


def _sample_png(size: int = 128) -> bytes:
    return base64.b64decode(_png_b64((size, size)))


def _schema_png(size: int = 512) -> bytes:
    image = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 190, 150, 310), radius=16, outline="black", width=5)
    draw.text((55, 245), "START", fill="black")
    draw.rectangle((190, 190, 320, 310), outline="black", width=5)
    draw.text((220, 245), "PROCESS", fill="black")
    draw.rounded_rectangle((360, 190, 490, 310), radius=16, outline="black", width=5)
    draw.text((400, 245), "END", fill="black")
    draw.line((150, 250, 190, 250), fill="black", width=5)
    draw.line((320, 250, 360, 250), fill="black", width=5)
    draw.polygon([(185, 243), (197, 250), (185, 257)], fill="black")
    draw.polygon([(355, 243), (367, 250), (355, 257)], fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _blank_png(size: int = 512) -> bytes:
    image = Image.new("RGB", (size, size), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _run_live_segment(coro):
    try:
        return asyncio.run(coro)
    except ImageGenerationError as error:
        print(
            "live Comfy segmentation error "
            f"code={error.code} details={error.details}"
        )
        raise


@pytest.mark.acceptance
@pytest.mark.skipif(not COMFY_URL, reason="COMFYUI_URL not set")
def test_live_comfy_segment_box() -> None:
    segmenter = asyncio.run(create_comfy_segmenter_if_ready())
    assert segmenter is not None
    if not asyncio.run(segmenter.capabilities())["modes"]["box"]:
        pytest.skip("BBoxFromCoordinates node is not installed in the running ComfyUI")

    started = time.monotonic()
    mask_bytes = _run_live_segment(
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
def test_live_comfy_text_selection_and_no_detection_error_mapping() -> None:
    segmenter = asyncio.run(create_comfy_segmenter_if_ready())
    assert segmenter is not None
    capabilities = asyncio.run(segmenter.capabilities())
    if not capabilities["modes"]["text"]:
        pytest.skip("DINO/SAM2 BBOX adapter is not installed in the running ComfyUI")
    schema_image = _schema_png()

    mask = _run_live_segment(
        segmenter.segment(SegmentRequest(image=schema_image, prompt="rectangle"))
    )
    with Image.open(BytesIO(mask)) as output_mask:
        assert output_mask.size == (512, 512)

    with pytest.raises(NotFoundError):
        asyncio.run(
            segmenter.segment(
                SegmentRequest(image=_blank_png(), prompt="person")
            )
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
