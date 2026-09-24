"""Unit tests for local image segmentation (mocked LLM box detection)."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from image_generation_errors import InvalidSegmentationInputError
from image_segmentation import (
    SegmentBox,
    SegmentPoint,
    SegmentRequest,
    validate_segment_request,
)
from image_segmentation_config import ImageSegmentationConfig
from image_segmentation_local import LocalImageSegmenter
from test_image_contract import _png_b64


def _segmentation_config(**overrides: object) -> ImageSegmentationConfig:
    base: dict[str, object] = {
        "enabled": False,
        "probe_timeout": 5.0,
        "segment_timeout": 30.0,
        "max_bytes": 512 * 1024,
        "max_dimension": 4096,
        "grounding_threshold": 0.3,
        "grounding_model_name": "GroundingDINO: SwinT OGC",
        "grounding_loader_class": "GroundingModelLoader",
        "grounding_detect_class": "GroundingDetector",
        "sam2_model": "sam2_hiera_small.safetensors",
        "sam2_segmentor": "single_image",
        "sam2_device": "cuda",
        "sam2_precision": "bf16",
        "sam2_loader_class": "DownloadAndLoadSAM2Model",
        "sam2_segment_class": "Sam2Segmentation",
    }
    base.update(overrides)
    return ImageSegmentationConfig(**base)  # type: ignore[arg-type]


def test_validate_segment_request_requires_prompt_points_or_box() -> None:
    png = base64.b64decode(_png_b64())
    with pytest.raises(InvalidSegmentationInputError):
        validate_segment_request(SegmentRequest(image=png))


def test_box_segment_produces_white_region_without_llm() -> None:
    png = base64.b64decode(_png_b64((32, 32)))
    segmenter = LocalImageSegmenter(config=_segmentation_config())
    mask_bytes = asyncio.run(
        segmenter.segment(
            SegmentRequest(
                image=png,
                box=SegmentBox(x1=0.25, y1=0.25, x2=0.75, y2=0.75),
            ),
        ),
    )

    with Image.open(BytesIO(mask_bytes)) as mask:
        assert mask.mode == "L"
        assert mask.size == (32, 32)
        center = mask.getpixel((16, 16))
        corner = mask.getpixel((0, 0))
    assert center == 255
    assert corner == 0


def test_prompt_segment_uses_multimodal_chat_box() -> None:
    png = base64.b64decode(_png_b64((24, 24)))
    segmenter = LocalImageSegmenter(config=_segmentation_config())

    with patch(
        "image_segmentation_local.multimodal_chat",
        new_callable=AsyncMock,
        return_value='{"box": [0.0, 0.0, 0.5, 0.5]}',
    ) as chat:
        mask_bytes = asyncio.run(
            segmenter.segment(
                SegmentRequest(image=png, prompt="top-left quadrant"),
            ),
        )

    chat.assert_awaited_once()
    with Image.open(BytesIO(mask_bytes)) as mask:
        assert mask.getpixel((4, 4)) == 255
        assert mask.getpixel((20, 20)) == 0


def test_points_refine_existing_mask() -> None:
    png = base64.b64decode(_png_b64((20, 20)))
    blank_mask = Image.new("L", (20, 20), 0)
    buffer = BytesIO()
    blank_mask.save(buffer, format="PNG")
    seed_mask = buffer.getvalue()

    segmenter = LocalImageSegmenter(config=_segmentation_config())
    mask_bytes = asyncio.run(
        segmenter.segment(
            SegmentRequest(
                image=png,
                mask=seed_mask,
                points=(SegmentPoint(x=0.5, y=0.5, label="include"),),
            ),
        ),
    )

    with Image.open(BytesIO(mask_bytes)) as mask:
        assert mask.getpixel((10, 10)) == 255
