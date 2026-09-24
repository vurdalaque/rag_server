"""ComfyUI segmentation workflow and not_found behavior (mocked)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from comfy_client import ComfyUIClient
from comfy_segmentation import ComfyImageSegmenter
from image_generation_errors import NotFoundError
from image_segmentation_config import ImageSegmentationConfig
from image_segmentation_types import SegmentRequest


def _config() -> ImageSegmentationConfig:
    return ImageSegmentationConfig(
        enabled=True,
        probe_timeout=5.0,
        segment_timeout=30.0,
        max_bytes=1024 * 1024,
        max_dimension=4096,
        grounding_threshold=0.3,
        grounding_model_name="GroundingDINO: SwinT OGC",
        grounding_loader_class="GroundingModelLoader",
        grounding_detect_class="GroundingDetector",
        sam2_model="sam2_hiera_small.safetensors",
        sam2_segmentor="single_image",
        sam2_device="cuda",
        sam2_precision="bf16",
        sam2_loader_class="DownloadAndLoadSAM2Model",
        sam2_segment_class="Sam2Segmentation",
    )


def _png_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(128, 64, 32)).save(buf, format="PNG")
    return buf.getvalue()


def test_text_not_found_skips_sam2() -> None:
    client = MagicMock(spec=ComfyUIClient)
    client.upload_image = AsyncMock(return_value=MagicMock(name="in.png"))
    client.run_workflow_to_history = AsyncMock(
        return_value=("pid-detect", {"pid-detect": {"outputs": {"detect": {}}}}),
    )
    client.extract_node_outputs = MagicMock(return_value={})
    client.run_workflow = AsyncMock()

    segmenter = ComfyImageSegmenter(_config(), client=client)
    segmenter._text_supported = True
    segmenter._modes = {"text": True, "points": True, "box": True, "mask_refinement": True}

    with pytest.raises(NotFoundError):
        asyncio.run(
            segmenter.segment(
                SegmentRequest(image=_png_bytes(), prompt="person"),
            ),
        )

    client.run_workflow.assert_not_awaited()


def test_spatial_segment_runs_sam_without_detect() -> None:
    client = MagicMock(spec=ComfyUIClient)
    client.upload_image = AsyncMock(return_value=MagicMock(name="in.png"))
    client.run_workflow = AsyncMock(
        return_value=[{"data": _png_bytes(), "mime_type": "image/png"}],
    )
    client.run_workflow_to_history = AsyncMock()

    segmenter = ComfyImageSegmenter(_config(), client=client)
    segmenter._text_supported = True

    asyncio.run(
        segmenter.segment(
            SegmentRequest(
                image=_png_bytes(),
                box=__import__(
                    "image_segmentation_types",
                    fromlist=["SegmentBox"],
                ).SegmentBox(x1=0.2, y1=0.2, x2=0.8, y2=0.8),
            ),
        ),
    )

    client.run_workflow_to_history.assert_not_awaited()
    client.run_workflow.assert_awaited_once()
