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


def _schema_png_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 220, 180), outline="black", width=6)
    draw.rectangle((280, 280, 430, 390), outline="black", width=6)
    draw.line((220, 130, 280, 330), fill="black", width=5)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_segmentation_probe_reports_text_only_when_grounding_model_is_available() -> None:
    object_info = {
        "DownloadAndLoadSAM2Model": {
            "input": {"required": {"model": [["sam2_hiera_small.safetensors"]]}}
        },
        "Sam2Segmentation": {},
        "LoadImage": {},
        "MaskToImage": {},
        "SaveImage": {},
        "GroundingModelLoader": {
            "input": {"required": {"model_name": [["GroundingDINO: SwinT OGC"]]}}
        },
        "GroundingDetector": {},
    }
    client = MagicMock(spec=ComfyUIClient)
    client.fetch_object_info = AsyncMock(return_value=object_info)
    segmenter = ComfyImageSegmenter(_config(), client=client)

    assert asyncio.run(segmenter.probe()) is True
    assert segmenter._modes["text"] is True

    object_info["GroundingModelLoader"]["input"]["required"]["model_name"] = [["other-model"]]
    segmenter = ComfyImageSegmenter(_config(), client=client)
    assert asyncio.run(segmenter.probe()) is True
    assert segmenter._modes["text"] is False
    assert segmenter._modes["points"] is True


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


def test_text_score_below_threshold_skips_sam2() -> None:
    client = MagicMock(spec=ComfyUIClient)
    client.upload_image = AsyncMock(return_value=MagicMock(name="in.png"))
    client.run_workflow_to_history = AsyncMock(
        return_value=("pid-detect", {"pid-detect": {"outputs": {}}}),
    )
    client.extract_node_outputs = MagicMock(
        return_value={"bboxes": [[5, 5, 40, 40]], "scores": [0.29]},
    )
    client.run_workflow = AsyncMock()

    segmenter = ComfyImageSegmenter(_config(), client=client)
    segmenter._text_supported = True

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(segmenter.segment(SegmentRequest(image=_schema_png_bytes(), prompt="person")))

    assert exc.value.code == "not_found"
    assert exc.value.details["score"] == 0.29
    client.run_workflow.assert_not_awaited()


def test_text_detection_above_threshold_runs_sam2_after_dino() -> None:
    events = []
    client = MagicMock(spec=ComfyUIClient)
    client.upload_image = AsyncMock(return_value=MagicMock(name="in.png"))

    async def detect(_workflow, *, request_id):
        events.append("dino")
        return "pid-detect", {"pid-detect": {"outputs": {}}}

    def extract(_history, _prompt_id, _node_id):
        return {"bboxes": [[5, 5, 40, 40]], "scores": [0.9]}

    async def segment(_workflow, *, request_id, timeout):
        events.append("sam2")
        return [{"data": _png_bytes(), "mime_type": "image/png"}]

    client.run_workflow_to_history = AsyncMock(side_effect=detect)
    client.extract_node_outputs = MagicMock(side_effect=extract)
    client.run_workflow = AsyncMock(side_effect=segment)
    segmenter = ComfyImageSegmenter(_config(), client=client)
    segmenter._text_supported = True

    mask = asyncio.run(segmenter.segment(SegmentRequest(image=_schema_png_bytes(), prompt="schema")))

    assert mask.startswith(b"\x89PNG\r\n\x1a\n")
    assert events == ["dino", "sam2"]
    assert segmenter.last_timings["total_ms"] >= 0
    assert segmenter.last_timings["detect_ms"] >= 0
    assert segmenter.last_timings["segment_ms"] >= 0


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
