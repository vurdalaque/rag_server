"""ComfyUI segmentation workflow and not_found behavior (mocked)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from comfy_client import ComfyUIClient
from comfy_segmentation import ComfyImageSegmenter
from image_generation_errors import ExecutionFailedError, NotFoundError
from image_segmentation_config import ImageSegmentationConfig
from image_segmentation_types import SegmentRequest


def _config() -> ImageSegmentationConfig:
    return ImageSegmentationConfig(
        enabled=True,
        probe_timeout=5.0,
        segment_timeout=30.0,
        max_bytes=1024 * 1024,
        max_dimension=4096,
        grounding_threshold=0.30,
        grounding_text_threshold=0.25,
        grounding_model_name="tiny",
        grounding_precision="bf16",
        grounding_device="cuda",
        grounding_loader_class="GroundingDINOLoader",
        grounding_detect_class="GroundingDINODetect",
        bbox_adapter_class="BBoxFromCoordinates",
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


def test_segmentation_probe_reports_text_only_when_grounding_model_is_available() -> None:
    object_info = {
        "DownloadAndLoadSAM2Model": {
            "input": {
                "required": {
                    "model": [["sam2_hiera_small.safetensors"]],
                    "segmentor": [["single_image", "video"]],
                    "device": [["cuda", "cpu"]],
                    "precision": [["bf16", "fp16"]],
                }
            },
            "output": ["SAM2MODEL"],
        },
        "Sam2Segmentation": {
            "input": {
                "required": {
                    "sam2_model": ["SAM2MODEL"],
                    "image": ["IMAGE"],
                    "keep_model_loaded": ["BOOLEAN"],
                },
                "optional": {"coordinates_positive": ["STRING"], "coordinates_negative": ["STRING"], "bboxes": ["BBOX"], "mask": ["MASK"]},
            },
            "output": ["MASK"],
        },
        "LoadImage": {},
        "PreviewImage": {},
        "PreviewAny": {},
        "MaskToImage": {},
        "SaveImage": {},
        "GroundingDINOLoader": {
            "input": {
                "required": {
                    "model": [["tiny", "base"]],
                    "precision": [["bf16", "fp32"]],
                    "device": [["cuda", "cpu"]],
                }
            },
            "output": ["GROUNDING_DINO_MODEL"],
        },
        "GroundingDINODetect": {
            "input": {
                "required": {
                    "grounding_dino": ["GROUNDING_DINO_MODEL"],
                    "image": ["IMAGE"],
                    "text": ["STRING"],
                    "box_threshold": ["FLOAT"],
                    "text_threshold": ["FLOAT"],
                }
            },
            "output": ["BBOX", "STRING"],
        },
        "BBoxFromCoordinates": {
            "input": {
                "required": {
                    "image": ["IMAGE"],
                    "x1": ["FLOAT"],
                    "y1": ["FLOAT"],
                    "x2": ["FLOAT"],
                    "y2": ["FLOAT"],
                }
            },
            "output": ["BBOX"],
        },
    }
    client = MagicMock(spec=ComfyUIClient)
    client.fetch_object_info = AsyncMock(return_value=object_info)
    segmenter = ComfyImageSegmenter(_config(), client=client)

    assert asyncio.run(segmenter.probe()) is True
    assert segmenter._modes["text"] is True

    object_info["GroundingDINOLoader"]["input"]["required"]["model"] = [["base"]]
    segmenter = ComfyImageSegmenter(_config(), client=client)
    assert asyncio.run(segmenter.probe()) is True
    assert segmenter._modes["text"] is False
    assert segmenter._modes["points"] is True

    object_info["GroundingDINOLoader"]["input"]["required"]["model"] = [["tiny", "base"]]
    object_info.pop("BBoxFromCoordinates")
    segmenter = ComfyImageSegmenter(_config(), client=client)
    assert asyncio.run(segmenter.probe()) is True
    assert segmenter._modes["text"] is False
    assert segmenter._modes["box"] is False


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


def test_grounding_node_no_detection_error_maps_to_not_found() -> None:
    client = MagicMock(spec=ComfyUIClient)
    client.upload_image = AsyncMock(return_value=MagicMock(name="in.png"))
    client.run_workflow_to_history = AsyncMock(
        side_effect=ExecutionFailedError(
            "ComfyUI execution failed",
            prompt_id="pid-detect",
            detail="Grounding DINO did not find 'person' above thresholds",
        )
    )
    client.run_workflow = AsyncMock()
    segmenter = ComfyImageSegmenter(_config(), client=client)
    segmenter._text_supported = True

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(segmenter.segment(SegmentRequest(image=_schema_png_bytes(), prompt="person")))

    assert exc.value.code == "not_found"
    client.run_workflow.assert_not_awaited()


def test_text_detection_above_threshold_runs_sam2_after_dino() -> None:
    events = []
    client = MagicMock(spec=ComfyUIClient)
    client.upload_image = AsyncMock(return_value=MagicMock(name="in.png"))

    async def detect(_workflow, *, request_id):
        events.append("dino")
        return "pid-detect", {"pid-detect": {"outputs": {}}}

    def extract(_history, _prompt_id, _node_id):
        return {"bboxes": [[5, 5, 40, 40]]}

    async def segment(_workflow, *, request_id, timeout):
        events.append("sam2")
        return [{"data": _png_bytes(), "mime_type": "image/png"}]

    client.run_workflow_to_history = AsyncMock(side_effect=detect)
    client.extract_node_outputs = MagicMock(side_effect=extract)
    client.run_workflow = AsyncMock(side_effect=segment)
    segmenter = ComfyImageSegmenter(_config(), client=client)
    segmenter._text_supported = True
    segmenter._modes = {"text": True, "points": True, "box": True, "mask_refinement": True}

    mask = asyncio.run(segmenter.segment(SegmentRequest(image=_schema_png_bytes(), prompt="schema")))

    assert mask.startswith(b"\x89PNG\r\n\x1a\n")
    assert events == ["dino", "sam2"]
    dino_workflow = client.run_workflow_to_history.await_args.args[0]
    assert dino_workflow["grounding_model"]["class_type"] == "GroundingDINOLoader"
    assert dino_workflow["grounding_model"]["inputs"] == {
        "model": "tiny",
        "precision": "bf16",
        "device": "cuda",
    }
    assert dino_workflow["detect"]["class_type"] == "GroundingDINODetect"
    assert dino_workflow["detect"]["inputs"]["text"] == "schema"
    assert dino_workflow["detect"]["inputs"]["box_threshold"] == 0.3
    assert dino_workflow["detect"]["inputs"]["text_threshold"] == 0.25
    sam_workflow = client.run_workflow.await_args.args[0]
    assert "detect" not in sam_workflow
    assert sam_workflow["bbox_adapter"]["class_type"] == "BBoxFromCoordinates"
    assert sam_workflow["bbox_adapter"]["inputs"]["image"] == ["load", 0]
    assert sam_workflow["segment"]["inputs"]["bboxes"] == ["bbox_adapter", 0]
    client.run_workflow_to_history.assert_awaited_once()
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
    segmenter._modes = {"text": True, "points": True, "box": True, "mask_refinement": True}

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
    workflow = client.run_workflow.await_args.args[0]
    assert "detect" not in workflow
    assert workflow["bbox_adapter"]["class_type"] == "BBoxFromCoordinates"
    assert workflow["bbox_adapter"]["inputs"] == {
        "image": ["load", 0],
        "x1": 0.2,
        "y1": 0.2,
        "x2": 0.8,
        "y2": 0.8,
    }
    assert workflow["segment"]["inputs"]["bboxes"] == ["bbox_adapter", 0]
    assert "coordinates_positive" not in workflow["segment"]["inputs"]
