"""ComfyUI SAM2 + Grounding DINO segmentation backend."""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from io import BytesIO
from typing import Any

from comfy_client import ComfyUIClient
from comfy_segment_workflow import (
    build_grounding_detect_workflow,
    build_sam2_segment_workflow,
)
from comfy_workflow import _combo_options
from image_concurrency import comfy_gpu_slot
from image_generation_config import ImageGenerationConfig
from image_generation_errors import (
    ExecutionFailedError,
    ImageGenerationError,
    NotFoundError,
    SegmentationFailedError,
)
from image_reference import canonicalize_reference_png_for_comfy
from image_segmentation_types import (
    SegmentBox,
    SegmentPoint,
    SegmentRequest,
    validate_segment_request,
)
from image_segmentation_config import (
    ImageSegmentationConfig,
    load_image_segmentation_config,
)
from image_timings import elapsed_ms, merge_timings_ms
from image_upscale_config import comfy_generation_config_for_upscale, load_image_upscale_config

logger = logging.getLogger(__name__)
_DINO_TEXT_DETECTION_PATTERN = re.compile(
    r"score\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"bbox\s*=\s*\[([^\]]+)\]"
)


def _pixel_points(
    points: tuple[SegmentPoint, ...],
    width: int,
    height: int,
    *,
    label: str,
) -> list[dict[str, float]]:
    selected = [p for p in points if p.label == label]
    coords: list[dict[str, float]] = []
    for point in selected:
        coords.append(
            {
                "x": round(point.x * max(width - 1, 1), 2),
                "y": round(point.y * max(height - 1, 1), 2),
            },
        )
    return coords


def _image_size(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        image.load()
        return image.size


def _socket_type(inputs: dict[str, Any], name: str) -> str | None:
    value = inputs.get(name)
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        return value[0]
    return None


def _parse_best_bbox(
    node_output: dict[str, Any],
    width: int,
    height: int,
) -> tuple[SegmentBox, float | None] | None:
    """Parse the highest-confidence box when scores are present, else top-1 output."""
    candidates: list[tuple[float | None, SegmentBox]] = []
    raw_bboxes = node_output.get("bboxes") or node_output.get("BBOX")
    raw_scores = node_output.get("scores") or node_output.get("confidences")

    if isinstance(raw_bboxes, list):
        for index, item in enumerate(raw_bboxes):
            try:
                raw_score = (
                    raw_scores[index]
                    if isinstance(raw_scores, list) and index < len(raw_scores)
                    else item[4] if isinstance(item, (list, tuple)) and len(item) > 4
                    else None
                )
                score = float(raw_score) if raw_score is not None else None
                if score is not None and not math.isfinite(score):
                    score = 0.0
                if not isinstance(item, (list, tuple)) or len(item) < 4:
                    continue
                values = [float(value) for value in item[:4]]
                box = _normalize_pixel_bbox(
                    SegmentBox(x1=values[0], y1=values[1], x2=values[2], y2=values[3]),
                    width,
                    height,
                )
            except (TypeError, ValueError):
                continue
            candidates.append((score, box))

    raw_text = node_output.get("text") or node_output.get("detections")
    text_blocks = [raw_text] if isinstance(raw_text, str) else raw_text
    if isinstance(text_blocks, list):
        for text in text_blocks:
            if not isinstance(text, str):
                continue
            for match in _DINO_TEXT_DETECTION_PATTERN.finditer(text):
                try:
                    score = float(match.group(1))
                    coords = [float(value.strip()) for value in match.group(2).split(",")]
                    if len(coords) != 4 or not math.isfinite(score):
                        continue
                    box = _normalize_pixel_bbox(
                        SegmentBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                        width,
                        height,
                    )
                except (TypeError, ValueError):
                    continue
                candidates.append((score, box))

    if not candidates:
        return None
    scored = [candidate for candidate in candidates if candidate[0] is not None]
    score, box = max(scored, key=lambda candidate: candidate[0]) if scored else candidates[0]
    return box, score


def _normalize_pixel_bbox(box: SegmentBox, width: int, height: int) -> SegmentBox:
    """If bbox values look like pixels, convert to normalized."""
    if max(box.x2, box.y2) > 1.0:
        return SegmentBox(
            x1=box.x1 / max(width - 1, 1),
            y1=box.y1 / max(height - 1, 1),
            x2=box.x2 / max(width - 1, 1),
            y2=box.y2 / max(height - 1, 1),
        )
    return box


class ComfyImageSegmenter:
    def __init__(
        self,
        config: ImageSegmentationConfig | None = None,
        *,
        client: ComfyUIClient | None = None,
        generation_config: ImageGenerationConfig | None = None,
    ) -> None:
        self._config = config or load_image_segmentation_config()
        upscale_cfg = load_image_upscale_config()
        self._comfy_config = generation_config or comfy_generation_config_for_upscale(
            upscale_cfg,
        )
        self._client = client
        self._text_supported = False
        self._modes = {
            "text": False,
            "points": False,
            "box": False,
            "mask_refinement": False,
        }

    def _client_instance(self) -> ComfyUIClient:
        if self._client is not None:
            return self._client
        return ComfyUIClient(self._comfy_config)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def probe(self) -> bool:
        if not self._config.enabled:
            return False
        try:
            client = self._client_instance()
            object_info = await client.fetch_object_info(
                timeout=self._config.probe_timeout,
            )
        except ImageGenerationError as error:
            logger.warning("segmentation ComfyUI probe failed: %s", error.message)
            return False

        required_sam = {
            self._config.sam2_loader_class,
            self._config.sam2_segment_class,
            "LoadImage",
            "MaskToImage",
            "SaveImage",
        }
        if not required_sam <= set(object_info.keys()):
            logger.warning("segmentation probe missing SAM2 workflow nodes")
            return False

        sam_loader = object_info.get(self._config.sam2_loader_class, {})
        sam_required = (sam_loader.get("input") or {}).get("required") or {}
        for field, configured in (
            ("model", self._config.sam2_model),
            ("segmentor", self._config.sam2_segmentor),
            ("device", self._config.sam2_device),
            ("precision", self._config.sam2_precision),
        ):
            options = _combo_options(sam_required.get(field))
            if not options or configured not in options:
                logger.warning("segmentation probe: configured SAM2 %s is unavailable", field)
                return False

        sam_loader_outputs = set(sam_loader.get("output") or [])
        sam_segment = object_info.get(self._config.sam2_segment_class, {})
        segment_schema = sam_segment.get("input") or {}
        segment_required_schema = segment_schema.get("required") or {}
        segment_required = set(segment_required_schema)
        if (
            "SAM2MODEL" not in sam_loader_outputs
            or not {"sam2_model", "image", "keep_model_loaded"} <= segment_required
            or _socket_type(segment_required_schema, "sam2_model") != "SAM2MODEL"
            or _socket_type(segment_required_schema, "image") != "IMAGE"
        ):
            logger.warning("segmentation probe: SAM2 segment input schema is incomplete")
            return False
        segment_inputs = dict(segment_required_schema)
        segment_inputs.update(segment_schema.get("optional") or {})
        if "MASK" not in set(sam_segment.get("output") or []):
            logger.warning("segmentation probe: SAM2 node does not expose a mask output")
            return False
        points_ok = (
            _socket_type(segment_inputs, "coordinates_positive") == "STRING"
            and _socket_type(segment_inputs, "coordinates_negative") == "STRING"
        )
        sam_accepts_bbox = _socket_type(segment_inputs, "bboxes") == "BBOX"
        bbox_adapter = object_info.get(self._config.bbox_adapter_class)
        bbox_inputs = ((bbox_adapter or {}).get("input") or {}).get("required") or {}
        bbox_adapter_ok = (
            isinstance(bbox_adapter, dict)
            and _socket_type(bbox_inputs, "image") == "IMAGE"
            and all(_socket_type(bbox_inputs, name) == "FLOAT" for name in ("x1", "y1", "x2", "y2"))
            and "BBOX" in set(bbox_adapter.get("output") or [])
        )
        box_ok = sam_accepts_bbox and bbox_adapter_ok
        mask_ok = _socket_type(segment_inputs, "mask") == "MASK" and points_ok

        text_ok = False
        grounding_loader = object_info.get(self._config.grounding_loader_class)
        grounding_detect = object_info.get(self._config.grounding_detect_class)
        if isinstance(grounding_loader, dict) and isinstance(grounding_detect, dict):
            grounding_required = (grounding_loader.get("input") or {}).get("required") or {}
            detect_required = (grounding_detect.get("input") or {}).get("required") or {}
            detect_outputs = grounding_detect.get("output") or []
            model_options = _combo_options(grounding_required.get("model"))
            precision_options = _combo_options(grounding_required.get("precision"))
            device_options = _combo_options(grounding_required.get("device"))
            grounding_outputs = set(grounding_loader.get("output") or [])
            text_ok = (
                self._config.grounding_model_name in model_options
                and self._config.grounding_precision in precision_options
                and self._config.grounding_device in device_options
                and "GROUNDING_DINO_MODEL" in grounding_outputs
                and _socket_type(detect_required, "grounding_dino") == "GROUNDING_DINO_MODEL"
                and _socket_type(detect_required, "image") == "IMAGE"
                and _socket_type(detect_required, "text") == "STRING"
                and _socket_type(detect_required, "box_threshold") == "FLOAT"
                and _socket_type(detect_required, "text_threshold") == "FLOAT"
                and "BBOX" in detect_outputs
                and "PreviewImage" in object_info
                and "PreviewAny" in object_info
                and box_ok
            )

        self._text_supported = text_ok
        self._modes = {
            "text": text_ok,
            "points": points_ok,
            "box": box_ok,
            "mask_refinement": mask_ok,
        }
        return bool(points_ok or box_ok or mask_ok or text_ok)

    async def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "comfyui",
            "max_bytes": self._config.max_bytes,
            "max_dimension": self._config.max_dimension,
            "mask_semantics": "selected_white_1_not_selected_black_0",
            "modes": dict(self._modes),
        }

    async def segment(self, request: SegmentRequest) -> bytes:
        validate_segment_request(request)
        started = time.monotonic()
        width, height = _image_size(request.image)
        png = canonicalize_reference_png_for_comfy(request.image, index=0)
        request_id = str(uuid.uuid4())
        client = self._client_instance()

        box = request.box
        has_points = bool(request.points)
        has_spatial = has_points or box is not None
        if has_points and not self._modes["points"]:
            raise SegmentationFailedError("point segmentation is not available on this server")
        if box is not None and not self._modes["box"]:
            raise SegmentationFailedError("box segmentation is not available on this server")
        if request.mask is not None and not self._modes["mask_refinement"]:
            raise SegmentationFailedError("mask refinement is not available on this server")

        detect_ms: float | None = None
        bbox_coordinates = (
            (box.x1, box.y1, box.x2, box.y2) if box is not None else None
        )
        if request.prompt and request.prompt.strip() and not has_spatial:
            if not self._text_supported:
                raise SegmentationFailedError(
                    "text segmentation is not available on this server",
                )
            t0 = time.monotonic()
            async with comfy_gpu_slot():
                uploaded = await client.upload_image(
                    f"seg_{request_id}.png",
                    png,
                    "image/png",
                )
                workflow = build_grounding_detect_workflow(
                    request_id=request_id,
                    uploaded_filename=uploaded.name,
                    prompt=request.prompt.strip(),
                    config=self._config,
                )
                try:
                    prompt_id, history = await client.run_workflow_to_history(
                        workflow,
                        request_id=request_id,
                    )
                except ExecutionFailedError as error:
                    detail = str(error.details.get("detail") or "")
                    if detail.lower().startswith("grounding dino did not find"):
                        raise NotFoundError(
                            "no object matched the text prompt",
                            prompt=request.prompt.strip(),
                            threshold=self._config.grounding_threshold,
                        ) from error
                    raise
            detect_ms = elapsed_ms(t0)
            node_output = client.extract_node_outputs(history, prompt_id, "detect")
            if not node_output:
                node_output = client.consume_websocket_node_output(prompt_id, "detect")
            preview_output = client.extract_node_outputs(history, prompt_id, "preview_detections")
            if not preview_output:
                preview_output = client.consume_websocket_node_output(prompt_id, "preview_detections")
            if not node_output and preview_output:
                node_output = preview_output
            logger.info(
                "COMFY DINO HISTORY prompt_id=%s fields=%s preview_fields=%s preview=%s",
                prompt_id,
                sorted(node_output),
                sorted(preview_output),
                str(preview_output)[:500],
            )
            raw_bboxes = node_output.get("bboxes") or node_output.get("BBOX")
            raw_scores = node_output.get("scores") or node_output.get("confidences")
            logger.info(
                "COMFY DINO OUTPUT prompt_id=%s fields=%s bbox_type=%s bbox_count=%s score_count=%s",
                prompt_id,
                sorted(node_output),
                type(raw_bboxes).__name__,
                len(raw_bboxes) if isinstance(raw_bboxes, list) else 0,
                len(raw_scores) if isinstance(raw_scores, list) else 0,
            )
            parsed = _parse_best_bbox(node_output, width, height)
            if parsed is None:
                raise NotFoundError(
                    "no object matched the text prompt",
                    prompt=request.prompt.strip(),
                    threshold=self._config.grounding_threshold,
                )
            box, score = parsed
            if score is not None and score < self._config.grounding_threshold:
                raise NotFoundError(
                    "best object match is below the configured confidence threshold",
                    prompt=request.prompt.strip(),
                    score=score,
                    threshold=self._config.grounding_threshold,
                )
            bbox_coordinates = (box.x1, box.y1, box.x2, box.y2)

        if box is None and not request.points:
            raise NotFoundError("segmentation input did not resolve to a region")

        pos = _pixel_points(request.points, width, height, label="include")
        neg = _pixel_points(request.points, width, height, label="exclude")

        mask_name: str | None = None
        if request.mask is not None:
            mask_png = canonicalize_reference_png_for_comfy(request.mask, index=0)
            uploaded_mask = await client.upload_image(
                f"seg_mask_{request_id}.png",
                mask_png,
                "image/png",
            )
            mask_name = uploaded_mask.name

        t0 = time.monotonic()
        async with comfy_gpu_slot():
            uploaded = await client.upload_image(
                f"seg_{request_id}.png",
                png,
                "image/png",
            )
            workflow = build_sam2_segment_workflow(
                request_id=request_id,
                uploaded_filename=uploaded.name,
                config=self._config,
                coordinates_positive=pos or None,
                coordinates_negative=neg or None,
                bbox_coordinates=bbox_coordinates,
                bbox_adapter_class=self._config.bbox_adapter_class,
                mask_upload_name=mask_name,
            )
            outputs = await client.run_workflow(
                workflow,
                request_id=f"{request_id}-sam",
                timeout=self._config.segment_timeout,
            )
        segment_ms = elapsed_ms(t0)

        if not outputs:
            raise SegmentationFailedError("segmentation returned no mask image")

        mask_bytes = outputs[0]["data"]
        timings = merge_timings_ms(
            total=elapsed_ms(started),
            detect=detect_ms,
            segment=segment_ms,
        )
        self._last_timings = timings
        return mask_bytes

    @property
    def last_timings(self) -> dict[str, float]:
        return getattr(self, "_last_timings", {})


async def create_comfy_segmenter_if_ready() -> ComfyImageSegmenter | None:
    config = load_image_segmentation_config()
    if not config.enabled:
        return None
    segmenter = ComfyImageSegmenter(config)
    if await segmenter.probe():
        return segmenter
    await segmenter.aclose()
    return None
