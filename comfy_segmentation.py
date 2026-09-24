"""ComfyUI SAM2 + Grounding DINO segmentation backend."""

from __future__ import annotations

import logging
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


def _pixel_bbox(box: SegmentBox, width: int, height: int) -> list[float]:
    return [
        round(min(box.x1, box.x2) * max(width - 1, 1), 2),
        round(min(box.y1, box.y2) * max(height - 1, 1), 2),
        round(max(box.x1, box.x2) * max(width - 1, 1), 2),
        round(max(box.y1, box.y2) * max(height - 1, 1), 2),
    ]


def _image_size(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        image.load()
        return image.size


def _parse_best_bbox(node_output: dict[str, Any]) -> SegmentBox | None:
    """Parse highest-confidence bbox from Comfy detect node output."""
    candidates: list[tuple[float, SegmentBox]] = []

    raw_bboxes = node_output.get("bboxes") or node_output.get("BBOX")
    raw_scores = node_output.get("scores") or node_output.get("confidences")

    if isinstance(raw_bboxes, list) and raw_bboxes:
        for index, item in enumerate(raw_bboxes):
            score = 1.0
            if isinstance(raw_scores, list) and index < len(raw_scores):
                try:
                    score = float(raw_scores[index])
                except (TypeError, ValueError):
                    score = 1.0
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                x1, y1, x2, y2 = (float(v) for v in item[:4])
                width_hint = max(x2, 1.0)
                height_hint = max(y2, 1.0)
                candidates.append(
                    (
                        score,
                        SegmentBox(
                            x1=x1 / width_hint if x1 > 1 else x1,
                            y1=y1 / height_hint if y1 > 1 else y1,
                            x2=x2 / width_hint if x2 > 1 else x2,
                            y2=y2 / height_hint if y2 > 1 else y2,
                        ),
                    ),
                )

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


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

        loader = object_info.get(self._config.sam2_loader_class, {})
        required_inputs = loader.get("input", {}).get("required", {})
        models = _combo_options(required_inputs.get("model"))
        if self._config.sam2_model not in models:
            logger.warning(
                "segmentation probe: SAM2 model %s not in Comfy combo",
                self._config.sam2_model,
            )
            return False

        text_ok = False
        if (
            self._config.grounding_loader_class in object_info
            and self._config.grounding_detect_class in object_info
        ):
            text_ok = True

        self._text_supported = text_ok
        self._modes = {
            "text": text_ok,
            "points": True,
            "box": True,
            "mask_refinement": True,
        }
        return True

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
        has_spatial = bool(request.points) or box is not None

        detect_ms: float | None = None
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
                prompt_id, history = await client.run_workflow_to_history(
                    workflow,
                    request_id=request_id,
                )
            detect_ms = elapsed_ms(t0)
            node_output = client.extract_node_outputs(history, prompt_id, "detect")
            parsed = _parse_best_bbox(node_output)
            if parsed is None:
                raise NotFoundError(
                    "no object matched the text prompt",
                    prompt=request.prompt.strip(),
                )
            box = _normalize_pixel_bbox(parsed, width, height)

        if box is None and not request.points:
            raise NotFoundError("segmentation input did not resolve to a region")

        pos = _pixel_points(request.points, width, height, label="include")
        neg = _pixel_points(request.points, width, height, label="exclude")
        bboxes = [_pixel_bbox(box, width, height)] if box is not None else None

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
                bboxes=bboxes,
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
