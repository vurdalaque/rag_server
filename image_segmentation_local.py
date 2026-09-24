"""PIL-based segmenter for unit tests only — not used in production."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
from io import BytesIO

import httpx
from PIL import Image, ImageDraw

from image_generation_errors import (
    InputTooLargeError,
    InvalidSegmentationInputError,
    SegmentationFailedError,
)
from image_segmentation_types import (
    SegmentBox,
    SegmentPoint,
    SegmentRequest,
    validate_segment_request,
)
from image_segmentation_types import _validate_box
from image_segmentation_config import (
    ImageSegmentationConfig,
    load_image_segmentation_config,
)
from llm_client import SAFETY_LLM_TIMEOUT, multimodal_chat

logger = logging.getLogger(__name__)

_SEGMENT_JSON_RE = re.compile(
    r"^```(?:json)?\s*(.*?)```\s*$",
    re.DOTALL | re.IGNORECASE,
)

_SEGMENT_SYSTEM_PROMPT = """You locate objects in images for segmentation.
Given an image and a short text description, respond with JSON only (no markdown):
{"box": [x1, y1, x2, y2]}
Coordinates are normalized floats in [0, 1] with origin at the top-left."""


def _guess_image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return "image/webp"
    guessed, _ = mimetypes.guess_type("segment.bin")
    return guessed or "image/png"


def _open_source_image(
    data: bytes,
    *,
    max_bytes: int,
    max_dimension: int,
) -> Image.Image:
    if len(data) > max_bytes:
        raise InputTooLargeError(
            "segmentation source image exceeds size limit",
            size_bytes=len(data),
            max_bytes=max_bytes,
        )
    image = Image.open(BytesIO(data))
    image.load()
    width, height = image.size
    if width > max_dimension or height > max_dimension:
        raise InputTooLargeError(
            "source image dimensions exceed limit",
            width=width,
            height=height,
            max_dimension=max_dimension,
        )
    return image


def _load_mask_image(data: bytes, *, size: tuple[int, int], max_bytes: int) -> Image.Image:
    if len(data) > max_bytes:
        raise InputTooLargeError("segmentation mask exceeds size limit", size_bytes=len(data))
    mask = Image.open(BytesIO(data))
    mask.load()
    if mask.size != size:
        raise InvalidSegmentationInputError("existing mask dimensions must match source image")
    return mask.convert("L")


def _norm_scalar_to_px(value: float, extent: int) -> int:
    if extent <= 1:
        return 0
    return max(0, min(extent - 1, int(round(value * (extent - 1)))))


def _apply_box(mask: Image.Image, box: SegmentBox) -> None:
    width, height = mask.size
    left = _norm_scalar_to_px(min(box.x1, box.x2), width)
    right = _norm_scalar_to_px(max(box.x1, box.x2), width)
    top = _norm_scalar_to_px(min(box.y1, box.y2), height)
    bottom = _norm_scalar_to_px(max(box.y1, box.y2), height)
    ImageDraw.Draw(mask).rectangle([left, top, right, bottom], fill=255)


def _apply_points(mask: Image.Image, points: tuple[SegmentPoint, ...]) -> None:
    width, height = mask.size
    radius = max(1, int(round(min((width, height)) * 0.02)))
    draw = ImageDraw.Draw(mask)
    for point in points:
        cx = _norm_scalar_to_px(point.x, width)
        cy = _norm_scalar_to_px(point.y, height)
        fill = 255 if point.label == "include" else 0
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill)


def _parse_segment_box_json(raw: str) -> SegmentBox:
    text = raw.strip()
    fence = _SEGMENT_JSON_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    parsed = json.loads(text)
    box_raw = parsed.get("box")
    x1, y1, x2, y2 = (float(v) for v in box_raw)
    box = SegmentBox(x1=x1, y1=y1, x2=x2, y2=y2)
    _validate_box(box)
    return box


def _encode_mask_png(mask: Image.Image) -> bytes:
    buffer = BytesIO()
    mask.convert("L").save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


class LocalImageSegmenter:
    """Test-only PIL segmenter — do not register in production."""

    def __init__(self, config: ImageSegmentationConfig | None = None) -> None:
        self._config = config or load_image_segmentation_config()

    async def probe(self) -> bool:
        return False

    async def segment(
        self,
        request: SegmentRequest,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> bytes:
        validate_segment_request(request)
        with _open_source_image(
            request.image,
            max_bytes=self._config.max_bytes,
            max_dimension=self._config.max_dimension,
        ) as source:
            width, height = source.size
            if request.mask is not None:
                mask = _load_mask_image(
                    request.mask,
                    size=(width, height),
                    max_bytes=self._config.max_bytes,
                )
            else:
                mask = Image.new("L", (width, height), 0)

            has_points = bool(request.points)
            has_box = request.box is not None
            has_prompt = bool(request.prompt and request.prompt.strip())

            if has_prompt and not has_box and not has_points:
                box = await self._prompt_to_box(request.image, request.prompt.strip(), client=client)
                _apply_box(mask, box)
            elif has_box:
                _apply_box(mask, request.box)

            if has_points:
                _apply_points(mask, request.points)

            return _encode_mask_png(mask)

    async def _prompt_to_box(
        self,
        image: bytes,
        prompt: str,
        *,
        client: httpx.AsyncClient | None,
    ) -> SegmentBox:
        mime_type = _guess_image_mime(image)
        encoded = base64.b64encode(image).decode("ascii")
        messages = [
            {"role": "system", "content": _SEGMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Select the region described by:\n{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    },
                ],
            },
        ]
        try:
            raw = await multimodal_chat(
                messages,
                thinking=False,
                response_format={"type": "json_object"},
                timeout=SAFETY_LLM_TIMEOUT,
                client=client,
            )
            return _parse_segment_box_json(raw)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as error:
            raise SegmentationFailedError(
                "could not derive a region from the text prompt",
                reason=str(error),
            ) from error
