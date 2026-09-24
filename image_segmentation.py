"""Segmentation types and ComfyUI-backed segmenter factory."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from comfy_segmentation import ComfyImageSegmenter, create_comfy_segmenter_if_ready
from image_segmentation_config import (
    ImageSegmentationConfig,
    load_image_segmentation_config,
)

PointLabel = Literal["include", "exclude"]

# Re-export dataclasses and validation from dedicated module to avoid cycles.
from image_segmentation_types import (  # noqa: E402
    SegmentBox,
    SegmentPoint,
    SegmentRequest,
    validate_segment_request,
)

__all__ = [
    "ComfyImageSegmenter",
    "ImageSegmenter",
    "PointLabel",
    "SegmentBox",
    "SegmentPoint",
    "SegmentRequest",
    "create_comfy_segmenter_if_ready",
    "load_image_segmentation_config",
    "validate_segment_request",
]


@runtime_checkable
class ImageSegmenter(Protocol):
    async def probe(self) -> bool: ...

    async def segment(self, request: SegmentRequest) -> bytes: ...

    async def capabilities(self) -> dict: ...

    @property
    def last_timings(self) -> dict: ...
