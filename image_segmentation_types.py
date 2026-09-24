"""Shared segmentation request types and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from image_generation_errors import InvalidSegmentationInputError

PointLabel = Literal["include", "exclude"]


@dataclass(frozen=True)
class SegmentPoint:
    x: float
    y: float
    label: PointLabel


@dataclass(frozen=True)
class SegmentBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class SegmentRequest:
    image: bytes
    prompt: str | None = None
    points: tuple[SegmentPoint, ...] = ()
    box: SegmentBox | None = None
    mask: bytes | None = None


def validate_segment_request(request: SegmentRequest) -> None:
    has_prompt = bool(request.prompt and request.prompt.strip())
    has_points = bool(request.points)
    has_box = request.box is not None
    has_mask = request.mask is not None

    if not (has_prompt or has_points or has_box):
        raise InvalidSegmentationInputError(
            "segmentation requires at least one of prompt, points, or box",
        )

    if has_mask and not has_points:
        raise InvalidSegmentationInputError(
            "existing mask is only supported together with points for refinement",
        )

    if has_mask and has_box:
        raise InvalidSegmentationInputError(
            "existing mask cannot be combined with box",
        )

    if has_box:
        _validate_box(request.box)

    for index, point in enumerate(request.points):
        _validate_point(point, index=index)


def _validate_point(point: SegmentPoint, *, index: int) -> None:
    if not (0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0):
        raise InvalidSegmentationInputError(
            "point coordinates must be normalized to [0, 1]",
            index=index,
            x=point.x,
            y=point.y,
        )
    if point.label not in ("include", "exclude"):
        raise InvalidSegmentationInputError(
            "point label must be include or exclude",
            index=index,
            label=point.label,
        )


def _validate_box(box: SegmentBox) -> None:
    coords = (box.x1, box.y1, box.x2, box.y2)
    if not all(0.0 <= value <= 1.0 for value in coords):
        raise InvalidSegmentationInputError(
            "box coordinates must be normalized to [0, 1]",
            box=coords,
        )
    if box.x1 > box.x2 or box.y1 > box.y2:
        raise InvalidSegmentationInputError(
            "box must satisfy x1 <= x2 and y1 <= y2",
            box=coords,
        )
