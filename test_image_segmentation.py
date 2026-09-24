"""Unit tests for segment_image request validation."""

from __future__ import annotations

import base64

import pytest

from image_generation_errors import InvalidSegmentationInputError
from image_segmentation import (
    SegmentBox,
    SegmentPoint,
    SegmentRequest,
    validate_segment_request,
)
from test_image_contract import _png_b64


def test_validate_segment_request_requires_prompt_points_or_box() -> None:
    png = base64.b64decode(_png_b64())
    with pytest.raises(InvalidSegmentationInputError):
        validate_segment_request(SegmentRequest(image=png))


def test_validate_mask_requires_points() -> None:
    png = base64.b64decode(_png_b64((16, 16)))
    with pytest.raises(InvalidSegmentationInputError):
        validate_segment_request(
            SegmentRequest(image=png, mask=png),
        )


def test_validate_mask_cannot_combine_with_box() -> None:
    png = base64.b64decode(_png_b64((16, 16)))
    with pytest.raises(InvalidSegmentationInputError):
        validate_segment_request(
            SegmentRequest(
                image=png,
                mask=png,
                box=SegmentBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9),
                points=(SegmentPoint(x=0.5, y=0.5, label="include"),),
            ),
        )


def test_validate_normalized_box() -> None:
    png = base64.b64decode(_png_b64())
    validate_segment_request(
        SegmentRequest(
            image=png,
            box=SegmentBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
        ),
    )
