"""MCP tool registration for analyze / segment / upscale."""

from __future__ import annotations

import base64
import logging
from typing import Annotated, Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import Field

from image_analysis import LlmImageAnalyzer
from image_generation_errors import (
    ImageGenerationError,
    InvalidMaskImageError,
    InvalidReferenceImageError,
    InvalidRequestError,
    InvalidSegmentationInputError,
)
from image_mcp_backends import ImageMcpBackends
from image_mcp_schemas import (
    ANALYZE_IMAGE_TOOL_DESCRIPTION,
    SEGMENT_IMAGE_TOOL_DESCRIPTION,
    UPSCALE_IMAGE_TOOL_DESCRIPTION,
    AnalyzeImageStructuredOutput,
    SegmentImageStructuredOutput,
    UpscaleImageStructuredOutput,
)
from image_reference import (
    decode_and_validate_mask_image,
    decode_and_validate_reference_image,
)
from image_segmentation import SegmentBox, SegmentPoint, SegmentRequest
from image_upscale import UpscaleRequest
from rag_metrics import track_mcp_tool

logger = logging.getLogger(__name__)

ErrorResultFn = Callable[[ImageGenerationError], CallToolResult]
CoerceImageFn = Callable[[Any, str], str]


def _parse_xy_points(
    raw: list[dict[str, Any]] | None,
    *,
    label: str,
    field: str,
) -> list[SegmentPoint]:
    if not raw:
        return []
    points: list[SegmentPoint] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InvalidRequestError(f"{field} must be objects", index=index)
        try:
            x = float(item["x"])
            y = float(item["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidRequestError(
                f"each {field} entry requires numeric x and y",
                index=index,
            ) from error
        points.append(SegmentPoint(x=x, y=y, label=label))
    return points


def _parse_segment_points(
    *,
    points: list[dict[str, Any]] | None = None,
    positive_points: list[dict[str, Any]] | None = None,
    negative_points: list[dict[str, Any]] | None = None,
) -> tuple[SegmentPoint, ...]:
    merged: list[SegmentPoint] = []
    if points:
        for index, item in enumerate(points):
            if not isinstance(item, dict):
                raise InvalidRequestError("points must be objects", index=index)
            label = str(item.get("label", "include")).strip().lower()
            try:
                x = float(item["x"])
                y = float(item["y"])
            except (KeyError, TypeError, ValueError) as error:
                raise InvalidRequestError(
                    "each point requires numeric x and y",
                    index=index,
                ) from error
            merged.append(SegmentPoint(x=x, y=y, label=label))
    merged.extend(_parse_xy_points(positive_points, label="include", field="positive_points"))
    merged.extend(_parse_xy_points(negative_points, label="exclude", field="negative_points"))
    return tuple(merged)


def _parse_segment_box(raw: dict[str, float] | None) -> SegmentBox | None:
    if raw is None:
        return None
    try:
        return SegmentBox(
            x1=float(raw["x1"]),
            y1=float(raw["y1"]),
            x2=float(raw["x2"]),
            y2=float(raw["y2"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidRequestError("box requires x1,y1,x2,y2") from error


def register_image_ops_tools(
    mcp: MCPServer,
    backends: ImageMcpBackends,
    *,
    error_result: ErrorResultFn,
    coerce_image: CoerceImageFn,
    registered_names: list[str],
) -> None:
    analyzer = backends.analyzer
    segmenter = backends.segmenter
    upscaler = backends.upscaler

    if analyzer is not None:

        @track_mcp_tool("analyze_image")
        async def analyze_image(
            images: Annotated[
                list[str],
                Field(
                    description=(
                        "One or more images as base64 PNG/JPEG/WebP "
                        "(same encoding as generate_image reference_images)."
                    ),
                ),
            ],
            instruction: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Optional analysis instruction or question.",
                ),
            ] = None,
        ) -> Annotated[CallToolResult, AnalyzeImageStructuredOutput]:
            decoded: list[bytes] = []
            mime_types: list[str] = []
            try:
                for index, encoded in enumerate(images):
                    payload = coerce_image(encoded, field="images")
                    blob = decode_and_validate_reference_image(payload, index=index)
                    decoded.append(blob)
                    mime_types.append("image/png")
            except (
                InvalidReferenceImageError,
                InvalidRequestError,
            ) as error:
                return error_result(error)

            try:
                result = await analyzer.analyze(
                    tuple(decoded),
                    tuple(mime_types),
                    instruction,
                )
            except ImageGenerationError as error:
                return error_result(error)

            structured = AnalyzeImageStructuredOutput(
                image_count=len(decoded),
                timings=result.timings,
            )
            return CallToolResult(
                content=[TextContent(type="text", text=result.text)],
                structured_content=structured.model_dump(mode="json"),
            )

        mcp.add_tool(analyze_image, description=ANALYZE_IMAGE_TOOL_DESCRIPTION)
        registered_names.append("analyze_image")

    if segmenter is not None:

        @track_mcp_tool("segment_image")
        async def segment_image(
            image: Annotated[str, Field(description="Source image as base64 PNG.")],
            prompt: Annotated[
                str | None,
                Field(default=None, description="Semantic region description."),
            ] = None,
            points: Annotated[
                list[dict[str, Any]] | None,
                Field(
                    default=None,
                    description="Deprecated: normalized points with label include/exclude.",
                ),
            ] = None,
            positive_points: Annotated[
                list[dict[str, Any]] | None,
                Field(default=None, description="Normalized include click points."),
            ] = None,
            negative_points: Annotated[
                list[dict[str, Any]] | None,
                Field(default=None, description="Normalized exclude click points."),
            ] = None,
            box: Annotated[
                dict[str, float] | None,
                Field(default=None, description="Normalized bounding box."),
            ] = None,
            mask: Annotated[
                str | None,
                Field(default=None, description="Optional mask PNG for point refinement."),
            ] = None,
        ) -> Annotated[CallToolResult, SegmentImageStructuredOutput]:
            try:
                image_bytes = decode_and_validate_reference_image(
                    coerce_image(image, field="image"),
                    index=0,
                )
                mask_bytes = None
                if mask is not None:
                    mask_bytes = decode_and_validate_mask_image(
                        coerce_image(mask, field="mask"),
                    )
                segment_points = _parse_segment_points(
                    points=points,
                    positive_points=positive_points,
                    negative_points=negative_points,
                )
                segment_box = _parse_segment_box(box)
                request = SegmentRequest(
                    image=image_bytes,
                    prompt=prompt,
                    points=segment_points,
                    box=segment_box,
                    mask=mask_bytes,
                )
            except (
                InvalidReferenceImageError,
                InvalidMaskImageError,
                InvalidRequestError,
                InvalidSegmentationInputError,
            ) as error:
                return error_result(error)

            try:
                mask_png = await segmenter.segment(request)
            except ImageGenerationError as error:
                return error_result(error)

            from PIL import Image
            from io import BytesIO

            with Image.open(BytesIO(image_bytes)) as source:
                width, height = source.size

            timings = getattr(segmenter, "last_timings", {}) or {}
            structured = SegmentImageStructuredOutput(
                width=width,
                height=height,
                timings=timings,
            )
            return CallToolResult(
                content=[
                    ImageContent(
                        type="image",
                        data=base64.b64encode(mask_png).decode("ascii"),
                        mimeType="image/png",
                    ),
                ],
                structured_content=structured.model_dump(mode="json"),
            )

        mcp.add_tool(segment_image, description=SEGMENT_IMAGE_TOOL_DESCRIPTION)
        registered_names.append("segment_image")

    if upscaler is not None:

        @track_mcp_tool("upscale_image")
        async def upscale_image(
            image: Annotated[str, Field(description="Source image as base64.")],
            scale: Annotated[
                float | None,
                Field(default=None, description="Upscale factor."),
            ] = None,
        ) -> Annotated[CallToolResult, UpscaleImageStructuredOutput]:
            try:
                image_bytes = decode_and_validate_reference_image(
                    coerce_image(image, field="image"),
                    index=0,
                )
                request = UpscaleRequest(
                    image=image_bytes,
                    scale=scale,
                )
            except (InvalidReferenceImageError, InvalidRequestError) as error:
                return error_result(error)

            try:
                result = await upscaler.upscale(request)
            except ImageGenerationError as error:
                return error_result(error)

            mime = "image/png"
            if result.image.startswith(b"\xff\xd8\xff"):
                mime = "image/jpeg"
            structured = UpscaleImageStructuredOutput(
                width=result.width,
                height=result.height,
                timings=result.timings,
            )
            return CallToolResult(
                content=[
                    ImageContent(
                        type="image",
                        data=base64.b64encode(result.image).decode("ascii"),
                        mimeType=mime,
                    ),
                ],
                structured_content=structured.model_dump(mode="json"),
            )

        mcp.add_tool(upscale_image, description=UPSCALE_IMAGE_TOOL_DESCRIPTION)
        registered_names.append("upscale_image")
