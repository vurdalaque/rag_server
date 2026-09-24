"""Conditional MCP tools for image generation."""

from __future__ import annotations

import base64
import json
import logging
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field
from mcp.types import CallToolResult, ImageContent, TextContent

from image_generation import GenerateImageRequest, ImageGenerationBackend
from image_generation_errors import (
    ImageGenerationError,
    InvalidMaskImageError,
    InvalidReferenceImageError,
    InvalidRequestError,
    InvalidSketchImageError,
    public_error_code,
)
from image_reference import (
    decode_and_validate_mask_image,
    decode_and_validate_reference_image,
    decode_and_validate_sketch_image,
)
from image_mcp_schemas import (
    GENERATE_IMAGE_TOOL_DESCRIPTION,
    GenerateImageInput,
    GenerateImageStructuredOutput,
    ImageGenerationCapabilitiesOutput,
    _MASK_FIELD_DESCRIPTION,
    _REFERENCE_IMAGES_FIELD_DESCRIPTION,
    _SKETCH_FIELD_DESCRIPTION,
    capabilities_from_backend,
    generate_image_input_json_schema,
)
from rag_metrics import track_mcp_tool

logger = logging.getLogger(__name__)

CORE_MCP_TOOL_NAMES: tuple[str, ...] = (
    "search_project",
    "web_search",
    "ask_project",
    "ping",
)

IMAGE_MCP_TOOL_NAMES: tuple[str, ...] = (
    "generate_image",
    "image_generation_capabilities",
)

_backend: ImageGenerationBackend | None = None
_registered = False


def image_tools_registered() -> bool:
    return _registered


def get_image_backend() -> ImageGenerationBackend | None:
    return _backend


def list_mcp_tool_names(
    *,
    include_image_tools: bool | None = None,
) -> list[str]:
    if include_image_tools is None:
        include_image_tools = _registered

    names = list(CORE_MCP_TOOL_NAMES)

    if include_image_tools:
        names.extend(IMAGE_MCP_TOOL_NAMES)

    return names


def _coerce_image_base64_payload(value: Any, *, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        data = value.get("data")
        if isinstance(data, str) and data.strip():
            return data
    raise InvalidRequestError(
        f"{field} must be a base64-encoded image",
        field=field,
    )


def _error_tool_result(error: ImageGenerationError) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": {
                            "code": public_error_code(error),
                            "message": error.message,
                            "details": error.details or None,
                        }
                    },
                    ensure_ascii=False,
                ),
            )
        ],
        is_error=True,
    )


def list_ping_tool_names() -> list[str]:
    """Tool names advertised by ping (excludes ping itself)."""
    return [name for name in list_mcp_tool_names() if name != "ping"]


def _remove_image_tools_from_server(mcp: MCPServer) -> None:
    for name in IMAGE_MCP_TOOL_NAMES:
        try:
            mcp.remove_tool(name)
        except ToolError:
            pass


def register_image_tools(
    mcp: MCPServer,
    backend: ImageGenerationBackend,
) -> None:
    global _backend, _registered

    _backend = backend
    _remove_image_tools_from_server(mcp)

    @track_mcp_tool("generate_image")
    async def generate_image(
        prompt: Annotated[str, Field(description=GenerateImageInput.model_fields["prompt"].description)],
        negative_prompt: Annotated[
            str | None,
            Field(description=GenerateImageInput.model_fields["negative_prompt"].description),
        ] = None,
        width: Annotated[
            int | None,
            Field(description=GenerateImageInput.model_fields["width"].description),
        ] = None,
        height: Annotated[
            int | None,
            Field(description=GenerateImageInput.model_fields["height"].description),
        ] = None,
        steps: Annotated[
            int | None,
            Field(description=GenerateImageInput.model_fields["steps"].description),
        ] = None,
        seed: Annotated[
            int | None,
            Field(description=GenerateImageInput.model_fields["seed"].description),
        ] = None,
        cfg: Annotated[
            float | None,
            Field(description=GenerateImageInput.model_fields["cfg"].description),
        ] = None,
        sampler: Annotated[
            str | None,
            Field(description=GenerateImageInput.model_fields["sampler"].description),
        ] = None,
        scheduler: Annotated[
            str | None,
            Field(description=GenerateImageInput.model_fields["scheduler"].description),
        ] = None,
        image_count: Annotated[
            int,
            Field(description=GenerateImageInput.model_fields["image_count"].description),
        ] = 1,
        reference_images: Annotated[
            list[str] | None,
            Field(description=_REFERENCE_IMAGES_FIELD_DESCRIPTION),
        ] = None,
        mask: Annotated[
            str | None,
            Field(description=_MASK_FIELD_DESCRIPTION),
        ] = None,
        sketch: Annotated[
            str | None,
            Field(description=_SKETCH_FIELD_DESCRIPTION),
        ] = None,
    ) -> Annotated[CallToolResult, GenerateImageStructuredOutput]:
        """Registered via ``add_tool``; see ``GENERATE_IMAGE_TOOL_DESCRIPTION``."""
        if _backend is None:
            logger.warning("generate_image rejected: image backend not registered")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": {
                                    "code": "backend_unavailable",
                                    "message": (
                                        "image generation is not available "
                                        "on this server"
                                    ),
                                }
                            },
                            ensure_ascii=False,
                        ),
                    )
                ],
                is_error=True,
            )

        ref_count = len(reference_images or [])
        logger.info(
            "MCP generate_image start prompt_chars=%s reference_images=%s "
            "has_mask=%s has_sketch=%s image_count=%s",
            len(prompt),
            ref_count,
            mask is not None,
            sketch is not None,
            image_count,
        )

        decoded_refs: list[bytes] = []
        decoded_mask: bytes | None = None
        decoded_sketch: bytes | None = None

        try:
            if reference_images:
                for index, encoded in enumerate(reference_images):
                    payload = _coerce_image_base64_payload(
                        encoded,
                        field="reference_images",
                    )
                    decoded_refs.append(
                        decode_and_validate_reference_image(payload, index=index),
                    )

            if mask is not None:
                decoded_mask = decode_and_validate_mask_image(
                    _coerce_image_base64_payload(mask, field="mask"),
                )

            if sketch is not None:
                decoded_sketch = decode_and_validate_sketch_image(
                    _coerce_image_base64_payload(sketch, field="sketch"),
                )
        except (
            InvalidReferenceImageError,
            InvalidMaskImageError,
            InvalidSketchImageError,
            InvalidRequestError,
        ) as error:
            return _error_tool_result(error)

        request = GenerateImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            image_count=image_count,
            reference_images=tuple(decoded_refs),
            mask=decoded_mask,
            sketch=decoded_sketch,
        )

        try:
            result = await _backend.generate(request)
        except ImageGenerationError as error:
            logger.warning(
                "MCP generate_image failed code=%s message=%s details=%s",
                public_error_code(error),
                error.message,
                error.details or {},
            )
            return _error_tool_result(error)

        content: list[ImageContent] = []

        for image in result.images:
            content.append(
                ImageContent(
                    type="image",
                    data=base64.b64encode(image.data).decode("ascii"),
                    mimeType=image.mime_type,
                )
            )

        structured = GenerateImageStructuredOutput.from_generation(
            seed=result.seed,
            prompt_id=result.prompt_id,
            image_count=len(result.images),
            timings=result.timings,
        )

        tool_result = CallToolResult(
            content=content,
            structured_content=structured.model_dump(mode="json"),
        )
        logger.info(
            "MCP RETURN generate_image prompt_id=%s artifacts=%s seed=%s",
            result.prompt_id,
            len(content),
            structured.seed,
        )
        return tool_result

    @track_mcp_tool("image_generation_capabilities")
    async def image_generation_capabilities() -> (
        Annotated[CallToolResult, ImageGenerationCapabilitiesOutput]
    ):
        """
        Report server image-generation limits, defaults, and sampler metadata.
        """
        if _backend is None:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": {
                                    "code": "backend_unavailable",
                                    "message": (
                                        "image generation is not available "
                                        "on this server"
                                    ),
                                }
                            },
                            ensure_ascii=False,
                        ),
                    )
                ],
                is_error=True,
            )

        payload = await _backend.capabilities()
        structured = capabilities_from_backend(payload)
        return CallToolResult(
            content=[],
            structured_content=structured.model_dump(mode="json"),
        )

    mcp.add_tool(
        generate_image,
        description=GENERATE_IMAGE_TOOL_DESCRIPTION,
    )
    mcp.add_tool(image_generation_capabilities)
    _registered = True
    logger.info("Registered MCP image tools: %s", ", ".join(IMAGE_MCP_TOOL_NAMES))


def reset_image_mcp_registration(mcp: MCPServer | None = None) -> None:
    """Test helper: clear module registration state and remove tools from ``mcp``."""
    global _backend, _registered
    if mcp is not None:
        _remove_image_tools_from_server(mcp)
    _backend = None
    _registered = False


def published_generate_image_input_schema() -> dict[str, Any]:
    """Input schema exposed on ``tools/list`` for ``generate_image``."""
    return generate_image_input_json_schema()
