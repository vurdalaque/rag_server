"""Conditional MCP tools for image generation."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Annotated, Any

from comfy_progress import reset_wait_progress, set_wait_progress
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field
from mcp.types import CallToolResult, ImageContent, TextContent

from image_generation import GenerateImageRequest
from image_mcp_backends import ImageMcpBackends
from image_mcp_capabilities import build_platform_capabilities_payload
from image_mcp_ops import register_image_ops_tools
from image_generation_errors import (
    ImageGenerationError,
    InternalImageGenerationError,
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
    "analyze_image",
    "segment_image",
    "upscale_image",
)

GENERATION_MCP_TOOL_NAMES: tuple[str, ...] = (
    "generate_image",
    "image_generation_capabilities",
)

IMAGE_OPS_TOOL_NAMES: tuple[str, ...] = (
    "analyze_image",
    "segment_image",
    "upscale_image",
)

_backends: ImageMcpBackends | None = None
_registered_tool_names: list[str] = []
_registered = False


def image_tools_registered() -> bool:
    return _registered


def get_image_backend():
    if _backends is None:
        return None
    return _backends.generation


def get_image_mcp_backends() -> ImageMcpBackends | None:
    return _backends


def list_mcp_tool_names(
    *,
    include_image_tools: bool | None = None,
) -> list[str]:
    if include_image_tools is None:
        include_image_tools = _registered

    names = list(CORE_MCP_TOOL_NAMES)

    if include_image_tools:
        if _registered_tool_names:
            names.extend(_registered_tool_names)
        elif _registered:
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


def register_image_mcp_backends(
    mcp: MCPServer,
    backends: ImageMcpBackends,
) -> None:
    """Register all available image MCP tools from ``backends``."""
    register_image_tools(mcp, backends.generation, backends=backends)


def register_image_tools(
    mcp: MCPServer,
    backend,
    *,
    backends: ImageMcpBackends | None = None,
) -> None:
    global _backends, _registered, _registered_tool_names

    if backends is None:
        backends = ImageMcpBackends(generation=backend)
    elif backend is not None and backends.generation is None:
        backends = ImageMcpBackends(
            generation=backend,
            analyzer=backends.analyzer,
            segmenter=backends.segmenter,
            upscaler=backends.upscaler,
        )

    _backends = backends
    _registered_tool_names = []
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
        *,
        ctx: Context,
    ) -> Annotated[CallToolResult, GenerateImageStructuredOutput]:
        """Registered via ``add_tool``; see ``GENERATE_IMAGE_TOOL_DESCRIPTION``."""
        if _backends is None or _backends.generation is None:
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

        async def _comfy_wait_progress(elapsed: float, state: str) -> None:
            try:
                await ctx.report_progress(
                    min(99.0, elapsed),
                    600.0,
                    f"comfyui:{state}",
                )
            except Exception:
                logger.debug("MCP progress notification failed", exc_info=True)

        progress_token = set_wait_progress(_comfy_wait_progress)
        try:
            try:
                result = await _backends.generation.generate(request)
            except asyncio.CancelledError:
                logger.warning(
                    "MCP generate_image cancelled (client likely disconnected)",
                )
                raise
            except ImageGenerationError as error:
                logger.warning(
                    "MCP generate_image failed code=%s message=%s details=%s",
                    public_error_code(error),
                    error.message,
                    error.details or {},
                )
                return _error_tool_result(error)
            except Exception as error:
                logger.exception("MCP generate_image unexpected error")
                return _error_tool_result(
                    InternalImageGenerationError(
                        "image generation failed unexpectedly",
                        reason=str(error),
                    ),
                )
        finally:
            reset_wait_progress(progress_token)

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
        logger.info(
            "MCP TOOL FINISHED generate_image prompt_id=%s",
            result.prompt_id,
        )
        return tool_result

    @track_mcp_tool("image_generation_capabilities")
    async def image_generation_capabilities() -> (
        Annotated[CallToolResult, ImageGenerationCapabilitiesOutput]
    ):
        """
        Report server image-generation limits, defaults, and sampler metadata.
        """
        if _backends is None or not _backends.any_available():
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": {
                                    "code": "backend_unavailable",
                                    "message": (
                                        "image capabilities are not available "
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

        payload = await build_platform_capabilities_payload(_backends)
        structured = capabilities_from_backend(payload)
        return CallToolResult(
            content=[],
            structured_content=structured.model_dump(mode="json"),
        )

    if backends.generation is not None:
        mcp.add_tool(
            generate_image,
            description=GENERATE_IMAGE_TOOL_DESCRIPTION,
        )
        _registered_tool_names.append("generate_image")

    register_image_ops_tools(
        mcp,
        backends,
        error_result=_error_tool_result,
        coerce_image=_coerce_image_base64_payload,
        registered_names=_registered_tool_names,
    )

    if backends.any_available():
        mcp.add_tool(image_generation_capabilities)
        _registered_tool_names.append("image_generation_capabilities")

    _registered = bool(_registered_tool_names)
    logger.info(
        "Registered MCP image tools: %s",
        ", ".join(_registered_tool_names),
    )


def reset_image_mcp_registration(mcp: MCPServer | None = None) -> None:
    """Test helper: clear module registration state and remove tools from ``mcp``."""
    global _backends, _registered, _registered_tool_names
    if mcp is not None:
        _remove_image_tools_from_server(mcp)
    _backends = None
    _registered_tool_names = []
    _registered = False


def published_generate_image_input_schema() -> dict[str, Any]:
    """Input schema exposed on ``tools/list`` for ``generate_image``."""
    return generate_image_input_json_schema()
