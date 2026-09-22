"""Conditional MCP tools for image generation."""

from __future__ import annotations

import base64
import json
import logging
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent

from image_generation import GenerateImageRequest, ImageGenerationBackend
from image_generation_errors import ImageGenerationError
from image_mcp_schemas import (
    GenerateImageStructuredOutput,
    ImageGenerationCapabilitiesOutput,
    capabilities_from_backend,
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


def list_ping_tool_names() -> list[str]:
    """Tool names advertised by ping (excludes ping itself)."""
    return [name for name in list_mcp_tool_names() if name != "ping"]


def register_image_tools(
    mcp: MCPServer,
    backend: ImageGenerationBackend,
) -> None:
    global _backend, _registered

    if _registered:
        logger.debug("image MCP tools already registered")
        return

    _backend = backend

    @track_mcp_tool("generate_image")
    async def generate_image(
        prompt: str,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        cfg: float | None = None,
        sampler: str | None = None,
        scheduler: str | None = None,
        image_count: int = 1,
        reference_images: list[str] | None = None,
    ) -> Annotated[CallToolResult, GenerateImageStructuredOutput]:
        """
        Generate one or more images from a text prompt (and optional reference images).

        Returns MCP image content blocks; metadata is in structured output (seed, timings).
        Reference images must be base64-encoded blobs.
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

        decoded_refs: list[bytes] = []

        if reference_images:
            for index, encoded in enumerate(reference_images):
                try:
                    decoded_refs.append(base64.b64decode(encoded, validate=True))
                except Exception as error:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=json.dumps(
                                    {
                                        "error": {
                                            "code": "invalid_reference_images",
                                            "message": (
                                                f"reference_images[{index}] "
                                                "is not valid base64"
                                            ),
                                            "details": {"reason": str(error)},
                                        }
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                        ],
                        is_error=True,
                    )

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
        )

        try:
            result = await _backend.generate(request)
        except ImageGenerationError as error:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": {
                                    "code": error.code,
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

        return CallToolResult(
            content=content,
            structured_content=structured.model_dump(mode="json"),
        )

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

    mcp.add_tool(generate_image)
    mcp.add_tool(image_generation_capabilities)
    _registered = True
    logger.info("Registered MCP image tools: %s", ", ".join(IMAGE_MCP_TOOL_NAMES))


def reset_image_mcp_registration() -> None:
    """Test helper: clear module registration state."""
    global _backend, _registered
    _backend = None
    _registered = False
