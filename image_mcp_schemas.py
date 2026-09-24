"""Structured MCP output models for image-generation tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ImageGenerationDefaultsOutput(BaseModel):
    resolution: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str


class CapabilityInputLimits(BaseModel):
    supported: bool
    max_count: int
    max_bytes: int | None = None
    semantics: str | None = None


class CapabilityInputsOutput(BaseModel):
    reference_images: CapabilityInputLimits
    mask: CapabilityInputLimits
    sketch: CapabilityInputLimits


class ResolutionLimitsOutput(BaseModel):
    min: int
    max: int


class ImageGenerationCapabilitiesOutput(BaseModel):
    backend: str
    max_images: int
    max_output_bytes: int
    inputs: CapabilityInputsOutput
    resolution: ResolutionLimitsOutput
    defaults: ImageGenerationDefaultsOutput
    safety_validation_enabled: bool
    samplers: dict[str, list[str]] = Field(default_factory=dict)
    # Deprecated compatibility aliases (remove after MCP acceptance).
    max_reference_images: int
    max_reference_bytes: int
    safety_enabled: bool


GENERATE_IMAGE_TOOL_DESCRIPTION = (
    "Generate one or more images from a text prompt with optional visual inputs: "
    "reference images (visual references / WHAT), sketch (composition and layout / HOW), "
    "and mask (soft spatial edit-region guidance / WHERE). "
    "Returns MCP image content blocks; metadata is in structured output (seed, timings)."
)

_REFERENCE_IMAGES_FIELD_DESCRIPTION = (
    "Optional PNG reference images as base64 strings (same encoding as MCP ImageContent.data). "
    "Visual references for subject, style, or content (WHAT)."
)
_MASK_FIELD_DESCRIPTION = (
    "Optional single PNG mask as base64. Soft spatial edit-region guidance (WHERE); "
    "not hard pixel locking outside the marked region."
)
_SKETCH_FIELD_DESCRIPTION = (
    "Optional single PNG sketch as base64. Composition, layout, shape, and color guidance (HOW); "
    "spatial guidance rather than strict style copying."
)


class GenerateImageInput(BaseModel):
    """Canonical MCP input schema for ``generate_image`` (flat JSON object)."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(description="Requested image operation or scene description.")
    negative_prompt: str | None = Field(
        default=None,
        description="Optional negative prompt.",
    )
    width: int | None = Field(
        default=None,
        description="Optional output width in pixels (square generation; must equal height if both set).",
    )
    height: int | None = Field(
        default=None,
        description="Optional output height in pixels (square generation; must equal width if both set).",
    )
    steps: int | None = Field(default=None, description="Optional sampler steps.")
    seed: int | None = Field(default=None, description="Optional random seed.")
    cfg: float | None = Field(default=None, description="Optional CFG scale.")
    sampler: str | None = Field(default=None, description="Optional sampler name.")
    scheduler: str | None = Field(default=None, description="Optional scheduler name.")
    image_count: int = Field(
        default=1,
        description="Number of images to return (1..server max).",
    )
    reference_images: list[str] | None = Field(
        default=None,
        description=_REFERENCE_IMAGES_FIELD_DESCRIPTION,
    )
    mask: str | None = Field(
        default=None,
        description=_MASK_FIELD_DESCRIPTION,
    )
    sketch: str | None = Field(
        default=None,
        description=_SKETCH_FIELD_DESCRIPTION,
    )


def generate_image_input_json_schema() -> dict[str, Any]:
    """Published ``tools/list`` inputSchema for ``generate_image``."""
    return GenerateImageInput.model_json_schema()


class GenerateImageStructuredOutput(BaseModel):
    """Metadata for a successful generate_image call (images stay in content blocks)."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    prompt_id: str | None = None
    image_count: int
    timings: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_generation(
        cls,
        *,
        seed: int | None,
        prompt_id: str | None,
        image_count: int,
        timings: dict[str, float] | None,
    ) -> GenerateImageStructuredOutput:
        return cls(
            seed=seed,
            prompt_id=prompt_id,
            image_count=image_count,
            timings=dict(timings or {}),
        )


def capabilities_from_backend(payload: dict[str, Any]) -> ImageGenerationCapabilitiesOutput:
    defaults_raw = payload.get("defaults") or {}
    samplers_raw = payload.get("samplers") or {}
    inputs_raw = payload.get("inputs") or {}
    resolution_raw = payload.get("resolution") or {}
    samplers: dict[str, list[str]] = {}

    for key, value in samplers_raw.items():
        if isinstance(value, list):
            samplers[str(key)] = [str(item) for item in value]

    def _input_limits(name: str) -> CapabilityInputLimits:
        block = inputs_raw.get(name) or {}
        if not isinstance(block, dict):
            block = {}
        return CapabilityInputLimits(
            supported=bool(block.get("supported", False)),
            max_count=int(block.get("max_count", 0)),
            max_bytes=(
                int(block["max_bytes"])
                if block.get("max_bytes") is not None
                else None
            ),
            semantics=(
                str(block["semantics"]) if block.get("semantics") is not None else None
            ),
        )

    safety_validation = bool(
        payload.get(
            "safety_validation_enabled",
            payload.get("safety_enabled", False),
        ),
    )

    return ImageGenerationCapabilitiesOutput(
        backend=str(payload.get("backend", "")),
        max_images=int(payload["max_images"]),
        max_output_bytes=int(payload["max_output_bytes"]),
        inputs=CapabilityInputsOutput(
            reference_images=_input_limits("reference_images"),
            mask=_input_limits("mask"),
            sketch=_input_limits("sketch"),
        ),
        resolution=ResolutionLimitsOutput(
            min=int(resolution_raw.get("min", 0)),
            max=int(resolution_raw.get("max", 0)),
        ),
        defaults=ImageGenerationDefaultsOutput(
            resolution=int(defaults_raw.get("resolution", 0)),
            steps=int(defaults_raw.get("steps", 0)),
            cfg=float(defaults_raw.get("cfg", 0)),
            sampler=str(defaults_raw.get("sampler", "")),
            scheduler=str(defaults_raw.get("scheduler", "")),
        ),
        safety_validation_enabled=safety_validation,
        samplers=samplers,
        max_reference_images=int(
            payload.get(
                "max_reference_images",
                (_input_limits("reference_images").max_count),
            ),
        ),
        max_reference_bytes=int(
            payload.get(
                "max_reference_bytes",
                (_input_limits("reference_images").max_bytes or 0),
            ),
        ),
        safety_enabled=safety_validation,
    )
