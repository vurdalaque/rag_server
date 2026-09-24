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


class AnalysisCapabilityOutput(BaseModel):
    supported: bool
    max_images: int
    max_bytes: int


class SegmentationModesOutput(BaseModel):
    text: bool
    points: bool
    box: bool
    mask_refinement: bool


class SegmentationCapabilityOutput(BaseModel):
    supported: bool
    max_bytes: int
    max_dimension: int
    mask_semantics: str
    modes: SegmentationModesOutput


class UpscaleCapabilityOutput(BaseModel):
    supported: bool
    max_input_bytes: int = 0
    max_output_bytes: int = 0
    max_input_dimension: int = 0
    max_output_dimension: int = 0
    supports_scale_factor: bool = False
    supports_target_dimensions: bool = False
    max_scale: float | None = None
    min_scale: float | None = None
    default_scale: float | None = None


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
    analysis: AnalysisCapabilityOutput
    segmentation: SegmentationCapabilityOutput
    upscale: UpscaleCapabilityOutput


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


ANALYZE_IMAGE_TOOL_DESCRIPTION = (
    "Read-only multimodal image analysis using the configured vision-language model. "
    "Supply one or more images and an optional instruction (e.g. describe, compare before/after)."
)

_SEGMENT_POINT_DESCRIPTION = (
    "Normalized point prompt with x and y in [0,1] and label include (select) or exclude."
)

_SEGMENT_BOX_DESCRIPTION = (
    "Normalized bounding box with x1,y1,x2,y2 in [0,1] (top-left to bottom-right)."
)


class AnalyzeImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images: list[str] = Field(
        description="One or more PNG/JPEG/WebP images as base64 (same encoding as reference_images).",
    )
    instruction: str | None = Field(
        default=None,
        description="Optional natural-language analysis instruction or question.",
    )


class SegmentPointInput(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    label: str = Field(description="include or exclude")


class SegmentImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str = Field(description="Source image as base64 PNG.")
    prompt: str | None = Field(
        default=None,
        description="Optional semantic description of the region to select (AI Select by text).",
    )
    points: list[SegmentPointInput] | None = Field(
        default=None,
        description="Deprecated: use positive_points / negative_points.",
    )
    positive_points: list[dict[str, float]] | None = Field(
        default=None,
        description="Optional normalized include click points (x, y in 0..1).",
    )
    negative_points: list[dict[str, float]] | None = Field(
        default=None,
        description="Optional normalized exclude click points (x, y in 0..1).",
    )
    box: dict[str, float] | None = Field(
        default=None,
        description=_SEGMENT_BOX_DESCRIPTION,
    )
    mask: str | None = Field(
        default=None,
        description=(
            "Optional existing mask PNG (base64) for refinement with points; "
            "selected=white, not selected=black."
        ),
    )


class UpscaleImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str = Field(description="Source image as base64.")
    scale: float | None = Field(
        default=None,
        description="Upscale factor (MVP supports 4). Defaults to server capability.",
    )


UPSCALE_IMAGE_TOOL_DESCRIPTION = (
    "Neural super-resolution upscale (not ordinary geometric resize). "
    "MVP supports scale=4 via the configured upscale model."
)

SEGMENT_IMAGE_TOOL_DESCRIPTION = (
    "Produce a pixel selection mask for the source image (AI Select). "
    "Combine text prompt, points, box, and optional mask refinement. "
    "Output mask: white=selected, black=not selected."
)


class AnalyzeImageStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_count: int
    timings: dict[str, float] = Field(default_factory=dict)


class SegmentImageStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    timings: dict[str, float] = Field(default_factory=dict)


class UpscaleImageStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    timings: dict[str, float] = Field(default_factory=dict)


def analyze_image_input_json_schema() -> dict[str, Any]:
    return AnalyzeImageInput.model_json_schema()


def segment_image_input_json_schema() -> dict[str, Any]:
    return SegmentImageInput.model_json_schema()


def upscale_image_input_json_schema() -> dict[str, Any]:
    return UpscaleImageInput.model_json_schema()


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

    analysis_raw = payload.get("analysis") or {}
    segment_raw = payload.get("segmentation") or {}
    upscale_raw = payload.get("upscale") or {}
    modes_raw = segment_raw.get("modes") or {}

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
        analysis=AnalysisCapabilityOutput(
            supported=bool(analysis_raw.get("supported", False)),
            max_images=int(analysis_raw.get("max_images", 0)),
            max_bytes=int(analysis_raw.get("max_bytes", 0)),
        ),
        segmentation=SegmentationCapabilityOutput(
            supported=bool(segment_raw.get("supported", False)),
            max_bytes=int(segment_raw.get("max_bytes", 0)),
            max_dimension=int(segment_raw.get("max_dimension", 0)),
            mask_semantics=str(
                segment_raw.get("mask_semantics", "selected_white_1_not_selected_black_0"),
            ),
            modes=SegmentationModesOutput(
                text=bool(modes_raw.get("text", False)),
                points=bool(modes_raw.get("points", False)),
                box=bool(modes_raw.get("box", False)),
                mask_refinement=bool(modes_raw.get("mask_refinement", False)),
            ),
        ),
        upscale=UpscaleCapabilityOutput(
            supported=bool(upscale_raw.get("supported", False)),
            max_input_bytes=int(upscale_raw.get("max_input_bytes", 0)),
            max_output_bytes=int(upscale_raw.get("max_output_bytes", 0)),
            max_input_dimension=int(upscale_raw.get("max_input_dimension", 0)),
            max_output_dimension=int(upscale_raw.get("max_output_dimension", 0)),
            supports_scale_factor=bool(upscale_raw.get("supports_scale_factor", False)),
            supports_target_dimensions=bool(
                upscale_raw.get("supports_target_dimensions", False),
            ),
            max_scale=(
                float(upscale_raw["max_scale"])
                if upscale_raw.get("max_scale") is not None
                else None
            ),
            min_scale=(
                float(upscale_raw["min_scale"])
                if upscale_raw.get("min_scale") is not None
                else None
            ),
            default_scale=(
                float(upscale_raw["default_scale"])
                if upscale_raw.get("default_scale") is not None
                else None
            ),
        ),
    )
