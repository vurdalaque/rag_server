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
