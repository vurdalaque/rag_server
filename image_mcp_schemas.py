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


class ImageGenerationCapabilitiesOutput(BaseModel):
    backend: str
    max_images: int
    max_output_bytes: int
    max_reference_images: int
    max_reference_bytes: int
    defaults: ImageGenerationDefaultsOutput
    safety_enabled: bool
    samplers: dict[str, list[str]] = Field(default_factory=dict)


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
    samplers: dict[str, list[str]] = {}

    for key, value in samplers_raw.items():
        if isinstance(value, list):
            samplers[str(key)] = [str(item) for item in value]

    return ImageGenerationCapabilitiesOutput(
        backend=str(payload.get("backend", "")),
        max_images=int(payload["max_images"]),
        max_output_bytes=int(payload["max_output_bytes"]),
        max_reference_images=int(payload["max_reference_images"]),
        max_reference_bytes=int(payload["max_reference_bytes"]),
        defaults=ImageGenerationDefaultsOutput(
            resolution=int(defaults_raw.get("resolution", 0)),
            steps=int(defaults_raw.get("steps", 0)),
            cfg=float(defaults_raw.get("cfg", 0)),
            sampler=str(defaults_raw.get("sampler", "")),
            scheduler=str(defaults_raw.get("scheduler", "")),
        ),
        safety_enabled=bool(payload.get("safety_enabled", False)),
        samplers=samplers,
    )
