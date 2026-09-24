"""Backends wired into Radius MCP image tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from image_analysis import LlmImageAnalyzer
from image_generation import ImageGenerationBackend
from image_segmentation import ImageSegmenter
from image_upscale import ImageUpscaler


@dataclass(frozen=True)
class ImageMcpBackends:
    generation: ImageGenerationBackend | None = None
    analyzer: LlmImageAnalyzer | None = None
    segmenter: ImageSegmenter | None = None
    upscaler: ImageUpscaler | None = None

    def any_available(self) -> bool:
        return any(
            (
                self.generation is not None,
                self.analyzer is not None,
                self.segmenter is not None,
                self.upscaler is not None,
            ),
        )
