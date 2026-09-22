"""Structured errors for image generation (MCP-friendly codes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImageGenerationError(Exception):
    """Base error with a stable machine-readable code."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class InvalidRequestError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("invalid_request", message, details)


class UnsupportedParameterError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("unsupported_parameter", message, details)


class TooManyImagesError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("too_many_images", message, details)


class UnsupportedMimeTypeError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("unsupported_image_mime", message, details)


class InputTooLargeError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("input_too_large", message, details)


class SafetyRejectedError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("safety_rejected", message, details)


class SafetyBackendError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("safety_backend_failure", message, details)


class BackendUnavailableError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("backend_unavailable", message, details)


class ComfyUIUnavailableError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("comfyui_unavailable", message, details)


class ComfyUIWorkflowRejectedError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("comfyui_workflow_rejected", message, details)


class ExecutionFailedError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("execution_failed", message, details)


class ImageGenerationTimeoutError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("timeout", message, details)


class CancelledError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("cancelled", message, details)


class OutputMissingError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("output_missing", message, details)


class OutputInvalidError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("output_invalid", message, details)


class InternalImageGenerationError(ImageGenerationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("internal_error", message, details)
