"""Decode and validate MCP image payloads (references, sketch, mask)."""

from __future__ import annotations

import base64
import hashlib
import re
from io import BytesIO
from typing import Literal, Type

from image_generation_errors import (
    ImageGenerationError,
    InvalidMaskImageError,
    InvalidReferenceImageError,
    InvalidSketchImageError,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DATA_URL_RE = re.compile(r"^data:[^;]+;base64,(.+)$", re.DOTALL | re.IGNORECASE)

ImageInputRole = Literal["reference", "mask", "sketch"]

_ROLE_ERROR_CLASS: dict[ImageInputRole, Type[ImageGenerationError]] = {
    "reference": InvalidReferenceImageError,
    "mask": InvalidMaskImageError,
    "sketch": InvalidSketchImageError,
}


def _role_error(
    role: ImageInputRole,
    message: str,
    **details: object,
) -> ImageGenerationError:
    error_class = _ROLE_ERROR_CLASS[role]
    return error_class(message, **details)


def normalize_reference_base64(encoded: str) -> str:
    """Strip data-URL prefix and whitespace from a base64 image payload."""
    text = encoded.strip()
    if not text:
        raise InvalidReferenceImageError(
            "reference image payload is empty",
            index=None,
        )
    match = _DATA_URL_RE.match(text)
    if match:
        text = match.group(1)
    return "".join(text.split())


def normalize_image_base64(encoded: str, *, role: ImageInputRole) -> str:
    text = encoded.strip()
    if not text:
        raise _role_error(role, f"{role} image payload is empty")
    match = _DATA_URL_RE.match(text)
    if match:
        text = match.group(1)
    return "".join(text.split())


def decode_image_base64(
    encoded: str,
    *,
    role: ImageInputRole,
    index: int | None = None,
) -> bytes:
    try:
        normalized = normalize_image_base64(encoded, role=role)
        data = base64.b64decode(normalized, validate=True)
    except ImageGenerationError:
        raise
    except Exception as error:
        details: dict[str, object] = {"reason": str(error)}
        if index is not None:
            details["index"] = index
        raise _role_error(
            role,
            f"{role} image is not valid base64",
            **details,
        ) from error

    if not data:
        details = {}
        if index is not None:
            details["index"] = index
        raise _role_error(role, f"{role} image decoded to empty bytes", **details)
    return data


def decode_reference_image_base64(encoded: str, *, index: int) -> bytes:
    return decode_image_base64(encoded, role="reference", index=index)


def validate_image_bytes(
    data: bytes,
    *,
    role: ImageInputRole,
    index: int | None = None,
) -> None:
    if len(data) < len(PNG_SIGNATURE) or not data.startswith(PNG_SIGNATURE):
        details: dict[str, object] = {"size_bytes": len(data)}
        if index is not None:
            details["index"] = index
        raise _role_error(
            role,
            f"{role} image must be a PNG (missing PNG signature)",
            **details,
        )

    try:
        from PIL import Image
    except ImportError as error:
        details = {}
        if index is not None:
            details["index"] = index
        raise _role_error(
            role,
            f"server cannot validate {role} images (Pillow not installed)",
            **details,
        ) from error

    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            width, height = image.size
    except OSError as error:
        details = {
            "reason": str(error),
            "size_bytes": len(data),
            "sha256_prefix": reference_sha256_prefix(data),
        }
        if index is not None:
            details["index"] = index
        raise _role_error(
            role,
            f"{role} image is corrupt or truncated",
            **details,
        ) from error
    except Exception as error:
        details = {"reason": str(error), "size_bytes": len(data)}
        if index is not None:
            details["index"] = index
        raise _role_error(
            role,
            f"{role} image could not be decoded",
            **details,
        ) from error

    if width < 1 or height < 1:
        details = {"width": width, "height": height}
        if index is not None:
            details["index"] = index
        raise _role_error(role, f"{role} image has invalid dimensions", **details)


def validate_reference_image_bytes(data: bytes, *, index: int) -> None:
    validate_image_bytes(data, role="reference", index=index)


def reference_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reference_sha256_prefix(data: bytes, hex_chars: int = 16) -> str:
    return reference_sha256(data)[:hex_chars]


def canonicalize_png_for_comfy(
    data: bytes,
    *,
    role: ImageInputRole,
    index: int | None = None,
) -> bytes:
    validate_image_bytes(data, role=role, index=index)

    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            converted = image.convert("RGBA")
        elif image.mode != "RGB":
            converted = image.convert("RGB")
        else:
            converted = image.copy()

        buffer = BytesIO()
        converted.save(buffer, format="PNG", compress_level=6)
        out = buffer.getvalue()

    validate_image_bytes(out, role=role, index=index)
    return out


def canonicalize_reference_png_for_comfy(data: bytes, *, index: int) -> bytes:
    return canonicalize_png_for_comfy(data, role="reference", index=index)


def decode_and_validate_reference_image(encoded: str, *, index: int) -> bytes:
    data = decode_reference_image_base64(encoded, index=index)
    return canonicalize_reference_png_for_comfy(data, index=index)


def decode_and_validate_mask_image(encoded: str) -> bytes:
    data = decode_image_base64(encoded, role="mask")
    return canonicalize_png_for_comfy(data, role="mask")


def decode_and_validate_sketch_image(encoded: str) -> bytes:
    data = decode_image_base64(encoded, role="sketch")
    return canonicalize_png_for_comfy(data, role="sketch")
