"""Decode and validate MCP reference image payloads."""

from __future__ import annotations

import base64
import hashlib
import re
from io import BytesIO

from image_generation_errors import InvalidReferenceImageError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DATA_URL_RE = re.compile(r"^data:[^;]+;base64,(.+)$", re.DOTALL | re.IGNORECASE)


def normalize_reference_base64(encoded: str) -> str:
    """Strip data-URL prefix and whitespace from a base64 reference payload."""
    text = encoded.strip()
    if not text:
        raise InvalidReferenceImageError(
            "reference image payload is empty",
            index=None,
        )
    match = _DATA_URL_RE.match(text)
    if match:
        text = match.group(1)
    # MCP clients sometimes wrap base64 across lines.
    return "".join(text.split())


def decode_reference_image_base64(encoded: str, *, index: int) -> bytes:
    """Decode one reference_images[] entry to raw bytes."""
    try:
        normalized = normalize_reference_base64(encoded)
        data = base64.b64decode(normalized, validate=True)
    except InvalidReferenceImageError:
        raise
    except Exception as error:
        raise InvalidReferenceImageError(
            "reference image is not valid base64",
            index=index,
            reason=str(error),
        ) from error

    if not data:
        raise InvalidReferenceImageError(
            "reference image decoded to empty bytes",
            index=index,
        )
    return data


def validate_reference_image_bytes(data: bytes, *, index: int) -> None:
    """Fully decode the image (not just the file header)."""
    if len(data) < len(PNG_SIGNATURE) or not data.startswith(PNG_SIGNATURE):
        raise InvalidReferenceImageError(
            "reference image must be a PNG (missing PNG signature)",
            index=index,
            size_bytes=len(data),
        )

    try:
        from PIL import Image
    except ImportError as error:
        raise InvalidReferenceImageError(
            "server cannot validate reference images (Pillow not installed)",
            index=index,
        ) from error

    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except OSError as error:
        raise InvalidReferenceImageError(
            "reference image is corrupt or truncated",
            index=index,
            reason=str(error),
            size_bytes=len(data),
            sha256_prefix=reference_sha256_prefix(data),
        ) from error
    except Exception as error:
        raise InvalidReferenceImageError(
            "reference image could not be decoded",
            index=index,
            reason=str(error),
            size_bytes=len(data),
        ) from error

    if width < 1 or height < 1:
        raise InvalidReferenceImageError(
            "reference image has invalid dimensions",
            index=index,
            width=width,
            height=height,
        )


def reference_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reference_sha256_prefix(data: bytes, hex_chars: int = 16) -> str:
    return reference_sha256(data)[:hex_chars]


def canonicalize_reference_png_for_comfy(data: bytes, *, index: int) -> bytes:
    """Re-encode to a fresh PNG so ComfyUI LoadImage reads a plain raster file.

    ComfyUI may route marginal/corrupt files through PyAV (VideoFromFile), which
    fails on static PNG with avcodec_receive_frame errors.
    """
    validate_reference_image_bytes(data, index=index)

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

    validate_reference_image_bytes(out, index=index)
    return out


def decode_and_validate_reference_image(encoded: str, *, index: int) -> bytes:
    data = decode_reference_image_base64(encoded, index=index)
    return canonicalize_reference_png_for_comfy(data, index=index)
