from __future__ import annotations

from typing import Any

_SENSITIVE_PAYLOAD_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "api-key",
        "x-api-key",
        "token",
        "password",
        "secret",
        "cookie",
    }
)


def sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(key, str) and key.lower() in _SENSITIVE_PAYLOAD_KEYS:
            continue
        if isinstance(value, dict):
            sanitized[key] = sanitize_payload(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_payload(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized
