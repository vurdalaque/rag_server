"""Wall-clock timing helpers for MCP image structured output."""

from __future__ import annotations

import time
from typing import Any


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0


def elapsed_ms(start_monotonic: float) -> float:
    return round((time.monotonic() - start_monotonic) * 1000.0, 1)


def merge_timings_ms(**stages: float | None) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in stages.items():
        if value is None:
            continue
        name = key if key.endswith("_ms") else f"{key}_ms"
        payload[name] = round(value, 1)
    if "total_ms" not in payload and payload:
        payload["total_ms"] = max(payload.values())
    return payload


def attach_total(timings: dict[str, Any], total_ms: float) -> dict[str, float]:
    merged = {str(k): float(v) for k, v in timings.items()}
    merged["total_ms"] = round(total_ms, 1)
    return merged
