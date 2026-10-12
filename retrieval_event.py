"""Backward-compatible shim; see radius_rag.observability.event."""

from radius_rag.observability.config import TELEMETRY_ENABLED
from radius_rag.observability import event as _event
from radius_rag.observability.sanitize import sanitize_payload

for _name in dir(_event):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_event, _name)
