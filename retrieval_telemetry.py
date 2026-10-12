"""Backward-compatible shim; see radius_rag.observability.telemetry.writer."""

from radius_rag.observability.storage.db import insert_events
from radius_rag.observability.telemetry import writer as _writer

for _name in dir(_writer):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_writer, _name)

__all__ = [n for n in globals() if not n.startswith("_")]
