"""Backward-compatible shim; see radius_rag.observability.evaluation.worker."""

from radius_rag.observability.evaluation import worker as _worker

for _name in dir(_worker):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_worker, _name)
