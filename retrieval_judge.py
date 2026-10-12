"""Backward-compatible shim; see radius_rag.observability.evaluation.judge."""

from radius_rag.observability.evaluation import judge as _judge

for _name in dir(_judge):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_judge, _name)
