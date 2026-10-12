"""Uvicorn entrypoint: rag_server:app (legacy imports via radius_rag._legacy_bootstrap)."""

from radius_rag._legacy_bootstrap import install_legacy_imports

install_legacy_imports()

from radius_rag.api.server import *  # noqa: F403
