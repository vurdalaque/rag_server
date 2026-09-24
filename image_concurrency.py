"""Bounded concurrency for external image backends."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from llm_params import env_int

_comfy_semaphore: asyncio.Semaphore | None = None
_vlm_semaphore: asyncio.Semaphore | None = None


def _comfy_limit() -> int:
    return max(1, env_int("IMAGE_COMFY_MAX_CONCURRENT", 4))


def _vlm_limit() -> int:
    return max(1, env_int("IMAGE_VLM_MAX_CONCURRENT", 8))


def _comfy_sem() -> asyncio.Semaphore:
    global _comfy_semaphore
    if _comfy_semaphore is None:
        _comfy_semaphore = asyncio.Semaphore(_comfy_limit())
    return _comfy_semaphore


def _vlm_sem() -> asyncio.Semaphore:
    global _vlm_semaphore
    if _vlm_semaphore is None:
        _vlm_semaphore = asyncio.Semaphore(_vlm_limit())
    return _vlm_semaphore


@asynccontextmanager
async def comfy_gpu_slot() -> AsyncIterator[None]:
    async with _comfy_sem():
        yield


@asynccontextmanager
async def vlm_slot() -> AsyncIterator[None]:
    async with _vlm_sem():
        yield


def reset_image_concurrency_for_tests() -> None:
    global _comfy_semaphore, _vlm_semaphore
    _comfy_semaphore = None
    _vlm_semaphore = None
