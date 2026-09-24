"""Concurrency limits for image backends."""

from __future__ import annotations

import asyncio

import image_concurrency


def test_comfy_slots_allow_parallel_acquire() -> None:
    image_concurrency.reset_image_concurrency_for_tests()

    async def run() -> None:
        async with image_concurrency.comfy_gpu_slot():
            async with image_concurrency.comfy_gpu_slot():
                return

        await asyncio.gather(
            asyncio.create_task(_hold_slot()),
            asyncio.create_task(_hold_slot()),
        )

    async def _hold_slot() -> None:
        async with image_concurrency.comfy_gpu_slot():
            await asyncio.sleep(0.01)

    asyncio.run(run())
