"""Concurrency limits for image backends."""

from __future__ import annotations

import asyncio

import image_concurrency


def test_comfy_and_sglang_limits_are_bounded_and_independent(monkeypatch) -> None:
    monkeypatch.setenv("COMFYUI_CONCURRENCY_LIMIT", "2")
    monkeypatch.setenv("SGLANG_CONCURRENCY_LIMIT", "1")
    image_concurrency.reset_image_concurrency_for_tests()

    async def run() -> tuple[int, int]:
        active = {"comfy": 0, "sglang": 0}
        maximum = {"comfy": 0, "sglang": 0}

        async def work(name, slot):
            async with slot():
                active[name] += 1
                maximum[name] = max(maximum[name], active[name])
                await asyncio.sleep(0.02)
                active[name] -= 1

        await asyncio.gather(
            *(work("comfy", image_concurrency.comfy_gpu_slot) for _ in range(6)),
            *(work("sglang", image_concurrency.vlm_slot) for _ in range(4)),
        )
        return maximum["comfy"], maximum["sglang"]

    try:
        comfy_maximum, sglang_maximum = asyncio.run(run())
        assert comfy_maximum == 2
        assert sglang_maximum == 1
    finally:
        image_concurrency.reset_image_concurrency_for_tests()


def test_comfy_and_sglang_work_can_overlap(monkeypatch) -> None:
    monkeypatch.setenv("COMFYUI_CONCURRENCY_LIMIT", "1")
    monkeypatch.setenv("SGLANG_CONCURRENCY_LIMIT", "1")
    image_concurrency.reset_image_concurrency_for_tests()

    async def run() -> None:
        started = {"comfy": asyncio.Event(), "sglang": asyncio.Event()}

        async def hold(name, slot):
            async with slot():
                started[name].set()
                await asyncio.wait_for(started["comfy"].wait(), timeout=1.0)
                await asyncio.wait_for(started["sglang"].wait(), timeout=1.0)

        await asyncio.wait_for(
            asyncio.gather(
                hold("comfy", image_concurrency.comfy_gpu_slot),
                hold("sglang", image_concurrency.vlm_slot),
            ),
            timeout=2.0,
        )

    try:
        asyncio.run(run())
    finally:
        image_concurrency.reset_image_concurrency_for_tests()


def test_failure_inside_one_slot_does_not_cancel_independent_work(monkeypatch) -> None:
    monkeypatch.setenv("COMFYUI_CONCURRENCY_LIMIT", "1")
    monkeypatch.setenv("SGLANG_CONCURRENCY_LIMIT", "1")
    image_concurrency.reset_image_concurrency_for_tests()

    async def run():
        async def fail_comfy():
            async with image_concurrency.comfy_gpu_slot():
                raise RuntimeError("comfy failed")

        async def complete_vlm():
            async with image_concurrency.vlm_slot():
                return "vlm complete"

        return await asyncio.gather(fail_comfy(), complete_vlm(), return_exceptions=True)

    try:
        errors = asyncio.run(run())
        assert isinstance(errors[0], RuntimeError)
        assert errors[1] == "vlm complete"
    finally:
        image_concurrency.reset_image_concurrency_for_tests()
