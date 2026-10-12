"""Optional MCP progress hooks during ComfyUI waits (context-local)."""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable

ComfyWaitProgress = Callable[[float, str], Awaitable[None]]

_VAR: contextvars.ContextVar[ComfyWaitProgress | None] = contextvars.ContextVar(
    "comfy_wait_progress",
    default=None,
)


def set_wait_progress(handler: ComfyWaitProgress | None) -> contextvars.Token[ComfyWaitProgress | None]:
    return _VAR.set(handler)


def reset_wait_progress(token: contextvars.Token[ComfyWaitProgress | None]) -> None:
    _VAR.reset(token)


async def notify_wait_progress(elapsed_seconds: float, history_state: str) -> None:
    handler = _VAR.get()
    if handler is None:
        return
    await handler(elapsed_seconds, history_state)
