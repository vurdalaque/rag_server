#!/usr/bin/env python3
"""Standalone process: apply DB migrations and run the retrieval evaluation worker."""

from __future__ import annotations

import asyncio
import logging
import signal

from radius_rag.observability.config import load_retrieval_config
from radius_rag.observability.storage.db import close_pool, run_migrations
from radius_rag.observability.evaluation.worker import start_evaluation_worker, stop_evaluation_worker

logger = logging.getLogger(__name__)


async def _run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_retrieval_config()
    if not config.database_url:
        raise SystemExit("RETRIEVAL_DATABASE_URL is required")

    applied = await run_migrations(config)
    if applied:
        logger.info("Applied migrations: %s", ", ".join(applied))

    stop_event = await start_evaluation_worker(config)
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda _signum, _frame: _request_stop())

    await stop_event.wait()
    await stop_evaluation_worker(stop_event)
    await close_pool()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
