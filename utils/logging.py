"""Loguru configuration shared by the API, scripts and workers."""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        enqueue=True,  # thread/process safe
    )
    _CONFIGURED = True


__all__ = ["logger", "setup_logging"]
