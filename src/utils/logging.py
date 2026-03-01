"""
Consistent logger setup for Fantasy F1 2026 pipeline modules.

All modules should use get_logger(__name__) so that log levels and format
are controlled from one place.

Usage:
    from src.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("Processing round %s", round_number)
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a logger for the given module name.

    Configures the root fantasy_f1 logger on first use: stdout handler,
    format with timestamp and level. Child loggers inherit the level.

    Args:
        name: Module name (typically __name__).
        level: Optional override for log level (e.g. logging.DEBUG).
               If None, defaults to INFO.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    # Configure root logger once so all modules (e.g. src.data.scoring) inherit
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(level or logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level or logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

    return logger
