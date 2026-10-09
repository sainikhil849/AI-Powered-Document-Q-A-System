"""Logging setup.

Logs carry diagnostic metadata (document IDs, chunk counts, node names,
timings, error types) but never document contents or full prompts.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once; later calls only adjust the level."""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(level)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
        _CONFIGURED = True
    # Third-party libraries are noisy at INFO and may log request bodies.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "pypdf", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
