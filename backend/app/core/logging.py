"""Minimal structured logging setup.

Logs go to stdout as key=value-style lines. Never log uploaded document
contents, API keys, or database passwords.
"""

import logging
from typing import Any

from fastapi import Request

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a single stdout handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def get_request_id_from_state(request: Request | None) -> str | None:
    """Safely extract request ID from request state."""
    if request and hasattr(request.state, 'request_id'):
        return request.state.request_id
    return None