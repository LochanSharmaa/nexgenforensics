"""Structured logging.

JSON in every environment except local development, where a human-readable
console renderer is used instead. `investigation_id`, `run_id` and `stage` bind
to the context so every line emitted during a pipeline stage carries them without
each call site repeating itself — which is what makes "show me everything that
happened during CRAWL on case 114" a grep rather than an archaeology exercise.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars

_configured = False


def configure_logging(*, level: str = "INFO", fmt: str = "json") -> None:
    """Idempotent. Safe to call from api, worker, CLI and tests."""
    global _configured  # noqa: PLW0603

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )

    # Third-party loggers are noisy at INFO and drown the signal.
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "asyncio", "botocore", "aiobotocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "console":
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


@contextmanager
def log_context(**bindings: Any) -> Iterator[None]:
    """Bind fields for the duration of a block, then restore.

    Unbinding rather than clearing matters: a stage running inside a request must
    not wipe the request's own bindings when it finishes.
    """
    bind_contextvars(**bindings)
    try:
        yield
    finally:
        unbind_contextvars(*bindings.keys())


def bind_investigation(investigation_id: str, **extra: Any) -> None:
    bind_contextvars(investigation_id=investigation_id, **extra)


def clear_context() -> None:
    clear_contextvars()


__all__ = [
    "bind_investigation",
    "clear_context",
    "configure_logging",
    "get_logger",
    "log_context",
]
