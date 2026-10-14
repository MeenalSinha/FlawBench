"""Central logging setup using the standard library.

All FlawBench loggers live under the ``flawbench`` namespace. :func:`configure_logging` is
idempotent: repeated calls (Streamlit re-executes scripts on every interaction) update the level
of the single managed handler instead of adding duplicates.
"""

from __future__ import annotations

import logging
import sys

ROOT_LOGGER_NAME = "flawbench"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_MARKER = "_flawbench_managed"


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure the ``flawbench`` logger once; later calls only adjust the level."""
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown log level: {level!r}")
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(numeric)
    logger.propagate = False  # avoid duplicate lines when a host (e.g. Streamlit) configures root
    managed = [h for h in logger.handlers if getattr(h, _HANDLER_MARKER, False)]
    if not managed:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger inside the ``flawbench`` namespace."""
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def log_failure(
    logger: logging.Logger, message: str, exc: BaseException, *, detailed: bool = False
) -> None:
    """Record a failure. Tracebacks are included only when detailed logging is enabled.

    The exception's technical ``detail`` (if any) is logged; raw audio content never is.
    """
    detail = getattr(exc, "detail", None)
    text = f"{message}: {type(exc).__name__}: {exc}" + (f" [{detail}]" if detail else "")
    if detailed:
        logger.error(text, exc_info=exc)
    else:
        logger.error(text)
