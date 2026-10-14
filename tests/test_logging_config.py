from __future__ import annotations

import logging

import pytest

from flawbench.exceptions import AudioDecodingError
from flawbench.logging_config import (
    _HANDLER_MARKER,
    ROOT_LOGGER_NAME,
    configure_logging,
    get_logger,
    log_failure,
)


def managed(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, _HANDLER_MARKER, False)]


@pytest.fixture(autouse=True)
def clean_logger():
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    saved = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers.clear()
    yield logger
    logger.handlers[:], logger.level, logger.propagate = saved[0], saved[1], saved[2]


def test_repeated_configuration_does_not_duplicate_handlers(clean_logger: logging.Logger) -> None:
    for _ in range(5):
        configure_logging("INFO")
    assert len(managed(clean_logger)) == 1


def test_level_follows_configuration(clean_logger: logging.Logger) -> None:
    configure_logging("DEBUG")
    assert clean_logger.level == logging.DEBUG
    configure_logging("error")
    assert clean_logger.level == logging.ERROR
    assert len(managed(clean_logger)) == 1


def test_invalid_level_rejected() -> None:
    with pytest.raises(ValueError):
        configure_logging("LOUD")


def test_named_loggers_live_under_namespace() -> None:
    assert get_logger("audio.loader").name == "flawbench.audio.loader"
    assert get_logger("flawbench.paths").name == "flawbench.paths"


def test_does_not_propagate_to_root(clean_logger: logging.Logger) -> None:
    configure_logging("INFO")
    assert clean_logger.propagate is False


def test_log_failure_includes_detail_and_optional_traceback(
    caplog: pytest.LogCaptureFixture, clean_logger: logging.Logger
) -> None:
    clean_logger.propagate = True  # let caplog observe records
    exc = AudioDecodingError("Could not decode.", detail="internal libsndfile message")
    with caplog.at_level(logging.ERROR, logger=ROOT_LOGGER_NAME):
        log_failure(get_logger("t"), "Load failed", exc, detailed=False)
        assert "internal libsndfile message" in caplog.text
        assert all(r.exc_info is None for r in caplog.records)
        caplog.clear()
        try:
            raise exc
        except AudioDecodingError as caught:
            log_failure(get_logger("t"), "Load failed", caught, detailed=True)
        assert any(r.exc_info for r in caplog.records)
