"""Pure input-validation helpers for audio.

These functions check *claims* (extension, size, metadata, sample values). Whether bytes are really
audio is decided by the decoder in :mod:`flawbench.audio.loader`; the extension is never trusted.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from flawbench.config import AudioSettings
from flawbench.exceptions import (
    AudioFileAccessError,
    EmptyInputError,
    FileTooLargeError,
    InvalidAudioError,
    UnsupportedFormatError,
)


def validate_extension(filename: str, settings: AudioSettings) -> str:
    """Return the normalized extension of ``filename`` or raise if it is not allowed."""
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        shown = ext or "(none)"
        raise UnsupportedFormatError(
            f"Unsupported file type {shown}. Allowed: {', '.join(settings.allowed_extensions)}."
        )
    return ext


def validate_size(size_bytes: int, settings: AudioSettings) -> None:
    """Reject empty input and input above the configured limit."""
    if size_bytes <= 0:
        raise EmptyInputError("The file is empty.")
    if size_bytes > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"The file exceeds the {settings.max_upload_mb} MB limit.",
            detail=f"size={size_bytes} bytes",
        )


def validate_file_path(path: Path, settings: AudioSettings) -> int:
    """Check existence, readability, extension and size using ``stat`` only; return size in bytes.

    The file is not read, so large files are not pulled into memory.
    """
    try:
        if not path.is_file():
            raise AudioFileAccessError("The audio file was not found.", detail=str(path))
        size = path.stat().st_size
    except OSError as exc:
        raise AudioFileAccessError(
            "The audio file could not be accessed.", detail=f"{path}: {exc}"
        ) from exc
    if not os.access(path, os.R_OK):
        raise AudioFileAccessError("The audio file is not readable.", detail=str(path))
    validate_extension(path.name, settings)
    validate_size(size, settings)
    return size


def validate_stream_properties(
    sample_rate: int, channels: int, frames: int, settings: AudioSettings
) -> None:
    """Validate header-level properties reported by the decoder."""
    if frames <= 0:
        raise InvalidAudioError("The audio contains no samples.")
    if sample_rate < settings.min_sample_rate_hz:
        raise InvalidAudioError(
            f"The sample rate ({sample_rate} Hz) is below the minimum of "
            f"{settings.min_sample_rate_hz} Hz."
        )
    if channels < 1 or channels > settings.max_channels:
        raise InvalidAudioError(
            f"The audio has {channels} channel(s); at most {settings.max_channels} are accepted."
        )


def validate_samples(samples: NDArray[np.floating], settings: AudioSettings) -> None:
    """Reject NaN/infinite samples when configured to do so."""
    if settings.reject_non_finite_samples and not np.isfinite(samples).all():
        bad = int(samples.size - np.count_nonzero(np.isfinite(samples)))
        raise InvalidAudioError(
            "The audio contains invalid (non-finite) sample values.", detail=f"{bad} samples"
        )
