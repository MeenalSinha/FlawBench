"""Exception hierarchy.

Every exception carries two messages:

* ``user_message``: concise, safe to show in the interface (no filesystem paths, no internals).
* ``detail``: technical context intended for logs (may contain paths and library messages).
"""

from __future__ import annotations


class FlawBenchError(Exception):
    """Base class for all expected FlawBench failures."""

    def __init__(self, user_message: str, *, detail: str | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail

    def public_message(self, *, expose_details: bool = False) -> str:
        """Return the message for the interface, optionally including technical detail."""
        if expose_details and self.detail:
            return f"{self.user_message} ({self.detail})"
        return self.user_message


class ConfigurationError(FlawBenchError):
    """Configuration is missing, malformed, or invalid."""


class StorageError(FlawBenchError):
    """A configured directory is missing, not a directory, or not accessible."""


class AudioInputError(FlawBenchError):
    """Base class for rejected or unreadable audio input."""


class UnsupportedFormatError(AudioInputError):
    """Extension or detected container format is not allowed by configuration."""


class EmptyInputError(AudioInputError):
    """The input contains no data."""


class FileTooLargeError(AudioInputError):
    """The input exceeds the configured size limit."""


class AudioFileAccessError(AudioInputError):
    """The audio file does not exist or cannot be read."""


class AudioDecodingError(AudioInputError):
    """The bytes are not decodable as audio."""


class InvalidAudioError(AudioInputError):
    """Audio decoded but violates configured validity rules (rate, channels, samples)."""


class DatasetError(FlawBenchError):
    """Base class for dataset-engine failures."""


class RegistryError(DatasetError):
    """Source registration or lookup failed (missing rights metadata, duplicates, corruption)."""


class InjectionError(DatasetError):
    """A flaw transformation request is invalid or its output failed verification."""


class GenerationError(DatasetError):
    """A variant could not be generated or committed."""


class FeatureError(FlawBenchError):
    """Base class for feature-extraction and comparison failures."""


class FeatureExtractionError(FeatureError):
    """Extraction failed on audio input (e.g. invalid length, unhandled shape, numerical fault)."""


class FeatureStoreError(FeatureError):
    """Persistence or retrieval of feature artifacts failed."""


class FeatureComparisonError(FeatureError):
    """Differential analysis between reference and variant features failed."""


class DetectionError(FlawBenchError):
    """Base class for Phase 4 flaw detection and temporal localization failures."""


class DetectionInputError(DetectionError):
    """Input resolution or construction of label-free detection input failed."""


class DetectionStoreError(DetectionError):
    """Persistence or retrieval of detection artifacts failed."""


class DetectionValidationError(DetectionError):
    """Integrity check of saved detection results failed."""
