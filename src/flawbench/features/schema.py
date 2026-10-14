"""Feature metadata schema and label-free input structures.

Ground-truth fields (flaw_type, severity, output_interval, source_interval) are
deliberately absent from every structure in this module. Feature extraction must
never receive or use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

FEATURE_SCHEMA_VERSION = "3.0"


class AudioRole(str, Enum):
    """Whether this audio is the unmodified reference or a generated variant."""

    REFERENCE = "reference"
    VARIANT = "variant"


@dataclass(frozen=True)
class AudioInput:
    """Label-free descriptor passed to numerical extraction functions.

    Contains only the information needed to locate and identify audio.
    Flaw annotations, severity, and ground-truth intervals are NEVER included.
    """

    # Identification
    artifact_id: str  # unique id for this extraction artifact (e.g. 'src-abc123')
    source_id: str  # registered source id
    sample_id: str | None  # dataset sample id; None for reference-only extractions
    role: AudioRole  # REFERENCE or VARIANT

    # Audio data (not the Phase 2 annotation)
    audio_path: Path
    sha256: str
    sample_rate: int
    num_channels: int
    num_samples: int
    duration_seconds: float


@dataclass
class ActivityInterval:
    """A contiguous energy-active (or silent) interval in a recording.

    This is a heuristic energy-based estimate — NOT ground-truth flaw annotation.
    start_sample and end_sample are half-open [start, end) in the recording's own timeline.
    """

    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    is_active: bool  # True = above threshold (active speech estimate), False = below (silence)
    mean_rms_db: float  # Mean RMS in dB over the interval


@dataclass
class FrameSchema:
    """Documents the canonical frame grid for a feature result.

    All feature arrays produced for a single recording use this same grid.
    Frame boundaries use half-open integer-sample ranges [start_sample, end_sample).
    """

    num_frames: int
    frame_length: int  # samples per frame (analysis window)
    hop_length: int  # samples between frame starts
    sample_rate: int
    # Per-frame arrays (shape = (num_frames,)):
    start_samples: Any  # int64 ndarray
    end_samples: Any  # int64 ndarray; may be clipped at audio length for last partial frame
    center_times_sec: Any  # float64 ndarray; center time = (start + end) / 2 / sr
    valid_samples: Any  # int64 ndarray; number of non-padded samples in each frame
    padded: Any  # bool ndarray; True if frame was zero-padded (partial final frame)

    # Configuration used to generate this schema (informational)
    frame_ms: float
    hop_ms: float


@dataclass
class FeatureMetadata:
    """Non-numeric provenance metadata stored alongside a feature artifact.

    Ground-truth fields (flaw_type, severity, intervals) are NEVER stored here.
    """

    schema_version: str
    artifact_id: str
    source_id: str
    sample_id: str | None
    role: str  # AudioRole value
    audio_sha256: str
    sample_rate: int
    duration_seconds: float
    num_channels: int
    channel_policy: str
    frame_ms: float
    hop_ms: float
    n_mfcc: int
    spectral_rolloff_p: float
    pitch_backend: str
    f0_min_hz: float
    f0_max_hz: float
    silence_threshold_db: float
    min_silence_ms: float
    config_fingerprint: str
    software: dict[str, Any]
    created_utc: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "source_id": self.source_id,
            "sample_id": self.sample_id,
            "role": self.role,
            "audio_sha256": self.audio_sha256,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "num_channels": self.num_channels,
            "channel_policy": self.channel_policy,
            "frame_ms": self.frame_ms,
            "hop_ms": self.hop_ms,
            "n_mfcc": self.n_mfcc,
            "spectral_rolloff_p": self.spectral_rolloff_p,
            "pitch_backend": self.pitch_backend,
            "f0_min_hz": self.f0_min_hz,
            "f0_max_hz": self.f0_max_hz,
            "silence_threshold_db": self.silence_threshold_db,
            "min_silence_ms": self.min_silence_ms,
            "config_fingerprint": self.config_fingerprint,
            "software": self.software,
            "created_utc": self.created_utc,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureMetadata:
        return cls(
            schema_version=str(data["schema_version"]),
            artifact_id=str(data["artifact_id"]),
            source_id=str(data["source_id"]),
            sample_id=data.get("sample_id"),
            role=str(data["role"]),
            audio_sha256=str(data["audio_sha256"]),
            sample_rate=int(data["sample_rate"]),
            duration_seconds=float(data["duration_seconds"]),
            num_channels=int(data["num_channels"]),
            channel_policy=str(data["channel_policy"]),
            frame_ms=float(data["frame_ms"]),
            hop_ms=float(data["hop_ms"]),
            n_mfcc=int(data["n_mfcc"]),
            spectral_rolloff_p=float(data["spectral_rolloff_p"]),
            pitch_backend=str(data["pitch_backend"]),
            f0_min_hz=float(data["f0_min_hz"]),
            f0_max_hz=float(data["f0_max_hz"]),
            silence_threshold_db=float(data["silence_threshold_db"]),
            min_silence_ms=float(data["min_silence_ms"]),
            config_fingerprint=str(data["config_fingerprint"]),
            software=dict(data["software"]),
            created_utc=str(data["created_utc"]),
            warnings=list(data.get("warnings", [])),
        )
