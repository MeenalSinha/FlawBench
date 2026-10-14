"""Phase 3 Acoustic Feature Extraction.

Extracts time-indexed acoustic features from raw audio using a canonical frame grid:
- RMS energy (linear and dB)
- Zero Crossing Rate (ZCR)
- Spectral Centroid, Bandwidth, Rolloff, and Flatness
- Mel-Frequency Cepstral Coefficients (MFCCs)
- Fundamental Frequency (F0 / pitch) with voicing mask and confidence
- Heuristic energy-based activity intervals (VAD-like, non-ground-truth)
- Summary statistics across the recording

Every feature is calculated directly from actual audio samples.
Ground truth flaw annotations are never accessed or used by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from numpy.typing import NDArray

from flawbench.audio.loader import LoadedAudio
from flawbench.config import Config, FeaturesSettings
from flawbench.exceptions import FeatureExtractionError
from flawbench.features.schema import (
    FEATURE_SCHEMA_VERSION,
    ActivityInterval,
    AudioInput,
    AudioRole,
    FeatureMetadata,
    FrameSchema,
)
from flawbench.provenance import collect_software_versions


@dataclass
class FeatureResult:
    """Container for time-indexed acoustic features extracted from an audio recording.

    All 1D and 2D arrays have their primary axis equal to ``frame_schema.num_frames``.
    """

    metadata: FeatureMetadata
    frame_schema: FrameSchema

    # Time series arrays (shape: (num_frames,))
    rms: NDArray[np.float64]
    rms_db: NDArray[np.float64]
    zcr: NDArray[np.float64]
    spectral_centroid: NDArray[np.float64]
    spectral_bandwidth: NDArray[np.float64]
    spectral_rolloff: NDArray[np.float64]
    spectral_flatness: NDArray[np.float64]
    f0_hz: NDArray[np.float64]  # NaN where unvoiced
    voiced_flag: NDArray[np.bool_]
    pitch_confidence: NDArray[np.float64]

    # MFCC matrix (shape: (num_frames, n_mfcc))
    mfcc: NDArray[np.float64]

    # Segmentation & summary
    activity_intervals: list[ActivityInterval] = field(default_factory=list)
    summary_stats: dict[str, float] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return self.frame_schema.num_frames

    @property
    def center_times_sec(self) -> NDArray[np.float64]:
        return np.asarray(self.frame_schema.center_times_sec, dtype=np.float64)

    def get_feature(self, name: str) -> NDArray[Any]:
        """Access a feature array by name."""
        mapping = {
            "rms": self.rms,
            "rms_db": self.rms_db,
            "zcr": self.zcr,
            "spectral_centroid": self.spectral_centroid,
            "spectral_bandwidth": self.spectral_bandwidth,
            "spectral_rolloff": self.spectral_rolloff,
            "spectral_flatness": self.spectral_flatness,
            "f0_hz": self.f0_hz,
            "voiced_flag": self.voiced_flag,
            "pitch_confidence": self.pitch_confidence,
            "mfcc": self.mfcc,
        }
        if name not in mapping:
            raise KeyError(f"Unknown feature {name!r}. Available: {list(mapping.keys())}")
        return mapping[name]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary representation."""
        return {
            "metadata": self.metadata.to_dict(),
            "frame_schema": {
                "num_frames": self.frame_schema.num_frames,
                "frame_length": self.frame_schema.frame_length,
                "hop_length": self.frame_schema.hop_length,
                "sample_rate": self.frame_schema.sample_rate,
                "frame_ms": self.frame_schema.frame_ms,
                "hop_ms": self.frame_schema.hop_ms,
            },
            "summary_stats": self.summary_stats,
            "activity_intervals": [
                {
                    "start_sample": i.start_sample,
                    "end_sample": i.end_sample,
                    "start_sec": i.start_sec,
                    "end_sec": i.end_sec,
                    "is_active": i.is_active,
                    "mean_rms_db": i.mean_rms_db,
                }
                for i in self.activity_intervals
            ],
        }

    def save(self, directory: Path) -> tuple[Path, Path]:
        """Save this FeatureResult to a directory as .npz (arrays) and .json (metadata).

        Returns:
            Tuple of (npz_path, json_path).
        """
        directory.mkdir(parents=True, exist_ok=True)
        artifact_id = self.metadata.artifact_id
        npz_path = directory / f"{artifact_id}.npz"
        json_path = directory / f"{artifact_id}.json"

        # Save arrays into compressed npz
        np.savez_compressed(
            npz_path,
            rms=self.rms,
            rms_db=self.rms_db,
            zcr=self.zcr,
            spectral_centroid=self.spectral_centroid,
            spectral_bandwidth=self.spectral_bandwidth,
            spectral_rolloff=self.spectral_rolloff,
            spectral_flatness=self.spectral_flatness,
            f0_hz=self.f0_hz,
            voiced_flag=self.voiced_flag,
            pitch_confidence=self.pitch_confidence,
            mfcc=self.mfcc,
            start_samples=self.frame_schema.start_samples,
            end_samples=self.frame_schema.end_samples,
            center_times_sec=self.frame_schema.center_times_sec,
            valid_samples=self.frame_schema.valid_samples,
            padded=self.frame_schema.padded,
        )

        # Save metadata and intervals to json
        doc = self.to_dict()
        json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return npz_path, json_path

    @classmethod
    def load(cls, npz_path: Path, json_path: Path) -> FeatureResult:
        """Load a FeatureResult from its .npz and .json files."""
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        metadata = FeatureMetadata.from_dict(doc["metadata"])
        summary_stats = doc.get("summary_stats", {})

        raw_intervals = doc.get("activity_intervals", [])
        activity_intervals = [
            ActivityInterval(
                start_sample=int(i["start_sample"]),
                end_sample=int(i["end_sample"]),
                start_sec=float(i["start_sec"]),
                end_sec=float(i["end_sec"]),
                is_active=bool(i["is_active"]),
                mean_rms_db=float(i["mean_rms_db"]),
            )
            for i in raw_intervals
        ]

        arrays = np.load(npz_path)
        frame_cfg = doc["frame_schema"]
        frame_schema = FrameSchema(
            num_frames=int(frame_cfg["num_frames"]),
            frame_length=int(frame_cfg["frame_length"]),
            hop_length=int(frame_cfg["hop_length"]),
            sample_rate=int(frame_cfg["sample_rate"]),
            start_samples=arrays["start_samples"],
            end_samples=arrays["end_samples"],
            center_times_sec=arrays["center_times_sec"],
            valid_samples=arrays["valid_samples"],
            padded=arrays["padded"],
            frame_ms=float(frame_cfg["frame_ms"]),
            hop_ms=float(frame_cfg["hop_ms"]),
        )

        return cls(
            metadata=metadata,
            frame_schema=frame_schema,
            rms=arrays["rms"],
            rms_db=arrays["rms_db"],
            zcr=arrays["zcr"],
            spectral_centroid=arrays["spectral_centroid"],
            spectral_bandwidth=arrays["spectral_bandwidth"],
            spectral_rolloff=arrays["spectral_rolloff"],
            spectral_flatness=arrays["spectral_flatness"],
            f0_hz=arrays["f0_hz"],
            voiced_flag=arrays["voiced_flag"],
            pitch_confidence=arrays["pitch_confidence"],
            mfcc=arrays["mfcc"],
            activity_intervals=activity_intervals,
            summary_stats=summary_stats,
        )


def _compute_activity_intervals(
    rms_db: NDArray[np.float64],
    frame_schema: FrameSchema,
    threshold_db: float,
    min_silence_ms: float,
) -> list[ActivityInterval]:
    """Segment audio into contiguous active and silent intervals based on RMS energy threshold."""
    if frame_schema.num_frames == 0:
        return []

    is_active_frame = rms_db >= threshold_db
    sr = frame_schema.sample_rate

    # Initial runs
    intervals: list[ActivityInterval] = []
    run_active = bool(is_active_frame[0])
    run_start_idx = 0

    for idx in range(1, frame_schema.num_frames):
        current_active = bool(is_active_frame[idx])
        if current_active != run_active:
            start_sample = int(frame_schema.start_samples[run_start_idx])
            end_sample = int(frame_schema.end_samples[idx - 1])
            mean_db = float(np.mean(rms_db[run_start_idx:idx]))
            intervals.append(
                ActivityInterval(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_sec=float(start_sample / sr),
                    end_sec=float(end_sample / sr),
                    is_active=run_active,
                    mean_rms_db=mean_db,
                )
            )
            run_active = current_active
            run_start_idx = idx

    # Final run
    start_sample = int(frame_schema.start_samples[run_start_idx])
    end_sample = int(frame_schema.end_samples[-1])
    mean_db = float(np.mean(rms_db[run_start_idx:]))
    intervals.append(
        ActivityInterval(
            start_sample=start_sample,
            end_sample=end_sample,
            start_sec=float(start_sample / sr),
            end_sec=float(end_sample / sr),
            is_active=run_active,
            mean_rms_db=mean_db,
        )
    )

    return intervals


def _compute_summary_stats(
    rms_db: NDArray[np.float64],
    zcr: NDArray[np.float64],
    spectral_centroid: NDArray[np.float64],
    spectral_bandwidth: NDArray[np.float64],
    spectral_rolloff: NDArray[np.float64],
    spectral_flatness: NDArray[np.float64],
    f0_hz: NDArray[np.float64],
    voiced_flag: NDArray[np.bool_],
    activity_intervals: list[ActivityInterval],
    duration_seconds: float,
    num_frames: int,
) -> dict[str, float]:
    """Compute global aggregate summary statistics."""
    stats: dict[str, float] = {
        "duration_seconds": float(duration_seconds),
        "num_frames": float(num_frames),
    }

    if num_frames == 0:
        return stats

    stats["mean_rms_db"] = float(np.mean(rms_db))
    stats["std_rms_db"] = float(np.std(rms_db))
    stats["max_rms_db"] = float(np.max(rms_db))
    stats["min_rms_db"] = float(np.min(rms_db))

    stats["mean_zcr"] = float(np.mean(zcr))
    stats["mean_spectral_centroid"] = float(np.mean(spectral_centroid))
    stats["mean_spectral_bandwidth"] = float(np.mean(spectral_bandwidth))
    stats["mean_spectral_rolloff"] = float(np.mean(spectral_rolloff))
    stats["mean_spectral_flatness"] = float(np.mean(spectral_flatness))

    voiced_count = int(np.sum(voiced_flag))
    stats["voiced_ratio"] = float(voiced_count / num_frames)

    if voiced_count > 0:
        voiced_f0 = f0_hz[voiced_flag]
        stats["f0_mean_voiced_hz"] = float(np.mean(voiced_f0))
        stats["f0_std_voiced_hz"] = float(np.std(voiced_f0))
        stats["f0_min_voiced_hz"] = float(np.min(voiced_f0))
        stats["f0_max_voiced_hz"] = float(np.max(voiced_f0))
    else:
        stats["f0_mean_voiced_hz"] = 0.0
        stats["f0_std_voiced_hz"] = 0.0
        stats["f0_min_voiced_hz"] = 0.0
        stats["f0_max_voiced_hz"] = 0.0

    active_duration = sum(i.end_sec - i.start_sec for i in activity_intervals if i.is_active)
    stats["active_speech_ratio"] = (
        float(active_duration / duration_seconds) if duration_seconds > 0 else 0.0
    )
    return stats


def extract_features(
    audio: LoadedAudio | NDArray[Any],
    sample_rate: int | None = None,
    feat_cfg: FeaturesSettings | None = None,
    *,
    audio_input: AudioInput | None = None,
    config: Config | None = None,
) -> FeatureResult:
    """Extract full time-indexed acoustic features from raw audio.

    Args:
        audio: LoadedAudio instance or 1D/2D float numpy array.
        sample_rate: Required if audio is a numpy array.
        feat_cfg: FeaturesSettings instance (defaults to config.features or standard defaults).
        audio_input: Optional label-free metadata describing the audio provenance.
        config: Optional root Config.

    Returns:
        FeatureResult containing frame schema, time-series arrays, and metadata.
    """
    if feat_cfg is None:
        if config is not None:
            feat_cfg = config.features
        else:
            from flawbench.config import load_config

            feat_cfg = load_config().features

    # Handle input type and downmixing
    if isinstance(audio, LoadedAudio):
        sr = audio.sample_rate
        num_channels = audio.num_channels
        mono_signal = audio.mono()

        num_samples = audio.num_samples
    elif isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise FeatureExtractionError(
                "sample_rate is required when extracting features from a raw numpy array."
            )
        sr = int(sample_rate)
        if audio.ndim == 1:
            num_channels = 1
            mono_signal = np.ascontiguousarray(audio, dtype=np.float32)
            num_samples = len(audio)
        elif audio.ndim == 2:
            # FlawBench convention: (channels, samples)
            if audio.shape[0] <= audio.shape[1] and audio.shape[0] <= 8:
                num_channels = audio.shape[0]
                mono_signal = np.ascontiguousarray(audio.mean(axis=0), dtype=np.float32)
                num_samples = audio.shape[1]
            else:
                num_channels = audio.shape[1]
                mono_signal = np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)
                num_samples = audio.shape[0]
        else:
            raise FeatureExtractionError(
                f"Audio array must be 1D or 2D, got shape {audio.shape}."
            )
    else:
        raise FeatureExtractionError(f"Unsupported audio type: {type(audio)}.")

    if num_samples == 0:
        raise FeatureExtractionError("Audio has 0 samples; cannot extract features.")

    duration_sec = num_samples / sr
    if duration_sec > feat_cfg.max_feature_seconds:
        raise FeatureExtractionError(
            f"Audio duration ({duration_sec:.2f}s) exceeds max allowed feature extraction duration "
            f"({feat_cfg.max_feature_seconds:.1f}s)."
        )

    # Calculate frame geometry
    frame_length = int(round(sr * (feat_cfg.frame_ms / 1000.0)))
    hop_length = int(round(sr * (feat_cfg.hop_ms / 1000.0)))
    if frame_length <= 0 or hop_length <= 0:
        raise FeatureExtractionError(
            f"Invalid frame/hop length in samples: frame={frame_length}, hop={hop_length}."
        )

    num_frames = int(math.ceil(num_samples / hop_length))
    target_length = (num_frames - 1) * hop_length + frame_length
    pad_amount = max(0, target_length - num_samples)
    y_padded = np.pad(mono_signal, (0, pad_amount), mode="constant")

    # Construct canonical FrameSchema
    start_samples = np.arange(num_frames, dtype=np.int64) * hop_length
    end_samples = np.minimum(num_samples, start_samples + frame_length).astype(np.int64)
    valid_samples = np.maximum(0, np.minimum(num_samples - start_samples, frame_length)).astype(np.int64)
    padded = ((start_samples + frame_length) > num_samples)
    center_times_sec = (start_samples + end_samples) / (2.0 * sr)

    frame_schema = FrameSchema(
        num_frames=num_frames,
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=sr,
        start_samples=start_samples,
        end_samples=end_samples,
        center_times_sec=center_times_sec,
        valid_samples=valid_samples,
        padded=padded,
        frame_ms=feat_cfg.frame_ms,
        hop_ms=feat_cfg.hop_ms,
    )

    # 1. RMS Energy
    rms_arr = librosa.feature.rms(
        y=y_padded, frame_length=frame_length, hop_length=hop_length, center=False
    )[0, :num_frames].astype(np.float64)
    # Floor RMS to 1e-6 (-120 dB) to avoid log of 0
    rms_floored = np.maximum(rms_arr, 1e-6)
    rms_db_arr = (20.0 * np.log10(rms_floored)).astype(np.float64)

    # 2. Zero Crossing Rate
    zcr_arr = librosa.feature.zero_crossing_rate(
        y=y_padded, frame_length=frame_length, hop_length=hop_length, center=False
    )[0, :num_frames].astype(np.float64)

    # 3. Spectral Features (Centroid, Bandwidth, Rolloff, Flatness)
    centroid_arr = librosa.feature.spectral_centroid(
        y=y_padded, sr=sr, n_fft=frame_length, hop_length=hop_length, center=False
    )[0, :num_frames].astype(np.float64)

    bandwidth_arr = librosa.feature.spectral_bandwidth(
        y=y_padded, sr=sr, n_fft=frame_length, hop_length=hop_length, center=False
    )[0, :num_frames].astype(np.float64)

    rolloff_arr = librosa.feature.spectral_rolloff(
        y=y_padded,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        roll_percent=feat_cfg.spectral_rolloff_p,
        center=False,
    )[0, :num_frames].astype(np.float64)

    flatness_arr = librosa.feature.spectral_flatness(
        y=y_padded, n_fft=frame_length, hop_length=hop_length, center=False
    )[0, :num_frames].astype(np.float64)

    # 4. MFCCs (transposed to (num_frames, n_mfcc))
    mfcc_raw = librosa.feature.mfcc(
        y=y_padded,
        sr=sr,
        n_mfcc=feat_cfg.n_mfcc,
        n_fft=frame_length,
        hop_length=hop_length,
        center=False,
    )[:, :num_frames]
    mfcc_arr = np.ascontiguousarray(mfcc_raw.T, dtype=np.float64)

    # 5. Pitch Estimation (F0, voiced mask, confidence)
    pitch_frame_len = max(
        frame_length,
        int(2 ** math.ceil(math.log2(max(1.0, 2.0 * sr / feat_cfg.f0_min_hz)))),
    )
    extra_pad = max(0, pitch_frame_len - frame_length)
    y_pitch_padded = np.pad(y_padded, (0, extra_pad), mode="constant")

    backend = feat_cfg.pitch_backend
    if backend == "librosa_pyin":
        raw_f0, pyin_voiced, pyin_prob = librosa.pyin(
            y_pitch_padded,
            fmin=feat_cfg.f0_min_hz,
            fmax=feat_cfg.f0_max_hz,
            sr=sr,
            frame_length=pitch_frame_len,
            hop_length=hop_length,
            center=False,
        )
        raw_f0 = raw_f0[:num_frames]
        pyin_voiced = pyin_voiced[:num_frames]
        pyin_prob = pyin_prob[:num_frames]
        voiced_flag = np.asarray(pyin_voiced, dtype=np.bool_)
        f0_hz = np.where(voiced_flag, raw_f0, np.nan).astype(np.float64)
        pitch_confidence = np.asarray(pyin_prob, dtype=np.float64)
    else:  # librosa_yin default or fallback
        raw_f0 = librosa.yin(
            y_pitch_padded,
            fmin=feat_cfg.f0_min_hz,
            fmax=feat_cfg.f0_max_hz,
            sr=sr,
            frame_length=pitch_frame_len,
            hop_length=hop_length,
            center=False,
        )[:num_frames]
        is_above_silence = rms_db_arr >= feat_cfg.silence_threshold_db
        in_pitch_range = (raw_f0 >= feat_cfg.f0_min_hz) & (raw_f0 <= feat_cfg.f0_max_hz)
        voiced_flag = is_above_silence & in_pitch_range & np.isfinite(raw_f0)
        f0_hz = np.where(voiced_flag, raw_f0, np.nan).astype(np.float64)
        pitch_confidence = np.where(voiced_flag, 1.0, 0.0).astype(np.float64)

    # 6. Activity Intervals
    activity_intervals = _compute_activity_intervals(
        rms_db_arr, frame_schema, feat_cfg.silence_threshold_db, feat_cfg.min_silence_ms
    )

    # 7. Summary Stats
    summary_stats = _compute_summary_stats(
        rms_db=rms_db_arr,
        zcr=zcr_arr,
        spectral_centroid=centroid_arr,
        spectral_bandwidth=bandwidth_arr,
        spectral_rolloff=rolloff_arr,
        spectral_flatness=flatness_arr,
        f0_hz=f0_hz,
        voiced_flag=voiced_flag,
        activity_intervals=activity_intervals,
        duration_seconds=duration_sec,
        num_frames=num_frames,
    )

    # Provenance metadata
    audio_sha256 = (
        audio_input.sha256
        if audio_input is not None
        else hashlib.sha256(mono_signal.tobytes()).hexdigest()
    )
    artifact_id = (
        audio_input.artifact_id
        if audio_input is not None
        else f"feat-{audio_sha256[:12]}"
    )
    source_id = audio_input.source_id if audio_input is not None else "direct_audio"
    sample_id = audio_input.sample_id if audio_input is not None else None
    role = audio_input.role.value if audio_input is not None else AudioRole.REFERENCE.value

    # Config fingerprint
    cfg_items = f"{feat_cfg.frame_ms}_{feat_cfg.hop_ms}_{feat_cfg.n_mfcc}_{feat_cfg.pitch_backend}_{feat_cfg.f0_min_hz}_{feat_cfg.f0_max_hz}"
    config_fingerprint = hashlib.sha256(cfg_items.encode("utf-8")).hexdigest()[:16]

    software = collect_software_versions()

    metadata = FeatureMetadata(
        schema_version=FEATURE_SCHEMA_VERSION,
        artifact_id=artifact_id,
        source_id=source_id,
        sample_id=sample_id,
        role=role,
        audio_sha256=audio_sha256,
        sample_rate=sr,
        duration_seconds=float(duration_sec),
        num_channels=num_channels,
        channel_policy=feat_cfg.channel_policy,
        frame_ms=feat_cfg.frame_ms,
        hop_ms=feat_cfg.hop_ms,
        n_mfcc=feat_cfg.n_mfcc,
        spectral_rolloff_p=feat_cfg.spectral_rolloff_p,
        pitch_backend=feat_cfg.pitch_backend,
        f0_min_hz=feat_cfg.f0_min_hz,
        f0_max_hz=feat_cfg.f0_max_hz,
        silence_threshold_db=feat_cfg.silence_threshold_db,
        min_silence_ms=feat_cfg.min_silence_ms,
        config_fingerprint=config_fingerprint,
        software=software,
        created_utc=datetime.now(timezone.utc).isoformat(),
        warnings=[],
    )

    return FeatureResult(
        metadata=metadata,
        frame_schema=frame_schema,
        rms=rms_arr,
        rms_db=rms_db_arr,
        zcr=zcr_arr,
        spectral_centroid=centroid_arr,
        spectral_bandwidth=bandwidth_arr,
        spectral_rolloff=rolloff_arr,
        spectral_flatness=flatness_arr,
        f0_hz=f0_hz,
        voiced_flag=voiced_flag,
        pitch_confidence=pitch_confidence,
        mfcc=mfcc_arr,
        activity_intervals=activity_intervals,
        summary_stats=summary_stats,
    )
