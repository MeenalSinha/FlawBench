"""Tests for Phase 3 acoustic feature extraction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from flawbench.config import load_config
from flawbench.exceptions import FeatureExtractionError
from flawbench.features.extractor import FeatureResult, extract_features
from flawbench.features.schema import FEATURE_SCHEMA_VERSION


def _generate_tone(
    freq_hz: float = 220.0,
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a clean single-frequency sine tone."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_extract_features_framing_and_shapes() -> None:
    """Canonical frame grid geometry matches sample count and hop size."""
    sr = 16000
    duration = 1.25  # 1.25s = 20,000 samples
    y = _generate_tone(freq_hz=440.0, duration_sec=duration, sample_rate=sr)

    cfg = load_config()
    feat_cfg = cfg.features

    res = extract_features(y, sample_rate=sr, feat_cfg=feat_cfg)

    frame_length = int(round(sr * (feat_cfg.frame_ms / 1000.0)))  # 25 ms = 400 samples
    hop_length = int(round(sr * (feat_cfg.hop_ms / 1000.0)))  # 10 ms = 160 samples
    expected_frames = int(math.ceil(len(y) / hop_length))  # ceil(20000 / 160) = 125

    assert frame_length == 400
    assert res.num_frames == expected_frames
    assert res.rms.shape == (expected_frames,)
    assert res.rms_db.shape == (expected_frames,)
    assert res.zcr.shape == (expected_frames,)
    assert res.spectral_centroid.shape == (expected_frames,)
    assert res.spectral_bandwidth.shape == (expected_frames,)
    assert res.spectral_rolloff.shape == (expected_frames,)
    assert res.spectral_flatness.shape == (expected_frames,)
    assert res.f0_hz.shape == (expected_frames,)
    assert res.voiced_flag.shape == (expected_frames,)
    assert res.pitch_confidence.shape == (expected_frames,)
    assert res.mfcc.shape == (expected_frames, feat_cfg.n_mfcc)

    # FrameSchema checks
    fs = res.frame_schema
    assert fs.num_frames == expected_frames
    assert fs.start_samples[0] == 0
    assert fs.start_samples[-1] == (expected_frames - 1) * hop_length
    assert fs.end_samples[-1] <= len(y)
    assert np.all(fs.end_samples >= fs.start_samples)
    assert np.all(fs.valid_samples > 0)
    assert len(res.center_times_sec) == expected_frames
    assert res.center_times_sec[-1] <= duration


def test_extract_pitch_accuracy_on_known_tone() -> None:
    """YIN pitch estimation accurately identifies the fundamental frequency."""
    sr = 16000
    target_f0 = 220.0
    y = _generate_tone(freq_hz=target_f0, duration_sec=0.8, sample_rate=sr, amplitude=0.4)

    res = extract_features(y, sample_rate=sr)

    # In steady tone portion (middle 50% of frames), pitch should be very close to 220 Hz
    mid_start = res.num_frames // 4
    mid_end = 3 * res.num_frames // 4
    voiced_mid = res.voiced_flag[mid_start:mid_end]
    assert np.mean(voiced_mid) > 0.8  # majority voiced

    est_f0 = res.f0_hz[mid_start:mid_end]
    valid_f0 = est_f0[np.isfinite(est_f0)]
    assert len(valid_f0) > 0
    mean_f0 = np.mean(valid_f0)
    assert abs(mean_f0 - target_f0) < 5.0, f"Expected {target_f0} Hz, got {mean_f0:.1f} Hz"


def test_silence_detection_and_activity_intervals() -> None:
    """Silence has low RMS energy, unvoiced flag, and creates silent intervals."""
    sr = 16000
    # 0.5s tone, then 0.5s silence
    tone = _generate_tone(freq_hz=300.0, duration_sec=0.5, sample_rate=sr, amplitude=0.3)
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    y = np.concatenate([tone, silence])

    res = extract_features(y, sample_rate=sr)

    # First half should have higher energy than second half
    mid_frame = res.num_frames // 2
    first_half_mean_db = np.mean(res.rms_db[:mid_frame])
    second_half_mean_db = np.mean(res.rms_db[mid_frame:])
    assert first_half_mean_db > second_half_mean_db + 20.0  # at least 20 dB difference

    # Activity intervals should segment into active and silent
    assert len(res.activity_intervals) >= 2
    has_active = any(i.is_active for i in res.activity_intervals)
    has_silent = any(not i.is_active for i in res.activity_intervals)
    assert has_active and has_silent


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """FeatureResult saves to .npz and .json and loads back identically."""
    sr = 16000
    y = _generate_tone(freq_hz=440.0, duration_sec=0.5, sample_rate=sr)
    original = extract_features(y, sample_rate=sr)

    npz_path, json_path = original.save(tmp_path)
    assert npz_path.is_file()
    assert json_path.is_file()

    loaded = FeatureResult.load(npz_path, json_path)

    assert loaded.num_frames == original.num_frames
    assert np.allclose(loaded.rms, original.rms)
    assert np.allclose(loaded.rms_db, original.rms_db)
    assert np.allclose(loaded.zcr, original.zcr)
    assert np.allclose(loaded.spectral_centroid, original.spectral_centroid)
    assert np.allclose(loaded.mfcc, original.mfcc)
    assert np.array_equal(loaded.voiced_flag, original.voiced_flag)
    assert len(loaded.activity_intervals) == len(original.activity_intervals)
    assert loaded.metadata.schema_version == FEATURE_SCHEMA_VERSION


def test_ground_truth_fields_never_present() -> None:
    """Feature metadata and results must never store ground-truth flaw fields."""
    y = _generate_tone(duration_sec=0.2, sample_rate=16000)
    res = extract_features(y, sample_rate=16000)
    meta_dict = res.metadata.to_dict()

    forbidden_fields = {
        "flaw_type",
        "severity",
        "output_interval",
        "source_interval",
        "source_anchor",
        "timeline",
    }
    assert not (set(meta_dict.keys()) & forbidden_fields)


def test_empty_audio_raises_error() -> None:
    """Attempting to extract features from 0 samples raises FeatureExtractionError."""
    with pytest.raises(FeatureExtractionError, match="0 samples"):
        extract_features(np.array([], dtype=np.float32), sample_rate=16000)
