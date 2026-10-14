"""Tests for Phase 3 reference comparison and differential time-series analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flawbench.exceptions import FeatureComparisonError
from flawbench.features.comparison import ComparisonResult, compare_features
from flawbench.features.extractor import extract_features
from flawbench.features.schema import AudioInput, AudioRole


def _sine(freq: float, dur: float, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_comparison_identical_audio_has_low_distance() -> None:
    """Comparing identical audio produces near-zero DTW cost and ~0 delta RMS."""
    sr = 16000
    y = _sine(freq=300.0, dur=0.6, sr=sr)

    ref_feat = extract_features(y, sample_rate=sr)
    var_feat = extract_features(y, sample_rate=sr)

    cmp_res = compare_features(ref_feat, var_feat)

    assert cmp_res.dtw_cost < 0.05
    assert abs(cmp_res.global_metrics["mean_delta_rms_db"]) < 0.1
    assert abs(cmp_res.global_metrics["duration_ratio"] - 1.0) < 0.01
    assert len(cmp_res.candidate_regions) == 0


def test_comparison_energy_reduction_detected() -> None:
    """Attenuating an audio segment creates negative delta RMS and energy_reduction candidate."""
    sr = 16000
    # Reference: 1.0s tone
    ref = _sine(freq=300.0, dur=1.0, sr=sr, amp=0.6)
    # Variant: attenuated in the middle 0.5s by -12 dB (gain = 0.25)
    var = ref.copy()
    mid_start = int(0.3 * sr)
    mid_end = int(0.8 * sr)
    var[mid_start:mid_end] *= 0.25

    ref_feat = extract_features(ref, sample_rate=sr)
    var_feat = extract_features(var, sample_rate=sr)

    cmp_res = compare_features(ref_feat, var_feat)

    assert cmp_res.global_metrics["min_delta_rms_db"] < -6.0
    # Should detect energy_reduction candidate
    er_candidates = [c for c in cmp_res.candidate_regions if c["flaw_hypothesis"] == "energy_reduction"]
    assert len(er_candidates) >= 1
    c = er_candidates[0]
    assert c["var_start_sec"] >= 0.2 and c["var_end_sec"] <= 0.9


def test_comparison_pause_insertion_detected() -> None:
    """Inserting a silent pause creates pause candidate region."""
    sr = 16000
    # Reference: 0.6s tone
    part1 = _sine(freq=350.0, dur=0.3, sr=sr)
    part2 = _sine(freq=350.0, dur=0.3, sr=sr)
    ref = np.concatenate([part1, part2])

    # Variant: insert 0.3s of silence in between
    pause = np.zeros(int(0.3 * sr), dtype=np.float32)
    var = np.concatenate([part1, pause, part2])

    ref_feat = extract_features(ref, sample_rate=sr)
    var_feat = extract_features(var, sample_rate=sr)

    cmp_res = compare_features(ref_feat, var_feat)

    assert cmp_res.global_metrics["duration_ratio"] > 1.3
    # Check for pause candidate
    pause_candidates = [c for c in cmp_res.candidate_regions if c["flaw_hypothesis"] == "pause_insertion"]
    assert len(pause_candidates) >= 1


def test_comparison_save_and_load(tmp_path: Path) -> None:
    """ComparisonResult saves to .npz and loads back accurately."""
    sr = 16000
    y1 = _sine(250.0, 0.4, sr)
    y2 = _sine(250.0, 0.4, sr, amp=0.2)

    ref_feat = extract_features(y1, sample_rate=sr)
    var_feat = extract_features(y2, sample_rate=sr)
    original = compare_features(ref_feat, var_feat)

    save_file = tmp_path / "comparison.npz"
    original.save(save_file)
    assert save_file.is_file()

    loaded = ComparisonResult.load(save_file)
    assert loaded.ref_artifact_id == original.ref_artifact_id
    assert loaded.variant_artifact_id == original.variant_artifact_id
    assert abs(loaded.dtw_cost - original.dtw_cost) < 1e-5
    assert np.allclose(loaded.delta_rms_db, original.delta_rms_db)
