"""Unit tests for Phase 4 flaw detection algorithms."""

from __future__ import annotations

import numpy as np

from flawbench.config import Config
from flawbench.detection.detectors import (
    EnergyReductionDetector,
    PauseDetector,
    TempoDetector,
)
from flawbench.detection.inputs import build_detection_input_from_features
from flawbench.features.extractor import extract_features


def _tone(freq: float, dur: float, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    """Generate a clean synthetic tone with speech-like harmonics."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # Fundamental + harmonics for rich MFCC alignment
    sig = amp * (
        0.6 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(4 * np.pi * freq * t)
        + 0.1 * np.sin(6 * np.pi * freq * t)
    )
    return sig.astype(np.float32)


def test_pause_detector_detects_inserted_pause(config: Config) -> None:
    sr = 16000
    # Reference: 1.0s tone
    part1 = _tone(freq=300.0, dur=0.5, sr=sr)
    part2 = _tone(freq=400.0, dur=0.5, sr=sr)
    ref = np.concatenate([part1, part2])

    # Variant: 0.4s silence inserted between part1 and part2
    silence = np.zeros(int(0.4 * sr), dtype=np.float32)
    var = np.concatenate([part1, silence, part2])

    ref_feat = extract_features(ref, sample_rate=sr, config=config)
    var_feat = extract_features(var, sample_rate=sr, config=config)

    inp = build_detection_input_from_features(ref_feat, var_feat)
    detector = PauseDetector(config.detection)
    predictions = detector.detect(inp)

    assert len(predictions) >= 1
    p = predictions[0]
    assert p.flaw_type == "pause_insertion"
    # Localized around 0.5s to 0.9s on variant timeline
    assert 0.4 <= p.start_sec <= 0.6
    assert 0.8 <= p.end_sec <= 1.0
    assert p.evidence["duration_difference_sec"] >= 0.25
    assert p.evidence_strength > 0.0
    assert p.start_sample < p.end_sample


def test_pause_detector_natural_silence_not_flagged(config: Config) -> None:
    """Natural silence present in both reference and variant is not detected as an injected pause."""
    sr = 16000
    part1 = _tone(freq=300.0, dur=0.4, sr=sr)
    silence = np.zeros(int(0.3 * sr), dtype=np.float32)
    part2 = _tone(freq=400.0, dur=0.4, sr=sr)
    both = np.concatenate([part1, silence, part2])

    ref_feat = extract_features(both, sample_rate=sr, config=config)
    var_feat = extract_features(both, sample_rate=sr, config=config)

    inp = build_detection_input_from_features(ref_feat, var_feat)
    detector = PauseDetector(config.detection)
    predictions = detector.detect(inp)

    # Identical silence duration means duration difference == 0, so no flaw detected!
    assert len(predictions) == 0


def _speech_like(dur: float, sr: int = 16000) -> np.ndarray:
    """Generate dynamic audio with time-varying harmonic structure (simulating formant transitions)."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    phase = 2 * np.pi * (180.0 * t + 120.0 * (t**2) / max(1e-4, dur))
    sig = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    return sig.astype(np.float32)


def test_tempo_detector_detects_slower_delivery(config: Config) -> None:
    sr = 16000
    ref_dur = 0.8
    var_dur = 1.4  # stretched, slower delivery (rate ~ 0.57)

    ref = _speech_like(dur=ref_dur, sr=sr)
    var = _speech_like(dur=var_dur, sr=sr)

    ref_feat = extract_features(ref, sample_rate=sr, config=config)
    var_feat = extract_features(var, sample_rate=sr, config=config)

    inp = build_detection_input_from_features(ref_feat, var_feat)
    detector = TempoDetector(config.detection)
    predictions = detector.detect(inp)

    assert len(predictions) >= 1
    p = predictions[0]
    assert p.flaw_type == "tempo_change"
    assert p.evidence["direction"] == "slower"
    assert p.evidence["estimated_relative_rate"] < 0.85
    assert p.start_sec < p.end_sec


def test_tempo_detector_detects_faster_delivery(config: Config) -> None:
    sr = 16000
    ref_dur = 1.2
    var_dur = 0.6  # compressed, faster delivery (rate ~ 2.0)

    ref = _speech_like(dur=ref_dur, sr=sr)
    var = _speech_like(dur=var_dur, sr=sr)

    ref_feat = extract_features(ref, sample_rate=sr, config=config)
    var_feat = extract_features(var, sample_rate=sr, config=config)

    inp = build_detection_input_from_features(ref_feat, var_feat)
    detector = TempoDetector(config.detection)
    predictions = detector.detect(inp)

    assert len(predictions) >= 1
    p = predictions[0]
    assert p.flaw_type == "tempo_change"
    assert p.evidence["direction"] == "faster"
    assert p.evidence["estimated_relative_rate"] > 1.15


def test_energy_reduction_detector(config: Config) -> None:
    sr = 16000
    ref = _tone(freq=350.0, dur=1.0, sr=sr, amp=0.6)
    var = ref.copy()
    # Attenuate 0.3s to 0.8s by -12 dB (factor 0.25)
    s = int(0.3 * sr)
    e = int(0.8 * sr)
    var[s:e] *= 0.25

    ref_feat = extract_features(ref, sample_rate=sr, config=config)
    var_feat = extract_features(var, sample_rate=sr, config=config)

    inp = build_detection_input_from_features(ref_feat, var_feat)
    detector = EnergyReductionDetector(config.detection)
    predictions = detector.detect(inp)

    assert len(predictions) >= 1
    p = predictions[0]
    assert p.flaw_type == "energy_reduction"
    assert p.evidence["mean_delta_rms_db"] < -6.0
    assert 0.2 <= p.start_sec <= 0.4
    assert 0.7 <= p.end_sec <= 0.9
    assert p.evidence_strength > 0.0


def test_identical_audio_yields_zero_detections(config: Config) -> None:
    sr = 16000
    sig = _tone(freq=300.0, dur=0.8, sr=sr)

    ref_feat = extract_features(sig, sample_rate=sr, config=config)
    var_feat = extract_features(sig, sample_rate=sr, config=config)

    inp = build_detection_input_from_features(ref_feat, var_feat)

    p_det = PauseDetector(config.detection).detect(inp)
    t_det = TempoDetector(config.detection).detect(inp)
    e_det = EnergyReductionDetector(config.detection).detect(inp)

    assert len(p_det) == 0
    assert len(t_det) == 0
    assert len(e_det) == 0


def test_inserted_pause_with_frozen_reference_yields_valid_result(config: Config) -> None:
    """Regression: a pause that holds the DTW reference position fixed also triggers the energy
    detector; its zero-length reference span used to make the whole result fail validation."""
    from flawbench.detection.engine import DetectionEngine
    from flawbench.detection.validator import DetectionValidator

    sr = 16000
    part1 = _tone(freq=300.0, dur=0.5, sr=sr)
    part2 = _tone(freq=400.0, dur=0.5, sr=sr)
    ref = np.concatenate([part1, part2])
    var = np.concatenate([part1, np.zeros(int(0.4 * sr), dtype=np.float32), part2])
    inp = build_detection_input_from_features(
        extract_features(ref, sample_rate=sr, config=config),
        extract_features(var, sample_rate=sr, config=config),
    )
    result = DetectionEngine(config.detection).detect(inp)
    assert DetectionValidator().validate_result(result) == []
    for p in result.predictions:
        if p.flaw_type != "pause_insertion" and p.ref_start_sec is not None:
            assert p.ref_end_sec is not None and p.ref_end_sec > p.ref_start_sec
