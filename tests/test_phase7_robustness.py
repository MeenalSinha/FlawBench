"""Phase 7 robustness tests (SYNTHETIC signals; they check behavior, not speech accuracy).

Expected behaviour is stated per test. Warnings are errors: numerical problems must be handled
explicitly, never emitted and ignored.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from flawbench.config import Config
from flawbench.detection.engine import DetectionEngine
from flawbench.detection.inputs import build_detection_input_from_features
from flawbench.features.extractor import extract_features

pytestmark = pytest.mark.filterwarnings("error")


def _tone(sr: int, dur: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return (amp * (np.sin(2 * np.pi * 220 * t) + 0.4 * np.sin(2 * np.pi * 440 * t))).astype(
        np.float32
    )


def _run(config: Config, ref: np.ndarray, var: np.ndarray, sr: int):  # type: ignore[no-untyped-def]
    rf = extract_features(ref, sample_rate=sr, config=config)
    vf = extract_features(var, sample_rate=sr, config=config)
    inp = build_detection_input_from_features(rf, vf, settings=config.detection)
    return DetectionEngine(config.detection).detect(inp), vf


@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100])
def test_identical_audio_at_each_sample_rate_gives_no_detection_and_correct_timeline(
    config: Config, sr: int
) -> None:
    audio = _tone(sr, 3.0)
    result, vf = _run(config, audio, audio.copy(), sr)
    assert result.predictions == []
    assert vf.center_times_sec[-1] <= 3.0 + 1e-6
    assert vf.center_times_sec[-1] >= 3.0 - 0.1  # last frame centre within 100 ms of the end


@pytest.mark.parametrize("dur", [0.2, 0.5, 1.0])
def test_short_recordings_do_not_crash_and_stay_in_bounds(config: Config, dur: float) -> None:
    sr = 16000
    ref = _tone(sr, dur)
    var = np.concatenate(
        [ref[: len(ref) // 2], np.zeros(sr // 4, np.float32), ref[len(ref) // 2 :]]
    )
    result, _ = _run(config, ref, var, sr)
    for p in result.predictions:
        assert 0 <= p.start_sample < p.end_sample <= result.total_samples


def test_silent_recordings_yield_zero_candidates_not_errors(config: Config) -> None:
    silent = np.zeros(16000 * 2, dtype=np.float32)
    result, vf = _run(config, silent, silent.copy(), 16000)
    assert result.predictions == []
    assert not np.isfinite(vf.f0_hz).any() or np.isnan(vf.f0_hz).any()  # no fabricated pitch


def test_constant_and_very_low_level_audio_are_handled_without_warnings(config: Config) -> None:
    sr = 16000
    for sig in (np.full(sr * 2, 0.1, np.float32), _tone(sr, 2.0, amp=1e-4)):
        result, _ = _run(config, sig, sig.copy(), sr)
        assert result.predictions == []


def test_missing_f0_is_never_fabricated(config: Config) -> None:
    rng = np.random.default_rng(0)
    noise = (0.05 * rng.standard_normal(16000 * 2)).astype(np.float32)
    vf = extract_features(noise, sample_rate=16000, config=config)
    voiced = np.asarray(vf.voiced_flag, dtype=bool)
    assert np.isnan(vf.f0_hz[~voiced]).all()  # unvoiced frames carry NaN, never an invented pitch


def test_no_numerical_warnings_for_pause_on_silent_reference(config: Config) -> None:
    sr = 16000
    ref = np.zeros(sr * 2, dtype=np.float32)
    var = _tone(sr, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result, _ = _run(config, ref, var, sr)
    assert isinstance(result.predictions, list)


@pytest.mark.parametrize(("sr", "expected"), [(8000, 96), (16000, 128), (44100, 128)])
def test_mel_band_count_avoids_empty_filters_without_changing_higher_rates(
    sr: int, expected: int
) -> None:
    from flawbench.features.extractor import mel_bands_for

    assert mel_bands_for(sr, int(round(sr * 0.025))) == expected
