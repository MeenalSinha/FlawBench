"""Injector unit tests on SYNTHETIC arrays (test fixtures, not speech)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import SYN_RATE, synthetic_signal
from flawbench.dataset import injectors
from flawbench.exceptions import InjectionError

SR = SYN_RATE


@pytest.fixture
def audio() -> np.ndarray:
    return synthetic_signal(4.0, seed=2)[None, :].copy()


def sine(freq: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :]


def dominant_hz(x: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(x[0]))
    return float(np.fft.rfftfreq(x.shape[1], 1 / SR)[np.argmax(spectrum)])


# ------------------------------------------------------------------------- pause
def test_pause_exact_samples_and_annotation(audio: np.ndarray) -> None:
    n, at, d = audio.shape[1], 20000, 8000
    before = audio.copy()
    r = injectors.insert_pause(audio, at_sample=at, silence_samples=d)
    out = r.samples
    assert out.shape == (1, n + d) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:, :at], audio[:, :at])
    assert not out[:, at : at + d].any()
    np.testing.assert_array_equal(out[:, at + d :], audio[:, at:])
    assert r.output_interval == (at, at + d)
    assert r.source_anchor == at and r.source_interval is None
    assert r.effective_parameters == {"silence_samples": d}
    np.testing.assert_array_equal(audio, before)  # input untouched


def test_pause_boundaries_are_deliberately_allowed(audio: np.ndarray) -> None:
    n = audio.shape[1]
    head = injectors.insert_pause(audio, at_sample=0, silence_samples=100)
    tail = injectors.insert_pause(audio, at_sample=n, silence_samples=100)
    assert head.output_interval == (0, 100) and tail.output_interval == (n, n + 100)
    np.testing.assert_array_equal(tail.samples[:, :n], audio)


@pytest.mark.parametrize("at, d", [(-1, 10), (10**9, 10), (100, 0), (100, -5)])
def test_pause_invalid_requests(audio: np.ndarray, at: int, d: int) -> None:
    with pytest.raises(InjectionError):
        injectors.insert_pause(audio, at_sample=at, silence_samples=d)


def test_pause_is_deterministic_and_multichannel(audio: np.ndarray) -> None:
    stereo = np.concatenate([audio, audio * 0.5])
    a = injectors.insert_pause(stereo, at_sample=500, silence_samples=50).samples
    b = injectors.insert_pause(stereo, at_sample=500, silence_samples=50).samples
    np.testing.assert_array_equal(a, b)
    assert a.shape[0] == 2 and not a[:, 500:550].any()


def test_rejects_bad_input_arrays() -> None:
    with pytest.raises(InjectionError):
        injectors.insert_pause(np.zeros(10, dtype=np.float32), at_sample=0, silence_samples=1)
    with pytest.raises(InjectionError):
        injectors.insert_pause(np.zeros((1, 10), dtype=np.float64), at_sample=0, silence_samples=1)


# ------------------------------------------------------------------------- tempo
@pytest.mark.parametrize("rate", [0.5, 0.7, 1.5, 2.0])
def test_tempo_annotation_matches_actual_output(audio: np.ndarray, rate: float) -> None:
    start, end = 16000, 48000
    before = audio.copy()
    r = injectors.change_tempo(audio, start=start, end=end, rate=rate)
    eff = r.effective_parameters
    measured = eff["output_segment_samples"]
    assert r.output_interval == (start, start + measured)
    assert r.source_interval == (start, end)
    assert r.samples.shape[1] == audio.shape[1] - (end - start) + measured
    # Measured, not assumed: within a few percent of the nominal stretch, and effective_rate agrees.
    assert measured == pytest.approx((end - start) / rate, rel=0.001)
    assert eff["effective_rate"] == pytest.approx((end - start) / measured)
    assert r.implementation["name"] == "audiotsm.wsola"
    np.testing.assert_array_equal(audio, before)


def test_tempo_slow_down_lengthens_and_speed_up_shortens(audio: np.ndarray) -> None:
    slow = injectors.change_tempo(audio, start=0, end=32000, rate=0.5)
    fast = injectors.change_tempo(audio, start=0, end=32000, rate=2.0)
    assert slow.samples.shape[1] > audio.shape[1] > fast.samples.shape[1]


def test_tempo_preserves_audio_outside_interval(audio: np.ndarray) -> None:
    start, end = 10000, 30000
    r = injectors.change_tempo(audio, start=start, end=end, rate=0.7)
    o_end = r.output_interval[1]
    np.testing.assert_array_equal(r.samples[:, :start], audio[:, :start])
    np.testing.assert_array_equal(r.samples[:, o_end:], audio[:, end:])


def test_tempo_is_pitch_preserving() -> None:
    x = sine(220.0, 3.0)
    r = injectors.change_tempo(x, start=8000, end=40000, rate=0.7)
    s, e = r.output_interval
    mid = r.samples[:, s + 2000 : e - 2000]
    assert abs(dominant_hz(mid) - 220.0) < 4.0
    assert np.isfinite(r.samples).all()


def test_tempo_is_repeatable(audio: np.ndarray) -> None:
    a = injectors.change_tempo(audio, start=1000, end=30000, rate=0.8).samples
    b = injectors.change_tempo(audio, start=1000, end=30000, rate=0.8).samples
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_tempo_invalid_rate(audio: np.ndarray, rate: float) -> None:
    with pytest.raises(InjectionError):
        injectors.change_tempo(audio, start=0, end=20000, rate=rate)


@pytest.mark.parametrize("start, end", [(-1, 100), (100, 100), (200, 100), (0, 10**9)])
def test_tempo_invalid_interval(audio: np.ndarray, start: int, end: int) -> None:
    with pytest.raises(InjectionError):
        injectors.change_tempo(audio, start=start, end=end, rate=0.8)


# ------------------------------------------------------------------------ energy
def test_energy_attenuation_exact(audio: np.ndarray) -> None:
    start, end, gain = 8000, 24000, -6.0
    before = audio.copy()
    r = injectors.reduce_energy(audio, start=start, end=end, gain_db=gain)
    mult = r.effective_parameters["amplitude_multiplier"]
    assert mult == pytest.approx(10 ** (gain / 20), rel=1e-6)
    assert r.effective_parameters["applied_gain_db"] == pytest.approx(gain, abs=1e-4)
    assert r.samples.shape == audio.shape
    np.testing.assert_array_equal(r.samples[:, :start], audio[:, :start])
    np.testing.assert_array_equal(r.samples[:, end:], audio[:, end:])
    np.testing.assert_allclose(r.samples[:, start:end], audio[:, start:end] * np.float32(mult))
    assert r.output_interval == r.source_interval == (start, end)
    np.testing.assert_array_equal(audio, before)


def test_energy_does_not_normalize_peak(audio: np.ndarray) -> None:
    r = injectors.reduce_energy(audio, start=0, end=audio.shape[1], gain_db=-20.0)
    assert np.abs(r.samples).max() == pytest.approx(np.abs(audio).max() * 0.1, rel=1e-4)


@pytest.mark.parametrize("gain", [0.0, 3.0, float("nan"), float("-inf")])
def test_energy_invalid_gain(audio: np.ndarray, gain: float) -> None:
    with pytest.raises(InjectionError):
        injectors.reduce_energy(audio, start=0, end=100, gain_db=gain)


def test_energy_invalid_interval_and_non_finite_input(audio: np.ndarray) -> None:
    with pytest.raises(InjectionError):
        injectors.reduce_energy(audio, start=100, end=50, gain_db=-6)
    bad = audio.copy()
    bad[0, 10] = np.nan
    with pytest.raises(InjectionError, match="non-finite"):
        injectors.reduce_energy(bad, start=0, end=100, gain_db=-6)
