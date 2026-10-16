from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from flawbench.exceptions import ParticipantInputError
from flawbench.participant.playback import (
    excerpt_bounds,
    participant_excerpt,
    reference_excerpt,
    slice_excerpt,
    to_wav_bytes,
)
from tests.explain_fixtures import ENERGY_EVIDENCE, PAUSE_EVIDENCE, bundle


def test_margin_is_applied_in_samples() -> None:
    b = excerpt_bounds(16000, 32000, sample_rate=16000, total_samples=160000, margin_sec=0.5)
    assert (b.start_sample, b.end_sample) == (8000, 40000)
    assert (b.region_start_sample, b.region_end_sample) == (16000, 32000)
    assert not b.clamped_start and not b.clamped_end


def test_clamped_to_recording_bounds() -> None:
    b = excerpt_bounds(100, 159900, sample_rate=16000, total_samples=160000, margin_sec=2.0)
    assert (b.start_sample, b.end_sample) == (0, 160000)
    assert b.clamped_start and b.clamped_end
    assert b.region_start_sample == 100 and b.region_end_sample == 159900


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100])
def test_sample_rates_scale_margin(rate: int) -> None:
    b = excerpt_bounds(rate, 2 * rate, sample_rate=rate, total_samples=10 * rate, margin_sec=1.0)
    assert (b.start_sample, b.end_sample) == (0, 3 * rate)
    assert b.duration_sec == pytest.approx(3.0)


def test_max_excerpt_cap_applies() -> None:
    b = excerpt_bounds(0, 16000 * 50, sample_rate=16000, total_samples=16000 * 60,
                       margin_sec=1.0, max_excerpt_sec=10.0)  # fmt: skip
    assert b.duration_sec == pytest.approx(10.0)


def test_invalid_inputs_rejected() -> None:
    for kwargs in (
        dict(start_sample=5, end_sample=1, sample_rate=16000, total_samples=100, margin_sec=0.0),
        dict(start_sample=0, end_sample=1, sample_rate=16000, total_samples=0, margin_sec=0.0),
        dict(start_sample=0, end_sample=1, sample_rate=16000, total_samples=100, margin_sec=-1.0),
    ):
        with pytest.raises(ParticipantInputError):
            excerpt_bounds(**kwargs)  # type: ignore[arg-type]


def test_excerpt_matches_selected_interval_and_does_not_modify_source() -> None:
    samples = np.arange(2 * 32000, dtype=np.float32).reshape(2, 32000) / 64000
    original = samples.copy()
    b = participant_excerpt(bundle("energy_reduction", ENERGY_EVIDENCE, start=8000, end=16000),
                            total_samples=32000, margin_sec=0.25, max_excerpt_sec=30.0)  # fmt: skip
    clip = slice_excerpt(samples, b)
    assert clip.shape == (2, 16000)
    assert np.array_equal(clip, original[:, 4000:20000])
    clip[:] = 0
    assert np.array_equal(samples, original)


def test_reference_anchor_gives_contextual_window_not_zero_length_clip() -> None:
    bd = bundle("pause_insertion", PAUSE_EVIDENCE, ref=(2.0, 2.0))
    b = reference_excerpt(bd, reference_sample_rate=16000, reference_total_samples=96000,
                          margin_sec=1.0, max_excerpt_sec=30.0)  # fmt: skip
    assert b is not None and (b.start_sample, b.end_sample) == (16000, 48000)
    assert b.region_start_sample == b.region_end_sample == 32000


def test_reference_interval_uses_reference_sample_rate() -> None:
    bd = bundle("energy_reduction", ENERGY_EVIDENCE, ref=(1.0, 2.0))
    b = reference_excerpt(bd, reference_sample_rate=22050, reference_total_samples=220500,
                          margin_sec=0.0, max_excerpt_sec=30.0)  # fmt: skip
    assert b is not None and (b.start_sample, b.end_sample) == (22050, 44100)


def test_missing_reference_correspondence_returns_none() -> None:
    bd = bundle("energy_reduction", ENERGY_EVIDENCE, ref=None)
    assert reference_excerpt(bd, reference_sample_rate=16000, reference_total_samples=16000,
                             margin_sec=1.0, max_excerpt_sec=30.0) is None  # fmt: skip


def test_wav_encoding_round_trips_without_quantization() -> None:
    x = (np.sin(np.linspace(0, 20, 4000)).astype(np.float32) * 0.3)[None, :]
    data, sr = sf.read(io.BytesIO(to_wav_bytes(x, 22050)), dtype="float32", always_2d=True)
    assert sr == 22050 and np.array_equal(data.T, x)
