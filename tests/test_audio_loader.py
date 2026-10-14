from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from flawbench.audio.loader import load_audio, probe_audio
from flawbench.config import Config, load_config
from flawbench.exceptions import (
    AudioDecodingError,
    AudioFileAccessError,
    EmptyInputError,
    FileTooLargeError,
    InvalidAudioError,
    UnsupportedFormatError,
)

MakeTone = Callable[..., Path]


def test_load_mono_wav_shape_rate_and_metadata(config: Config, make_tone: MakeTone) -> None:
    path = make_tone(sample_rate=16000, seconds=0.5)
    audio = load_audio(path, config.audio)
    assert audio.samples.shape == (1, 8000)
    assert audio.samples.dtype == np.float32
    assert audio.samples.flags["C_CONTIGUOUS"]
    assert audio.sample_rate == 16000
    meta = audio.metadata
    assert (meta.channels, meta.frames, meta.container_format) == (1, 8000, "WAV")
    assert meta.duration_seconds == pytest.approx(0.5)
    assert meta.file_size_bytes == path.stat().st_size
    assert meta.filename == "tone.wav"
    assert audio.duration_seconds == pytest.approx(0.5)


def test_samples_are_not_normalized_or_resampled(config: Config, make_tone: MakeTone) -> None:
    original = (0.25 * np.sin(np.linspace(0, 20, 4410))).astype(np.float32)
    path = make_tone("quiet.wav", sample_rate=44100, data=original, subtype="FLOAT")
    audio = load_audio(path, config.audio)
    assert audio.sample_rate == 44100  # native rate preserved
    np.testing.assert_array_equal(audio.samples[0], original)  # amplitude untouched
    assert np.abs(audio.samples).max() == pytest.approx(0.25, abs=1e-3)


def test_pcm16_is_scaled_to_unit_range(config: Config, make_tone: MakeTone) -> None:
    ints = np.array([0, 16384, -16384, 32767], dtype=np.int16).repeat(500)
    path = make_tone("pcm.wav", data=ints, subtype="PCM_16")
    samples = load_audio(path, config.audio).samples[0]
    assert samples.max() < 1.0 and samples.min() >= -1.0
    assert samples[500] == pytest.approx(0.5, abs=1e-4)


def test_stereo_layout_and_explicit_mono_downmix(config: Config, make_tone: MakeTone) -> None:
    left = np.full(1600, 0.5, dtype=np.float32)
    right = np.full(1600, -0.1, dtype=np.float32)
    path = make_tone("st.wav", data=np.stack([left, right], axis=1), subtype="FLOAT")
    audio = load_audio(path, config.audio)
    assert audio.samples.shape == (2, 1600)
    np.testing.assert_allclose(audio.samples[0], 0.5)
    np.testing.assert_allclose(audio.samples[1], -0.1)
    mono = audio.mono()
    assert mono.shape == (1600,)
    np.testing.assert_allclose(mono, 0.2, atol=1e-6)
    assert audio.samples.shape == (2, 1600)  # downmix did not mutate the loaded audio


def test_flac_round_trip(config: Config, make_tone: MakeTone) -> None:
    path = make_tone("clip.flac", sample_rate=22050, seconds=0.3, subtype="PCM_16")
    audio = load_audio(path, config.audio)
    assert audio.metadata.container_format == "FLAC"
    assert audio.sample_rate == 22050


def test_probe_reads_header_only_and_matches_load(config: Config, make_tone: MakeTone) -> None:
    path = make_tone(seconds=1.0)
    assert probe_audio(path, config.audio) == load_audio(path, config.audio).metadata


def test_bytes_and_stream_sources(config: Config, make_tone: MakeTone) -> None:
    path = make_tone()
    data = path.read_bytes()
    from_bytes = load_audio(data, config.audio, filename="upload.wav")
    from_stream = load_audio(io.BytesIO(data), config.audio, filename="upload.wav")
    np.testing.assert_array_equal(from_bytes.samples, from_stream.samples)
    assert from_bytes.metadata.file_size_bytes == len(data)
    with pytest.raises(UnsupportedFormatError, match="filename"):
        load_audio(data, config.audio)


def test_undecodable_content_with_audio_extension(config: Config, tmp_path: Path) -> None:
    fake = tmp_path / "fake.wav"
    fake.write_bytes(b"this is definitely not audio" * 10)
    with pytest.raises(AudioDecodingError) as info:
        load_audio(fake, config.audio)
    assert str(tmp_path) not in info.value.public_message()


def test_truncated_file_is_rejected(config: Config, make_tone: MakeTone, tmp_path: Path) -> None:
    good = make_tone("full.flac", seconds=1.0, subtype="PCM_16").read_bytes()
    cut = tmp_path / "cut.flac"
    cut.write_bytes(good[: len(good) // 2])
    with pytest.raises(AudioDecodingError):
        load_audio(cut, config.audio)


def test_content_must_match_an_allowed_format(
    project_root: Path, make_tone: MakeTone, tmp_path: Path
) -> None:
    wav_only = load_config(
        project_root=project_root, env={"FLAWBENCH_AUDIO_EXTENSIONS": ".wav"}
    ).audio
    flac_bytes = make_tone("real.flac").read_bytes()
    disguised = tmp_path / "disguised.wav"
    disguised.write_bytes(flac_bytes)
    with pytest.raises(UnsupportedFormatError, match="content"):
        load_audio(disguised, wav_only)


def test_unsupported_extension_missing_and_empty(config: Config, tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("hello")
    with pytest.raises(UnsupportedFormatError):
        load_audio(txt, config.audio)
    with pytest.raises(AudioFileAccessError):
        load_audio(tmp_path / "gone.wav", config.audio)
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(EmptyInputError):
        load_audio(empty, config.audio)
    with pytest.raises(EmptyInputError):
        load_audio(b"", config.audio, filename="x.wav")


def test_size_limit_is_enforced_before_decoding(project_root: Path, make_tone: MakeTone) -> None:
    tiny = load_config(project_root=project_root, env={"FLAWBENCH_MAX_UPLOAD_MB": "1"}).audio
    big = make_tone("big.wav", sample_rate=16000, seconds=40.0, subtype="PCM_16")  # ~1.2 MB
    assert big.stat().st_size > tiny.max_upload_bytes
    with pytest.raises(FileTooLargeError):
        load_audio(big, tiny)
    with pytest.raises(FileTooLargeError):
        probe_audio(big.read_bytes(), tiny, filename="big.wav")


def test_sample_rate_and_channel_rules(project_root: Path, make_tone: MakeTone) -> None:
    strict = load_config(
        project_root=project_root,
        env={"FLAWBENCH_MIN_SAMPLE_RATE_HZ": "16000", "FLAWBENCH_MAX_CHANNELS": "1"},
    ).audio
    with pytest.raises(InvalidAudioError, match="sample rate"):
        load_audio(make_tone("lo.wav", sample_rate=8000), strict)
    with pytest.raises(InvalidAudioError, match="channel"):
        load_audio(make_tone("st.wav", channels=2), strict)


def test_non_finite_samples_rejected_or_allowed_by_config(
    config: Config, make_tone: MakeTone
) -> None:
    data = np.zeros(1600, dtype=np.float32)
    data[10] = np.nan
    data[20] = np.inf
    path = make_tone("nan.wav", data=data, subtype="FLOAT")
    with pytest.raises(InvalidAudioError, match="non-finite"):
        load_audio(path, config.audio)
    lenient = replace(config.audio, reject_non_finite_samples=False)
    assert np.isnan(load_audio(path, lenient).samples).any()


def test_zero_length_audio_is_rejected(config: Config, tmp_path: Path) -> None:
    path = tmp_path / "zero.wav"
    sf.write(path, np.zeros(0, dtype=np.float32), 16000)
    with pytest.raises(InvalidAudioError, match="no samples"):
        load_audio(path, config.audio)
