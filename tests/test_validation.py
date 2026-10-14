from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flawbench.config import AudioSettings, Config
from flawbench.exceptions import (
    AudioFileAccessError,
    EmptyInputError,
    FileTooLargeError,
    InvalidAudioError,
    UnsupportedFormatError,
)
from flawbench.validation import (
    validate_extension,
    validate_file_path,
    validate_samples,
    validate_size,
    validate_stream_properties,
)


@pytest.fixture
def audio(config: Config) -> AudioSettings:
    return config.audio


def test_extension_is_case_insensitive(audio: AudioSettings) -> None:
    assert validate_extension("Take1.WAV", audio) == ".wav"


@pytest.mark.parametrize("name", ["notes.txt", "noextension", "clip.exe", "archive.wav.zip"])
def test_unsupported_extensions(audio: AudioSettings, name: str) -> None:
    with pytest.raises(UnsupportedFormatError):
        validate_extension(name, audio)


def test_size_rules(audio: AudioSettings) -> None:
    validate_size(1, audio)
    validate_size(audio.max_upload_bytes, audio)
    with pytest.raises(EmptyInputError):
        validate_size(0, audio)
    with pytest.raises(FileTooLargeError):
        validate_size(audio.max_upload_bytes + 1, audio)


def test_file_path_checks(audio: AudioSettings, tmp_path: Path) -> None:
    with pytest.raises(AudioFileAccessError):
        validate_file_path(tmp_path / "missing.wav", audio)
    with pytest.raises(AudioFileAccessError):
        validate_file_path(tmp_path, audio)  # a directory
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(EmptyInputError):
        validate_file_path(empty, audio)
    ok = tmp_path / "ok.wav"
    ok.write_bytes(b"12345")
    assert validate_file_path(ok, audio) == 5  # size only; contents are not inspected here


def test_user_message_hides_paths_but_detail_keeps_them(
    audio: AudioSettings, tmp_path: Path
) -> None:
    with pytest.raises(AudioFileAccessError) as info:
        validate_file_path(tmp_path / "secret_dir.wav", audio)
    assert str(tmp_path) not in info.value.public_message()
    assert str(tmp_path) in info.value.public_message(expose_details=True)


@pytest.mark.parametrize(
    ("rate", "channels", "frames"),
    [(16000, 1, 0), (4000, 1, 100), (16000, 0, 100), (16000, 3, 100)],
)
def test_invalid_stream_properties(
    audio: AudioSettings, rate: int, channels: int, frames: int
) -> None:
    with pytest.raises(InvalidAudioError):
        validate_stream_properties(rate, channels, frames, audio)


def test_valid_stream_properties(audio: AudioSettings) -> None:
    validate_stream_properties(44100, 2, 10, audio)


def test_non_finite_samples_policy(audio: AudioSettings) -> None:
    bad = np.array([[0.0, np.nan, 0.1, np.inf]], dtype=np.float32)
    with pytest.raises(InvalidAudioError, match="non-finite"):
        validate_samples(bad, audio)
    lenient = replace(audio, reject_non_finite_samples=False)
    validate_samples(bad, lenient)
    validate_samples(np.zeros((1, 4), dtype=np.float32), audio)
