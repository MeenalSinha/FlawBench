"""Audio probing and loading.

Array convention (documented contract for all later phases):

* ``LoadedAudio.samples`` has shape ``(num_channels, num_samples)``, dtype ``float32``,
  C-contiguous, with integer PCM scaled to ``[-1.0, 1.0)`` by the decoder.
* Mono audio is shape ``(1, num_samples)``; use :meth:`LoadedAudio.mono` for an explicit,
  opt-in downmix to shape ``(num_samples,)``.
* Nothing is resampled, normalized, trimmed, or downmixed implicitly. ``sample_rate`` is the
  rate stored in the file.

Accepted sources: a path, raw ``bytes``, or a binary file-like object. For non-path sources,
``filename`` is required so the extension rule can be applied.
"""

from __future__ import annotations

import io
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from flawbench.config import AudioSettings
from flawbench.exceptions import (
    AudioDecodingError,
    AudioFileAccessError,
    UnsupportedFormatError,
)
from flawbench.validation import (
    validate_extension,
    validate_file_path,
    validate_samples,
    validate_size,
    validate_stream_properties,
)

AudioSource = Path | str | bytes | BinaryIO

SAMPLE_DTYPE = np.float32


@dataclass(frozen=True)
class AudioMetadata:
    """Facts about an audio input, taken from the decoder's header information."""

    filename: str
    container_format: str
    subtype: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    file_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded audio plus its metadata. See the module docstring for the array contract."""

    samples: NDArray[np.float32]
    sample_rate: int
    metadata: AudioMetadata

    @property
    def num_channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return self.num_channels

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / self.sample_rate

    def mono(self) -> NDArray[np.float32]:
        """Explicit downmix: mean over channels, shape ``(num_samples,)``."""
        return np.ascontiguousarray(self.samples.mean(axis=0, dtype=np.float32))


def _prepare(
    source: AudioSource, filename: str | None, settings: AudioSettings
) -> tuple[Path | BinaryIO, str, int]:
    """Validate cheap claims (existence, extension, size) and return (handle, name, size)."""
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        size = validate_file_path(path, settings)
        return path, path.name, size
    if isinstance(source, (bytes, bytearray)):
        if filename is None:
            raise UnsupportedFormatError("A filename is required to validate in-memory audio.")
        validate_extension(filename, settings)
        validate_size(len(source), settings)
        return io.BytesIO(bytes(source)), filename, len(source)
    if filename is None:
        raise UnsupportedFormatError("A filename is required to validate stream input.")
    validate_extension(filename, settings)
    stream: BinaryIO = source
    try:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(0)
    except (OSError, ValueError) as exc:
        raise AudioFileAccessError("The audio stream could not be read.", detail=str(exc)) from exc
    validate_size(size, settings)
    return stream, filename, size


def _info(handle: Path | BinaryIO) -> sf._SoundFileInfo:
    try:
        return sf.info(handle)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioDecodingError(
            "The file could not be decoded as audio.", detail=str(exc)
        ) from exc
    except OSError as exc:
        raise AudioFileAccessError("The audio file could not be read.", detail=str(exc)) from exc


def _metadata(
    info: sf._SoundFileInfo, name: str, size: int, settings: AudioSettings
) -> AudioMetadata:
    if info.format not in settings.allowed_formats:
        raise UnsupportedFormatError(
            "The file content is not in an allowed audio format.",
            detail=f"detected format {info.format}",
        )
    validate_stream_properties(info.samplerate, info.channels, info.frames, settings)
    return AudioMetadata(
        filename=name,
        container_format=info.format,
        subtype=info.subtype,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        duration_seconds=info.frames / info.samplerate,
        file_size_bytes=size,
    )


def probe_audio(
    source: AudioSource, settings: AudioSettings, *, filename: str | None = None
) -> AudioMetadata:
    """Validate and describe audio by reading only its header (no sample data is loaded)."""
    handle, name, size = _prepare(source, filename, settings)
    return _metadata(_info(handle), name, size, settings)


def load_audio(
    source: AudioSource, settings: AudioSettings, *, filename: str | None = None
) -> LoadedAudio:
    """Validate, decode, and return audio with its metadata.

    Raises:
        AudioInputError: (or a subclass) for any rejected or undecodable input.
    """
    handle, name, size = _prepare(source, filename, settings)
    metadata = _metadata(_info(handle), name, size, settings)
    if not isinstance(handle, Path):
        handle.seek(0)
    try:
        data, rate = sf.read(handle, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioDecodingError("The audio data could not be decoded.", detail=str(exc)) from exc
    except OSError as exc:
        raise AudioFileAccessError("The audio file could not be read.", detail=str(exc)) from exc
    if rate != metadata.sample_rate or data.shape[0] != metadata.frames:
        raise AudioDecodingError(
            "The audio appears truncated or corrupt.",
            detail=f"header frames={metadata.frames}, decoded={data.shape[0]}",
        )
    samples = np.ascontiguousarray(data.T, dtype=SAMPLE_DTYPE)
    validate_samples(samples, settings)
    return LoadedAudio(samples=samples, sample_rate=int(rate), metadata=metadata)
