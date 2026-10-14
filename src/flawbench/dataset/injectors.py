"""Pure audio transformations. Each takes an independent ``(channels, samples)`` float32 array,
never mutates it, and returns the new audio together with exact, sample-based annotations.

Annotations are measured from the produced signal, not predicted from a duration formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

import numpy as np
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter
from numpy.typing import NDArray

from flawbench.exceptions import InjectionError

Samples = NDArray[np.float32]

TEMPO_IMPLEMENTATION = "audiotsm.wsola"
#: Zero samples appended to a segment before WSOLA so its tail is not truncated by the algorithm's
#: analysis window; the output is then cut to the segment's nominal stretched length. Algorithmic
#: constant, not a runtime setting.
_WSOLA_TAIL_PAD = 4096


@dataclass(frozen=True)
class InjectionResult:
    samples: Samples
    source_interval: tuple[int, int] | None  # [start, end) in reference samples
    source_anchor: int | None  # reference sample index for point flaws (pause)
    output_interval: tuple[int, int]  # [start, end) in generated samples
    effective_parameters: dict[str, Any]
    implementation: dict[str, str] = field(default_factory=dict)


def _check_input(samples: Samples) -> None:
    if samples.ndim != 2 or samples.shape[1] == 0 or samples.dtype != np.float32:
        raise InjectionError("Audio must be a non-empty float32 array shaped (channels, samples).")


def _check_finite(out: Samples, what: str) -> None:
    if not np.isfinite(out).all():
        raise InjectionError(f"{what} produced non-finite samples.")


def insert_pause(samples: Samples, *, at_sample: int, silence_samples: int) -> InjectionResult:
    """Insert ``silence_samples`` of digital silence before reference sample ``at_sample``.

    ``0 <= at_sample <= n``: 0 prepends and ``n`` appends silence (deliberately allowed; such
    placements are not speech pauses and callers decide whether they are meaningful).
    Output interval is ``[at_sample, at_sample + silence_samples)`` on the generated timeline.
    """
    _check_input(samples)
    channels, n = samples.shape
    if not (0 <= at_sample <= n):
        raise InjectionError(f"Insertion point {at_sample} is outside [0, {n}] samples.")
    if silence_samples < 1:
        raise InjectionError("Silence must be at least one sample long.")
    silence = np.zeros((channels, silence_samples), dtype=np.float32)
    out = np.concatenate([samples[:, :at_sample], silence, samples[:, at_sample:]], axis=1)
    return InjectionResult(
        samples=out,
        source_interval=None,
        source_anchor=at_sample,
        output_interval=(at_sample, at_sample + silence_samples),
        effective_parameters={"silence_samples": int(silence_samples)},
    )


def change_tempo(samples: Samples, *, start: int, end: int, rate: float) -> InjectionResult:
    """Time-stretch reference ``[start, end)`` by ``rate`` (pitch-preserving WSOLA).

    ``rate < 1`` slows (longer segment), ``rate > 1`` speeds up. Audio outside the interval is
    copied unchanged and shifted. The output interval is the *measured* stretched segment.
    """
    _check_input(samples)
    channels, n = samples.shape
    if not (0 <= start < end <= n):
        raise InjectionError(f"Interval [{start}, {end}) is outside [0, {n}] samples.")
    if not (np.isfinite(rate) and rate > 0):
        raise InjectionError("Tempo rate must be positive and finite.")
    segment = np.ascontiguousarray(samples[:, start:end])
    padded = np.concatenate([segment, np.zeros((channels, _WSOLA_TAIL_PAD), np.float32)], axis=1)
    writer = ArrayWriter(channels)
    wsola(channels, speed=float(rate)).run(ArrayReader(padded), writer)
    stretched = np.asarray(writer.data, dtype=np.float32)
    nominal = round((end - start) / rate)
    if stretched.shape[1] < nominal:
        raise InjectionError(
            "Time-stretch produced a shorter segment than expected; segment too short."
        )
    stretched = np.ascontiguousarray(stretched[:, :nominal])
    if stretched.shape[1] < 1:
        raise InjectionError("Time-stretch produced an empty segment.")
    _check_finite(stretched, "Time-stretch")
    out = np.concatenate([samples[:, :start], stretched, samples[:, end:]], axis=1)
    measured = int(stretched.shape[1])
    return InjectionResult(
        samples=out,
        source_interval=(start, end),
        source_anchor=None,
        output_interval=(start, start + measured),
        effective_parameters={
            "requested_rate": float(rate),
            "source_segment_samples": int(end - start),
            "output_segment_samples": measured,
            "effective_rate": (end - start) / measured,
            "output_peak_abs": float(np.abs(stretched).max()),
        },
        implementation={
            "name": TEMPO_IMPLEMENTATION,
            "version": metadata.version("audiotsm"),
            "pitch_preserving": "true",
            "notes": "library default WSOLA; segment zero-padded, cut to nominal length",
        },
    )


def reduce_energy(samples: Samples, *, start: int, end: int, gain_db: float) -> InjectionResult:
    """Scale reference ``[start, end)`` by ``10**(gain_db/20)``; duration is unchanged.

    No normalization is applied. This changes signal amplitude only; it is not a perceptual
    loudness model.
    """
    _check_input(samples)
    n = samples.shape[1]
    if not (0 <= start < end <= n):
        raise InjectionError(f"Interval [{start}, {end}) is outside [0, {n}] samples.")
    if not (np.isfinite(gain_db) and gain_db < 0):
        raise InjectionError("Gain must be a finite negative number of decibels.")
    multiplier = np.float32(10.0 ** (gain_db / 20.0))
    out = samples.copy()
    out[:, start:end] = samples[:, start:end] * multiplier
    _check_finite(out, "Attenuation")
    return InjectionResult(
        samples=out,
        source_interval=(start, end),
        source_anchor=None,
        output_interval=(start, end),
        effective_parameters={
            "requested_gain_db": float(gain_db),
            "amplitude_multiplier": float(multiplier),
            "applied_gain_db": float(20.0 * np.log10(float(multiplier))),
        },
    )
