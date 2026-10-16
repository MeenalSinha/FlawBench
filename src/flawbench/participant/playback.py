"""Interval-to-audio mapping for excerpt playback.

Integer sample indices are authoritative. Margins are converted to whole samples with the
recording's own sample rate and clamped to the real recording length. Excerpts are copies; the
source arrays are never modified.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from flawbench.exceptions import ParticipantInputError
from flawbench.explainability.schema import REF_ANCHOR, REF_INTERVAL, EvidenceBundle


@dataclass(frozen=True)
class ExcerptBounds:
    """Half-open sample range of an excerpt and of the flagged region inside it."""

    start_sample: int
    end_sample: int
    region_start_sample: int
    region_end_sample: int
    sample_rate: int
    clamped_start: bool
    clamped_end: bool

    @property
    def duration_sec(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


def excerpt_bounds(
    start_sample: int,
    end_sample: int,
    *,
    sample_rate: int,
    total_samples: int,
    margin_sec: float,
    max_excerpt_sec: float | None = None,
) -> ExcerptBounds:
    """Region plus margin, clamped to ``[0, total_samples)``; honours a maximum excerpt length."""
    if sample_rate <= 0 or total_samples <= 0:
        raise ParticipantInputError("Cannot build an excerpt from an empty recording.")
    if end_sample < start_sample or margin_sec < 0:
        raise ParticipantInputError("Invalid excerpt interval or margin.")
    margin = int(round(margin_sec * sample_rate))
    lo, hi = start_sample - margin, end_sample + margin
    clamped_start, clamped_end = lo < 0, hi > total_samples
    lo, hi = max(0, lo), min(total_samples, hi)
    r0, r1 = max(lo, min(start_sample, hi)), max(lo, min(end_sample, hi))
    if max_excerpt_sec is not None:
        cap = int(round(max_excerpt_sec * sample_rate))
        if hi - lo > cap:
            hi = lo + cap
            r1 = min(r1, hi)
            r0 = min(r0, hi)
            clamped_end = True
    if hi <= lo:
        raise ParticipantInputError("The excerpt lies outside the recording.")
    return ExcerptBounds(lo, hi, r0, r1, sample_rate, clamped_start, clamped_end)


def slice_excerpt(samples: NDArray[np.float32], bounds: ExcerptBounds) -> NDArray[np.float32]:
    """Copy of ``samples`` (shape ``(channels, n)``) for ``bounds``."""
    return np.array(samples[:, bounds.start_sample : bounds.end_sample], copy=True)


def participant_excerpt(
    bundle: EvidenceBundle, *, total_samples: int, margin_sec: float, max_excerpt_sec: float
) -> ExcerptBounds:
    v = bundle.variant
    return excerpt_bounds(
        v.start_sample,
        v.end_sample,
        sample_rate=v.sample_rate,
        total_samples=total_samples,
        margin_sec=margin_sec,
        max_excerpt_sec=max_excerpt_sec,
    )


def reference_excerpt(
    bundle: EvidenceBundle,
    *,
    reference_sample_rate: int,
    reference_total_samples: int,
    margin_sec: float,
    max_excerpt_sec: float,
) -> ExcerptBounds | None:
    """Reference excerpt, or None when no reliable correspondence exists.

    An anchor (zero-length reference span) yields a contextual window around the anchor point
    rather than an invalid zero-length clip; the flagged region inside it is empty.
    """
    ref = bundle.reference
    if ref.status not in {REF_INTERVAL, REF_ANCHOR} or ref.start_sec is None:
        return None
    sr = reference_sample_rate
    start = int(np.floor(ref.start_sec * sr))
    end = start if ref.status == REF_ANCHOR else int(np.ceil((ref.end_sec or ref.start_sec) * sr))
    start = min(start, reference_total_samples)
    end = min(max(end, start), reference_total_samples)
    return excerpt_bounds(
        start,
        end,
        sample_rate=sr,
        total_samples=reference_total_samples,
        margin_sec=margin_sec,
        max_excerpt_sec=max_excerpt_sec,
    )


def to_wav_bytes(samples: NDArray[np.float32], sample_rate: int) -> bytes:
    """Encode float samples ``(channels, n)`` as 32-bit float WAV without quantization."""
    buf = io.BytesIO()
    sf.write(buf, samples.T, sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()
