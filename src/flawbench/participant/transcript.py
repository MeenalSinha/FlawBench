"""Text-level transcript comparison. There is no audio-text alignment backend in FlawBench, so
this reports text similarity only and never produces word timestamps."""

from __future__ import annotations

import difflib
import re

from flawbench.exceptions import ParticipantInputError
from flawbench.explainability.schema import TranscriptCheck

_WORD = re.compile(r"[\w']+", re.UNICODE)
MAX_DIFFERENCES = 8


def read_transcript_bytes(data: bytes, filename: str, allowed_formats: tuple[str, ...]) -> str:
    """Decode an uploaded transcript, enforcing the configured formats and UTF-8 text."""
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in allowed_formats:
        raise ParticipantInputError(
            f"Transcript format {suffix or '(none)'} is not enabled.",
            detail=f"allowed: {', '.join(allowed_formats)}",
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParticipantInputError("Transcript must be UTF-8 text.", detail=str(exc)) from exc


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def check_transcript(
    reference_text: str | None, participant_text: str | None, warning_threshold: float
) -> TranscriptCheck:
    ref_ok = bool(reference_text and reference_text.strip())
    part_ok = bool(participant_text and participant_text.strip())
    if not (ref_ok and part_ok):
        note = (
            "No transcript comparison was performed: "
            + ("the reference has no transcript. " if not ref_ok else "")
            + ("No participant transcript was supplied." if not part_ok else "")
        ).strip()
        return TranscriptCheck(False, ref_ok, part_ok, note=note)
    a, b = _words(reference_text or ""), _words(participant_text or "")
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    diffs: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or len(diffs) >= MAX_DIFFERENCES:
            continue
        diffs.append(
            f"{tag}: reference '{' '.join(a[i1:i2])}' / participant '{' '.join(b[j1:j2])}'"
        )
    similarity = matcher.ratio()
    return TranscriptCheck(
        performed=True,
        reference_available=True,
        participant_supplied=True,
        similarity=similarity,
        reference_word_count=len(a),
        participant_word_count=len(b),
        differences=tuple(diffs),
        below_warning_threshold=similarity < warning_threshold,
        alignment_available=False,
        note="Word-sequence similarity only. No audio-text alignment or word timestamps exist.",
    )
