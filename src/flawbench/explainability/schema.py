"""Phase 6 explanation schema.

Data flow (no ground-truth annotation appears anywhere in this chain)::

    DetectionResult -> EvidenceBundle (validated facts) -> ExplanationCard -> ParticipantAnalysis

Interval convention: variant (participant) intervals are half-open sample ranges
``[start_sample, end_sample)``; integer samples are authoritative and seconds are derived from
the recording's own sample rate. Reference correspondence is expressed in reference-timeline
seconds only, and is kept strictly separate from the participant interval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EXPLANATION_SCHEMA_VERSION = "6.0"

# Fact roles
ROLE_MEASUREMENT = "measurement"
ROLE_REFERENCE = "reference"
ROLE_VARIANT = "variant"
ROLE_DIFFERENCE = "difference"
ROLE_THRESHOLD = "threshold"
ROLE_CONTEXT = "context"

# Reference correspondence status
REF_INTERVAL = "interval"
REF_ANCHOR = "anchor"
REF_UNAVAILABLE = "unavailable"

# Overall analysis status
STATUS_COMPLETED = "completed"
STATUS_NO_CANDIDATES = "completed_no_candidates"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

# Stage status
STAGE_COMPLETED = "completed"
STAGE_FAILED = "failed"
STAGE_SKIPPED = "skipped"

CARD_OK = "ok"
CARD_VALIDATION_FAILED = "validation_failed"

STRENGTH_MEANING = (
    "Detector evidence strength: an uncalibrated heuristic score scaled from zero to one that "
    "combines the detector's own measurements. It is not a probability and not a confidence."
)


@dataclass(frozen=True)
class EvidenceFact:
    """One validated measurement or threshold taken from real detector or comparison output."""

    key: str
    label: str
    value: float | None
    unit: str
    role: str
    source: str
    definition: str
    valid: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VariantInterval:
    """Participant-timeline interval, half-open ``[start_sample, end_sample)``."""

    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def start_sec(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_sec(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def duration_sec(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "sample_rate": self.sample_rate,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": self.duration_sec,
            "convention": "half_open_samples",
        }


@dataclass(frozen=True)
class ReferenceCorrespondence:
    """Where the detector's DTW comparison places this event on the reference timeline."""

    status: str  # interval | anchor | unavailable
    start_sec: float | None = None
    end_sec: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceBundle:
    """Validated evidence for a single prediction. The only input templates may read."""

    prediction_id: str
    flaw_type: str
    detector_id: str
    variant: VariantInterval
    reference: ReferenceCorrespondence
    facts: tuple[EvidenceFact, ...]
    evidence_strength: float
    decision_threshold: float | None
    detector_warnings: tuple[str, ...] = ()
    alignment_warnings: tuple[str, ...] = ()
    was_merged: bool = False
    ignored_evidence_keys: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    categorical: dict[str, str] = field(default_factory=dict)

    def fact(self, key: str) -> EvidenceFact | None:
        for f in self.facts:
            if f.key == key:
                return f
        return None

    def valid_value(self, key: str) -> float | None:
        f = self.fact(key)
        return f.value if f is not None and f.valid and f.value is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "flaw_type": self.flaw_type,
            "detector_id": self.detector_id,
            "variant_interval": self.variant.to_dict(),
            "reference_correspondence": self.reference.to_dict(),
            "facts": [f.to_dict() for f in self.facts],
            "evidence_strength": self.evidence_strength,
            "evidence_strength_meaning": STRENGTH_MEANING,
            "decision_threshold": self.decision_threshold,
            "detector_warnings": list(self.detector_warnings),
            "alignment_warnings": list(self.alignment_warnings),
            "was_merged": self.was_merged,
            "ignored_evidence_keys": list(self.ignored_evidence_keys),
            "categorical_evidence": dict(self.categorical),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class NumericClaim:
    """A number that appears in explanation prose, with the fact it was rendered from."""

    fact_key: str
    rendered: str
    value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExplanationCard:
    """Deterministic, evidence-linked explanation for one prediction."""

    prediction_id: str
    flaw_type: str
    title: str
    observation: str
    evidence_lines: tuple[str, ...]
    interpretation: str
    suggested_review: str
    limitations: tuple[str, ...]
    warnings: tuple[str, ...]
    detector_notes: tuple[str, ...]
    numeric_claims: tuple[NumericClaim, ...]
    missing_facts: tuple[str, ...]
    bundle: EvidenceBundle
    status: str = CARD_OK
    validation_problems: tuple[str, ...] = ()

    def prose_fields(self) -> list[str]:
        """Template-generated text that must contain only traceable numbers."""
        return [
            self.title,
            self.observation,
            *self.evidence_lines,
            self.interpretation,
            self.suggested_review,
            *self.limitations,
            *self.warnings,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "flaw_type": self.flaw_type,
            "title": self.title,
            "observation": self.observation,
            "evidence": list(self.evidence_lines),
            "interpretation": self.interpretation,
            "suggested_review": self.suggested_review,
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "detector_notes": list(self.detector_notes),
            "numeric_claims": [c.to_dict() for c in self.numeric_claims],
            "missing_facts": list(self.missing_facts),
            "status": self.status,
            "validation_problems": list(self.validation_problems),
            "evidence_bundle": self.bundle.to_dict(),
        }


@dataclass(frozen=True)
class RecordingInfo:
    role: str  # reference | participant
    origin: str  # registered_reference | registered_variant | uploaded
    label: str
    source_id: str | None
    sample_id: str | None
    filename: str | None
    sha256: str
    sample_rate: int
    channels: int
    duration_seconds: float
    container_format: str | None = None
    subtype: str | None = None
    is_synthetic_fixture: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageRecord:
    name: str
    status: str
    message: str = ""
    duration_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TranscriptCheck:
    """Text-level comparison only. No audio-text alignment backend exists in FlawBench."""

    performed: bool
    reference_available: bool
    participant_supplied: bool
    similarity: float | None = None
    reference_word_count: int | None = None
    participant_word_count: int | None = None
    differences: tuple[str, ...] = ()
    below_warning_threshold: bool = False
    alignment_available: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["differences"] = list(self.differences)
        return d


@dataclass(frozen=True)
class BenchmarkContext:
    """Aggregate detector performance from a saved evaluation run (never per-prediction)."""

    available: bool
    run_id: str | None = None
    split: str | None = None
    is_synthetic_only: bool | None = None
    sample_count: int | None = None
    per_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParticipantAnalysis:
    """Complete, serializable record of one participant analysis session."""

    session_id: str
    created_utc: str
    mode: str
    status: str
    reference: RecordingInfo | None
    participant: RecordingInfo | None
    stages: list[StageRecord]
    supported_categories: list[str]
    categories_completed: list[str]
    categories_failed: dict[str, str]
    cards: list[ExplanationCard]
    warnings: list[str]
    limitations: list[str]
    transcript: TranscriptCheck | None
    alignment: dict[str, Any]
    detection: dict[str, Any]
    provenance: dict[str, Any]
    sort_policy: str
    failure: str | None = None
    benchmark: BenchmarkContext | None = None
    schema_version: str = EXPLANATION_SCHEMA_VERSION

    @property
    def prediction_count(self) -> int:
        return len(self.cards)

    @property
    def categories_detected(self) -> list[str]:
        return sorted({c.flaw_type for c in self.cards})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_utc": self.created_utc,
            "mode": self.mode,
            "status": self.status,
            "failure": self.failure,
            "reference": self.reference.to_dict() if self.reference else None,
            "participant": self.participant.to_dict() if self.participant else None,
            "stages": [s.to_dict() for s in self.stages],
            "supported_categories": list(self.supported_categories),
            "categories_completed": list(self.categories_completed),
            "categories_failed": dict(self.categories_failed),
            "prediction_count": self.prediction_count,
            "categories_detected": self.categories_detected,
            "sort_policy": self.sort_policy,
            "cards": [c.to_dict() for c in self.cards],
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "alignment": dict(self.alignment),
            "detection": dict(self.detection),
            "provenance": dict(self.provenance),
            "benchmark": self.benchmark.to_dict() if self.benchmark else None,
        }
