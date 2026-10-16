"""Evidence-fact extraction: FlawPrediction + real comparison context -> validated EvidenceBundle.

Nothing is invented here. Every numeric fact is copied from the prediction's own ``evidence``
dictionary (or from its validated interval) and checked for type and finiteness. Evidence keys the
detectors do not document are listed as ignored rather than interpreted. Measurements are never
reinterpreted across detectors: each detector has its own specification below, taken from the
Phase 4 implementation (``flawbench.detection.detectors``).

Conventions inherited from Phase 3/4 (verified in code):

* ``delta_rms_db = participant_rms_db - reference_rms_db`` (negative means the participant is
  quieter).
* ``local_slope = participant_time_step / reference_time_step``; the tempo detector reports
  ``estimated_relative_rate = 1 / mean_slope``. A rate below one means the participant covers the
  same reference content more slowly; above one means faster.
* A pause insertion that has no counterpart in the reference has ``ref_start_sec ==
  ref_end_sec`` (a zero-length reference span), which is represented as an anchor point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from flawbench.detection.schema import DetectionResult, FlawPrediction
from flawbench.explainability.schema import (
    REF_ANCHOR,
    REF_INTERVAL,
    REF_UNAVAILABLE,
    ROLE_CONTEXT,
    ROLE_DIFFERENCE,
    ROLE_MEASUREMENT,
    ROLE_REFERENCE,
    ROLE_THRESHOLD,
    ROLE_VARIANT,
    EvidenceBundle,
    EvidenceFact,
    ReferenceCorrespondence,
    VariantInterval,
)

SRC_EVIDENCE = "prediction.evidence"
SRC_PREDICTION = "prediction"
SRC_COMPARISON = "comparison"

CONSISTENCY_ABS_TOL = 0.05  # rounding slack for cross-field checks (detectors round to 2-4 dp)
CONSISTENCY_REL_TOL = 0.02


@dataclass(frozen=True)
class FactSpec:
    key: str
    label: str
    unit: str
    role: str
    definition: str


FACT_SPECS: dict[str, tuple[FactSpec, ...]] = {
    "pause_insertion": (
        FactSpec(
            "variant_pause_duration_sec",
            "participant pause duration",
            "s",
            ROLE_VARIANT,
            "Length of the flagged region on the participant timeline, from DTW-aligned frame "
            "times.",
        ),
        FactSpec(
            "reference_duration_sec",
            "aligned reference duration",
            "s",
            ROLE_REFERENCE,
            "Length of the DTW-matched reference span; zero when the reference position did "
            "not advance during the participant pause.",
        ),
        FactSpec(
            "duration_difference_sec",
            "duration difference (participant minus reference)",
            "s",
            ROLE_DIFFERENCE,
            "Participant pause duration minus aligned reference duration.",
        ),
        FactSpec(
            "mean_variant_rms_db",
            "mean participant RMS level",
            "dB",
            ROLE_VARIANT,
            "Mean frame RMS level (dB relative to full scale) of the participant over the region.",
        ),
        FactSpec(
            "threshold_silence_db",
            "silence threshold",
            "dB",
            ROLE_THRESHOLD,
            "Frame RMS level at or below which a participant frame counts as silent.",
        ),
        FactSpec(
            "min_duration_sec",
            "minimum pause duration",
            "s",
            ROLE_THRESHOLD,
            "Configured minimum duration for a pause candidate.",
        ),
        FactSpec(
            "ref_frozen_fraction",
            "reference-frozen fraction",
            "fraction",
            ROLE_CONTEXT,
            "Fraction of alignment steps in the region where the reference position did not "
            "advance.",
        ),
    ),
    "tempo_change": (
        FactSpec(
            "estimated_relative_rate",
            "estimated relative rate",
            "ratio",
            ROLE_MEASUREMENT,
            "Estimated progression rate of the participant relative to the reference "
            "(reference time covered per unit of participant time); below one is slower than "
            "the reference, above one is faster.",
        ),
        FactSpec(
            "mean_warp_slope",
            "mean warp slope",
            "ratio",
            ROLE_MEASUREMENT,
            "Mean participant time elapsed per unit of reference time along the DTW path "
            "(excluding pause-like steps).",
        ),
        FactSpec(
            "rate_deviation",
            "deviation from equal pace",
            "ratio",
            ROLE_DIFFERENCE,
            "Absolute difference between the estimated relative rate and equal pace.",
        ),
        FactSpec(
            "threshold_min_deviation",
            "minimum deviation threshold",
            "ratio",
            ROLE_THRESHOLD,
            "Configured minimum deviation from equal pace for a tempo candidate.",
        ),
        FactSpec(
            "duration_sec",
            "detector region length",
            "s",
            ROLE_MEASUREMENT,
            "Length of the detector's region on the participant timeline before consolidation.",
        ),
    ),
    "energy_reduction": (
        FactSpec(
            "mean_delta_rms_db",
            "mean RMS level difference (participant minus reference)",
            "dB",
            ROLE_DIFFERENCE,
            "Mean over the region of participant RMS dB minus aligned reference RMS dB; "
            "negative means the participant is quieter.",
        ),
        FactSpec(
            "min_delta_rms_db",
            "largest single-frame reduction",
            "dB",
            ROLE_DIFFERENCE,
            "Most negative frame-level RMS dB difference in the region.",
        ),
        FactSpec(
            "mean_ref_rms_db",
            "mean reference RMS level",
            "dB",
            ROLE_REFERENCE,
            "Mean aligned reference frame RMS level (dB relative to full scale).",
        ),
        FactSpec(
            "mean_var_rms_db",
            "mean participant RMS level",
            "dB",
            ROLE_VARIANT,
            "Mean participant frame RMS level (dB relative to full scale).",
        ),
        FactSpec(
            "threshold_gain_db",
            "energy-reduction threshold",
            "dB",
            ROLE_THRESHOLD,
            "Configured (negative) dB difference at or below which a frame qualifies.",
        ),
        FactSpec(
            "duration_sec",
            "detector region length",
            "s",
            ROLE_MEASUREMENT,
            "Length of the detector's region on the participant timeline before consolidation.",
        ),
    ),
}

CATEGORICAL_KEYS: dict[str, tuple[str, ...]] = {"tempo_change": ("direction",)}
SUPPORTED_CATEGORIES: tuple[str, ...] = tuple(FACT_SPECS)


@dataclass(frozen=True)
class AlignmentContext:
    """Measurable properties of the reference/participant pair that bound interpretation."""

    reference_duration_sec: float | None = None
    participant_duration_sec: float | None = None
    normalized_dtw_cost: float | None = None
    duration_ratio_warning: float = 2.0
    transcript_below_threshold: bool = False

    @property
    def duration_ratio(self) -> float | None:
        a, b = self.reference_duration_sec, self.participant_duration_sec
        if not a or not b or a <= 0 or b <= 0:
            return None
        return max(a, b) / min(a, b)

    @property
    def length_mismatch(self) -> bool:
        ratio = self.duration_ratio
        return ratio is not None and ratio > self.duration_ratio_warning

    @property
    def uncertain(self) -> bool:
        return self.length_mismatch or self.transcript_below_threshold


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(CONSISTENCY_ABS_TOL, CONSISTENCY_REL_TOL * max(abs(a), abs(b)))


def _make_fact(spec: FactSpec, raw: Any, *, present: bool) -> EvidenceFact:
    if not present:
        return EvidenceFact(
            spec.key,
            spec.label,
            None,
            spec.unit,
            spec.role,
            SRC_EVIDENCE,
            spec.definition,
            valid=False,
            note="not present in the detector output",
        )
    value = _finite(raw)
    if value is None:
        return EvidenceFact(
            spec.key,
            spec.label,
            None,
            spec.unit,
            spec.role,
            SRC_EVIDENCE,
            spec.definition,
            valid=False,
            note="rejected: value is not a finite number",
        )
    return EvidenceFact(
        spec.key, spec.label, value, spec.unit, spec.role, SRC_EVIDENCE, spec.definition
    )


def _invalidate(facts: dict[str, EvidenceFact], key: str, note: str) -> None:
    f = facts.get(key)
    if f is not None and f.valid:
        facts[key] = EvidenceFact(
            f.key, f.label, None, f.unit, f.role, f.source, f.definition, valid=False, note=note
        )


def _cross_check(flaw_type: str, facts: dict[str, EvidenceFact], cat: dict[str, str]) -> None:
    """Reject facts that contradict each other instead of presenting them side by side."""

    def val(key: str) -> float | None:
        f = facts.get(key)
        return f.value if f is not None and f.valid else None

    if flaw_type == "pause_insertion":
        var, ref, diff = (
            val("variant_pause_duration_sec"),
            val("reference_duration_sec"),
            val("duration_difference_sec"),
        )
        if var is not None and ref is not None and diff is not None and not _close(var - ref, diff):
            _invalidate(
                facts,
                "duration_difference_sec",
                "inconsistent with the participant and reference durations",
            )
    elif flaw_type == "tempo_change":
        rate, slope = val("estimated_relative_rate"), val("mean_warp_slope")
        if rate is not None and slope is not None and slope > 0 and not _close(rate, 1.0 / slope):
            _invalidate(facts, "estimated_relative_rate", "inconsistent with the warp slope")
            _invalidate(facts, "mean_warp_slope", "inconsistent with the relative rate")
        rate = val("estimated_relative_rate")
        direction = cat.get("direction")
        if rate is not None and direction in {"slower", "faster"}:
            if (direction == "slower") != (rate < 1.0):
                cat.pop("direction", None)
                cat["direction_rejected"] = "detector direction label contradicts its own rate"
        elif rate is None and direction in {"slower", "faster"}:
            cat["direction_unverified"] = direction
            cat.pop("direction", None)
    elif flaw_type == "energy_reduction":
        d, r, v = val("mean_delta_rms_db"), val("mean_ref_rms_db"), val("mean_var_rms_db")
        if d is not None and r is not None and v is not None and not _close(v - r, d):
            _invalidate(
                facts, "mean_delta_rms_db", "inconsistent with the mean reference/participant RMS"
            )


def _reference_correspondence(
    pred: FlawPrediction, facts: dict[str, EvidenceFact], context: AlignmentContext
) -> ReferenceCorrespondence:
    r0, r1 = pred.ref_start_sec, pred.ref_end_sec
    if r0 is None or r1 is None:
        return ReferenceCorrespondence(
            REF_UNAVAILABLE, note="The detector produced no reference correspondence."
        )
    if not (math.isfinite(r0) and math.isfinite(r1)) or r0 < 0 or r1 < r0:
        return ReferenceCorrespondence(
            REF_UNAVAILABLE, note="The reference correspondence was invalid and is not used."
        )
    ref_dur = context.reference_duration_sec
    if ref_dur is not None and r1 > ref_dur + 0.05:
        return ReferenceCorrespondence(
            REF_UNAVAILABLE,
            note="The reference correspondence lies beyond the end of the reference recording.",
        )
    zero_span = (r1 - r0) <= 1e-9
    if pred.flaw_type == "pause_insertion" and zero_span:
        return ReferenceCorrespondence(
            REF_ANCHOR,
            r0,
            r0,
            "The participant pause has no counterpart in the reference; it maps to a single "
            "anchor point on the reference timeline.",
        )
    if zero_span:
        return ReferenceCorrespondence(
            REF_UNAVAILABLE, note="The reference span had zero length for a non-pause event."
        )
    return ReferenceCorrespondence(REF_INTERVAL, r0, r1)


def extract_bundle(
    pred: FlawPrediction,
    *,
    sample_rate: int,
    context: AlignmentContext,
    provenance: dict[str, Any] | None = None,
) -> EvidenceBundle:
    """Turn one prediction into a validated evidence bundle."""
    specs = FACT_SPECS.get(pred.flaw_type)
    if specs is None:
        raise ValueError(f"Unsupported flaw type for explanation: {pred.flaw_type!r}")
    evidence = pred.evidence
    facts: dict[str, EvidenceFact] = {
        spec.key: _make_fact(spec, evidence.get(spec.key), present=spec.key in evidence)
        for spec in specs
    }
    categorical: dict[str, str] = {}
    for key in CATEGORICAL_KEYS.get(pred.flaw_type, ()):
        raw = evidence.get(key)
        if isinstance(raw, str) and raw:
            categorical[key] = raw
    _cross_check(pred.flaw_type, facts, categorical)

    variant = VariantInterval(pred.start_sample, pred.end_sample, sample_rate)
    ordered: list[EvidenceFact] = [
        EvidenceFact(
            "variant_start_sec",
            "interval start",
            variant.start_sec,
            "s",
            ROLE_VARIANT,
            SRC_PREDICTION,
            "Start of the flagged interval on the participant timeline (start_sample / "
            "sample rate).",
        ),
        EvidenceFact(
            "variant_end_sec",
            "interval end",
            variant.end_sec,
            "s",
            ROLE_VARIANT,
            SRC_PREDICTION,
            "End (exclusive) of the flagged interval on the participant timeline.",
        ),
        EvidenceFact(
            "variant_duration_sec",
            "interval length",
            variant.duration_sec,
            "s",
            ROLE_VARIANT,
            SRC_PREDICTION,
            "Length of the flagged interval ((end_sample - start_sample) / sample rate).",
        ),
    ]
    reference = _reference_correspondence(pred, facts, context)
    if reference.status in {REF_INTERVAL, REF_ANCHOR} and reference.start_sec is not None:
        label = "reference anchor point" if reference.status == REF_ANCHOR else "reference start"
        ordered.append(
            EvidenceFact(
                "reference_start_sec",
                label,
                reference.start_sec,
                "s",
                ROLE_REFERENCE,
                SRC_PREDICTION,
                "Reference-timeline position matched by the DTW comparison.",
            )
        )
        if reference.status == REF_INTERVAL and reference.end_sec is not None:
            ordered.append(
                EvidenceFact(
                    "reference_end_sec",
                    "reference end",
                    reference.end_sec,
                    "s",
                    ROLE_REFERENCE,
                    SRC_PREDICTION,
                    "End of the DTW-matched reference interval.",
                )
            )
    ordered.append(
        EvidenceFact(
            "evidence_strength",
            "detector evidence strength",
            pred.evidence_strength,
            "score",
            ROLE_CONTEXT,
            SRC_PREDICTION,
            "Uncalibrated heuristic score from zero to one; not a probability or confidence.",
        )
    )
    if context.normalized_dtw_cost is not None and math.isfinite(context.normalized_dtw_cost):
        ordered.append(
            EvidenceFact(
                "normalized_dtw_cost",
                "normalized DTW cost",
                float(context.normalized_dtw_cost),
                "cost",
                ROLE_CONTEXT,
                SRC_COMPARISON,
                "Mean per-step feature distance along the DTW path for the whole recording. "
                "Lower means the aligned frames are more similar; no calibrated reliability "
                "threshold is defined.",
            )
        )
    ratio = context.duration_ratio
    if ratio is not None and math.isfinite(ratio):
        ordered.append(
            EvidenceFact(
                "recording_duration_ratio",
                "recording length ratio",
                float(ratio),
                "ratio",
                ROLE_CONTEXT,
                SRC_COMPARISON,
                "Length of the longer recording divided by the length of the shorter one.",
            )
        )
    if pred.decision_threshold is not None and math.isfinite(pred.decision_threshold):
        ordered.append(
            EvidenceFact(
                "decision_threshold",
                "detector decision threshold",
                float(pred.decision_threshold),
                "",
                ROLE_THRESHOLD,
                SRC_PREDICTION,
                "The primary threshold the detector reports for this category.",
            )
        )
    ordered.extend(facts[spec.key] for spec in specs)

    known = {s.key for s in specs} | set(CATEGORICAL_KEYS.get(pred.flaw_type, ())) | {"was_merged"}
    ignored = tuple(sorted(k for k in evidence if k not in known))

    alignment_warnings: list[str] = []
    if abs(variant.start_sec - pred.start_sec) > 1e-3 or abs(variant.end_sec - pred.end_sec) > 1e-3:
        alignment_warnings.append(
            "The prediction's stored seconds differ slightly from its sample bounds; the "
            "sample bounds are authoritative and are used for all displayed times."
        )

    return EvidenceBundle(
        prediction_id=pred.prediction_id,
        flaw_type=pred.flaw_type,
        detector_id=pred.detector_id,
        variant=variant,
        reference=reference,
        facts=tuple(ordered),
        evidence_strength=pred.evidence_strength,
        decision_threshold=pred.decision_threshold,
        detector_warnings=tuple(pred.warnings),
        alignment_warnings=tuple(alignment_warnings),
        was_merged=bool(evidence.get("was_merged", False)),
        ignored_evidence_keys=ignored,
        provenance=dict(provenance or {}),
        categorical=categorical,
    )


def detection_provenance(result: DetectionResult) -> dict[str, Any]:
    return {
        "detection_run_id": result.run_id,
        "detector_version": result.detector_version,
        "config_fingerprint": result.config_fingerprint,
        "ref_artifact_id": result.ref_artifact_id,
        "variant_artifact_id": result.variant_artifact_id,
        "ref_checksum": result.ref_checksum,
        "variant_checksum": result.variant_checksum,
        "sample_rate": result.sample_rate,
    }


def build_bundles(result: DetectionResult, context: AlignmentContext) -> list[EvidenceBundle]:
    """Build evidence bundles for every prediction in a (previously validated) result."""
    prov = detection_provenance(result)
    return [
        extract_bundle(p, sample_rate=result.sample_rate, context=context, provenance=prov)
        for p in result.predictions
    ]
