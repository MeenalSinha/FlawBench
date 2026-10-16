"""Deterministic prediction builders for explanation tests (synthetic evidence, no audio)."""

from __future__ import annotations

from typing import Any

from flawbench.detection.schema import FlawPrediction
from flawbench.explainability.facts import AlignmentContext, extract_bundle

SR = 16000

PAUSE_EVIDENCE: dict[str, Any] = {
    "variant_pause_duration_sec": 0.62,
    "reference_duration_sec": 0.0,
    "duration_difference_sec": 0.62,
    "mean_variant_rms_db": -90.5,
    "threshold_silence_db": -35.0,
    "min_duration_sec": 0.15,
    "ref_frozen_fraction": 1.0,
}
TEMPO_EVIDENCE: dict[str, Any] = {
    "direction": "slower",
    "estimated_relative_rate": 0.7,
    "mean_warp_slope": 1.4286,
    "rate_deviation": 0.3,
    "threshold_min_deviation": 0.15,
    "duration_sec": 1.2,
}
ENERGY_EVIDENCE: dict[str, Any] = {
    "mean_delta_rms_db": -8.0,
    "min_delta_rms_db": -11.0,
    "mean_ref_rms_db": -20.0,
    "mean_var_rms_db": -28.0,
    "threshold_gain_db": -4.0,
    "duration_sec": 0.9,
}


def make_pred(
    flaw: str,
    evidence: dict[str, Any],
    *,
    start: int = 16000,
    end: int = 32000,
    ref: tuple[float, float] | None = (1.0, 1.0),
    strength: float = 0.6,
) -> FlawPrediction:
    return FlawPrediction(
        prediction_id=f"pred_{flaw}_test",
        flaw_type=flaw,
        start_sample=start,
        end_sample=end,
        start_sec=start / SR,
        end_sec=end / SR,
        duration_sec=(end - start) / SR,
        ref_start_sec=ref[0] if ref else None,
        ref_end_sec=ref[1] if ref else None,
        evidence=dict(evidence),
        evidence_strength=strength,
        detector_id=f"{flaw}_detector",
        decision_threshold=0.15,
    )


def ctx(**kw: Any) -> AlignmentContext:
    base: dict[str, Any] = {
        "reference_duration_sec": 6.0,
        "participant_duration_sec": 6.5,
        "normalized_dtw_cost": 0.3,
        "duration_ratio_warning": 2.0,
    }
    base.update(kw)
    return AlignmentContext(**base)


def bundle(flaw: str, evidence: dict[str, Any], **kw: Any):  # type: ignore[no-untyped-def]
    ref = kw.pop("ref", (1.0, 1.0) if flaw == "pause_insertion" else (1.0, 2.0))
    context = kw.pop("context", ctx())
    return extract_bundle(make_pred(flaw, evidence, ref=ref, **kw), sample_rate=SR, context=context)
