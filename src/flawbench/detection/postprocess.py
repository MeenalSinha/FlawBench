"""Phase 4 Prediction Post-Processing and Consolidation.

Implements candidate merging, short candidate filtering, boundary verification,
and category preservation.
"""

from __future__ import annotations

import uuid
from typing import Any

from flawbench.detection.schema import FlawPrediction


def _merge_ref(a: float | None, b: float | None, pick: Any) -> float | None:
    """Combine two optional reference times; 0.0 is a valid time, so test for None explicitly."""
    if a is None:
        return b
    if b is None:
        return a
    return float(pick(a, b))


def postprocess_predictions(
    candidates: list[FlawPrediction],
    *,
    merge_gap_sec: float,
    min_candidate_duration_sec: float,
    total_samples: int,
    sample_rate: int,
) -> list[FlawPrediction]:
    """Consolidate, merge adjacent intervals of the same flaw category, and filter short candidates.

    Invariants:
    1. Only candidates of the same `flaw_type` can merge.
    2. Merging requires temporal separation <= `merge_gap_sec`.
    3. Candidates shorter than `min_candidate_duration_sec` are discarded.
    4. Boundaries are strictly validated [start_sample, end_sample) within [0, total_samples].
    5. Chronologically ordered output.
    """
    if not candidates:
        return []

    # Group by flaw_type
    by_category: dict[str, list[FlawPrediction]] = {}
    for c in candidates:
        by_category.setdefault(c.flaw_type, []).append(c)

    consolidated: list[FlawPrediction] = []

    for flaw_type, items in by_category.items():
        # Sort by start_sample
        items.sort(key=lambda p: (p.start_sample, p.end_sample))

        merged_category: list[FlawPrediction] = []
        for cand in items:
            if not merged_category:
                merged_category.append(cand)
                continue

            last = merged_category[-1]
            gap_sec = cand.start_sec - last.end_sec

            if gap_sec <= merge_gap_sec:
                # Merge last and cand
                start_sample = min(last.start_sample, cand.start_sample)
                end_sample = max(last.end_sample, cand.end_sample)
                start_sec = start_sample / sample_rate
                end_sec = end_sample / sample_rate
                duration_sec = end_sec - start_sec

                ref_s = _merge_ref(last.ref_start_sec, cand.ref_start_sec, min)
                ref_e = _merge_ref(last.ref_end_sec, cand.ref_end_sec, max)

                combined_evidence: dict[str, Any] = dict(last.evidence)
                combined_evidence.update({f"merged_{k}": v for k, v in cand.evidence.items()})
                combined_evidence["was_merged"] = True

                new_pred = FlawPrediction(
                    prediction_id=f"pred_merged_{uuid.uuid4().hex[:8]}",
                    flaw_type=flaw_type,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_sec=round(start_sec, 4),
                    end_sec=round(end_sec, 4),
                    duration_sec=round(duration_sec, 4),
                    ref_start_sec=round(ref_s, 4) if ref_s is not None else None,
                    ref_end_sec=round(ref_e, 4) if ref_e is not None else None,
                    evidence=combined_evidence,
                    evidence_strength=round(max(last.evidence_strength, cand.evidence_strength), 3),
                    detector_id=f"{last.detector_id}+merged",
                    decision_threshold=last.decision_threshold,
                    warnings=tuple(dict.fromkeys(list(last.warnings) + list(cand.warnings))),
                )
                merged_category[-1] = new_pred
            else:
                merged_category.append(cand)

        # Filter by min duration and valid bounds
        for p in merged_category:
            if p.duration_sec < min_candidate_duration_sec:
                continue
            if p.start_sample < 0 or p.end_sample > total_samples or p.start_sample >= p.end_sample:
                continue
            consolidated.append(p)

    # Sort all final predictions chronologically
    consolidated.sort(key=lambda p: (p.start_sample, p.end_sample, p.flaw_type))
    return consolidated
