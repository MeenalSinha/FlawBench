"""Deterministic one-to-one event matching and temporal IoU computation."""

from __future__ import annotations

from typing import Any

from flawbench.detection.schema import FlawPrediction
from flawbench.evaluation.schema import EventMatch


def compute_interval_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    """Compute temporal Intersection-over-Union (IoU) for two half-open intervals [s, e).

    Integer sample bounds are authoritative. Returns 0.0 if either interval is invalid.
    """
    if e1 <= s1 or e2 <= s2:
        return 0.0
    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def match_sample_events(
    sample_id: str,
    source_id: str,
    ground_truth_events: list[dict[str, Any]],
    predictions: list[FlawPrediction],
    sample_rate: int,
    *,
    iou_threshold: float = 0.5,
    require_category_match: bool = True,
) -> list[EventMatch]:
    """Perform deterministic one-to-one bipartite event matching between predictions and GT.

    Parameters:
        sample_id: Unique variant recording identifier.
        source_id: Original reference source identifier.
        ground_truth_events: Ground-truth flaw records (with 'flaw_type' and 'output_interval').
        predictions: Predicted flaws from independent detection.
        sample_rate: Audio sampling rate for error time conversions.
        iou_threshold: Minimum temporal IoU required for a valid match.
        require_category_match: If True, only pairs with identical flaw_type can match.

    Returns:
        List of EventMatch records (matched pairs, unmatched predictions, and unmatched GT).
    """
    # 1. Prepare candidates
    # Format ground truth records
    gt_list: list[dict[str, Any]] = []
    for idx, gt in enumerate(ground_truth_events):
        flaw_type = gt.get("flaw_type")
        out_int = gt.get("output_interval") or gt.get("annotation", {}).get("output_interval", {})
        start_samp = int(out_int.get("start_sample", 0))
        end_samp = int(out_int.get("end_sample", 0))
        start_sec = float(
            out_int.get("start_sec", start_samp / sample_rate if sample_rate > 0 else 0.0)
        )
        end_sec = float(out_int.get("end_sec", end_samp / sample_rate if sample_rate > 0 else 0.0))
        gt_list.append(
            {
                "gt_id": f"gt-{idx + 1}",
                "flaw_type": flaw_type,
                "start_sample": start_samp,
                "end_sample": end_samp,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )

    # 2. Build candidate match matrix
    candidate_pairs: list[tuple[float, int, int]] = []  # (iou, gt_idx, pred_idx)
    for g_idx, g in enumerate(gt_list):
        for p_idx, p in enumerate(predictions):
            cat_match = g["flaw_type"] == p.flaw_type
            if require_category_match and not cat_match:
                continue

            iou = compute_interval_iou(
                g["start_sample"], g["end_sample"], p.start_sample, p.end_sample
            )
            if iou >= iou_threshold:
                candidate_pairs.append((iou, g_idx, p_idx))

    # 3. Deterministic greedy one-to-one matching: highest IoU first
    # Tie-breaking on start sample and indices guarantees 100% deterministic reproducibility
    candidate_pairs.sort(
        key=lambda item: (
            item[0],
            -predictions[item[2]].start_sample,
            -gt_list[item[1]]["start_sample"],
            -item[2],
            -item[1],
        ),
        reverse=True,
    )

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    results: list[EventMatch] = []

    for iou, g_idx, p_idx in candidate_pairs:
        if g_idx in matched_gt or p_idx in matched_pred:
            continue
        matched_gt.add(g_idx)
        matched_pred.add(p_idx)

        g = gt_list[g_idx]
        p = predictions[p_idx]
        cat_match = g["flaw_type"] == p.flaw_type
        start_err = (p.start_sample - g["start_sample"]) / sample_rate
        end_err = (p.end_sample - g["end_sample"]) / sample_rate
        dur_err = (
            p.end_sample - p.start_sample - (g["end_sample"] - g["start_sample"])
        ) / sample_rate

        results.append(
            EventMatch(
                sample_id=sample_id,
                source_id=source_id,
                match_status="matched",
                iou=iou,
                category_match=cat_match,
                prediction_id=p.prediction_id,
                ground_truth_id=g["gt_id"],
                pred_flaw_type=p.flaw_type,
                gt_flaw_type=g["flaw_type"],
                pred_start_sample=p.start_sample,
                pred_end_sample=p.end_sample,
                gt_start_sample=g["start_sample"],
                gt_end_sample=g["end_sample"],
                pred_start_sec=p.start_sec,
                pred_end_sec=p.end_sec,
                gt_start_sec=g["start_sec"],
                gt_end_sec=g["end_sec"],
                start_error_sec=start_err,
                end_error_sec=end_err,
                duration_error_sec=dur_err,
            )
        )

    # 4. Record unmatched predictions (Spurious false positives)
    for p_idx, p in enumerate(predictions):
        if p_idx not in matched_pred:
            results.append(
                EventMatch(
                    sample_id=sample_id,
                    source_id=source_id,
                    match_status="unmatched_prediction",
                    iou=0.0,
                    category_match=False,
                    prediction_id=p.prediction_id,
                    pred_flaw_type=p.flaw_type,
                    pred_start_sample=p.start_sample,
                    pred_end_sample=p.end_sample,
                    pred_start_sec=p.start_sec,
                    pred_end_sec=p.end_sec,
                )
            )

    # 5. Record unmatched ground truth (Missed false negatives)
    for g_idx, g in enumerate(gt_list):
        if g_idx not in matched_gt:
            results.append(
                EventMatch(
                    sample_id=sample_id,
                    source_id=source_id,
                    match_status="unmatched_ground_truth",
                    iou=0.0,
                    category_match=False,
                    ground_truth_id=g["gt_id"],
                    gt_flaw_type=g["flaw_type"],
                    gt_start_sample=g["start_sample"],
                    gt_end_sample=g["end_sample"],
                    gt_start_sec=g["start_sec"],
                    gt_end_sec=g["end_sec"],
                )
            )

    return results
