"""Quantitative classification and temporal localization metric calculations."""

from __future__ import annotations

import statistics
from typing import Any

from flawbench.evaluation.matching import match_sample_events
from flawbench.evaluation.schema import (
    AggregateMetrics,
    CategoryMetrics,
    EventMatch,
    LocalizationPoint,
    safe_div,
)


def compute_category_metrics(
    flaw_type: str,
    matches: list[EventMatch],
    *,
    policy: str = "undefined",
) -> CategoryMetrics:
    """Calculate precision, recall, F1, and temporal localization errors for one category."""
    # Matched events of this category (True Positives)
    tp_matches = [
        m
        for m in matches
        if m.match_status == "matched" and m.gt_flaw_type == flaw_type and m.category_match
    ]
    tp = len(tp_matches)

    # False Positives: predictions of this category that are not valid TPs
    # 1. Unmatched predictions of this category
    # 2. Matched predictions of this category where GT was a different category
    fp_unmatched = sum(
        1
        for m in matches
        if m.match_status == "unmatched_prediction" and m.pred_flaw_type == flaw_type
    )
    fp_misclass = sum(
        1
        for m in matches
        if m.match_status == "matched"
        and m.pred_flaw_type == flaw_type
        and m.gt_flaw_type != flaw_type
    )
    fp = fp_unmatched + fp_misclass

    # False Negatives: GT events of this category that are not valid TPs
    # 1. Unmatched GT of this category
    # 2. GT of this category matched to a prediction of a different category
    fn_unmatched = sum(
        1
        for m in matches
        if m.match_status == "unmatched_ground_truth" and m.gt_flaw_type == flaw_type
    )
    fn_misclass = sum(
        1
        for m in matches
        if m.match_status == "matched"
        and m.gt_flaw_type == flaw_type
        and m.pred_flaw_type != flaw_type
    )
    fn = fn_unmatched + fn_misclass

    gt_count = tp + fn
    pred_count = tp + fp

    prec = safe_div(float(tp), float(tp + fp), policy=policy)
    rec = safe_div(float(tp), float(tp + fn), policy=policy)
    if prec is not None and rec is not None and (prec + rec) > 0.0:
        f1: float | None = 2.0 * prec * rec / (prec + rec)
    else:
        f1 = 0.0 if policy == "zero" else None

    # Localization statistics for true positives
    ious = [m.iou for m in tp_matches]
    mean_iou = statistics.mean(ious) if ious else None
    median_iou = statistics.median(ious) if ious else None

    start_errs = [abs(m.start_error_sec) for m in tp_matches if m.start_error_sec is not None]
    end_errs = [abs(m.end_error_sec) for m in tp_matches if m.end_error_sec is not None]
    dur_errs = [abs(m.duration_error_sec) for m in tp_matches if m.duration_error_sec is not None]

    mean_start_err = statistics.mean(start_errs) if start_errs else None
    mean_end_err = statistics.mean(end_errs) if end_errs else None
    mean_dur_err = statistics.mean(dur_errs) if dur_errs else None

    return CategoryMetrics(
        flaw_type=flaw_type,
        gt_count=gt_count,
        pred_count=pred_count,
        matched_count=tp,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=prec,
        recall=rec,
        f1=f1,
        mean_iou=mean_iou,
        median_iou=median_iou,
        mean_abs_start_error_sec=mean_start_err,
        mean_abs_end_error_sec=mean_end_err,
        mean_abs_dur_error_sec=mean_dur_err,
    )


def compute_aggregate_metrics(
    per_category: dict[str, CategoryMetrics],
    all_matches: list[EventMatch],
    *,
    policy: str = "undefined",
) -> AggregateMetrics:
    """Calculate micro- and macro-averaged metrics across all evaluated categories."""
    total_gt = sum(c.gt_count for c in per_category.values())
    total_pred = sum(c.pred_count for c in per_category.values())
    total_tp = sum(c.tp for c in per_category.values())
    total_fp = sum(c.fp for c in per_category.values())
    total_fn = sum(c.fn for c in per_category.values())

    micro_prec = safe_div(float(total_tp), float(total_tp + total_fp), policy=policy)
    micro_rec = safe_div(float(total_tp), float(total_tp + total_fn), policy=policy)
    if micro_prec is not None and micro_rec is not None and (micro_prec + micro_rec) > 0.0:
        micro_f1: float | None = 2.0 * micro_prec * micro_rec / (micro_prec + micro_rec)
    else:
        micro_f1 = 0.0 if policy == "zero" else None

    # Macro averaging: mean across categories that have defined metrics
    valid_precs = [c.precision for c in per_category.values() if c.precision is not None]
    valid_recs = [c.recall for c in per_category.values() if c.recall is not None]
    valid_f1s = [c.f1 for c in per_category.values() if c.f1 is not None]

    macro_prec = (
        statistics.mean(valid_precs) if valid_precs else (0.0 if policy == "zero" else None)
    )
    macro_rec = statistics.mean(valid_recs) if valid_recs else (0.0 if policy == "zero" else None)
    macro_f1 = statistics.mean(valid_f1s) if valid_f1s else (0.0 if policy == "zero" else None)

    all_matched_ious = [m.iou for m in all_matches if m.match_status == "matched"]
    mean_iou = statistics.mean(all_matched_ious) if all_matched_ious else None
    median_iou = statistics.median(all_matched_ious) if all_matched_ious else None

    return AggregateMetrics(
        total_gt_events=total_gt,
        total_predictions=total_pred,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        micro_precision=micro_prec,
        micro_recall=micro_rec,
        micro_f1=micro_f1,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        macro_f1=macro_f1,
        mean_iou=mean_iou,
        median_iou=median_iou,
    )


def compute_localization_curve(
    sample_records: list[dict[str, Any]],
    thresholds: tuple[float, ...],
    *,
    require_category_match: bool = True,
    policy: str = "undefined",
) -> list[LocalizationPoint]:
    """Compute localization performance (event recall) across multiple IoU thresholds."""
    points: list[LocalizationPoint] = []
    for thresh in thresholds:
        total_gt = 0
        total_matched = 0
        for item in sample_records:
            sample_id = item["sample_id"]
            source_id = item["source_id"]
            gt_events = item["gt_events"]
            preds = item["predictions"]
            sr = item["sample_rate"]
            matches = match_sample_events(
                sample_id,
                source_id,
                gt_events,
                preds,
                sr,
                iou_threshold=thresh,
                require_category_match=require_category_match,
            )
            matched_here = sum(
                1
                for m in matches
                if m.match_status == "matched" and (not require_category_match or m.category_match)
            )
            total_matched += matched_here
            total_gt += len(gt_events)

        rec = safe_div(float(total_matched), float(total_gt), policy=policy)
        points.append(
            LocalizationPoint(
                iou_threshold=thresh,
                matched_count=total_matched,
                gt_count=total_gt,
                recall=rec,
            )
        )
    return points
