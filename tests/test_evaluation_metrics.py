"""Unit tests for Phase 5 classification metrics, aggregation, and localization curves."""

from __future__ import annotations

from flawbench.detection.schema import FlawPrediction
from flawbench.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_category_metrics,
    compute_localization_curve,
)
from flawbench.evaluation.schema import EventMatch, safe_div


def test_safe_div() -> None:
    assert safe_div(10.0, 2.0) == 5.0
    assert safe_div(0.0, 0.0, policy="undefined") is None
    assert safe_div(0.0, 0.0, policy="zero") == 0.0
    assert safe_div(5.0, 0.0, policy="undefined") is None


def test_category_metrics_perfect_scores() -> None:
    match1 = EventMatch(
        sample_id="s1",
        source_id="src1",
        match_status="matched",
        iou=0.9,
        category_match=True,
        pred_flaw_type="pause_insertion",
        gt_flaw_type="pause_insertion",
        pred_start_sample=1000,
        pred_end_sample=2000,
        gt_start_sample=1000,
        gt_end_sample=2000,
        start_error_sec=0.0,
        end_error_sec=0.0,
        duration_error_sec=0.0,
    )

    pm = compute_category_metrics("pause_insertion", [match1])
    assert pm.gt_count == 1
    assert pm.pred_count == 1
    assert pm.tp == 1
    assert pm.fp == 0
    assert pm.fn == 0
    assert pm.precision == 1.0
    assert pm.recall == 1.0
    assert pm.f1 == 1.0
    assert pm.mean_iou == 0.9
    assert pm.mean_abs_start_error_sec == 0.0


def test_category_metrics_with_fp_and_fn() -> None:
    # 1 matched, 1 false alarm (FP), 1 missed (FN)
    matches = [
        EventMatch(
            sample_id="s1",
            source_id="src1",
            match_status="matched",
            iou=0.8,
            category_match=True,
            pred_flaw_type="tempo_change",
            gt_flaw_type="tempo_change",
            start_error_sec=0.01,
            end_error_sec=0.02,
            duration_error_sec=0.01,
        ),
        EventMatch(
            sample_id="s1",
            source_id="src1",
            match_status="unmatched_prediction",
            iou=0.0,
            category_match=False,
            pred_flaw_type="tempo_change",
            gt_flaw_type=None,
        ),
        EventMatch(
            sample_id="s1",
            source_id="src1",
            match_status="unmatched_ground_truth",
            iou=0.0,
            category_match=False,
            pred_flaw_type=None,
            gt_flaw_type="tempo_change",
        ),
    ]

    tm = compute_category_metrics("tempo_change", matches)
    assert tm.tp == 1
    assert tm.fp == 1
    assert tm.fn == 1
    # Precision = 1 / (1 + 1) = 0.5
    # Recall = 1 / (1 + 1) = 0.5
    # F1 = 0.5
    assert tm.precision == 0.5
    assert tm.recall == 0.5
    assert tm.f1 == 0.5
    assert tm.mean_iou == 0.8
    assert tm.mean_abs_start_error_sec == 0.01


def test_micro_and_macro_aggregation() -> None:
    cat_metrics = {
        "pause_insertion": compute_category_metrics("pause_insertion", []),
        "tempo_change": compute_category_metrics("tempo_change", []),
    }
    agg = compute_aggregate_metrics(cat_metrics, [])
    assert agg.micro_precision is None
    assert agg.macro_precision is None


def test_localization_curve() -> None:
    # GT interval [1000, 3000) (len 2000)
    # Pred interval [1000, 2500) (len 1500)
    # Intersection = 1500, Union = 2000 -> IoU = 0.75
    gt_event = {
        "flaw_type": "pause_insertion",
        "output_interval": {
            "start_sample": 1000,
            "end_sample": 3000,
            "start_sec": 0.1,
            "end_sec": 0.3,
        },
    }
    pred = FlawPrediction(
        prediction_id="pred-loc",
        flaw_type="pause_insertion",
        start_sample=1000,
        end_sample=2500,
        start_sec=0.1,
        end_sec=0.25,
        duration_sec=0.15,
        evidence_strength=0.9,
        detector_id="PauseDetector",
    )

    record = {
        "sample_id": "test_s",
        "source_id": "src1",
        "gt_events": [gt_event],
        "predictions": [pred],
        "sample_rate": 16000,
    }

    curve = compute_localization_curve([record], thresholds=(0.5, 0.7, 0.8, 0.9))
    assert len(curve) == 4
    # IoU is 0.75:
    # At threshold 0.5 and 0.7 -> matches (recall = 1.0)
    # At threshold 0.8 and 0.9 -> does not match (recall = 0.0)
    pt_05 = [p for p in curve if p.iou_threshold == 0.5][0]
    pt_08 = [p for p in curve if p.iou_threshold == 0.8][0]
    assert pt_05.matched_count == 1
    assert pt_05.recall == 1.0
    assert pt_08.matched_count == 0
    assert pt_08.recall == 0.0
