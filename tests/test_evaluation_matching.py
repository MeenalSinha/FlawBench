"""Unit tests for FlawBench Phase 5 event matching and IoU calculation."""

from __future__ import annotations

import pytest

from flawbench.detection.schema import FlawPrediction
from flawbench.evaluation.matching import compute_interval_iou, match_sample_events


def test_iou_exact_match() -> None:
    assert compute_interval_iou(100, 200, 100, 200) == 1.0
    assert compute_interval_iou(0, 16000, 0, 16000) == 1.0


def test_iou_disjoint_intervals() -> None:
    assert compute_interval_iou(100, 200, 300, 400) == 0.0
    assert compute_interval_iou(100, 200, 200, 300) == 0.0  # half-open boundary


def test_iou_partial_overlap() -> None:
    # A = [100, 300) (len 200), B = [200, 400) (len 200)
    # Intersection = [200, 300) (len 100)
    # Union = [100, 400) (len 300)
    # IoU = 100 / 300 = 1/3
    iou = compute_interval_iou(100, 300, 200, 400)
    assert pytest.approx(iou, rel=1e-4) == 1.0 / 3.0


def test_iou_contained_interval() -> None:
    # A = [100, 200) (len 100), B = [100, 400) (len 300)
    # Intersection = 100, Union = 300 -> IoU = 1/3
    assert pytest.approx(compute_interval_iou(100, 200, 100, 400), rel=1e-4) == 1.0 / 3.0


def test_iou_invalid_intervals() -> None:
    assert compute_interval_iou(200, 100, 100, 200) == 0.0
    assert compute_interval_iou(100, 100, 100, 200) == 0.0


def test_one_to_one_bipartite_matching() -> None:
    # 1 GT pause, 2 overlapping predicted pauses
    gt_events = [
        {
            "flaw_type": "pause_insertion",
            "output_interval": {
                "start_sample": 16000,
                "end_sample": 32000,
                "start_sec": 1.0,
                "end_sec": 2.0,
            },
        }
    ]

    # Pred 1: near exact match (IoU ~ 0.94)
    pred1 = FlawPrediction(
        prediction_id="pred-1",
        flaw_type="pause_insertion",
        start_sample=16000,
        end_sample=31000,
        start_sec=1.0,
        end_sec=1.9375,
        duration_sec=0.9375,
        evidence_strength=0.9,
        detector_id="PauseDetector",
    )
    # Pred 2: lower IoU match (IoU ~ 0.5)
    pred2 = FlawPrediction(
        prediction_id="pred-2",
        flaw_type="pause_insertion",
        start_sample=16000,
        end_sample=24000,
        start_sec=1.0,
        end_sec=1.5,
        duration_sec=0.5,
        evidence_strength=0.7,
        detector_id="PauseDetector",
    )

    matches = match_sample_events(
        sample_id="test_samp_1",
        source_id="src_1",
        ground_truth_events=gt_events,
        predictions=[pred1, pred2],
        sample_rate=16000,
        iou_threshold=0.5,
        require_category_match=True,
    )

    assert len(matches) == 2
    matched = [m for m in matches if m.match_status == "matched"]
    unmatched_pred = [m for m in matches if m.match_status == "unmatched_prediction"]

    # Exactly 1 event receives matching credit
    assert len(matched) == 1
    assert matched[0].pred_start_sample == 16000
    assert matched[0].pred_end_sample == 31000
    assert matched[0].iou > 0.9

    # The other prediction is marked as an unmatched prediction (false positive)
    assert len(unmatched_pred) == 1
    assert unmatched_pred[0].pred_start_sample == 16000
    assert unmatched_pred[0].pred_end_sample == 24000


def test_category_mismatch_filtering() -> None:
    gt_events = [
        {
            "flaw_type": "pause_insertion",
            "output_interval": {
                "start_sample": 16000,
                "end_sample": 32000,
                "start_sec": 1.0,
                "end_sec": 2.0,
            },
        }
    ]
    # Prediction has same interval but wrong category
    pred = FlawPrediction(
        prediction_id="pred-wrong-cat",
        flaw_type="energy_reduction",
        start_sample=16000,
        end_sample=32000,
        start_sec=1.0,
        end_sec=2.0,
        duration_sec=1.0,
        evidence_strength=0.8,
        detector_id="EnergyReductionDetector",
    )

    # When require_category_match=True, they cannot match
    matches = match_sample_events(
        sample_id="test_samp_2",
        source_id="src_1",
        ground_truth_events=gt_events,
        predictions=[pred],
        sample_rate=16000,
        iou_threshold=0.5,
        require_category_match=True,
    )
    statuses = [m.match_status for m in matches]
    assert "matched" not in statuses
    assert "unmatched_ground_truth" in statuses
    assert "unmatched_prediction" in statuses


def test_unmatched_ground_truth_when_below_min_iou() -> None:
    gt_events = [
        {
            "flaw_type": "pause_insertion",
            "output_interval": {
                "start_sample": 1000,
                "end_sample": 2000,
                "start_sec": 0.1,
                "end_sec": 0.2,
            },
        }
    ]
    pred = FlawPrediction(
        prediction_id="pred-low-iou",
        flaw_type="pause_insertion",
        start_sample=1800,
        end_sample=3000,
        start_sec=0.18,
        end_sec=0.3,
        duration_sec=0.12,
        evidence_strength=0.8,
        detector_id="PauseDetector",
    )
    # IoU = 200 / 2000 = 0.1, below min_iou 0.5
    matches = match_sample_events(
        sample_id="test_samp_3",
        source_id="src_1",
        ground_truth_events=gt_events,
        predictions=[pred],
        sample_rate=16000,
        iou_threshold=0.5,
        require_category_match=True,
    )
    assert any(m.match_status == "unmatched_ground_truth" for m in matches)
    assert any(m.match_status == "unmatched_prediction" for m in matches)
