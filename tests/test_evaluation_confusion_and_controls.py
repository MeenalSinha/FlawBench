"""Unit tests for Phase 5 confusion matrix, negative identity controls, and bootstrap CI."""

from __future__ import annotations

from flawbench.evaluation.bootstrap import compute_source_level_bootstrap
from flawbench.evaluation.confusion import build_confusion_matrix
from flawbench.evaluation.schema import EventMatch, SampleEvaluation


def test_confusion_matrix_correct_and_spurious() -> None:
    matches = [
        # Correctly matched pause
        EventMatch(
            sample_id="s1",
            source_id="src1",
            match_status="matched",
            iou=0.85,
            category_match=True,
            pred_flaw_type="pause_insertion",
            gt_flaw_type="pause_insertion",
        ),
        # Spurious detection (no ground truth)
        EventMatch(
            sample_id="s1",
            source_id="src1",
            match_status="unmatched_prediction",
            iou=0.0,
            category_match=False,
            pred_flaw_type="energy_reduction",
            gt_flaw_type=None,
        ),
        # Missed ground truth event
        EventMatch(
            sample_id="s2",
            source_id="src2",
            match_status="unmatched_ground_truth",
            iou=0.0,
            category_match=False,
            pred_flaw_type=None,
            gt_flaw_type="tempo_change",
        ),
    ]

    conf_mat = build_confusion_matrix(matches)
    matrix = conf_mat.matrix

    # Correct pause: row "pause_insertion", col "pause_insertion"
    assert matrix["pause_insertion"]["pause_insertion"] == 1
    # Spurious energy_reduction: row "none", col "energy_reduction"
    assert matrix["none"]["energy_reduction"] == 1
    # Missed tempo_change: row "tempo_change", col "missed"
    assert matrix["tempo_change"]["missed"] == 1


def test_source_level_bootstrap_computation() -> None:
    # Build sample evaluations from multiple sources
    sample_evals = []
    for i in range(1, 6):
        src_id = f"source_{i}"
        match = EventMatch(
            sample_id=f"sample_{i}",
            source_id=src_id,
            match_status="matched",
            iou=0.8,
            category_match=True,
            pred_flaw_type="pause_insertion",
            gt_flaw_type="pause_insertion",
            pred_start_sample=1000,
            pred_end_sample=2000,
            gt_start_sample=1000,
            gt_end_sample=2000,
        )
        sample_evals.append(
            SampleEvaluation(
                sample_id=f"sample_{i}",
                source_id=src_id,
                split="test",
                num_gt=1,
                num_pred=1,
                matches=[match],
            )
        )

    res = compute_source_level_bootstrap(
        sample_evaluations=sample_evals,
        replicates=50,
        seed=42,
    )

    assert res["status"] == "completed"
    assert res["source_count"] == 5
    assert res["replicates"] == 50
    assert "metrics" in res
    assert "micro_f1" in res["metrics"]
    ci = res["metrics"]["micro_f1"]
    assert ci["ci_lower_2_5"] == 1.0
    assert ci["ci_upper_97_5"] == 1.0
