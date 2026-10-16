"""Integration tests for EvaluationStore, EvaluationValidator, Runner, and CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from flawbench.__main__ import main
from flawbench.config import load_config
from flawbench.evaluation.reports import generate_markdown_report
from flawbench.evaluation.schema import (
    EVALUATION_SCHEMA_VERSION,
    AggregateMetrics,
    CategoryMetrics,
    ConfusionMatrix,
    EvaluationReport,
    EventMatch,
    SampleEvaluation,
)
from flawbench.evaluation.store import EvaluationStore
from flawbench.evaluation.validator import EvaluationValidator
from flawbench.exceptions import EvaluationValidationError
from flawbench.paths import ProjectPaths


@pytest.fixture
def test_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectPaths:
    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(tmp_path))
    cfg = load_config()
    paths = ProjectPaths(cfg)
    paths.ensure_output_directories()
    return paths


def test_evaluation_store_and_validator(test_paths: ProjectPaths) -> None:
    store = EvaluationStore(test_paths)
    validator = EvaluationValidator()

    cat_m = CategoryMetrics(
        flaw_type="pause_insertion",
        gt_count=2,
        pred_count=2,
        matched_count=2,
        tp=2,
        fp=0,
        fn=0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        mean_iou=0.9,
        median_iou=0.9,
        mean_abs_start_error_sec=0.0,
        mean_abs_end_error_sec=0.0,
        mean_abs_dur_error_sec=0.0,
    )
    agg_m = AggregateMetrics(
        total_gt_events=2,
        total_predictions=2,
        total_tp=2,
        total_fp=0,
        total_fn=0,
        micro_precision=1.0,
        micro_recall=1.0,
        micro_f1=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        mean_iou=0.9,
        median_iou=0.9,
    )
    match = EventMatch(
        sample_id="test_samp",
        source_id="src_1",
        match_status="matched",
        iou=0.9,
        category_match=True,
        pred_flaw_type="pause_insertion",
        gt_flaw_type="pause_insertion",
        pred_start_sample=1000,
        pred_end_sample=2000,
        gt_start_sample=1000,
        gt_end_sample=2000,
    )
    conf_mat = ConfusionMatrix(
        categories=["pause_insertion"],
        matrix={
            "pause_insertion": {"pause_insertion": 2, "missed": 0},
            "none": {"pause_insertion": 0, "missed": 0},
        },
    )

    report = EvaluationReport(
        schema_version=EVALUATION_SCHEMA_VERSION,
        run_id="eval_test_001",
        split="test",
        sample_count=1,
        source_count=1,
        excluded_count=0,
        exclusion_reasons={},
        per_category={"pause_insertion": cat_m},
        aggregate=agg_m,
        localization_curve=[],
        confusion_matrix=conf_mat,
        created_at="2026-10-15T00:00:00Z",
        detector_version="0.1.0",
        detector_config_fingerprint="abc",
        eval_config_fingerprint="def",
        is_synthetic_only=True,
        sample_evaluations=[SampleEvaluation("test_samp", "src_1", "test", 1, 1, [match])],
    )

    # 1. Validation in-memory
    warnings = validator.validate_report(report)
    assert any("synthetic" in w.lower() for w in warnings)

    # 2. Saving to store
    md_content = generate_markdown_report(report)
    run_dir = store.save_evaluation(report, markdown_report=md_content, overwrite=True)
    assert run_dir.is_dir()
    assert (run_dir / "evaluation_report.json").is_file()
    assert (run_dir / "event_matches.jsonl").is_file()
    assert (run_dir / "report.md").is_file()

    # 3. Listing runs
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "eval_test_001"
    assert runs[0]["micro_f1"] == 1.0

    # 4. Loading run
    loaded = store.load_evaluation("eval_test_001")
    assert loaded.run_id == report.run_id
    assert loaded.sample_count == 1
    assert "pause_insertion" in loaded.per_category

    # 5. Validation on disk
    val_report, val_warnings = validator.validate_run_directory(run_dir)
    assert val_report.run_id == "eval_test_001"


def test_validator_rejects_corrupted_metrics() -> None:
    validator = EvaluationValidator()

    # Out of range precision (> 1.0)
    cat_bad = CategoryMetrics(
        flaw_type="pause_insertion",
        gt_count=2,
        pred_count=2,
        matched_count=2,
        tp=2,
        fp=0,
        fn=0,
        precision=1.5,
        recall=1.0,
        f1=1.0,
        mean_iou=0.9,
        median_iou=0.9,
        mean_abs_start_error_sec=0.0,
        mean_abs_end_error_sec=0.0,
        mean_abs_dur_error_sec=0.0,
    )
    agg_m = AggregateMetrics(
        total_gt_events=2,
        total_predictions=2,
        total_tp=2,
        total_fp=0,
        total_fn=0,
        micro_precision=1.0,
        micro_recall=1.0,
        micro_f1=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        mean_iou=0.9,
        median_iou=0.9,
    )
    conf_mat = ConfusionMatrix(categories=[], matrix={})

    report = EvaluationReport(
        schema_version=EVALUATION_SCHEMA_VERSION,
        run_id="bad_run",
        split="test",
        sample_count=1,
        source_count=1,
        excluded_count=0,
        exclusion_reasons={},
        per_category={"pause_insertion": cat_bad},
        aggregate=agg_m,
        localization_curve=[],
        confusion_matrix=conf_mat,
        created_at="now",
        detector_version="0.1.0",
        detector_config_fingerprint="abc",
        eval_config_fingerprint="def",
        is_synthetic_only=False,
    )

    with pytest.raises(EvaluationValidationError, match="Invalid precision"):
        validator.validate_report(report)


def test_cli_evaluate_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(tmp_path))

    # Initialize directories
    rc = main(["init"])
    assert rc == 0
    capsys.readouterr()

    # Test evaluate list on empty
    rc = main(["evaluate", "list"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Evaluation runs: 0" in captured.out

    # Run evaluation on empty dataset
    rc = main(["evaluate", "run", "--split", "test", "--no-controls"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Evaluation completed" in captured.out

    # List again -> 1 run
    rc = main(["evaluate", "list"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Evaluation runs: 1" in captured.out
