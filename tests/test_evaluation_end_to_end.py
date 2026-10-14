"""End-to-end formal evaluation and benchmarking integration tests."""

from __future__ import annotations

import json
from typing import Any

from flawbench.config import Config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.splits import build_splits
from flawbench.evaluation.runner import EvaluationRunner
from flawbench.evaluation.store import EvaluationStore
from flawbench.evaluation.validator import EvaluationValidator
from flawbench.paths import ProjectPaths


def test_end_to_end_formal_evaluation(
    ds_config: Config,
    source: Any,
) -> None:
    """Execute complete end-to-end Phase 5 evaluation on generated variants."""
    paths = ProjectPaths(ds_config)
    gen = DatasetGenerator(ds_config)

    # 1. Generate 3 variants with known injections
    res_pause = gen.generate_variant(
        source.source_id,
        "pause_insertion",
        severity="medium",
        placement={"mode": "explicit", "at_sec": 1.5},
    )
    res_tempo = gen.generate_variant(
        source.source_id,
        "tempo_change",
        severity="medium",
        placement={"mode": "explicit", "start_sec": 1.0},
    )
    res_energy = gen.generate_variant(
        source.source_id,
        "energy_reduction",
        severity="medium",
        placement={"mode": "explicit", "start_sec": 2.0},
    )

    assert res_pause.record.sample_id
    assert res_tempo.record.sample_id
    assert res_energy.record.sample_id

    # 2. Build source-level splits
    splits_res = build_splits(ds_config, paths, force=True)
    assert splits_res.assignments

    # 3. Run formal evaluation across all variants
    runner = EvaluationRunner(ds_config, paths)
    report, run_dir = runner.run(split="all", include_controls=True, enable_bootstrap=False)

    # 4. Verify evaluation report structure & counts
    assert report.sample_count == 3, f"exclusions: {report.exclusion_reasons}"
    assert report.source_count == 1
    assert isinstance(report.is_synthetic_only, bool)
    assert run_dir.is_dir()
    assert (run_dir / "evaluation_report.json").is_file()
    assert (run_dir / "event_matches.jsonl").is_file()
    assert (run_dir / "report.md").is_file()

    # 5. Verify event matches
    matches = []
    with open(run_dir / "event_matches.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                matches.append(json.loads(line))
    assert len(matches) >= 3
    for m in matches:
        assert "match_status" in m
        assert "iou" in m
        assert 0.0 <= m["iou"] <= 1.0

    # 6. Verify per-category metrics exist
    for cat in ["pause_insertion", "tempo_change", "energy_reduction"]:
        assert cat in report.per_category
        cat_m = report.per_category[cat]
        assert cat_m.gt_count >= 1

    # 7. Verify localization tolerance curve
    assert len(report.localization_curve) == len(ds_config.evaluation.localization_iou_thresholds)
    for pt in report.localization_curve:
        assert pt.gt_count == 3
        if pt.recall is not None:
            assert 0.0 <= pt.recall <= 1.0

    # 8. Verify clean identity controls
    assert report.controls is not None
    assert report.controls.clean_controls_evaluated >= 1

    # 9. Verify manifest entry
    store = EvaluationStore(paths)
    runs = store.list_runs()
    assert any(r["run_id"] == report.run_id for r in runs)

    # 10. Verify validation passes
    validator = EvaluationValidator()
    val_rep, warnings = validator.validate_run_directory(run_dir)
    assert val_rep.run_id == report.run_id
    assert any("small" in w.lower() for w in warnings)
