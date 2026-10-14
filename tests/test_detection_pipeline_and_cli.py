"""Integration tests for DetectionPipeline and CLI detect commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flawbench.__main__ import main
from flawbench.config import Config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.schema import SourceRecord
from flawbench.detection.pipeline import DetectionPipeline
from flawbench.paths import ProjectPaths


def test_detection_pipeline_and_cli_end_to_end(
    ds_config: Config,
    ds_root: Path,
    source: SourceRecord,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(ds_root))
    paths = ProjectPaths(ds_config)
    gen = DatasetGenerator(ds_config)

    # 1. Generate two variants: one pause_insertion and one tempo_change
    res_pause = gen.generate_variant(
        source.source_id,
        "pause_insertion",
        severity="medium",
        placement={"mode": "explicit", "at_sec": 2.0},
    )
    _res_tempo = gen.generate_variant(
        source.source_id,
        "tempo_change",
        severity="medium",
        placement={"mode": "explicit", "start_sec": 2.0},
    )

    sample_id = res_pause.record.sample_id
    pipeline = DetectionPipeline(config=ds_config, paths=paths)

    # 2. Pipeline single sample detection
    det_res = pipeline.detect_sample(sample_id, force=True)
    assert det_res.sample_id == sample_id
    assert det_res.source_id == source.source_id
    assert det_res.duration_seconds > 0

    # 3. Cache hit verification
    cached_res = pipeline.detect_sample(sample_id, force=False)
    assert cached_res.run_id == det_res.run_id

    # 4. Pipeline batch detection
    batch_summary = pipeline.detect_all(force=False)
    assert batch_summary["total"] == 2
    assert batch_summary["processed"] == 2
    assert batch_summary["failed"] == 0

    # 5. CLI: detect --sample <sample_id>
    export_file = paths.output_dir / "test_pred_export.json"
    code = main(
        [
            "detect",
            "--sample",
            sample_id,
            "--export",
            str(export_file),
        ]
    )
    assert code == 0
    # Note: if ds_config has no config_file on disk, pass env or run without explicit --config
    # Let's test main() with project root set or direct dispatch
    out, _ = capsys.readouterr()
    assert export_file.is_file()
    exported_data = json.loads(export_file.read_text(encoding="utf-8"))
    assert exported_data["sample_id"] == sample_id

    # 6. CLI: detect list
    code_list = main(["detect", "list"])
    assert code_list == 0
    out_list, _ = capsys.readouterr()
    assert "Stored detections:" in out_list

    # 7. CLI: detect validate
    code_val = main(["detect", "validate"])
    out_val, _ = capsys.readouterr()
    assert "Detection results validation:" in out_val
    assert code_val == 0
