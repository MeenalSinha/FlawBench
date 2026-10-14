"""Unit tests for Phase 4 detection schema and data structures."""

from __future__ import annotations

from pathlib import Path

import pytest

from flawbench.detection.schema import DetectionResult, FlawPrediction
from flawbench.exceptions import DetectionValidationError


def test_flaw_prediction_valid_bounds() -> None:
    pred = FlawPrediction(
        prediction_id="pred_test_1",
        flaw_type="pause_insertion",
        start_sample=1600,
        end_sample=3200,
        start_sec=0.1,
        end_sec=0.2,
        duration_sec=0.1,
        ref_start_sec=0.1,
        ref_end_sec=0.15,
        evidence={"mean_var_rms_db": -42.0},
        evidence_strength=0.85,
        detector_id="pause_detector_v4",
        decision_threshold=0.15,
    )
    assert pred.start_sample == 1600
    assert pred.end_sample == 3200
    assert pred.duration_sec == 0.1
    assert pred.evidence["mean_var_rms_db"] == -42.0

    d = pred.to_dict()
    assert d["prediction_id"] == "pred_test_1"
    assert d["start_sample"] == 1600

    restored = FlawPrediction.from_dict(d)
    assert restored.prediction_id == pred.prediction_id
    assert restored.start_sample == pred.start_sample
    assert restored.evidence_strength == pred.evidence_strength


@pytest.mark.parametrize(
    "start_sample,end_sample,start_sec,end_sec,strength",
    [
        (-1, 100, 0.0, 1.0, 0.5),  # negative start sample
        (100, 50, 0.0, 1.0, 0.5),  # inverted samples
        (0, 100, -0.5, 1.0, 0.5),  # negative start_sec
        (0, 100, 1.0, 0.5, 0.5),  # inverted seconds
        (0, 100, 0.0, 1.0, -0.1),  # strength < 0
        (0, 100, 0.0, 1.0, 1.1),  # strength > 1
    ],
)
def test_flaw_prediction_rejects_invalid_bounds(
    start_sample: int,
    end_sample: int,
    start_sec: float,
    end_sec: float,
    strength: float,
) -> None:
    with pytest.raises(DetectionValidationError):
        FlawPrediction(
            prediction_id="pred_bad",
            flaw_type="tempo_change",
            start_sample=start_sample,
            end_sample=end_sample,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=abs(end_sec - start_sec),
            evidence_strength=strength,
        )


def test_detection_result_serialization_and_atomic_save(tmp_path: Path) -> None:
    pred = FlawPrediction(
        prediction_id="pred_001",
        flaw_type="energy_reduction",
        start_sample=8000,
        end_sample=16000,
        start_sec=0.5,
        end_sec=1.0,
        duration_sec=0.5,
        ref_start_sec=0.5,
        ref_end_sec=1.0,
        evidence={"mean_delta_rms_db": -12.5},
        evidence_strength=0.9,
        detector_id="energy_detector_v4",
    )
    res = DetectionResult(
        schema_version="4.0",
        run_id="run_test_123",
        sample_id="sample-001",
        source_id="source-001",
        ref_artifact_id="ref-source-001",
        variant_artifact_id="var-sample-001",
        ref_checksum="a" * 64,
        variant_checksum="b" * 64,
        detector_version="4.0",
        config_fingerprint="c" * 16,
        created_at="2026-10-14T12:00:00Z",
        sample_rate=16000,
        total_samples=32000,
        duration_seconds=2.0,
        predictions=[pred],
        raw_candidates_count=2,
        metadata={"test_run": True},
    )

    save_path = tmp_path / "detections" / "sample-001_detection.json"
    written = res.save(save_path)
    assert written.is_file()

    loaded = DetectionResult.load(written)
    assert loaded.sample_id == "sample-001"
    assert loaded.source_id == "source-001"
    assert len(loaded.predictions) == 1
    assert loaded.predictions[0].flaw_type == "energy_reduction"
    assert loaded.predictions[0].evidence["mean_delta_rms_db"] == -12.5
    assert loaded.raw_candidates_count == 2
