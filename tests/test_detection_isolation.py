"""Mandatory Ground-Truth Isolation and Leakage Prevention Tests.

Verifies:
1. Detection receives strictly label-free data structures (DetectionInput).
2. Changing ground-truth flaw labels or injection annotations in the manifest
   without changing audio files produces IDENTICAL detection predictions.
3. Inferred detection intervals are mathematically derived from acoustic evidence,
   never copied from ground-truth injection intervals.
4. DetectionResult does not leak or contain forbidden ground-truth fields.
"""

from __future__ import annotations

from flawbench.config import Config
from flawbench.dataset.io import write_jsonl
from flawbench.dataset.schema import SourceRecord
from flawbench.detection.pipeline import DetectionPipeline
from flawbench.detection.schema import DetectionInput, DetectionResult, FlawPrediction
from flawbench.paths import ProjectPaths


def test_detection_input_contains_no_ground_truth_fields() -> None:
    """DetectionInput dataclass must not have flaw_type, severity, or ground-truth interval fields."""
    forbidden = {
        "flaw_type",
        "severity",
        "output_interval",
        "source_interval",
        "injected_interval",
        "experimental_severity",
        "ground_truth",
    }
    field_names = set(DetectionInput.__dataclass_fields__.keys())
    leakage = field_names.intersection(forbidden)
    assert not leakage, f"DetectionInput contains forbidden ground-truth fields: {leakage}"


def test_modifying_manifest_ground_truth_does_not_affect_detection(
    ds_config: Config,
    source: SourceRecord,
) -> None:
    """Proves that altering ground-truth flaw annotations in the manifest leaves predictions 100% identical."""
    paths = ProjectPaths(ds_config)
    from flawbench.dataset.generator import DatasetGenerator
    from flawbench.dataset.io import read_jsonl

    gen = DatasetGenerator(ds_config)
    gen_res = gen.generate_variant(
        source.source_id,
        "pause_insertion",
        severity="medium",
        placement={"mode": "explicit", "at_sec": 2.0},
    )
    sample_id = gen_res.record.sample_id

    pipeline = DetectionPipeline(config=ds_config, paths=paths)
    result_a = pipeline.detect_sample(sample_id, force=True)

    # Read the manifest and alter the ground-truth annotation fields
    rows = read_jsonl(paths.manifest_file, what="manifest")
    assert len(rows) == 1
    # Mutate annotation completely
    rows[0]["annotation"]["flaw_type"] = "energy_reduction"
    rows[0]["annotation"]["severity"] = "extreme"
    rows[0]["annotation"]["output_interval"]["start_sample"] = 100
    rows[0]["annotation"]["output_interval"]["end_sample"] = 200
    write_jsonl(paths.manifest_file, rows)

    # Re-run detection on the modified manifest
    result_b = pipeline.detect_sample(sample_id, force=True)

    # Predictions MUST remain identical because detector does not inspect annotation fields!
    assert len(result_a.predictions) > 0
    assert len(result_a.predictions) == len(result_b.predictions)
    for p_a, p_b in zip(result_a.predictions, result_b.predictions, strict=True):
        assert p_a.flaw_type == p_b.flaw_type
        assert p_a.start_sample == p_b.start_sample
        assert p_a.end_sample == p_b.end_sample
        assert p_a.evidence == p_b.evidence


def test_predictions_do_not_contain_ground_truth_leakage() -> None:
    pred = FlawPrediction(
        prediction_id="p1",
        flaw_type="pause_insertion",
        start_sample=8000,
        end_sample=12800,
        start_sec=0.5,
        end_sec=0.8,
        duration_sec=0.3,
        evidence={"mean_var_rms_db": -45.0},
        evidence_strength=0.8,
    )
    res = DetectionResult(
        schema_version="4.0",
        run_id="run_1",
        sample_id="s1",
        source_id="src1",
        ref_artifact_id="ref1",
        variant_artifact_id="var1",
        ref_checksum="a" * 64,
        variant_checksum="b" * 64,
        detector_version="4.0",
        config_fingerprint="c" * 16,
        created_at="2026-10-14T00:00:00Z",
        sample_rate=16000,
        total_samples=16000,
        duration_seconds=1.0,
        predictions=[pred],
        raw_candidates_count=1,
    )

    data = res.to_dict()
    # Ensure neither result nor predictions contain ground-truth keys
    forbidden = {
        "ground_truth",
        "output_interval",
        "source_interval",
        "severity",
        "target_interval",
    }
    for key in forbidden:
        assert key not in data
        assert key not in data["predictions"][0]
        assert key not in data["predictions"][0]["evidence"]
