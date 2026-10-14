"""Unit tests for DetectionStore and DetectionValidator."""

from __future__ import annotations

from pathlib import Path

from flawbench.detection.schema import DetectionResult, FlawPrediction
from flawbench.detection.store import DetectionStore
from flawbench.detection.validator import DetectionValidator


def _make_dummy_result(
    sample_id: str = "sample-001",
    ref_checksum: str = "a" * 64,
    variant_checksum: str = "b" * 64,
    config_fingerprint: str = "c" * 16,
) -> DetectionResult:
    pred = FlawPrediction(
        prediction_id="pred_1",
        flaw_type="pause_insertion",
        start_sample=8000,
        end_sample=16000,
        start_sec=0.5,
        end_sec=1.0,
        duration_sec=0.5,
        ref_start_sec=0.5,
        ref_end_sec=0.8,
        evidence={"mean_var_rms_db": -42.0},
        evidence_strength=0.8,
        detector_id="pause_detector_v4",
    )
    return DetectionResult(
        schema_version="4.0",
        run_id="run_1",
        sample_id=sample_id,
        source_id="src-001",
        ref_artifact_id="ref-001",
        variant_artifact_id="var-001",
        ref_checksum=ref_checksum,
        variant_checksum=variant_checksum,
        detector_version="4.0",
        config_fingerprint=config_fingerprint,
        created_at="2026-10-14T12:00:00Z",
        sample_rate=16000,
        total_samples=32000,
        duration_seconds=2.0,
        predictions=[pred],
        raw_candidates_count=1,
    )


def test_detection_store_save_and_manifest(tmp_path: Path) -> None:
    store = DetectionStore(detection_dir=tmp_path / "detections")
    res = _make_dummy_result()

    saved_path = store.save(res)
    assert saved_path.is_file()
    assert store.manifest_file.is_file()

    manifest_entries = store.list_manifest()
    assert len(manifest_entries) == 1
    assert manifest_entries[0]["sample_id"] == "sample-001"
    assert manifest_entries[0]["predictions_count"] == 1


def test_detection_store_caching_and_invalidation(tmp_path: Path) -> None:
    store = DetectionStore(detection_dir=tmp_path / "detections", cache_policy="skip_if_valid")
    res = _make_dummy_result(
        sample_id="s1",
        ref_checksum="1" * 64,
        variant_checksum="2" * 64,
        config_fingerprint="fp1",
    )
    store.save(res)

    # 1. Cache hit with identical parameters
    hit = store.get_valid(
        "s1", variant_checksum="2" * 64, ref_checksum="1" * 64, config_fingerprint="fp1"
    )
    assert hit is not None
    assert hit.sample_id == "s1"

    # 2. Invalidation: changed variant checksum
    miss1 = store.get_valid(
        "s1", variant_checksum="9" * 64, ref_checksum="1" * 64, config_fingerprint="fp1"
    )
    assert miss1 is None

    # 3. Invalidation: changed config fingerprint
    miss2 = store.get_valid(
        "s1", variant_checksum="2" * 64, ref_checksum="1" * 64, config_fingerprint="fp_new"
    )
    assert miss2 is None

    # 4. Cache policy always_recompute bypasses cache
    recompute_store = DetectionStore(
        detection_dir=tmp_path / "detections", cache_policy="always_recompute"
    )
    miss3 = recompute_store.get_valid(
        "s1", variant_checksum="2" * 64, ref_checksum="1" * 64, config_fingerprint="fp1"
    )
    assert miss3 is None


def test_detection_validator_clean_result() -> None:
    validator = DetectionValidator()
    res = _make_dummy_result()
    problems = validator.validate_result(res)
    assert problems == []


def test_detection_validator_detects_flaws(tmp_path: Path) -> None:
    validator = DetectionValidator()

    # Out-of-bounds prediction
    p_bad = FlawPrediction(
        prediction_id="p_bad",
        flaw_type="tempo_change",
        start_sample=40000,  # exceeds total_samples=32000
        end_sample=45000,
        start_sec=2.5,
        end_sec=2.8125,
        duration_sec=0.3125,
        evidence={"rate": float("nan")},  # non-finite!
        evidence_strength=0.5,
    )

    res = DetectionResult(
        schema_version="4.0",
        run_id="run_bad",
        sample_id="",  # empty!
        source_id="src1",
        ref_artifact_id="ref1",
        variant_artifact_id="var1",
        ref_checksum="a" * 64,
        variant_checksum="b" * 64,
        detector_version="4.0",
        config_fingerprint="c" * 16,
        created_at="2026-10-14T00:00:00Z",
        sample_rate=16000,
        total_samples=32000,
        duration_seconds=2.0,
        predictions=[p_bad],
        raw_candidates_count=1,
        metadata={"ground_truth": "leaked!"},  # forbidden!
    )

    problems = validator.validate_result(res)
    assert any("sample_id is empty" in p for p in problems)
    assert any("exceeds total_samples" in p for p in problems)
    assert any("non-finite" in p for p in problems)
    assert any("forbidden" in p or "leakage" in p.lower() for p in problems)
