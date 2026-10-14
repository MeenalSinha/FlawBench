"""Tests for Phase 3 FeaturePipeline integration with dataset registry and generator."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from flawbench.config import load_config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.registry import SourceRegistry
from flawbench.features.pipeline import FeaturePipeline
from flawbench.paths import ProjectPaths


def _write_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(rate * seconds)) / rate
    y = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    sf.write(str(path), y, rate, format="WAV")


@pytest.fixture
def test_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ProjectPaths, str, str]:
    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("FLAWBENCH_ALLOW_OVERWRITE", "true")
    config = load_config()
    paths = ProjectPaths(config)
    paths.ensure_output_directories()

    # 1. Register reference
    wav_path = paths.reference_dir / "sample_ref.wav"
    _write_wav(wav_path, seconds=1.0)
    registry = SourceRegistry(config, paths)
    source = registry.register(
        wav_path,
        transcript_text="Test utterance for feature pipeline",
        source_type="self_recorded",
        attribution="Audio Engineer",
        license="CC0-1.0",
        rights_confirmed=True,
    )

    # 2. Generate a variant
    generator = DatasetGenerator(config, paths, registry)
    res = generator.generate_variant(
        source_id=source.source_id,
        flaw_type="energy_reduction",
        parameters={"gain_db": -12.0, "segment_sec": 0.4},
        placement={"mode": "explicit", "start_sec": 0.3},
    )
    sample_id = res.record.sample_id
    return paths, source.source_id, sample_id


def test_pipeline_extract_and_compare(test_dataset: tuple[ProjectPaths, str, str]) -> None:
    paths, source_id, sample_id = test_dataset
    config = load_config()
    pipeline = FeaturePipeline(config, paths)

    # Extract reference
    ref_feat = pipeline.extract_reference(source_id)
    assert ref_feat.num_frames > 0
    assert ref_feat.metadata.role == "reference"

    # Extract variant
    var_feat = pipeline.extract_variant(sample_id)
    assert var_feat.num_frames > 0
    assert var_feat.metadata.role == "variant"

    # Compare
    cmp_res = pipeline.compare_variant_to_reference(sample_id)
    assert cmp_res.ref_source_id == source_id
    assert cmp_res.variant_sample_id == sample_id
    assert cmp_res.path_length > 0

    # Verify manifest
    manifest_entries = pipeline.store.list_manifest()
    assert len(manifest_entries) == 2

    # Export summaries
    summaries = pipeline.export_feature_summaries()
    assert len(summaries) == 2
    roles = {s["role"] for s in summaries}
    assert roles == {"reference", "variant"}


def test_pipeline_extract_dataset_all(test_dataset: tuple[ProjectPaths, str, str]) -> None:
    paths, source_id, sample_id = test_dataset
    config = load_config()
    pipeline = FeaturePipeline(config, paths)

    counts = pipeline.extract_dataset()
    assert counts["references_extracted"] == 1
    assert counts["variants_extracted"] == 1

    all_comparisons = pipeline.compare_all()
    assert len(all_comparisons) == 1
