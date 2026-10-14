"""Tests for Phase 3 FeatureStore persistence, lookup, and cache management."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flawbench.exceptions import FeatureStoreError
from flawbench.features.extractor import extract_features
from flawbench.features.schema import AudioInput, AudioRole
from flawbench.features.store import FeatureStore


def _synthetic_result(artifact_id: str, audio_sha: str = "abc123456789") -> tuple[np.ndarray, AudioInput]:
    sr = 16000
    y = np.sin(2 * np.pi * 300 * np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)).astype(np.float32)
    inp = AudioInput(
        artifact_id=artifact_id,
        source_id="src-test",
        sample_id=None,
        role=AudioRole.REFERENCE,
        audio_path=Path("dummy.wav"),
        sha256=audio_sha,
        sample_rate=sr,
        num_channels=1,
        num_samples=len(y),
        duration_seconds=0.3,
    )
    return y, inp


def test_store_save_load_and_manifest(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    y, inp = _synthetic_result("feat-test-1")
    feat = extract_features(y, sample_rate=16000, audio_input=inp)

    npz_path, json_path = store.save(feat)
    assert npz_path.is_file()
    assert json_path.is_file()
    assert store.manifest_file.is_file()

    entries = store.list_manifest()
    assert len(entries) == 1
    assert entries[0].artifact_id == "feat-test-1"
    assert entries[0].audio_sha256 == "abc123456789"

    loaded = store.load("feat-test-1")
    assert loaded.metadata.artifact_id == "feat-test-1"
    assert loaded.num_frames == feat.num_frames


def test_store_cache_hit(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path, cache_policy="skip_if_valid")
    y, inp = _synthetic_result("feat-cached", audio_sha="sha-match-123")
    feat = extract_features(y, sample_rate=16000, audio_input=inp)
    store.save(feat)

    cached = store.get_by_checksum("sha-match-123", feat.metadata.config_fingerprint)
    assert cached is not None
    assert cached.metadata.artifact_id == "feat-cached"

    # Different fingerprint or sha should miss
    assert store.get_by_checksum("sha-match-123", "wrong_fingerprint") is None
    assert store.get_by_checksum("different_sha", feat.metadata.config_fingerprint) is None


def test_store_validation_detects_missing_file(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    y, inp = _synthetic_result("feat-delete-me")
    feat = extract_features(y, sample_rate=16000, audio_input=inp)
    npz_path, _ = store.save(feat)

    val1 = store.validate_store()
    assert val1["is_valid"] is True

    # Delete the npz file
    npz_path.unlink()
    val2 = store.validate_store()
    assert val2["is_valid"] is False
    assert "feat-delete-me" in val2["missing_artifacts"]


def test_store_load_missing_raises_error(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    with pytest.raises(FeatureStoreError, match="does not exist"):
        store.load("nonexistent-artifact")
