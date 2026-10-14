"""Phase 3 Feature Extraction & Comparison Pipeline.

Connects the acoustic feature extractor, reference comparator, and feature store
to the Phase 2 dataset manifest and source registry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from flawbench.audio.loader import load_audio
from flawbench.config import Config, load_config
from flawbench.dataset.io import read_jsonl_lenient
from flawbench.dataset.registry import SourceRegistry
from flawbench.dataset.schema import SampleRecord
from flawbench.exceptions import DatasetError, FeatureError
from flawbench.features.comparison import ComparisonResult, compare_features
from flawbench.features.extractor import FeatureResult, extract_features
from flawbench.features.schema import AudioInput, AudioRole
from flawbench.features.store import FeatureStore
from flawbench.paths import ProjectPaths, resolve_within


class FeaturePipeline:
    """Orchestrates acoustic feature extraction and reference comparison across a dataset."""

    def __init__(
        self,
        config: Config | None = None,
        paths: ProjectPaths | None = None,
        store: FeatureStore | None = None,
    ) -> None:
        self.config = config or load_config()
        self.paths = paths or ProjectPaths(self.config)
        self.store = store or FeatureStore(
            features_dir=self.paths.features_dir,
            manifest_filename=self.config.features.feature_manifest_filename,
            cache_policy=self.config.features.cache_policy,
        )
        self.registry = SourceRegistry(self.config, self.paths)

    def _config_fingerprint(self) -> str:
        f = self.config.features
        raw = f"{f.frame_ms}_{f.hop_ms}_{f.n_mfcc}_{f.pitch_backend}_{f.f0_min_hz}_{f.f0_max_hz}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ---------------------------------------------------------------- Reference extraction
    def extract_reference(self, source_id: str, force: bool = False) -> FeatureResult:
        """Extract acoustic features for a registered reference recording."""
        record = self.registry.get(source_id)
        path = self.registry.audio_path(record)

        fingerprint = self._config_fingerprint()
        if not force:
            cached = self.store.get_by_checksum(record.sha256, fingerprint)
            if cached is not None:
                return cached

        audio = self.registry.load_audio(record)
        audio_input = AudioInput(
            artifact_id=f"ref-{record.source_id}",
            source_id=record.source_id,
            sample_id=None,
            role=AudioRole.REFERENCE,
            audio_path=path,
            sha256=record.sha256,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
            num_samples=audio.num_samples,
            duration_seconds=audio.duration_seconds,
        )

        result = extract_features(
            audio=audio,
            feat_cfg=self.config.features,
            audio_input=audio_input,
            config=self.config,
        )
        self.store.save(result)
        return result

    def extract_all_references(self, force: bool = False) -> list[FeatureResult]:
        """Extract acoustic features for all registered references."""
        results: list[FeatureResult] = []
        for record in self.registry.all():
            results.append(self.extract_reference(record.source_id, force=force))
        return results

    # ---------------------------------------------------------------- Variant extraction
    def _load_manifest_sample(self, sample_id: str) -> tuple[SampleRecord, Path]:
        """Find a sample in manifest.jsonl and resolve its audio file."""
        if not self.paths.manifest_file.is_file():
            raise FeatureError(
                "Dataset manifest does not exist.", detail=str(self.paths.manifest_file)
            )

        rows, problems = read_jsonl_lenient(self.paths.manifest_file)
        if problems:
            raise DatasetError("Dataset manifest has malformed lines.")

        for _, raw in rows:
            if raw.get("sample_id") == sample_id:
                rec = SampleRecord.from_dict(raw)
                path = resolve_within(self.paths.generated_dir, str(rec.data["generated"]["path"]))
                return rec, path

        raise FeatureError(f"Sample '{sample_id}' not found in manifest.")

    def extract_variant(self, sample_id: str, force: bool = False) -> FeatureResult:
        """Extract acoustic features for a generated variant."""
        record, path = self._load_manifest_sample(sample_id)
        gen = record.data["generated"]
        sha256 = str(gen["sha256"])

        fingerprint = self._config_fingerprint()
        if not force:
            cached = self.store.get_by_checksum(sha256, fingerprint)
            if cached is not None:
                return cached

        audio = load_audio(path, self.config.audio)
        audio_input = AudioInput(
            artifact_id=f"var-{sample_id}",
            source_id=record.source_id,
            sample_id=sample_id,
            role=AudioRole.VARIANT,
            audio_path=path,
            sha256=sha256,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
            num_samples=audio.num_samples,
            duration_seconds=audio.duration_seconds,
        )

        result = extract_features(
            audio=audio,
            feat_cfg=self.config.features,
            audio_input=audio_input,
            config=self.config,
        )
        self.store.save(result)
        return result

    def extract_all_variants(self, force: bool = False) -> list[FeatureResult]:
        """Extract acoustic features for all committed variants in the manifest."""
        if not self.paths.manifest_file.is_file():
            return []

        rows, _ = read_jsonl_lenient(self.paths.manifest_file)
        results: list[FeatureResult] = []
        for _, raw in rows:
            sample_id = raw.get("sample_id")
            if sample_id:
                results.append(self.extract_variant(str(sample_id), force=force))
        return results

    def extract_dataset(self, force: bool = False) -> dict[str, int]:
        """Run feature extraction for all references and all variants in the dataset."""
        ref_results = self.extract_all_references(force=force)
        var_results = self.extract_all_variants(force=force)
        return {
            "references_extracted": len(ref_results),
            "variants_extracted": len(var_results),
            "total_artifacts": len(ref_results) + len(var_results),
        }

    # ---------------------------------------------------------------- Reference comparison
    def compare_variant_to_reference(
        self, sample_id: str, force_extract: bool = False
    ) -> ComparisonResult:
        """Compare a variant against its reference recording."""
        rec, _ = self._load_manifest_sample(sample_id)
        ref_feat = self.extract_reference(rec.source_id, force=force_extract)
        var_feat = self.extract_variant(sample_id, force=force_extract)

        result = compare_features(ref_feat, var_feat)

        # Save comparison bundle
        comp_dir = self.paths.features_dir / "comparisons"
        comp_dir.mkdir(parents=True, exist_ok=True)
        save_path = comp_dir / f"{sample_id}_vs_{rec.source_id}.npz"
        result.save(save_path)
        return result

    def compare_all(self, force_extract: bool = False) -> list[ComparisonResult]:
        """Compare all committed variants in the manifest to their respective references."""
        if not self.paths.manifest_file.is_file():
            return []

        rows, _ = read_jsonl_lenient(self.paths.manifest_file)
        comparisons: list[ComparisonResult] = []
        for _, raw in rows:
            sample_id = raw.get("sample_id")
            if sample_id:
                comparisons.append(
                    self.compare_variant_to_reference(str(sample_id), force_extract=force_extract)
                )
        return comparisons

    # ---------------------------------------------------------------- Summaries & Export
    def export_feature_summaries(self, output_path: Path | None = None) -> list[dict[str, Any]]:
        """Export tabular summary metrics for all stored feature artifacts."""
        entries = self.store.list_manifest()
        summaries: list[dict[str, Any]] = []

        for e in entries:
            try:
                feat = self.store.load(e.artifact_id)
                item: dict[str, Any] = {
                    "artifact_id": e.artifact_id,
                    "source_id": e.source_id,
                    "sample_id": e.sample_id,
                    "role": e.role,
                    "num_frames": feat.num_frames,
                    "duration_seconds": feat.metadata.duration_seconds,
                    "sample_rate": feat.metadata.sample_rate,
                }
                item.update(feat.summary_stats)
                summaries.append(item)
            except Exception:
                continue

        if output_path is not None:
            output_path = Path(output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

        return summaries
