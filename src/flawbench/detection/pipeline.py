"""Phase 4 Flaw Detection Pipeline.

Coordinates dataset sample resolution, label-free feature comparison, detection engine
inference, caching, and result persistence.
"""

from __future__ import annotations

import logging
from typing import Any

from flawbench.config import Config, load_config
from flawbench.dataset.io import read_jsonl_lenient
from flawbench.detection.engine import DetectionEngine
from flawbench.detection.inputs import build_detection_input
from flawbench.detection.schema import DetectionResult
from flawbench.detection.store import DetectionStore
from flawbench.detection.validator import DetectionValidator
from flawbench.exceptions import DetectionError
from flawbench.features.pipeline import FeaturePipeline
from flawbench.paths import ProjectPaths

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """End-to-end flaw detection and temporal localization pipeline."""

    def __init__(
        self,
        config: Config | None = None,
        paths: ProjectPaths | None = None,
        store: DetectionStore | None = None,
        engine: DetectionEngine | None = None,
        feature_pipeline: FeaturePipeline | None = None,
    ) -> None:
        self.config = config or load_config()
        self.paths = paths or ProjectPaths(self.config)
        self.store = store or DetectionStore(
            detection_dir=self.paths.detection_dir,
            manifest_filename=self.config.detection.detection_manifest_filename,
            cache_policy=self.config.detection.cache_policy,
        )
        self.engine = engine or DetectionEngine(self.config.detection)
        self.feature_pipeline = feature_pipeline or FeaturePipeline(
            config=self.config, paths=self.paths
        )
        self.validator = DetectionValidator()

    def detect_sample(self, sample_id: str, force: bool = False) -> DetectionResult:
        """Run independent flaw detection on a single dataset sample."""
        config_fp = self.engine.config_fingerprint()

        # Check if sample exists in manifest first to get checksums for cache lookup
        manifest_file = self.paths.manifest_file
        if not manifest_file.is_file():
            raise DetectionError("Dataset manifest does not exist.", detail=str(manifest_file))

        rows, _ = read_jsonl_lenient(manifest_file)
        target_raw: dict[str, Any] | None = None
        for _, raw in rows:
            if raw.get("sample_id") == sample_id:
                target_raw = raw
                break

        if target_raw is None:
            raise DetectionError(f"Sample '{sample_id}' not found in dataset manifest.")

        gen = target_raw.get("generated", {})
        variant_checksum = str(gen.get("sha256", ""))
        source_id = str(target_raw.get("source_id", ""))

        ref_record = self.feature_pipeline.registry.get(source_id)
        ref_checksum = ref_record.sha256

        if not force:
            cached = self.store.get_valid(
                sample_id=sample_id,
                variant_checksum=variant_checksum,
                ref_checksum=ref_checksum,
                config_fingerprint=config_fp,
            )
            if cached is not None:
                logger.info(f"Using cached detection result for sample '{sample_id}'.")
                return cached

        # Build label-free input
        inp = build_detection_input(
            sample_id, self.feature_pipeline, force_extract=force, settings=self.config.detection
        )

        # Run inference
        result = self.engine.detect(inp)

        # Validate
        problems = self.validator.validate_result(result)
        if problems:
            raise DetectionError(
                f"Generated detection result failed validation: {'; '.join(problems)}",
                detail=str(problems),
            )

        # Persist
        self.store.save(result)
        return result

    def detect_all(self, force: bool = False) -> dict[str, Any]:
        """Run flaw detection across all committed samples in the dataset."""
        manifest_file = self.paths.manifest_file
        if not manifest_file.is_file():
            return {
                "total": 0,
                "processed": 0,
                "cached": 0,
                "failed": 0,
                "with_candidates": 0,
                "zero_candidates": 0,
                "results": [],
            }

        rows, _ = read_jsonl_lenient(manifest_file)
        results: list[DetectionResult] = []
        cached_count = 0
        failed_count = 0
        with_candidates_count = 0
        zero_candidates_count = 0

        config_fp = self.engine.config_fingerprint()

        for _, raw in rows:
            sample_id = raw.get("sample_id")
            if not sample_id:
                continue
            sample_id = str(sample_id)

            try:
                # Check cache status
                gen = raw.get("generated", {})
                v_sha = str(gen.get("sha256", ""))
                s_id = str(raw.get("source_id", ""))
                ref_rec = self.feature_pipeline.registry.get(s_id)
                r_sha = ref_rec.sha256

                is_cached = (
                    not force
                    and self.store.get_valid(sample_id, v_sha, r_sha, config_fp) is not None
                )

                res = self.detect_sample(sample_id, force=force)
                results.append(res)

                if is_cached:
                    cached_count += 1
                if len(res.predictions) > 0:
                    with_candidates_count += 1
                else:
                    zero_candidates_count += 1

            except Exception as exc:
                logger.error(f"Detection failed on sample '{sample_id}': {exc}")
                failed_count += 1

        return {
            "total": len(rows),
            "processed": len(results),
            "cached": cached_count,
            "failed": failed_count,
            "with_candidates": with_candidates_count,
            "zero_candidates": zero_candidates_count,
            "results": results,
        }

    def list_detections(self) -> list[dict[str, Any]]:
        """List summary records from detections_manifest.jsonl."""
        return self.store.list_manifest()

    def validate_detections(self) -> dict[str, list[str]]:
        """Validate all saved detection artifacts in the detections directory."""
        return self.validator.validate_all(self.paths.detection_dir)
