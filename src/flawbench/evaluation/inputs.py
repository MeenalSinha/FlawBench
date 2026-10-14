"""Evaluation cohort resolution, integrity checks, and ground-truth pairing."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from flawbench.config import Config, load_config
from flawbench.dataset.io import read_jsonl_lenient
from flawbench.detection.pipeline import DetectionPipeline
from flawbench.detection.schema import FlawPrediction
from flawbench.exceptions import EvaluationInputError
from flawbench.paths import ProjectPaths

logger = logging.getLogger("flawbench.evaluation.inputs")


@dataclass
class EvaluationCohortItem:
    """A verified evaluation sample with isolated predictions and ground-truth events."""

    sample_id: str
    source_id: str
    split: str
    sample_rate: int
    duration_seconds: float
    ref_checksum: str
    variant_checksum: str
    ground_truth_events: list[dict[str, Any]]
    predicted_events: list[FlawPrediction]
    raw_record: dict[str, Any] = field(default_factory=dict)

    def to_record_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching compute_localization_curve sample format."""
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "gt_events": self.ground_truth_events,
            "predictions": self.predicted_events,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class EvaluationCohort:
    """Resolved cohort of evaluation samples with strict ground-truth isolation."""

    items: list[EvaluationCohortItem]
    excluded_samples: dict[str, list[str]]
    total_cohort_count: int
    is_synthetic_only: bool
    reference_sources: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def resolve_cohort(
        cls,
        paths: ProjectPaths,
        split: str = "test",
        config: Config | None = None,
        sample_id: str | None = None,
        force_detect: bool = False,
    ) -> EvaluationCohort:
        """Resolve evaluation cohort records with verified checksums and frozen predictions."""
        cfg = config or load_config()
        det_pipeline = DetectionPipeline(config=cfg, paths=paths)

        manifest_file = paths.manifest_file
        if not manifest_file.is_file():
            return cls(
                items=[],
                excluded_samples={"missing_dataset_manifest": [str(manifest_file)]},
                total_cohort_count=0,
                is_synthetic_only=True,
                reference_sources=[],
            )

        rows, problems = read_jsonl_lenient(manifest_file)
        if problems:
            logger.warning("Manifest lenient read issues: %s", problems)

        # Build split map: source_id -> split_name
        splits_file = paths.splits_file
        source_to_split: dict[str, str] = {}
        if splits_file.is_file():
            try:
                data = json.loads(splits_file.read_text(encoding="utf-8"))
                for s_name, s_ids in data.get("splits", {}).items():
                    for sid in s_ids:
                        source_to_split[sid] = s_name
            except Exception as exc:
                raise EvaluationInputError(
                    f"Failed to read splits file: {exc}", detail=str(splits_file)
                ) from exc

        # Read source registry to determine provenance (real speech vs synthetic)
        sources_file = paths.sources_file
        source_types: dict[str, str] = {}
        ref_sources: list[dict[str, Any]] = []
        if sources_file.is_file():
            s_rows, _ = read_jsonl_lenient(sources_file)
            for _, s_rec in s_rows:
                s_id = s_rec.get("source_id")
                if s_id:
                    source_types[s_id] = s_rec.get("source_type", "unknown")
                    ref_sources.append(s_rec)

        items: list[EvaluationCohortItem] = []
        excluded: dict[str, list[str]] = {}
        candidate_count = 0

        for _, record in rows:
            sid = record.get("sample_id")
            source_id = record.get("source_id")
            if not sid or not source_id:
                excluded.setdefault("malformed_record", []).append(str(record))
                continue

            if sample_id is not None and sid != sample_id:
                continue

            src_split = source_to_split.get(
                source_id, "test" if not source_to_split else "unassigned"
            )
            if split != "all" and src_split != split:
                continue

            candidate_count += 1

            # Check variant audio file existence & checksum
            gen_data = record.get("generated", {})
            rel_var = gen_data.get("path") or record.get("output_file")
            if not rel_var:
                excluded.setdefault("missing_output_file_field", []).append(sid)
                continue

            try:
                from flawbench.paths import resolve_within

                var_file = resolve_within(paths.generated_dir, str(rel_var))
            except Exception:
                var_file = paths.generated_dir / rel_var

            if not var_file.is_file():
                excluded.setdefault("output_file_not_found", []).append(str(var_file))
                continue

            # Verify variant checksum
            expected_cksum = gen_data.get("sha256") or record.get("output_sha256")
            actual_cksum = hashlib.sha256(var_file.read_bytes()).hexdigest()
            if expected_cksum and actual_cksum != expected_cksum:
                excluded.setdefault("checksum_mismatch", []).append(
                    f"{sid}: expected {expected_cksum[:8]}, got {actual_cksum[:8]}"
                )
                continue

            # Isolation Step 1: Freeze and retrieve independent predictions FIRST
            # Detection never sees the ground truth record annotations
            try:
                det_res = det_pipeline.detect_sample(sid, force=force_detect)
                predictions = list(det_res.predictions)
                sample_rate = det_res.sample_rate
                duration_seconds = det_res.duration_seconds
                ref_checksum = det_res.ref_checksum
                variant_checksum = det_res.variant_checksum
            except Exception as exc:
                excluded.setdefault("detection_failure", []).append(f"{sid}: {exc}")
                continue

            # Isolation Step 2: AFTER predictions are frozen, extract ground truth event
            gt_events: list[dict[str, Any]] = []
            anno = record.get("annotation", {})
            flaw_type = anno.get("flaw_type") or record.get("flaw_type")
            if flaw_type:
                gt_events.append(
                    {
                        "flaw_type": flaw_type,
                        "output_interval": anno.get("output_interval")
                        or record.get("output_interval", {}),
                        "annotation": anno,
                        "raw_record": record,
                    }
                )

            items.append(
                EvaluationCohortItem(
                    sample_id=sid,
                    source_id=source_id,
                    split=src_split,
                    sample_rate=sample_rate,
                    duration_seconds=duration_seconds,
                    ref_checksum=ref_checksum,
                    variant_checksum=variant_checksum,
                    ground_truth_events=gt_events,
                    predicted_events=predictions,
                    raw_record=record,
                )
            )

        # Determine if cohort is synthetic only
        is_synthetic = True
        for it in items:
            stype = source_types.get(it.source_id, "synthetic")
            if stype in ("self_recorded", "licensed", "public_domain", "other_permissioned"):
                is_synthetic = False
                break

        return cls(
            items=items,
            excluded_samples=excluded,
            total_cohort_count=candidate_count,
            is_synthetic_only=is_synthetic,
            reference_sources=ref_sources,
        )
