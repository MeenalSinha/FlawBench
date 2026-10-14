"""Phase 4 Detection Result Store.

Manages persistence of DetectionResult JSON files and maintains the
detection manifest with caching and checksum-based invalidation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flawbench.dataset.io import read_jsonl_lenient
from flawbench.detection.schema import DetectionResult
from flawbench.exceptions import DetectionStoreError


class DetectionStore:
    """Persists and retrieves flaw detection artifacts and maintains detections_manifest.jsonl."""

    def __init__(
        self,
        detection_dir: Path,
        manifest_filename: str = "detections_manifest.jsonl",
        cache_policy: str = "skip_if_valid",
    ) -> None:
        self.detection_dir = Path(detection_dir).resolve()
        self.manifest_file = self.detection_dir / manifest_filename
        self.cache_policy = cache_policy
        self.detection_dir.mkdir(parents=True, exist_ok=True)

    def _result_path(self, sample_id: str) -> Path:
        return self.detection_dir / f"{sample_id}_detection.json"

    def get(self, sample_id: str) -> DetectionResult | None:
        """Load a stored DetectionResult if the file exists and is valid JSON."""
        path = self._result_path(sample_id)
        if not path.is_file():
            return None
        try:
            return DetectionResult.load(path)
        except Exception as exc:
            raise DetectionStoreError(
                f"Failed to load detection result for sample '{sample_id}'.",
                detail=str(exc),
            ) from exc

    def get_valid(
        self,
        sample_id: str,
        variant_checksum: str,
        ref_checksum: str,
        config_fingerprint: str,
    ) -> DetectionResult | None:
        """Retrieve a cached DetectionResult if valid under current checksums and configuration."""
        if self.cache_policy == "always_recompute":
            return None

        result = self.get(sample_id)
        if result is None:
            return None

        # Verify checksums and config fingerprint
        if (
            result.variant_checksum == variant_checksum
            and result.ref_checksum == ref_checksum
            and result.config_fingerprint == config_fingerprint
        ):
            return result

        return None

    def _upsert_manifest_entry(self, entry: dict[str, Any]) -> None:
        entries = self.list_manifest()
        updated = False
        new_entries: list[dict[str, Any]] = []
        for e in entries:
            if e.get("sample_id") == entry.get("sample_id"):
                new_entries.append(entry)
                updated = True
            else:
                new_entries.append(e)
        if not updated:
            new_entries.append(entry)

        temp_file = self.manifest_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")
        temp_file.replace(self.manifest_file)

    def save(self, result: DetectionResult) -> Path:
        """Save a DetectionResult and record an entry in the manifest."""
        path = self._result_path(result.sample_id)
        result.save(path)

        # Record in manifest
        entry = {
            "sample_id": result.sample_id,
            "source_id": result.source_id,
            "run_id": result.run_id,
            "artifact_path": path.name,
            "ref_checksum": result.ref_checksum,
            "variant_checksum": result.variant_checksum,
            "config_fingerprint": result.config_fingerprint,
            "detector_version": result.detector_version,
            "predictions_count": len(result.predictions),
            "raw_candidates_count": result.raw_candidates_count,
            "created_at": result.created_at,
        }
        self._upsert_manifest_entry(entry)
        return path

    def list_manifest(self) -> list[dict[str, Any]]:
        """List all entries recorded in detections_manifest.jsonl."""
        if not self.manifest_file.is_file():
            return []
        rows, problems = read_jsonl_lenient(self.manifest_file)
        if problems:
            raise DetectionStoreError("Detection manifest contains malformed lines.")
        return [raw for _, raw in rows]
