"""Phase 3 Feature Storage and Artifact Persistence.

Manages persistent storage, lookup, and validation of feature artifacts (.npz and .json),
along with the centralized JSONL feature manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flawbench.exceptions import FeatureStoreError
from flawbench.features.extractor import FeatureResult
from flawbench.features.schema import FEATURE_SCHEMA_VERSION, AudioRole


@dataclass(frozen=True)
class FeatureManifestEntry:
    artifact_id: str
    source_id: str
    sample_id: str | None
    role: str
    audio_sha256: str
    config_fingerprint: str
    schema_version: str
    duration_seconds: float
    num_frames: int
    npz_path: str
    json_path: str
    created_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "source_id": self.source_id,
            "sample_id": self.sample_id,
            "role": self.role,
            "audio_sha256": self.audio_sha256,
            "config_fingerprint": self.config_fingerprint,
            "schema_version": self.schema_version,
            "duration_seconds": self.duration_seconds,
            "num_frames": self.num_frames,
            "npz_path": self.npz_path,
            "json_path": self.json_path,
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureManifestEntry:
        return cls(
            artifact_id=str(data["artifact_id"]),
            source_id=str(data["source_id"]),
            sample_id=data.get("sample_id"),
            role=str(data["role"]),
            audio_sha256=str(data["audio_sha256"]),
            config_fingerprint=str(data["config_fingerprint"]),
            schema_version=str(data.get("schema_version", FEATURE_SCHEMA_VERSION)),
            duration_seconds=float(data["duration_seconds"]),
            num_frames=int(data["num_frames"]),
            npz_path=str(data["npz_path"]),
            json_path=str(data["json_path"]),
            created_utc=str(data["created_utc"]),
        )


class FeatureStore:
    """Storage manager for extracted acoustic features."""

    def __init__(
        self,
        features_dir: Path,
        manifest_filename: str = "feature_manifest.jsonl",
        cache_policy: str = "skip_if_valid",
    ) -> None:
        self.features_dir = Path(features_dir).resolve()
        self.manifest_file = self.features_dir / manifest_filename
        self.cache_policy = cache_policy

    def save(self, result: FeatureResult) -> tuple[Path, Path]:
        """Save feature result and record entry in manifest."""
        self.features_dir.mkdir(parents=True, exist_ok=True)
        npz_path, json_path = result.save(self.features_dir)

        # Update manifest
        entry = FeatureManifestEntry(
            artifact_id=result.metadata.artifact_id,
            source_id=result.metadata.source_id,
            sample_id=result.metadata.sample_id,
            role=result.metadata.role,
            audio_sha256=result.metadata.audio_sha256,
            config_fingerprint=result.metadata.config_fingerprint,
            schema_version=result.metadata.schema_version,
            duration_seconds=result.metadata.duration_seconds,
            num_frames=result.num_frames,
            npz_path=npz_path.name,
            json_path=json_path.name,
            created_utc=result.metadata.created_utc,
        )

        self._upsert_manifest_entry(entry)
        return npz_path, json_path

    def load(self, artifact_id: str) -> FeatureResult:
        """Load feature result by artifact id."""
        npz_path = self.features_dir / f"{artifact_id}.npz"
        json_path = self.features_dir / f"{artifact_id}.json"

        if not npz_path.is_file() or not json_path.is_file():
            raise FeatureStoreError(
                f"Feature artifact '{artifact_id}' does not exist on disk.",
                detail=f"expected npz: {npz_path}, json: {json_path}",
            )
        try:
            return FeatureResult.load(npz_path, json_path)
        except Exception as exc:
            raise FeatureStoreError(
                f"Failed to load feature artifact '{artifact_id}'.", detail=str(exc)
            ) from exc

    def list_manifest(self) -> list[FeatureManifestEntry]:
        """Read and return all entries in the feature manifest."""
        if not self.manifest_file.is_file():
            return []

        entries: list[FeatureManifestEntry] = []
        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        entries.append(FeatureManifestEntry.from_dict(json.loads(stripped)))
        except Exception as exc:
            raise FeatureStoreError("Failed to parse feature manifest.", detail=str(exc)) from exc
        return entries

    def get_by_checksum(
        self, audio_sha256: str, config_fingerprint: str
    ) -> FeatureResult | None:
        """Look up cached FeatureResult by audio checksum and config fingerprint."""
        if self.cache_policy == "always_recompute":
            return None

        entries = self.list_manifest()
        for e in reversed(entries):
            if e.audio_sha256 == audio_sha256 and e.config_fingerprint == config_fingerprint:
                npz = self.features_dir / e.npz_path
                json_p = self.features_dir / e.json_path
                if npz.is_file() and json_p.is_file():
                    try:
                        return FeatureResult.load(npz, json_p)
                    except Exception:
                        continue
        return None

    def get_by_source_id(
        self, source_id: str, role: str = AudioRole.REFERENCE.value
    ) -> FeatureResult | None:
        """Look up FeatureResult by registered source id and role."""
        for e in reversed(self.list_manifest()):
            if e.source_id == source_id and e.role == role:
                npz = self.features_dir / e.npz_path
                json_p = self.features_dir / e.json_path
                if npz.is_file() and json_p.is_file():
                    try:
                        return FeatureResult.load(npz, json_p)
                    except Exception:
                        continue
        return None

    def get_by_sample_id(self, sample_id: str) -> FeatureResult | None:
        """Look up FeatureResult by dataset variant sample id."""
        for e in reversed(self.list_manifest()):
            if e.sample_id == sample_id:
                npz = self.features_dir / e.npz_path
                json_p = self.features_dir / e.json_path
                if npz.is_file() and json_p.is_file():
                    try:
                        return FeatureResult.load(npz, json_p)
                    except Exception:
                        continue
        return None

    def _upsert_manifest_entry(self, new_entry: FeatureManifestEntry) -> None:
        """Update existing manifest entry or append if new."""
        entries = self.list_manifest()
        updated = False
        new_entries: list[FeatureManifestEntry] = []
        for e in entries:
            if e.artifact_id == new_entry.artifact_id:
                new_entries.append(new_entry)
                updated = True
            else:
                new_entries.append(e)

        if not updated:
            new_entries.append(new_entry)

        # Atomic rewrite
        temp_file = self.manifest_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            for e in new_entries:
                f.write(json.dumps(e.to_dict()) + "\n")
        temp_file.replace(self.manifest_file)

    def validate_store(self) -> dict[str, Any]:
        """Validate all manifest records against disk."""
        entries = self.list_manifest()
        total = len(entries)
        missing_files: list[str] = []
        valid_count = 0

        for e in entries:
            npz = self.features_dir / e.npz_path
            json_p = self.features_dir / e.json_path
            if not npz.is_file() or not json_p.is_file():
                missing_files.append(e.artifact_id)
            else:
                valid_count += 1

        return {
            "total_entries": total,
            "valid_entries": valid_count,
            "missing_artifacts": missing_files,
            "is_valid": len(missing_files) == 0,
        }
