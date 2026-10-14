"""Phase 4 Flaw Detection & Temporal Localization Schema.

Defines:
- Label-free DetectionInput strictly decoupled from ground-truth annotations.
- FlawPrediction representing an independently detected flaw candidate interval.
- Versioned DetectionResult representing the consolidated outcome for a sample.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flawbench.exceptions import DetectionValidationError
from flawbench.features.comparison import ComparisonResult
from flawbench.features.extractor import FeatureResult


@dataclass(frozen=True)
class DetectionInput:
    """Label-free detection input container.

    Enforces strict ground-truth isolation:
    This data structure contains only acoustic signals, extracted features,
    and signal metadata (durations, sample rates, checksums).
    It contains NO ground-truth flaw annotations, injection intervals,
    or experimental severities.
    """

    sample_id: str
    source_id: str
    ref_features: FeatureResult
    var_features: FeatureResult
    comparison: ComparisonResult
    sample_rate: int
    ref_sample_rate: int
    duration_seconds: float
    ref_duration_seconds: float
    ref_checksum: str
    variant_checksum: str


@dataclass(frozen=True)
class FlawPrediction:
    """An independently detected flaw interval on the variant recording timeline."""

    prediction_id: str
    flaw_type: str
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float
    ref_start_sec: float | None = None
    ref_end_sec: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    evidence_strength: float = 0.0
    detector_id: str = ""
    decision_threshold: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.start_sample < 0 or self.end_sample < 0:
            raise DetectionValidationError(
                "Prediction sample bounds must be non-negative.",
                detail=f"start_sample={self.start_sample}, end_sample={self.end_sample}",
            )
        if self.start_sample >= self.end_sample:
            raise DetectionValidationError(
                "Prediction start_sample must be strictly less than end_sample "
                "(half-open [start, end)).",
                detail=f"start_sample={self.start_sample}, end_sample={self.end_sample}",
            )
        if self.start_sec < 0.0 or self.end_sec < 0.0:
            raise DetectionValidationError(
                "Prediction time bounds must be non-negative.",
                detail=f"start_sec={self.start_sec}, end_sec={self.end_sec}",
            )
        if self.start_sec >= self.end_sec:
            raise DetectionValidationError(
                "Prediction start_sec must be strictly less than end_sec.",
                detail=f"start_sec={self.start_sec}, end_sec={self.end_sec}",
            )
        if not (0.0 <= self.evidence_strength <= 1.0):
            raise DetectionValidationError(
                "Prediction evidence_strength must be in [0.0, 1.0].",
                detail=f"evidence_strength={self.evidence_strength}",
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to JSON-serializable dictionary."""
        d = asdict(self)
        d["warnings"] = list(self.warnings)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlawPrediction:
        """Construct a FlawPrediction from a dictionary."""
        return cls(
            prediction_id=str(data["prediction_id"]),
            flaw_type=str(data["flaw_type"]),
            start_sample=int(data["start_sample"]),
            end_sample=int(data["end_sample"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            duration_sec=float(data["duration_sec"]),
            ref_start_sec=float(data["ref_start_sec"])
            if data.get("ref_start_sec") is not None
            else None,
            ref_end_sec=float(data["ref_end_sec"]) if data.get("ref_end_sec") is not None else None,
            evidence=dict(data.get("evidence", {})),
            evidence_strength=float(data.get("evidence_strength", 0.0)),
            detector_id=str(data.get("detector_id", "")),
            decision_threshold=float(data["decision_threshold"])
            if data.get("decision_threshold") is not None
            else None,
            warnings=tuple(data.get("warnings", ())),
        )


@dataclass
class DetectionResult:
    """Consolidated flaw detection and temporal localization output for a speech sample."""

    schema_version: str
    run_id: str
    sample_id: str
    source_id: str
    ref_artifact_id: str
    variant_artifact_id: str
    ref_checksum: str
    variant_checksum: str
    detector_version: str
    config_fingerprint: str
    created_at: str
    sample_rate: int
    total_samples: int
    duration_seconds: float
    predictions: list[FlawPrediction]
    raw_candidates_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert detection result to serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "ref_artifact_id": self.ref_artifact_id,
            "variant_artifact_id": self.variant_artifact_id,
            "ref_checksum": self.ref_checksum,
            "variant_checksum": self.variant_checksum,
            "detector_version": self.detector_version,
            "config_fingerprint": self.config_fingerprint,
            "created_at": self.created_at,
            "sample_rate": self.sample_rate,
            "total_samples": self.total_samples,
            "duration_seconds": self.duration_seconds,
            "predictions_count": len(self.predictions),
            "raw_candidates_count": self.raw_candidates_count,
            "predictions": [p.to_dict() for p in self.predictions],
            "metadata": self.metadata,
        }

    def save(self, filepath: Path) -> Path:
        """Atomically persist detection result as JSON file."""
        filepath = Path(filepath).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp = filepath.with_suffix(f".tmp.{filepath.suffix.lstrip('.')}")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> DetectionResult:
        """Load and deserialize DetectionResult from JSON file."""
        filepath = Path(filepath).resolve()
        data = json.loads(filepath.read_text(encoding="utf-8"))
        predictions = [FlawPrediction.from_dict(p) for p in data.get("predictions", [])]
        return cls(
            schema_version=str(data["schema_version"]),
            run_id=str(data["run_id"]),
            sample_id=str(data["sample_id"]),
            source_id=str(data["source_id"]),
            ref_artifact_id=str(data["ref_artifact_id"]),
            variant_artifact_id=str(data["variant_artifact_id"]),
            ref_checksum=str(data["ref_checksum"]),
            variant_checksum=str(data["variant_checksum"]),
            detector_version=str(data["detector_version"]),
            config_fingerprint=str(data["config_fingerprint"]),
            created_at=str(data["created_at"]),
            sample_rate=int(data["sample_rate"]),
            total_samples=int(data["total_samples"]),
            duration_seconds=float(data["duration_seconds"]),
            predictions=predictions,
            raw_candidates_count=int(data.get("raw_candidates_count", len(predictions))),
            metadata=dict(data.get("metadata", {})),
        )
