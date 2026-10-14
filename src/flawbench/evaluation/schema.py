"""Phase 5 Evaluation schema, data classes, and serialization."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from flawbench.exceptions import EvaluationValidationError

EVALUATION_SCHEMA_VERSION = "v1.0.0"


@dataclass(frozen=True)
class EventMatch:
    """Detailed record of a match between a prediction and ground-truth event (or unmatched)."""

    sample_id: str
    source_id: str
    match_status: str  # "matched" | "unmatched_prediction" | "unmatched_ground_truth"
    iou: float
    category_match: bool
    prediction_id: str | None = None
    ground_truth_id: str | None = None
    pred_flaw_type: str | None = None
    gt_flaw_type: str | None = None
    pred_start_sample: int | None = None
    pred_end_sample: int | None = None
    gt_start_sample: int | None = None
    gt_end_sample: int | None = None
    pred_start_sec: float | None = None
    pred_end_sec: float | None = None
    gt_start_sec: float | None = None
    gt_end_sec: float | None = None
    start_error_sec: float | None = None  # pred_start_sec - gt_start_sec
    end_error_sec: float | None = None  # pred_end_sec - gt_end_sec
    duration_error_sec: float | None = None  # pred_duration - gt_duration

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventMatch:
        return cls(**data)


@dataclass(frozen=True)
class CategoryMetrics:
    """Metrics evaluated for a specific flaw category."""

    flaw_type: str
    gt_count: int
    pred_count: int
    matched_count: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    mean_iou: float | None
    median_iou: float | None
    mean_abs_start_error_sec: float | None
    mean_abs_end_error_sec: float | None
    mean_abs_dur_error_sec: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoryMetrics:
        return cls(**data)


@dataclass(frozen=True)
class LocalizationPoint:
    """Point on the localization-tolerance curve."""

    iou_threshold: float
    matched_count: int
    gt_count: int
    recall: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LocalizationPoint:
        return cls(**data)


@dataclass(frozen=True)
class AggregateMetrics:
    """Overall micro- and macro-averaged metrics across all evaluated categories."""

    total_gt_events: int
    total_predictions: int
    total_tp: int
    total_fp: int
    total_fn: int
    micro_precision: float | None
    micro_recall: float | None
    micro_f1: float | None
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    mean_iou: float | None
    median_iou: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregateMetrics:
        return cls(**data)


@dataclass(frozen=True)
class ConfusionMatrix:
    """Event-level categorization confusion matrix."""

    categories: list[str]
    matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": list(self.categories),
            "matrix": {k: dict(v) for k, v in self.matrix.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfusionMatrix:
        return cls(
            categories=list(data["categories"]),
            matrix={k: dict(v) for k, v in data["matrix"].items()},
        )


@dataclass(frozen=True)
class ControlMetrics:
    """Metrics evaluated on negative/identity controls."""

    total_controls: int
    clean_controls_evaluated: int
    controls_with_false_positives: int
    total_false_positive_events: int
    false_positive_rate_per_control: float | None
    false_positive_rate_per_sec: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlMetrics:
        return cls(**data)


@dataclass(frozen=True)
class SampleEvaluation:
    """Per-sample evaluation summary."""

    sample_id: str
    source_id: str
    split: str
    num_gt: int
    num_pred: int
    matches: list[EventMatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "split": self.split,
            "num_gt": self.num_gt,
            "num_pred": self.num_pred,
            "matches": [m.to_dict() for m in self.matches],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampleEvaluation:
        matches = [EventMatch.from_dict(m) for m in data.get("matches", [])]
        return cls(
            sample_id=str(data["sample_id"]),
            source_id=str(data["source_id"]),
            split=str(data["split"]),
            num_gt=int(data["num_gt"]),
            num_pred=int(data["num_pred"]),
            matches=matches,
        )


@dataclass(frozen=True)
class EvaluationReport:
    """Complete, self-contained, reproducible evaluation report."""

    schema_version: str
    run_id: str
    split: str
    sample_count: int
    source_count: int
    excluded_count: int
    exclusion_reasons: dict[str, list[str]]
    per_category: dict[str, CategoryMetrics]
    aggregate: AggregateMetrics
    localization_curve: list[LocalizationPoint]
    confusion_matrix: ConfusionMatrix
    created_at: str
    detector_version: str
    detector_config_fingerprint: str
    eval_config_fingerprint: str
    is_synthetic_only: bool
    controls: ControlMetrics | None = None
    bootstrap_ci: dict[str, Any] | None = None
    sample_evaluations: list[SampleEvaluation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "split": self.split,
            "sample_count": self.sample_count,
            "source_count": self.source_count,
            "excluded_count": self.excluded_count,
            "exclusion_reasons": {k: list(v) for k, v in self.exclusion_reasons.items()},
            "per_category": {k: v.to_dict() for k, v in self.per_category.items()},
            "aggregate": self.aggregate.to_dict(),
            "localization_curve": [p.to_dict() for p in self.localization_curve],
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "controls": self.controls.to_dict() if self.controls else None,
            "bootstrap_ci": self.bootstrap_ci,
            "sample_evaluations": [s.to_dict() for s in self.sample_evaluations],
            "created_at": self.created_at,
            "detector_version": self.detector_version,
            "detector_config_fingerprint": self.detector_config_fingerprint,
            "eval_config_fingerprint": self.eval_config_fingerprint,
            "is_synthetic_only": self.is_synthetic_only,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationReport:
        try:
            per_cat = {
                k: CategoryMetrics.from_dict(v) for k, v in data.get("per_category", {}).items()
            }
            aggregate = AggregateMetrics.from_dict(data["aggregate"])
            loc_curve = [LocalizationPoint.from_dict(p) for p in data.get("localization_curve", [])]
            conf_mat = ConfusionMatrix.from_dict(data["confusion_matrix"])
            controls = ControlMetrics.from_dict(data["controls"]) if data.get("controls") else None
            sample_evals = [
                SampleEvaluation.from_dict(s) for s in data.get("sample_evaluations", [])
            ]
            return cls(
                schema_version=str(data.get("schema_version", EVALUATION_SCHEMA_VERSION)),
                run_id=str(data["run_id"]),
                split=str(data["split"]),
                sample_count=int(data["sample_count"]),
                source_count=int(data["source_count"]),
                excluded_count=int(data.get("excluded_count", 0)),
                exclusion_reasons=dict(data.get("exclusion_reasons", {})),
                per_category=per_cat,
                aggregate=aggregate,
                localization_curve=loc_curve,
                confusion_matrix=conf_mat,
                controls=controls,
                bootstrap_ci=data.get("bootstrap_ci"),
                sample_evaluations=sample_evals,
                created_at=str(data["created_at"]),
                detector_version=str(data["detector_version"]),
                detector_config_fingerprint=str(data["detector_config_fingerprint"]),
                eval_config_fingerprint=str(data["eval_config_fingerprint"]),
                is_synthetic_only=bool(data.get("is_synthetic_only", False)),
                metadata=dict(data.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationValidationError(
                f"Failed to deserialize EvaluationReport: {exc}", detail=str(exc)
            ) from exc

    def validate_bounds(self) -> None:
        """Verify internal consistency of metrics and numbers."""
        if self.sample_count < 0:
            raise EvaluationValidationError(f"sample_count must be >= 0: {self.sample_count}")
        if self.source_count < 0:
            raise EvaluationValidationError(f"source_count must be >= 0: {self.source_count}")
        for k, cat in self.per_category.items():
            if cat.tp < 0 or cat.fp < 0 or cat.fn < 0:
                raise EvaluationValidationError(f"Negative counts in category {k}")
            if cat.precision is not None and not (0.0 <= cat.precision <= 1.0):
                raise EvaluationValidationError(f"Invalid precision in {k}: {cat.precision}")
            if cat.recall is not None and not (0.0 <= cat.recall <= 1.0):
                raise EvaluationValidationError(f"Invalid recall in {k}: {cat.recall}")
            if cat.f1 is not None and not (0.0 <= cat.f1 <= 1.0):
                raise EvaluationValidationError(f"Invalid f1 in {k}: {cat.f1}")
            if cat.mean_iou is not None and not (0.0 <= cat.mean_iou <= 1.0):
                raise EvaluationValidationError(f"Invalid mean_iou in {k}: {cat.mean_iou}")


def safe_div(numerator: float, denominator: float, *, policy: str = "undefined") -> float | None:
    """Safe division respecting zero-denominator policy."""
    if denominator == 0.0 or math.isnan(denominator) or math.isnan(numerator):
        return 0.0 if policy == "zero" else None
    res = numerator / denominator
    return float(res) if math.isfinite(res) else (0.0 if policy == "zero" else None)
