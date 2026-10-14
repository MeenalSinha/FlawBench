"""FlawBench Phase 5: Formal Evaluation, Temporal Localization Metrics & Benchmarking.

Provides independent, reproducible evaluation of speech-flaw detector predictions
against ground-truth annotations using deterministic one-to-one temporal IoU matching.
"""

from __future__ import annotations

from flawbench.evaluation.bootstrap import compute_source_level_bootstrap
from flawbench.evaluation.confusion import build_confusion_matrix
from flawbench.evaluation.controls import evaluate_identity_controls
from flawbench.evaluation.inputs import EvaluationCohort, EvaluationCohortItem
from flawbench.evaluation.matching import compute_interval_iou, match_sample_events
from flawbench.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_category_metrics,
    compute_localization_curve,
)
from flawbench.evaluation.reports import generate_markdown_report
from flawbench.evaluation.runner import EvaluationRunner
from flawbench.evaluation.schema import (
    AggregateMetrics,
    CategoryMetrics,
    ConfusionMatrix,
    ControlMetrics,
    EvaluationReport,
    EventMatch,
    LocalizationPoint,
    SampleEvaluation,
)
from flawbench.evaluation.store import EvaluationStore
from flawbench.evaluation.validator import EvaluationValidator

__all__ = [
    "AggregateMetrics",
    "CategoryMetrics",
    "ConfusionMatrix",
    "ControlMetrics",
    "EvaluationCohort",
    "EvaluationCohortItem",
    "EvaluationReport",
    "EvaluationRunner",
    "EvaluationStore",
    "EvaluationValidator",
    "EventMatch",
    "LocalizationPoint",
    "SampleEvaluation",
    "build_confusion_matrix",
    "compute_aggregate_metrics",
    "compute_category_metrics",
    "compute_interval_iou",
    "compute_localization_curve",
    "compute_source_level_bootstrap",
    "evaluate_identity_controls",
    "generate_markdown_report",
    "match_sample_events",
]
