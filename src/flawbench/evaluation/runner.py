"""Evaluation runner orchestrating Phase 5 evaluation workflows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flawbench import __version__
from flawbench.config import VALID_DETECTION_CATEGORIES, Config
from flawbench.evaluation.bootstrap import compute_source_level_bootstrap
from flawbench.evaluation.confusion import build_confusion_matrix
from flawbench.evaluation.controls import evaluate_identity_controls
from flawbench.evaluation.inputs import EvaluationCohort
from flawbench.evaluation.matching import match_sample_events
from flawbench.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_category_metrics,
    compute_localization_curve,
)
from flawbench.evaluation.reports import generate_markdown_report
from flawbench.evaluation.schema import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationReport,
    EventMatch,
    SampleEvaluation,
)
from flawbench.evaluation.store import EvaluationStore
from flawbench.evaluation.validator import EvaluationValidator
from flawbench.paths import ProjectPaths


def _fingerprint(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class EvaluationRunner:
    """Orchestrates end-to-end evaluation runs."""

    def __init__(self, config: Config, paths: ProjectPaths) -> None:
        self.config = config
        self.paths = paths
        self.store = EvaluationStore(paths)
        self.validator = EvaluationValidator()

    def run(
        self,
        split: str = "test",
        sample_id: str | None = None,
        run_id: str | None = None,
        include_controls: bool | None = None,
        enable_bootstrap: bool | None = None,
    ) -> tuple[EvaluationReport, Path]:
        """Execute a formal evaluation run on a dataset split or single sample."""
        eval_cfg = self.config.evaluation
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        if run_id is None:
            prefix = f"eval_{split}" if sample_id is None else f"eval_{sample_id}"
            run_id = f"{prefix}_{timestamp}"

        # 1. Resolve cohort
        cohort = EvaluationCohort.resolve_cohort(
            paths=self.paths,
            split=split,
            config=self.config,
            sample_id=sample_id,
        )

        # 2. Perform event matching for each sample
        sample_evaluations: list[SampleEvaluation] = []
        all_matches: list[EventMatch] = []

        for item in cohort.items:
            matches = match_sample_events(
                sample_id=item.sample_id,
                source_id=item.source_id,
                ground_truth_events=item.ground_truth_events,
                predictions=item.predicted_events,
                sample_rate=item.sample_rate,
                iou_threshold=eval_cfg.matching_iou_threshold,
                require_category_match=True,
            )
            all_matches.extend(matches)
            sample_eval = SampleEvaluation(
                sample_id=item.sample_id,
                source_id=item.source_id,
                split=item.split,
                num_gt=len(item.ground_truth_events),
                num_pred=len(item.predicted_events),
                matches=matches,
            )
            sample_evaluations.append(sample_eval)

        # 3. Calculate category metrics
        per_category = {
            cat: compute_category_metrics(cat, all_matches, policy=eval_cfg.zero_denominator_policy)
            for cat in VALID_DETECTION_CATEGORIES
        }

        # 4. Calculate aggregate metrics
        aggregate = compute_aggregate_metrics(
            per_category, all_matches, policy=eval_cfg.zero_denominator_policy
        )

        # 5. Localization-tolerance curve
        sample_records = [item.to_record_dict() for item in cohort.items]
        loc_curve = compute_localization_curve(
            sample_records,
            tuple(eval_cfg.localization_iou_thresholds),
            require_category_match=True,
            policy=eval_cfg.zero_denominator_policy,
        )

        # 6. Event-level confusion matrix
        confusion_mat = build_confusion_matrix(all_matches)

        # 7. Negative controls (identity comparisons)
        ctrl_metrics = None
        should_run_controls = (
            include_controls if include_controls is not None else eval_cfg.include_identity_controls
        )
        if should_run_controls and sample_id is None and cohort.reference_sources:
            ctrl_metrics = evaluate_identity_controls(
                config=self.config,
                paths=self.paths,
                reference_sources=cohort.reference_sources,
            )

        # 8. Source-level bootstrap confidence intervals
        bootstrap_data: dict[str, Any] | None = None
        reps = eval_cfg.bootstrap_replicates
        if enable_bootstrap is True and reps <= 0:
            reps = 100
        elif enable_bootstrap is False:
            reps = 0

        if reps > 0 and len(sample_evaluations) > 1:
            bootstrap_data = compute_source_level_bootstrap(
                sample_evaluations=sample_evaluations,
                replicates=reps,
                seed=eval_cfg.bootstrap_seed,
                policy=eval_cfg.zero_denominator_policy,
            )

        det_fingerprint = _fingerprint(self.config.detection.__dict__)
        eval_fingerprint = _fingerprint(self.config.evaluation.__dict__)

        # 9. Build EvaluationReport
        report = EvaluationReport(
            schema_version=EVALUATION_SCHEMA_VERSION,
            run_id=run_id,
            split=split,
            sample_count=len(cohort.items),
            source_count=len(set(it.source_id for it in cohort.items)),
            excluded_count=sum(len(v) for v in cohort.excluded_samples.values()),
            exclusion_reasons=cohort.excluded_samples,
            per_category=per_category,
            aggregate=aggregate,
            localization_curve=loc_curve,
            confusion_matrix=confusion_mat,
            controls=ctrl_metrics,
            bootstrap_ci=bootstrap_data,
            sample_evaluations=sample_evaluations,
            created_at=datetime.now(timezone.utc).isoformat(),
            detector_version=__version__,
            detector_config_fingerprint=det_fingerprint,
            eval_config_fingerprint=eval_fingerprint,
            is_synthetic_only=cohort.is_synthetic_only,
            metadata={"matching_iou_threshold": eval_cfg.matching_iou_threshold},
        )

        # 10. Validate report
        self.validator.validate_report(report)

        # 11. Generate markdown report & persist
        markdown = generate_markdown_report(report)
        run_dir = self.store.save_evaluation(
            report=report,
            markdown_report=markdown,
            all_matches=all_matches,
            overwrite=True,
        )

        return report, run_dir
