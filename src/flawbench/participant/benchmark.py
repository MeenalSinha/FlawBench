"""Aggregate benchmark context from saved evaluation runs (never per-prediction correctness)."""

from __future__ import annotations

from flawbench.config import Config
from flawbench.explainability.schema import BenchmarkContext
from flawbench.paths import ProjectPaths


def load_benchmark_context(config: Config, paths: ProjectPaths | None = None) -> BenchmarkContext:
    """Read the most recent valid saved evaluation run. Nothing is computed or launched."""
    from flawbench.evaluation.store import EvaluationStore

    paths = paths or ProjectPaths(config)
    store = EvaluationStore(paths)
    runs = store.list_runs()
    for entry in reversed(runs):
        run_id = entry.get("run_id")
        if not run_id:
            continue
        try:
            report = store.load_evaluation(str(run_id))
        except Exception:
            continue
        per_cat = {
            k: {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "mean_iou": m.mean_iou,
                "ground_truth_events": m.gt_count,
                "predictions": m.pred_count,
            }
            for k, m in report.per_category.items()
        }
        note = (
            "Aggregate detector performance on a synthetic-fixture benchmark. It does not show "
            "generalization to natural speech and does not state how likely any single "
            "prediction is to be correct."
            if report.is_synthetic_only
            else "Aggregate detector performance on the evaluation cohort; it does not state "
            "how likely any single prediction is to be correct."
        )
        return BenchmarkContext(
            True,
            report.run_id,
            report.split,
            report.is_synthetic_only,
            report.sample_count,
            per_cat,
            note,
        )
    return BenchmarkContext(False, note="No valid saved evaluation run is available.")
