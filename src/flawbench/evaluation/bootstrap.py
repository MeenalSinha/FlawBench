"""Source-level cluster bootstrap confidence interval estimation."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from flawbench.config import VALID_DETECTION_CATEGORIES
from flawbench.evaluation.metrics import compute_aggregate_metrics, compute_category_metrics
from flawbench.evaluation.schema import EventMatch, SampleEvaluation


def compute_source_level_bootstrap(
    sample_evaluations: list[SampleEvaluation],
    *,
    replicates: int = 100,
    seed: int = 1337,
    categories: tuple[str, ...] = VALID_DETECTION_CATEGORIES,
    policy: str = "undefined",
) -> dict[str, Any]:
    """Compute empirical 95% confidence intervals by resampling source recordings with replacement.

    Preserves source-level cluster integrity: all samples derived from a source
    are resampled together.
    """
    # Group samples by source_id
    by_source: dict[str, list[SampleEvaluation]] = defaultdict(list)
    for s in sample_evaluations:
        by_source[s.source_id].append(s)

    unique_sources = list(by_source.keys())
    k = len(unique_sources)

    if k < 3:
        return {
            "status": "insufficient_sources",
            "message": f"Source count {k} < 3 is insufficient for meaningful cluster bootstrap.",
            "source_count": k,
            "replicates": 0,
        }

    if replicates <= 0:
        return {"status": "disabled", "replicates": 0}

    rng = random.Random(seed)
    micro_f1_samples: list[float] = []
    macro_f1_samples: list[float] = []
    mean_iou_samples: list[float] = []

    for _ in range(replicates):
        resampled_sources = rng.choices(unique_sources, k=k)
        boot_matches: list[EventMatch] = []
        for src_id in resampled_sources:
            for s_eval in by_source[src_id]:
                boot_matches.extend(s_eval.matches)

        per_cat = {
            cat: compute_category_metrics(cat, boot_matches, policy=policy) for cat in categories
        }
        agg = compute_aggregate_metrics(per_cat, boot_matches, policy=policy)

        if agg.micro_f1 is not None:
            micro_f1_samples.append(agg.micro_f1)
        if agg.macro_f1 is not None:
            macro_f1_samples.append(agg.macro_f1)
        if agg.mean_iou is not None:
            mean_iou_samples.append(agg.mean_iou)

    def _percentiles(values: list[float]) -> dict[str, float] | None:
        if len(values) < 5:
            return None
        values.sort()
        low_idx = int(round(0.025 * (len(values) - 1)))
        high_idx = int(round(0.975 * (len(values) - 1)))
        return {
            "ci_lower_2_5": round(values[low_idx], 4),
            "median": round(values[len(values) // 2], 4),
            "ci_upper_97_5": round(values[high_idx], 4),
        }

    return {
        "status": "completed",
        "replicates": replicates,
        "source_count": k,
        "seed": seed,
        "metrics": {
            "micro_f1": _percentiles(micro_f1_samples),
            "macro_f1": _percentiles(macro_f1_samples),
            "mean_iou": _percentiles(mean_iou_samples),
        },
    }
