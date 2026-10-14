"""Clean negative control generation and false-positive evaluation."""

from __future__ import annotations

import logging
from typing import Any

from flawbench.config import Config
from flawbench.detection.engine import DetectionEngine
from flawbench.detection.inputs import build_detection_input_from_features
from flawbench.evaluation.schema import ControlMetrics, safe_div
from flawbench.features.pipeline import FeaturePipeline
from flawbench.paths import ProjectPaths

logger = logging.getLogger("flawbench.evaluation.controls")


def evaluate_identity_controls(
    config: Config,
    paths: ProjectPaths,
    reference_sources: list[dict[str, Any]],
    *,
    engine: DetectionEngine | None = None,
) -> ControlMetrics:
    """Evaluate clean speech false-positive rate using identity reference comparisons.

    An identity control uses a clean reference recording as both reference and variant.
    The expected set of flaw events is strictly empty. Any candidate produced by the
    detector represents an unprompted false positive on clean speech.
    """
    if engine is None:
        engine = DetectionEngine(config.detection)

    feat_pipeline = FeaturePipeline(config, paths)
    total_controls = len(reference_sources)
    clean_evaluated = 0
    controls_with_fp = 0
    total_fp_events = 0
    total_dur_sec = 0.0

    for src in reference_sources:
        src_id = src.get("source_id")
        if not src_id:
            continue

        try:
            ref_feat = feat_pipeline.extract_reference(str(src_id))
            det_input = build_detection_input_from_features(
                ref_features=ref_feat,
                var_features=ref_feat,
                sample_id=f"ctrl-{src_id}",
                source_id=str(src_id),
            )
            result = engine.detect(det_input)
            preds = result.predictions

            clean_evaluated += 1
            dur = det_input.duration_seconds
            total_dur_sec += dur

            if preds:
                controls_with_fp += 1
                total_fp_events += len(preds)

        except Exception as exc:
            logger.warning("Failed to evaluate clean identity control for %s: %s", src_id, exc)

    fp_rate_control = safe_div(
        float(total_fp_events),
        float(clean_evaluated),
        policy=config.evaluation.zero_denominator_policy,
    )
    fp_rate_sec = safe_div(
        float(total_fp_events),
        float(total_dur_sec),
        policy=config.evaluation.zero_denominator_policy,
    )

    return ControlMetrics(
        total_controls=total_controls,
        clean_controls_evaluated=clean_evaluated,
        controls_with_false_positives=controls_with_fp,
        total_false_positive_events=total_fp_events,
        false_positive_rate_per_control=fp_rate_control,
        false_positive_rate_per_sec=fp_rate_sec,
    )
