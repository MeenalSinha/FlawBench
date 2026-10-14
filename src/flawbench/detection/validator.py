"""Phase 4 Detection Result Integrity Validator.

Validates saved detection artifacts against schema, boundary, and isolation rules:
- Checks half-open sample boundaries [start_sample, end_sample)
- Checks timestamp consistency and sample-rate calculation
- Validates numerical evidence values (finite, non-NaN)
- Verifies absence of ground-truth leakage
"""

from __future__ import annotations

import math
from pathlib import Path

from flawbench.config import VALID_DETECTION_CATEGORIES
from flawbench.detection.schema import DetectionResult, FlawPrediction


class DetectionValidator:
    """Audits detection results for structural integrity, valid bounds,
    and ground-truth isolation.
    """

    FORBIDDEN_GROUND_TRUTH_KEYS = {
        "ground_truth",
        "injected_interval",
        "output_interval",
        "source_interval",
        "severity",
        "experimental_severity",
        "injected_flaw",
    }

    def validate_prediction(
        self,
        pred: FlawPrediction,
        total_samples: int,
        sample_rate: int,
        duration_seconds: float,
    ) -> list[str]:
        """Validate an individual FlawPrediction."""
        problems: list[str] = []

        if not pred.prediction_id:
            problems.append("prediction_id is empty")

        if pred.flaw_type not in VALID_DETECTION_CATEGORIES:
            problems.append(
                f"Unknown flaw_type '{pred.flaw_type}'. "
                f"Expected one of {VALID_DETECTION_CATEGORIES}."
            )

        # Sample bounds
        if pred.start_sample < 0:
            problems.append(f"start_sample {pred.start_sample} is negative")
        if pred.end_sample > total_samples:
            problems.append(f"end_sample {pred.end_sample} exceeds total_samples {total_samples}")
        if pred.start_sample >= pred.end_sample:
            problems.append(f"start_sample {pred.start_sample} >= end_sample {pred.end_sample}")

        # Seconds bounds
        if pred.start_sec < 0.0:
            problems.append(f"start_sec {pred.start_sec} is negative")
        if pred.start_sec >= pred.end_sec:
            problems.append(f"start_sec {pred.start_sec} >= end_sec {pred.end_sec}")

        # Duration consistency
        calc_dur = (pred.end_sample - pred.start_sample) / sample_rate
        if abs(calc_dur - pred.duration_sec) > 0.01:
            problems.append(
                f"duration_sec {pred.duration_sec} inconsistent with sample duration {calc_dur:.4f}"
            )

        # Reference correspondence
        if pred.ref_start_sec is not None and pred.ref_end_sec is not None:
            if pred.ref_start_sec < 0.0:
                problems.append(f"ref_start_sec {pred.ref_start_sec} is negative")
            # For pause insertion, reference anchor is instantaneous (ref_start_sec == ref_end_sec)
            if pred.flaw_type == "pause_insertion":
                if pred.ref_start_sec > pred.ref_end_sec:
                    problems.append(
                        f"ref_start_sec {pred.ref_start_sec} > ref_end_sec {pred.ref_end_sec}"
                    )
            else:
                if pred.ref_start_sec >= pred.ref_end_sec:
                    problems.append(
                        f"ref_start_sec {pred.ref_start_sec} >= ref_end_sec {pred.ref_end_sec}"
                    )

        # Evidence strength
        if not (0.0 <= pred.evidence_strength <= 1.0):
            problems.append(f"evidence_strength {pred.evidence_strength} outside [0.0, 1.0]")

        # Finite evidence check
        for k, v in pred.evidence.items():
            if isinstance(v, (int, float)) and not math.isfinite(v):
                problems.append(f"Evidence field '{k}' has non-finite value: {v}")

            # Ground-truth leakage check
            if k.lower() in self.FORBIDDEN_GROUND_TRUTH_KEYS:
                problems.append(f"Ground-truth leakage detected: forbidden evidence key '{k}'")

        return problems

    def validate_result(self, result: DetectionResult) -> list[str]:
        """Validate a full DetectionResult."""
        problems: list[str] = []

        if not result.sample_id:
            problems.append("sample_id is empty")
        if not result.source_id:
            problems.append("source_id is empty")
        if not result.ref_checksum:
            problems.append("ref_checksum is empty")
        if not result.variant_checksum:
            problems.append("variant_checksum is empty")
        if result.sample_rate <= 0:
            problems.append(f"sample_rate {result.sample_rate} must be positive")
        if result.total_samples <= 0:
            problems.append(f"total_samples {result.total_samples} must be positive")
        if result.duration_seconds <= 0:
            problems.append(f"duration_seconds {result.duration_seconds} must be positive")

        # Ground truth leakage in metadata
        for k in result.metadata:
            if k.lower() in self.FORBIDDEN_GROUND_TRUTH_KEYS:
                problems.append(f"Ground-truth leakage detected in metadata: '{k}'")

        # Validate each prediction
        last_start = -1
        for i, pred in enumerate(result.predictions):
            pred_problems = self.validate_prediction(
                pred,
                total_samples=result.total_samples,
                sample_rate=result.sample_rate,
                duration_seconds=result.duration_seconds,
            )
            for p in pred_problems:
                problems.append(f"Prediction [{i}] ({pred.prediction_id}): {p}")

            if pred.start_sample < last_start:
                problems.append(
                    f"Predictions not sorted chronologically: [{i}] "
                    f"start {pred.start_sample} < previous {last_start}"
                )
            last_start = pred.start_sample

        return problems

    def validate_file(self, filepath: Path) -> list[str]:
        """Load and validate a detection result JSON file."""
        filepath = Path(filepath).resolve()
        if not filepath.is_file():
            return [f"File does not exist: {filepath}"]
        try:
            result = DetectionResult.load(filepath)
        except Exception as exc:
            return [f"Failed to parse detection result from {filepath.name}: {exc}"]
        return self.validate_result(result)

    def validate_all(self, detection_dir: Path) -> dict[str, list[str]]:
        """Validate all detection JSON files in a directory."""
        detection_dir = Path(detection_dir).resolve()
        results: dict[str, list[str]] = {}
        if not detection_dir.is_dir():
            return results

        for p in sorted(detection_dir.glob("*_detection.json")):
            results[p.name] = self.validate_file(p)

        return results
