"""Validation of evaluation reports and saved artifacts for FlawBench Phase 5."""

from __future__ import annotations

import json
from pathlib import Path

from flawbench.evaluation.schema import EvaluationReport
from flawbench.exceptions import EvaluationValidationError


class EvaluationValidator:
    """Validates in-memory and persisted evaluation reports."""

    def validate_report(self, report: EvaluationReport) -> list[str]:
        """Validate an in-memory EvaluationReport.

        Returns a list of warning messages (if any). Raises EvaluationValidationError
        if fatal integrity violations are found.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not report.run_id.strip():
            errors.append("run_id must not be empty.")
        if not report.detector_version.strip():
            errors.append("detector_version must not be empty.")
        if report.sample_count < 0:
            errors.append(f"sample_count must be non-negative, got {report.sample_count}.")

        # Check internal bounds on category metrics & aggregate metrics
        try:
            report.validate_bounds()
        except EvaluationValidationError as exc:
            errors.append(str(exc))

        if report.is_synthetic_only:
            warnings.append(
                "Evaluation was performed on synthetic audio signals only. "
                "Results do not demonstrate generalization to natural human speech."
            )

        if report.sample_count < 10:
            warnings.append(
                f"Cohort size is small ({report.sample_count} samples). "
                "Statistical metrics and confidence intervals may be unstable."
            )

        if errors:
            raise EvaluationValidationError(
                "EvaluationReport validation failed:\n  - " + "\n  - ".join(errors)
            )

        return warnings

    def validate_run_directory(self, run_dir: Path) -> tuple[EvaluationReport, list[str]]:
        """Validate files in an evaluation run directory and return the parsed report."""
        if not run_dir.exists() or not run_dir.is_dir():
            raise EvaluationValidationError(f"Evaluation directory does not exist: {run_dir}")

        report_file = run_dir / "evaluation_report.json"
        if not report_file.exists():
            raise EvaluationValidationError(f"Missing required report file: {report_file}")

        try:
            with open(report_file, encoding="utf-8") as f:
                data = json.load(f)
            report = EvaluationReport.from_dict(data)
        except Exception as exc:
            raise EvaluationValidationError(
                f"Failed to parse report file {report_file}: {exc}"
            ) from exc

        matches_file = run_dir / "event_matches.jsonl"
        if matches_file.exists():
            try:
                with open(matches_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            json.loads(line)
            except Exception as exc:
                raise EvaluationValidationError(
                    f"Corrupt event_matches.jsonl in {run_dir}: {exc}"
                ) from exc

        warnings = self.validate_report(report)
        return report, warnings
