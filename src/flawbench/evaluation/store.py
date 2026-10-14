"""Evaluation persistence and storage for FlawBench Phase 5.

Manages saving and loading evaluation artifacts, reports, event-match logs,
and maintaining the centralized evaluations manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flawbench.evaluation.schema import EvaluationReport, EventMatch
from flawbench.exceptions import EvaluationStoreError
from flawbench.paths import ProjectPaths


class EvaluationStore:
    """Handles persistence of evaluation runs and manifests."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.evaluations_dir = paths.evaluations_dir
        self.manifest_path = paths.evaluations_manifest_file

    def get_run_dir(self, run_id: str) -> Path:
        """Return the directory path for a given evaluation run."""
        return self.evaluations_dir / run_id

    def save_evaluation(
        self,
        report: EvaluationReport,
        markdown_report: str | None = None,
        all_matches: list[EventMatch] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Save an evaluation run's artifacts.

        Writes:
          - evaluation_report.json
          - event_matches.jsonl
          - report.md (if provided)
          - Appends/updates entry in evaluations_manifest.jsonl
        """
        run_dir = self.get_run_dir(report.run_id)
        if run_dir.exists() and not overwrite:
            raise EvaluationStoreError(
                f"Evaluation run directory already exists: {run_dir}. "
                "Set overwrite=True to replace."
            )

        run_dir.mkdir(parents=True, exist_ok=True)

        report_file = run_dir / "evaluation_report.json"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
        except Exception as exc:
            raise EvaluationStoreError(f"Failed to write evaluation report: {exc}") from exc

        # Extract matches either from argument or from sample_evaluations
        matches_to_save: list[EventMatch] = []
        if all_matches is not None:
            matches_to_save = all_matches
        elif report.sample_evaluations:
            for s in report.sample_evaluations:
                matches_to_save.extend(s.matches)

        # Write event matches
        matches_file = run_dir / "event_matches.jsonl"
        try:
            with open(matches_file, "w", encoding="utf-8") as f:
                for match in matches_to_save:
                    f.write(json.dumps(match.to_dict()) + "\n")
        except Exception as exc:
            raise EvaluationStoreError(f"Failed to write event matches: {exc}") from exc

        # Write markdown report if present
        if markdown_report is not None:
            md_file = run_dir / "report.md"
            try:
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(markdown_report)
            except Exception as exc:
                raise EvaluationStoreError(f"Failed to write report.md: {exc}") from exc

        # Update manifest
        self._record_in_manifest(report)
        return run_dir

    def _record_in_manifest(self, report: EvaluationReport) -> None:
        """Update or append evaluation summary to evaluations_manifest.jsonl."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_entry = {
            "run_id": report.run_id,
            "created_at": report.created_at,
            "split": report.split,
            "sample_count": report.sample_count,
            "source_count": report.source_count,
            "excluded_count": report.excluded_count,
            "detector_version": report.detector_version,
            "is_synthetic_only": report.is_synthetic_only,
            "micro_f1": report.aggregate.micro_f1,
            "macro_f1": report.aggregate.macro_f1,
            "mean_iou": report.aggregate.mean_iou,
        }

        existing_entries: list[dict[str, Any]] = []
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            existing_entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        # Replace if existing run_id, otherwise append
        updated = False
        for idx, entry in enumerate(existing_entries):
            if entry.get("run_id") == report.run_id:
                existing_entries[idx] = manifest_entry
                updated = True
                break
        if not updated:
            existing_entries.append(manifest_entry)

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for entry in existing_entries:
                f.write(json.dumps(entry) + "\n")

    def load_evaluation(self, run_id: str) -> EvaluationReport:
        """Load an EvaluationReport by its run_id."""
        run_dir = self.get_run_dir(run_id)
        report_file = run_dir / "evaluation_report.json"
        if not report_file.exists():
            raise EvaluationStoreError(f"Evaluation report not found: {report_file}")

        try:
            with open(report_file, encoding="utf-8") as f:
                data = json.load(f)
            return EvaluationReport.from_dict(data)
        except Exception as exc:
            raise EvaluationStoreError(f"Failed to load evaluation report {run_id}: {exc}") from exc

    def list_runs(self) -> list[dict[str, Any]]:
        """List all completed evaluation runs from the manifest."""
        if not self.manifest_path.exists():
            return []
        runs = []
        with open(self.manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return runs
