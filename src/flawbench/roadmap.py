"""Project phases. Informational only: status values describe what exists in this repository."""

from __future__ import annotations

from dataclasses import dataclass

IMPLEMENTED = "Implemented"
PLANNED = "Planned (not implemented)"
IN_PROGRESS = "In progress (release candidate; see docs/phase_7_validation_report.md)"


@dataclass(frozen=True)
class Phase:
    number: int
    title: str
    summary: str
    status: str


PHASES: tuple[Phase, ...] = (
    Phase(
        1,
        "Foundation",
        "Configuration, paths, logging, audio input validation, Streamlit foundation, provenance.",
        IMPLEMENTED,
    ),
    Phase(
        2,
        "Dataset and ground-truth engine",
        "Reference registration, controlled flaw injection, ground-truth annotations, manifest, "
        "provenance, dataset validation.",
        IMPLEMENTED,
    ),
    Phase(
        3,
        "Feature extraction",
        "Time-indexed acoustic features for reference and modified recordings.",
        IMPLEMENTED,
    ),
    Phase(
        4,
        "Flaw detection",
        "Independent, label-blind detection and temporal localization.",
        IMPLEMENTED,
    ),
    Phase(
        5,
        "Evaluation",
        "Comparison of predictions against ground truth and temporal IoU benchmarking.",
        IMPLEMENTED,
    ),
    Phase(
        6,
        "Explanations and final dashboard",
        "Participant analysis, evidence-linked explanations, interval playback, reports and "
        "dashboard.",
        IMPLEMENTED,
    ),
    Phase(
        7,
        "Quality assurance and reliability hardening",
        "Detector-reliability fixes, numerical-stability policy, regression, robustness and "
        "security tests, and release-candidate validation on synthetic fixtures.",
        IMPLEMENTED,
    ),
    Phase(
        8,
        "Release preparation, demo engineering, packaging, and handover",
        "Release candidate verification, demo runbook and script, reproducible packaging, "
        "comprehensive documentation, licensing/AI disclosure, and handover readiness.",
        IMPLEMENTED,
    ),
)
