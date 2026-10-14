"""Project phases. Informational only: status values describe what exists in this repository."""

from __future__ import annotations

from dataclasses import dataclass

IMPLEMENTED = "Implemented"
PLANNED = "Planned (not implemented)"


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
        PLANNED,
    ),
    Phase(5, "Evaluation", "Comparison of predictions against ground truth and metrics.", PLANNED),
    Phase(
        6,
        "Explanations and final dashboard",
        "Evidence-linked explanations, reference-versus-participant visualization.",
        PLANNED,
    ),
)
