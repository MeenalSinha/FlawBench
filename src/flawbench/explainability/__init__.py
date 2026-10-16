"""Phase 6 evidence-linked explanation engine (deterministic, no external services)."""

from flawbench.explainability.builder import build_card, explain_result, sort_cards
from flawbench.explainability.facts import (
    SUPPORTED_CATEGORIES,
    AlignmentContext,
    build_bundles,
    extract_bundle,
)
from flawbench.explainability.schema import (
    EXPLANATION_SCHEMA_VERSION,
    EvidenceBundle,
    EvidenceFact,
    ExplanationCard,
    ParticipantAnalysis,
)
from flawbench.explainability.validator import validate_analysis, validate_card

__all__ = [
    "EXPLANATION_SCHEMA_VERSION",
    "SUPPORTED_CATEGORIES",
    "AlignmentContext",
    "EvidenceBundle",
    "EvidenceFact",
    "ExplanationCard",
    "ParticipantAnalysis",
    "build_bundles",
    "build_card",
    "explain_result",
    "extract_bundle",
    "sort_cards",
    "validate_analysis",
    "validate_card",
]
