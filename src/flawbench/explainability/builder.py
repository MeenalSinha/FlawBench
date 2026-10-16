"""Explanation builder: validated DetectionResult -> verified ExplanationCards.

Inputs are limited to the detection result, real comparison measurements and recording metadata
(through :class:`AlignmentContext`). Ground-truth annotations are not an input anywhere in this
package; ``tests/test_explainability_grounding.py`` enforces that at import level.
"""

from __future__ import annotations

from flawbench.config import ExplanationSettings
from flawbench.detection.schema import DetectionResult
from flawbench.detection.validator import DetectionValidator
from flawbench.exceptions import ExplanationError, ExplanationInputError
from flawbench.explainability.facts import AlignmentContext, build_bundles
from flawbench.explainability.schema import (
    CARD_OK,
    CARD_VALIDATION_FAILED,
    EvidenceBundle,
    ExplanationCard,
)
from flawbench.explainability.templates import (
    CardText,
    render_card_text,
    render_fallback_text,
)
from flawbench.explainability.validator import validate_card


def _to_card(
    bundle: EvidenceBundle, text: CardText, status: str, problems: list[str]
) -> ExplanationCard:
    return ExplanationCard(
        prediction_id=bundle.prediction_id,
        flaw_type=bundle.flaw_type,
        title=text.title,
        observation=text.observation,
        evidence_lines=tuple(text.evidence_lines),
        interpretation=text.interpretation,
        suggested_review=text.suggested_review,
        limitations=tuple(text.limitations),
        warnings=tuple(text.warnings),
        detector_notes=bundle.detector_warnings,
        numeric_claims=tuple(text.claims),
        missing_facts=tuple(text.missing),
        bundle=bundle,
        status=status,
        validation_problems=tuple(problems),
    )


def build_card(
    bundle: EvidenceBundle, context: AlignmentContext, verbosity: str
) -> ExplanationCard:
    """Render and verify a card; fall back to a facts-only card if verification fails."""
    card = _to_card(bundle, render_card_text(bundle, context, verbosity), CARD_OK, [])
    problems = validate_card(card)
    if not problems:
        return card
    fallback = _to_card(
        bundle, render_fallback_text(bundle, problems), CARD_VALIDATION_FAILED, problems
    )
    fallback_problems = validate_card(fallback)
    if fallback_problems:
        raise ExplanationError(
            "An explanation could not be verified and no safe fallback was possible.",
            detail="; ".join(problems + fallback_problems),
        )
    return fallback


def sort_cards(cards: list[ExplanationCard], policy: str) -> list[ExplanationCard]:
    """Chronological by default; ``evidence_strength`` ranks by the detector's own heuristic."""
    if policy == "evidence_strength":
        return sorted(
            cards, key=lambda c: (-c.bundle.evidence_strength, c.bundle.variant.start_sample)
        )
    return sorted(
        cards,
        key=lambda c: (c.bundle.variant.start_sample, c.bundle.variant.end_sample, c.flaw_type),
    )


def explain_result(
    result: DetectionResult, context: AlignmentContext, settings: ExplanationSettings
) -> list[ExplanationCard]:
    """Build verified explanation cards for every prediction of a valid detection result.

    Raises:
        ExplanationInputError: if the detection result fails Phase 4 validation. An invalid
            result never produces a success-style explanation.
    """
    problems = DetectionValidator().validate_result(result)
    if problems:
        raise ExplanationInputError(
            "The detection result failed validation and cannot be explained.",
            detail="; ".join(problems[:5]),
        )
    bundles = build_bundles(result, context)
    cards = [build_card(b, context, settings.verbosity) for b in bundles]
    return sort_cards(cards, settings.sort_policy)


__all__ = ["build_card", "explain_result", "sort_cards"]
