"""Verification of explanation cards.

The templates are deterministic, but verification is still enforced in code so that a template
edit or a malformed evidence record cannot silently introduce an untraceable number or an
unsupported claim:

* every number in template prose must match a recorded :class:`NumericClaim`;
* every claim must re-render, via the single formatter, from a valid fact in the bundle;
* prose must not contain percentages, invented-confidence wording, or causal/psychological claims.
"""

from __future__ import annotations

import re

from flawbench.explainability.schema import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NO_CANDIDATES,
    STATUS_PARTIAL,
    ExplanationCard,
    ParticipantAnalysis,
)
from flawbench.explainability.templates import format_value

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Phrases that would assert something the measurements cannot establish.
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"%",
    r"\bnervous",
    r"\banxious",
    r"\baudience",
    r"\bboring",
    r"lack(?:s|ed)? (?:of )?confidence",
    r"\bunprofessional",
    r"\bobjectively",
    r"\bprove[sd]?\b",
    r"\bproof\b",
    r"\boverall (?:speech )?(?:score|quality)",
    r"\bquality score",
    r"\bflawless",
    r"\bcaused\b",
    r"\bthe speaker was\b",
    r"\byou were\b",
    r"\b(?:high|low) confidence",
    r"\bconfident that\b",
)
_FORBIDDEN = tuple(re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS)


def validate_card(card: ExplanationCard) -> list[str]:
    """Return problems found in ``card``; an empty list means the card is traceable and safe."""
    problems: list[str] = []
    bundle = card.bundle
    if card.prediction_id != bundle.prediction_id:
        problems.append("card prediction_id does not match its evidence bundle")
    if card.flaw_type != bundle.flaw_type:
        problems.append("card flaw_type does not match its evidence bundle")
    if bundle.variant.start_sample >= bundle.variant.end_sample:
        problems.append("variant interval is empty or reversed")
    for name, text in (
        ("title", card.title),
        ("observation", card.observation),
        ("interpretation", card.interpretation),
        ("suggested_review", card.suggested_review),
    ):
        if not text.strip():
            problems.append(f"{name} is empty")
    if not card.limitations:
        problems.append("limitations section is empty")

    allowed: set[str] = set()
    for claim in card.numeric_claims:
        fact = bundle.fact(claim.fact_key)
        if fact is None or not fact.valid or fact.value is None:
            problems.append(f"claim '{claim.fact_key}' has no valid supporting fact")
            continue
        if abs(fact.value - claim.value) > 1e-12:
            problems.append(f"claim '{claim.fact_key}' value differs from its fact")
        if claim.rendered != format_value(fact.value, fact.unit):
            problems.append(f"claim '{claim.fact_key}' rendering differs from the fact value")
        allowed.update(NUMBER_RE.findall(claim.rendered))

    for text in card.prose_fields():
        for token in NUMBER_RE.findall(text):
            if token not in allowed:
                problems.append(f"untraceable number {token!r} in: {text[:80]}")
        for pattern in _FORBIDDEN:
            if pattern.search(text):
                problems.append(f"forbidden wording ({pattern.pattern}) in: {text[:80]}")
    return problems


def validate_analysis(analysis: ParticipantAnalysis) -> list[str]:
    """Check that an analysis' status agrees with its content (no success-style failures)."""
    problems: list[str] = []
    status = analysis.status
    if status == STATUS_FAILED:
        if analysis.cards:
            problems.append("a failed analysis must not carry explanation cards")
        if not analysis.failure:
            problems.append("a failed analysis must state its failure")
    elif status == STATUS_NO_CANDIDATES:
        if analysis.cards:
            problems.append("a zero-candidate analysis must not carry explanation cards")
        if analysis.categories_failed:
            problems.append("a complete zero-candidate analysis cannot have failed categories")
    elif status == STATUS_PARTIAL:
        if not analysis.categories_failed:
            problems.append("a partial analysis must name its failed categories")
    elif status == STATUS_COMPLETED:
        if analysis.categories_failed:
            problems.append("a completed analysis cannot have failed categories")
        if not analysis.cards:
            problems.append("a completed analysis with candidates must carry cards")
    else:
        problems.append(f"unknown analysis status {status!r}")
    for card in analysis.cards:
        problems.extend(f"{card.prediction_id}: {p}" for p in validate_card(card))
    return problems
