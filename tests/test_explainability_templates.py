from __future__ import annotations

import re

import pytest

from flawbench.config import Config
from flawbench.explainability.builder import build_card, sort_cards
from flawbench.explainability.validator import NUMBER_RE, validate_card
from tests.explain_fixtures import ENERGY_EVIDENCE, PAUSE_EVIDENCE, TEMPO_EVIDENCE, bundle, ctx


def _text(card) -> str:  # type: ignore[no-untyped-def]
    return "\n".join(card.prose_fields())


def test_pause_card_uses_actual_values_and_anchor() -> None:
    card = build_card(bundle("pause_insertion", PAUSE_EVIDENCE, ref=(1.9, 1.9)), ctx(), "standard")
    t = _text(card)
    assert card.status == "ok" and validate_card(card) == []
    assert "0.62 s" in t and "1.00 s" in t and "2.00 s" in t  # duration, start, end
    assert "anchor point near 1.90 s" in t and "-35.00 dB" in t and "0.15 s" in t


def test_pause_without_reference_correspondence_says_so() -> None:
    card = build_card(bundle("pause_insertion", PAUSE_EVIDENCE, ref=None), ctx(), "standard")
    assert "No reliable reference correspondence" in _text(card)
    assert "anchor" not in _text(card)


def test_tempo_direction_follows_documented_rate_convention() -> None:
    slow = build_card(bundle("tempo_change", TEMPO_EVIDENCE), ctx(), "standard")
    assert "slower than the reference" in slow.observation and "0.700" in _text(slow)
    fast_ev = dict(
        TEMPO_EVIDENCE, direction="faster", estimated_relative_rate=1.4, mean_warp_slope=0.7143
    )
    fast = build_card(bundle("tempo_change", fast_ev), ctx(), "standard")
    assert "faster than the reference" in fast.observation


def test_tempo_uncertain_alignment_withholds_direction() -> None:
    card = build_card(
        bundle("tempo_change", TEMPO_EVIDENCE), ctx(participant_duration_sec=20.0), "standard"
    )
    assert "slower than" not in card.observation and "different from" in card.observation
    assert any("uncertain" in w for w in card.warnings)


def test_energy_card_sign_and_no_psychological_claims() -> None:
    card = build_card(bundle("energy_reduction", ENERGY_EVIDENCE), ctx(), "standard")
    t = _text(card)
    assert "-8.00 dB" in t and "-11.00 dB" in t and "lower measured signal energy" in t
    assert "not a direct measure of confidence" in t
    assert validate_card(card) == []


def test_non_negative_energy_difference_is_flagged_not_asserted() -> None:
    card = build_card(
        bundle(
            "energy_reduction", dict(ENERGY_EVIDENCE, mean_delta_rms_db=2.0, mean_var_rms_db=-18.0)
        ),
        ctx(),
        "standard",
    )
    assert "lower measured signal energy" not in card.observation
    assert any("unexpected" in w for w in card.warnings)


def test_missing_measurement_produces_no_invented_number() -> None:
    ev = {k: v for k, v in ENERGY_EVIDENCE.items() if k != "mean_delta_rms_db"}
    card = build_card(bundle("energy_reduction", ev), ctx(), "standard")
    assert "mean_delta_rms_db" in card.missing_facts
    assert "-8.00" not in _text(card) and "-11.00" not in _text(card)
    assert validate_card(card) == []


def test_every_number_is_traceable_in_all_verbosities() -> None:
    cases = [
        ("pause_insertion", PAUSE_EVIDENCE), ("tempo_change", TEMPO_EVIDENCE),
        ("energy_reduction", ENERGY_EVIDENCE),
    ]  # fmt: skip
    for verbosity in ("concise", "standard", "detailed"):
        for flaw, ev in cases:
            card = build_card(bundle(flaw, ev), ctx(), verbosity)
            allowed = {n for c in card.numeric_claims for n in NUMBER_RE.findall(c.rendered)}
            assert card.status == "ok"
            for text in card.prose_fields():
                assert set(NUMBER_RE.findall(text)) <= allowed, (verbosity, flaw, text)


def test_verbosity_changes_amount_of_evidence() -> None:
    b = bundle("pause_insertion", PAUSE_EVIDENCE)
    concise = build_card(b, ctx(), "concise")
    detailed = build_card(b, ctx(), "detailed")
    assert len(concise.evidence_lines) < len(detailed.evidence_lines)


def test_strength_is_described_as_uncalibrated_not_probability() -> None:
    t = _text(build_card(bundle("pause_insertion", PAUSE_EVIDENCE), ctx(), "standard"))
    assert "uncalibrated heuristic score" in t and "not a calibrated probability" in t
    assert "%" not in t


def test_merged_prediction_is_disclosed() -> None:
    card = build_card(
        bundle("pause_insertion", dict(PAUSE_EVIDENCE, was_merged=True)), ctx(), "standard"
    )
    assert any("first region only" in w for w in card.warnings)


def test_validator_rejects_untraceable_numbers_and_forbidden_wording() -> None:
    from dataclasses import replace

    card = build_card(bundle("pause_insertion", PAUSE_EVIDENCE), ctx(), "standard")
    assert any(
        "untraceable" in p for p in validate_card(replace(card, observation="A 7.77 s pause."))
    )
    bad = replace(card, interpretation="The speaker was nervous and lost the audience.")
    assert any("forbidden" in p for p in validate_card(bad))
    assert any(
        "%" in p or "forbidden" in p
        for p in validate_card(replace(card, suggested_review="90% sure"))
    )


def test_template_failing_verification_falls_back_to_facts_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flawbench.explainability import builder, templates

    original = templates.render_card_text

    def corrupt(b, c, v):  # type: ignore[no-untyped-def]
        t = original(b, c, v)
        t.evidence_lines.append("The speaker was nervous (99 percent).")
        return t

    monkeypatch.setattr(builder, "render_card_text", corrupt)
    card = build_card(bundle("pause_insertion", PAUSE_EVIDENCE), ctx(), "standard")
    assert card.status == "validation_failed" and card.validation_problems
    assert "nervous" not in _text(card)


def test_sorting_policies(config: Config) -> None:
    a = build_card(
        bundle("pause_insertion", PAUSE_EVIDENCE, start=32000, end=40000, strength=0.9),
        ctx(),
        "standard",
    )
    b = build_card(
        bundle("energy_reduction", ENERGY_EVIDENCE, start=0, end=8000, strength=0.2),
        ctx(),
        "standard",
    )
    assert sort_cards([a, b], "chronological")[0] is b
    assert sort_cards([a, b], "evidence_strength")[0] is a
    assert re.fullmatch(r"[a-z_]+", config.explanation.sort_policy)
