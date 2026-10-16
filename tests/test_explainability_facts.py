from __future__ import annotations

import math

import pytest

from flawbench.explainability.schema import REF_ANCHOR, REF_INTERVAL, REF_UNAVAILABLE
from tests.explain_fixtures import (
    ENERGY_EVIDENCE,
    PAUSE_EVIDENCE,
    SR,
    TEMPO_EVIDENCE,
    bundle,
    ctx,
    make_pred,
)


def test_facts_come_from_prediction_evidence() -> None:
    b = bundle("pause_insertion", PAUSE_EVIDENCE)
    assert b.valid_value("variant_pause_duration_sec") == 0.62
    assert b.valid_value("duration_difference_sec") == 0.62
    assert b.fact("min_duration_sec").unit == "s"  # type: ignore[union-attr]
    assert b.fact("mean_variant_rms_db").unit == "dB"  # type: ignore[union-attr]


def test_interval_uses_samples_and_sample_rate() -> None:
    b = bundle("energy_reduction", ENERGY_EVIDENCE, start=8000, end=24000)
    assert b.variant.start_sample == 8000 and b.variant.end_sample == 24000
    assert b.valid_value("variant_start_sec") == pytest.approx(0.5)
    assert b.valid_value("variant_end_sec") == pytest.approx(1.5)
    assert b.valid_value("variant_duration_sec") == pytest.approx(1.0)


def test_zero_length_pause_reference_is_an_anchor() -> None:
    b = bundle("pause_insertion", PAUSE_EVIDENCE, ref=(1.9, 1.9))
    assert b.reference.status == REF_ANCHOR
    assert b.valid_value("reference_start_sec") == 1.9
    assert b.fact("reference_end_sec") is None


def test_positive_reference_span_is_an_interval_and_stays_separate() -> None:
    b = bundle("tempo_change", TEMPO_EVIDENCE, ref=(2.0, 3.0), start=16000, end=24000)
    assert b.reference.status == REF_INTERVAL
    assert b.valid_value("reference_start_sec") == 2.0 and b.valid_value("variant_start_sec") == 1.0


@pytest.mark.parametrize("flaw", ["tempo_change", "energy_reduction"])
def test_degenerate_reference_for_non_pause_is_unavailable(flaw: str) -> None:
    ev = TEMPO_EVIDENCE if flaw == "tempo_change" else ENERGY_EVIDENCE
    assert bundle(flaw, ev, ref=(1.0, 1.0)).reference.status == REF_UNAVAILABLE
    assert bundle(flaw, ev, ref=None).reference.status == REF_UNAVAILABLE


def test_reference_beyond_reference_recording_is_not_used() -> None:
    b = bundle("tempo_change", TEMPO_EVIDENCE, ref=(10.0, 11.0))
    assert b.reference.status == REF_UNAVAILABLE


@pytest.mark.parametrize("bad", [math.nan, math.inf, "x", None, True])
def test_non_finite_or_wrong_type_evidence_is_rejected(bad: object) -> None:
    ev = dict(PAUSE_EVIDENCE, variant_pause_duration_sec=bad)
    f = bundle("pause_insertion", ev).fact("variant_pause_duration_sec")
    assert f is not None and not f.valid and f.value is None and f.note


def test_missing_measurement_is_marked_unavailable_not_invented() -> None:
    ev = {k: v for k, v in ENERGY_EVIDENCE.items() if k != "min_delta_rms_db"}
    f = bundle("energy_reduction", ev).fact("min_delta_rms_db")
    assert f is not None and not f.valid and f.value is None


def test_undocumented_evidence_keys_are_ignored_not_interpreted() -> None:
    b = bundle("pause_insertion", dict(PAUSE_EVIDENCE, mystery_score=9.9, merged_x=1.0))
    assert "mystery_score" in b.ignored_evidence_keys
    assert b.fact("mystery_score") is None


def test_inconsistent_fields_are_rejected() -> None:
    b = bundle("pause_insertion", dict(PAUSE_EVIDENCE, duration_difference_sec=5.0))
    assert not b.fact("duration_difference_sec").valid  # type: ignore[union-attr]
    t = bundle("tempo_change", dict(TEMPO_EVIDENCE, mean_warp_slope=3.0))
    assert not t.fact("estimated_relative_rate").valid  # type: ignore[union-attr]


def test_tempo_direction_contradicting_rate_is_dropped() -> None:
    b = bundle("tempo_change", dict(TEMPO_EVIDENCE, direction="faster"))
    assert "direction" not in b.categorical and "direction_rejected" in b.categorical
    ok = bundle("tempo_change", TEMPO_EVIDENCE)
    assert ok.categorical["direction"] == "slower"


def test_merged_flag_and_alignment_context_recorded() -> None:
    b = bundle("pause_insertion", dict(PAUSE_EVIDENCE, was_merged=True))
    assert b.was_merged
    c = ctx(reference_duration_sec=2.0, participant_duration_sec=9.0)
    assert c.length_mismatch and c.uncertain


def test_unsupported_flaw_type_raises() -> None:
    from flawbench.explainability.facts import extract_bundle

    with pytest.raises(ValueError):
        extract_bundle(make_pred("pitch_flatten", {}), sample_rate=SR, context=ctx())
