"""Unit tests for Phase 4 prediction post-processing."""

from __future__ import annotations

from flawbench.detection.postprocess import postprocess_predictions
from flawbench.detection.schema import FlawPrediction


def test_postprocess_merges_adjacent_same_category() -> None:
    # Two pause candidates separated by 0.03s (< 0.05s merge_gap_sec)
    p1 = FlawPrediction(
        prediction_id="p1",
        flaw_type="pause_insertion",
        start_sample=16000,
        end_sample=24000,
        start_sec=1.0,
        end_sec=1.5,
        duration_sec=0.5,
        evidence={"mean_var_rms_db": -40.0},
        evidence_strength=0.7,
    )
    p2 = FlawPrediction(
        prediction_id="p2",
        flaw_type="pause_insertion",
        start_sample=24480,
        end_sample=32000,
        start_sec=1.53,
        end_sec=2.0,
        duration_sec=0.47,
        evidence={"mean_var_rms_db": -45.0},
        evidence_strength=0.8,
    )

    merged = postprocess_predictions(
        [p1, p2],
        merge_gap_sec=0.05,
        min_candidate_duration_sec=0.08,
        total_samples=48000,
        sample_rate=16000,
    )

    assert len(merged) == 1
    m = merged[0]
    assert m.flaw_type == "pause_insertion"
    assert m.start_sample == 16000
    assert m.end_sample == 32000
    assert m.start_sec == 1.0
    assert m.end_sec == 2.0
    assert m.duration_sec == 1.0
    assert m.evidence_strength == 0.8
    assert m.evidence.get("was_merged") is True


def test_postprocess_does_not_merge_different_categories() -> None:
    # Overlapping or adjacent candidates of DIFFERENT flaw categories must remain distinct
    p1 = FlawPrediction(
        prediction_id="p1",
        flaw_type="pause_insertion",
        start_sample=16000,
        end_sample=24000,
        start_sec=1.0,
        end_sec=1.5,
        duration_sec=0.5,
        evidence_strength=0.7,
    )
    p2 = FlawPrediction(
        prediction_id="p2",
        flaw_type="energy_reduction",
        start_sample=20000,
        end_sample=28000,
        start_sec=1.25,
        end_sec=1.75,
        duration_sec=0.5,
        evidence_strength=0.6,
    )

    result = postprocess_predictions(
        [p1, p2],
        merge_gap_sec=0.1,
        min_candidate_duration_sec=0.08,
        total_samples=48000,
        sample_rate=16000,
    )

    assert len(result) == 2
    types = {r.flaw_type for r in result}
    assert types == {"pause_insertion", "energy_reduction"}


def test_postprocess_filters_short_candidates() -> None:
    p_short = FlawPrediction(
        prediction_id="p_short",
        flaw_type="tempo_change",
        start_sample=8000,
        end_sample=8640,
        start_sec=0.5,
        end_sec=0.54,
        duration_sec=0.04,  # < 0.08s min duration
        evidence_strength=0.5,
    )
    p_valid = FlawPrediction(
        prediction_id="p_valid",
        flaw_type="tempo_change",
        start_sample=16000,
        end_sample=24000,
        start_sec=1.0,
        end_sec=1.5,
        duration_sec=0.5,
        evidence_strength=0.7,
    )

    result = postprocess_predictions(
        [p_short, p_valid],
        merge_gap_sec=0.05,
        min_candidate_duration_sec=0.08,
        total_samples=48000,
        sample_rate=16000,
    )

    assert len(result) == 1
    assert result[0].prediction_id == "p_valid"
