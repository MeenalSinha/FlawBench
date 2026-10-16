"""Phase 7 regression tests.

All audio is SYNTHETIC (harmonic tone plus noise from ``synthetic_signal``). These tests pin
detector behavior on controlled injections produced by the real Phase 2 injectors; annotations
(the injector's output interval) are read only AFTER detection, for assertions, and are never
passed to a detector.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest

from flawbench.config import Config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.injectors import change_tempo, insert_pause, reduce_energy
from flawbench.dataset.splits import build_splits
from flawbench.detection.detectors import (
    EnergyReductionDetector,
    PauseDetector,
    TempoDetector,
    windowed_warp_slope,
)
from flawbench.detection.engine import DetectionEngine
from flawbench.detection.inputs import build_detection_input_from_features
from flawbench.detection.schema import DetectionResult
from flawbench.evaluation.runner import EvaluationRunner
from flawbench.exceptions import EvaluationStoreError
from flawbench.features.comparison import _f0_correlation
from flawbench.features.extractor import extract_features
from flawbench.paths import ProjectPaths
from tests.conftest import SYN_RATE, synthetic_signal


def _detect(config: Config, ref: np.ndarray, var: np.ndarray) -> DetectionResult:
    rf = extract_features(ref, sample_rate=SYN_RATE, config=config)
    vf = extract_features(var, sample_rate=SYN_RATE, config=config)
    inp = build_detection_input_from_features(rf, vf, settings=config.detection)
    return DetectionEngine(config.detection).detect(inp)


def _kinds(result: DetectionResult) -> list[str]:
    return sorted(p.flaw_type for p in result.predictions)


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


@pytest.fixture
def ref_audio() -> np.ndarray:
    return synthetic_signal(6.0, seed=3)[np.newaxis, :]


@pytest.mark.parametrize("rate", [0.6, 0.7, 1.3, 1.5])
def test_tempo_change_detected_with_correct_direction_and_localization(
    config: Config, ref_audio: np.ndarray, rate: float
) -> None:
    inj = change_tempo(ref_audio, start=2 * SYN_RATE, end=4 * SYN_RATE, rate=rate)
    result = _detect(config, ref_audio[0], inj.samples[0])
    tempo = [p for p in result.predictions if p.flaw_type == "tempo_change"]
    assert len(tempo) == 1, _kinds(result)
    p = tempo[0]
    assert p.evidence["direction"] == ("slower" if rate < 1 else "faster")
    assert abs(p.evidence["estimated_relative_rate"] - rate) / rate < 0.15
    truth = (inj.output_interval[0] / SYN_RATE, inj.output_interval[1] / SYN_RATE)
    assert _iou((p.start_sec, p.end_sec), truth) >= 0.7
    assert _kinds(result) == ["tempo_change"]


def test_inserted_pause_is_not_reported_as_energy_reduction_or_tempo(
    config: Config, ref_audio: np.ndarray
) -> None:
    inj = insert_pause(ref_audio, at_sample=3 * SYN_RATE, silence_samples=SYN_RATE)
    result = _detect(config, ref_audio[0], inj.samples[0])
    assert _kinds(result) == ["pause_insertion"]


def test_strong_attenuation_is_not_reported_as_pause_or_tempo(
    config: Config, ref_audio: np.ndarray
) -> None:
    inj = reduce_energy(ref_audio, start=2 * SYN_RATE, end=4 * SYN_RATE, gain_db=-24.0)
    result = _detect(config, ref_audio[0], inj.samples[0])
    assert _kinds(result) == ["energy_reduction"]
    assert result.predictions[0].evidence["mean_delta_rms_db"] < -15.0


def test_identical_audio_yields_no_predictions(config: Config, ref_audio: np.ndarray) -> None:
    assert _detect(config, ref_audio[0], ref_audio[0].copy()).predictions == []


def test_prediction_ids_are_deterministic(config: Config, ref_audio: np.ndarray) -> None:
    inj = change_tempo(ref_audio, start=2 * SYN_RATE, end=4 * SYN_RATE, rate=0.7)
    a = _detect(config, ref_audio[0], inj.samples[0])
    b = _detect(config, ref_audio[0], inj.samples[0])
    assert a.predictions and [p.prediction_id for p in a.predictions] == [
        p.prediction_id for p in b.predictions
    ]


def test_detector_fingerprint_depends_on_version_and_every_logic_setting(config: Config) -> None:
    base = DetectionEngine(config.detection).config_fingerprint()
    changes: dict[str, Any] = {
        "tempo_slope_window_sec": 0.9,
        "tempo_max_inactive_fraction": 0.5,
        "pause_max_voiced_fraction": 0.6,
        "alignment_metric": "cosine",
        "alignment_exclude_energy_coefficient": False,
    }
    for name, value in changes.items():
        changed = dataclasses.replace(config.detection, **{name: value})
        assert DetectionEngine(changed).config_fingerprint() != base, name
    engine = DetectionEngine(config.detection)
    engine.detector_version = "9.9"
    assert engine.config_fingerprint() != base


def test_detectors_expose_v5_ids(config: Config) -> None:
    assert PauseDetector.detector_id.endswith("_v5")
    assert TempoDetector.detector_id.endswith("_v5")
    assert EnergyReductionDetector.detector_id.endswith("_v5")


# ----------------------------------------------------------------- slope estimator


def test_windowed_slope_recovers_constant_warp_rate() -> None:
    # variant advances 3 frames for every 2 reference frames -> slope 1.5
    var = np.arange(300)
    ref = (var * 2) // 3
    path = np.stack([ref, var], axis=1).astype(np.int64)
    s = windowed_warp_slope(
        path,
        np.zeros(300, dtype=bool),
        hop_sec=0.01,
        window_sec=0.5,
        max_inactive_fraction=0.2,
    )
    assert np.nanmedian(s) == pytest.approx(1.5, abs=0.05)
    assert np.isfinite(s).mean() > 0.9


def test_windowed_slope_is_unavailable_when_reference_is_frozen_or_silent() -> None:
    var = np.arange(200)
    ref = np.minimum(var, 80)  # reference stops advancing at frame 80
    path = np.stack([ref, var], axis=1).astype(np.int64)
    s = windowed_warp_slope(
        path,
        np.zeros(200, dtype=bool),
        hop_sec=0.01,
        window_sec=0.5,
        max_inactive_fraction=0.2,
    )
    assert np.isnan(s[150:]).all()  # frozen reference: no slope, never a fabricated number
    silent = np.zeros(200, dtype=bool)
    silent[:] = True
    s2 = windowed_warp_slope(
        np.stack([var, var], axis=1).astype(np.int64),
        silent,
        hop_sec=0.01,
        window_sec=0.5,
        max_inactive_fraction=0.2,
    )
    assert np.isnan(s2).all()


# ----------------------------------------------------------- correlation policy


@pytest.mark.parametrize(
    ("ref", "var", "status"),
    [
        (np.full(10, 200.0), np.linspace(100, 190, 10), "unavailable_constant_f0"),
        (np.linspace(100, 190, 10), np.full(10, 200.0), "unavailable_constant_f0"),
        (
            np.array([100.0, 110.0]),
            np.array([100.0, 120.0]),
            "unavailable_insufficient_voiced_points",
        ),
        (np.array([]), np.array([]), "unavailable_insufficient_voiced_points"),
        (np.array([1.0, 2.0, np.nan, 4.0]), np.arange(4.0), "unavailable_non_finite_input"),
        (np.arange(5.0), np.arange(6.0), "unavailable_shape_mismatch"),
    ],
)
def test_f0_correlation_undefined_is_unavailable_not_zero_and_warning_free(
    ref: np.ndarray, var: np.ndarray, status: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        value, got = _f0_correlation(ref, var)
    assert value is None
    assert got == status


def test_f0_correlation_defined_case() -> None:
    x = np.linspace(100, 200, 20)
    value, status = _f0_correlation(x, 2 * x + 5)
    assert status == "ok"
    assert value == pytest.approx(1.0)
    anti, _ = _f0_correlation(x, -x)
    assert anti == pytest.approx(-1.0)


# ------------------------------------------------- evaluation identity and overwrite


def test_evaluation_report_identifies_detector_and_refuses_silent_overwrite(
    ds_config: Config, source: Any
) -> None:
    paths = ProjectPaths(ds_config)
    DatasetGenerator(ds_config).generate_variant(
        source.source_id,
        "pause_insertion",
        severity="medium",
        placement={"mode": "explicit", "at_sec": 1.5},
    )
    build_splits(ds_config, paths, force=True)
    runner = EvaluationRunner(ds_config, paths)
    report, run_dir = runner.run(split="all", run_id="rc1", include_controls=False)

    engine = DetectionEngine(ds_config.detection)
    assert report.detector_version == engine.detector_version
    assert report.detector_config_fingerprint == engine.config_fingerprint()
    assert report.metadata["flawbench_version"]
    md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert f"**Detector Version**: {engine.detector_version}" in md
    assert engine.config_fingerprint() in md
    assert "non-synthetic" not in md  # registry cannot tell synthetic from recorded audio
    before = (run_dir / "evaluation_report.json").read_bytes()

    with pytest.raises(EvaluationStoreError):
        runner.run(split="all", run_id="rc1", include_controls=False)
    assert (run_dir / "evaluation_report.json").read_bytes() == before  # untouched

    runner.run(split="all", run_id="rc1", include_controls=False, overwrite=True)


def test_pause_voicing_guard_blocks_attenuated_speech_even_with_legacy_alignment(
    config: Config, ref_audio: np.ndarray
) -> None:
    """Under the Phase 6 alignment (cosine, c0 included) a -24 dB segment freezes the DTW
    reference and sits below the silence threshold, which used to yield a pause candidate.
    The segment is still voiced, so it must not be reported as a pause."""
    legacy = dataclasses.replace(
        config.detection, alignment_metric="cosine", alignment_exclude_energy_coefficient=False
    )
    inj = reduce_energy(ref_audio, start=2 * SYN_RATE, end=4 * SYN_RATE, gain_db=-24.0)
    rf = extract_features(ref_audio[0], sample_rate=SYN_RATE, config=config)
    vf = extract_features(inj.samples[0], sample_rate=SYN_RATE, config=config)
    inp = build_detection_input_from_features(rf, vf, settings=legacy)
    assert PauseDetector(legacy).detect(inp) == []


def test_auto_generated_run_ids_are_unique_within_the_same_second(
    ds_config: Config, source: Any
) -> None:
    paths = ProjectPaths(ds_config)
    DatasetGenerator(ds_config).generate_variant(
        source.source_id,
        "pause_insertion",
        severity="medium",
        placement={"mode": "explicit", "at_sec": 1.5},
    )
    build_splits(ds_config, paths, force=True)
    runner = EvaluationRunner(ds_config, paths)
    ids = {runner.run(split="all", include_controls=False)[0].run_id for _ in range(3)}
    assert len(ids) == 3  # three runs, three distinct directories, none overwritten
