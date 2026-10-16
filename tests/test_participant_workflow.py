"""Participant-analysis workflow: registered and uploaded modes, failures, persistence."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from flawbench.config import Config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.injectors import insert_pause
from flawbench.dataset.registry import SourceRegistry
from flawbench.dataset.schema import SourceRecord
from flawbench.exceptions import ParticipantInputError
from flawbench.explainability.schema import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NO_CANDIDATES,
    STATUS_PARTIAL,
)
from flawbench.participant import ParticipantAnalyzer
from tests.conftest import REGISTER_DEFAULTS, SYN_RATE


def _wav_bytes(samples: np.ndarray, rate: int = SYN_RATE) -> bytes:
    """Encode ``(channels, n)`` or ``(n,)`` float samples as 16-bit WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, samples.T if samples.ndim == 2 else samples, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _reference_samples(registry: SourceRegistry, source: SourceRecord) -> np.ndarray:
    return registry.load_audio(source).samples


def _pause_upload(registry: SourceRegistry, source: SourceRecord, seconds: float = 1.0) -> bytes:
    ref = _reference_samples(registry, source)
    out = insert_pause(ref, at_sample=int(1.5 * SYN_RATE), silence_samples=int(seconds * SYN_RATE))
    return _wav_bytes(out.samples)


@pytest.fixture
def analyzer(ds_config: Config) -> ParticipantAnalyzer:
    return ParticipantAnalyzer(ds_config)


def test_upload_analysis_detects_pause_with_cards(
    analyzer: ParticipantAnalyzer, registry: SourceRegistry, source: SourceRecord
) -> None:
    session = analyzer.analyze_upload(source.source_id, _pause_upload(registry, source), "me.wav")
    a = session.analysis
    assert a.status in {STATUS_COMPLETED, STATUS_PARTIAL}
    assert "pause_insertion" in a.categories_detected
    assert a.participant is not None and a.participant.origin == "uploaded"
    assert a.participant.sample_rate == SYN_RATE
    assert [s.status for s in a.stages].count("completed") == 6
    assert session.artifacts is not None
    pause = next(c for c in a.cards if c.flaw_type == "pause_insertion")
    assert pause.status == "ok"
    assert pause.bundle.reference.status in {"anchor", "interval", "unavailable"}


def test_upload_is_not_persisted(
    analyzer: ParticipantAnalyzer,
    registry: SourceRegistry,
    source: SourceRecord,
    ds_root: Path,
) -> None:
    upload = _pause_upload(registry, source)
    analyzer.analyze_upload(source.source_id, upload, "private.wav")  # warm reference features
    before = {p: p.stat().st_size for p in ds_root.rglob("*") if p.is_file()}
    analyzer.analyze_upload(source.source_id, upload, "private.wav")
    after = {p: p.stat().st_size for p in ds_root.rglob("*") if p.is_file()}
    assert before == after
    assert not any("private" in p.name for p in after)


def test_zero_candidate_result_is_not_a_failure(
    analyzer: ParticipantAnalyzer, registry: SourceRegistry, source: SourceRecord
) -> None:
    same = _wav_bytes(_reference_samples(registry, source))
    a = analyzer.analyze_upload(source.source_id, same, "same.wav").analysis
    assert a.status == STATUS_NO_CANDIDATES
    assert a.cards == [] and a.failure is None


@pytest.mark.parametrize(
    ("data", "name"), [(b"", "a.wav"), (b"not audio at all", "a.wav"), (b"RIFFxxxx", "a.wav")]
)
def test_invalid_upload_fails_distinctly(
    analyzer: ParticipantAnalyzer, source: SourceRecord, data: bytes, name: str
) -> None:
    a = analyzer.analyze_upload(source.source_id, data, name).analysis
    assert a.status == STATUS_FAILED and a.cards == [] and a.failure
    assert a.stages[0].status == "failed"
    assert all(s.status == "skipped" for s in a.stages[1:])


def test_missing_reference_fails(analyzer: ParticipantAnalyzer) -> None:
    a = analyzer.analyze_upload("", b"x", "a.wav").analysis
    assert a.status == STATUS_FAILED
    a = analyzer.analyze_upload("src_missing", b"x", "a.wav").analysis
    assert a.status == STATUS_FAILED


def test_unknown_registered_variant_fails(analyzer: ParticipantAnalyzer) -> None:
    assert analyzer.analyze_registered("sample_nope").analysis.status == STATUS_FAILED


def test_partial_detector_failure_is_reported(
    analyzer: ParticipantAnalyzer,
    registry: SourceRegistry,
    source: SourceRecord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flawbench.detection import detectors

    def boom(self: object, inp: object) -> list[object]:
        raise RuntimeError("simulated detector crash")

    monkeypatch.setattr(detectors.EnergyReductionDetector, "detect", boom)
    a = analyzer.analyze_upload(source.source_id, _pause_upload(registry, source), "p.wav").analysis
    assert a.status == STATUS_PARTIAL
    assert "energy_reduction" in a.categories_failed
    assert "energy_reduction" not in a.categories_completed
    assert any("partial" in w.lower() for w in a.warnings)


def test_different_sample_rate_is_handled(
    analyzer: ParticipantAnalyzer, registry: SourceRegistry, source: SourceRecord
) -> None:
    import librosa

    ref = _reference_samples(registry, source)
    out = insert_pause(ref, at_sample=int(1.5 * SYN_RATE), silence_samples=SYN_RATE).samples
    res = librosa.resample(out.astype(np.float32), orig_sr=SYN_RATE, target_sr=22050, axis=-1)
    a = analyzer.analyze_upload(source.source_id, _wav_bytes(res, 22050), "r.wav").analysis
    assert a.status != STATUS_FAILED
    assert a.participant is not None and a.participant.sample_rate == 22050
    for c in a.cards:
        assert c.bundle.variant.sample_rate == 22050


def test_registered_variant_analysis_ignores_hidden_annotations(
    analyzer: ParticipantAnalyzer, ds_config: Config, source: SourceRecord
) -> None:
    gen = DatasetGenerator(ds_config)
    r = gen.generate_variant(
        source.source_id,
        "pause_insertion",
        severity="medium",
        placement={"mode": "explicit", "at_sec": 1.5},
    )
    sid = r.record.sample_id
    variants = analyzer.registered_variants(source.source_id)
    assert [v.sample_id for v in variants] == [sid]
    assert not hasattr(variants[0], "annotation")

    first = analyzer.analyze_registered(sid).analysis
    assert first.status in {STATUS_COMPLETED, STATUS_PARTIAL}
    text = json.dumps(first.to_dict())
    assert "ground_truth" not in text and "annotation" not in text

    from flawbench.paths import ProjectPaths

    mp = ProjectPaths(ds_config).manifest_file
    assert mp.is_file()
    rows = [json.loads(line) for line in mp.read_text().splitlines() if line.strip()]
    for row in rows:
        ann = row["annotation"]
        ann["flaw_type"] = "energy_reduction"
        ann["output_interval"] = {
            "start_sample": 1,
            "end_sample": 2,
            "start_sec": 0.0,
            "end_sec": 0.0,
        }
    mp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    second = analyzer.analyze_registered(sid).analysis

    def strip(card: object) -> dict:
        d = card.to_dict()  # type: ignore[attr-defined]
        d.pop("prediction_id")
        d["evidence_bundle"].pop("prediction_id")
        d["evidence_bundle"].pop("provenance")  # run ids are random per detection run
        return d

    assert [strip(c) for c in first.cards] == [strip(c) for c in second.cards]
    assert first.cards


def test_explicit_registration_requires_rights_and_cleans_up(
    analyzer: ParticipantAnalyzer, registry: SourceRegistry, source: SourceRecord
) -> None:
    upload = _pause_upload(registry, source)
    n = len(registry.all())
    with pytest.raises(ParticipantInputError):
        analyzer.register_upload(
            upload,
            "u.wav",
            source_type="self_recorded",
            attribution="me",
            license="CC0",
            rights_confirmed=False,
            transcript_text="t",
        )
    assert len(registry.all()) == n
    sid = analyzer.register_upload(
        upload,
        "u.wav",
        source_type=REGISTER_DEFAULTS["source_type"],
        attribution="me",
        license="CC0",
        rights_confirmed=True,
        transcript_text="synthetic upload",
    )
    assert len(registry.all()) == n + 1 and registry.get(sid)
    import tempfile

    assert not list(Path(tempfile.gettempdir()).glob("flawbench_upload_*"))


def test_registration_disabled_by_policy(
    ds_config: Config, registry: SourceRegistry, source: SourceRecord
) -> None:
    from dataclasses import replace

    cfg = replace(
        ds_config, explanation=replace(ds_config.explanation, upload_persistence="transient_only")
    )
    with pytest.raises(ParticipantInputError):
        ParticipantAnalyzer(cfg).register_upload(
            b"x",
            "u.wav",
            source_type="self_recorded",
            attribution="m",
            license="l",
            rights_confirmed=True,
            transcript_text="t",
        )


def test_cli_explain_registered_and_upload(
    ds_config: Config,
    registry: SourceRegistry,
    source: SourceRecord,
    ds_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from flawbench.__main__ import main

    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(ds_root))
    wav = ds_root / "mine.wav"
    wav.write_bytes(_pause_upload(registry, source))
    assert main(["explain", "upload", "--source", source.source_id, "--audio", str(wav)]) == 0
    out = capsys.readouterr().out
    assert "pause" in out.lower() and "stage compare_and_detect: completed" in out
    assert not (ds_root / "outputs" / "participant_reports").exists()  # no export unrequested
    assert (
        main(["explain", "upload", "--source", source.source_id, "--audio", str(wav), "--export"])
        == 0
    )
    assert list((ds_root / "outputs" / "participant_reports").glob("*.json"))
    assert main(["explain", "registered", "--sample", "smp-missing"]) == 1
