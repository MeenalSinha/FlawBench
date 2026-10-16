from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from flawbench.config import Config, load_config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.io import sha256_file
from flawbench.dataset.schema import SourceRecord
from flawbench.dataset.validator import validate_dataset


@pytest.fixture
def built(ds_config: Config, source: SourceRecord) -> DatasetGenerator:
    g = DatasetGenerator(ds_config)
    g.generate_variant(
        source.source_id,
        "pause_insertion",
        severity="medium",
        placement={"mode": "explicit", "at_sec": 2.0},
    )
    g.generate_variant(
        source.source_id,
        "tempo_change",
        severity="medium",
        placement={"mode": "explicit", "start_sec": 1.0},
    )
    g.generate_variant(
        source.source_id,
        "energy_reduction",
        severity="high",
        placement={"mode": "explicit", "start_sec": 3.0},
    )
    return g


def codes(report) -> set[str]:  # type: ignore[no-untyped-def]
    return {f.code for f in report.findings}


def edit_manifest(g: DatasetGenerator, index: int, mutate) -> None:  # type: ignore[no-untyped-def]
    lines = g.paths.manifest_file.read_text().splitlines()
    rec = json.loads(lines[index])
    mutate(rec)
    lines[index] = json.dumps(rec)
    g.paths.manifest_file.write_text("\n".join(lines) + "\n")


def test_valid_dataset_reports_real_counts(built: DatasetGenerator) -> None:
    report = validate_dataset(built.config)
    assert report.ok and not report.errors
    assert report.counts["sources"] == 1 and report.counts["samples"] == 3
    assert not report.is_empty and report.to_dict()["ok"] is True
    json.loads(report.to_json())


def test_empty_dataset_is_reported_honestly(ds_config: Config) -> None:
    report = validate_dataset(ds_config)
    assert report.is_empty and report.counts["sources"] == 0 and report.counts["samples"] == 0
    assert codes(report) == {"empty_dataset"} and report.to_dict()["empty_dataset"] is True


def test_missing_generated_file(built: DatasetGenerator) -> None:
    next(built.paths.dataset_generated_dir.glob("*.wav")).unlink()
    report = validate_dataset(built.config)
    assert not report.ok and "sample_file_missing" in codes(report)


def test_missing_and_modified_source(built: DatasetGenerator, ds_root: Path) -> None:
    ref = ds_root / "data" / "reference" / "fixture_a.wav"
    ref.write_bytes(ref.read_bytes()[:-2] + b"\x07\x07")
    assert "source_checksum_mismatch" in codes(validate_dataset(built.config))
    ref.unlink()
    report = validate_dataset(built.config)
    assert not report.ok and "source_missing" in codes(report)


def test_invalid_interval_detected(built: DatasetGenerator) -> None:
    def mutate(r: dict) -> None:  # type: ignore[type-arg]
        r["annotation"]["output_interval"]["end_sample"] = 10**9

    edit_manifest(built, 0, mutate)
    report = validate_dataset(built.config)
    assert not report.ok and "sample_bad_interval" in codes(report)

    def mutate2(r: dict) -> None:  # type: ignore[type-arg]
        iv = r["annotation"]["output_interval"]
        iv["start_sample"], iv["end_sample"] = iv["end_sample"], iv["start_sample"]

    edit_manifest(built, 1, mutate2)
    assert "sample_bad_interval" in codes(validate_dataset(built.config))


def test_inconsistent_annotation_detected(built: DatasetGenerator) -> None:
    def mutate(r: dict) -> None:  # type: ignore[type-arg]
        r["annotation"]["effective_parameters"]["silence_samples"] = 5

    edit_manifest(built, 0, mutate)
    assert "sample_inconsistent_annotation" in codes(validate_dataset(built.config))


def test_generated_checksum_mismatch(built: DatasetGenerator) -> None:
    f = next(built.paths.dataset_generated_dir.glob("*.wav"))
    f.write_bytes(f.read_bytes() + b"\x00")
    assert "sample_checksum_mismatch" in codes(validate_dataset(built.config))


def test_deep_check_catches_audio_that_contradicts_the_annotation(built: DatasetGenerator) -> None:
    """Re-record a correct-looking file with different content and a matching checksum/duration:
    only content verification can notice that the ground truth is wrong."""
    first = built.samples()[0]  # pause at 2.0 s
    path = built.paths.generated_dir / first.data["generated"]["path"]
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data[40000:40100, 0] = 0.25  # corrupt audio *inside* the silent interval
    sf.write(path, data, sr, subtype="FLOAT", format="WAV")

    def mutate(r: dict) -> None:  # type: ignore[type-arg]
        r["generated"]["sha256"] = sha256_file(path)

    edit_manifest(built, 0, mutate)
    shallow = validate_dataset(built.config, deep=False)
    deep = validate_dataset(built.config, deep=True)
    assert shallow.ok and not deep.ok and "sample_content_mismatch" in codes(deep)


def test_deep_check_detects_wrong_tempo_and_energy_claims(built: DatasetGenerator) -> None:
    def mutate(r: dict) -> None:  # type: ignore[type-arg]
        r["annotation"]["effective_parameters"]["amplitude_multiplier"] = 0.9

    edit_manifest(built, 2, mutate)
    assert "sample_content_mismatch" in codes(validate_dataset(built.config))


def test_duplicate_ids_and_malformed_lines(built: DatasetGenerator) -> None:
    lines = built.paths.manifest_file.read_text().splitlines()
    built.paths.manifest_file.write_text(
        "\n".join([*lines, lines[0], "{broken", '["list"]']) + "\n"
    )
    report = validate_dataset(built.config)
    assert not report.ok
    assert {"sample_duplicate_id", "manifest_malformed"} <= codes(report)
    assert report.counts["samples"] == 3  # valid records are still counted once


def test_unsafe_paths_are_rejected(built: DatasetGenerator) -> None:
    for bad in ("../../escape.wav", "/etc/passwd", "a/../../b.wav"):

        def mutate(r: dict, bad: str = bad) -> None:  # type: ignore[type-arg]
            r["generated"]["path"] = bad

        edit_manifest(built, 0, mutate)
        assert "sample_unsafe_path" in codes(validate_dataset(built.config)), bad
    lines = built.paths.sources_file.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["audio_path"] = "../outside.wav"
    built.paths.sources_file.write_text(json.dumps(rec) + "\n")
    assert "source_unsafe_path" in codes(validate_dataset(built.config))


def test_missing_rights_and_transcript_are_errors(built: DatasetGenerator) -> None:
    rec = json.loads(built.paths.sources_file.read_text().splitlines()[0])
    rec["license"] = ""
    rec["transcript_text"] = " "
    rec["rights"]["use_confirmed"] = False
    built.paths.sources_file.write_text(json.dumps(rec) + "\n")
    got = codes(validate_dataset(built.config))
    assert {"source_no_rights_info", "source_no_transcript", "source_rights_unconfirmed"} <= got


def test_unknown_redistribution_is_a_warning_not_an_error(ds_root: Path, ds_config: Config) -> None:
    from flawbench.dataset.registry import SourceRegistry
    from tests.conftest import REGISTER_DEFAULTS, write_fixture

    SourceRegistry(ds_config).register(
        write_fixture(ds_root, "r.wav", seed=6),
        **{**REGISTER_DEFAULTS, "redistribution": "unknown"},
    )
    report = validate_dataset(ds_config)
    assert report.ok and "source_not_redistributable" in codes(report)


def test_partial_generation_leftovers_and_orphans_are_flagged(built: DatasetGenerator) -> None:
    d = built.paths.dataset_generated_dir
    (d / ".smp-halfdone.wav.tmp").write_bytes(b"partial")
    (d / "smp-unregistered.wav").write_bytes(b"x")
    report = validate_dataset(built.config)
    assert report.ok  # leftovers do not corrupt registered samples...
    assert {"partial_file", "unregistered_file"} <= codes(report)  # ...but are surfaced


def test_non_committed_status_and_missing_fields(built: DatasetGenerator) -> None:
    def mutate(r: dict) -> None:  # type: ignore[type-arg]
        r["status"] = "pending"

    edit_manifest(built, 0, mutate)
    assert "sample_not_committed" in codes(validate_dataset(built.config))

    def mutate2(r: dict) -> None:  # type: ignore[type-arg]
        del r["annotation"]

    edit_manifest(built, 1, mutate2)
    assert "sample_invalid" in codes(validate_dataset(built.config))


def test_duration_mismatch_between_record_and_file(built: DatasetGenerator) -> None:
    def mutate(r: dict) -> None:  # type: ignore[type-arg]
        r["generated"]["frames"] += 1

    edit_manifest(built, 0, mutate)
    assert not validate_dataset(built.config).ok


def test_samples_referencing_unregistered_source(built: DatasetGenerator) -> None:
    built.paths.sources_file.write_text("")
    report = validate_dataset(built.config)
    assert not report.ok and "sample_source_missing" in codes(report)


def test_undecodable_generated_file_with_matching_checksum(built: DatasetGenerator) -> None:
    first = built.samples()[0]
    path = built.paths.generated_dir / first.data["generated"]["path"]
    path.write_bytes(b"garbage" * 100)

    def mutate(r: dict) -> None:  # type: ignore[type-arg]
        r["generated"]["sha256"] = sha256_file(path)

    edit_manifest(built, 0, mutate)
    assert "sample_undecodable" in codes(validate_dataset(built.config))


def test_validation_handles_float_and_pcm_subtypes(ds_root: Path, source: SourceRecord) -> None:
    for subtype in ("PCM_16", "PCM_24"):
        cfg = load_config(
            project_root=ds_root,
            env={
                "FLAWBENCH_OUTPUT_SUBTYPE": subtype,
                "FLAWBENCH_DATASET_ID": f"ds-{subtype.lower()}",
            },
        )
        g = DatasetGenerator(cfg)
        for flaw, pl in (
            ("pause_insertion", {"mode": "explicit", "at_sec": 1.0}),
            ("tempo_change", {"mode": "explicit", "start_sec": 1.0}),
            ("energy_reduction", {"mode": "explicit", "start_sec": 1.0}),
        ):
            g.generate_variant(source.source_id, flaw, severity="low", placement=pl)
        assert validate_dataset(cfg).ok, subtype
        data = np.asarray(sf.info(next(g.paths.dataset_generated_dir.glob("*.wav"))).subtype)
        assert str(data) == subtype
