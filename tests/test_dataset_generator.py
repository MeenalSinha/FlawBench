from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flawbench.audio.loader import load_audio
from flawbench.config import Config, load_config
from flawbench.dataset.generator import (
    DatasetGenerator,
    derive_seed,
    generated_audio_settings,
)
from flawbench.dataset.io import read_jsonl, sha256_file
from flawbench.dataset.plan import parse_plan
from flawbench.dataset.registry import SourceRegistry
from flawbench.dataset.schema import SampleRecord, SourceRecord
from flawbench.dataset.validator import validate_dataset
from flawbench.exceptions import DatasetError, GenerationError, InjectionError, RegistryError
from tests.conftest import write_fixture

EXPLICIT = {"mode": "explicit"}


@pytest.fixture
def gen(ds_config: Config, source: SourceRecord) -> DatasetGenerator:  # source registers first
    return DatasetGenerator(ds_config)


def pause_args() -> dict:  # type: ignore[type-arg]
    return {"severity": "medium", "placement": {"mode": "explicit", "at_sec": 2.0}}


def test_single_pause_variant_end_to_end(
    gen: DatasetGenerator, source: SourceRecord, ds_root: Path
) -> None:
    ref_path = ds_root / "data" / "reference" / "fixture_a.wav"
    ref_before = sha256_file(ref_path)
    res = gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    assert res.status == "generated"
    rec = res.record.data
    ann = rec["annotation"]
    assert ann["output_interval"]["start_sample"] == 32000
    assert ann["output_interval"]["end_sample"] == 48000  # 1.0 s at 16 kHz
    assert ann["output_interval"]["start_sec"] == 2.0 and ann["output_interval"]["end_sec"] == 3.0
    assert ann["source_interval"] is None and ann["source_anchor"] == {"sample": 32000, "sec": 2.0}
    assert (ann["severity"], ann["parameters"]) == ("medium", {"duration_sec": 1.0})
    assert ann["effective_parameters"] == {"silence_samples": 16000}
    assert rec["seed"] is None  # deterministic operation: no invented seed
    # Generated file matches the record.
    audio = load_audio(res.audio_path, generated_audio_settings(gen.config))
    assert audio.num_samples == rec["generated"]["frames"] == 96000 + 16000
    assert sha256_file(res.audio_path) == rec["generated"]["sha256"]
    assert not audio.samples[:, 32000:48000].any()
    # Source is intact.
    assert sha256_file(ref_path) == ref_before == rec["reference"]["sha256"]
    # Manifest parses back into the same record.
    assert [s.sample_id for s in gen.samples()] == [res.record.sample_id]
    assert validate_dataset(gen.config).ok


@pytest.mark.parametrize(
    "flaw, severity, placement",
    [
        ("pause_insertion", "low", {"mode": "explicit", "at_sec": 1.0}),
        ("tempo_change", "medium", {"mode": "explicit", "start_sec": 1.0}),
        ("tempo_change", "low", {"mode": "explicit", "start_sec": 0.5}),
        ("energy_reduction", "high", {"mode": "explicit", "start_sec": 2.5}),
    ],
)
def test_each_flaw_passes_deep_validation(
    gen: DatasetGenerator,
    source: SourceRecord,
    flaw: str,
    severity: str,
    placement: dict,  # type: ignore[type-arg]
) -> None:
    res = gen.generate_variant(source.source_id, flaw, severity=severity, placement=placement)
    ann = res.record.data["annotation"]
    assert ann["flaw_type"] == flaw  # exactly one flaw per variant
    report = validate_dataset(gen.config, deep=True)
    assert report.ok, report.to_json()


def test_tempo_annotation_uses_measured_length(gen: DatasetGenerator, source: SourceRecord) -> None:
    res = gen.generate_variant(
        source.source_id,
        "tempo_change",
        severity="high",
        placement={"mode": "explicit", "start_sec": 1.0},
    )
    d = res.record.data
    ann = d["annotation"]
    out_len = ann["output_interval"]["end_sample"] - ann["output_interval"]["start_sample"]
    assert out_len == ann["effective_parameters"]["output_segment_samples"]
    assert d["generated"]["frames"] == 96000 - 32000 + out_len
    assert ann["source_interval"]["end_sample"] - ann["source_interval"]["start_sample"] == 32000
    assert ann["source_interval"] != ann["output_interval"]  # timelines are not conflated
    assert d["transformation"]["implementation"]["name"] == "audiotsm.wsola"


def test_identical_request_is_idempotent_with_stable_id(
    gen: DatasetGenerator, source: SourceRecord
) -> None:
    first = gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    second = gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    assert second.status == "skipped_existing"
    assert second.record.sample_id == first.record.sample_id
    assert len(gen.samples()) == 1
    other = gen.generate_variant(
        source.source_id,
        "pause_insertion",
        severity="high",
        placement={"mode": "explicit", "at_sec": 2.0},
    )
    assert other.record.sample_id != first.record.sample_id and len(gen.samples()) == 2


def test_output_bytes_are_reproducible_across_time(
    gen: DatasetGenerator, source: SourceRecord, ds_root: Path
) -> None:
    import time

    a = gen.generate_variant(
        source.source_id,
        "energy_reduction",
        severity="low",
        placement={"mode": "explicit", "start_sec": 1.0},
    )
    sha_a = sha256_file(a.audio_path)
    time.sleep(1.1)  # libsndfile's PEAK chunk timestamp would change the bytes
    a.audio_path.unlink()
    manifest = gen.paths.manifest_file
    manifest.write_text("")
    b = gen.generate_variant(
        source.source_id,
        "energy_reduction",
        severity="low",
        placement={"mode": "explicit", "start_sec": 1.0},
    )
    assert sha256_file(b.audio_path) == sha_a


def test_random_placement_is_seeded_recorded_and_reproducible(
    gen: DatasetGenerator, source: SourceRecord
) -> None:
    rnd = {"mode": "random"}
    with pytest.raises(GenerationError, match="explicit seed"):
        gen.generate_variant(source.source_id, "energy_reduction", severity="low", placement=rnd)
    a = gen.generate_variant(
        source.source_id, "energy_reduction", severity="low", placement=rnd, seed=7
    )
    again = gen.generate_variant(
        source.source_id, "energy_reduction", severity="low", placement=rnd, seed=7
    )
    assert again.status == "skipped_existing" and again.record.sample_id == a.record.sample_id
    seed = a.record.data["seed"]
    assert seed["run_seed"] == 7 and seed["used_for"] == "placement"
    key = f"{source.source_id}:energy_reduction:low"
    assert seed["sample_seed"] == derive_seed(7, key)
    iv = a.record.data["annotation"]["output_interval"]
    margin = int(gen.config.dataset.placement_margin_sec * 16000)
    assert margin <= iv["start_sample"] and iv["end_sample"] <= 96000 - margin
    different = gen.generate_variant(
        source.source_id, "energy_reduction", severity="low", placement=rnd, seed=8
    )
    assert different.record.data["annotation"]["output_interval"] != iv


def test_explicit_placement_ignores_seed_and_records_none(
    gen: DatasetGenerator, source: SourceRecord
) -> None:
    res = gen.generate_variant(source.source_id, "pause_insertion", seed=99, **pause_args())
    assert res.record.data["seed"] is None


def test_random_placement_rejected_when_recording_too_short(ds_root: Path) -> None:
    cfg = load_config(project_root=ds_root, env={"FLAWBENCH_PLACEMENT_MARGIN_SEC": "2.5"})
    reg = SourceRegistry(cfg)
    rec = reg.register(
        write_fixture(ds_root, "short.wav", seconds=4.0),
        transcript_text="t",
        source_type="self_recorded",
        attribution="a",
        license="l",
        rights_confirmed=True,
    )
    with pytest.raises(InjectionError, match="too short"):
        DatasetGenerator(cfg).generate_variant(
            rec.source_id, "pause_insertion", severity="low", placement={"mode": "random"}, seed=1
        )


def test_custom_parameters_are_labelled_and_validated(
    gen: DatasetGenerator, source: SourceRecord
) -> None:
    res = gen.generate_variant(
        source.source_id,
        "pause_insertion",
        severity="low",
        parameters={"duration_sec": 0.25},
        placement={"mode": "explicit", "at_sec": 1.0},
    )
    ann = res.record.data["annotation"]
    assert (ann["severity"], ann["base_severity"], ann["parameters"]) == (
        "custom",
        "low",
        {"duration_sec": 0.25},
    )
    with pytest.raises(GenerationError, match="bounds"):
        gen.generate_variant(
            source.source_id,
            "pause_insertion",
            parameters={"duration_sec": 500},
            placement={"mode": "explicit", "at_sec": 1.0},
        )
    with pytest.raises(GenerationError, match="No injector|no configured|Unknown flaw"):
        gen.generate_variant(
            source.source_id,
            "pitch_flattening",
            severity="low",
            placement={"mode": "random"},
            seed=1,
        )


def test_invalid_placement_fails_without_partial_files(
    gen: DatasetGenerator, source: SourceRecord
) -> None:
    with pytest.raises(InjectionError):
        gen.generate_variant(
            source.source_id,
            "pause_insertion",
            severity="low",
            placement={"mode": "explicit", "at_sec": 600.0},
        )
    with pytest.raises(InjectionError):
        gen.generate_variant(
            source.source_id,
            "energy_reduction",
            severity="low",
            placement={"mode": "explicit", "start_sec": 5.5},
        )  # 2 s segment overruns
    assert gen.samples() == []
    gdir = gen.paths.dataset_generated_dir
    assert not gdir.exists() or not any(gdir.iterdir())


def test_output_limit_enforced(ds_root: Path, source: SourceRecord) -> None:
    cfg = load_config(project_root=ds_root, env={"FLAWBENCH_MAX_OUTPUT_SECONDS": "6.5"})
    with pytest.raises(GenerationError, match="limit"):
        DatasetGenerator(cfg).generate_variant(
            source.source_id,
            "pause_insertion",
            severity="high",  # +2 s -> 8 s > 6.5 s
            placement={"mode": "explicit", "at_sec": 1.0},
        )


def test_source_modified_after_registration_blocks_generation(
    gen: DatasetGenerator, source: SourceRecord, ds_root: Path
) -> None:
    ref = ds_root / "data" / "reference" / "fixture_a.wav"
    ref.write_bytes(ref.read_bytes()[:-4] + b"\x00\x00\x00\x00")
    with pytest.raises(RegistryError, match="checksum mismatch"):
        gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    assert gen.samples() == []


def test_overwrite_policy_and_history(gen: DatasetGenerator, source: SourceRecord) -> None:
    first = gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    first.audio_path.write_bytes(b"tampered")
    with pytest.raises(GenerationError, match="already exists"):
        gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    assert first.audio_path.read_bytes() == b"tampered"  # refused: not replaced
    again = gen.generate_variant(
        source.source_id, "pause_insertion", overwrite=True, **pause_args()
    )
    assert again.status == "regenerated"
    hist = again.record.data["history"]
    assert (
        len(hist) == 1
        and hist[0]["previous_generated_sha256"] == first.record.data["generated"]["sha256"]
    )
    assert validate_dataset(gen.config).ok and len(gen.samples()) == 1


def test_overwrite_allowed_from_configuration(ds_root: Path, source: SourceRecord) -> None:
    cfg = load_config(project_root=ds_root, env={"FLAWBENCH_ALLOW_OVERWRITE": "true"})
    g = DatasetGenerator(cfg)
    first = g.generate_variant(source.source_id, "pause_insertion", **pause_args())
    first.audio_path.write_bytes(b"x")
    assert (
        g.generate_variant(source.source_id, "pause_insertion", **pause_args()).status
        == "regenerated"
    )


def test_orphan_identical_file_is_adopted_and_foreign_file_refused(
    gen: DatasetGenerator, source: SourceRecord
) -> None:
    res = gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    gen.paths.manifest_file.write_text("")  # simulate crash between file commit and manifest write
    assert any(f.code == "unregistered_file" for f in validate_dataset(gen.config).findings)
    adopted = gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    assert adopted.status == "generated" and validate_dataset(gen.config).ok
    gen.paths.manifest_file.write_text("")
    res.audio_path.write_bytes(b"foreign")
    with pytest.raises(GenerationError, match="unregistered file"):
        gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    assert res.audio_path.read_bytes() == b"foreign"


def test_manifest_write_failure_rolls_back_the_output(
    gen: DatasetGenerator, source: SourceRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    import flawbench.dataset.generator as g

    def boom(*a: object, **k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(g, "write_jsonl", boom)
    with pytest.raises(OSError):
        gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    gdir = gen.paths.dataset_generated_dir
    assert not list(gdir.iterdir()) and gen.samples() == []


def test_provenance_fields_are_recorded(gen: DatasetGenerator, source: SourceRecord) -> None:
    from flawbench.provenance import config_fingerprint

    res = gen.generate_variant(
        source.source_id, "pause_insertion", plan_fingerprint="abc", **pause_args()
    )
    d = res.record.data
    assert d["schema_version"] == "1.0" and d["status"] == "committed"
    assert d["dataset_id"] == "flawbench-v1" and d["plan_fingerprint"] == "abc"
    assert d["config_fingerprint"] == config_fingerprint(gen.config, sections=("dataset",))
    sw = d["software"]
    assert sw["flawbench_version"] and sw["python"] and sw["libsndfile"]
    assert {"numpy", "soundfile", "audiotsm"} <= set(sw["packages"])
    assert d["generated_utc"].endswith("+00:00")
    assert d["transcript_ref"]["source_id"] == source.source_id
    assert d["generated"]["path"] == f"flawbench-v1/{d['sample_id']}.wav"
    SampleRecord.from_dict(d)


def test_no_ground_truth_leaks_into_generated_audio_or_name(
    gen: DatasetGenerator, source: SourceRecord
) -> None:
    res = gen.generate_variant(
        source.source_id,
        "tempo_change",
        severity="high",
        placement={"mode": "explicit", "start_sec": 1.0},
    )
    name = res.audio_path.name
    blob = res.audio_path.read_bytes()
    for token in ("tempo", "pause", "energy", "high", "severity", "start_sec"):
        assert token not in name and token.encode() not in blob


def test_pcm_output_subtype_is_supported_and_validated(ds_root: Path, source: SourceRecord) -> None:
    cfg = load_config(project_root=ds_root, env={"FLAWBENCH_OUTPUT_SUBTYPE": "PCM_16"})
    g = DatasetGenerator(cfg)
    res = g.generate_variant(
        source.source_id,
        "energy_reduction",
        severity="medium",
        placement={"mode": "explicit", "start_sec": 1.0},
    )
    assert res.record.data["generated"]["subtype"] == "PCM_16"
    assert validate_dataset(cfg).ok


# -------------------------------------------------------------------------- batch
def make_plan(variants: list, seed: int | None = 5):  # type: ignore[no-untyped-def,type-arg]
    return parse_plan({"plan_version": 1, "name": "t", "seed": seed, "variants": variants})


def test_batch_generation_and_rerun(gen: DatasetGenerator, source: SourceRecord) -> None:
    plan = make_plan(
        [
            {
                "source": "*",
                "flaw": "pause_insertion",
                "severity": "low",
                "placement": {"mode": "explicit", "at_sec": 1.0},
            },
            {
                "source": source.source_id,
                "flaw": "energy_reduction",
                "severity": "low",
                "placement": {"mode": "random"},
                "count": 3,
            },
        ]
    )
    batch = gen.run_plan(plan)
    assert batch.ok and batch.counts == {"generated": 4}
    assert len(gen.samples()) == 4 and validate_dataset(gen.config).ok
    again = gen.run_plan(plan)
    assert again.counts == {"skipped_existing": 4} and len(gen.samples()) == 4
    runs = read_jsonl(gen.paths.runs_file, what="runs")
    assert len(runs) == 2 and runs[0]["plan_fingerprint"] == plan.fingerprint
    assert runs[0]["provenance"]["random_seed"] == 5 and runs[0]["complete_success"] is True
    # Replicates have distinct derived seeds.
    seeds = {s.data["seed"]["sample_seed"] for s in gen.samples() if s.data["seed"]}
    assert len(seeds) == 3


def test_batch_partial_failure_is_reported_not_hidden(
    gen: DatasetGenerator, source: SourceRecord
) -> None:
    plan = make_plan(
        [
            {
                "source": source.source_id,
                "flaw": "pause_insertion",
                "severity": "low",
                "placement": {"mode": "explicit", "at_sec": 1.0},
            },
            {
                "source": source.source_id,
                "flaw": "pause_insertion",
                "severity": "low",
                "placement": {"mode": "explicit", "at_sec": 500.0},
            },
        ],
        seed=None,
    )
    batch = gen.run_plan(plan)
    assert not batch.ok and batch.counts == {"generated": 1, "failed": 1}
    failed = next(o for o in batch.outcomes if o.status == "failed")
    assert failed.sample_id is None and "outside" in (failed.error or "")
    assert len(gen.samples()) == 1  # only the successful variant is registered
    assert not [p for p in gen.paths.dataset_generated_dir.iterdir() if p.name.startswith(".")]
    run = read_jsonl(gen.paths.runs_file, what="runs")[-1]
    assert run["complete_success"] is False and run["counts"] == {"failed": 1, "generated": 1}
    assert validate_dataset(gen.config).ok  # the committed part is still a valid dataset


def test_plan_is_validated_before_anything_is_generated(
    gen: DatasetGenerator, source: SourceRecord
) -> None:
    bad_source = make_plan(
        [
            {
                "source": "src-nope",
                "flaw": "pause_insertion",
                "severity": "low",
                "placement": {"mode": "explicit", "at_sec": 1.0},
            }
        ],
        seed=None,
    )
    with pytest.raises(DatasetError, match="unregistered"):
        gen.run_plan(bad_source)
    bad_severity = make_plan(
        [
            {
                "source": source.source_id,
                "flaw": "pause_insertion",
                "severity": "low",
                "placement": {"mode": "explicit", "at_sec": 1.0},
            },
            {
                "source": source.source_id,
                "flaw": "pause_insertion",
                "severity": "enormous",
                "placement": {"mode": "explicit", "at_sec": 1.0},
            },
        ],
        seed=None,
    )
    with pytest.raises(DatasetError, match="Unknown severity"):
        gen.run_plan(bad_severity)
    assert gen.samples() == []  # the valid first entry was NOT run


def test_variant_limit_enforced(ds_root: Path, source: SourceRecord) -> None:
    cfg = load_config(project_root=ds_root, env={"FLAWBENCH_MAX_VARIANTS_PER_RUN": "2"})
    plan = make_plan(
        [
            {
                "source": source.source_id,
                "flaw": "pause_insertion",
                "severity": "low",
                "placement": {"mode": "explicit", "at_sec": 1.0},
                "count": 3,
            }
        ],
        seed=None,
    )
    with pytest.raises(DatasetError, match="limit"):
        DatasetGenerator(cfg).run_plan(plan)


@pytest.mark.parametrize(
    "data, fragment",
    [
        ({"plan_version": 2, "variants": [{}]}, "plan_version"),
        ({"plan_version": 1, "variants": []}, "non-empty"),
        ({"plan_version": 1, "extra": 1, "variants": [{}]}, "Unknown plan key"),
        (
            {
                "plan_version": 1,
                "variants": [
                    {"source": "a", "flaw": "x", "placement": {"mode": "random"}, "severity": "low"}
                ],
            },
            "unknown flaw",
        ),
        (
            {
                "plan_version": 1,
                "variants": [
                    {
                        "source": "a",
                        "flaw": "pause_insertion",
                        "placement": {"mode": "random"},
                        "severity": "low",
                    }
                ],
            },
            "seed",
        ),
        (
            {
                "plan_version": 1,
                "seed": 1,
                "variants": [
                    {
                        "source": "a",
                        "flaw": "pause_insertion",
                        "placement": {"mode": "explicit", "start_sec": 1},
                        "severity": "low",
                    }
                ],
            },
            "at_sec",
        ),
        (
            {
                "plan_version": 1,
                "variants": [
                    {
                        "source": "a",
                        "flaw": "pause_insertion",
                        "placement": {"mode": "explicit", "at_sec": 1},
                    }
                ],
            },
            "severity or explicit",
        ),
        (
            {
                "plan_version": 1,
                "variants": [
                    {
                        "source": "a",
                        "flaw": "pause_insertion",
                        "placement": {"mode": "explicit", "at_sec": 1},
                        "severity": "low",
                        "count": 0,
                    }
                ],
            },
            "count",
        ),
    ],
)
def test_invalid_plans(data: dict, fragment: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(DatasetError, match=fragment):
        parse_plan(data)


def test_plan_fingerprint_changes_with_content_and_seed_override() -> None:
    v = [
        {
            "source": "a",
            "flaw": "pause_insertion",
            "severity": "low",
            "placement": {"mode": "random"},
        }
    ]
    p1 = parse_plan({"plan_version": 1, "seed": 1, "variants": v})
    assert p1.fingerprint == parse_plan({"plan_version": 1, "seed": 1, "variants": v}).fingerprint
    assert p1.fingerprint != parse_plan({"plan_version": 1, "seed": 2, "variants": v}).fingerprint
    assert (
        parse_plan({"plan_version": 1, "variants": v}, seed_override=1).fingerprint
        == p1.fingerprint
    )


def test_manifest_records_are_valid_json_lines(gen: DatasetGenerator, source: SourceRecord) -> None:
    gen.generate_variant(source.source_id, "pause_insertion", **pause_args())
    lines = gen.paths.manifest_file.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["record_type"] == "sample"
    assert np.isfinite(json.loads(lines[0])["generated"]["duration_seconds"])


def test_independent_rebuilds_reproduce_identical_ids_and_bytes(tmp_path: Path) -> None:
    """Same source bytes + config + plan + seed in two separate project roots -> same outputs."""
    results = []
    for tag in ("one", "two"):
        root = tmp_path / tag
        (root / "data" / "reference").mkdir(parents=True)
        cfg = load_config(project_root=root, env={})
        rec = SourceRegistry(cfg).register(
            write_fixture(root, "f.wav", seed=1),
            transcript_text="t",
            source_type="self_recorded",
            attribution="a",
            license="l",
            rights_confirmed=True,
        )
        plan = make_plan(
            [
                {
                    "source": "*",
                    "flaw": "pause_insertion",
                    "severity": "medium",
                    "placement": {"mode": "random"},
                },
                {
                    "source": "*",
                    "flaw": "tempo_change",
                    "severity": "low",
                    "placement": {"mode": "random"},
                },
                {
                    "source": "*",
                    "flaw": "energy_reduction",
                    "severity": "high",
                    "placement": {"mode": "random"},
                },
            ],
            seed=2026,
        )
        DatasetGenerator(cfg).run_plan(plan)
        samples = DatasetGenerator(cfg).samples()
        results.append(
            {
                s.sample_id: (
                    s.data["generated"]["sha256"],
                    s.data["annotation"]["output_interval"],
                    s.data["annotation"]["source_interval"],
                )
                for s in samples
            }
        )
        assert rec.source_id  # identical source bytes give the same source id too
    assert results[0] == results[1] and len(results[0]) == 3


def test_example_plan_file_is_valid() -> None:
    from flawbench.dataset.plan import load_plan

    plan = load_plan(Path(__file__).resolve().parents[1] / "configs" / "plan.example.yaml")
    assert plan.uses_random_placement and plan.seed == 42 and len(plan.entries) == 3
