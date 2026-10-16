from __future__ import annotations

import json
from pathlib import Path

import pytest

from flawbench.__main__ import main
from flawbench.config import Config, load_config
from flawbench.dataset.export import export_samples
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.registry import SourceRegistry
from flawbench.dataset.schema import SourceRecord
from flawbench.dataset.splits import build_splits
from flawbench.exceptions import DatasetError
from tests.conftest import REGISTER_DEFAULTS, write_fixture


def register_many(cfg: Config, root: Path, n: int, start: int = 0) -> list[str]:
    reg = SourceRegistry(cfg)
    return [
        reg.register(
            write_fixture(root, f"s{i}.wav", seed=i, seconds=1.0), **REGISTER_DEFAULTS
        ).source_id
        for i in range(start, start + n)
    ]


# ------------------------------------------------------------------------- splits
def test_split_assignment_follows_ratios_and_is_persisted(ds_root: Path, ds_config: Config) -> None:
    ids = register_many(ds_config, ds_root, 10)
    r1 = build_splits(ds_config)
    # Largest-remainder allocation of 10 sources at 0.7/0.15/0.15 (tie broken by split name).
    assert r1.counts(["train", "validation", "test"]) == {"train": 7, "validation": 1, "test": 2}
    assert sorted(r1.assignments) == sorted(ids)
    r2 = build_splits(ds_config)
    assert r2.assignments == r1.assignments  # rebuilding does not reshuffle
    extra = register_many(ds_config, ds_root, 2, start=10)
    r3 = build_splits(ds_config)
    assert all(r3.assignments[k] == v for k, v in r1.assignments.items())
    assert set(extra) <= set(r3.assignments)


def test_split_is_deterministic_for_a_seed(ds_root: Path, tmp_path: Path) -> None:
    results = []
    for tag in ("a", "b"):
        root = tmp_path / tag
        (root / "data" / "reference").mkdir(parents=True)
        cfg = load_config(project_root=root, env={})
        register_many(cfg, root, 6)
        results.append(build_splits(cfg).assignments)
    assert results[0] == results[1]


def test_small_collections_get_honest_warnings(
    ds_root: Path, ds_config: Config, source: SourceRecord
) -> None:
    result = build_splits(ds_config)
    text = " ".join(result.warnings)
    assert "received no sources" in text and "generalization" in text
    assert sum(result.counts(["train", "validation", "test"]).values()) == 1


def test_all_variants_share_their_sources_split(ds_config: Config, source: SourceRecord) -> None:
    g = DatasetGenerator(ds_config)
    for sev in ("low", "medium"):
        g.generate_variant(
            source.source_id,
            "pause_insertion",
            severity=sev,
            placement={"mode": "explicit", "at_sec": 1.0},
        )
    split = build_splits(ds_config).assignments
    assert {s.source_id for s in g.samples()} == set(
        split
    )  # splitting is per source, not per variant


def test_split_errors(ds_config: Config, ds_root: Path) -> None:
    with pytest.raises(DatasetError, match="No sources"):
        build_splits(ds_config)
    register_many(ds_config, ds_root, 3)
    build_splits(ds_config)
    other = load_config(project_root=ds_root, env={"FLAWBENCH_SPLIT_RATIOS": '{"a":0.5,"b":0.5}'})
    with pytest.raises(DatasetError, match="no longer configured"):
        build_splits(other)
    assert build_splits(other, force=True).assignments


# ------------------------------------------------------------------------- export
@pytest.fixture
def with_samples(ds_config: Config, source: SourceRecord) -> DatasetGenerator:
    g = DatasetGenerator(ds_config)
    g.generate_variant(
        source.source_id,
        "pause_insertion",
        severity="low",
        placement={"mode": "explicit", "at_sec": 1.0},
    )
    return g


def test_export_includes_only_redistributable_and_never_overwrites(
    with_samples: DatasetGenerator, ds_config: Config, ds_root: Path
) -> None:
    res = export_samples(ds_config, "share-1")
    assert len(res.exported_samples) == 1 and not res.refused
    files = {
        p.relative_to(res.destination).as_posix() for p in res.destination.rglob("*") if p.is_file()
    }
    assert {"sources.jsonl", "manifest.jsonl", "EXPORT.json"} <= files
    assert any(f.startswith("generated/") for f in files) and any(
        f.startswith("reference/") for f in files
    )
    with pytest.raises(DatasetError, match="already exists"):
        export_samples(ds_config, "share-1")
    with pytest.raises(DatasetError, match="letters"):
        export_samples(ds_config, "../escape")
    with pytest.raises(DatasetError, match="Unknown sample"):
        export_samples(ds_config, "x", ["smp-nope"])


def test_export_refuses_sources_without_redistribution_rights(
    ds_root: Path, ds_config: Config
) -> None:
    reg = SourceRegistry(ds_config)
    rec = reg.register(
        write_fixture(ds_root, "p.wav", seed=3),
        **{**REGISTER_DEFAULTS, "redistribution": "unknown"},
    )
    DatasetGenerator(ds_config).generate_variant(
        rec.source_id,
        "pause_insertion",
        severity="low",
        placement={"mode": "explicit", "at_sec": 1.0},
    )
    with pytest.raises(DatasetError, match="no selected sample"):
        export_samples(ds_config, "blocked")
    assert not (ds_config.storage.output_dir / "exports" / "blocked").exists()


# --------------------------------------------------------------------------- CLI
@pytest.fixture
def cli_env(ds_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(ds_root))
    return ds_root


def test_cli_full_workflow(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ref = write_fixture(cli_env, "cli.wav", seed=1)
    (cli_env / "t.txt").write_text("synthetic")
    assert main(["init"]) == 0
    assert main(["dataset", "list-sources"]) == 0
    assert "No sources are registered" in capsys.readouterr().out
    assert main(["dataset", "validate"]) == 0
    assert "empty" in capsys.readouterr().out
    base = [
        "dataset",
        "register-reference",
        str(ref),
        "--transcript-file",
        str(cli_env / "t.txt"),
        "--source-type",
        "self_recorded",
        "--attribution",
        "test",
        "--license",
        "CC0-1.0",
        "--redistribution",
        "permitted",
    ]
    assert main(base) == 1  # --rights-confirmed missing
    assert "Rights must be confirmed" in capsys.readouterr().err
    assert main([*base, "--rights-confirmed"]) == 0
    sid = capsys.readouterr().out.split()[1]
    assert (
        main(
            [
                "dataset",
                "generate",
                "--source",
                sid,
                "--flaw",
                "pause_insertion",
                "--severity",
                "medium",
                "--at-sec",
                "2.0",
            ]
        )
        == 0
    )
    assert "output_interval=[2.0000, 3.0000)" in capsys.readouterr().out
    assert (
        main(
            [
                "dataset",
                "generate",
                "--source",
                sid,
                "--flaw",
                "energy_reduction",
                "--severity",
                "low",
                "--random-placement",
            ]
        )
        == 1
    )  # seed required
    assert "seed" in capsys.readouterr().err
    assert (
        main(
            [
                "dataset",
                "generate",
                "--source",
                sid,
                "--flaw",
                "energy_reduction",
                "--severity",
                "low",
                "--random-placement",
                "--seed",
                "3",
                "--param",
                "gain_db=-9",
            ]
        )
        == 0
    )
    report = cli_env / "report.json"
    assert main(["dataset", "validate", "--report", str(report)]) == 0
    data = json.loads(report.read_text())
    assert data["ok"] and data["counts"]["samples"] == 2 and data["counts"]["sources"] == 1
    assert main(["dataset", "split"]) == 0
    assert main(["dataset", "export", "--name", "cli-share"]) == 0
    assert main(["dataset", "validate"]) == 0
    assert main(["check"]) == 0  # Phase 1 command still healthy with a populated dataset


def test_cli_plan_and_flag_conflicts_and_failures(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ref = write_fixture(cli_env, "p.wav")
    assert (
        main(
            [
                "dataset",
                "register-reference",
                str(ref),
                "--transcript-text",
                "x",
                "--source-type",
                "public_domain",
                "--attribution",
                "a",
                "--license",
                "PD",
                "--rights-confirmed",
            ]
        )
        == 0
    )
    capsys.readouterr()
    plan = cli_env / "plan.yaml"
    plan.write_text(
        "plan_version: 1\nname: cli\nvariants:\n"
        "  - {source: '*', flaw: pause_insertion, severity: low, placement: {mode: explicit, at_sec: 1.0}}\n"
        "  - {source: '*', flaw: pause_insertion, severity: low, placement: {mode: explicit, at_sec: 999}}\n"
    )
    assert main(["dataset", "generate", "--plan", str(plan)]) == 1  # one variant fails
    cap = capsys.readouterr()
    assert '"failed": 1' in cap.out and "FAILED" in cap.err
    assert main(["dataset", "generate", "--plan", str(plan), "--flaw", "pause_insertion"]) == 1
    assert "cannot be combined" in capsys.readouterr().err
    assert main(["dataset", "generate"]) == 1
    assert main(["dataset", "generate", "--source", "x", "--flaw", "pause_insertion"]) == 1
    assert main(["dataset", "generate", "--bogus"]) == 2
    assert (
        main(
            [
                "dataset",
                "generate",
                "--source",
                "x",
                "--flaw",
                "pause_insertion",
                "--at-sec",
                "1",
                "--param",
                "nonsense",
            ]
        )
        == 2
    )


def test_cli_help_lists_the_documented_actions(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["dataset", "--help"]) == 0
    out = capsys.readouterr().out
    for action in ("register-reference", "list-sources", "generate", "validate", "split", "export"):
        assert action in out


def test_phase1_cli_commands_accept_config_option_in_both_positions(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_file = cli_env / "c.yaml"
    cfg_file.write_text("app:\n  name: Renamed\n")
    assert main(["--config", str(cfg_file), "config"]) == 0
    assert json.loads(capsys.readouterr().out)["app"]["name"] == "Renamed"
    assert main(["config", "--config", str(cfg_file)]) == 0
    assert json.loads(capsys.readouterr().out)["app"]["name"] == "Renamed"
    assert main(["provenance"]) == 0 and "libsndfile" in capsys.readouterr().out
