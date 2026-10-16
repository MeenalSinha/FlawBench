from __future__ import annotations

import json
from pathlib import Path

import pytest

from flawbench.__main__ import main
from flawbench.config import Config, load_config
from flawbench.paths import ProjectPaths
from flawbench.provenance import build_provenance, config_fingerprint
from flawbench.roadmap import IMPLEMENTED, IN_PROGRESS, PHASES, PLANNED
from flawbench.status import collect_status


def test_fingerprint_is_stable_and_sensitive_to_changes(project_root: Path, config: Config) -> None:
    again = load_config(project_root=project_root, env={})
    changed = load_config(project_root=project_root, env={"FLAWBENCH_MAX_UPLOAD_MB": "7"})
    assert config_fingerprint(config) == config_fingerprint(again)
    assert config_fingerprint(config) != config_fingerprint(changed)


def test_provenance_records_versions_seed_and_config(config: Config) -> None:
    record = build_provenance(config, seed=1234)
    assert record["random_seed"] == 1234
    assert record["config_sha256"] == config_fingerprint(config)
    assert record["packages"]["numpy"] != "not installed"
    assert record["libsndfile"]
    json.dumps(record)


def test_status_reflects_real_directory_state(config: Config) -> None:
    paths = ProjectPaths(config)
    before = {i.label: i for i in collect_status(config, paths)}
    assert not before["Generated directory (output)"].ok
    paths.ensure_output_directories()
    config.storage.reference_dir.mkdir(parents=True)
    after = {i.label: i for i in collect_status(config, paths)}
    assert all(i.ok for i in after.values())
    assert after["Audio loader"].message.startswith("soundfile")


def test_status_hides_paths_unless_requested(config: Config) -> None:
    paths = ProjectPaths(config)
    hidden = " ".join(i.message for i in collect_status(config, paths))
    shown = " ".join(i.message for i in collect_status(config, paths, expose_details=True))
    assert str(config.project_root) not in hidden
    assert str(config.project_root) in shown


def test_roadmap_statuses_match_the_verified_state() -> None:
    # Phases 1-8 are implemented and verified
    assert [p.number for p in PHASES if p.status == IMPLEMENTED] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [p.number for p in PHASES if p.status == PLANNED] == []
    assert [p.number for p in PHASES if p.status == IN_PROGRESS] == []


def test_cli_config_check_init_and_errors(
    project_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(project_root))
    assert main(["config"]) == 0
    assert json.loads(capsys.readouterr().out)["project_root"] == str(project_root.resolve())
    assert main(["check"]) == 1  # nothing exists yet: honest failure
    assert main(["init"]) == 0
    (project_root / "data/reference").mkdir(parents=True, exist_ok=True)
    capsys.readouterr()
    assert main(["check"]) == 0
    monkeypatch.setenv("FLAWBENCH_LOG_LEVEL", "LOUD")
    assert main(["check"]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_cli_release_check(
    project_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(project_root))
    assert main(["init"]) == 0
    (project_root / "data/reference").mkdir(parents=True, exist_ok=True)
    (project_root / "app.py").write_text("import streamlit as st\nst.title('FlawBench')\n")
    capsys.readouterr()
    res = main(["release-check", "--json"])
    out = capsys.readouterr().out
    assert res == 0
    data = json.loads(out)
    assert data["is_ready"] is True
    assert data["version"] == "0.1.0"
    assert data["passed_count"] >= 20
