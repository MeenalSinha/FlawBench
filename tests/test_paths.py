from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from flawbench.config import Config, load_config
from flawbench.exceptions import StorageError
from flawbench.paths import ProjectPaths, ensure_writable_directory, resolve_within


def test_derived_directories_follow_configuration(config: Config) -> None:
    paths = ProjectPaths(config)
    assert paths.reports_dir == paths.output_dir / "reports"
    assert paths.figures_dir == paths.output_dir / "figures"
    assert paths.reference_dir == config.storage.reference_dir
    assert paths.detection_dir == config.detection.detection_dir
    assert paths.evaluations_dir == config.evaluation.evaluations_dir


def test_ensure_output_directories_creates_nested_writable_dirs_only(config: Config) -> None:
    paths = ProjectPaths(config)
    created = paths.ensure_output_directories()
    assert all(p.is_dir() for p in created)
    assert {p.name for p in created} == {
        "generated",
        "metadata",
        "reports",
        "figures",
        "features",
        "detections",
        "evaluations",
    }
    assert not paths.reference_dir.exists()  # input directory is never created


def test_nested_relative_configuration(project_root: Path) -> None:
    cfg = load_config(project_root=project_root, env={"FLAWBENCH_OUTPUT_DIR": "a/b/c/out"})
    paths = ProjectPaths(cfg)
    paths.ensure_output_directories()
    assert (project_root / "a/b/c/out/reports").is_dir()


def test_input_directory_check_reports_missing_and_non_directory(config: Config) -> None:
    paths = ProjectPaths(config)
    with pytest.raises(StorageError, match="does not exist"):
        paths.check_input_directory()
    paths.reference_dir.parent.mkdir(parents=True)
    paths.reference_dir.write_text("file")
    with pytest.raises(StorageError, match="not a directory"):
        paths.check_input_directory()


def test_input_directory_check_succeeds_when_present(config: Config) -> None:
    config.storage.reference_dir.mkdir(parents=True)
    assert ProjectPaths(config).check_input_directory() == config.storage.reference_dir


def test_ensure_writable_directory_rejects_file(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(StorageError):
        ensure_writable_directory(f)
    with pytest.raises(StorageError):
        ensure_writable_directory(f / "child")


@pytest.mark.skipif(
    sys.platform.startswith("win") or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX permission semantics; root bypasses them",
)
def test_unwritable_directory_is_reported(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(StorageError, match="not writable"):
            ensure_writable_directory(locked)
    finally:
        locked.chmod(0o700)


def test_inspect_reflects_real_state_without_creating(config: Config) -> None:
    paths = ProjectPaths(config)
    before = {s.name: s for s in paths.inspect()}
    assert not any(s.accessible for s in before.values())
    assert not paths.generated_dir.exists()
    paths.ensure_output_directories()
    config.storage.reference_dir.mkdir(parents=True)
    after = {s.name: s for s in paths.inspect()}
    assert all(s.accessible for s in after.values())
    assert after["reference"].role == "input"
    assert after["generated"].role == "output"


def test_resolve_within_blocks_traversal(tmp_path: Path) -> None:
    assert resolve_within(tmp_path, "a", "b.wav") == (tmp_path / "a" / "b.wav").resolve()
    with pytest.raises(StorageError):
        resolve_within(tmp_path, "..", "escape.wav")
    with pytest.raises(StorageError):
        resolve_within(tmp_path, "/etc/passwd")


def test_no_manual_separator_assumptions(config: Config) -> None:
    for status in ProjectPaths(config).inspect():
        assert isinstance(status.path, Path)
