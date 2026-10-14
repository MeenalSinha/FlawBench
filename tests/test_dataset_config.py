from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flawbench.config import DEFAULT_PARAMETER_BOUNDS, DEFAULT_SEVERITY_PROFILES, load_config
from flawbench.dataset.params import resolve_parameters
from flawbench.exceptions import ConfigurationError, InjectionError


def cfg(root: Path, **env: str):  # type: ignore[no-untyped-def]
    return load_config(project_root=root, env=env)


def test_dataset_defaults_validate_and_paths_follow_metadata_dir(project_root: Path) -> None:
    from flawbench.paths import ProjectPaths

    c = cfg(project_root, FLAWBENCH_METADATA_DIR="meta")
    paths = ProjectPaths(c)
    assert paths.manifest_file == (project_root / "meta" / "manifest.jsonl").resolve()
    assert paths.sources_file.parent == paths.metadata_dir
    assert c.dataset.severity_profiles == DEFAULT_SEVERITY_PROFILES


def test_severity_profiles_can_be_overridden_from_environment_json(project_root: Path) -> None:
    profiles = copy.deepcopy(DEFAULT_SEVERITY_PROFILES)
    profiles["pause_insertion"] = {"tiny": {"duration_sec": 0.2}}
    c = cfg(project_root, FLAWBENCH_SEVERITY_PROFILES=json.dumps(profiles))
    assert list(c.dataset.severity_profiles["pause_insertion"]) == ["tiny"]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (
            lambda p: p["energy_reduction"]["low"].update(gain_db=3.0),
            "outside the configured bounds",
        ),
        (lambda p: p["pause_insertion"]["low"].update(duration_sec=999), "bounds"),
        (lambda p: p["tempo_change"]["low"].pop("rate"), "must be exactly"),
        (lambda p: p.update(bogus={"x": {}}), "Unknown flaw type"),
        (lambda p: p["pause_insertion"].update(custom={"duration_sec": 1}), "reserved"),
        (
            lambda p: p["pause_insertion"].update(**{"bad name": {"duration_sec": 1}}),
            "Invalid severity",
        ),
    ],
)
def test_invalid_profiles_are_rejected_at_load_time(
    project_root: Path, mutate, fragment: str
) -> None:  # type: ignore[no-untyped-def]
    profiles = copy.deepcopy(DEFAULT_SEVERITY_PROFILES)
    mutate(profiles)
    with pytest.raises(ConfigurationError, match=fragment):
        cfg(project_root, FLAWBENCH_SEVERITY_PROFILES=json.dumps(profiles))


def test_invalid_bounds_rejected(project_root: Path) -> None:
    bounds = copy.deepcopy(DEFAULT_PARAMETER_BOUNDS)
    bounds["energy_reduction"]["gain_db"]["max"] = 1.0
    with pytest.raises(ConfigurationError, match="negative"):
        cfg(project_root, FLAWBENCH_PARAMETER_BOUNDS=json.dumps(bounds))
    del bounds["tempo_change"]
    with pytest.raises(ConfigurationError, match="missing"):
        cfg(project_root, FLAWBENCH_PARAMETER_BOUNDS=json.dumps(bounds))


@pytest.mark.parametrize(
    "env, fragment",
    [
        ({"FLAWBENCH_SPLIT_RATIOS": '{"train":0.5,"test":0.2}'}, "summing to 1"),
        ({"FLAWBENCH_SPLIT_RATIOS": "not json"}, "JSON"),
        ({"FLAWBENCH_SOURCES_FILENAME": "a/b.jsonl"}, "plain file name"),
        ({"FLAWBENCH_MANIFEST_FILENAME": "sources.jsonl"}, "different"),
        ({"FLAWBENCH_OUTPUT_SUBTYPE": "MP3"}, "output_subtype"),
        ({"FLAWBENCH_DATASET_ID": "bad id!"}, "dataset_id"),
        ({"FLAWBENCH_MAX_OUTPUT_SECONDS": "0"}, "max_output_seconds"),
        ({"FLAWBENCH_MAX_VARIANTS_PER_RUN": "0"}, "max_variants_per_run"),
        ({"FLAWBENCH_PLACEMENT_MARGIN_SEC": "-1"}, "placement_margin_sec"),
        ({"FLAWBENCH_SPLIT_SEED": "x"}, "split_seed"),
    ],
)
def test_invalid_dataset_settings(project_root: Path, env: dict[str, str], fragment: str) -> None:
    with pytest.raises(ConfigurationError, match=fragment):
        load_config(project_root=project_root, env=env)


def test_resolve_parameters_labels_overrides_as_custom() -> None:
    d = cfg(Path.cwd()).dataset
    args = (d.severity_profiles, d.parameter_bounds)
    assert resolve_parameters("pause_insertion", "medium", None, *args) == (
        "medium",
        None,
        {"duration_sec": 1.0},
    )
    label, base, params = resolve_parameters("tempo_change", "low", {"segment_sec": 1.0}, *args)
    assert (label, base, params) == ("custom", "low", {"rate": 0.85, "segment_sec": 1.0})
    with pytest.raises(InjectionError, match="severity profile or explicit"):
        resolve_parameters("pause_insertion", None, None, *args)
    with pytest.raises(InjectionError, match="Unknown severity"):
        resolve_parameters("pause_insertion", "extreme", None, *args)
    with pytest.raises(InjectionError, match="bounds"):
        resolve_parameters("pause_insertion", None, {"duration_sec": 0.0001}, *args)
    with pytest.raises(InjectionError, match="finite number"):
        resolve_parameters("pause_insertion", None, {"duration_sec": float("nan")}, *args)
