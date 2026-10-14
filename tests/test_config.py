from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from flawbench.config import ENV_VARIABLES, defaults, load_config
from flawbench.exceptions import ConfigurationError

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_yaml(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_defaults_resolve_against_project_root(project_root: Path) -> None:
    cfg = load_config(project_root=project_root, env={})
    assert cfg.storage.reference_dir == (project_root / "data/reference").resolve()
    assert cfg.storage.output_dir == (project_root / "outputs").resolve()
    assert cfg.app.environment == "development"
    assert cfg.config_file is None


def test_working_directory_does_not_affect_relative_paths(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = load_config(project_root=project_root, env={})
    assert cfg.storage.generated_dir.is_relative_to(project_root.resolve())


def test_example_config_file_loads_and_matches_defaults(project_root: Path) -> None:
    example = REPO_ROOT / "configs" / "config.example.yaml"
    cfg = load_config(example, project_root=project_root, env={})
    baseline = load_config(project_root=project_root, env={})
    assert cfg.to_dict() | {"config_file": None} == baseline.to_dict() | {"config_file": None}
    assert cfg.config_file == example.resolve()


def test_default_config_file_in_project_is_discovered(project_root: Path) -> None:
    (project_root / "configs").mkdir()
    write_yaml(project_root / "configs/config.yaml", {"app": {"log_level": "debug"}})
    assert load_config(project_root=project_root, env={}).app.log_level == "DEBUG"


def test_precedence_overrides_over_env_over_file_over_defaults(project_root: Path) -> None:
    file = write_yaml(
        project_root / "c.yaml",
        {"audio": {"max_upload_mb": 10, "max_channels": 4}, "app": {"name": "FromFile"}},
    )
    env = {"FLAWBENCH_MAX_UPLOAD_MB": "20", "FLAWBENCH_MAX_CHANNELS": "3"}
    cfg = load_config(
        file, project_root=project_root, env=env, overrides={"audio": {"max_upload_mb": 30}}
    )
    assert cfg.audio.max_upload_mb == 30  # override beats env and file
    assert cfg.audio.max_channels == 3  # env beats file
    assert cfg.app.name == "FromFile"  # file beats default
    assert cfg.audio.min_sample_rate_hz == 8000  # default survives


def test_env_var_selects_config_file(project_root: Path) -> None:
    file = write_yaml(project_root / "alt.yaml", {"ui": {"layout": "wide"}})
    cfg = load_config(project_root=project_root, env={"FLAWBENCH_CONFIG": str(file)})
    assert cfg.ui.layout == "wide"


def test_dotenv_file_is_used_but_real_env_wins(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project_root / ".env").write_text(
        "# comment\nFLAWBENCH_LOG_LEVEL=warning\nFLAWBENCH_APP_NAME='From Dotenv'\n"
    )
    monkeypatch.delenv("FLAWBENCH_APP_NAME", raising=False)
    monkeypatch.setenv("FLAWBENCH_LOG_LEVEL", "ERROR")
    cfg = load_config(project_root=project_root)
    assert cfg.app.log_level == "ERROR"
    assert cfg.app.name == "From Dotenv"


def test_malformed_dotenv_is_reported(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (project_root / ".env").write_text("NOT_A_PAIR\n")
    with pytest.raises(ConfigurationError, match=r"Malformed line 1"):
        load_config(project_root=project_root)


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({"FLAWBENCH_LOG_LEVEL": "LOUD"}, "log_level"),
        ({"FLAWBENCH_ENV": "staging"}, "environment"),
        ({"FLAWBENCH_MAX_UPLOAD_MB": "-5"}, "max_upload_mb"),
        ({"FLAWBENCH_MAX_UPLOAD_MB": "ten"}, "max_upload_mb"),
        ({"FLAWBENCH_MAX_CHANNELS": "0"}, "max_channels"),
        ({"FLAWBENCH_LAYOUT": "tiny"}, "layout"),
        ({"FLAWBENCH_SHOW_ROADMAP": "maybe"}, "show_roadmap"),
        ({"FLAWBENCH_AUDIO_EXTENSIONS": ".xyz"}, "extension"),
        ({"FLAWBENCH_AUDIO_EXTENSIONS": " , "}, "allowed_extensions"),
        ({"FLAWBENCH_REFERENCE_DIR": " "}, "reference_dir"),
    ],
)
def test_invalid_environment_values_raise_helpful_errors(
    project_root: Path, env: dict[str, str], fragment: str
) -> None:
    with pytest.raises(ConfigurationError, match=fragment):
        load_config(project_root=project_root, env=env)


def test_extensions_are_normalized(project_root: Path) -> None:
    cfg = load_config(project_root=project_root, env={"FLAWBENCH_AUDIO_EXTENSIONS": "WAV, .Flac"})
    assert cfg.audio.allowed_extensions == (".wav", ".flac")
    assert cfg.audio.extension_names == ["wav", "flac"]
    assert cfg.audio.allowed_formats >= {"WAV", "FLAC"}


def test_bool_parsing_accepts_common_spellings(project_root: Path) -> None:
    cfg = load_config(
        project_root=project_root,
        env={"FLAWBENCH_SHOW_ROADMAP": "off", "FLAWBENCH_DETAILED_LOGGING": "YES"},
    )
    assert cfg.ui.show_roadmap is False
    assert cfg.diagnostics.detailed_logging is True


def test_missing_explicit_config_file(project_root: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(project_root / "nope.yaml", project_root=project_root, env={})


@pytest.mark.parametrize("content", ["a: [unclosed", "- just\n- a list\n"])
def test_malformed_config_files(project_root: Path, content: str) -> None:
    path = project_root / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ConfigurationError):
        load_config(path, project_root=project_root, env={})


def test_empty_config_file_means_defaults(project_root: Path) -> None:
    path = project_root / "empty.yaml"
    path.write_text("")
    assert load_config(path, project_root=project_root, env={}).app.name == "FlawBench"


def test_unknown_section_and_key_are_rejected(project_root: Path) -> None:
    for data, fragment in (({"bogus": {}}, "bogus"), ({"app": {"nmae": "x"}}, "app.nmae")):
        path = write_yaml(project_root / "c.yaml", data)
        with pytest.raises(ConfigurationError, match=fragment):
            load_config(path, project_root=project_root, env={})


def test_storage_path_that_is_a_file_is_rejected(project_root: Path) -> None:
    (project_root / "afile").write_text("x")
    with pytest.raises(ConfigurationError, match="not a directory"):
        load_config(project_root=project_root, env={"FLAWBENCH_OUTPUT_DIR": "afile"})


def test_duplicate_storage_directories_are_rejected(project_root: Path) -> None:
    with pytest.raises(ConfigurationError, match="different directories"):
        load_config(project_root=project_root, env={"FLAWBENCH_GENERATED_DIR": "data/reference"})


def test_production_forbids_exposing_error_details(project_root: Path) -> None:
    with pytest.raises(ConfigurationError, match="production"):
        load_config(
            project_root=project_root,
            env={"FLAWBENCH_ENV": "production", "FLAWBENCH_EXPOSE_ERROR_DETAILS": "true"},
        )


def test_missing_project_root(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="project root"):
        load_config(project_root=tmp_path / "missing", env={})


def test_absolute_storage_path_is_kept(project_root: Path, tmp_path: Path) -> None:
    target = tmp_path / "abs_out"
    cfg = load_config(project_root=project_root, env={"FLAWBENCH_OUTPUT_DIR": str(target)})
    assert cfg.storage.output_dir == target.resolve()


def test_every_setting_has_exactly_one_env_var_and_default() -> None:
    assert len(set(ENV_VARIABLES)) == len(ENV_VARIABLES)
    flat = [(s, k) for s, body in defaults().items() for k in body]
    assert len(flat) == len(ENV_VARIABLES)


def test_example_env_file_documents_every_variable() -> None:
    text = (REPO_ROOT / ".env.example").read_text()
    missing = [name for name in ENV_VARIABLES if name not in text]
    assert not missing


def test_to_dict_is_json_serializable(config) -> None:  # type: ignore[no-untyped-def]
    import json

    json.dumps(config.to_dict())
