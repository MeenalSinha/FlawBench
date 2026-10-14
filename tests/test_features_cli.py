"""Tests for Phase 3 features CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from flawbench.__main__ import main
from flawbench.config import load_config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.registry import SourceRegistry
from flawbench.paths import ProjectPaths


def _write_wav(path: Path, seconds: float = 0.5, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(rate * seconds)) / rate
    y = (0.3 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
    sf.write(str(path), y, rate, format="WAV")


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, str]:
    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("FLAWBENCH_ALLOW_OVERWRITE", "true")
    config = load_config()
    paths = ProjectPaths(config)
    paths.ensure_output_directories()

    wav_path = paths.reference_dir / "cli_ref.wav"
    _write_wav(wav_path, seconds=0.6)
    reg = SourceRegistry(config, paths)
    source = reg.register(
        wav_path,
        transcript_text="CLI test speech recording",
        source_type="self_recorded",
        attribution="CLI Tester",
        license="MIT",
        rights_confirmed=True,
    )

    gen = DatasetGenerator(config, paths, reg)
    v_res = gen.generate_variant(
        source_id=source.source_id,
        flaw_type="pause_insertion",
        parameters={"duration_sec": 0.3},
        placement={"mode": "explicit", "at_sec": 0.2},
    )
    return tmp_path, source.source_id, v_res.record.sample_id


def test_cli_features_extract_and_list(cli_env: tuple[Path, str, str], capsys: pytest.CaptureFixture[str]) -> None:
    _, source_id, sample_id = cli_env

    # 1. Extract reference
    ret = main(["features", "extract", "--source", source_id])
    assert ret == 0
    captured = capsys.readouterr()
    assert f"ref-{source_id}" in captured.out

    # 2. Extract variant
    ret = main(["features", "extract", "--sample", sample_id])
    assert ret == 0
    captured = capsys.readouterr()
    assert f"var-{sample_id}" in captured.out

    # 3. List
    ret = main(["features", "list"])
    assert ret == 0
    captured = capsys.readouterr()
    assert f"ref-{source_id}" in captured.out
    assert f"var-{sample_id}" in captured.out

    # 4. Validate
    ret = main(["features", "validate"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Feature store valid: True" in captured.out


def test_cli_features_compare_and_export(cli_env: tuple[Path, str, str], capsys: pytest.CaptureFixture[str]) -> None:
    tmp_path, source_id, sample_id = cli_env

    # Compare sample
    ret = main(["features", "compare", "--sample", sample_id])
    assert ret == 0
    captured = capsys.readouterr()
    assert "DTW cost:" in captured.out
    assert "Path length:" in captured.out

    # Export to file
    out_file = tmp_path / "feat_summaries.json"
    ret = main(["features", "export", "--output", str(out_file)])
    assert ret == 0
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data) >= 2
