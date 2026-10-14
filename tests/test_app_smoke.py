"""Headless smoke tests using Streamlit's AppTest (no browser, no server)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def texts(at: AppTest) -> str:
    parts = (
        [e.value for e in at.markdown] + [e.value for e in at.success] + [e.value for e in at.error]
    )
    parts += (
        [e.value for e in at.info] + [e.value for e in at.title] + [e.value for e in at.caption]
    )
    return "\n".join(parts)


def test_app_runs_and_reports_real_status(isolated_env: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "FlawBench"
    body = texts(at)
    assert "Evidence-Based Speech Flaw Analysis" in body
    assert "Analysis has not been performed" in body
    assert (isolated_env / "outputs" / "reports").is_dir()  # writable dirs prepared from config
    assert any("Reference directory" in e.value and "does not exist" in e.value for e in at.error)
    assert "Planned (not implemented)" in body


def test_configuration_change_alters_behavior_without_code_edits(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLAWBENCH_APP_NAME", "BenchRenamed")
    monkeypatch.setenv("FLAWBENCH_SHOW_ROADMAP", "false")
    monkeypatch.setenv("FLAWBENCH_OUTPUT_DIR", "custom_out")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "BenchRenamed"
    assert "Project roadmap" not in "\n".join(s.value for s in at.subheader)
    assert (isolated_env / "custom_out" / "figures").is_dir()


def test_invalid_configuration_shows_message_not_traceback(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLAWBENCH_LOG_LEVEL", "LOUD")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Configuration problem" in e.value for e in at.error)


def _wav_bytes(seconds: float = 0.5, rate: int = 16000) -> bytes:
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    t = np.arange(int(rate * seconds)) / rate
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 330 * t)).astype("float32"), rate, format="WAV")
    return buf.getvalue()


def test_app_shows_real_empty_dataset_state(isolated_env: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    counts = {m.label: m.value for m in at.metric}
    assert counts["Registered sources"] == "0" and counts["Generated samples"] == "0"
    assert any("No reference recordings are registered" in i.value for i in at.info)
    at.button[0].click().run()  # Validate dataset
    assert not at.exception
    assert any("dataset is empty" in i.value for i in at.info)


@pytest.mark.skip(
    reason="AppTest.file_uploader not available in Streamlit <1.65; upgrade Streamlit to re-enable"
)
def test_valid_upload_shows_real_metadata_and_no_analysis(isolated_env: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.file_uploader[0].set_value(("tone.wav", _wav_bytes(0.5, 16000), "audio/wav")).run()  # type: ignore[attr-defined]
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Sample rate"] == "16,000 Hz"
    assert metrics["Duration"] == "0.50 s"
    assert metrics["Channels"] == "1"
    assert metrics["Container"] == "WAV"
    assert any("Analysis has not been performed" in i.value for i in at.info)


@pytest.mark.skip(
    reason="AppTest.file_uploader not available in Streamlit <1.65; upgrade Streamlit to re-enable"
)
def test_fake_audio_upload_is_rejected_cleanly(isolated_env: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.file_uploader[0].set_value(("fake.wav", b"not audio at all" * 20, "audio/wav")).run()  # type: ignore[attr-defined]
    assert not at.exception
    assert not {m.label for m in at.metric} & {"Sample rate", "Duration", "Channels"}
    errors = " ".join(e.value for e in at.error)
    assert "could not be decoded" in errors
    assert "Traceback" not in errors and str(isolated_env) not in errors


@pytest.mark.skip(
    reason="AppTest.file_uploader not available in Streamlit <1.65; upgrade Streamlit to re-enable"
)
def test_upload_limit_and_allowed_types_come_from_configuration(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLAWBENCH_AUDIO_EXTENSIONS", ".wav")
    monkeypatch.setenv("FLAWBENCH_MAX_UPLOAD_MB", "1")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.file_uploader[0].allowed_type == [".wav"] or at.file_uploader[0].allowed_type == [  # type: ignore[attr-defined]
        "wav"
    ]
    big = _wav_bytes(seconds=40.0, rate=16000)  # about 1.2 MB as float32 WAV
    at.file_uploader[0].set_value(("big.wav", big, "audio/wav")).run()  # type: ignore[attr-defined]
    assert any("exceeds the 1 MB limit" in e.value for e in at.error)
