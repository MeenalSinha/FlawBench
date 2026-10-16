"""Streamlit AppTest integration for the participant workflow (no browser, no external API)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from flawbench.config import load_config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.registry import SourceRegistry
from tests.conftest import REGISTER_DEFAULTS, write_fixture

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FLAWBENCH_PROJECT_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def populated(env_root: Path) -> Path:
    (env_root / "data" / "reference").mkdir(parents=True)
    cfg = load_config()
    path = write_fixture(env_root, "fix.wav", seed=1)
    src = SourceRegistry(cfg).register(path, **REGISTER_DEFAULTS)
    DatasetGenerator(cfg).generate_variant(
        src.source_id, "pause_insertion", severity="medium",
        placement={"mode": "explicit", "at_sec": 1.5},
    )  # fmt: skip
    return env_root


def _go(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    return at


def _all_text(at: AppTest) -> str:
    elements = [*at.markdown, *at.info, *at.error, *at.warning, *at.success]
    parts = [e.value for e in elements]
    return "\n".join(parts + [c.value for c in at.caption])


def test_overview_status_comes_from_real_checks(env_root: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    body = _all_text(at)
    assert "Detection engine" in body and "Evaluation reports" in body
    assert "No saved evaluation runs" in body and "None run in this session" in body
    assert "All systems operational" not in body


def test_analysis_pages_show_empty_state_before_any_run(env_root: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    for page in ("timeline", "evidence", "guidance"):
        _go(at, page)
        assert any("No analysis has been run" in i.value for i in at.info), page


def test_analyze_page_without_references_explains_why(env_root: Path) -> None:
    at = _go(AppTest.from_file(APP, default_timeout=60).run(), "analyze")
    assert any("No reference recordings are registered" in i.value for i in at.info)


def test_registered_variant_end_to_end_through_ui(populated: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    _go(at, "analyze")
    assert not at.error
    at.button(key="fb_run").click().run()
    assert not at.exception
    text = _all_text(at)
    assert "Analysis summary" in "\n".join(s.value for s in at.subheader)
    assert "Candidates" in {m.label for m in at.metric}
    for label_field in ("output_interval", "effective_parameters", "silence_samples", "severity"):
        assert label_field not in text

    _go(at, "timeline")
    assert at.selectbox(key="flawbench_selected_prediction").options
    assert len(at.get("audio")) >= 1  # a participant excerpt is playable
    assert any("Observation." in m.value for m in at.markdown)
    assert any("not a probability" in c.value for c in at.caption)

    _go(at, "evidence")
    assert any("Measured values" in s.value for s in at.subheader)
    _go(at, "guidance")
    assert any("Suggested review." in m.value for m in at.markdown)
    assert any("Benchmark results are unavailable" in i.value for i in at.info)


def test_export_button_writes_files_only_on_click(populated: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    _go(at, "analyze")
    at.button(key="fb_run").click().run()
    reports = populated / "outputs" / "participant_reports"
    assert not reports.exists() or not list(reports.glob("*"))
    _go(at, "guidance")
    at.button(key="flawbench_export").click().run()
    assert not at.exception
    assert len(list(reports.glob("*.json"))) == 1 and list(reports.glob("*.md"))


def test_evaluation_lab_does_not_run_a_benchmark_on_open(populated: Path) -> None:
    at = _go(AppTest.from_file(APP, default_timeout=60).run(), "evaluation")
    assert any("separate from Participant Analysis" in i.value for i in at.info)
    assert not (populated / "outputs" / "evaluations").exists() or not any(
        (populated / "outputs" / "evaluations").iterdir()
    )
    options = at.selectbox[0].options
    assert set(options) == {"test", "validation", "train", "all"}


def test_default_page_is_configurable(env_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLAWBENCH_EXPLAIN_DEFAULT_PAGE", "analyze")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "analyze"


def test_uploaded_participant_recording_through_ui(populated: Path) -> None:
    import io

    import numpy as np
    import soundfile as sf

    from flawbench.dataset.injectors import insert_pause

    cfg = load_config()
    ref = SourceRegistry(cfg).load_audio(SourceRegistry(cfg).all()[0]).samples
    var = insert_pause(ref, at_sample=24000, silence_samples=16000).samples
    buf = io.BytesIO()
    sf.write(buf, var.T.astype(np.float32), 16000, format="WAV")

    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "analyze")
    at.radio(key="fb_mode").set_value("uploaded_participant").run()
    at.file_uploader(key="fb_upload").set_value(("me.wav", buf.getvalue(), "audio/wav")).run()
    assert not at.exception
    assert {"Format", "Sample rate", "Channels", "Duration"} <= {m.label for m in at.metric}
    before = sorted(p.name for p in (populated / "data").rglob("*") if p.is_file())
    at.button(key="fb_run").click().run()
    assert not at.exception
    assert "Candidates" in {m.label for m in at.metric}
    after = sorted(p.name for p in (populated / "data").rglob("*") if p.is_file())
    assert [n for n in after if n not in before and "me" in n.split(".")[0]] == []
    assert not any(n == "me.wav" for n in after)
