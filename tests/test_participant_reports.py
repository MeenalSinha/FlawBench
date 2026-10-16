from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from flawbench.config import Config
from flawbench.dataset.registry import SourceRegistry
from flawbench.dataset.schema import SourceRecord
from flawbench.exceptions import ReportError
from flawbench.explainability.reports import export_report, render_markdown
from flawbench.participant import ParticipantAnalyzer
from tests.test_participant_workflow import _pause_upload, _reference_samples, _wav_bytes


@pytest.fixture
def session(ds_config: Config, registry: SourceRegistry, source: SourceRecord):  # type: ignore[no-untyped-def]
    a = ParticipantAnalyzer(ds_config)
    return a.analyze_upload(source.source_id, _pause_upload(registry, source), "p.wav")


def _cfg(cfg: Config, tmp_path: Path, **kw: object) -> Config:
    return replace(cfg, explanation=replace(cfg.explanation, reports_dir=tmp_path / "rep", **kw))


def test_all_formats_written_and_match_saved_json(
    ds_config: Config, session, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    cfg = _cfg(ds_config, tmp_path)
    out = export_report(session.analysis, cfg.explanation)
    assert set(out) == {"json", "markdown", "html", "csv"}
    saved = json.loads(out["json"].read_text())
    assert saved == json.loads(json.dumps(session.analysis.to_dict()))
    md = out["markdown"].read_text()
    for card in saved["cards"]:
        assert card["prediction_id"] in md
        for line in card["evidence"]:
            assert line in md
        for claim in card["numeric_claims"]:
            assert claim["rendered"] in md
    rows = list(csv.DictReader(io.StringIO(out["csv"].read_text())))
    assert len(rows) == saved["prediction_count"] == len(saved["cards"])
    assert {r["prediction_id"] for r in rows} == {c["prediction_id"] for c in saved["cards"]}
    assert saved["participant"]["sha256"] in md and saved["reference"]["sha256"] in md
    assert "<pre>" in out["html"].read_text()


def test_report_contains_provenance_limitations_and_no_audio(
    ds_config: Config, session, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    out = export_report(session.analysis, _cfg(ds_config, tmp_path).explanation)
    md = out["markdown"].read_text()
    assert "detector_config_fingerprint" in md and "Limitations" in md
    assert "Benchmark context" in md
    assert max(p.stat().st_size for p in out.values()) < 200_000  # no embedded audio
    assert b"RIFF" not in out["json"].read_bytes()


def test_export_does_not_rerun_or_modify_analysis(
    ds_config: Config, session, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    before = json.dumps(session.analysis.to_dict(), sort_keys=True)
    export_report(session.analysis, _cfg(ds_config, tmp_path).explanation)
    assert json.dumps(session.analysis.to_dict(), sort_keys=True) == before


def test_html_escapes_content(ds_config: Config, session, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    session.analysis.warnings.append("<script>alert(1)</script>")
    out = export_report(session.analysis, _cfg(ds_config, tmp_path).explanation, formats=("html",))
    html = out["html"].read_text()
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_failed_and_no_candidate_reports_are_marked(
    ds_config: Config, registry: SourceRegistry, source: SourceRecord, tmp_path: Path
) -> None:
    a = ParticipantAnalyzer(ds_config)
    failed = a.analyze_upload(source.source_id, b"junk", "x.wav").analysis
    md = render_markdown(json.loads(json.dumps(failed.to_dict())))
    assert "FAILED" in md and "No candidates are reported" in md
    same = a.analyze_upload(
        source.source_id, _wav_bytes(_reference_samples(registry, source)), "s.wav"
    )
    md2 = render_markdown(json.loads(json.dumps(same.analysis.to_dict())))
    assert "does not show that the delivery is free of problems" in md2


def test_partial_report_is_marked(ds_config: Config, session) -> None:  # type: ignore[no-untyped-def]
    d = json.loads(json.dumps(session.analysis.to_dict()))
    d["status"], d["categories_failed"] = "partial", {"energy_reduction": "boom"}
    md = render_markdown(d)
    assert "PARTIAL ANALYSIS" in md and "FAILED energy_reduction" in md


def test_invalid_or_disabled_format_rejected(ds_config: Config, session, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cfg = _cfg(ds_config, tmp_path, export_formats=("json", "markdown"))
    with pytest.raises(ReportError):
        export_report(session.analysis, cfg.explanation, formats=("pdf",))
    with pytest.raises(ReportError):
        export_report(session.analysis, cfg.explanation, formats=("html",))


def test_unwritable_output_path_raises_report_error(
    ds_config: Config, session, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cfg = replace(
        ds_config, explanation=replace(ds_config.explanation, reports_dir=blocker / "sub")
    )
    with pytest.raises(ReportError):
        export_report(session.analysis, cfg.explanation)


def test_session_id_cannot_escape_report_directory(
    ds_config: Config, session, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    session.analysis.session_id = "../../evil"
    out = export_report(session.analysis, _cfg(ds_config, tmp_path).explanation)
    assert all(p.resolve().is_relative_to((tmp_path / "rep").resolve()) for p in out.values())


def test_reports_are_not_saved_automatically_by_default(
    ds_config: Config, session, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    assert ds_config.explanation.save_reports_automatically is False
    assert not (tmp_path / "rep").exists()
