"""Phase 7 end-to-end and security-boundary tests.

SYNTHETIC functional tests: audio is a generated harmonic tone, not speech. Every stage uses the
real implementation; nothing downstream is constructed by hand.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from flawbench.config import Config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.injectors import insert_pause
from flawbench.dataset.registry import SourceRegistry
from flawbench.dataset.schema import SourceRecord
from flawbench.dataset.splits import build_splits
from flawbench.dataset.validator import validate_dataset
from flawbench.detection.pipeline import DetectionPipeline
from flawbench.evaluation.runner import EvaluationRunner
from flawbench.explainability.reports import export_report
from flawbench.participant import ParticipantAnalyzer
from flawbench.paths import ProjectPaths
from tests.conftest import SYN_RATE


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path, *, skip: tuple[str, ...] = ()) -> dict[str, str]:
    return {
        str(p.relative_to(root)): _sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not any(s in p.parts for s in skip)
    }


def test_full_pipeline_isolation_and_integrity(
    ds_config: Config, ds_root: Path, source: SourceRecord
) -> None:
    paths = ProjectPaths(ds_config)
    ref_path = ds_root / "data" / "reference" / "fixture_a.wav"
    ref_before = _sha(ref_path)

    gen = DatasetGenerator(ds_config)
    flaws = {
        "pause_insertion": {"mode": "explicit", "at_sec": 2.0},
        "tempo_change": {"mode": "explicit", "start_sec": 1.0},
        "energy_reduction": {"mode": "explicit", "start_sec": 2.0},
    }
    samples = {
        f: gen.generate_variant(source.source_id, f, severity="medium", placement=pl).record
        for f, pl in flaws.items()
    }
    assert len({r.sample_id for r in samples.values()}) == 3

    # generated audio checksums match the manifest
    for rec in samples.values():
        gen_path = paths.generated_dir / rec.data["generated"]["path"]
        assert _sha(gen_path) == rec.data["generated"]["sha256"]

    report = validate_dataset(ds_config)
    assert not report.errors

    build_splits(ds_config, paths, force=True)

    pipe = DetectionPipeline(ds_config)
    first = {sid: pipe.detect_sample(sid) for sid in (r.sample_id for r in samples.values())}
    for rec in samples.values():
        res = first[rec.sample_id]
        assert res.source_id == source.source_id
        assert res.variant_checksum == rec.data["generated"]["sha256"]
        assert res.ref_checksum == source.sha256
        assert all(0 <= p.start_sample < p.end_sample <= res.total_samples for p in res.predictions)

    # Ground-truth isolation: corrupt every annotation, force re-detection, predictions identical.
    manifest = paths.manifest_file
    original_manifest = manifest.read_text(encoding="utf-8")
    mutated = []
    for line in original_manifest.splitlines():
        row = json.loads(line)
        if row.get("record_type") == "sample":
            row["annotation"]["output_interval"] = {
                "start_sample": 1, "end_sample": 2, "start_sec": 0.0, "end_sec": 0.0001,
            }  # fmt: skip
        mutated.append(json.dumps(row))
    manifest.write_text("\n".join(mutated) + "\n", encoding="utf-8")
    try:
        again = {sid: pipe.detect_sample(sid, force=True) for sid in first}
    finally:
        manifest.write_text(original_manifest, encoding="utf-8")
    for sid in first:
        assert [
            (p.prediction_id, p.start_sample, p.end_sample) for p in again[sid].predictions
        ] == [(p.prediction_id, p.start_sample, p.end_sample) for p in first[sid].predictions]

    # Evaluation joins annotations after inference; the three flaw types are found.
    runner = EvaluationRunner(ds_config, paths)
    ev, run_dir = runner.run(split="all", run_id="e2e", include_controls=True)
    assert ev.sample_count == 3
    assert ev.aggregate is not None and ev.aggregate.micro_recall == 1.0
    det_files = {sid: _sha(paths.detection_dir / f"{sid}_detection.json") for sid in first}
    ev_json = (run_dir / "evaluation_report.json").read_bytes()

    # Participant analysis on a registered variant, then export every format.
    analyzer = ParticipantAnalyzer(ds_config)
    session = analyzer.analyze_registered(samples["pause_insertion"].sample_id)
    written = export_report(
        session.analysis, ds_config.explanation, formats=("markdown", "html", "csv")
    )
    assert set(written) >= {"json", "markdown", "html", "csv"}
    assert (
        "annotation" not in written["json"].read_text(encoding="utf-8").lower().split('"cards"')[0]
    )

    # Read-only analysis left everything else untouched.
    assert _sha(ref_path) == ref_before
    assert {sid: _sha(paths.detection_dir / f"{sid}_detection.json") for sid in first} == det_files
    assert (run_dir / "evaluation_report.json").read_bytes() == ev_json


def test_upload_exports_neutralise_hostile_text(
    ds_config: Config, registry: SourceRegistry, source: SourceRecord
) -> None:
    ref = registry.load_audio(source).samples
    out = insert_pause(ref, at_sample=int(1.5 * SYN_RATE), silence_samples=SYN_RATE)
    buf = io.BytesIO()
    sf.write(buf, out.samples.T, SYN_RATE, format="WAV", subtype="PCM_16")
    analyzer = ParticipantAnalyzer(ds_config)
    evil_name = '=HYPERLINK("http://evil","x")<script>alert(1)</script>../../escape.wav'
    session = analyzer.analyze_upload(
        source.source_id,
        buf.getvalue(),
        evil_name,
        participant_transcript="<script>alert('x')</script> =cmd|' /C calc'!A0",
    )
    written = export_report(
        session.analysis, ds_config.explanation, formats=("markdown", "html", "csv")
    )
    reports_dir = ds_config.explanation.reports_dir.resolve()
    for path in written.values():
        assert path.resolve().parent == reports_dir  # no traversal out of the reports directory
    html_text = written["html"].read_text(encoding="utf-8")
    assert "<script>" not in html_text
    md_text = written["markdown"].read_text(encoding="utf-8")
    assert "<script>" not in md_text and "<b>" not in md_text  # no raw inline HTML from filenames
    assert len(written["csv"].read_text(encoding="utf-8").splitlines()) >= 2  # real data rows
    for row in written["csv"].read_text(encoding="utf-8").splitlines()[1:]:
        for cell in row.split(","):
            assert not cell.startswith(("=", "+", "-@", "@")), cell
    assert not any(p.name == "escape.wav" for p in Path(reports_dir.parent).rglob("*"))


def test_repeated_registered_analysis_does_not_duplicate_samples_or_overwrite(
    ds_config: Config, source: SourceRecord
) -> None:
    sample = DatasetGenerator(ds_config).generate_variant(
        source.source_id, "pause_insertion", severity="medium",
        placement={"mode": "explicit", "at_sec": 2.0},
    ).record  # fmt: skip
    analyzer = ParticipantAnalyzer(ds_config)
    s1 = analyzer.analyze_registered(sample.sample_id)
    s2 = analyzer.analyze_registered(sample.sample_id)
    paths = ProjectPaths(ds_config)
    rows = [json.loads(line) for line in paths.manifest_file.read_text().splitlines() if line]
    assert sum(1 for r in rows if r.get("sample_id") == sample.sample_id) == 1
    assert s1.analysis.session_id != s2.analysis.session_id  # each run is a new, unique session
    p1 = export_report(s1.analysis, ds_config.explanation, formats=("markdown",))["json"]
    p2 = export_report(s2.analysis, ds_config.explanation, formats=("markdown",))["json"]
    assert p1 != p2 and p1.exists() and p2.exists()


_ = (np, Any)
