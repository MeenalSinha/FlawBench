"""Dataset auditing against the real files and records.

``validate_dataset`` inspects the source registry, the manifest, the generated audio on disk, and
(in deep mode) re-derives the ground truth from the actual samples: the audio outside the annotated
interval must equal the reference, a pause interval must be digital silence, an attenuated
interval must equal the reference scaled by the recorded multiplier, and so on.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from flawbench.audio.loader import load_audio
from flawbench.config import Config
from flawbench.dataset.generator import GENERATED_EXTENSION, generated_audio_settings
from flawbench.dataset.io import read_jsonl_lenient, sha256_file
from flawbench.dataset.params import FLAW_ENERGY, FLAW_PAUSE, FLAW_TEMPO, PARAM_SPECS
from flawbench.dataset.registry import ensure_relative_posix
from flawbench.dataset.schema import (
    STATUS_COMMITTED,
    TIMELINE_CONVENTION,
    SampleRecord,
    SourceRecord,
)
from flawbench.exceptions import AudioInputError, DatasetError
from flawbench.paths import ProjectPaths, resolve_within

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    subject: str | None = None


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    deep: bool = True

    def add(self, severity: str, code: str, message: str, subject: str | None = None) -> None:
        self.findings.append(Finding(severity, code, message, subject))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def ok(self) -> bool:
        """True only when there are no errors (warnings do not invalidate ground truth)."""
        return not self.errors

    @property
    def is_empty(self) -> bool:
        return self.counts.get("sources", 0) == 0 and self.counts.get("samples", 0) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "empty_dataset": self.is_empty,
            "deep": self.deep,
            "counts": dict(self.counts),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "findings": [vars(f) for f in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _atol(subtype: str) -> float:
    return {"FLOAT": 0.0, "PCM_16": 1.0 / 2**15, "PCM_24": 1.0 / 2**23}.get(subtype, 1.0 / 2**15)


def _same(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=atol))


def _interval_ok(iv: dict[str, Any], frames: int, sr: int) -> str | None:
    try:
        s, e = int(iv["start_sample"]), int(iv["end_sample"])
        ss, es = float(iv["start_sec"]), float(iv["end_sec"])
    except (KeyError, TypeError, ValueError):
        return "interval fields are missing or not numeric"
    if not (0 <= s < e <= frames):
        return f"interval [{s}, {e}) is not within [0, {frames}] samples"
    if abs(ss - s / sr) > 1e-9 or abs(es - e / sr) > 1e-9:
        return "interval seconds do not match sample indices"
    return None


class _Validator:
    def __init__(self, config: Config, paths: ProjectPaths, deep: bool) -> None:
        self.config = config
        self.paths = paths
        self.deep = deep
        self.report = ValidationReport(deep=deep)
        self.gen_settings = generated_audio_settings(config)

    def run(self) -> ValidationReport:
        sources = self._sources()
        samples = self._samples(sources)
        self._orphans(samples)
        self._splits(sources)
        self.report.counts.update({"sources": len(sources), "samples": len(samples)})
        if not sources and not samples:
            self.report.add(INFO, "empty_dataset", "No sources or samples are registered.")
        return self.report

    # ------------------------------------------------------------------ sources
    def _sources(self) -> dict[str, SourceRecord]:
        rep = self.report
        rows, problems = read_jsonl_lenient(self.paths.sources_file)
        for p in problems:
            rep.add(ERROR, "sources_malformed", p, "sources")
        out: dict[str, SourceRecord] = {}
        sha_seen: dict[str, str] = {}
        for line, raw in rows:
            subject = f"sources:{line}"
            try:
                rec = SourceRecord.from_dict(raw)
            except DatasetError as exc:
                rep.add(ERROR, "source_invalid", exc.user_message, subject)
                continue
            if rec.source_id in out:
                rep.add(
                    ERROR, "source_duplicate_id", f"Duplicate source id {rec.source_id}.", subject
                )
                continue
            if rec.sha256 in sha_seen:
                rep.add(
                    WARNING,
                    "source_duplicate_audio",
                    f"{rec.source_id} has the same audio as {sha_seen[rec.sha256]}.",
                    subject,
                )
            sha_seen[rec.sha256] = rec.source_id
            out[rec.source_id] = rec
            self._check_source(rec)
        return out

    def _check_source(self, rec: SourceRecord) -> None:
        rep, sid = self.report, rec.source_id
        d = rec.data
        if not str(d["transcript_text"]).strip():
            rep.add(ERROR, "source_no_transcript", "Transcript is empty.", sid)
        if not str(d["license"]).strip() or not str(d["attribution"]).strip():
            rep.add(ERROR, "source_no_rights_info", "Licence or attribution is missing.", sid)
        if not d["rights"].get("use_confirmed"):
            rep.add(ERROR, "source_rights_unconfirmed", "Use rights are not confirmed.", sid)
        if rec.redistribution != "permitted":
            rep.add(
                WARNING,
                "source_not_redistributable",
                f"Redistribution is '{rec.redistribution}': not cleared for public release.",
                sid,
            )
        if not ensure_relative_posix(str(d["audio_path"])):
            rep.add(ERROR, "source_unsafe_path", "Registered audio path is unsafe.", sid)
            return
        try:
            path = resolve_within(self.paths.reference_dir, str(d["audio_path"]))
        except Exception:
            rep.add(
                ERROR,
                "source_unsafe_path",
                "Registered audio path escapes the reference directory.",
                sid,
            )
            return
        if not path.is_file():
            rep.add(ERROR, "source_missing", "Reference audio file is missing.", sid)
            return
        if sha256_file(path) != rec.sha256:
            rep.add(
                ERROR,
                "source_checksum_mismatch",
                "Reference audio changed since registration.",
                sid,
            )
            return
        try:
            audio = load_audio(path, self.config.audio)
        except AudioInputError as exc:
            rep.add(ERROR, "source_undecodable", exc.user_message, sid)
            return
        m = audio.metadata
        if (m.sample_rate, m.channels, m.frames) != (d["sample_rate"], d["channels"], d["frames"]):
            rep.add(
                ERROR, "source_metadata_mismatch", "Audio metadata differs from the record.", sid
            )

    # ------------------------------------------------------------------ samples
    def _samples(self, sources: dict[str, SourceRecord]) -> dict[str, SampleRecord]:
        rep = self.report
        rows, problems = read_jsonl_lenient(self.paths.manifest_file)
        for p in problems:
            rep.add(ERROR, "manifest_malformed", p, "manifest")
        out: dict[str, SampleRecord] = {}
        for line, raw in rows:
            subject = f"manifest:{line}"
            try:
                rec = SampleRecord.from_dict(raw)
            except DatasetError as exc:
                rep.add(ERROR, "sample_invalid", exc.user_message, subject)
                continue
            if raw.get("status") != STATUS_COMMITTED:
                rep.add(
                    ERROR,
                    "sample_not_committed",
                    f"Record status is {raw.get('status')!r}.",
                    rec.sample_id,
                )
                continue
            if rec.sample_id in out:
                rep.add(
                    ERROR, "sample_duplicate_id", f"Duplicate sample id {rec.sample_id}.", subject
                )
                continue
            out[rec.sample_id] = rec
            self._check_sample(rec, sources)
        return out

    def _check_sample(self, rec: SampleRecord, sources: dict[str, SourceRecord]) -> None:
        rep, sid, d = self.report, rec.sample_id, rec.data
        ann, gen, ref = d["annotation"], d["generated"], d["reference"]
        if d["dataset_id"] != self.config.dataset.dataset_id:
            rep.add(
                WARNING,
                "sample_other_dataset",
                f"Sample belongs to dataset {d['dataset_id']!r}.",
                sid,
            )
        flaw = ann["flaw_type"]
        if flaw not in PARAM_SPECS:
            rep.add(ERROR, "sample_unknown_flaw", f"Unknown flaw type {flaw!r}.", sid)
            return
        if set(ann["parameters"]) != set(PARAM_SPECS[flaw]):
            rep.add(ERROR, "sample_bad_parameters", "Parameters do not match the flaw type.", sid)
        if ann["timeline"] != TIMELINE_CONVENTION:
            rep.add(ERROR, "sample_bad_timeline", "Unknown timeline convention.", sid)
        if rec.source_id not in sources:
            rep.add(
                ERROR, "sample_source_missing", f"Source {rec.source_id} is not registered.", sid
            )
            return
        source = sources[rec.source_id]
        if ref["sha256"] != source.sha256:
            rep.add(
                ERROR,
                "sample_reference_checksum",
                "Recorded reference checksum differs from the registry.",
                sid,
            )

        sr, frames = int(gen["sample_rate"]), int(gen["frames"])
        err = _interval_ok(ann["output_interval"], frames, sr)
        if err:
            rep.add(ERROR, "sample_bad_interval", f"output_interval: {err}.", sid)
            return
        ref_frames = int(ref["frames"])
        if ann["source_interval"] is not None:
            err = _interval_ok(ann["source_interval"], ref_frames, sr)
            if err:
                rep.add(ERROR, "sample_bad_interval", f"source_interval: {err}.", sid)
                return
        self._check_arithmetic(rec, ref_frames, frames)

        if not ensure_relative_posix(str(gen["path"])):
            rep.add(ERROR, "sample_unsafe_path", "Generated audio path is unsafe.", sid)
            return
        try:
            path = resolve_within(self.paths.generated_dir, str(gen["path"]))
        except Exception:
            rep.add(
                ERROR, "sample_unsafe_path", "Generated path escapes the generated directory.", sid
            )
            return
        if not path.is_file():
            rep.add(ERROR, "sample_file_missing", "Generated audio file is missing.", sid)
            return
        if sha256_file(path) != gen["sha256"]:
            rep.add(
                ERROR, "sample_checksum_mismatch", "Generated audio changed since generation.", sid
            )
            return
        try:
            audio = load_audio(path, self.gen_settings)
        except AudioInputError as exc:
            rep.add(ERROR, "sample_undecodable", exc.user_message, sid)
            return
        tol = self.config.dataset.duration_tolerance_sec
        if (
            audio.num_samples != frames
            or abs(audio.duration_seconds - gen["duration_seconds"]) > tol
        ):
            rep.add(
                ERROR,
                "sample_duration_mismatch",
                f"Recorded {frames} frames, file has {audio.num_samples}.",
                sid,
            )
            return
        if (audio.sample_rate, audio.num_channels) != (sr, int(gen["channels"])):
            rep.add(
                ERROR,
                "sample_metadata_mismatch",
                "Sample rate or channels differ from the record.",
                sid,
            )
            return
        if self.deep:
            self._check_content(rec, source, audio.samples)

    def _check_arithmetic(self, rec: SampleRecord, ref_frames: int, gen_frames: int) -> None:
        ann, sid = rec.data["annotation"], rec.sample_id
        out = ann["output_interval"]
        out_len = int(out["end_sample"]) - int(out["start_sample"])
        flaw = ann["flaw_type"]
        eff = ann["effective_parameters"]
        bad: str | None = None
        if flaw == FLAW_PAUSE:
            if ann["source_anchor"] is None or ann["source_interval"] is not None:
                bad = "pause needs a source anchor and no source interval"
            elif int(out["start_sample"]) != int(ann["source_anchor"]["sample"]):
                bad = "pause output interval must start at the source anchor"
            elif out_len != eff.get("silence_samples") or gen_frames != ref_frames + out_len:
                bad = "frame counts do not match the inserted silence"
        elif flaw == FLAW_TEMPO:
            src = ann["source_interval"]
            if src is None:
                bad = "tempo change needs a source interval"
            else:
                src_len = int(src["end_sample"]) - int(src["start_sample"])
                if int(out["start_sample"]) != int(src["start_sample"]):
                    bad = "output interval must start where the source interval starts"
                elif gen_frames != ref_frames - src_len + out_len:
                    bad = "frame counts do not match the measured stretched segment"
        elif flaw == FLAW_ENERGY:
            src = ann["source_interval"]
            if src is None or src != {k: out[k] for k in src}:
                bad = "energy reduction intervals must coincide"
            elif gen_frames != ref_frames:
                bad = "energy reduction must not change the duration"
        if bad:
            self.report.add(ERROR, "sample_inconsistent_annotation", bad + ".", sid)

    def _check_content(self, rec: SampleRecord, source: SourceRecord, gen: np.ndarray) -> None:
        sid, ann = rec.sample_id, rec.data["annotation"]
        try:
            ref = load_audio(
                resolve_within(self.paths.reference_dir, str(source.data["audio_path"])),
                self.config.audio,
            ).samples
        except Exception:
            return  # already reported by the source check
        atol = _atol(str(rec.data["generated"]["subtype"]))
        out = ann["output_interval"]
        o_s, o_e = int(out["start_sample"]), int(out["end_sample"])
        flaw, ok = ann["flaw_type"], True
        if flaw == FLAW_PAUSE:
            n = ref.shape[1]
            a = int(ann["source_anchor"]["sample"])
            ok = (
                _same(gen[:, :a], ref[:, :a], atol)
                and not np.any(np.abs(gen[:, o_s:o_e]) > atol)
                and _same(gen[:, o_e:], ref[:, a:n], atol)
            )
        elif flaw == FLAW_TEMPO:
            s_e = int(ann["source_interval"]["end_sample"])
            ok = _same(gen[:, :o_s], ref[:, :o_s], atol) and _same(gen[:, o_e:], ref[:, s_e:], atol)
        elif flaw == FLAW_ENERGY:
            mult = float(ann["effective_parameters"]["amplitude_multiplier"])
            ok = (
                _same(gen[:, :o_s], ref[:, :o_s], atol)
                and _same(gen[:, o_e:], ref[:, o_e:], atol)
                and np.allclose(
                    gen[:, o_s:o_e], ref[:, o_s:o_e] * np.float32(mult), rtol=1e-5, atol=atol
                )
            )
        if not ok:
            self.report.add(
                ERROR,
                "sample_content_mismatch",
                "Audio does not match the annotation (ground truth would be wrong).",
                sid,
            )

    # -------------------------------------------------------------- file audits
    def _orphans(self, samples: dict[str, SampleRecord]) -> None:
        root = self.paths.dataset_generated_dir
        if not root.is_dir():
            return
        registered = {Path(str(s.data["generated"]["path"])).name for s in samples.values()}
        for p in sorted(root.iterdir()):
            if p.name.startswith(".") and (p.name.endswith(".tmp") or p.name.endswith(".prev")):
                self.report.add(WARNING, "partial_file", "Interrupted generation leftover.", p.name)
            elif p.suffix == GENERATED_EXTENSION and p.name not in registered:
                self.report.add(
                    WARNING, "unregistered_file", "Generated file has no manifest record.", p.name
                )

    def _splits(self, sources: dict[str, SourceRecord]) -> None:
        path = self.paths.splits_file
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            assignments = data["assignments"]
        except (OSError, ValueError, KeyError, TypeError):
            self.report.add(
                ERROR, "splits_malformed", "The splits file cannot be parsed.", "splits"
            )
            return
        for sid in sources:
            if sid not in assignments:
                self.report.add(
                    WARNING, "split_missing_source", "Source has no split assignment.", sid
                )
        for sid in assignments:
            if sid not in sources:
                self.report.add(
                    WARNING, "split_unknown_source", "Split refers to an unknown source.", sid
                )


def validate_dataset(
    config: Config, paths: ProjectPaths | None = None, *, deep: bool = True
) -> ValidationReport:
    """Audit the dataset described by ``config``. Counts and findings come from real files."""
    report = _Validator(config, paths or ProjectPaths(config), deep).run()
    kinds = Counter(f.code for f in report.findings)
    report.counts["errors"] = len(report.errors)
    report.counts["warnings"] = len(report.warnings)
    report.counts.update({f"finding:{k}": v for k, v in kinds.items()})
    return report
