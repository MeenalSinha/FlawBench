"""Participant-analysis workflow: real feature extraction, comparison, detection, explanation.

Both modes run the same established components (audio loader, ``extract_features``,
``compare_features`` via ``build_detection_input_from_features``, ``DetectionEngine``,
``DetectionValidator``). Uploaded audio is held in memory only; nothing is written to the dataset
unless :meth:`ParticipantAnalyzer.register_upload` is called explicitly with rights metadata.
No ground-truth annotation is read by this module.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flawbench import __version__
from flawbench.audio.loader import LoadedAudio, load_audio
from flawbench.config import Config
from flawbench.dataset.io import read_jsonl_lenient
from flawbench.dataset.registry import SourceRegistry
from flawbench.detection.engine import DetectionEngine
from flawbench.detection.inputs import build_detection_input_from_features
from flawbench.detection.schema import DetectionResult
from flawbench.detection.validator import DetectionValidator
from flawbench.exceptions import (
    ExplanationError,
    FlawBenchError,
    ParticipantInputError,
)
from flawbench.explainability.builder import explain_result
from flawbench.explainability.facts import SUPPORTED_CATEGORIES, AlignmentContext
from flawbench.explainability.schema import (
    STAGE_COMPLETED,
    STAGE_FAILED,
    STAGE_SKIPPED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NO_CANDIDATES,
    STATUS_PARTIAL,
    ExplanationCard,
    ParticipantAnalysis,
    RecordingInfo,
    StageRecord,
    TranscriptCheck,
)
from flawbench.explainability.validator import validate_analysis
from flawbench.features.extractor import FeatureResult, extract_features
from flawbench.features.pipeline import FeaturePipeline
from flawbench.features.schema import AudioInput, AudioRole
from flawbench.participant.benchmark import load_benchmark_context
from flawbench.participant.transcript import check_transcript
from flawbench.paths import ProjectPaths
from flawbench.provenance import collect_software_versions, config_fingerprint

MODE_REGISTERED = "registered_variant"
MODE_UPLOAD = "uploaded_participant"

STAGES = (
    "validate_inputs",
    "load_audio",
    "extract_features",
    "compare_and_detect",
    "validate_detection",
    "build_explanations",
)

STANDING_LIMITATIONS = (
    "Only pause, tempo and energy-reduction deviations are analyzed. Pitch-related and other "
    "delivery aspects were not evaluated, so a result does not cover delivery as a whole.",
    "Findings are candidates produced by configured detection rules on measured acoustic "
    "differences from the reference. They are not confirmed errors.",
    "Reference-based comparison is most meaningful when both recordings contain corresponding "
    "spoken content. No audio-text alignment backend exists, so no word-level references "
    "are given.",
    "Benchmark results were measured on controlled fixtures and do not by themselves show "
    "performance on a given participant's natural speech. Source types are declared at "
    "registration and are not independently verified.",
)


@dataclass(frozen=True)
class RegisteredVariantRef:
    """Whitelisted, label-free view of a manifest row. Annotations are deliberately absent."""

    sample_id: str
    source_id: str
    generated_path: str
    sha256: str


@dataclass
class AnalysisArtifacts:
    """In-memory material for the UI (playback and plots). Never serialized."""

    reference_audio: LoadedAudio
    participant_audio: LoadedAudio
    reference_features: FeatureResult
    participant_features: FeatureResult
    detection_input: Any
    detection_result: DetectionResult | None = None


@dataclass
class AnalysisSession:
    analysis: ParticipantAnalysis
    artifacts: AnalysisArtifacts | None = field(default=None, repr=False)


def _recording_info(
    role: str,
    origin: str,
    label: str,
    audio: LoadedAudio,
    sha256: str,
    source_id: str | None,
    sample_id: str | None,
    synthetic: bool | None,
) -> RecordingInfo:
    m = audio.metadata
    return RecordingInfo(
        role=role,
        origin=origin,
        label=label,
        source_id=source_id,
        sample_id=sample_id,
        filename=m.filename,
        sha256=sha256,
        sample_rate=audio.sample_rate,
        channels=audio.num_channels,
        duration_seconds=audio.duration_seconds,
        container_format=m.container_format,
        subtype=m.subtype,
        is_synthetic_fixture=synthetic,
    )


class ParticipantAnalyzer:
    def __init__(self, config: Config, paths: ProjectPaths | None = None) -> None:
        self.config = config
        self.paths = paths or ProjectPaths(config)
        self.registry = SourceRegistry(config, self.paths)
        self.features = FeaturePipeline(config, self.paths)

    # ------------------------------------------------------------------ listing
    def references(self) -> list[Any]:
        if not self.paths.sources_file.is_file():
            return []
        return self.registry.all()

    def registered_variants(self, source_id: str | None = None) -> list[RegisteredVariantRef]:
        """Manifest rows reduced to label-free fields (no annotation is read into the object)."""
        if not self.paths.manifest_file.is_file():
            return []
        rows, _ = read_jsonl_lenient(self.paths.manifest_file)
        out: list[RegisteredVariantRef] = []
        for _, rec in rows:
            if rec.get("record_type") != "sample":
                continue
            gen = rec.get("generated") or {}
            if not rec.get("sample_id") or not gen.get("path"):
                continue
            if source_id and rec.get("source_id") != source_id:
                continue
            out.append(
                RegisteredVariantRef(
                    str(rec["sample_id"]),
                    str(rec["source_id"]),
                    str(gen["path"]),
                    str(gen.get("sha256", "")),
                )
            )
        return out

    # ------------------------------------------------------------------ entry points
    def analyze_registered(self, sample_id: str) -> AnalysisSession:
        def load() -> tuple[Any, LoadedAudio, LoadedAudio, str, str, str | None]:
            ref_match = next(
                (v for v in self.registered_variants() if v.sample_id == sample_id), None
            )
            if ref_match is None:
                raise ParticipantInputError(f"Registered variant {sample_id!r} was not found.")
            record = self.registry.get(ref_match.source_id)
            _, var_path = self.features._load_manifest_sample(sample_id)
            ref_audio = self.registry.load_audio(record)
            var_audio = load_audio(var_path, self.config.audio)
            return record, ref_audio, var_audio, ref_match.sha256, ref_match.source_id, sample_id

        return self._run(MODE_REGISTERED, load, participant_transcript=None, upload=None)

    def analyze_upload(
        self,
        source_id: str,
        data: bytes,
        filename: str,
        participant_transcript: str | None = None,
    ) -> AnalysisSession:
        def load() -> tuple[Any, LoadedAudio, LoadedAudio, str, str, str | None]:
            if not data:
                raise ParticipantInputError("No participant recording was provided.")
            if not source_id:
                raise ParticipantInputError("No reference recording was selected.")
            record = self.registry.get(source_id)
            ref_audio = self.registry.load_audio(record)
            var_audio = load_audio(data, self.config.audio, filename=filename)
            return record, ref_audio, var_audio, hashlib.sha256(data).hexdigest(), source_id, None

        return self._run(
            MODE_UPLOAD, load, participant_transcript=participant_transcript, upload=filename
        )

    # ------------------------------------------------------------------ core
    def _run(
        self,
        mode: str,
        load: Any,
        *,
        participant_transcript: str | None,
        upload: str | None,
    ) -> AnalysisSession:
        stages: list[StageRecord] = []
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        created = datetime.now(UTC).isoformat(timespec="seconds")
        ref_info = part_info = None
        transcript: TranscriptCheck | None = None

        def fail(stage: str, exc: Exception, t0: float) -> AnalysisSession:
            stages.append(StageRecord(stage, STAGE_FAILED, str(exc), time.perf_counter() - t0))
            for name in STAGES[STAGES.index(stage) + 1 :]:
                stages.append(StageRecord(name, STAGE_SKIPPED, "earlier stage failed"))
            analysis = self._assemble(
                session_id,
                created,
                mode,
                STATUS_FAILED,
                ref_info,
                part_info,
                stages,
                [],
                [],
                {},
                transcript,
                {},
                {},
                f"{stage}: {exc}",
                [],
            )
            return AnalysisSession(analysis)

        t0 = time.perf_counter()
        try:
            record, ref_audio, var_audio, var_sha, source_id, sample_id = load()
        except (FlawBenchError, OSError, ValueError, KeyError) as exc:
            stage = "validate_inputs"
            return fail(stage, exc, t0)
        stages.append(StageRecord("validate_inputs", STAGE_COMPLETED, "", time.perf_counter() - t0))
        stages.append(StageRecord("load_audio", STAGE_COMPLETED, "", 0.0))

        synthetic = record.data.get("source_type") == "synthetic"
        ref_sha = str(record.data.get("sha256") or record.data.get("checksum_sha256") or "")
        ref_info = _recording_info(
            "reference",
            "registered_reference",
            f"Reference {source_id}",
            ref_audio,
            ref_sha,
            source_id,
            None,
            synthetic,
        )
        part_info = _recording_info(
            "participant",
            "uploaded" if mode == MODE_UPLOAD else "registered_variant",
            upload or f"Registered variant {sample_id}",
            var_audio,
            var_sha,
            source_id,
            sample_id,
            None,
        )
        transcript = check_transcript(
            record.data.get("transcript_text"),
            participant_transcript,
            self.config.explanation.transcript_similarity_warning,
        )

        t0 = time.perf_counter()
        try:
            ref_feat = self.features.extract_reference(source_id)
            if mode == MODE_REGISTERED and sample_id:
                var_feat = self.features.extract_variant(sample_id)
            else:
                var_feat = extract_features(
                    var_audio.samples,
                    sample_rate=var_audio.sample_rate,
                    config=self.config,
                    audio_input=AudioInput(
                        artifact_id=f"upl-{var_sha[:12]}",
                        source_id=source_id,
                        sample_id=None,
                        role=AudioRole.VARIANT,
                        audio_path=Path("<in-memory upload>"),
                        sha256=var_sha,
                        sample_rate=var_audio.sample_rate,
                        num_channels=var_audio.num_channels,
                        num_samples=var_audio.num_samples,
                        duration_seconds=var_audio.duration_seconds,
                    ),
                )
        except (FlawBenchError, OSError, ValueError) as exc:
            return fail("extract_features", exc, t0)
        stages.append(
            StageRecord("extract_features", STAGE_COMPLETED, "", time.perf_counter() - t0)
        )

        t0 = time.perf_counter()
        engine = DetectionEngine(self.config.detection)
        try:
            inp = build_detection_input_from_features(
                ref_feat,
                var_feat,
                sample_id=sample_id or f"upl-{var_sha[:12]}",
                source_id=source_id,
            )
            result = engine.detect(inp, tolerate_detector_failures=True)
        except (FlawBenchError, ValueError) as exc:
            return fail("compare_and_detect", exc, t0)
        stages.append(
            StageRecord("compare_and_detect", STAGE_COMPLETED, "", time.perf_counter() - t0)
        )

        problems = DetectionValidator().validate_result(result)
        if problems:
            return fail(
                "validate_detection", ExplanationError("; ".join(problems[:3])), time.perf_counter()
            )
        stages.append(StageRecord("validate_detection", STAGE_COMPLETED, "", 0.0))

        ctx = AlignmentContext(
            ref_audio.duration_seconds,
            var_audio.duration_seconds,
            inp.comparison.normalized_dtw_cost,
            self.config.explanation.alignment_duration_ratio_warning,
            transcript.below_warning_threshold,
        )
        t0 = time.perf_counter()
        try:
            cards = explain_result(result, ctx, self.config.explanation)
        except ExplanationError as exc:
            return fail("build_explanations", exc, t0)
        stages.append(
            StageRecord("build_explanations", STAGE_COMPLETED, "", time.perf_counter() - t0)
        )

        det_to_cat = {d.detector_id: d.flaw_type for d in engine.detectors}
        failed = {
            det_to_cat.get(k, k): v
            for k, v in (result.metadata.get("detector_failures") or {}).items()
        }
        completed = list(result.metadata.get("detectors_completed") or [])
        status = STATUS_PARTIAL if failed else (STATUS_COMPLETED if cards else STATUS_NO_CANDIDATES)
        warnings: list[str] = []
        if failed:
            warnings.append(
                "Some detectors failed: " + ", ".join(sorted(failed)) + ". Results are partial."
            )
        if ctx.length_mismatch:
            warnings.append("The recordings differ substantially in length.")
        if transcript.below_warning_threshold:
            warnings.append("The supplied transcript differs noticeably from the reference.")
        alignment = {
            "normalized_dtw_cost": inp.comparison.normalized_dtw_cost,
            "recording_duration_ratio": ctx.duration_ratio,
            "alignment_uncertain": ctx.uncertain,
        }
        from flawbench.detection.schema import DetectionResult as _DR  # noqa: F401

        analysis = self._assemble(
            session_id,
            created,
            mode,
            status,
            ref_info,
            part_info,
            stages,
            cards,
            completed,
            failed,
            transcript,
            alignment,
            {
                "detection_run_id": result.run_id,
                "detector_version": result.detector_version,
                "detector_config_fingerprint": result.config_fingerprint,
                "enabled_detectors": list(self.config.detection.enabled_detectors),
                "prediction_ids": [c.prediction_id for c in cards],
            },
            None,
            warnings,
        )
        sw = collect_software_versions()
        analysis.provenance = {
            "flawbench_version": __version__,
            "config_fingerprint": config_fingerprint(
                self.config, sections=("features", "detection", "explanation")
            ),
            "reference_feature_artifact": ref_feat.metadata.artifact_id,
            "participant_feature_artifact": var_feat.metadata.artifact_id,
            "software": sw,
        }
        bad = validate_analysis(analysis)
        if bad:
            return fail(
                "build_explanations", ExplanationError("; ".join(bad[:3])), time.perf_counter()
            )
        if self.config.explanation.show_benchmark_context:
            analysis.benchmark = load_benchmark_context(self.config, self.paths)
        art = AnalysisArtifacts(ref_audio, var_audio, ref_feat, var_feat, inp, result)
        return AnalysisSession(analysis, art)

    def _assemble(
        self,
        session_id: str,
        created: str,
        mode: str,
        status: str,
        ref: RecordingInfo | None,
        part: RecordingInfo | None,
        stages: list[StageRecord],
        cards: list[ExplanationCard],
        completed: list[str],
        failed: dict[str, str],
        transcript: TranscriptCheck | None,
        alignment: dict[str, Any],
        detection: dict[str, Any],
        failure: str | None,
        warnings: list[str],
    ) -> ParticipantAnalysis:
        return ParticipantAnalysis(
            session_id=session_id,
            created_utc=created,
            mode=mode,
            status=status,
            reference=ref,
            participant=part,
            stages=stages,
            supported_categories=list(SUPPORTED_CATEGORIES),
            categories_completed=completed,
            categories_failed=failed,
            cards=cards,
            warnings=warnings,
            limitations=list(STANDING_LIMITATIONS),
            transcript=transcript,
            alignment=alignment,
            detection=detection,
            provenance={},
            sort_policy=self.config.explanation.sort_policy,
            failure=failure,
        )

    # ------------------------------------------------------------------ explicit persistence
    def register_upload(
        self,
        data: bytes,
        filename: str,
        *,
        source_type: str,
        attribution: str,
        license: str,
        rights_confirmed: bool,
        transcript_text: str,  # noqa: A002
    ) -> str:
        """Permanently register an uploaded recording as a reference source.

        Only reachable through an explicit call, only if the configured persistence policy
        permits it, and only with rights metadata. A temporary file is used and removed.
        """
        if self.config.explanation.upload_persistence != "explicit_registration":
            raise ParticipantInputError("Registering uploads is disabled by configuration.")
        if not rights_confirmed:
            raise ParticipantInputError("Rights must be confirmed before registration.")
        load_audio(data, self.config.audio, filename=filename)  # validate before writing
        suffix = Path(filename).suffix or ".wav"
        with tempfile.TemporaryDirectory(prefix="flawbench_upload_") as tmp:
            tmp_path = Path(tmp) / f"upload{suffix}"
            tmp_path.write_bytes(data)
            record = self.registry.register(
                tmp_path,
                transcript_text=transcript_text,
                source_type=source_type,
                attribution=attribution,
                license=license,
                rights_confirmed=rights_confirmed,
                copy=True,
            )
        return str(record.source_id)
