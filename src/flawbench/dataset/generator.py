"""Transactional variant and batch generation.

``DatasetGenerator.generate_variant`` is the programmatic API (no CLI or Streamlit imports).
A variant is committed only when: the audio was written to a temporary file, decoded back and
verified, the source checksum is unchanged, and the manifest record was written atomically.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from flawbench import __version__
from flawbench.audio.loader import LoadedAudio
from flawbench.config import AudioSettings, Config
from flawbench.dataset import injectors
from flawbench.dataset.io import fingerprint, read_jsonl, sha256_bytes, sha256_file, write_jsonl
from flawbench.dataset.params import (
    FLAW_ENERGY,
    FLAW_PAUSE,
    FLAW_TEMPO,
    resolve_parameters,
)
from flawbench.dataset.plan import PLACEMENT_EXPLICIT, PLACEMENT_RANDOM, Plan, validate_placement
from flawbench.dataset.registry import SourceRegistry, utc_now
from flawbench.dataset.schema import (
    RECORD_SAMPLE,
    SCHEMA_VERSION,
    STATUS_COMMITTED,
    TIMELINE_CONVENTION,
    SampleRecord,
    SourceRecord,
    anchor_dict,
    interval_dict,
)
from flawbench.exceptions import DatasetError, FlawBenchError, GenerationError, InjectionError
from flawbench.logging_config import get_logger
from flawbench.paths import ProjectPaths, ensure_writable_directory
from flawbench.provenance import build_provenance, config_fingerprint

logger = get_logger("dataset.generator")

STATUS_GENERATED = "generated"
STATUS_REGENERATED = "regenerated"
STATUS_SKIPPED = "skipped_existing"
STATUS_FAILED = "failed"
SAMPLE_ID_PREFIX = "smp-"
SAMPLE_ID_HEX_LENGTH = 16
GENERATED_EXTENSION = ".wav"
_UNLIMITED_UPLOAD_MB = 1_000_000


def generated_audio_settings(config: Config) -> AudioSettings:
    """Audio settings for reading *generated* files: WAV only, size bounded by dataset limits."""
    return replace(
        config.audio, allowed_extensions=(GENERATED_EXTENSION,), max_upload_mb=_UNLIMITED_UPLOAD_MB
    )


_SFC_SET_ADD_PEAK_CHUNK = 0x1050


def write_wav_deterministic(
    path: Path, samples: np.ndarray, sample_rate: int, subtype: str
) -> None:
    """Write ``(channels, samples)`` audio as WAV with byte-for-byte reproducible output.

    libsndfile normally adds a PEAK chunk containing the current time, which would make file
    checksums differ between identical runs. That chunk is disabled through a libsndfile command
    (uses python-soundfile's low-level handle; verified in the test-suite).
    """
    with sf.SoundFile(
        path, "w", samplerate=sample_rate, channels=samples.shape[0], subtype=subtype, format="WAV"
    ) as handle:
        status = sf._snd.sf_command(handle._file, _SFC_SET_ADD_PEAK_CHUNK, sf._ffi.NULL, 0)
        if status != 0:
            raise GenerationError("Could not disable the non-deterministic WAV peak chunk.")
        handle.write(np.ascontiguousarray(samples.T))


def derive_seed(run_seed: int, key: str) -> int:
    """Stable 64-bit per-sample seed from a run seed and a variant key."""
    digest = hashlib.sha256(f"{run_seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class GenerationResult:
    status: str
    record: SampleRecord
    audio_path: Path


@dataclass(frozen=True)
class VariantOutcome:
    key: str
    status: str
    sample_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchResult:
    run_id: str
    plan_name: str
    plan_fingerprint: str
    started_utc: str
    finished_utc: str
    outcomes: tuple[VariantOutcome, ...]

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.outcomes:
            out[o.status] = out.get(o.status, 0) + 1
        return out

    @property
    def ok(self) -> bool:
        return all(o.status != STATUS_FAILED for o in self.outcomes)


def _software() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "soundfile", "audiotsm"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "flawbench_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libsndfile": sf.__libsndfile_version__,
        "packages": packages,
    }


class DatasetGenerator:
    def __init__(
        self,
        config: Config,
        paths: ProjectPaths | None = None,
        registry: SourceRegistry | None = None,
    ) -> None:
        self.config = config
        self.paths = paths or ProjectPaths(config)
        self.registry = registry or SourceRegistry(config, self.paths)

    # ---------------------------------------------------------------- manifest
    def samples(self) -> list[SampleRecord]:
        return [
            SampleRecord.from_dict(r) for r in read_jsonl(self.paths.manifest_file, what="manifest")
        ]

    # ----------------------------------------------------------------- placement
    def _resolve_placement(
        self,
        flaw: str,
        placement: dict[str, Any],
        params: dict[str, float],
        audio: LoadedAudio,
        seed: int | None,
        key: str,
    ) -> tuple[dict[str, int], dict[str, Any] | None]:
        """Return resolved integer sample placement and the seed record (None if no randomness)."""
        placement = validate_placement(flaw, placement, "placement")
        sr, n = audio.sample_rate, audio.num_samples
        length = (
            max(1, round(params["duration_sec"] * sr))
            if flaw == FLAW_PAUSE
            else max(1, round(params["segment_sec"] * sr))
        )
        seed_record: dict[str, Any] | None = None
        if placement["mode"] == PLACEMENT_EXPLICIT:
            pos = round(placement["at_sec" if flaw == FLAW_PAUSE else "start_sec"] * sr)
        elif placement["mode"] == PLACEMENT_RANDOM:
            if seed is None:
                raise GenerationError("Random placement requires an explicit seed.")
            sample_seed = derive_seed(seed, key)
            rng = np.random.default_rng(sample_seed)
            margin = round(self.config.dataset.placement_margin_sec * sr)
            lo = margin
            hi = n - margin if flaw == FLAW_PAUSE else n - length - margin
            if hi < lo:
                raise InjectionError(
                    "The recording is too short for random placement with the configured margin."
                )
            pos = int(rng.integers(lo, hi + 1))
            seed_record = {
                "run_seed": int(seed),
                "sample_seed": sample_seed,
                "used_for": "placement",
            }
        else:  # pragma: no cover - validate_placement rejects other modes
            raise GenerationError("Unsupported placement mode.")
        return {"position_sample": int(pos), "length_samples": int(length)}, seed_record

    # --------------------------------------------------------------- generation
    def generate_variant(
        self,
        source_id: str,
        flaw_type: str,
        *,
        severity: str | None = None,
        parameters: dict[str, Any] | None = None,
        placement: dict[str, Any],
        seed: int | None = None,
        overwrite: bool | None = None,
        plan_fingerprint: str | None = None,
        variant_key: str | None = None,
    ) -> GenerationResult:
        """Generate, verify and commit one variant containing exactly one injected flaw."""
        ds = self.config.dataset
        allow_overwrite = ds.allow_overwrite if overwrite is None else overwrite
        label, base_severity, params = resolve_parameters(
            flaw_type,
            severity,
            parameters,
            ds.severity_profiles,
            ds.parameter_bounds,
            error=GenerationError,
        )
        source = self.registry.get(source_id)
        audio = self.registry.load_audio(source)  # verifies the source checksum first
        sr = audio.sample_rate
        key = variant_key or f"{source_id}:{flaw_type}:{label}"
        resolved, seed_record = self._resolve_placement(
            flaw_type, placement, params, audio, seed, key
        )

        pos, length = resolved["position_sample"], resolved["length_samples"]
        if flaw_type == FLAW_PAUSE:
            result = injectors.insert_pause(audio.samples, at_sample=pos, silence_samples=length)
        elif flaw_type == FLAW_TEMPO:
            result = injectors.change_tempo(
                audio.samples, start=pos, end=pos + length, rate=params["rate"]
            )
        elif flaw_type == FLAW_ENERGY:
            result = injectors.reduce_energy(
                audio.samples, start=pos, end=pos + length, gain_db=params["gain_db"]
            )
        else:
            raise GenerationError(f"No injector is implemented for '{flaw_type}'.")

        out_frames = int(result.samples.shape[1])
        if out_frames / sr > ds.max_output_seconds:
            raise GenerationError(
                f"The output would exceed the {ds.max_output_seconds:g} s generation limit."
            )

        identity = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": ds.dataset_id,
            "source_sha256": source.sha256,
            "flaw_type": flaw_type,
            "severity": label,
            "parameters": params,
            "placement": resolved,
            "output_subtype": ds.output_subtype,
        }
        sample_id = f"{SAMPLE_ID_PREFIX}{fingerprint(identity)[:SAMPLE_ID_HEX_LENGTH]}"
        relative = f"{ds.dataset_id}/{sample_id}{GENERATED_EXTENSION}"
        final = self.paths.dataset_generated_dir / f"{sample_id}{GENERATED_EXTENSION}"
        ensure_writable_directory(self.paths.dataset_generated_dir)
        ensure_writable_directory(self.paths.metadata_dir)

        tmp = final.with_name(f".{final.name}.tmp")
        try:
            self._write_and_verify(tmp, result.samples, sr)
            new_sha = sha256_file(tmp)
            problems = self.registry.verify(source)
            if problems:
                raise GenerationError(f"Source integrity check failed: {problems[0]}")
            record = self._build_record(
                source,
                sample_id,
                relative,
                new_sha,
                audio,
                result,
                label,
                base_severity,
                params,
                resolved,
                seed_record,
                plan_fingerprint,
                flaw_type,
                sr,
            )
            status = self._commit(tmp, final, record, allow_overwrite)
        finally:
            tmp.unlink(missing_ok=True)
        stored = next(s for s in self.samples() if s.sample_id == sample_id)
        logger.info("%s %s (%s, %s)", status, sample_id, flaw_type, label)
        return GenerationResult(status, stored, final)

    def _write_and_verify(self, tmp: Path, samples: np.ndarray, sr: int) -> None:
        subtype = self.config.dataset.output_subtype
        try:
            write_wav_deterministic(tmp, samples, sr, subtype)
            data, rate = sf.read(tmp, dtype="float32", always_2d=True)
        except (sf.LibsndfileError, RuntimeError, OSError) as exc:
            raise GenerationError(
                "The generated audio could not be written.", detail=str(exc)
            ) from exc
        if rate != sr or data.shape != samples.T.shape or not np.isfinite(data).all():
            raise GenerationError("The generated audio failed read-back verification.")
        atol = 0.0 if subtype == "FLOAT" else 1.0 / 2 ** (15 if subtype == "PCM_16" else 23)
        if not np.allclose(data.T, samples, rtol=0.0, atol=atol):
            raise GenerationError("The generated audio differs from the computed signal.")

    def _build_record(
        self,
        source: SourceRecord,
        sample_id: str,
        relative: str,
        new_sha: str,
        audio: LoadedAudio,
        result: injectors.InjectionResult,
        label: str,
        base_severity: str | None,
        params: dict[str, float],
        resolved: dict[str, int],
        seed_record: dict[str, Any] | None,
        plan_fp: str | None,
        flaw_type: str,
        sr: int,
    ) -> dict[str, Any]:
        out_frames = int(result.samples.shape[1])
        src_iv = result.source_interval
        implementation = result.implementation or {"name": "numpy", "operation": flaw_type}
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": RECORD_SAMPLE,
            "status": STATUS_COMMITTED,
            "dataset_id": self.config.dataset.dataset_id,
            "sample_id": sample_id,
            "source_id": source.source_id,
            "reference": {
                "path": source.data["audio_path"],
                "sha256": source.sha256,
                "frames": audio.num_samples,
                "duration_seconds": audio.num_samples / sr,
            },
            "generated": {
                "path": relative,
                "sha256": new_sha,
                "frames": out_frames,
                "duration_seconds": out_frames / sr,
                "sample_rate": sr,
                "channels": audio.num_channels,
                "subtype": self.config.dataset.output_subtype,
            },
            "transcript_ref": {
                "source_id": source.source_id,
                "sha256": sha256_bytes(str(source.data["transcript_text"]).encode("utf-8")),
            },
            "annotation": {
                "flaw_type": flaw_type,
                "severity": label,
                "base_severity": base_severity,
                "parameters": params,
                "effective_parameters": result.effective_parameters,
                "source_interval": interval_dict(*src_iv, sr) if src_iv else None,
                "source_anchor": (
                    anchor_dict(result.source_anchor, sr)
                    if result.source_anchor is not None
                    else None
                ),
                "output_interval": interval_dict(*result.output_interval, sr),
                "timeline": TIMELINE_CONVENTION,
            },
            "seed": seed_record,
            "generated_utc": utc_now(),
            "software": _software(),
            "config_fingerprint": config_fingerprint(self.config, sections=("dataset",)),
            "plan_fingerprint": plan_fp,
            "transformation": {"implementation": implementation, "placement": resolved},
            "history": [],
        }

    def _commit(self, tmp: Path, final: Path, record: dict[str, Any], allow_overwrite: bool) -> str:
        records = read_jsonl(self.paths.manifest_file, what="manifest")
        existing = next((r for r in records if r.get("sample_id") == record["sample_id"]), None)
        new_sha = record["generated"]["sha256"]
        prev: Path | None = None
        status = STATUS_GENERATED

        if existing is not None:
            same = (
                final.is_file() and sha256_file(final) == existing["generated"]["sha256"] == new_sha
            )
            if same:
                return STATUS_SKIPPED
            if not allow_overwrite:
                raise GenerationError(
                    f"Sample {record['sample_id']} already exists with different or missing "
                    "content "
                    "(different library version?). Re-run with overwrite enabled to replace it."
                )
            record["history"] = [
                *existing.get("history", []),
                {
                    "replaced_utc": utc_now(),
                    "previous_generated_sha256": existing["generated"]["sha256"],
                    "previous_config_fingerprint": existing.get("config_fingerprint"),
                    "previous_software": existing.get("software"),
                },
            ]
            status = STATUS_REGENERATED
        elif final.exists() and sha256_file(final) != new_sha and not allow_overwrite:
            raise GenerationError(
                f"An unregistered file already occupies {final.name} with different content; "
                "refusing to overwrite it (enable overwrite or remove it)."
            )

        if final.exists():
            prev = final.with_name(f".{final.name}.prev")
            os.replace(final, prev)
        os.replace(tmp, final)
        try:
            kept = [r for r in records if r.get("sample_id") != record["sample_id"]]
            write_jsonl(self.paths.manifest_file, [*kept, record])
        except BaseException:
            final.unlink(missing_ok=True)
            if prev is not None:
                os.replace(prev, final)
            raise
        if prev is not None:
            prev.unlink(missing_ok=True)
        return status

    # --------------------------------------------------------------------- batch
    def run_plan(self, plan: Plan, *, overwrite: bool | None = None) -> BatchResult:
        """Run a plan. The whole plan is validated before anything is generated."""
        ds = self.config.dataset
        registered = [s.source_id for s in self.registry.all()]
        specs = plan.expand(registered)
        if not specs:
            raise DatasetError("The plan expands to no variants (no registered sources?).")
        if len(specs) > ds.max_variants_per_run:
            raise DatasetError(
                f"The plan expands to {len(specs)} variants; the limit is "
                f"{ds.max_variants_per_run}."
            )
        for spec in specs:
            if spec.source not in registered:
                raise DatasetError(f"Plan references unregistered source '{spec.source}'.")
            resolve_parameters(
                spec.flaw,
                spec.severity,
                spec.parameters,
                ds.severity_profiles,
                ds.parameter_bounds,
                error=DatasetError,
            )

        started = utc_now()
        outcomes: list[VariantOutcome] = []
        for spec in specs:
            try:
                res = self.generate_variant(
                    spec.source,
                    spec.flaw,
                    severity=spec.severity,
                    parameters=spec.parameters or None,
                    placement=spec.placement,
                    seed=plan.seed,
                    overwrite=overwrite,
                    plan_fingerprint=plan.fingerprint,
                    variant_key=spec.key,
                )
                outcomes.append(VariantOutcome(spec.key, res.status, res.record.sample_id))
            except FlawBenchError as exc:
                logger.error("Variant %s failed: %s [%s]", spec.key, exc.user_message, exc.detail)
                outcomes.append(VariantOutcome(spec.key, STATUS_FAILED, None, exc.user_message))
            except Exception as exc:
                logger.exception("Variant %s failed unexpectedly", spec.key)
                outcomes.append(
                    VariantOutcome(
                        spec.key, STATUS_FAILED, None, f"unexpected {type(exc).__name__}"
                    )
                )
        finished = utc_now()
        run_id = f"run-{fingerprint({'plan': plan.fingerprint, 'started': started})[:12]}"
        result = BatchResult(
            run_id, plan.name, plan.fingerprint, started, finished, tuple(outcomes)
        )
        self._record_run(result, plan)
        return result

    def _record_run(self, result: BatchResult, plan: Plan) -> None:
        runs = (
            read_jsonl(self.paths.runs_file, what="run log")
            if self.paths.runs_file.exists()
            else []
        )
        entry = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "run",
            "run_id": result.run_id,
            "plan_name": result.plan_name,
            "plan_fingerprint": result.plan_fingerprint,
            "plan_seed": plan.seed,
            "started_utc": result.started_utc,
            "finished_utc": result.finished_utc,
            "counts": result.counts,
            "complete_success": result.ok,
            "outcomes": [
                {"key": o.key, "status": o.status, "sample_id": o.sample_id, "error": o.error}
                for o in result.outcomes
            ],
            "provenance": build_provenance(self.config, seed=plan.seed),
        }
        write_jsonl(self.paths.runs_file, [*runs, entry])
