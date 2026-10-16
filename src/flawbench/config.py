"""Centralized, validated configuration.

Precedence (highest first):

1. Explicit runtime overrides (``overrides`` argument to :func:`load_config`)
2. Environment variables (real environment, then a ``.env`` file in the project root)
3. Configuration file (explicit path, ``FLAWBENCH_CONFIG``, or ``configs/config.yaml``)
4. Application defaults (the ``_FIELDS`` table below: the single definition of every setting)

Relative storage paths are resolved against the *project root*, never the working directory.
The project root is, in order: the ``project_root`` argument, ``FLAWBENCH_PROJECT_ROOT``, or the
nearest ancestor of this package containing ``pyproject.toml`` (falling back to the cwd).
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from flawbench.dataset.params import is_identifier, validate_bounds, validate_profiles
from flawbench.exceptions import ConfigurationError

ENV_CONFIG_FILE = "FLAWBENCH_CONFIG"
ENV_PROJECT_ROOT = "FLAWBENCH_PROJECT_ROOT"
DEFAULT_CONFIG_FILE = Path("configs") / "config.yaml"
DOTENV_FILE = ".env"

VALID_ENVIRONMENTS = ("development", "production")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
VALID_LAYOUTS = ("centered", "wide")
VALID_OUTPUT_SUBTYPES = ("FLOAT", "PCM_16", "PCM_24")
VALID_FEATURE_CHANNEL_POLICIES = ("mono_mix",)  # extensible; only mono downmix implemented in P3
VALID_PITCH_BACKENDS = ("librosa_yin", "librosa_pyin")  # librosa backends for F0 estimation
VALID_FEATURE_CACHE_POLICIES = ("skip_if_valid", "always_recompute")
VALID_DETECTION_CATEGORIES = ("pause_insertion", "tempo_change", "energy_reduction")
VALID_DETECTION_CACHE_POLICIES = ("skip_if_valid", "always_recompute")
VALID_ALIGNMENT_METRICS = ("euclidean", "cosine")
#: Single source for the alignment defaults (also used where no settings object is available).
DEFAULT_ALIGNMENT_METRIC = "euclidean"
DEFAULT_ALIGNMENT_EXCLUDE_ENERGY_COEFFICIENT = True
VALID_EVALUATION_SPLITS = ("test", "validation", "train", "all")
VALID_ZERO_DENOMINATOR_POLICIES = ("undefined", "zero")
VALID_UI_PAGES = (
    "overview",
    "analyze",
    "timeline",
    "evidence",
    "guidance",
    "evaluation",
    "technical",
)
VALID_ANALYSIS_MODES = ("registered_variant", "uploaded_participant")
VALID_VERBOSITY = ("concise", "standard", "detailed")
VALID_SORT_POLICIES = ("chronological", "evidence_strength")
VALID_UPLOAD_PERSISTENCE = ("transient_only", "explicit_registration")
VALID_EXPORT_FORMATS = ("markdown", "html", "json", "csv")
VALID_TRANSCRIPT_FORMATS = (".txt", ".md")


DEFAULT_SEVERITY_PROFILES: dict[str, dict[str, dict[str, float]]] = {
    "pause_insertion": {
        "low": {"duration_sec": 0.5},
        "medium": {"duration_sec": 1.0},
        "high": {"duration_sec": 2.0},
    },
    "tempo_change": {
        "low": {"rate": 0.85, "segment_sec": 2.0},
        "medium": {"rate": 0.7, "segment_sec": 2.0},
        "high": {"rate": 0.5, "segment_sec": 2.0},
    },
    "energy_reduction": {
        "low": {"gain_db": -6.0, "segment_sec": 2.0},
        "medium": {"gain_db": -12.0, "segment_sec": 2.0},
        "high": {"gain_db": -24.0, "segment_sec": 2.0},
    },
}
DEFAULT_PARAMETER_BOUNDS: dict[str, dict[str, dict[str, float]]] = {
    "pause_insertion": {"duration_sec": {"min": 0.05, "max": 30.0}},
    "tempo_change": {
        "rate": {"min": 0.25, "max": 4.0},
        "segment_sec": {"min": 0.25, "max": 30.0},
    },
    "energy_reduction": {
        "gain_db": {"min": -60.0, "max": -0.5},
        "segment_sec": {"min": 0.05, "max": 30.0},
    },
}
DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.7, "validation": 0.15, "test": 0.15}

# Domain invariant (not a runtime setting): which libsndfile container names each extension may
# legitimately contain. Used to reject files whose content does not match an allowed format.
EXTENSION_FORMATS: dict[str, tuple[str, ...]] = {
    ".wav": ("WAV", "WAVEX", "RF64"),
    ".wave": ("WAV", "WAVEX", "RF64"),
    ".flac": ("FLAC",),
    ".ogg": ("OGG",),
    ".oga": ("OGG",),
    ".mp3": ("MP3",),
    ".aiff": ("AIFF",),
    ".aif": ("AIFF",),
}


@dataclass(frozen=True)
class _Field:
    section: str
    key: str
    # "str" | "path" | "int" | "float" | "bool" | "list" | "float_list" | "dict" | "filename"
    kind: str
    default: Any
    env: str


_FIELDS: tuple[_Field, ...] = (
    _Field("app", "name", "str", "FlawBench", "FLAWBENCH_APP_NAME"),
    _Field("app", "environment", "str", "development", "FLAWBENCH_ENV"),
    _Field("app", "log_level", "str", "INFO", "FLAWBENCH_LOG_LEVEL"),
    _Field("storage", "reference_dir", "path", "data/reference", "FLAWBENCH_REFERENCE_DIR"),
    _Field("storage", "generated_dir", "path", "data/generated", "FLAWBENCH_GENERATED_DIR"),
    _Field("storage", "metadata_dir", "path", "data/metadata", "FLAWBENCH_METADATA_DIR"),
    _Field("storage", "output_dir", "path", "outputs", "FLAWBENCH_OUTPUT_DIR"),
    _Field(
        "audio",
        "allowed_extensions",
        "list",
        [".wav", ".flac", ".ogg", ".mp3"],
        "FLAWBENCH_AUDIO_EXTENSIONS",
    ),
    _Field("audio", "max_upload_mb", "int", 50, "FLAWBENCH_MAX_UPLOAD_MB"),
    _Field("audio", "min_sample_rate_hz", "int", 8000, "FLAWBENCH_MIN_SAMPLE_RATE_HZ"),
    _Field("audio", "max_channels", "int", 2, "FLAWBENCH_MAX_CHANNELS"),
    _Field("audio", "reject_non_finite_samples", "bool", True, "FLAWBENCH_REJECT_NON_FINITE"),
    _Field("ui", "page_title", "str", "FlawBench - Speech Flaw Analysis", "FLAWBENCH_PAGE_TITLE"),
    _Field("ui", "layout", "str", "wide", "FLAWBENCH_LAYOUT"),
    _Field("ui", "show_roadmap", "bool", True, "FLAWBENCH_SHOW_ROADMAP"),
    _Field("diagnostics", "detailed_logging", "bool", False, "FLAWBENCH_DETAILED_LOGGING"),
    _Field("diagnostics", "expose_error_details", "bool", False, "FLAWBENCH_EXPOSE_ERROR_DETAILS"),
    _Field("dataset", "dataset_id", "str", "flawbench-v1", "FLAWBENCH_DATASET_ID"),
    _Field(
        "dataset", "sources_filename", "filename", "sources.jsonl", "FLAWBENCH_SOURCES_FILENAME"
    ),
    _Field(
        "dataset", "manifest_filename", "filename", "manifest.jsonl", "FLAWBENCH_MANIFEST_FILENAME"
    ),
    _Field("dataset", "runs_filename", "filename", "runs.jsonl", "FLAWBENCH_RUNS_FILENAME"),
    _Field("dataset", "splits_filename", "filename", "splits.json", "FLAWBENCH_SPLITS_FILENAME"),
    _Field("dataset", "copy_reference", "bool", False, "FLAWBENCH_COPY_REFERENCE"),
    _Field("dataset", "allow_overwrite", "bool", False, "FLAWBENCH_ALLOW_OVERWRITE"),
    _Field("dataset", "output_subtype", "str", "FLOAT", "FLAWBENCH_OUTPUT_SUBTYPE"),
    _Field("dataset", "placement_margin_sec", "float", 0.25, "FLAWBENCH_PLACEMENT_MARGIN_SEC"),
    _Field("dataset", "max_output_seconds", "float", 600.0, "FLAWBENCH_MAX_OUTPUT_SECONDS"),
    _Field("dataset", "max_variants_per_run", "int", 1000, "FLAWBENCH_MAX_VARIANTS_PER_RUN"),
    _Field("dataset", "duration_tolerance_sec", "float", 0.001, "FLAWBENCH_DURATION_TOLERANCE_SEC"),
    _Field("dataset", "split_seed", "int", 1337, "FLAWBENCH_SPLIT_SEED"),
    _Field("dataset", "split_ratios", "dict", DEFAULT_SPLIT_RATIOS, "FLAWBENCH_SPLIT_RATIOS"),
    _Field(
        "dataset",
        "severity_profiles",
        "dict",
        DEFAULT_SEVERITY_PROFILES,
        "FLAWBENCH_SEVERITY_PROFILES",
    ),
    _Field(
        "dataset",
        "parameter_bounds",
        "dict",
        DEFAULT_PARAMETER_BOUNDS,
        "FLAWBENCH_PARAMETER_BOUNDS",
    ),
    # ---- Phase 3: feature-extraction settings -----------------------------------------
    _Field("features", "frame_ms", "float", 25.0, "FLAWBENCH_FEAT_FRAME_MS"),
    _Field("features", "hop_ms", "float", 10.0, "FLAWBENCH_FEAT_HOP_MS"),
    _Field("features", "n_mfcc", "int", 13, "FLAWBENCH_FEAT_N_MFCC"),
    _Field("features", "spectral_rolloff_p", "float", 0.85, "FLAWBENCH_FEAT_ROLLOFF_P"),
    _Field("features", "pitch_backend", "str", "librosa_pyin", "FLAWBENCH_FEAT_PITCH_BACKEND"),
    _Field("features", "f0_min_hz", "float", 75.0, "FLAWBENCH_FEAT_F0_MIN_HZ"),
    _Field("features", "f0_max_hz", "float", 600.0, "FLAWBENCH_FEAT_F0_MAX_HZ"),
    _Field("features", "channel_policy", "str", "mono_mix", "FLAWBENCH_FEAT_CHANNEL_POLICY"),
    _Field("features", "silence_threshold_db", "float", -40.0, "FLAWBENCH_FEAT_SILENCE_DB"),
    _Field("features", "min_silence_ms", "float", 50.0, "FLAWBENCH_FEAT_MIN_SILENCE_MS"),
    _Field("features", "features_dir", "path", "data/features", "FLAWBENCH_FEATURES_DIR"),
    _Field(
        "features",
        "feature_manifest_filename",
        "filename",
        "feature_manifest.jsonl",
        "FLAWBENCH_FEAT_MANIFEST_FILENAME",
    ),
    _Field("features", "cache_policy", "str", "skip_if_valid", "FLAWBENCH_FEAT_CACHE_POLICY"),
    _Field("features", "max_feature_seconds", "float", 600.0, "FLAWBENCH_FEAT_MAX_SECONDS"),
    _Field("features", "schema_version", "str", "3.0", "FLAWBENCH_FEAT_SCHEMA_VERSION"),
    # ---- Phase 4: detection settings --------------------------------------------------
    _Field(
        "detection",
        "enabled_detectors",
        "list",
        ["pause_insertion", "tempo_change", "energy_reduction"],
        "FLAWBENCH_DETECT_ENABLED",
    ),
    _Field(
        "detection",
        "pause_silence_threshold_db",
        "float",
        -35.0,
        "FLAWBENCH_DETECT_PAUSE_THRESH_DB",
    ),
    _Field(
        "detection",
        "pause_min_duration_sec",
        "float",
        0.15,
        "FLAWBENCH_DETECT_PAUSE_MIN_SEC",
    ),
    _Field(
        "detection",
        "pause_min_difference_sec",
        "float",
        0.10,
        "FLAWBENCH_DETECT_PAUSE_DIFF_SEC",
    ),
    _Field(
        "detection",
        "tempo_ratio_min_deviation",
        "float",
        0.15,
        "FLAWBENCH_DETECT_TEMPO_DEV",
    ),
    _Field(
        "detection",
        "tempo_min_duration_sec",
        "float",
        0.25,
        "FLAWBENCH_DETECT_TEMPO_MIN_SEC",
    ),
    _Field(
        "detection",
        "alignment_metric",
        "str",
        "euclidean",
        "FLAWBENCH_DETECT_ALIGN_METRIC",
    ),
    _Field(
        "detection",
        "alignment_exclude_energy_coefficient",
        "bool",
        True,
        "FLAWBENCH_DETECT_ALIGN_NO_C0",
    ),
    _Field(
        "detection",
        "tempo_slope_window_sec",
        "float",
        0.5,
        "FLAWBENCH_DETECT_TEMPO_WINDOW_SEC",
    ),
    _Field(
        "detection",
        "tempo_max_inactive_fraction",
        "float",
        0.2,
        "FLAWBENCH_DETECT_TEMPO_MAX_INACTIVE",
    ),
    _Field(
        "detection",
        "pause_max_voiced_fraction",
        "float",
        0.25,
        "FLAWBENCH_DETECT_PAUSE_MAX_VOICED",
    ),
    _Field(
        "detection",
        "energy_reduction_threshold_db",
        "float",
        -4.0,
        "FLAWBENCH_DETECT_ENERGY_THRESH_DB",
    ),
    _Field(
        "detection",
        "energy_reduction_min_duration_sec",
        "float",
        0.15,
        "FLAWBENCH_DETECT_ENERGY_MIN_SEC",
    ),
    _Field(
        "detection",
        "merge_gap_sec",
        "float",
        0.05,
        "FLAWBENCH_DETECT_MERGE_GAP_SEC",
    ),
    _Field(
        "detection",
        "min_candidate_duration_sec",
        "float",
        0.08,
        "FLAWBENCH_DETECT_MIN_CAND_SEC",
    ),
    _Field(
        "detection",
        "detection_dir",
        "path",
        "data/detections",
        "FLAWBENCH_DETECTIONS_DIR",
    ),
    _Field(
        "detection",
        "detection_manifest_filename",
        "filename",
        "detections_manifest.jsonl",
        "FLAWBENCH_DETECT_MANIFEST_FILENAME",
    ),
    _Field(
        "detection",
        "cache_policy",
        "str",
        "skip_if_valid",
        "FLAWBENCH_DETECT_CACHE_POLICY",
    ),
    _Field(
        "detection",
        "schema_version",
        "str",
        "4.0",
        "FLAWBENCH_DETECT_SCHEMA_VERSION",
    ),
    _Field(
        "evaluation",
        "default_split",
        "str",
        "test",
        "FLAWBENCH_EVAL_DEFAULT_SPLIT",
    ),
    _Field(
        "evaluation",
        "matching_iou_threshold",
        "float",
        0.5,
        "FLAWBENCH_EVAL_IOU_THRESH",
    ),
    _Field(
        "evaluation",
        "localization_iou_thresholds",
        "float_list",
        [0.3, 0.5, 0.7, 0.9],
        "FLAWBENCH_EVAL_LOC_IOU_THRESHOLDS",
    ),
    _Field(
        "evaluation",
        "require_category_match",
        "bool",
        True,
        "FLAWBENCH_EVAL_REQUIRE_CAT_MATCH",
    ),
    _Field(
        "evaluation",
        "zero_denominator_policy",
        "str",
        "undefined",
        "FLAWBENCH_EVAL_ZERO_DENOM_POLICY",
    ),
    _Field(
        "evaluation",
        "bootstrap_replicates",
        "int",
        0,
        "FLAWBENCH_EVAL_BOOTSTRAP_REPLICATES",
    ),
    _Field(
        "evaluation",
        "bootstrap_seed",
        "int",
        1337,
        "FLAWBENCH_EVAL_BOOTSTRAP_SEED",
    ),
    _Field(
        "evaluation",
        "include_identity_controls",
        "bool",
        True,
        "FLAWBENCH_EVAL_INCLUDE_CONTROLS",
    ),
    _Field(
        "evaluation",
        "evaluations_dir",
        "path",
        "outputs/evaluations",
        "FLAWBENCH_EVALUATIONS_DIR",
    ),
    _Field(
        "evaluation",
        "evaluations_manifest_filename",
        "filename",
        "evaluations_manifest.jsonl",
        "FLAWBENCH_EVAL_MANIFEST_FILENAME",
    ),
    _Field(
        "evaluation",
        "schema_version",
        "str",
        "5.0",
        "FLAWBENCH_EVAL_SCHEMA_VERSION",
    ),
    # ------------------------------------------------------------ Phase 6: explanations & UI
    _Field("explanation", "default_page", "str", "overview", "FLAWBENCH_EXPLAIN_DEFAULT_PAGE"),
    _Field(
        "explanation",
        "analysis_modes",
        "list",
        ["registered_variant", "uploaded_participant"],
        "FLAWBENCH_EXPLAIN_ANALYSIS_MODES",
    ),
    _Field("explanation", "verbosity", "str", "standard", "FLAWBENCH_EXPLAIN_VERBOSITY"),
    _Field("explanation", "sort_policy", "str", "chronological", "FLAWBENCH_EXPLAIN_SORT_POLICY"),
    _Field(
        "explanation",
        "playback_context_sec",
        "float",
        1.0,
        "FLAWBENCH_EXPLAIN_PLAYBACK_CONTEXT_SEC",
    ),
    _Field("explanation", "max_excerpt_sec", "float", 30.0, "FLAWBENCH_EXPLAIN_MAX_EXCERPT_SEC"),
    _Field(
        "explanation",
        "show_technical_evidence",
        "bool",
        True,
        "FLAWBENCH_EXPLAIN_SHOW_TECHNICAL_EVIDENCE",
    ),
    _Field(
        "explanation",
        "transcript_formats",
        "list",
        [".txt"],
        "FLAWBENCH_EXPLAIN_TRANSCRIPT_FORMATS",
    ),
    _Field(
        "explanation",
        "transcript_similarity_warning",
        "float",
        0.8,
        "FLAWBENCH_EXPLAIN_TRANSCRIPT_SIMILARITY_WARNING",
    ),
    _Field(
        "explanation",
        "alignment_duration_ratio_warning",
        "float",
        2.0,
        "FLAWBENCH_EXPLAIN_ALIGNMENT_DURATION_RATIO_WARNING",
    ),
    _Field(
        "explanation",
        "upload_persistence",
        "str",
        "explicit_registration",
        "FLAWBENCH_EXPLAIN_UPLOAD_PERSISTENCE",
    ),
    _Field(
        "explanation",
        "reports_dir",
        "path",
        "outputs/participant_reports",
        "FLAWBENCH_EXPLAIN_REPORTS_DIR",
    ),
    _Field(
        "explanation",
        "export_formats",
        "list",
        ["markdown", "html", "json", "csv"],
        "FLAWBENCH_EXPLAIN_EXPORT_FORMATS",
    ),
    _Field(
        "explanation",
        "save_reports_automatically",
        "bool",
        False,
        "FLAWBENCH_EXPLAIN_SAVE_REPORTS_AUTOMATICALLY",
    ),
    _Field(
        "explanation",
        "show_benchmark_context",
        "bool",
        True,
        "FLAWBENCH_EXPLAIN_SHOW_BENCHMARK_CONTEXT",
    ),
)


_SCHEMA: dict[str, dict[str, _Field]] = {}
for _f in _FIELDS:
    _SCHEMA.setdefault(_f.section, {})[_f.key] = _f

ENV_VARIABLES: tuple[str, ...] = tuple(f.env for f in _FIELDS)


@dataclass(frozen=True)
class AppSettings:
    name: str
    environment: str
    log_level: str


@dataclass(frozen=True)
class StorageSettings:
    reference_dir: Path
    generated_dir: Path
    metadata_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class AudioSettings:
    allowed_extensions: tuple[str, ...]
    max_upload_mb: int
    min_sample_rate_hz: int
    max_channels: int
    reject_non_finite_samples: bool

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def extension_names(self) -> list[str]:
        """Extensions without the leading dot (the form Streamlit's uploader expects)."""
        return [ext.lstrip(".") for ext in self.allowed_extensions]

    @property
    def allowed_formats(self) -> frozenset[str]:
        """Container format names permitted by the allowed extensions."""
        return frozenset(fmt for ext in self.allowed_extensions for fmt in EXTENSION_FORMATS[ext])


@dataclass(frozen=True)
class UISettings:
    page_title: str
    layout: str
    show_roadmap: bool


@dataclass(frozen=True)
class DiagnosticsSettings:
    detailed_logging: bool
    expose_error_details: bool


@dataclass(frozen=True)
class DatasetSettings:
    dataset_id: str
    sources_filename: str
    manifest_filename: str
    runs_filename: str
    splits_filename: str
    copy_reference: bool
    allow_overwrite: bool
    output_subtype: str
    placement_margin_sec: float
    max_output_seconds: float
    max_variants_per_run: int
    duration_tolerance_sec: float
    split_seed: int
    split_ratios: dict[str, float]
    severity_profiles: dict[str, dict[str, dict[str, float]]]
    parameter_bounds: dict[str, dict[str, dict[str, float]]]


@dataclass(frozen=True)
class FeaturesSettings:
    """Phase 3 acoustic feature-extraction settings."""

    frame_ms: float
    hop_ms: float
    n_mfcc: int
    spectral_rolloff_p: float
    pitch_backend: str
    f0_min_hz: float
    f0_max_hz: float
    channel_policy: str
    silence_threshold_db: float
    min_silence_ms: float
    features_dir: Path
    feature_manifest_filename: str
    cache_policy: str
    max_feature_seconds: float
    schema_version: str


@dataclass(frozen=True)
class DetectionSettings:
    """Phase 4 flaw-detection and temporal localization settings."""

    enabled_detectors: tuple[str, ...]
    pause_silence_threshold_db: float
    pause_min_duration_sec: float
    pause_min_difference_sec: float
    tempo_ratio_min_deviation: float
    tempo_min_duration_sec: float
    alignment_metric: str
    alignment_exclude_energy_coefficient: bool
    tempo_slope_window_sec: float
    tempo_max_inactive_fraction: float
    pause_max_voiced_fraction: float
    energy_reduction_threshold_db: float
    energy_reduction_min_duration_sec: float
    merge_gap_sec: float
    min_candidate_duration_sec: float
    detection_dir: Path
    detection_manifest_filename: str
    cache_policy: str
    schema_version: str


@dataclass(frozen=True)
class EvaluationSettings:
    """Phase 5 evaluation and benchmarking settings."""

    default_split: str
    matching_iou_threshold: float
    localization_iou_thresholds: tuple[float, ...]
    require_category_match: bool
    zero_denominator_policy: str
    bootstrap_replicates: int
    bootstrap_seed: int
    include_identity_controls: bool
    evaluations_dir: Path
    evaluations_manifest_filename: str
    schema_version: str


@dataclass(frozen=True)
class ExplanationSettings:
    """Phase 6 participant-analysis, explanation and report settings."""

    default_page: str
    analysis_modes: tuple[str, ...]
    verbosity: str
    sort_policy: str
    playback_context_sec: float
    max_excerpt_sec: float
    show_technical_evidence: bool
    transcript_formats: tuple[str, ...]
    transcript_similarity_warning: float
    alignment_duration_ratio_warning: float
    upload_persistence: str
    reports_dir: Path
    export_formats: tuple[str, ...]
    save_reports_automatically: bool
    show_benchmark_context: bool


@dataclass(frozen=True)
class Config:
    app: AppSettings
    storage: StorageSettings
    audio: AudioSettings
    ui: UISettings
    diagnostics: DiagnosticsSettings
    dataset: DatasetSettings
    features: FeaturesSettings
    detection: DetectionSettings
    evaluation: EvaluationSettings
    explanation: ExplanationSettings
    project_root: Path
    config_file: Path | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot (used for effective-config display and run provenance)."""
        data = asdict(self)
        data["storage"] = {k: str(v) for k, v in data["storage"].items()}
        data["audio"]["allowed_extensions"] = list(self.audio.allowed_extensions)
        data["features"]["features_dir"] = str(self.features.features_dir)
        data["detection"]["detection_dir"] = str(self.detection.detection_dir)
        data["detection"]["enabled_detectors"] = list(self.detection.enabled_detectors)
        data["evaluation"]["evaluations_dir"] = str(self.evaluation.evaluations_dir)
        data["evaluation"]["localization_iou_thresholds"] = list(
            self.evaluation.localization_iou_thresholds
        )
        data["explanation"]["analysis_modes"] = list(self.explanation.analysis_modes)
        data["explanation"]["transcript_formats"] = list(self.explanation.transcript_formats)
        data["explanation"]["export_formats"] = list(self.explanation.export_formats)
        data["explanation"]["reports_dir"] = str(self.explanation.reports_dir)
        data["project_root"] = str(self.project_root)
        data["config_file"] = str(self.config_file) if self.config_file else None
        return data


def defaults() -> dict[str, dict[str, Any]]:
    """Return a fresh copy of the application defaults."""
    out: dict[str, dict[str, Any]] = {}
    for f in _FIELDS:
        value = copy.deepcopy(f.default)
        out.setdefault(f.section, {})[f.key] = value
    return out


def discover_project_root() -> Path:
    """Nearest ancestor of this package containing ``pyproject.toml``, else the cwd."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return values
    except OSError as exc:
        raise ConfigurationError(
            f"The {DOTENV_FILE} file could not be read.", detail=f"{path}: {exc}"
        ) from exc
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(
                f"Malformed line {number} in {DOTENV_FILE}.", detail=f"{path}:{number}"
            )
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _coerce(field: _Field, value: Any, source: str) -> Any:
    label = f"{field.section}.{field.key}"

    def bad(expected: str) -> ConfigurationError:
        return ConfigurationError(
            f"Invalid value for '{label}' from {source}: expected {expected}.",
            detail=f"got {value!r}",
        )

    if field.kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise bad("a boolean (true/false)")
    if field.kind == "int":
        if isinstance(value, bool):
            raise bad("an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
        raise bad("an integer")
    if field.kind == "list":
        if isinstance(value, str):
            items: list[Any] = [part for part in value.split(",") if part.strip()]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise bad("a list or comma-separated string")
        if not all(isinstance(i, str) for i in items):
            raise bad("a list of strings")
        normalized = []
        for item in items:
            s = item.strip()
            if field.key == "allowed_extensions":
                s = s.lower()
                normalized.append(s if s.startswith(".") else f".{s}")
            else:
                normalized.append(s)
        return normalized
    if field.kind == "float_list":
        if isinstance(value, str):
            raw_items = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, (list, tuple)):
            raw_items = list(value)
        else:
            raise bad("a list of numbers or comma-separated string")
        parsed_floats: list[float] = []
        for item in raw_items:
            try:
                num = float(item)
                if not math.isfinite(num):
                    raise bad("a list of finite numbers")
                parsed_floats.append(num)
            except (ValueError, TypeError):
                raise bad("a list of numbers") from None
        return parsed_floats
    if field.kind == "float":
        if isinstance(value, bool):
            raise bad("a number")
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError:
                raise bad("a number") from None
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
        raise bad("a finite number")
    if field.kind == "dict":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise bad("a JSON object") from None
        if not isinstance(value, dict):
            raise bad("a mapping")
        return copy.deepcopy(value)
    if field.kind == "filename":
        if not isinstance(value, str) or not value.strip():
            raise bad("a non-empty file name")
        name = value.strip()
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise bad("a plain file name without directories")
        return name
    # "str" and "path"
    if not isinstance(value, str) or not value.strip():
        raise bad("a non-empty string")
    return value.strip()


def _load_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigurationError(
            "The configuration file does not exist.", detail=str(path)
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            "The configuration file could not be read.", detail=f"{path}: {exc}"
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            "The configuration file is not valid YAML.", detail=f"{path}: {exc}"
        ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            "The configuration file must contain a mapping of sections.", detail=str(path)
        )
    return data


def _apply_layer(merged: dict[str, dict[str, Any]], layer: Mapping[str, Any], source: str) -> None:
    for section, body in layer.items():
        if section not in _SCHEMA:
            raise ConfigurationError(
                f"Unknown configuration section '{section}' in {source}.",
                detail=f"valid sections: {', '.join(_SCHEMA)}",
            )
        if not isinstance(body, Mapping):
            raise ConfigurationError(f"Section '{section}' in {source} must be a mapping.")
        for key, value in body.items():
            field = _SCHEMA[section].get(key)
            if field is None:
                raise ConfigurationError(
                    f"Unknown setting '{section}.{key}' in {source}.",
                    detail=f"valid keys: {', '.join(_SCHEMA[section])}",
                )
            merged[section][key] = _coerce(field, value, source)


def _resolve_root(project_root: Path | str | None, env: Mapping[str, str]) -> Path:
    chosen = project_root or env.get(ENV_PROJECT_ROOT) or discover_project_root()
    root = Path(chosen).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError("The project root is not a directory.", detail=str(root))
    return root


def _resolve_config_file(
    config_path: Path | str | None, env: Mapping[str, str], root: Path
) -> Path | None:
    explicit = config_path or env.get(ENV_CONFIG_FILE)
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidate = root / DEFAULT_CONFIG_FILE
    return candidate if candidate.is_file() else None


def _resolve_dir(root: Path, raw: str, label: str) -> Path:
    path = Path(raw).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ConfigurationError(
            f"'{label}' points to something that is not a directory.", detail=str(resolved)
        )
    return resolved


def _build_dataset(ds: dict[str, Any]) -> DatasetSettings:
    if not is_identifier(ds["dataset_id"]):
        raise ConfigurationError(
            "dataset.dataset_id may contain only letters, digits, '.', '_' and '-'.",
            detail=f"got {ds['dataset_id']!r}",
        )
    names = [
        ds[k] for k in ("sources_filename", "manifest_filename", "runs_filename", "splits_filename")
    ]
    if len(set(names)) != len(names):
        raise ConfigurationError("The dataset metadata file names must all be different.")
    if ds["output_subtype"] not in VALID_OUTPUT_SUBTYPES:
        raise ConfigurationError(
            f"dataset.output_subtype must be one of: {', '.join(VALID_OUTPUT_SUBTYPES)}.",
            detail=f"got {ds['output_subtype']!r}",
        )
    for key in ("placement_margin_sec", "max_output_seconds", "duration_tolerance_sec"):
        if ds[key] <= 0 and not (key == "placement_margin_sec" and ds[key] == 0):
            raise ConfigurationError(f"dataset.{key} must be positive.", detail=f"got {ds[key]}")
    if ds["placement_margin_sec"] < 0:
        raise ConfigurationError("dataset.placement_margin_sec must not be negative.")
    if ds["max_variants_per_run"] < 1:
        raise ConfigurationError("dataset.max_variants_per_run must be at least 1.")
    ratios = ds["split_ratios"]
    if (
        not ratios
        or not all(isinstance(k, str) and is_identifier(k) for k in ratios)
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
            for v in ratios.values()
        )
        or abs(sum(ratios.values()) - 1.0) > 1e-6
    ):
        raise ConfigurationError(
            "dataset.split_ratios must map split names to non-negative numbers summing to 1.",
            detail=f"got {ratios!r}",
        )
    validate_bounds(ds["parameter_bounds"])
    validate_profiles(ds["severity_profiles"], ds["parameter_bounds"])
    return DatasetSettings(
        dataset_id=ds["dataset_id"],
        sources_filename=ds["sources_filename"],
        manifest_filename=ds["manifest_filename"],
        runs_filename=ds["runs_filename"],
        splits_filename=ds["splits_filename"],
        copy_reference=bool(ds["copy_reference"]),
        allow_overwrite=bool(ds["allow_overwrite"]),
        output_subtype=ds["output_subtype"],
        placement_margin_sec=float(ds["placement_margin_sec"]),
        max_output_seconds=float(ds["max_output_seconds"]),
        max_variants_per_run=int(ds["max_variants_per_run"]),
        duration_tolerance_sec=float(ds["duration_tolerance_sec"]),
        split_seed=int(ds["split_seed"]),
        split_ratios={k: float(v) for k, v in ratios.items()},
        severity_profiles=ds["severity_profiles"],
        parameter_bounds=ds["parameter_bounds"],
    )


def _build_features(ft: dict[str, Any], root: Path) -> FeaturesSettings:
    """Validate and construct :class:`FeaturesSettings`."""
    if ft["frame_ms"] <= 0:
        raise ConfigurationError(
            "features.frame_ms must be positive.", detail=f"got {ft['frame_ms']}"
        )
    if ft["hop_ms"] <= 0:
        raise ConfigurationError("features.hop_ms must be positive.", detail=f"got {ft['hop_ms']}")
    if ft["hop_ms"] > ft["frame_ms"]:
        raise ConfigurationError(
            "features.hop_ms must not exceed features.frame_ms.",
            detail=f"hop={ft['hop_ms']} frame={ft['frame_ms']}",
        )
    if ft["n_mfcc"] < 1:
        raise ConfigurationError(
            "features.n_mfcc must be at least 1.", detail=f"got {ft['n_mfcc']}"
        )
    if not (0.0 < ft["spectral_rolloff_p"] < 1.0):
        raise ConfigurationError(
            "features.spectral_rolloff_p must be in (0, 1).",
            detail=f"got {ft['spectral_rolloff_p']}",
        )
    if ft["pitch_backend"] not in VALID_PITCH_BACKENDS:
        raise ConfigurationError(
            f"features.pitch_backend must be one of: {', '.join(VALID_PITCH_BACKENDS)}.",
            detail=f"got {ft['pitch_backend']!r}",
        )
    if ft["f0_min_hz"] <= 0:
        raise ConfigurationError(
            "features.f0_min_hz must be positive.", detail=f"got {ft['f0_min_hz']}"
        )
    if ft["f0_max_hz"] <= ft["f0_min_hz"]:
        raise ConfigurationError(
            "features.f0_max_hz must be greater than features.f0_min_hz.",
            detail=f"min={ft['f0_min_hz']} max={ft['f0_max_hz']}",
        )
    if ft["channel_policy"] not in VALID_FEATURE_CHANNEL_POLICIES:
        raise ConfigurationError(
            f"features.channel_policy must be one of: {', '.join(VALID_FEATURE_CHANNEL_POLICIES)}.",
            detail=f"got {ft['channel_policy']!r}",
        )
    if ft["silence_threshold_db"] >= 0:
        raise ConfigurationError(
            "features.silence_threshold_db must be negative (dB below full scale).",
            detail=f"got {ft['silence_threshold_db']}",
        )
    if ft["min_silence_ms"] <= 0:
        raise ConfigurationError(
            "features.min_silence_ms must be positive.", detail=f"got {ft['min_silence_ms']}"
        )
    if ft["cache_policy"] not in VALID_FEATURE_CACHE_POLICIES:
        raise ConfigurationError(
            f"features.cache_policy must be one of: {', '.join(VALID_FEATURE_CACHE_POLICIES)}.",
            detail=f"got {ft['cache_policy']!r}",
        )
    if ft["max_feature_seconds"] <= 0:
        raise ConfigurationError(
            "features.max_feature_seconds must be positive.",
            detail=f"got {ft['max_feature_seconds']}",
        )
    features_dir = _resolve_dir(root, ft["features_dir"], "features.features_dir")
    return FeaturesSettings(
        frame_ms=float(ft["frame_ms"]),
        hop_ms=float(ft["hop_ms"]),
        n_mfcc=int(ft["n_mfcc"]),
        spectral_rolloff_p=float(ft["spectral_rolloff_p"]),
        pitch_backend=str(ft["pitch_backend"]),
        f0_min_hz=float(ft["f0_min_hz"]),
        f0_max_hz=float(ft["f0_max_hz"]),
        channel_policy=str(ft["channel_policy"]),
        silence_threshold_db=float(ft["silence_threshold_db"]),
        min_silence_ms=float(ft["min_silence_ms"]),
        features_dir=features_dir,
        feature_manifest_filename=str(ft["feature_manifest_filename"]),
        cache_policy=str(ft["cache_policy"]),
        max_feature_seconds=float(ft["max_feature_seconds"]),
        schema_version=str(ft["schema_version"]),
    )


def _build_detection(dt: dict[str, Any], root: Path) -> DetectionSettings:
    detectors = list(dict.fromkeys(dt["enabled_detectors"]))
    if not detectors:
        raise ConfigurationError("detection.enabled_detectors must not be empty.")
    invalid_detectors = [d for d in detectors if d not in VALID_DETECTION_CATEGORIES]
    if invalid_detectors:
        raise ConfigurationError(
            f"Unsupported detector(s): {', '.join(invalid_detectors)}.",
            detail=f"supported: {', '.join(VALID_DETECTION_CATEGORIES)}",
        )
    if dt["pause_silence_threshold_db"] > 0:
        raise ConfigurationError(
            "detection.pause_silence_threshold_db must be negative or zero (dB scale).",
            detail=f"got {dt['pause_silence_threshold_db']}",
        )
    if dt["pause_min_duration_sec"] <= 0:
        raise ConfigurationError(
            "detection.pause_min_duration_sec must be positive.",
            detail=f"got {dt['pause_min_duration_sec']}",
        )
    if dt["pause_min_difference_sec"] < 0:
        raise ConfigurationError(
            "detection.pause_min_difference_sec must be non-negative.",
            detail=f"got {dt['pause_min_difference_sec']}",
        )
    if dt["tempo_ratio_min_deviation"] <= 0:
        raise ConfigurationError(
            "detection.tempo_ratio_min_deviation must be positive.",
            detail=f"got {dt['tempo_ratio_min_deviation']}",
        )
    if dt["tempo_min_duration_sec"] <= 0:
        raise ConfigurationError(
            "detection.tempo_min_duration_sec must be positive.",
            detail=f"got {dt['tempo_min_duration_sec']}",
        )
    if dt["alignment_metric"] not in VALID_ALIGNMENT_METRICS:
        raise ConfigurationError(
            f"detection.alignment_metric must be one of: {', '.join(VALID_ALIGNMENT_METRICS)}.",
            detail=f"got {dt['alignment_metric']!r}",
        )
    if dt["tempo_slope_window_sec"] <= 0:
        raise ConfigurationError(
            "detection.tempo_slope_window_sec must be positive.",
            detail=f"got {dt['tempo_slope_window_sec']}",
        )
    if not 0.0 <= dt["tempo_max_inactive_fraction"] <= 1.0:
        raise ConfigurationError(
            "detection.tempo_max_inactive_fraction must be within [0, 1].",
            detail=f"got {dt['tempo_max_inactive_fraction']}",
        )
    if not 0.0 <= dt["pause_max_voiced_fraction"] <= 1.0:
        raise ConfigurationError(
            "detection.pause_max_voiced_fraction must be within [0, 1].",
            detail=f"got {dt['pause_max_voiced_fraction']}",
        )
    if dt["energy_reduction_threshold_db"] >= 0:
        raise ConfigurationError(
            "detection.energy_reduction_threshold_db must be negative (dB drop).",
            detail=f"got {dt['energy_reduction_threshold_db']}",
        )
    if dt["energy_reduction_min_duration_sec"] <= 0:
        raise ConfigurationError(
            "detection.energy_reduction_min_duration_sec must be positive.",
            detail=f"got {dt['energy_reduction_min_duration_sec']}",
        )
    if dt["merge_gap_sec"] < 0:
        raise ConfigurationError(
            "detection.merge_gap_sec must be non-negative.",
            detail=f"got {dt['merge_gap_sec']}",
        )
    if dt["min_candidate_duration_sec"] <= 0:
        raise ConfigurationError(
            "detection.min_candidate_duration_sec must be positive.",
            detail=f"got {dt['min_candidate_duration_sec']}",
        )
    if dt["cache_policy"] not in VALID_DETECTION_CACHE_POLICIES:
        raise ConfigurationError(
            f"detection.cache_policy must be one of: {', '.join(VALID_DETECTION_CACHE_POLICIES)}.",
            detail=f"got {dt['cache_policy']!r}",
        )

    detection_dir = _resolve_dir(root, dt["detection_dir"], "detection.detection_dir")
    return DetectionSettings(
        enabled_detectors=tuple(detectors),
        pause_silence_threshold_db=float(dt["pause_silence_threshold_db"]),
        pause_min_duration_sec=float(dt["pause_min_duration_sec"]),
        pause_min_difference_sec=float(dt["pause_min_difference_sec"]),
        tempo_ratio_min_deviation=float(dt["tempo_ratio_min_deviation"]),
        tempo_min_duration_sec=float(dt["tempo_min_duration_sec"]),
        alignment_metric=str(dt["alignment_metric"]),
        alignment_exclude_energy_coefficient=bool(dt["alignment_exclude_energy_coefficient"]),
        tempo_slope_window_sec=float(dt["tempo_slope_window_sec"]),
        tempo_max_inactive_fraction=float(dt["tempo_max_inactive_fraction"]),
        pause_max_voiced_fraction=float(dt["pause_max_voiced_fraction"]),
        energy_reduction_threshold_db=float(dt["energy_reduction_threshold_db"]),
        energy_reduction_min_duration_sec=float(dt["energy_reduction_min_duration_sec"]),
        merge_gap_sec=float(dt["merge_gap_sec"]),
        min_candidate_duration_sec=float(dt["min_candidate_duration_sec"]),
        detection_dir=detection_dir,
        detection_manifest_filename=str(dt["detection_manifest_filename"]),
        cache_policy=str(dt["cache_policy"]),
        schema_version=str(dt["schema_version"]),
    )


def _build_evaluation(ev: dict[str, Any], root: Path) -> EvaluationSettings:
    default_split = str(ev["default_split"]).lower()
    if default_split not in VALID_EVALUATION_SPLITS:
        raise ConfigurationError(
            f"evaluation.default_split must be one of: {', '.join(VALID_EVALUATION_SPLITS)}.",
            detail=f"got {default_split!r}",
        )
    iou_thresh = float(ev["matching_iou_threshold"])
    if not (0.0 < iou_thresh <= 1.0):
        raise ConfigurationError(
            "evaluation.matching_iou_threshold must be in (0.0, 1.0].",
            detail=f"got {iou_thresh}",
        )
    loc_thresholds = [float(x) for x in ev["localization_iou_thresholds"]]
    if not loc_thresholds:
        raise ConfigurationError("evaluation.localization_iou_thresholds must not be empty.")
    for t in loc_thresholds:
        if not (0.0 < t <= 1.0):
            raise ConfigurationError(
                "All thresholds in evaluation.localization_iou_thresholds must be in (0.0, 1.0].",
                detail=f"got {t}",
            )
    zero_denom = str(ev["zero_denominator_policy"]).lower()
    if zero_denom not in VALID_ZERO_DENOMINATOR_POLICIES:
        raise ConfigurationError(
            f"evaluation.zero_denominator_policy must be one of: "
            f"{', '.join(VALID_ZERO_DENOMINATOR_POLICIES)}.",
            detail=f"got {zero_denom!r}",
        )
    reps = int(ev["bootstrap_replicates"])
    if reps < 0:
        raise ConfigurationError(
            "evaluation.bootstrap_replicates must be non-negative.",
            detail=f"got {reps}",
        )
    evaluations_dir = _resolve_dir(root, ev["evaluations_dir"], "evaluation.evaluations_dir")
    return EvaluationSettings(
        default_split=default_split,
        matching_iou_threshold=iou_thresh,
        localization_iou_thresholds=tuple(sorted(loc_thresholds)),
        require_category_match=bool(ev["require_category_match"]),
        zero_denominator_policy=zero_denom,
        bootstrap_replicates=reps,
        bootstrap_seed=int(ev["bootstrap_seed"]),
        include_identity_controls=bool(ev["include_identity_controls"]),
        evaluations_dir=evaluations_dir,
        evaluations_manifest_filename=str(ev["evaluations_manifest_filename"]),
        schema_version=str(ev["schema_version"]),
    )


def _choice(section: str, key: str, value: Any, valid: tuple[str, ...]) -> str:
    text = str(value).strip().lower()
    if text not in valid:
        raise ConfigurationError(
            f"{section}.{key} must be one of: {', '.join(valid)}.", detail=f"got {value!r}"
        )
    return text


def _choice_list(
    section: str, key: str, values: list[str], valid: tuple[str, ...]
) -> tuple[str, ...]:
    items = list(dict.fromkeys(v.strip().lower() for v in values))
    if not items:
        raise ConfigurationError(f"{section}.{key} must not be empty.")
    bad = [i for i in items if i not in valid]
    if bad:
        raise ConfigurationError(
            f"Unsupported value(s) in {section}.{key}: {', '.join(bad)}.",
            detail=f"supported: {', '.join(valid)}",
        )
    return tuple(items)


def _build_explanation(ex: dict[str, Any], root: Path) -> ExplanationSettings:
    transcript_formats = [
        t if t.startswith(".") else f".{t}" for t in (str(x) for x in ex["transcript_formats"])
    ]
    context = float(ex["playback_context_sec"])
    if not (0.0 <= context <= 10.0):
        raise ConfigurationError(
            "explanation.playback_context_sec must be between 0 and 10 seconds.",
            detail=f"got {context}",
        )
    max_excerpt = float(ex["max_excerpt_sec"])
    if max_excerpt <= 0:
        raise ConfigurationError(
            "explanation.max_excerpt_sec must be positive.", detail=f"got {max_excerpt}"
        )
    similarity = float(ex["transcript_similarity_warning"])
    if not (0.0 <= similarity <= 1.0):
        raise ConfigurationError(
            "explanation.transcript_similarity_warning must be in [0, 1].",
            detail=f"got {similarity}",
        )
    ratio = float(ex["alignment_duration_ratio_warning"])
    if ratio < 1.0:
        raise ConfigurationError(
            "explanation.alignment_duration_ratio_warning must be at least 1.0.",
            detail=f"got {ratio}",
        )
    return ExplanationSettings(
        default_page=_choice("explanation", "default_page", ex["default_page"], VALID_UI_PAGES),
        analysis_modes=_choice_list(
            "explanation", "analysis_modes", ex["analysis_modes"], VALID_ANALYSIS_MODES
        ),
        verbosity=_choice("explanation", "verbosity", ex["verbosity"], VALID_VERBOSITY),
        sort_policy=_choice("explanation", "sort_policy", ex["sort_policy"], VALID_SORT_POLICIES),
        playback_context_sec=context,
        max_excerpt_sec=max_excerpt,
        show_technical_evidence=bool(ex["show_technical_evidence"]),
        transcript_formats=_choice_list(
            "explanation", "transcript_formats", transcript_formats, VALID_TRANSCRIPT_FORMATS
        ),
        transcript_similarity_warning=similarity,
        alignment_duration_ratio_warning=ratio,
        upload_persistence=_choice(
            "explanation", "upload_persistence", ex["upload_persistence"], VALID_UPLOAD_PERSISTENCE
        ),
        reports_dir=_resolve_dir(root, ex["reports_dir"], "explanation.reports_dir"),
        export_formats=_choice_list(
            "explanation", "export_formats", ex["export_formats"], VALID_EXPORT_FORMATS
        ),
        save_reports_automatically=bool(ex["save_reports_automatically"]),
        show_benchmark_context=bool(ex["show_benchmark_context"]),
    )


def _build(values: dict[str, dict[str, Any]], root: Path, config_file: Path | None) -> Config:
    app, storage, audio, ui, diag, ds, ft, dt, ev, ex = (
        values["app"],
        values["storage"],
        values["audio"],
        values["ui"],
        values["diagnostics"],
        values["dataset"],
        values["features"],
        values["detection"],
        values["evaluation"],
        values["explanation"],
    )

    environment = str(app["environment"]).lower()
    if environment not in VALID_ENVIRONMENTS:
        raise ConfigurationError(
            f"app.environment must be one of: {', '.join(VALID_ENVIRONMENTS)}.",
            detail=f"got {app['environment']!r}",
        )
    log_level = str(app["log_level"]).upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigurationError(
            f"app.log_level must be one of: {', '.join(VALID_LOG_LEVELS)}.",
            detail=f"got {app['log_level']!r}",
        )
    layout = str(ui["layout"]).lower()
    if layout not in VALID_LAYOUTS:
        raise ConfigurationError(
            f"ui.layout must be one of: {', '.join(VALID_LAYOUTS)}.", detail=f"got {ui['layout']!r}"
        )

    for key, minimum in (("max_upload_mb", 1), ("min_sample_rate_hz", 1), ("max_channels", 1)):
        if audio[key] < minimum:
            raise ConfigurationError(
                f"audio.{key} must be at least {minimum}.", detail=f"got {audio[key]}"
            )
    extensions = list(dict.fromkeys(audio["allowed_extensions"]))
    if not extensions:
        raise ConfigurationError("audio.allowed_extensions must not be empty.")
    unsupported = [e for e in extensions if e not in EXTENSION_FORMATS]
    if unsupported:
        raise ConfigurationError(
            f"Unsupported audio extension(s): {', '.join(unsupported)}.",
            detail=f"supported: {', '.join(EXTENSION_FORMATS)}",
        )

    if environment == "production" and diag["expose_error_details"]:
        raise ConfigurationError(
            "diagnostics.expose_error_details must be false when app.environment is production."
        )

    dirs = {
        key: _resolve_dir(root, storage[key], f"storage.{key}")
        for key in ("reference_dir", "generated_dir", "metadata_dir", "output_dir")
    }
    seen: dict[Path, str] = {}
    for key, path in dirs.items():
        if path in seen:
            raise ConfigurationError(
                f"storage.{key} and storage.{seen[path]} must be different directories.",
                detail=str(path),
            )
        seen[path] = key

    dataset = _build_dataset(ds)
    features = _build_features(ft, root)
    detection = _build_detection(dt, root)
    evaluation = _build_evaluation(ev, root)
    explanation = _build_explanation(ex, root)

    return Config(
        app=AppSettings(str(app["name"]), environment, log_level),
        storage=StorageSettings(**dirs),
        audio=AudioSettings(
            tuple(extensions),
            int(audio["max_upload_mb"]),
            int(audio["min_sample_rate_hz"]),
            int(audio["max_channels"]),
            bool(audio["reject_non_finite_samples"]),
        ),
        ui=UISettings(str(ui["page_title"]), layout, bool(ui["show_roadmap"])),
        diagnostics=DiagnosticsSettings(
            bool(diag["detailed_logging"]), bool(diag["expose_error_details"])
        ),
        dataset=dataset,
        features=features,
        detection=detection,
        evaluation=evaluation,
        explanation=explanation,
        project_root=root,
        config_file=config_file,
    )


def load_config(
    config_path: Path | str | None = None,
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    env: Mapping[str, str] | None = None,
    project_root: Path | str | None = None,
) -> Config:
    """Load, merge, and validate configuration.

    Args:
        config_path: Explicit YAML file. Must exist if given.
        overrides: Highest-precedence settings, shaped like the YAML file.
        env: Environment mapping. Defaults to ``os.environ`` layered over the project's ``.env``.
            When supplied, ``.env`` is not read (useful for hermetic tests).
        project_root: Base for relative paths. See module docstring for the default.

    Raises:
        ConfigurationError: for any missing, malformed, or invalid setting.
    """
    env_source: Mapping[str, str]
    if env is None:
        root_for_env = _resolve_root(project_root, os.environ)
        env_source = {**_parse_dotenv(root_for_env / DOTENV_FILE), **os.environ}
    else:
        env_source = env
    root = _resolve_root(project_root, env_source)
    config_file = _resolve_config_file(config_path, env_source, root)

    merged = defaults()
    if config_file is not None:
        _apply_layer(merged, _load_file(config_file), f"config file '{config_file.name}'")
    env_layer: dict[str, dict[str, Any]] = {}
    for f in _FIELDS:
        if f.env in env_source:
            env_layer.setdefault(f.section, {})[f.key] = env_source[f.env]
    _apply_layer(merged, env_layer, "environment variables")
    if overrides:
        _apply_layer(merged, overrides, "runtime overrides")

    return _build(merged, root, config_file)
