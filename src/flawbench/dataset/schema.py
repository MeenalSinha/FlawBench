"""Record schemas and the timeline convention.

Timeline convention
-------------------
Every interval is half-open, ``[start, end)``, measured from the start of the recording it
refers to. Integer sample indices are authoritative; seconds are derived as ``index / sample_rate``.
``output_interval`` always refers to the *generated* recording. ``source_interval`` (when a flaw
has one) refers to the *reference* recording and is stored separately.

See ``docs/dataset_schema.md`` for the full field reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flawbench.exceptions import DatasetError

SCHEMA_VERSION = "1.0"
RECORD_SOURCE = "source"
RECORD_SAMPLE = "sample"
STATUS_COMMITTED = "committed"
TIMELINE_CONVENTION = "half_open_samples_and_seconds"

SOURCE_TYPES = ("self_recorded", "public_domain", "licensed", "other_permissioned")
REDISTRIBUTION_VALUES = ("permitted", "prohibited", "unknown")

SOURCE_REQUIRED = (
    "schema_version",
    "record_type",
    "source_id",
    "original_filename",
    "audio_path",
    "transcript_text",
    "sha256",
    "sample_rate",
    "channels",
    "frames",
    "duration_seconds",
    "source_type",
    "attribution",
    "license",
    "rights",
    "registered_utc",
    "flawbench_version",
)
SAMPLE_REQUIRED = (
    "schema_version",
    "record_type",
    "status",
    "dataset_id",
    "sample_id",
    "source_id",
    "reference",
    "generated",
    "transcript_ref",
    "annotation",
    "seed",
    "generated_utc",
    "software",
    "config_fingerprint",
    "plan_fingerprint",
    "transformation",
)
REFERENCE_REQUIRED = ("path", "sha256", "frames", "duration_seconds")
GENERATED_REQUIRED = (
    "path",
    "sha256",
    "frames",
    "duration_seconds",
    "sample_rate",
    "channels",
    "subtype",
)
ANNOTATION_REQUIRED = (
    "flaw_type",
    "severity",
    "parameters",
    "effective_parameters",
    "source_interval",
    "source_anchor",
    "output_interval",
    "timeline",
)
INTERVAL_KEYS = ("start_sample", "end_sample", "start_sec", "end_sec")


def interval_dict(start: int, end: int, sample_rate: int) -> dict[str, Any]:
    """Build an interval record from integer sample bounds (seconds derived from samples)."""
    return {
        "start_sample": int(start),
        "end_sample": int(end),
        "start_sec": start / sample_rate,
        "end_sec": end / sample_rate,
    }


def anchor_dict(sample: int, sample_rate: int) -> dict[str, Any]:
    return {"sample": int(sample), "sec": sample / sample_rate}


def _require(data: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise DatasetError(f"Record is missing required field(s) in {where}: {', '.join(missing)}.")


@dataclass(frozen=True)
class SourceRecord:
    """A registered reference recording. Thin typed wrapper over the JSON record."""

    data: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRecord:
        _require(data, SOURCE_REQUIRED, "source record")
        if data["record_type"] != RECORD_SOURCE:
            raise DatasetError("Record is not a source record.")
        _require(data["rights"], ("use_confirmed", "redistribution"), "source rights")
        return cls(dict(data))

    @property
    def source_id(self) -> str:
        return str(self.data["source_id"])

    @property
    def sha256(self) -> str:
        return str(self.data["sha256"])

    @property
    def redistribution(self) -> str:
        return str(self.data["rights"]["redistribution"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class SampleRecord:
    """A generated variant with its ground-truth annotation."""

    data: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampleRecord:
        _require(data, SAMPLE_REQUIRED, "sample record")
        if data["record_type"] != RECORD_SAMPLE:
            raise DatasetError("Record is not a sample record.")
        _require(data["reference"], REFERENCE_REQUIRED, "sample.reference")
        _require(data["generated"], GENERATED_REQUIRED, "sample.generated")
        _require(data["annotation"], ANNOTATION_REQUIRED, "sample.annotation")
        _require(data["annotation"]["output_interval"], INTERVAL_KEYS, "output_interval")
        return cls(dict(data))

    @property
    def sample_id(self) -> str:
        return str(self.data["sample_id"])

    @property
    def source_id(self) -> str:
        return str(self.data["source_id"])

    @property
    def flaw_type(self) -> str:
        return str(self.data["annotation"]["flaw_type"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)
