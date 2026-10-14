"""Registry of reference recordings (``sources.jsonl``).

Registration validates the audio with the Phase 1 loader, requires a transcript and rights
metadata, computes a SHA-256, and refuses duplicates. Registered paths are stored relative to the
configured reference directory; a file outside it is only accepted when copying is enabled
(``dataset.copy_reference`` or an explicit argument), and an existing file is never overwritten.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flawbench import __version__
from flawbench.audio.loader import LoadedAudio, load_audio
from flawbench.config import Config
from flawbench.dataset.io import read_jsonl, sha256_file, write_jsonl
from flawbench.dataset.params import is_identifier
from flawbench.dataset.schema import (
    RECORD_SOURCE,
    REDISTRIBUTION_VALUES,
    SCHEMA_VERSION,
    SOURCE_TYPES,
    SourceRecord,
)
from flawbench.exceptions import AudioInputError, DatasetError, RegistryError
from flawbench.logging_config import get_logger
from flawbench.paths import ProjectPaths, resolve_within

logger = get_logger("dataset.registry")
SOURCE_ID_PREFIX = "src-"
SOURCE_ID_HEX_LENGTH = 12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SourceRegistry:
    def __init__(self, config: Config, paths: ProjectPaths | None = None) -> None:
        self.config = config
        self.paths = paths or ProjectPaths(config)

    # ------------------------------------------------------------------ reading
    def all(self) -> list[SourceRecord]:
        return [
            SourceRecord.from_dict(r)
            for r in read_jsonl(self.paths.sources_file, what="source registry")
        ]

    def get(self, source_id: str) -> SourceRecord:
        for record in self.all():
            if record.source_id == source_id:
                return record
        raise RegistryError(f"Source '{source_id}' is not registered.")

    def audio_path(self, record: SourceRecord) -> Path:
        """Resolve the stored relative path, refusing anything outside the reference directory."""
        return resolve_within(self.paths.reference_dir, str(record.data["audio_path"]))

    def verify(self, record: SourceRecord) -> list[str]:
        """Return integrity problems (empty list means the file exists and matches its checksum)."""
        try:
            path = self.audio_path(record)
        except DatasetError as exc:
            return [exc.user_message]
        except Exception as exc:  # StorageError for traversal
            return [str(exc)]
        if not path.is_file():
            return [f"audio file for '{record.source_id}' is missing"]
        if sha256_file(path) != record.sha256:
            return [f"checksum mismatch for '{record.source_id}' (file changed since registration)"]
        return []

    def load_audio(self, record: SourceRecord) -> LoadedAudio:
        """Load a source after verifying its checksum. Raises if the file changed."""
        problems = self.verify(record)
        if problems:
            raise RegistryError(problems[0])
        return load_audio(self.audio_path(record), self.config.audio)

    # -------------------------------------------------------------- registration
    def register(
        self,
        audio_path: Path | str,
        *,
        transcript_text: str | None = None,
        transcript_file: Path | str | None = None,
        source_type: str,
        attribution: str,
        license: str,  # noqa: A002 - field name in the schema
        rights_confirmed: bool,
        redistribution: str = "unknown",
        source_id: str | None = None,
        copy: bool | None = None,
        notes: str | None = None,
    ) -> SourceRecord:
        """Register a reference recording. Nothing is written unless every check passes."""
        if source_type not in SOURCE_TYPES:
            raise RegistryError(f"source_type must be one of: {', '.join(SOURCE_TYPES)}.")
        if redistribution not in REDISTRIBUTION_VALUES:
            raise RegistryError(
                f"redistribution must be one of: {', '.join(REDISTRIBUTION_VALUES)}."
            )
        if not attribution.strip():
            raise RegistryError("Attribution (origin of the recording) is required.")
        if not license.strip():
            raise RegistryError("Licence or permission information is required.")
        if not rights_confirmed:
            raise RegistryError(
                "Rights must be confirmed: you need permission to process and modify the recording."
            )
        text = self._read_transcript(transcript_text, transcript_file)

        source = Path(audio_path).expanduser()
        try:
            source = source.resolve(strict=True)
        except OSError as exc:
            raise RegistryError("The audio file was not found.", detail=f"{source}: {exc}") from exc
        try:
            audio = load_audio(source, self.config.audio)
        except AudioInputError as exc:
            raise RegistryError(
                f"Audio validation failed: {exc.user_message}", detail=exc.detail
            ) from exc
        checksum = sha256_file(source)

        existing = self.all()
        for rec in existing:
            if rec.sha256 == checksum:
                raise RegistryError(f"This recording is already registered as '{rec.source_id}'.")
        chosen_id = source_id or f"{SOURCE_ID_PREFIX}{checksum[:SOURCE_ID_HEX_LENGTH]}"
        if not is_identifier(chosen_id):
            raise RegistryError("source_id may contain only letters, digits, '.', '_' and '-'.")
        if any(rec.source_id == chosen_id for rec in existing):
            raise RegistryError(f"Source id '{chosen_id}' is already registered.")

        do_copy = self.config.dataset.copy_reference if copy is None else copy
        reference_dir = self.paths.reference_dir.resolve()
        copied_to: Path | None = None
        if source.is_relative_to(reference_dir):
            relative = source.relative_to(reference_dir).as_posix()
        elif do_copy:
            relative = source.name
            destination = resolve_within(reference_dir, relative)
            if destination.exists():
                raise RegistryError(
                    "A file with this name already exists in the reference directory; "
                    "refusing to overwrite it.",
                    detail=str(destination),
                )
            reference_dir.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            copied_to = destination
            if sha256_file(destination) != checksum:
                destination.unlink(missing_ok=True)
                raise RegistryError("The copied reference file failed checksum verification.")
        else:
            raise RegistryError(
                "The audio file is outside the reference directory. Place it there, or enable "
                "copying (--copy / dataset.copy_reference)."
            )

        meta = audio.metadata
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_type": RECORD_SOURCE,
            "source_id": chosen_id,
            "original_filename": source.name,
            "audio_path": relative,
            "transcript_text": text,
            "transcript_file": Path(transcript_file).name if transcript_file else None,
            "sha256": checksum,
            "sample_rate": meta.sample_rate,
            "channels": meta.channels,
            "frames": meta.frames,
            "duration_seconds": meta.duration_seconds,
            "container_format": meta.container_format,
            "subtype": meta.subtype,
            "source_type": source_type,
            "attribution": attribution.strip(),
            "license": license.strip(),
            "rights": {
                "use_confirmed": True,
                "redistribution": redistribution,
                "cleared_for_public_redistribution": redistribution == "permitted",
            },
            "notes": notes,
            "registered_utc": utc_now(),
            "flawbench_version": __version__,
        }
        try:
            write_jsonl(self.paths.sources_file, [r.to_dict() for r in existing] + [record])
        except OSError as exc:
            if copied_to is not None:
                copied_to.unlink(missing_ok=True)
            raise RegistryError(
                "The source registry could not be written.", detail=str(exc)
            ) from exc
        logger.info("Registered source %s (%s frames)", chosen_id, meta.frames)
        return SourceRecord.from_dict(record)

    @staticmethod
    def _read_transcript(text: str | None, file: Path | str | None) -> str:
        if (text is None) == (file is None):
            raise RegistryError("Provide exactly one of transcript text or a transcript file.")
        if file is not None:
            try:
                text = Path(file).expanduser().read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise RegistryError(
                    "The transcript file could not be read.", detail=str(exc)
                ) from exc
        assert text is not None
        if not text.strip():
            raise RegistryError("The transcript is empty.")
        return text.strip()


def ensure_relative_posix(value: str) -> bool:
    """True when ``value`` is a safe relative POSIX-style path (no absolute path or ``..``)."""
    if not value or value.startswith(("/", "\\")) or ":" in value.split("/")[0]:
        return False
    return ".." not in value.replace("\\", "/").split("/") and not os.path.isabs(value)
