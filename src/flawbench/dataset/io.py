"""Checksums, atomic writes, and JSON Lines helpers.

Writers replace whole files atomically (write to a temporary sibling, fsync, ``os.replace``) so a
crash can never leave a half-written manifest that parses as valid. The engine assumes a single
writer process at a time (documented limitation; no locking is implemented).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from flawbench.exceptions import DatasetError

_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> str:
    """Deterministic JSON used for fingerprints and identifiers."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(data: Any) -> str:
    return sha256_bytes(canonical_json(data).encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=True) for r in records]
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))


def read_jsonl_lenient(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    """Return ``([(line_number, record)], problems)``; malformed lines are reported, not raised."""
    records: list[tuple[int, dict[str, Any]]] = []
    problems: list[str] = []
    if not path.exists():
        return records, problems
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return records, [f"cannot read file: {exc}"]
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(value, dict):
            problems.append(f"line {number}: record is not a JSON object")
            continue
        records.append((number, value))
    return records, problems


def read_jsonl(path: Path, *, what: str) -> list[dict[str, Any]]:
    """Strict reader: any malformed line raises :class:`DatasetError`."""
    records, problems = read_jsonl_lenient(path)
    if problems:
        raise DatasetError(
            f"The {what} file is corrupt and must be repaired before use.",
            detail=f"{path}: {'; '.join(problems)}",
        )
    return [r for _, r in records]
