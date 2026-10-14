"""Source-level split assignment, persisted so rebuilds never reshuffle existing sources.

All variants of a source inherit the source's split. Initial assignment shuffles the sorted source
ids with a seeded generator and allocates counts by largest remainder. Later-registered sources are
assigned by a seeded hash draw and never move existing ones.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flawbench.config import Config
from flawbench.dataset.io import atomic_write_text
from flawbench.dataset.registry import SourceRegistry, utc_now
from flawbench.exceptions import DatasetError
from flawbench.paths import ProjectPaths

#: Fewer sources than non-empty splits means at least one split is empty.
MIN_SOURCES_FOR_STATISTICS = 10


@dataclass
class SplitResult:
    assignments: dict[str, str]
    warnings: list[str] = field(default_factory=list)

    def counts(self, names: list[str]) -> dict[str, int]:
        return {n: sum(1 for v in self.assignments.values() if v == n) for n in names}


def _allocate(n: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {k: n * v for k, v in ratios.items()}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    left = n - sum(base.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:left]:
        base[k] += 1
    return base


def _hash_draw(seed: int, source_id: str, ratios: dict[str, float]) -> str:
    digest = hashlib.sha256(f"{seed}|{source_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    acc = 0.0
    for name in sorted(ratios):
        acc += ratios[name]
        if u < acc:
            return name
    return sorted(ratios)[-1]


def build_splits(
    config: Config, paths: ProjectPaths | None = None, *, force: bool = False
) -> SplitResult:
    paths = paths or ProjectPaths(config)
    ds = config.dataset
    ids = sorted(s.source_id for s in SourceRegistry(config, paths).all())
    if not ids:
        raise DatasetError("No sources are registered; nothing to split.")
    existing: dict[str, str] = {}
    if paths.splits_file.exists() and not force:
        try:
            existing = dict(
                json.loads(paths.splits_file.read_text(encoding="utf-8"))["assignments"]
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DatasetError("The splits file is corrupt.", detail=str(exc)) from exc
        unknown = {v for v in existing.values() if v not in ds.split_ratios}
        if unknown:
            raise DatasetError(
                f"Existing assignments use split(s) no longer configured: {sorted(unknown)}. "
                "Use --force to rebuild."
            )
    assignments = {k: v for k, v in existing.items() if k in ids}
    new_ids = [i for i in ids if i not in assignments]
    if not assignments:
        rng = np.random.default_rng(ds.split_seed)
        shuffled = list(new_ids)
        rng.shuffle(shuffled)
        quota = _allocate(len(shuffled), ds.split_ratios)
        cursor = 0
        for name in sorted(quota):
            for sid in shuffled[cursor : cursor + quota[name]]:
                assignments[sid] = name
            cursor += quota[name]
    else:
        for sid in new_ids:
            assignments[sid] = _hash_draw(ds.split_seed, sid, ds.split_ratios)

    result = SplitResult(dict(sorted(assignments.items())))
    wanted = [n for n, r in ds.split_ratios.items() if r > 0]
    empty = [n for n in wanted if result.counts(wanted)[n] == 0]
    if empty:
        result.warnings.append(
            f"Split(s) {empty} received no sources: {len(ids)} source recording(s) cannot fill "
            "every configured split."
        )
    if len(ids) < MIN_SOURCES_FOR_STATISTICS:
        result.warnings.append(
            f"Only {len(ids)} source recording(s): these splits cannot support claims about "
            "generalization."
        )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "seed": ds.split_seed,
        "ratios": ds.split_ratios,
        "created_utc": utc_now(),
        "assignments": result.assignments,
        "warnings": result.warnings,
    }
    atomic_write_text(paths.splits_file, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return result
