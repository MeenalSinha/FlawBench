"""Export selected samples for sharing. Only items whose source rights say redistribution is
'permitted' are exported; nothing is published and an existing destination is never overwritten."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from flawbench.config import Config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.io import write_jsonl
from flawbench.dataset.params import is_identifier
from flawbench.dataset.registry import SourceRegistry
from flawbench.exceptions import DatasetError
from flawbench.paths import ProjectPaths, ensure_writable_directory, resolve_within


@dataclass(frozen=True)
class ExportResult:
    destination: Path
    exported_samples: tuple[str, ...]
    refused: dict[str, str]


def export_samples(
    config: Config,
    name: str,
    sample_ids: list[str] | None = None,
    paths: ProjectPaths | None = None,
) -> ExportResult:
    if not is_identifier(name):
        raise DatasetError("Export name may contain only letters, digits, '.', '_' and '-'.")
    paths = paths or ProjectPaths(config)
    registry = SourceRegistry(config, paths)
    samples = DatasetGenerator(config, paths, registry).samples()
    if sample_ids is not None:
        known = {s.sample_id for s in samples}
        missing = [i for i in sample_ids if i not in known]
        if missing:
            raise DatasetError(f"Unknown sample id(s): {', '.join(missing)}.")
        samples = [s for s in samples if s.sample_id in set(sample_ids)]
    sources = {s.source_id: s for s in registry.all()}
    refused: dict[str, str] = {}
    chosen = []
    for s in samples:
        src = sources.get(s.source_id)
        if src is None or src.redistribution != "permitted":
            refused[s.sample_id] = "source redistribution is not 'permitted'"
        else:
            chosen.append(s)
    if not chosen:
        raise DatasetError(
            "Nothing can be exported: no selected sample has redistribution permitted."
        )

    dest = resolve_within(paths.exports_dir, name)
    if dest.exists():
        raise DatasetError("The export destination already exists; refusing to overwrite it.")
    ensure_writable_directory(paths.exports_dir)
    dest.mkdir()
    used = {s.source_id for s in chosen}
    for s in chosen:
        target = dest / "generated" / Path(str(s.data["generated"]["path"])).name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(resolve_within(paths.generated_dir, str(s.data["generated"]["path"])), target)
    for sid in used:
        ref = sources[sid]
        target = dest / "reference" / Path(str(ref.data["audio_path"])).name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(registry.audio_path(ref), target)
    write_jsonl(dest / "sources.jsonl", [sources[i].to_dict() for i in sorted(used)])
    write_jsonl(dest / "manifest.jsonl", [s.to_dict() for s in chosen])
    (dest / "EXPORT.json").write_text(
        json.dumps({"samples": [s.sample_id for s in chosen], "refused": refused}, indent=2) + "\n"
    )
    return ExportResult(dest, tuple(s.sample_id for s in chosen), refused)
