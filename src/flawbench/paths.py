"""Project path management.

All modules obtain directories from :class:`ProjectPaths` (built from a validated
:class:`~flawbench.config.Config`) or receive an explicit path by injection. Nothing else in the
package defines directory locations.

Roles:

* *input* (read-only): the reference-audio directory. Never created or written by FlawBench.
* *output* (writable): generated audio, metadata, and outputs (``reports/`` and ``figures/``).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from flawbench.config import Config
from flawbench.exceptions import StorageError

REPORTS_SUBDIR = "reports"
FIGURES_SUBDIR = "figures"
EXPORTS_SUBDIR = "exports"


@dataclass(frozen=True)
class DirectoryStatus:
    """Result of inspecting one configured directory (inspection never creates anything)."""

    name: str
    role: str  # "input" or "output"
    path: Path
    exists: bool
    accessible: bool
    problem: str | None = None


def ensure_writable_directory(path: Path) -> Path:
    """Create ``path`` if needed and verify that files can really be written there.

    Raises:
        StorageError: if the path is a file, cannot be created, or is not writable.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, NotADirectoryError) as exc:
        raise StorageError("A storage location is not a directory.", detail=str(path)) from exc
    except OSError as exc:
        raise StorageError(
            "A storage directory could not be created.", detail=f"{path}: {exc}"
        ) from exc
    try:
        with tempfile.TemporaryFile(dir=path):
            pass
    except OSError as exc:
        raise StorageError("A storage directory is not writable.", detail=f"{path}: {exc}") from exc
    return path


def resolve_within(base: Path, *parts: str | os.PathLike[str]) -> Path:
    """Join ``parts`` onto ``base`` and refuse results that escape ``base`` (e.g. ``../x``)."""
    base_resolved = base.resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise StorageError(
            "The requested location is outside the permitted directory.",
            detail=f"{candidate} not under {base_resolved}",
        )
    return candidate


class ProjectPaths:
    """Resolved storage locations derived from configuration."""

    def __init__(self, config: Config) -> None:
        storage = config.storage
        self.project_root: Path = config.project_root
        self.reference_dir: Path = storage.reference_dir
        self.generated_dir: Path = storage.generated_dir
        self.metadata_dir: Path = storage.metadata_dir
        self.output_dir: Path = storage.output_dir
        self.reports_dir: Path = self.output_dir / REPORTS_SUBDIR
        self.figures_dir: Path = self.output_dir / FIGURES_SUBDIR
        self.exports_dir: Path = self.output_dir / EXPORTS_SUBDIR
        ds = config.dataset
        self.dataset_id: str = ds.dataset_id
        self.sources_file: Path = self.metadata_dir / ds.sources_filename
        self.manifest_file: Path = self.metadata_dir / ds.manifest_filename
        self.runs_file: Path = self.metadata_dir / ds.runs_filename
        self.splits_file: Path = self.metadata_dir / ds.splits_filename
        # Phase 3: feature storage
        ft = config.features
        self.features_dir: Path = ft.features_dir
        self.feature_manifest_file: Path = self.features_dir / ft.feature_manifest_filename
        # Phase 4: detection storage
        dt = config.detection
        self.detection_dir: Path = dt.detection_dir
        self.detection_manifest_file: Path = self.detection_dir / dt.detection_manifest_filename
        # Phase 5: evaluation storage
        ev = config.evaluation
        self.evaluations_dir: Path = ev.evaluations_dir
        self.evaluations_manifest_file: Path = (
            self.evaluations_dir / ev.evaluations_manifest_filename
        )

    @property
    def dataset_generated_dir(self) -> Path:
        """Directory holding this dataset's generated audio (created lazily by the generator)."""
        return self.generated_dir / self.dataset_id

    @property
    def writable_directories(self) -> dict[str, Path]:
        return {
            "generated": self.generated_dir,
            "metadata": self.metadata_dir,
            "reports": self.reports_dir,
            "figures": self.figures_dir,
            "features": self.features_dir,
            "detections": self.detection_dir,
            "evaluations": self.evaluations_dir,
        }

    def ensure_output_directories(self) -> list[Path]:
        """Create and verify every writable directory. Never touches the input directory."""
        return [ensure_writable_directory(p) for p in self.writable_directories.values()]

    def check_input_directory(self) -> Path:
        """Verify the reference directory exists and is readable (it is never created)."""
        path = self.reference_dir
        if not path.exists():
            raise StorageError("The reference-audio directory does not exist.", detail=str(path))
        if not path.is_dir():
            raise StorageError("The reference-audio location is not a directory.", detail=str(path))
        if not os.access(path, os.R_OK | os.X_OK):
            raise StorageError("The reference-audio directory is not readable.", detail=str(path))
        return path

    def inspect(self) -> tuple[DirectoryStatus, ...]:
        """Report the real state of every configured directory without modifying anything."""
        statuses = [self._inspect("reference", "input", self.reference_dir)]
        statuses += [
            self._inspect(name, "output", path) for name, path in self.writable_directories.items()
        ]
        return tuple(statuses)

    @staticmethod
    def _inspect(name: str, role: str, path: Path) -> DirectoryStatus:
        if not path.exists():
            return DirectoryStatus(name, role, path, False, False, "directory does not exist")
        if not path.is_dir():
            return DirectoryStatus(name, role, path, True, False, "path is not a directory")
        needed = os.R_OK | os.X_OK if role == "input" else os.R_OK | os.W_OK | os.X_OK
        if not os.access(path, needed):
            return DirectoryStatus(name, role, path, True, False, "insufficient permissions")
        return DirectoryStatus(name, role, path, True, True)
