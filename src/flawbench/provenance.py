"""Run provenance helpers (foundation for reproducible dataset generation and evaluation).

Phase 2 records the output of :func:`build_provenance` next to every generated artifact, together
with its own transformation parameters and random seed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

import soundfile as sf

from flawbench import __version__
from flawbench.config import Config

TRACKED_PACKAGES = ("numpy", "soundfile", "PyYAML", "streamlit", "audiotsm", "librosa", "plotly")


def config_fingerprint(config: Config, *, sections: tuple[str, ...] | None = None) -> str:
    """Stable SHA-256 of the effective configuration.

    With ``sections`` only those top-level sections are hashed (for example ``("dataset",)``),
    which excludes machine-specific paths and gives a portable behavior fingerprint.
    """
    data = config.to_dict()
    if sections is not None:
        data = {name: data[name] for name in sections}
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def collect_software_versions() -> dict[str, str]:
    """Collect system and library software versions for provenance tracking."""
    versions = package_versions()
    versions["python"] = sys.version.split()[0]
    versions["libsndfile"] = getattr(sf, "__libsndfile_version__", "unknown")
    versions["platform"] = platform.platform()
    versions["flawbench"] = __version__
    return versions


def build_provenance(config: Config, *, seed: int | None = None) -> dict[str, Any]:
    """Collect software versions, platform, configuration snapshot and an optional random seed."""
    return {
        "flawbench_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libsndfile": sf.__libsndfile_version__,
        "packages": package_versions(),
        "config_sha256": config_fingerprint(config),
        "config": config.to_dict(),
        "random_seed": seed,
    }
