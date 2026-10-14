"""Real component status, gathered by inspecting the running application (no canned messages)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from flawbench.config import Config
from flawbench.paths import ProjectPaths


@dataclass(frozen=True)
class StatusItem:
    label: str
    ok: bool
    message: str


def audio_backend_status() -> StatusItem:
    """Check that the audio decoder imports and report its real versions."""
    try:
        sf = importlib.import_module("soundfile")
        importlib.import_module("flawbench.audio.loader")
    except ImportError as exc:
        return StatusItem("Audio loader", False, f"Unavailable: {exc}")
    return StatusItem(
        "Audio loader", True, f"soundfile {sf.__version__}, libsndfile {sf.__libsndfile_version__}"
    )


def dataset_status(config: Config, paths: ProjectPaths) -> StatusItem:
    """Count registered sources and samples from the real registry files."""
    from flawbench.dataset.generator import DatasetGenerator
    from flawbench.dataset.registry import SourceRegistry
    from flawbench.exceptions import FlawBenchError

    try:
        registry = SourceRegistry(config, paths)
        n_sources = len(registry.all())
        n_samples = len(DatasetGenerator(config, paths, registry).samples())
    except FlawBenchError as exc:
        return StatusItem("Dataset registry", False, exc.user_message)
    note = " (empty)" if n_sources == 0 and n_samples == 0 else ""
    return StatusItem(
        "Dataset registry", True, f"{n_sources} source(s), {n_samples} sample(s){note}"
    )


def features_status(config: Config, paths: ProjectPaths) -> StatusItem:
    """Check feature extraction engine and count stored feature artifacts."""
    try:
        lr = importlib.import_module("librosa")
        from flawbench.features.store import FeatureStore

        store = FeatureStore(paths.features_dir, config.features.feature_manifest_filename)
        entries = store.list_manifest()
        n_artifacts = len(entries)
    except Exception as exc:
        return StatusItem("Feature engine", False, f"Unavailable: {exc}")
    note = " (empty)" if n_artifacts == 0 else ""
    return StatusItem(
        "Feature engine", True, f"librosa {lr.__version__}, {n_artifacts} artifact(s){note}"
    )


def collect_status(
    config: Config, paths: ProjectPaths, *, expose_details: bool = False
) -> list[StatusItem]:
    """Build the status list shown in the UI and by ``python -m flawbench check``.

    Directory paths are shown only when ``expose_details`` is true.
    """
    source = config.config_file.name if config.config_file else "defaults and environment only"
    items = [
        StatusItem("Configuration", True, f"Loaded ({source})"),
        StatusItem("Environment", True, config.app.environment),
        StatusItem("Log level", True, config.app.log_level),
    ]
    for d in paths.inspect():
        suffix = f" at {d.path}" if expose_details else ""
        if d.accessible:
            msg = ("Readable" if d.role == "input" else "Readable and writable") + suffix
        else:
            msg = f"{d.problem or 'Not accessible'}{suffix}"
        items.append(StatusItem(f"{d.name.capitalize()} directory ({d.role})", d.accessible, msg))
    items.append(audio_backend_status())
    items.append(dataset_status(config, paths))
    items.append(features_status(config, paths))
    return items

