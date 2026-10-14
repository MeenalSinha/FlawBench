"""Shared fixtures. All audio is synthetic test material generated inside tmp_path.

The sine tones below exist only to exercise file loading. They are not speech and are never
written outside pytest's temporary directories.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from flawbench.config import Config, load_config


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def config(project_root: Path) -> Config:
    """Hermetic default configuration rooted in a temporary project."""
    return load_config(project_root=project_root, env={})


@pytest.fixture
def make_tone(tmp_path: Path) -> Callable[..., Path]:
    """Write a synthetic test tone and return its path."""

    def _make(
        name: str = "tone.wav",
        *,
        sample_rate: int = 16000,
        seconds: float = 0.5,
        channels: int = 1,
        subtype: str | None = None,
        data: np.ndarray | None = None,
    ) -> Path:
        path = tmp_path / name
        if data is None:
            t = np.arange(int(sample_rate * seconds)) / sample_rate
            tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
        sf.write(path, data, sample_rate, subtype=subtype)
        return path

    return _make


# ---------------------------------------------------------------- Phase 2 fixtures
# All audio below is SYNTHETIC TEST MATERIAL (noisy harmonic tones). It is not speech, is created
# inside pytest temporary directories only, and is never presented as reference data.

from flawbench.dataset.registry import SourceRegistry  # noqa: E402
from flawbench.dataset.schema import SourceRecord  # noqa: E402

SYN_RATE = 16000


def synthetic_signal(seconds: float = 6.0, seed: int = 0, rate: int = SYN_RATE) -> np.ndarray:
    """Deterministic float32 signal shaped (samples,), amplitude below 0.5."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    env = 0.25 + 0.2 * np.sin(2 * np.pi * 1.3 * t) ** 2
    tone = np.sin(2 * np.pi * (150 + 10 * seed) * t) + 0.5 * np.sin(2 * np.pi * 450 * t)
    return (env * tone / 1.5 + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.fixture
def ds_root(tmp_path: Path) -> Path:
    root = tmp_path / "dsproj"
    (root / "data" / "reference").mkdir(parents=True)
    return root


@pytest.fixture
def ds_config(ds_root: Path) -> Config:
    return load_config(project_root=ds_root, env={})


def write_fixture(root: Path, name: str, *, seed: int = 0, seconds: float = 6.0) -> Path:
    path = root / "data" / "reference" / name
    sf.write(path, synthetic_signal(seconds, seed), SYN_RATE, subtype="PCM_16")
    return path


REGISTER_DEFAULTS = {
    "transcript_text": "synthetic test fixture",
    "source_type": "self_recorded",
    "attribution": "generated inside the test-suite",
    "license": "CC0-1.0 (synthetic)",
    "rights_confirmed": True,
    "redistribution": "permitted",
}


@pytest.fixture
def registry(ds_config: Config) -> SourceRegistry:
    return SourceRegistry(ds_config)


@pytest.fixture
def source(ds_root: Path, registry: SourceRegistry) -> SourceRecord:
    path = write_fixture(ds_root, "fixture_a.wav", seed=1)
    return registry.register(path, **REGISTER_DEFAULTS)
