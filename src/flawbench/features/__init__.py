"""Acoustic feature-extraction package (Phase 3).

Public API:
    ``extract_features(audio, sample_rate, feat_cfg)`` → :class:`FeatureResult`
    ``FeatureResult`` — time-indexed acoustic features from one audio recording.
    ``FeaturePipeline`` — connects extractor to Phase 2 registry and dataset.
    ``FeatureStore`` — persists and validates feature artifacts.
"""

from flawbench.features.comparison import ComparisonResult, compare_features
from flawbench.features.extractor import FeatureResult, extract_features
from flawbench.features.pipeline import FeaturePipeline
from flawbench.features.schema import AudioRole
from flawbench.features.store import FeatureStore

__all__ = [
    "AudioRole",
    "ComparisonResult",
    "FeaturePipeline",
    "FeatureResult",
    "FeatureStore",
    "compare_features",
    "extract_features",
]
