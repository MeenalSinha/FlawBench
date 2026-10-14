"""Phase 4 Label-Free Detection Input Builder.

Ensures strict ground-truth isolation by extracting only acoustic features,
timing information, and checksums from audio data without reading or exposing
any ground-truth flaw annotations.
"""

from __future__ import annotations

from flawbench.dataset.io import read_jsonl_lenient
from flawbench.detection.schema import DetectionInput
from flawbench.exceptions import DetectionInputError
from flawbench.features.comparison import compare_features
from flawbench.features.extractor import FeatureResult
from flawbench.features.pipeline import FeaturePipeline


def build_detection_input_from_features(
    ref_features: FeatureResult,
    var_features: FeatureResult,
    sample_id: str = "test-sample",
    source_id: str = "test-source",
) -> DetectionInput:
    """Construct a label-free DetectionInput directly from FeatureResults.

    Useful for hermetic testing, synthetic audio evaluation, and programmatic detection.
    """
    comparison = compare_features(ref_features, var_features)
    return DetectionInput(
        sample_id=sample_id,
        source_id=source_id,
        ref_features=ref_features,
        var_features=var_features,
        comparison=comparison,
        sample_rate=var_features.metadata.sample_rate,
        ref_sample_rate=ref_features.metadata.sample_rate,
        duration_seconds=var_features.metadata.duration_seconds,
        ref_duration_seconds=ref_features.metadata.duration_seconds,
        ref_checksum=ref_features.metadata.audio_sha256,
        variant_checksum=var_features.metadata.audio_sha256,
    )


def build_detection_input(
    sample_id: str,
    feature_pipeline: FeaturePipeline,
    force_extract: bool = False,
) -> DetectionInput:
    """Resolve a dataset sample and construct a label-free DetectionInput.

    Only retrieves:
    - sample_id
    - source_id
    - reference audio features
    - variant audio features
    - differential comparison

    Strictly ignores and does not touch any flaw annotations or ground-truth intervals.
    """
    manifest_file = feature_pipeline.paths.manifest_file
    if not manifest_file.is_file():
        raise DetectionInputError("Dataset manifest does not exist.", detail=str(manifest_file))

    rows, problems = read_jsonl_lenient(manifest_file)
    if problems:
        raise DetectionInputError("Dataset manifest contains malformed lines.")

    target_source_id: str | None = None
    for _, raw in rows:
        if raw.get("sample_id") == sample_id:
            target_source_id = str(raw.get("source_id"))
            break

    if target_source_id is None:
        raise DetectionInputError(f"Sample '{sample_id}' not found in dataset manifest.")

    try:
        ref_feat = feature_pipeline.extract_reference(target_source_id, force=force_extract)
        var_feat = feature_pipeline.extract_variant(sample_id, force=force_extract)
        comparison = compare_features(ref_feat, var_feat)
    except Exception as exc:
        raise DetectionInputError(
            f"Failed to extract features or comparison for sample '{sample_id}': {exc}",
            detail=str(exc),
        ) from exc

    return DetectionInput(
        sample_id=sample_id,
        source_id=target_source_id,
        ref_features=ref_feat,
        var_features=var_feat,
        comparison=comparison,
        sample_rate=var_feat.metadata.sample_rate,
        ref_sample_rate=ref_feat.metadata.sample_rate,
        duration_seconds=var_feat.metadata.duration_seconds,
        ref_duration_seconds=ref_feat.metadata.duration_seconds,
        ref_checksum=ref_feat.metadata.audio_sha256,
        variant_checksum=var_feat.metadata.audio_sha256,
    )
