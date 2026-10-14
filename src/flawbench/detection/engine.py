"""Phase 4 Flaw Detection Engine.

Orchestrates configured flaw detectors, executes inference over label-free inputs,
and post-processes candidate intervals into consolidated predictions.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from flawbench.config import DetectionSettings
from flawbench.detection.detectors import (
    EnergyReductionDetector,
    PauseDetector,
    TempoDetector,
)
from flawbench.detection.postprocess import postprocess_predictions
from flawbench.detection.schema import DetectionInput, DetectionResult, FlawPrediction


class DetectionEngine:
    """Coordinates evidence-based flaw detectors to produce consolidated predictions."""

    detector_version: str = "4.0"

    def __init__(self, config: DetectionSettings) -> None:
        self.config = config
        self.detectors: list[Any] = []
        for cat in config.enabled_detectors:
            if cat == "pause_insertion":
                self.detectors.append(PauseDetector(config))
            elif cat == "tempo_change":
                self.detectors.append(TempoDetector(config))
            elif cat == "energy_reduction":
                self.detectors.append(EnergyReductionDetector(config))

    def config_fingerprint(self) -> str:
        """Compute deterministic fingerprint of effective detection settings."""
        cfg = self.config
        raw = (
            f"{','.join(cfg.enabled_detectors)}_"
            f"{cfg.pause_silence_threshold_db}_{cfg.pause_min_duration_sec}_{cfg.pause_min_difference_sec}_"
            f"{cfg.tempo_ratio_min_deviation}_{cfg.tempo_min_duration_sec}_"
            f"{cfg.energy_reduction_threshold_db}_{cfg.energy_reduction_min_duration_sec}_"
            f"{cfg.merge_gap_sec}_{cfg.min_candidate_duration_sec}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def detect(self, inp: DetectionInput) -> DetectionResult:
        """Run all enabled detectors and consolidate predictions on the variant timeline."""
        raw_candidates: list[FlawPrediction] = []
        for detector in self.detectors:
            candidates = detector.detect(inp)
            raw_candidates.extend(candidates)

        sample_rate = inp.sample_rate
        total_samples = int(round(inp.duration_seconds * sample_rate))

        consolidated = postprocess_predictions(
            raw_candidates,
            merge_gap_sec=self.config.merge_gap_sec,
            min_candidate_duration_sec=self.config.min_candidate_duration_sec,
            total_samples=total_samples,
            sample_rate=sample_rate,
        )

        return DetectionResult(
            schema_version=self.config.schema_version,
            run_id=f"detrun_{uuid.uuid4().hex[:8]}",
            sample_id=inp.sample_id,
            source_id=inp.source_id,
            ref_artifact_id=inp.ref_features.metadata.artifact_id,
            variant_artifact_id=inp.var_features.metadata.artifact_id,
            ref_checksum=inp.ref_checksum,
            variant_checksum=inp.variant_checksum,
            detector_version=self.detector_version,
            config_fingerprint=self.config_fingerprint(),
            created_at=datetime.now(timezone.utc).isoformat(),
            sample_rate=sample_rate,
            total_samples=total_samples,
            duration_seconds=inp.duration_seconds,
            predictions=consolidated,
            raw_candidates_count=len(raw_candidates),
            metadata={
                "enabled_detectors": list(self.config.enabled_detectors),
                "cache_policy": self.config.cache_policy,
            },
        )
