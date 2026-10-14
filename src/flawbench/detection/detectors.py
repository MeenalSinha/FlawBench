"""Phase 4 Flaw Detectors.

Implements three independent, evidence-based speech flaw detectors:
1. PauseDetector: Identifies inserted pauses and extended silences ungrounded in reference.
2. TempoDetector: Identifies local speaking rate acceleration and deceleration.
3. EnergyReductionDetector: Identifies sustained regional signal energy attenuation.

All detectors consume ONLY label-free DetectionInput and return FlawPrediction candidates.
"""

from __future__ import annotations

import math
import uuid

import numpy as np

from flawbench.config import DetectionSettings
from flawbench.detection.schema import DetectionInput, FlawPrediction


class PauseDetector:
    """Detects inserted or abnormally prolonged pauses in the variant recording."""

    detector_id: str = "pause_detector_v4"
    flaw_type: str = "pause_insertion"

    def __init__(self, config: DetectionSettings) -> None:
        self.config = config

    def detect(self, inp: DetectionInput) -> list[FlawPrediction]:
        """Detect pause insertions on the variant timeline."""
        comp = inp.comparison
        n_points = len(comp.warp_path)
        if n_points == 0:
            return []

        sample_rate = inp.sample_rate
        total_samples = int(round(inp.duration_seconds * sample_rate))
        silence_thresh = self.config.pause_silence_threshold_db
        min_duration = self.config.pause_min_duration_sec
        min_diff = self.config.pause_min_difference_sec

        var_times = comp.var_times_sec
        ref_times = comp.ref_times_sec
        var_rms = comp.aligned_var_rms_db
        ref_rms = comp.aligned_ref_rms_db

        predictions: list[FlawPrediction] = []

        # Find continuous regions in warp path where variant advances while reference is frozen,
        # or where variant RMS is below silence threshold while reference was active.
        in_pause = False
        start_idx = 0

        for i in range(n_points):
            ref_frozen = i > 0 and comp.warp_path[i, 0] == comp.warp_path[i - 1, 0]
            var_quiet = var_rms[i] <= silence_thresh
            # Active in reference if ref_rms > silence_thresh
            ref_active = ref_rms[i] > silence_thresh + 3.0

            # Region qualifies if variant is quiet or ref is frozen
            is_pause_frame = (var_quiet and (ref_active or ref_frozen)) or (
                ref_frozen and var_quiet
            )

            if is_pause_frame and not in_pause:
                in_pause = True
                start_idx = max(0, i - 1) if ref_frozen else i
            elif (not is_pause_frame or i == n_points - 1) and in_pause:
                in_pause = False
                end_idx = i

                v_start_sec = float(var_times[start_idx])
                v_end_sec = float(var_times[end_idx])
                var_dur = v_end_sec - v_start_sec

                r_start_sec = float(ref_times[start_idx])
                r_end_sec = float(ref_times[end_idx])
                ref_dur = max(0.0, r_end_sec - r_start_sec)

                dur_diff = var_dur - ref_dur

                if var_dur >= min_duration and dur_diff >= min_diff:
                    # Calculate sample boundaries
                    start_sample = max(0, int(math.floor(v_start_sec * sample_rate)))
                    end_sample = min(
                        total_samples,
                        max(start_sample + 1, int(math.ceil(v_end_sec * sample_rate))),
                    )

                    true_start_sec = start_sample / sample_rate
                    true_end_sec = end_sample / sample_rate
                    true_dur_sec = true_end_sec - true_start_sec

                    region_var_rms = float(np.mean(var_rms[start_idx : end_idx + 1]))
                    frozen_count = sum(
                        1
                        for k in range(start_idx + 1, end_idx + 1)
                        if comp.warp_path[k, 0] == comp.warp_path[k - 1, 0]
                    )
                    frozen_fraction = float(frozen_count / max(1, end_idx - start_idx))

                    # Evidence strength calculation [0.0, 1.0]
                    # Scaled by duration difference and silence depth
                    dur_score = min(1.0, dur_diff / 1.0)
                    silence_score = min(1.0, max(0.0, (silence_thresh - region_var_rms) / 20.0))
                    strength = round(
                        float(0.6 * dur_score + 0.4 * max(frozen_fraction, silence_score)), 3
                    )
                    strength = max(0.1, min(1.0, strength))

                    pred_id = f"pred_pause_{uuid.uuid4().hex[:8]}"
                    predictions.append(
                        FlawPrediction(
                            prediction_id=pred_id,
                            flaw_type=self.flaw_type,
                            start_sample=start_sample,
                            end_sample=end_sample,
                            start_sec=round(true_start_sec, 4),
                            end_sec=round(true_end_sec, 4),
                            duration_sec=round(true_dur_sec, 4),
                            ref_start_sec=round(r_start_sec, 4),
                            ref_end_sec=round(r_end_sec, 4),
                            evidence={
                                "variant_pause_duration_sec": round(var_dur, 4),
                                "reference_duration_sec": round(ref_dur, 4),
                                "duration_difference_sec": round(dur_diff, 4),
                                "mean_variant_rms_db": round(region_var_rms, 2),
                                "threshold_silence_db": silence_thresh,
                                "min_duration_sec": min_duration,
                                "ref_frozen_fraction": round(frozen_fraction, 3),
                            },
                            evidence_strength=strength,
                            detector_id=self.detector_id,
                            decision_threshold=min_duration,
                        )
                    )

        return predictions


class TempoDetector:
    """Detects local tempo deviations (speedup or slowdown) relative to reference."""

    detector_id: str = "tempo_detector_v4"
    flaw_type: str = "tempo_change"

    def __init__(self, config: DetectionSettings) -> None:
        self.config = config

    def detect(self, inp: DetectionInput) -> list[FlawPrediction]:
        """Detect local tempo changes on the variant timeline."""
        comp = inp.comparison
        n_points = len(comp.warp_path)
        if n_points == 0:
            return []

        sample_rate = inp.sample_rate
        total_samples = int(round(inp.duration_seconds * sample_rate))
        min_dev = self.config.tempo_ratio_min_deviation
        min_duration = self.config.tempo_min_duration_sec

        var_times = comp.var_times_sec
        ref_times = comp.ref_times_sec
        local_slope = comp.local_slope
        var_rms = comp.aligned_var_rms_db

        predictions: list[FlawPrediction] = []

        # Local slope = d_var / d_ref.
        # Relative rate r = 1.0 / local_slope.
        # If r < 1.0 - min_dev: speech is slower in variant (stretched).
        # If r > 1.0 + min_dev: speech is faster in variant (compressed).
        # We exclude pure pause regions (where local_slope == 10.0 or var_rms <= -40 dB).
        in_tempo = False
        start_idx = 0
        current_direction: str | None = None

        for i in range(n_points):
            slope = local_slope[i]
            is_pause_artifact = slope >= 9.0 or var_rms[i] <= -40.0

            direction: str | None = None
            if not is_pause_artifact and np.isfinite(slope) and slope > 0:
                rel_rate = 1.0 / slope
                if rel_rate < 1.0 - min_dev:
                    direction = "slower"
                elif rel_rate > 1.0 + min_dev:
                    direction = "faster"

            if direction is not None and not in_tempo:
                in_tempo = True
                start_idx = i
                current_direction = direction
            elif in_tempo and (direction != current_direction or i == n_points - 1):
                in_tempo = False
                end_idx = i

                v_start_sec = float(var_times[start_idx])
                v_end_sec = float(var_times[end_idx])
                duration = v_end_sec - v_start_sec

                if duration >= min_duration:
                    region_slopes = local_slope[start_idx : end_idx + 1]
                    valid_slopes = region_slopes[np.isfinite(region_slopes) & (region_slopes < 9.0)]
                    if len(valid_slopes) > 0:
                        mean_slope = float(np.mean(valid_slopes))
                        mean_rate = float(1.0 / max(1e-4, mean_slope))
                        deviation = abs(mean_rate - 1.0)

                        start_sample = max(0, int(math.floor(v_start_sec * sample_rate)))
                        end_sample = min(
                            total_samples,
                            max(start_sample + 1, int(math.ceil(v_end_sec * sample_rate))),
                        )

                        true_start_sec = start_sample / sample_rate
                        true_end_sec = end_sample / sample_rate
                        true_dur_sec = true_end_sec - true_start_sec

                        r_start_sec = float(ref_times[start_idx])
                        r_end_sec = float(ref_times[end_idx])

                        # Evidence strength [0.0, 1.0]
                        dev_score = min(1.0, deviation / 0.5)
                        dur_score = min(1.0, duration / 1.5)
                        strength = round(float(0.6 * dev_score + 0.4 * dur_score), 3)
                        strength = max(0.1, min(1.0, strength))

                        pred_id = f"pred_tempo_{uuid.uuid4().hex[:8]}"
                        predictions.append(
                            FlawPrediction(
                                prediction_id=pred_id,
                                flaw_type=self.flaw_type,
                                start_sample=start_sample,
                                end_sample=end_sample,
                                start_sec=round(true_start_sec, 4),
                                end_sec=round(true_end_sec, 4),
                                duration_sec=round(true_dur_sec, 4),
                                ref_start_sec=round(r_start_sec, 4),
                                ref_end_sec=round(r_end_sec, 4),
                                evidence={
                                    "direction": current_direction,
                                    "estimated_relative_rate": round(mean_rate, 3),
                                    "mean_warp_slope": round(mean_slope, 3),
                                    "rate_deviation": round(deviation, 3),
                                    "threshold_min_deviation": min_dev,
                                    "duration_sec": round(duration, 4),
                                },
                                evidence_strength=strength,
                                detector_id=self.detector_id,
                                decision_threshold=min_dev,
                            )
                        )

                # Reset or continue if immediately transitioning
                if direction is not None:
                    in_tempo = True
                    start_idx = i
                    current_direction = direction
                else:
                    current_direction = None

        return predictions


class EnergyReductionDetector:
    """Detects sustained regional signal energy reduction relative to reference."""

    detector_id: str = "energy_reduction_detector_v4"
    flaw_type: str = "energy_reduction"

    def __init__(self, config: DetectionSettings) -> None:
        self.config = config

    def detect(self, inp: DetectionInput) -> list[FlawPrediction]:
        """Detect sustained energy drops on the variant timeline."""
        comp = inp.comparison
        n_points = len(comp.warp_path)
        if n_points == 0:
            return []

        sample_rate = inp.sample_rate
        total_samples = int(round(inp.duration_seconds * sample_rate))
        threshold_db = self.config.energy_reduction_threshold_db  # negative, e.g. -4.0 dB
        min_duration = self.config.energy_reduction_min_duration_sec

        var_times = comp.var_times_sec
        ref_times = comp.ref_times_sec
        delta_rms_db = comp.delta_rms_db
        ref_rms = comp.aligned_ref_rms_db
        var_rms = comp.aligned_var_rms_db

        predictions: list[FlawPrediction] = []

        in_drop = False
        start_idx = 0

        for i in range(n_points):
            val = delta_rms_db[i]
            # Must be a valid finite number
            # Exclude ambient silence regions where reference itself is quiet (e.g. < -45 dB)
            # and variant is quiet, because ambient background drop is not speech energy reduction
            is_speech_in_ref = ref_rms[i] > -45.0
            is_energy_drop = np.isfinite(val) and (val <= threshold_db) and is_speech_in_ref

            if is_energy_drop and not in_drop:
                in_drop = True
                start_idx = i
            elif (not is_energy_drop or i == n_points - 1) and in_drop:
                in_drop = False
                end_idx = i

                v_start_sec = float(var_times[start_idx])
                v_end_sec = float(var_times[end_idx])
                duration = v_end_sec - v_start_sec

                if duration >= min_duration:
                    region_drops = delta_rms_db[start_idx : end_idx + 1]
                    valid_drops = region_drops[np.isfinite(region_drops)]
                    if len(valid_drops) > 0:
                        mean_drop = float(np.mean(valid_drops))
                        min_drop = float(np.min(valid_drops))
                        mean_ref = float(np.mean(ref_rms[start_idx : end_idx + 1]))
                        mean_var = float(np.mean(var_rms[start_idx : end_idx + 1]))

                        start_sample = max(0, int(math.floor(v_start_sec * sample_rate)))
                        end_sample = min(
                            total_samples,
                            max(start_sample + 1, int(math.ceil(v_end_sec * sample_rate))),
                        )

                        true_start_sec = start_sample / sample_rate
                        true_end_sec = end_sample / sample_rate
                        true_dur_sec = true_end_sec - true_start_sec

                        r_start_sec = float(ref_times[start_idx])
                        r_end_sec = float(ref_times[end_idx])

                        # Evidence strength [0.0, 1.0]
                        # Scaled by drop magnitude below threshold and duration
                        drop_excess = abs(mean_drop) - abs(threshold_db)
                        drop_score = min(1.0, max(0.0, drop_excess / 15.0))
                        dur_score = min(1.0, duration / 1.5)
                        strength = round(float(0.6 * drop_score + 0.4 * dur_score), 3)
                        strength = max(0.1, min(1.0, strength))

                        pred_id = f"pred_energy_{uuid.uuid4().hex[:8]}"
                        predictions.append(
                            FlawPrediction(
                                prediction_id=pred_id,
                                flaw_type=self.flaw_type,
                                start_sample=start_sample,
                                end_sample=end_sample,
                                start_sec=round(true_start_sec, 4),
                                end_sec=round(true_end_sec, 4),
                                duration_sec=round(true_dur_sec, 4),
                                ref_start_sec=round(r_start_sec, 4),
                                ref_end_sec=round(r_end_sec, 4),
                                evidence={
                                    "mean_delta_rms_db": round(mean_drop, 2),
                                    "min_delta_rms_db": round(min_drop, 2),
                                    "mean_ref_rms_db": round(mean_ref, 2),
                                    "mean_var_rms_db": round(mean_var, 2),
                                    "threshold_gain_db": threshold_db,
                                    "duration_sec": round(duration, 4),
                                },
                                evidence_strength=strength,
                                detector_id=self.detector_id,
                                decision_threshold=threshold_db,
                            )
                        )

        return predictions
