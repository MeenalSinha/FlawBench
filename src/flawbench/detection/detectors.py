"""Phase 4 Flaw Detectors.

Implements three independent, evidence-based speech flaw detectors:
1. PauseDetector: Identifies inserted pauses and extended silences ungrounded in reference.
2. TempoDetector: Identifies local speaking rate acceleration and deceleration.
3. EnergyReductionDetector: Identifies sustained regional signal energy attenuation.

All detectors consume ONLY label-free DetectionInput and return FlawPrediction candidates.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from numpy.typing import NDArray

from flawbench.config import DetectionSettings
from flawbench.detection.schema import DetectionInput, FlawPrediction


def _reference_span(r_start_sec: float, r_end_sec: float) -> tuple[float | None, float | None]:
    """Reference span for non-pause detectors, or (None, None) if it is degenerate.

    A non-pause event must correspond to a positive-length reference interval. When the DTW path
    holds the reference position fixed across the whole region (for example, an energy drop that
    coincides with an inserted pause), there is no valid reference interval; reporting a
    zero-length one makes the whole detection result fail validation, so the correspondence is
    reported as unavailable instead.
    """
    if r_end_sec > r_start_sec:
        return round(r_start_sec, 4), round(r_end_sec, 4)
    return None, None


def _prediction_id(kind: str, inp: DetectionInput, start_sample: int, end_sample: int) -> str:
    """Deterministic prediction identifier.

    Derived from the sample, the variant checksum and the predicted interval, so the same audio
    and the same detector output always carry the same identifier (no random component).
    """
    raw = f"{inp.sample_id}|{inp.variant_checksum}|{kind}|{start_sample}|{end_sample}"
    return f"pred_{kind}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]}"


def _inactive_mask(inp: DetectionInput, silence_threshold_db: float) -> NDArray[np.bool_]:
    """Per warp-path point: True where the aligned variant frame is silent AND unvoiced.

    Silence is treated as pause evidence, never as energy-reduction or tempo evidence.
    """
    comp = inp.comparison
    var_idx = comp.warp_path[:, 1]
    voiced = np.asarray(inp.var_features.voiced_flag, dtype=bool)[var_idx]
    quiet = comp.aligned_var_rms_db <= silence_threshold_db
    return np.asarray(quiet & ~voiced, dtype=bool)


def windowed_warp_slope(
    warp_path: NDArray[np.int64],
    inactive: NDArray[np.bool_],
    *,
    hop_sec: float,
    window_sec: float,
    max_inactive_fraction: float,
) -> NDArray[np.float64]:
    """Local DTW warp slope (variant frames advanced per reference frame) over a time window.

    The window spans ``window_sec`` of the *variant* timeline, measured in frames rather than in
    path points: a staircase DTW path has between one and several path points per frame, so a
    fixed number of path points gives a quantized, unstable slope. The result is NaN
    (unavailable) where the window holds no reference progress (frozen reference), too little
    variant progress, or more silent frames than ``max_inactive_fraction``. NaN is never
    interpreted as evidence.
    """
    n = len(warp_path)
    ref_idx = warp_path[:, 0]
    var_idx = warp_path[:, 1]
    half = max(1, int(round(window_sec / max(hop_sec, 1e-9) / 2.0)))
    lo = np.searchsorted(var_idx, var_idx - half, side="left")
    hi = np.searchsorted(var_idx, var_idx + half, side="right") - 1
    d_var = (var_idx[hi] - var_idx[lo]).astype(np.float64)
    d_ref = (ref_idx[hi] - ref_idx[lo]).astype(np.float64)
    csum = np.concatenate([[0], np.cumsum(inactive.astype(np.int64))])
    count = (hi - lo + 1).astype(np.float64)
    inactive_frac = (csum[hi + 1] - csum[lo]) / np.maximum(count, 1.0)
    valid = (d_ref > 0) & (d_var >= half) & (inactive_frac <= max_inactive_fraction)
    slope = np.full(n, np.nan, dtype=np.float64)
    slope[valid] = d_var[valid] / d_ref[valid]
    return slope


class PauseDetector:
    """Detects inserted or abnormally prolonged pauses in the variant recording."""

    detector_id: str = "pause_detector_v5"
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
        voiced_aligned = np.asarray(inp.var_features.voiced_flag, dtype=bool)[comp.warp_path[:, 1]]

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

                region_voiced = float(np.mean(voiced_aligned[start_idx : end_idx + 1]))
                if (
                    var_dur >= min_duration
                    and dur_diff >= min_diff
                    and region_voiced <= self.config.pause_max_voiced_fraction
                ):
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

                    pred_id = _prediction_id("pause", inp, start_sample, end_sample)
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
                                "variant_voiced_fraction": round(region_voiced, 3),
                            },
                            evidence_strength=strength,
                            detector_id=self.detector_id,
                            decision_threshold=min_duration,
                        )
                    )

        return predictions


class TempoDetector:
    """Detects local tempo deviations (speedup or slowdown) relative to reference."""

    detector_id: str = "tempo_detector_v5"
    flaw_type: str = "tempo_change"

    def __init__(self, config: DetectionSettings) -> None:
        self.config = config

    def detect(self, inp: DetectionInput) -> list[FlawPrediction]:
        """Detect local tempo changes on the variant timeline.

        Convention: the window slope is ``d_variant / d_reference`` (variant frames advanced per
        reference frame). The relative speaking rate is its inverse: ``rate < 1`` means the
        variant is slower (stretched), ``rate > 1`` means faster (compressed). A window with no
        reference progress, or dominated by silence, yields no slope (NaN) and therefore no
        tempo evidence.
        """
        comp = inp.comparison
        n_points = len(comp.warp_path)
        if n_points == 0 or inp.var_features.num_frames < 2:
            return []

        sample_rate = inp.sample_rate
        total_samples = int(round(inp.duration_seconds * sample_rate))
        min_dev = self.config.tempo_ratio_min_deviation
        min_duration = self.config.tempo_min_duration_sec

        var_times = comp.var_times_sec
        ref_times = comp.ref_times_sec
        hop_sec = float(np.median(np.diff(inp.var_features.center_times_sec)))
        inactive = _inactive_mask(inp, self.config.pause_silence_threshold_db)
        slope = windowed_warp_slope(
            comp.warp_path,
            inactive,
            hop_sec=hop_sec,
            window_sec=self.config.tempo_slope_window_sec,
            max_inactive_fraction=self.config.tempo_max_inactive_fraction,
        )

        directions: list[str | None] = []
        for i in range(n_points):
            direction: str | None = None
            s_i = slope[i]
            if np.isfinite(s_i) and s_i > 0:
                rel_rate = 1.0 / s_i
                if rel_rate < 1.0 - min_dev:
                    direction = "slower"
                elif rel_rate > 1.0 + min_dev:
                    direction = "faster"
            directions.append(direction)

        predictions: list[FlawPrediction] = []
        i = 0
        while i < n_points:
            direction = directions[i]
            if direction is None:
                i += 1
                continue
            j = i
            while j + 1 < n_points and directions[j + 1] == direction:
                j += 1
            start_idx, end_idx = i, j
            i = j + 1

            v_start_sec = float(var_times[start_idx])
            v_end_sec = float(var_times[end_idx])
            duration = v_end_sec - v_start_sec
            if duration < min_duration:
                continue

            d_var = float(comp.warp_path[end_idx, 1] - comp.warp_path[start_idx, 1])
            d_ref = float(comp.warp_path[end_idx, 0] - comp.warp_path[start_idx, 0])
            if d_ref <= 0 or d_var <= 0:
                continue
            mean_slope = d_var / d_ref
            mean_rate = 1.0 / mean_slope
            deviation = abs(mean_rate - 1.0)

            start_sample = max(0, int(math.floor(v_start_sec * sample_rate)))
            end_sample = min(
                total_samples,
                max(start_sample + 1, int(math.ceil(v_end_sec * sample_rate))),
            )
            true_start_sec = start_sample / sample_rate
            true_end_sec = end_sample / sample_rate
            true_dur_sec = true_end_sec - true_start_sec

            ref_start, ref_end = _reference_span(
                float(ref_times[start_idx]), float(ref_times[end_idx])
            )

            # Evidence strength [0.0, 1.0]: uncalibrated heuristic, not a probability.
            dev_score = min(1.0, deviation / 0.5)
            dur_score = min(1.0, duration / 1.5)
            strength = round(float(0.6 * dev_score + 0.4 * dur_score), 3)
            strength = max(0.1, min(1.0, strength))

            predictions.append(
                FlawPrediction(
                    prediction_id=_prediction_id("tempo", inp, start_sample, end_sample),
                    flaw_type=self.flaw_type,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_sec=round(true_start_sec, 4),
                    end_sec=round(true_end_sec, 4),
                    duration_sec=round(true_dur_sec, 4),
                    ref_start_sec=ref_start,
                    ref_end_sec=ref_end,
                    evidence={
                        "direction": direction,
                        "estimated_relative_rate": round(mean_rate, 3),
                        "mean_warp_slope": round(mean_slope, 3),
                        "rate_deviation": round(deviation, 3),
                        "threshold_min_deviation": min_dev,
                        "duration_sec": round(duration, 4),
                        "slope_window_sec": self.config.tempo_slope_window_sec,
                    },
                    evidence_strength=strength,
                    detector_id=self.detector_id,
                    decision_threshold=min_dev,
                )
            )

        return predictions


class EnergyReductionDetector:
    """Detects sustained regional signal energy reduction relative to reference."""

    detector_id: str = "energy_reduction_detector_v5"
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
        inactive = _inactive_mask(inp, self.config.pause_silence_threshold_db)

        in_drop = False
        start_idx = 0

        for i in range(n_points):
            val = delta_rms_db[i]
            # Must be a valid finite number
            # Exclude ambient silence regions where reference itself is quiet (e.g. < -45 dB)
            # and variant is quiet, because ambient background drop is not speech energy reduction
            is_speech_in_ref = ref_rms[i] > -45.0
            is_energy_drop = (
                np.isfinite(val)
                and (val <= threshold_db)
                and is_speech_in_ref
                and not inactive[i]  # silent + unvoiced variant frame: pause evidence only
            )

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

                        ref_start, ref_end = _reference_span(
                            float(ref_times[start_idx]), float(ref_times[end_idx])
                        )

                        # Evidence strength [0.0, 1.0]
                        # Scaled by drop magnitude below threshold and duration
                        drop_excess = abs(mean_drop) - abs(threshold_db)
                        drop_score = min(1.0, max(0.0, drop_excess / 15.0))
                        dur_score = min(1.0, duration / 1.5)
                        strength = round(float(0.6 * drop_score + 0.4 * dur_score), 3)
                        strength = max(0.1, min(1.0, strength))

                        pred_id = _prediction_id("energy", inp, start_sample, end_sample)
                        predictions.append(
                            FlawPrediction(
                                prediction_id=pred_id,
                                flaw_type=self.flaw_type,
                                start_sample=start_sample,
                                end_sample=end_sample,
                                start_sec=round(true_start_sec, 4),
                                end_sec=round(true_end_sec, 4),
                                duration_sec=round(true_dur_sec, 4),
                                ref_start_sec=ref_start,
                                ref_end_sec=ref_end,
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
