"""Phase 3 Reference Comparison & Differential Time-Series Analysis.

Aligns acoustic feature trajectories of reference and variant recordings using
Dynamic Time Warping (DTW) and computes differential acoustic profiles:
- Delta RMS energy over aligned time
- Spectral distance trajectory (MFCC distance)
- Fundamental frequency (F0) drift on co-voiced frames
- Local time dilation / tempo slope analysis
- Global distance metrics and acoustic anomaly candidates
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from numpy.typing import NDArray

from flawbench.exceptions import FeatureComparisonError
from flawbench.features.extractor import FeatureResult


@dataclass
class ComparisonResult:
    """Differential acoustic comparison between a reference and a variant recording."""

    ref_artifact_id: str
    variant_artifact_id: str
    ref_source_id: str
    variant_sample_id: str | None
    dtw_cost: float
    normalized_dtw_cost: float

    # Warp path coordinates shape: (P, 2) where col 0 is ref_frame_idx, col 1 is var_frame_idx
    warp_path: NDArray[np.int64]

    # Aligned timelines (shape: (P,))
    ref_times_sec: NDArray[np.float64]
    var_times_sec: NDArray[np.float64]

    # Differential feature curves along warp path (shape: (P,))
    aligned_ref_rms_db: NDArray[np.float64]
    aligned_var_rms_db: NDArray[np.float64]
    delta_rms_db: NDArray[np.float64]
    aligned_ref_f0: NDArray[np.float64]
    aligned_var_f0: NDArray[np.float64]
    delta_f0_hz: NDArray[np.float64]
    spectral_distance: NDArray[np.float64]
    local_slope: NDArray[np.float64]

    # Aggregated metrics
    global_metrics: dict[str, float | str | None] = field(default_factory=dict)
    candidate_regions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def path_length(self) -> int:
        return len(self.warp_path)

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison result to a serializable dictionary summary."""
        return {
            "ref_artifact_id": self.ref_artifact_id,
            "variant_artifact_id": self.variant_artifact_id,
            "ref_source_id": self.ref_source_id,
            "variant_sample_id": self.variant_sample_id,
            "dtw_cost": self.dtw_cost,
            "normalized_dtw_cost": self.normalized_dtw_cost,
            "path_length": self.path_length,
            "global_metrics": self.global_metrics,
            "candidate_regions": self.candidate_regions,
        }

    def save(self, filepath: Path) -> Path:
        """Save comparison arrays and metadata to a .npz bundle."""
        filepath = Path(filepath).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        summary_json = json.dumps(self.to_dict())

        np.savez_compressed(
            filepath,
            warp_path=self.warp_path,
            ref_times_sec=self.ref_times_sec,
            var_times_sec=self.var_times_sec,
            aligned_ref_rms_db=self.aligned_ref_rms_db,
            aligned_var_rms_db=self.aligned_var_rms_db,
            delta_rms_db=self.delta_rms_db,
            aligned_ref_f0=self.aligned_ref_f0,
            aligned_var_f0=self.aligned_var_f0,
            delta_f0_hz=self.delta_f0_hz,
            spectral_distance=self.spectral_distance,
            local_slope=self.local_slope,
            summary_json=np.array(summary_json),
        )
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> ComparisonResult:
        """Load a ComparisonResult from a saved .npz bundle."""
        data = np.load(filepath)
        summary = json.loads(str(data["summary_json"]))

        return cls(
            ref_artifact_id=summary["ref_artifact_id"],
            variant_artifact_id=summary["variant_artifact_id"],
            ref_source_id=summary["ref_source_id"],
            variant_sample_id=summary["variant_sample_id"],
            dtw_cost=float(summary["dtw_cost"]),
            normalized_dtw_cost=float(summary["normalized_dtw_cost"]),
            warp_path=data["warp_path"],
            ref_times_sec=data["ref_times_sec"],
            var_times_sec=data["var_times_sec"],
            aligned_ref_rms_db=data["aligned_ref_rms_db"],
            aligned_var_rms_db=data["aligned_var_rms_db"],
            delta_rms_db=data["delta_rms_db"],
            aligned_ref_f0=data["aligned_ref_f0"],
            aligned_var_f0=data["aligned_var_f0"],
            delta_f0_hz=data["delta_f0_hz"],
            spectral_distance=data["spectral_distance"],
            local_slope=data["local_slope"],
            global_metrics=summary.get("global_metrics", {}),
            candidate_regions=summary.get("candidate_regions", []),
        )


def _detect_candidate_flaw_regions(
    warp_path: NDArray[np.int64],
    ref_times: NDArray[np.float64],
    var_times: NDArray[np.float64],
    delta_rms_db: NDArray[np.float64],
    local_slope: NDArray[np.float64],
    var_rms_db: NDArray[np.float64],
) -> list[dict[str, Any]]:
    """Heuristically identify candidate acoustic flaw regions based on differential features."""
    candidates: list[dict[str, Any]] = []
    n_points = len(warp_path)
    if n_points == 0:
        return candidates

    # 1. Energy reduction candidate search (sustained negative delta_rms_db < -4 dB)
    in_energy_drop = False
    drop_start = 0
    for i in range(n_points):
        if delta_rms_db[i] <= -4.0 and not in_energy_drop:
            in_energy_drop = True
            drop_start = i
        elif (delta_rms_db[i] > -4.0 or i == n_points - 1) and in_energy_drop:
            in_energy_drop = False
            drop_end = i
            duration = var_times[drop_end] - var_times[drop_start]
            if duration >= 0.1:  # min 100 ms
                mean_drop = float(np.mean(delta_rms_db[drop_start : drop_end + 1]))
                candidates.append(
                    {
                        "flaw_hypothesis": "energy_reduction",
                        "var_start_sec": round(float(var_times[drop_start]), 4),
                        "var_end_sec": round(float(var_times[drop_end]), 4),
                        "ref_start_sec": round(float(ref_times[drop_start]), 4),
                        "ref_end_sec": round(float(ref_times[drop_end]), 4),
                        "duration_sec": round(float(duration), 4),
                        "mean_delta_rms_db": round(mean_drop, 2),
                        "min_delta_rms_db": round(
                            float(np.min(delta_rms_db[drop_start : drop_end + 1])), 2
                        ),
                    }
                )

    # 2. Pause insertion candidate search (variant advances while reference is frozen or low energy)
    in_pause = False
    pause_start = 0
    for i in range(1, n_points):
        ref_frozen = warp_path[i, 0] == warp_path[i - 1, 0]
        var_quiet = var_rms_db[i] <= -35.0
        is_pause_point = ref_frozen or var_quiet

        if is_pause_point and not in_pause:
            in_pause = True
            pause_start = i - 1
        elif (not is_pause_point or i == n_points - 1) and in_pause:
            in_pause = False
            pause_end = i
            duration = var_times[pause_end] - var_times[pause_start]
            if duration >= 0.15:  # min 150 ms
                candidates.append(
                    {
                        "flaw_hypothesis": "pause_insertion",
                        "var_start_sec": round(float(var_times[pause_start]), 4),
                        "var_end_sec": round(float(var_times[pause_end]), 4),
                        "ref_anchor_sec": round(float(ref_times[pause_start]), 4),
                        "duration_sec": round(float(duration), 4),
                        "mean_var_rms_db": round(
                            float(np.mean(var_rms_db[pause_start : pause_end + 1])), 2
                        ),
                    }
                )

    # 3. Tempo change candidate search (significant deviation of local slope from 1.0)
    in_tempo = False
    tempo_start = 0
    for i in range(n_points):
        slope = local_slope[i]
        is_tempo_dev = (slope < 0.85 or slope > 1.20) and np.isfinite(slope)
        if is_tempo_dev and not in_tempo:
            in_tempo = True
            tempo_start = i
        elif (not is_tempo_dev or i == n_points - 1) and in_tempo:
            in_tempo = False
            tempo_end = i
            duration = var_times[tempo_end] - var_times[tempo_start]
            if duration >= 0.3:  # min 300 ms
                mean_slope = float(np.mean(local_slope[tempo_start : tempo_end + 1]))
                candidates.append(
                    {
                        "flaw_hypothesis": "tempo_change",
                        "var_start_sec": round(float(var_times[tempo_start]), 4),
                        "var_end_sec": round(float(var_times[tempo_end]), 4),
                        "ref_start_sec": round(float(ref_times[tempo_start]), 4),
                        "ref_end_sec": round(float(ref_times[tempo_end]), 4),
                        "duration_sec": round(float(duration), 4),
                        "mean_relative_rate": round(1.0 / max(1e-4, mean_slope), 3),
                    }
                )

    return candidates


#: Minimum number of jointly voiced aligned points for a Pearson correlation to be reported.
#: Algorithmic constant of the statistic (fewer points make the coefficient meaningless).
_MIN_CORRELATION_POINTS = 4


def _f0_correlation(
    ref_f0: NDArray[np.float64], var_f0: NDArray[np.float64]
) -> tuple[float | None, str]:
    """Pearson correlation of jointly voiced F0, with an explicit validity policy.

    Returns ``(value, status)``. ``value`` is ``None`` whenever the coefficient is mathematically
    undefined or not meaningful; ``status`` then states why. Undefined is never replaced by 0.0:
    zero would assert "no linear relation", which is a measured outcome, not a missing one.
    """
    ref = np.asarray(ref_f0, dtype=np.float64)
    var = np.asarray(var_f0, dtype=np.float64)
    if ref.shape != var.shape:
        return None, "unavailable_shape_mismatch"
    if ref.size < _MIN_CORRELATION_POINTS:
        return None, "unavailable_insufficient_voiced_points"
    if not (np.isfinite(ref).all() and np.isfinite(var).all()):
        return None, "unavailable_non_finite_input"
    if float(np.ptp(ref)) == 0.0 or float(np.ptp(var)) == 0.0:
        return None, "unavailable_constant_f0"
    corr = float(np.corrcoef(ref, var)[0, 1])
    if not np.isfinite(corr):
        return None, "unavailable_undefined"
    return corr, "ok"


def compare_features(
    ref_features: FeatureResult,
    var_features: FeatureResult,
    *,
    metric: str = "cosine",
    exclude_energy_coefficient: bool = False,
) -> ComparisonResult:
    """Perform differential time-series analysis and reference comparison using DTW alignment.

    Args:
        ref_features: FeatureResult of the reference audio recording.
        var_features: FeatureResult of the variant/modified audio recording.
        metric: Distance metric for DTW (default: 'cosine', or 'euclidean').
        exclude_energy_coefficient: Drop MFCC c0 (the energy term) from the alignment cost so the
            warp path does not depend on loudness. Only the alignment changes; the differential
            RMS and spectral-distance outputs still use the full feature set.

    Returns:
        ComparisonResult containing warp path, differential curves, and anomaly candidates.
    """
    if ref_features.num_frames == 0:
        raise FeatureComparisonError("Reference feature result contains 0 frames.")
    if var_features.num_frames == 0:
        raise FeatureComparisonError("Variant feature result contains 0 frames.")

    # Prepare feature matrices for DTW: (n_features, n_frames)
    X = ref_features.mfcc.T
    Y = var_features.mfcc.T
    if exclude_energy_coefficient:
        if X.shape[0] < 2 or Y.shape[0] < 2:
            raise FeatureComparisonError(
                "Cannot exclude the energy coefficient: fewer than 2 MFCC coefficients."
            )
        X, Y = X[1:], Y[1:]

    if X.shape[0] != Y.shape[0]:
        raise FeatureComparisonError(
            f"MFCC feature dimension mismatch: reference has {X.shape[0]}, "
            f"variant has {Y.shape[0]}."
        )

    try:
        cost_matrix, wp = librosa.sequence.dtw(X=X, Y=Y, metric=metric)
    except Exception as exc:
        raise FeatureComparisonError(f"DTW alignment failed: {exc}") from exc

    # Librosa DTW returns warp path in reverse order: (N_ref - 1, N_var - 1) -> (0, 0)
    # Reverse to natural chronological order:
    warp_path = np.ascontiguousarray(wp[::-1], dtype=np.int64)
    total_cost = float(cost_matrix[-1, -1])
    path_len = len(warp_path)
    norm_cost = total_cost / max(1, path_len)

    ref_indices = warp_path[:, 0]
    var_indices = warp_path[:, 1]

    ref_center_times = ref_features.center_times_sec
    var_center_times = var_features.center_times_sec

    aligned_ref_times = ref_center_times[ref_indices]
    aligned_var_times = var_center_times[var_indices]

    # Differential RMS
    aligned_ref_rms_db = ref_features.rms_db[ref_indices]
    aligned_var_rms_db = var_features.rms_db[var_indices]
    delta_rms_db = (aligned_var_rms_db - aligned_ref_rms_db).astype(np.float64)

    # Differential F0
    aligned_ref_f0 = ref_features.f0_hz[ref_indices]
    aligned_var_f0 = var_features.f0_hz[var_indices]
    both_voiced = np.isfinite(aligned_ref_f0) & np.isfinite(aligned_var_f0)
    delta_f0_hz = np.where(both_voiced, aligned_var_f0 - aligned_ref_f0, np.nan).astype(np.float64)

    # Spectral distance (Euclidean distance across MFCCs along warp path)
    ref_mfcc_aligned = ref_features.mfcc[ref_indices]
    var_mfcc_aligned = var_features.mfcc[var_indices]
    spectral_distance = np.linalg.norm(ref_mfcc_aligned - var_mfcc_aligned, axis=1).astype(
        np.float64
    )

    # Local slope calculation: dt_var / dt_ref with sliding window smoothing
    window = 5
    half = window // 2
    local_slope = np.ones(path_len, dtype=np.float64)
    for i in range(path_len):
        start = max(0, i - half)
        end = min(path_len - 1, i + half)
        d_var = var_indices[end] - var_indices[start]
        d_ref = ref_indices[end] - ref_indices[start]
        if d_ref > 0:
            local_slope[i] = float(d_var / d_ref)
        elif d_var > 0:
            local_slope[i] = 10.0  # reference paused while variant continued
        else:
            local_slope[i] = 1.0

    # Global metrics
    ref_dur = ref_features.metadata.duration_seconds
    var_dur = var_features.metadata.duration_seconds
    duration_ratio = float(var_dur / max(1e-4, ref_dur))

    f0_corr, f0_corr_status = _f0_correlation(
        aligned_ref_f0[both_voiced], aligned_var_f0[both_voiced]
    )

    global_metrics = {
        "dtw_total_cost": total_cost,
        "dtw_normalized_cost": norm_cost,
        "duration_ratio": duration_ratio,
        "ref_duration_sec": ref_dur,
        "variant_duration_sec": var_dur,
        "mean_delta_rms_db": float(np.mean(delta_rms_db)),
        "max_delta_rms_db": float(np.max(delta_rms_db)),
        "min_delta_rms_db": float(np.min(delta_rms_db)),
        "mean_spectral_distance": float(np.mean(spectral_distance)),
        # None means "unavailable" (see f0_voiced_correlation_status); it is never coerced to 0.
        "f0_voiced_correlation": f0_corr,
        "f0_voiced_correlation_status": f0_corr_status,
        "dtw_metric": metric,
        "dtw_excludes_energy_coefficient": exclude_energy_coefficient,
    }

    # Candidate flaw region detection
    candidates = _detect_candidate_flaw_regions(
        warp_path=warp_path,
        ref_times=aligned_ref_times,
        var_times=aligned_var_times,
        delta_rms_db=delta_rms_db,
        local_slope=local_slope,
        var_rms_db=aligned_var_rms_db,
    )

    return ComparisonResult(
        ref_artifact_id=ref_features.metadata.artifact_id,
        variant_artifact_id=var_features.metadata.artifact_id,
        ref_source_id=ref_features.metadata.source_id,
        variant_sample_id=var_features.metadata.sample_id,
        dtw_cost=total_cost,
        normalized_dtw_cost=norm_cost,
        warp_path=warp_path,
        ref_times_sec=aligned_ref_times,
        var_times_sec=aligned_var_times,
        aligned_ref_rms_db=aligned_ref_rms_db,
        aligned_var_rms_db=aligned_var_rms_db,
        delta_rms_db=delta_rms_db,
        aligned_ref_f0=aligned_ref_f0,
        aligned_var_f0=aligned_var_f0,
        delta_f0_hz=delta_f0_hz,
        spectral_distance=spectral_distance,
        local_slope=local_slope,
        global_metrics=global_metrics,
        candidate_regions=candidates,
    )
