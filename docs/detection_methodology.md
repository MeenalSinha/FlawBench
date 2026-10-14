# FlawBench Detection Methodology (Phase 4)

## 1. Overview and Core Objective

The central objective of Phase 4 in FlawBench is:
**Given a reference speech recording and a modified variant of the same speech, independently identify which delivery flaws are present, determine their locations on the variant timeline, and provide measurable acoustic evidence supporting each detection—without accessing ground-truth annotations.**

FlawBench supports three canonical speech flaw categories in Phase 4:
1. `pause_insertion`: Unusually long pauses appearing in a modified recording unsupported by the reference.
2. `tempo_change`: Local delivery rate acceleration or deceleration relative to the reference.
3. `energy_reduction`: Sustained attenuation in signal energy within a region relative to the reference.

---

## 2. Mathematical Decision Rules & Detectors

### 2.1 Pause Insertion Detector (`PauseDetector`)

#### Input Features
- Aligned warp path $W = \{(i_k, j_k)\}_{k=0}^{K-1}$ between reference frames $i$ and variant frames $j$.
- Variant RMS energy in decibels $\text{RMS}_{var}(j)$ and reference RMS energy $\text{RMS}_{ref}(i)$.
- Configured silence energy threshold $\theta_{\text{silence}} \le -45.0\text{ dB}$.

#### Detection Logic
When an inserted pause occurs in a modified recording, the temporal alignment progression between reference and variant displays a distinctive signature:
1. **Reference Freeze / Variant Advancement**: While the variant audio continues to progress across silent frames $j \in [j_{start}, j_{end}]$, the corresponding reference index $i$ remains virtually frozen at the pause insertion point $i^*$.
2. **Variant Inactivity**: Throughout this frozen span, the variant's acoustic energy satisfies:
   $$\text{RMS}_{var}(j) \le \theta_{\text{silence}}$$
3. **Contrastive Silence Difference**: The duration of this quiet interval in the variant is compared to the corresponding reference span. If:
   $$\Delta t = t_{var\_dur} - t_{ref\_dur} \ge \theta_{\text{min\_pause\_diff}} \quad \text{and} \quad t_{var\_dur} \ge \theta_{\text{min\_pause\_dur}}$$
   a `pause_insertion` candidate is flagged.

#### Evidence Measurements
- `variant_pause_dur_sec`: Measured variant pause duration $\Delta t_{var}$.
- `ref_pause_dur_sec`: Measured reference pause duration $\Delta t_{ref}$.
- `duration_diff_sec`: $\Delta t = \Delta t_{var} - \Delta t_{ref}$.
- `min_variant_rms_db`: Lowest RMS energy (dB) observed in the candidate window.
- `mean_variant_rms_db`: Average RMS energy (dB) in the candidate window.
- `threshold_pause_dur_sec`: Configured decision boundary $\theta_{\text{min\_pause\_dur}}$.

#### Evidence Strength
$$\text{strength} = \min\left(1.0, \frac{\Delta t}{2.0 \cdot \theta_{\text{min\_pause\_dur}}}\right) \in [0.0, 1.0]$$

---

### 2.2 Tempo Deviation Detector (`TempoDetector`)

#### Mathematical Ratio Convention
The local warp slope represents the rate of temporal progression in the variant relative to the reference. For a local alignment window $[k - w, k + w]$ on the warp path $W = \{(i_k, j_k)\}$:
$$s(j) = \frac{\Delta t_{var}}{\Delta t_{ref}} = \frac{(j_{k+w} - j_{k-w}) \cdot \text{hop\_sec}}{(i_{k+w} - i_{k-w}) \cdot \text{hop\_sec}} = \frac{j_{k+w} - j_{k-w}}{i_{k+w} - i_{k-w}}$$

- **Normal progression**: $s(j) \approx 1.0$ (variant and reference progress at identical speaking rate).
- **Slower delivery (Pacing deceleration)**: The variant takes more elapsed time to cover the same spoken content. Thus, $\Delta t_{var} > \Delta t_{ref}$, yielding:
  $$s(j) \ge \theta_{\text{slower}} \quad (\text{default: } 1.30)$$
- **Faster delivery (Pacing acceleration)**: The variant takes less elapsed time to cover the same spoken content. Thus, $\Delta t_{var} < \Delta t_{ref}$, yielding:
  $$s(j) \le \theta_{\text{faster}} \quad (\text{default: } 0.75)$$

#### Detection Logic
1. Continuous contiguous frames where $s(j) \ge \theta_{\text{slower}}$ or $s(j) \le \theta_{\text{faster}}$ are identified.
2. The candidate interval is retained only if the sustained duration exceeds the configured threshold:
   $$\text{duration} = (j_{end} - j_{start}) \cdot \text{hop\_sec} \ge \theta_{\text{min\_tempo\_dur}} \quad (\text{default: } 0.40\text{ s})$$
3. Flaw category is emitted as `tempo_change`, with the specific directional anomaly (`slower` or `faster`) recorded in the evidence dictionary.

#### Evidence Measurements
- `direction`: `"slower"` or `"faster"`.
- `mean_warp_slope`: Mean local warp slope $\bar{s}$ over the interval.
- `median_warp_slope`: Median local warp slope over the interval.
- `slope_deviation`: Absolute deviation from unit progression $|\bar{s} - 1.0|$.
- `threshold_used`: Applied ratio threshold ($\theta_{\text{slower}}$ or $\theta_{\text{faster}}$).
- `sustained_duration_sec`: Duration of the anomalous tempo region in seconds.

#### Evidence Strength
$$\text{strength} = \min\left(1.0, \frac{|\bar{s} - 1.0|}{|\theta - 1.0|} \cdot \frac{\text{duration}}{2.0 \cdot \theta_{\text{min\_dur}}}\right) \in [0.0, 1.0]$$

---

### 2.3 Energy Reduction Detector (`EnergyReductionDetector`)

#### Sign Convention and Formulations
Aligned frame-level energy difference is calculated using the canonical convention:
$$\Delta \text{RMS}_{dB}(j) = \text{RMS}_{var\_dB}(j) - \text{RMS}_{ref\_dB}(i_j)$$
where $i_j$ is the reference frame paired with variant frame $j$ via the DTW warp path.

A negative value indicates that the variant signal is quieter than the corresponding reference speech.

#### Detection Logic
1. **Silence Exclusion**: Silent frames where either $\text{RMS}_{ref}(i) \le \theta_{\text{silence}}$ or $\text{RMS}_{var}(j) \le \theta_{\text{silence}}$ are excluded from energy attenuation analysis to prevent silence/pause regions from triggering spurious energy reduction flags.
2. **Threshold Test**: Active speech frames satisfying:
   $$\Delta \text{RMS}_{dB}(j) \le \theta_{\text{energy\_diff}} \quad (\text{default: } -6.0\text{ dB})$$
   are gathered into contiguous clusters.
3. **Sustained Duration**: Clusters spanning at least $\theta_{\text{min\_energy\_dur}}$ (default: 0.30 s) are registered as `energy_reduction` candidates.

#### Evidence Measurements
- `mean_delta_rms_db`: Average decibel energy reduction $\overline{\Delta \text{RMS}_{dB}}$.
- `max_drop_db`: Peak negative attenuation $\min_j \Delta \text{RMS}_{dB}(j)$.
- `mean_var_rms_db`: Average variant energy in candidate region.
- `mean_ref_rms_db`: Average reference energy in aligned region.
- `threshold_db`: Applied threshold $\theta_{\text{energy\_diff}}$.
- `sustained_duration_sec`: Measured duration of the attenuation span.

#### Evidence Strength
$$\text{strength} = \min\left(1.0, \frac{|\overline{\Delta \text{RMS}_{dB}}|}{2.0 \cdot |\theta_{\text{energy\_diff}}|}\right) \in [0.0, 1.0]$$

---

## 3. Post-Processing and Consolidation

The post-processing pipeline receives raw candidates from all enabled detectors and executes deterministic consolidation:

1. **Category Isolation**: Detections from different flaw categories (e.g., a pause vs an energy reduction) are never merged together, preserving independent acoustic phenomena.
2. **Gap Merging**: Adjacent candidates of the *same* flaw category separated by a gap less than or equal to `merge_gap_sec` (default: 0.20 s) are merged into a single cohesive interval.
3. **Short Candidate Filtering**: Candidates shorter than `min_candidate_duration_sec` (default: 0.15 s) are filtered out to eliminate transient alignment jitter.
4. **Half-Open Interval Validation**: All intervals enforce the half-open canonical bounds:
   $$[start\_sample, end\_sample) \quad \text{where } 0 \le start\_sample < end\_sample \le total\_samples$$
   $$[start\_sec, end\_sec) \quad \text{where } 0.0 \le start\_sec < end\_sec \le duration\_seconds$$
5. **Chronological Sorting**: Final predictions are sorted strictly by $start\_sample$.

---

## 4. Ground-Truth Isolation & Architectural Boundaries

A critical requirement of Phase 4 is complete insulation from ground-truth annotations:
- Detectors accept only `DetectionInput`, which contains the audio signal, sample rates, acoustic feature arrays, and DTW differential curves.
- The fields `flaw_type`, `severity`, `output_interval`, `source_interval`, and injection placement seeds are completely inaccessible to the detection engine.
- Automated tests (`tests/test_detection_isolation.py`) prove that modifying `manifest.jsonl` annotations without altering audio produces identical predictions.
- The `DetectionValidator` scans output artifacts and rejects any results containing forbidden annotation keys.

---

## 5. Limitations & Caveats

1. **Acoustic Evidence vs. Human Competence**: The evidence strength score measures the magnitude of deviation from the reference speech. It is an algorithmic evidence score, **not** a calibrated human subjective competence score or probability.
2. **DTW Alignment Sensitivity**: Dynamic Time Warping assumes monotonic alignment of spoken words. Severe acoustic distortion or out-of-order phrase insertions can cause alignment artifacts.
3. **Natural Pauses vs. Injected Pauses**: Natural breath pauses present in both the reference and variant are not flagged because the detector explicitly computes contrastive pause differences ($\Delta t_{var} - \Delta t_{ref}$). However, conversational pauses occurring during spontaneous speech variation may trigger candidates if thresholds are configured too sensitively.
4. **Separation from Evaluation**: Phase 4 focuses on independent inference. Quantitative evaluation metrics (Precision, Recall, F1, IoU against ground truth) are formally computed in Phase 5.
