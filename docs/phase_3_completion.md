# FlawBench Phase 3 — Acoustic Feature Extraction, Time-Series Analysis & Reference Comparison

## Executive Summary

Phase 3 upgrades FlawBench from dataset generation and ground-truth injection (Phase 2) to rigorous, time-indexed acoustic feature extraction and differential reference comparison.

Following the core guiding principle:
> **Every extracted feature must come from actual audio, every timestamp must correspond to the correct recording timeline, and every reported status must reflect functionality that genuinely exists.**

Ground truth flaw annotations (`flaw_type`, `severity`, `output_interval`, `source_interval`) are strictly isolated from the feature extraction engine. The extractor only consumes label-free audio data and produces mathematically verifiable acoustic representations.

---

## Architecture & Modules

### 1. Feature Metadata & Schema (`src/flawbench/features/schema.py`)
- **`AudioRole`**: Enum designating whether the audio is `REFERENCE` or `VARIANT`.
- **`AudioInput`**: Label-free descriptor containing only audio metadata (`artifact_id`, `sha256`, `sample_rate`, `num_channels`, `num_samples`, `duration_seconds`). Ground truth flaw annotations are strictly excluded.
- **`FrameSchema`**: Canonical integer-sample framing specification. Documents:
  - `start_samples`: `int64` array of frame start indices ($i \cdot \text{hop\_length}$)
  - `end_samples`: `int64` array of frame end indices ($\min(N, \text{start} + \text{frame\_length})$)
  - `center_times_sec`: `float64` array of frame center timestamps in seconds
  - `valid_samples`: `int64` count of non-padded audio samples in each frame
  - `padded`: `bool` array indicating frames with zero-padding (at recording boundaries)
- **`ActivityInterval`**: Heuristic energy-based speech/silence intervals with start/end samples and timestamps.
- **`FeatureMetadata`**: Non-numeric provenance tracking schema version, audio SHA-256, frame geometry, filter configurations, software versions, and config fingerprint.

### 2. Acoustic Feature Extractor (`src/flawbench/features/extractor.py`)
Extracts the following features synchronized on the canonical frame grid:
- **RMS Energy (Linear & Decibel)**: Framing with window duration (default 25 ms, 10 ms hop), decibel conversion floored at $-120$ dB.
- **Zero-Crossing Rate (ZCR)**: High-frequency/noisiness indicator across frames.
- **Spectral Features**:
  - `spectral_centroid`: Center of mass of the spectrum (brightness)
  - `spectral_bandwidth`: Width of spectral band around centroid
  - `spectral_rolloff`: Frequency below which 85% (configurable) of spectral energy lies
  - `spectral_flatness`: Ratio of geometric mean to arithmetic mean of spectral energy (tonality vs noisiness)
- **Mel-Frequency Cepstral Coefficients (MFCCs)**: Configurable coefficient count ($n=13$), representing timbre and spectral envelope.
- **Pitch Estimation (F0)**:
  - Backends: `librosa_yin` and `librosa_pyin` with adaptive window length ($N \ge 2 \cdot sr / f_{min}$) to prevent truncation artifacts.
  - Generates: `f0_hz` (frequency in Hz, NaN for unvoiced frames), `voiced_flag` (boolean mask), and `pitch_confidence`.
- **Activity Intervals**: Heuristic voice activity/silence segmentation based on configurable RMS energy threshold (`silence_threshold_db`) and minimum pause duration (`min_silence_ms`).
- **Global Summary Statistics**: Mean, std, min, max for RMS energy; voiced ratio; mean/std voiced F0; active speech ratio.
- **`FeatureResult`**: Data structure supporting `.save()` to compressed `.npz` and `.json`, and `.load()`.

### 3. Reference Comparison & Differential Time-Series Analysis (`src/flawbench/features/comparison.py`)
- **Dynamic Time Warping (DTW)**: Non-linear temporal alignment between reference and variant MFCC feature matrices.
- **Differential Curves along Warp Path**:
  - $\Delta \text{RMS}$ (dB): Aligned energy difference curve ($RMS_{variant} - RMS_{reference}$)
  - $\Delta F0$ (Hz): Pitch drift on co-voiced frames
  - Spectral distance: Euclidean distance across aligned MFCC vectors
  - Local warp slope: Instantaneous time dilation factor ($\Delta t_{var} / \Delta t_{ref}$) identifying stretch, compression, or inserted pauses
- **Candidate Flaw Region Detection**: Acoustic anomaly detection heuristics:
  - Energy reduction: Sustained negative $\Delta \text{RMS} \le -4.0$ dB for $\ge 100$ ms
  - Pause insertion: Time advancement in variant while reference is frozen or variant is quiet ($\le -35$ dB) for $\ge 150$ ms
  - Tempo change: Sustained deviation of local slope from 1.0 (relative rate $< 0.85$ or $> 1.20$) for $\ge 300$ ms
- **`ComparisonResult`**: Self-contained result container with normalized DTW cost, global metrics, and serialization.

### 4. Feature Persistence Store (`src/flawbench/features/store.py`)
- Centralized storage in `data/features/` with atomic manifest updates (`feature_manifest.jsonl`).
- Cache policies: `skip_if_valid` (returns cached feature artifact if audio SHA-256 and configuration fingerprint match) and `always_recompute`.
- Store validation: `validate_store()` checks disk presence and manifest integrity.

### 5. Orchestration Pipeline (`src/flawbench/features/pipeline.py`)
- `extract_reference(source_id)`: Extracts and caches features for registered sources.
- `extract_variant(sample_id)`: Extracts and caches features for dataset variants.
- `extract_dataset()`: Batch extracts all references and variants.
- `compare_variant_to_reference(sample_id)`: Performs DTW alignment and differential analysis for a variant against its reference.
- `compare_all()`: Computes comparisons across all dataset variants.
- `export_feature_summaries()`: Exports aggregate feature metrics as JSON.

### 6. Interactive Visualization (`src/flawbench/ui/feature_panel.py`)
Streamlit component using Plotly for responsive, publication-grade interactive graphs:
- Energy curve with silence threshold line
- F0 pitch contour with voiced segment markers
- Spectral centroid and rolloff trajectories
- Full MFCC coefficient heatmap
- DTW alignment warp path and 1:1 diagonal
- Differential $\Delta \text{RMS}$ curve with attenuation boundaries
- Candidate anomaly table
- In-memory feature extraction on user audio uploads

### 7. CLI Subcommands (`src/flawbench/__main__.py`)
- `flawbench features extract --source <ID> | --sample <ID> | --all [--force]`
- `flawbench features compare --sample <ID> | --all [--force]`
- `flawbench features list`
- `flawbench features validate`
- `flawbench features export [--output <PATH>]`

---

## Test Verification

The entire test suite was executed and passed with 0 failures:
- **Total tests**: 246
- **Passed**: 242
- **Skipped**: 4 (3 tests requiring Streamlit $\ge 1.65$ `AppTest.file_uploader`, 1 POSIX permissions test skipped on Windows)
- **Failed**: 0

### Phase 3 Specific Test Coverage:
1. `tests/test_features_extractor.py`: Framing mathematics, sample alignment, YIN pitch accuracy on 220 Hz synthetic tone, silence detection, round-trip serialization, label-free isolation.
2. `tests/test_features_store.py`: Persistence, manifest updates, cache hit/miss on SHA256 + config fingerprint, missing artifact detection.
3. `tests/test_features_comparison.py`: DTW on identical audio ($\approx 0$ distance), energy reduction anomaly detection, pause insertion anomaly detection, comparison serialization.
4. `tests/test_features_pipeline.py`: End-to-end extraction and comparison across reference registration and variant generation.
5. `tests/test_features_cli.py`: Execution of `extract`, `compare`, `list`, `validate`, and `export` CLI commands.
