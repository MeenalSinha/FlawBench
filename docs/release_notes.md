# FlawBench Release Notes

## Version: v0.1.0 (Release Candidate 1)
**Date**: October 2026  
**License**: MIT  
**Repository**: [MeenalSinha/FlawBench](https://github.com/MeenalSinha/FlawBench)

---

## 1. Overview

FlawBench v0.1.0-rc1 is the initial release candidate of the **FlawBench** speech flaw analysis and benchmark platform. FlawBench establishes an evidence-based foundation for vocal delivery assessment by introducing **Parametric Flaw Injection with Exact Ground Truth**. 

Rather than relying on subjective human scores or ungrounded generative language model feedback, FlawBench creates controlled speech modifications with known ground-truth intervals, extracts frame-level acoustic features, detects delivery anomalies blindly, and rigorously benchmarks performance using 1-to-1 temporal Intersection-over-Union (IoU) event matching.

---

## 2. Implemented Capabilities (Phases 1–8)

### Phase 1 — System Foundation & Provenance
- Centralized configuration system supporting YAML configuration files, hierarchical overrides, and environment variables.
- Pure audio validation and decoding engine using `libsndfile` and `soundfile` without implicit resampling or silent distortion.
- Runtime path management with security guards against directory traversal.
- Structured, idempotent logging and real-time component health checks.
- Cryptographic provenance tracking (SHA-256 configuration fingerprints, system package versions).

### Phase 2 — Dataset & Ground-Truth Engine
- Reference recording registry with transcript storage, rights verification, and licensing metadata.
- Deterministic, seeded flaw injectors:
  - `pause_insertion`: Injects silence segments with configurable duration and boundary placement.
  - `tempo_change`: Modifies local delivery rate using WSOLA (`audiotsm`) without pitch distortion.
  - `energy_reduction`: Attenuates localized signal amplitude by a specified decibel level.
- Transactional sample generation with exact integer-sample ground-truth annotations `[start_sample, end_sample)`.
- Dataset splitting (source-level separation to prevent data leakage between train, validation, and test splits).
- Deep dataset integrity validator checking audio headers, durations, and manifest alignment.

### Phase 3 — Acoustic Feature Extraction & Differential Analysis
- Canonical 25 ms / 10 ms acoustic frame grid.
- Comprehensive acoustic feature extraction:
  - RMS energy envelope & zero-crossing rate (ZCR).
  - Spectral centroid, bandwidth, rolloff (85%), and flatness.
  - Mel-Frequency Cepstral Coefficients (MFCCs 0–12).
  - Fundamental frequency (F0 pitch) estimation via `librosa.pyin`.
- Dynamic Time Warping (DTW) nonlinear alignment between reference and participant recordings.
- Differential feature analysis producing localized acoustic distance metrics.
- Feature persistence store with manifest tracking and validation.

### Phase 4 — Independent Flaw Detection & Localization
- Completely label-blind detection architecture: detectors cannot access ground truth during inference.
- Three specialized acoustic detectors:
  - `PauseDetector`: Identifies anomalous silence intervals using energy thresholds and voicing guards.
  - `TempoDetector`: Computes local warp slope over moving time windows to detect pacing deviations.
  - `EnergyReductionDetector`: Flags sustained amplitude dropouts while ignoring unvoiced silence.
- Post-processing pipeline with candidate merging and minimum duration filtering.
- Immutable detection artifact store with cache invalidation and integrity validation.

### Phase 5 — Formal Evaluation & Benchmarking Engine
- Deterministic 1-to-1 event matching using temporal IoU thresholding ($\theta_{\text{IoU}} = 0.5$).
- Classification metrics: Precision, Recall, F1 (per-category and micro/macro aggregates).
- Temporal localization metrics: Mean/Median IoU, boundary start/end MAE, duration MAE.
- Multi-threshold localization curves ($\theta_{\text{IoU}} \in [0.3, 0.9]$).
- Negative identity controls: verification that clean references trigger zero false alarms.
- Source-level cluster bootstrap 95% confidence intervals.
- Cataloged evaluation store with structured Markdown and JSON reporting.

### Phase 6 — Participant Dashboard & Explainability
- Participant analysis workflow supporting registered samples and in-memory uploaded audio files.
- Deterministic Explanation Cards translating physical acoustic measurements into natural delivery guidance.
- Strict numeric traceability: every numerical claim in an explanation is cross-verified against detection facts.
- Audio playback of anomaly intervals with configurable pre- and post-roll context margins.
- Multi-format report export: Markdown, JSON, HTML, and CSV.
- Interactive multi-page Streamlit dashboard (`app.py`).

### Phase 7 — Quality Assurance & Reliability Hardening
- Comprehensive regression test suite consisting of **415 automated tests** across all subsystems.
- Detector reliability hardening:
  - Windowed tempo slope replacing quantized path-point differentiation.
  - Voicing and silence guards preventing energy/pause cross-category false alarms.
  - Exclusion of MFCC c0 from DTW alignment to prevent level steps from distorting temporal warping.
- Numerical stability policy: clean handling of silent frames and unvoiced pitch correlations (`None` rather than 0.0).
- Deterministic, content-hashed prediction identifiers replacing random UUIDs.

### Phase 8 — Release Engineering & Packaging
- Repeatable release-validation CLI command (`python -m flawbench release-check`).
- Standalone demonstration runbook, spoken script, and visual shot list (`docs/demo_runbook.md`).
- Licensing audit and third-party attribution documentation (`docs/licensing_and_disclosure.md`).
- Clean distribution package structure with validated checksums.

---

## 3. Verified Benchmark Results

The following metrics were established on the frozen **held-out evaluation cohort** (`heldout_rc`):
- **Cohort**: 8 independent source recordings, 24 controlled variants (3 per source: pause, tempo, energy).
- **Ground Truth Events**: 24.
- **Detector Version**: 5.0 (Configuration fingerprint: `ee4c7f85d2c2ad13`).

| Metric | Pause Insertion | Tempo Change | Energy Reduction | Overall (Micro) | Overall (Macro) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Precision** | **1.000** | 0.857 | **1.000** | **0.957** | 0.952 |
| **Recall** | **1.000** | 0.750 | **1.000** | **0.917** | 0.917 |
| **F1-Score** | **1.000** | 0.800 | **1.000** | **0.936** | 0.933 |
| **Mean IoU** | 0.979 | 0.926 | 0.993 | **0.970** | 0.966 |
| **Start MAE (s)** | 0.009 | 0.069 | 0.010 | **0.029** | 0.029 |
| **End MAE (s)** | 0.008 | 0.058 | 0.005 | **0.023** | 0.024 |

*Key Findings*:
- Pause insertion and energy reduction achieved perfect precision and recall (1.000) with sub-15ms boundary localization accuracy.
- Tempo change achieved 0.857 precision and 0.750 recall (6/8 detected, 1 false positive); the 2 missed detections occurred at the boundary condition where the injected rate deviation (0.85) equaled the minimum detection threshold (0.15).
- Negative identity controls produced 0 false alarms.

---

## 4. Bug Fixes & Refactorings in this Release

1. **Test Environment Disambiguation**: Resolved module collisions with site-packages by packaging `tests/` with `__init__.py` and standardizing fixture imports.
2. **Streamlit AppTest Uploader Compatibility**: Upgraded Streamlit dependency constraints to `>=1.56.0,<2` to enable `AppTest.file_uploader` support.
3. **DTW Alignment Robustness**: Excluded energy-dependent MFCC coefficient 0 from alignment calculations, eliminating spurious warp distortions caused by volume changes.
4. **XSS & Hostile Text Neutralization**: Enforced backtick escaping on participant-uploaded filenames in exported Markdown and HTML reports.
5. **Deterministic Run IDs**: Replaced second-resolution timestamp run IDs with collision-proof unique identifiers.

---

## 5. Known Limitations & Scope Disclosures

- **Synthetic Fixtures vs. Natural Human Speech**: FlawBench's formal benchmarks were established on controlled parametric audio fixtures. These verify algorithmic mechanics, feature responses, and mathematical correctness, but do not prove performance on conversational speech.
- **Heuristic Evidence Scores**: Detector evidence strengths represent acoustic distance magnitudes relative to reference features. They are not calibrated Bayesian posterior probabilities of human delivery flaw severity.
- **Phonetic & Word-Level Alignment**: FlawBench operates on frame-level acoustic grids; forced audio-text alignment and word-level segmentation are deferred to future phases.
- **Delivery Dimensions**: The current release focuses on temporal pacing, pause timing, and vocal energy. Pitch flattening, pronunciation, and intonation contour coaching are planned extensions.
