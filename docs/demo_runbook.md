# FlawBench Demonstration Runbook & Presentation Guide

This runbook provides complete, reproducible instructions for presenting and demonstrating FlawBench. It includes environment preparation, sample workflows, a minute-by-minute spoken script, a shot-by-shot visual guide, and troubleshooting recovery procedures.

---

## 1. Demonstration Overview & Objectives

The goal of the demonstration is to prove that FlawBench works on real data without mockups or pre-fabricated animations.

The demonstration proves:
1. **The Core Innovation**: Parametric flaw injection with exact ground truth solves the fundamental challenge of speech coaching evaluation—how to objectively benchmark detection algorithms when real-world delivery flaws have subjective, contested boundaries.
2. **Acoustic Differential Analysis**: Dynamic Time Warping (DTW) and canonical frame acoustic features (RMS energy, spectral centroid, MFCCs, F0 pitch) provide contrastive representations between reference and participant audio.
3. **Independent Detection**: Detection algorithms identify candidate flaws blindly—without accessing ground truth labels or metadata.
4. **Quantitative Benchmarking (Evaluation Lab)**: Rigorous 1-to-1 temporal IoU matching scores precision, recall, F1, and boundary localization accuracy against known ground truth.
5. **Evidence-Linked Explanations**: Explanation cards link observations directly to physical measurements (decibels, seconds, ratios) and enable synchronized excerpt audio playback.
6. **Transparent Limitations**: The system clearly distinguishes controlled synthetic-fixture benchmarks from natural human speech claims.

---

## 2. Environment Preparation

### 2.1 System Prerequisites
- Python 3.10+ (tested on Python 3.12 and 3.14)
- Working virtual environment with dependencies installed:
  ```bash
  pip install -e ".[dev]"
  ```
- Terminal open in repository root: `FlawBench/`

### 2.2 Pre-Flight Validation Command
Before presenting, run the automated release check to verify environment readiness:
```bash
python -m flawbench release-check
```
Ensure all required checks display `[PASS]` and status is `RELEASE READY`.

### 2.3 Starting the Interactive Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 3. End-to-End Sample Workflow for the Demo

The repository includes tools to generate a clean synthetic demo sample on demand if none exists in the working directory:

### Step A: Initialize directories & register reference
```bash
python -m flawbench init
```

To create a reference and controlled variants using the included test fixtures:
```python
# Quick python one-liner to generate a 4-second reference tone and register it
import soundfile as sf
import numpy as np
from tests.conftest import synthetic_signal, SYN_RATE

ref_path = "data/reference/demo_ref.wav"
sf.write(ref_path, synthetic_signal(4.0, 1), SYN_RATE)
```
Then register the reference via CLI:
```bash
python -m flawbench dataset register-reference data/reference/demo_ref.wav \
  --transcript-text "Welcome to the FlawBench speech evaluation demonstration." \
  --source-type self_recorded \
  --attribution "FlawBench Demo Team" \
  --license "MIT / Demonstration Audio" \
  --rights-confirmed --redistribution permitted
```

### Step B: Generate controlled variant
```bash
python -m flawbench dataset generate --source $(python -m flawbench dataset list-sources | head -n 1 | awk '{print $1}') \
  --flaw pause_insertion --severity medium --at-sec 1.5
```

### Step C: Extract features, detect, and evaluate
```bash
python -m flawbench features extract --all
python -m flawbench detect --all
python -m flawbench evaluate run --split all --run-id demo_run
```

---

## 4. Spoken Demo Script & Shot List (3–4 Minutes)

### Scene 1: Introduction & Problem Statement (0:00 – 0:35)
- **Application View**: Overview Page (`app.py`, default landing view).
- **User Action**: Show page title and system health metrics. Scroll through component status table showing all subsystems `[OK]`.
- **Narration**:
  > "Public speaking and vocal delivery coaching tools typically rely on black-box language models or subjective human ratings. When a speaker is told their pacing was 'a bit rushed' or their pauses were 'awkward', there is rarely an exact, verifiable ground truth to benchmark whether that feedback was correct.
  >
  > FlawBench introduces *Parametric Flaw Injection with Exact Ground Truth*. By starting from clean reference speech and applying controlled, mathematically defined acoustic transformations—such as pause insertions, local tempo modifications, and energy reductions—we produce flawed audio with microsecond-accurate ground-truth boundaries. This allows us to independently evaluate acoustic detectors with scientific rigor."

### Scene 2: The Dataset & Controlled Ground Truth (0:35 – 1:15)
- **Application View**: Sidebar -> Navigation: "Dataset & Technical Details".
- **User Action**: Inspect Registered Reference Recordings table and Generated Variants catalog. Point to the Ground Truth intervals `[start_sample, end_sample]`.
- **Narration**:
  > "Under Dataset and Technical Details, we see our registered source audio and controlled variants. Notice that each generated variation records the exact transformation parameters—such as an inserted 1.0-second pause or a 30% tempo deceleration—along with its integer sample interval and cryptographic SHA-256 checksum.
  >
  > Crucially, these ground-truth annotations are quarantined. The detection algorithms and participant-facing explanation layers never have access to these labels during inference."

### Scene 3: Acoustic Features & Differential Alignment (1:15 – 1:55)
- **Application View**: Sidebar -> Navigation: "Acoustic Features & Alignment".
- **User Action**: Select a variant. Observe the interactive Plotly multi-panel time-series: RMS energy, zero-crossing rate, spectral centroid, and MFCC contours. Show the Dynamic Time Warping (DTW) alignment curve.
- **Narration**:
  > "In the Acoustic Features panel, FlawBench computes frame-level representations over 25-millisecond windows with a 10-millisecond hop. Using Dynamic Time Warping on energy-normalized spectral coefficients, the system computes the non-linear temporal correspondence between the reference and the variant.
  >
  > This reveals acoustic anomalies: local warp slopes deviate from 1.0 when tempo shifts, and differential energy curves highlight localized volume dropouts or inserted silence."

### Scene 4: Independent Detection & Formal Evaluation (1:55 – 2:40)
- **Application View**: Sidebar -> Navigation: "Evaluation Lab".
- **User Action**: Select the held-out benchmark run (`heldout_rc`). Review the KPI cards (Micro F1, Precision, Recall, Mean IoU) and the IoU Localization Tolerance curve ($\theta \in [0.3, 0.9]$).
- **Narration**:
  > "In the Evaluation Lab, we benchmark predictions against frozen ground truth. Using deterministic 1-to-1 event matching with half-open temporal Intersection over Union (IoU), we score detection performance.
  >
  > On our held-out 24-sample benchmark cohort across 8 distinct sources, FlawBench's detector achieved a 0.957 micro-precision, 0.917 recall, and a 0.936 F1-score, with a 0.970 mean IoU localization accuracy. Pause insertion and energy reduction achieved 100% precision and recall; tempo detection achieved 75% recall, with misses confined to mild threshold-boundary cases. Negative identity controls demonstrated zero false alarms."

### Scene 5: Participant Dashboard & Evidence-Linked Explanations (2:40 – 3:30)
- **Application View**: Sidebar -> Navigation: "Analyze Speech" (Participant Dashboard).
- **User Action**: Select a sample. Click **Run Analysis**. Expand an Explanation Card. Demonstrate audio excerpt playback of the exact anomaly interval. Show report export buttons (Markdown, JSON, HTML, CSV).
- **Narration**:
  > "Finally, let's look at the participant experience. A user selects or uploads an audio file to compare against a reference. FlawBench runs the full pipeline in memory without persisting unauthorized audio.
  >
  > When candidate flaws are detected, the system generates deterministic Explanation Cards. Notice that every number in the explanation—such as 'silence duration: 1.00 s' or 'attenuation: -12.00 dB'—is strictly validated against physical acoustic measurements. The user can play back the exact flagged interval with context, examine suggested delivery adjustments, and export audit-ready reports in Markdown, JSON, HTML, or CSV."

### Scene 6: Limitations & Conclusion (3:30 – 4:00)
- **Application View**: Overview Page -> Project Limitations section.
- **User Action**: Highlight the documented limitations in the UI and README.
- **Narration**:
  > "In summary, FlawBench bridges speech coaching and objective ML benchmarking. To maintain academic and scientific integrity, we explicitly disclose: our benchmarks were measured on controlled acoustic fixtures; natural human speech validation requires future corpora with verified multi-annotator agreement; and detector evidence scores are heuristic acoustic deviations, not calibrated probabilities.
  >
  > FlawBench is release-ready, fully tested with 416 automated tests, and packaged with reproducible CLI workflows."

---

## 5. Troubleshooting & Demo Recovery Plan

| Scenario | Symptom | Immediate Recovery Action |
|:---|:---|:---|
| **Streamlit does not open** | Terminal port conflict | Run `streamlit run app.py --server.port 8502` |
| **No reference audio registered** | Dataset metrics show `0 sources` | Run `python -m flawbench init` then register the demo reference following Step A above |
| **Detector returns 0 candidates** | "No flaw candidates detected" message | Explain that zero candidates is a valid, honest result (not an error). Select a known variant (e.g. pause insertion at 1.5s) to show positive detections |
| **Audio playback disabled** | Browser autoplay policy blocks sound | Click the play icon on the Streamlit audio widget manually |
| **Export file path permission issue** | Cannot write report | Verify writable directories with `python -m flawbench check` |

---

## 6. What NOT to Claim During the Demonstration

1. **Do not claim human speech generalization**: The benchmark cohort consists of controlled synthetic/parametric fixtures. Do not claim FlawBench solves spontaneous conversation coaching.
2. **Do not describe evidence strength as probability**: It is a normalized distance metric relative to reference acoustic features, not a Bayesian posterior probability.
3. **Do not claim word-level phonetic alignment**: FlawBench currently operates on acoustic frame grids; forced phonetic alignment is a planned Phase 9 extension.
4. **Do not claim medical or psychological diagnostics**: FlawBench measures acoustic speech delivery anomalies, not clinical speech pathology or speaker emotional states.
