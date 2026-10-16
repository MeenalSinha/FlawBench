# FlawBench Evaluation Report: `heldout_rc`

- **Generated At**: 2026-10-16T03:06:46.567050+00:00
- **Detector Version**: 5.0
- **Detector Config Fingerprint**: `ee4c7f85d2c2ad13`
- **FlawBench Version**: 0.1.0
- **Evaluation Split**: `all`
- **Cohort Size**: 24 samples across 8 sources
- **Audio Provenance**: Audio origin is taken from registration metadata only and is not independently verified. The registry has no synthetic source type, so synthetic fixtures are NOT identified by this field; treat results as unvalidated for natural speech unless annotated, licensed speech was used.

---

## 1. Executive Summary

| Metric | Micro-Averaged Value | Macro-Averaged Value |
| :--- | :---: | :---: |
| **Precision** | 0.957 | 0.952 |
| **Recall** | 0.917 | 0.917 |
| **F1-Score** | 0.936 | 0.933 |
| **Mean Matched IoU** | 0.970 | - |

## 2. Per-Category Classification Performance

| Category | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `energy_reduction` | 8 | 8 | 8 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.993 |
| `pause_insertion` | 8 | 8 | 8 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.979 |
| `tempo_change` | 8 | 7 | 6 | 1 | 2 | 0.857 | 0.750 | 0.800 | 0.926 |

## 3. Temporal Localization & Boundary Errors

| Category | Mean Start Err (s) | Mean End Err (s) | Mean Dur Err (s) |
| :--- | :---: | :---: | :---: |
| `energy_reduction` | 0.0100 | 0.0050 | 0.0112 |
| `pause_insertion` | 0.0088 | 0.0075 | 0.0163 |
| `tempo_change` | 0.0692 | 0.0578 | 0.1222 |

## 4. Localization-Tolerance Curve

Shows detection recall across varying temporal IoU overlap thresholds:

| IoU Threshold | Total Ground Truth | Matched Events | Recall |
| :---: | :---: | :---: | :---: |
| 0.30 | 24 | 22 | 0.917 |
| 0.50 | 24 | 22 | 0.917 |
| 0.70 | 24 | 22 | 0.917 |
| 0.90 | 24 | 20 | 0.833 |

## 5. Event-Level Confusion Matrix

Rows represent Ground Truth categories (or `'none'` for spurious false alarms).
Columns represent Predicted categories (or `'missed'` for undetected events).

| GT \ Pred | `pause_insertion` | `tempo_change` | `energy_reduction` | `missed` |
| :--- | :---: | :---: | :---: | :---: |
| `pause_insertion` | 8 | 0 | 0 | 0 |
| `tempo_change` | 0 | 6 | 0 | 2 |
| `energy_reduction` | 0 | 0 | 8 | 0 |
| `none` | 0 | 1 | 0 | 0 |

## 6. Clean Negative Controls Analysis

- **Evaluated Identity Controls**: 8
- **Controls Triggering Predictions**: 0
- **Total Spurious Predictions**: 0
- **False Alarm Rate per Clean Control**: 0.000

---
*Report generated automatically by FlawBench Evaluation Engine (Phase 5).*