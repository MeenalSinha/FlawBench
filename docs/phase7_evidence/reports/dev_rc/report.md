# FlawBench Evaluation Report: `dev_rc`

- **Generated At**: 2026-10-16T03:07:04.176135+00:00
- **Detector Version**: 5.0
- **Detector Config Fingerprint**: `ee4c7f85d2c2ad13`
- **FlawBench Version**: 0.1.0
- **Evaluation Split**: `all`
- **Cohort Size**: 12 samples across 4 sources
- **Audio Provenance**: Audio origin is taken from registration metadata only and is not independently verified. The registry has no synthetic source type, so synthetic fixtures are NOT identified by this field; treat results as unvalidated for natural speech unless annotated, licensed speech was used.

---

## 1. Executive Summary

| Metric | Micro-Averaged Value | Macro-Averaged Value |
| :--- | :---: | :---: |
| **Precision** | 1.000 | 1.000 |
| **Recall** | 1.000 | 1.000 |
| **F1-Score** | 1.000 | 1.000 |
| **Mean Matched IoU** | 0.975 | - |

## 2. Per-Category Classification Performance

| Category | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `energy_reduction` | 4 | 4 | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.992 |
| `pause_insertion` | 4 | 4 | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.980 |
| `tempo_change` | 4 | 4 | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.954 |

## 3. Temporal Localization & Boundary Errors

| Category | Mean Start Err (s) | Mean End Err (s) | Mean Dur Err (s) |
| :--- | :---: | :---: | :---: |
| `energy_reduction` | 0.0025 | 0.0138 | 0.0150 |
| `pause_insertion` | 0.0125 | 0.0075 | 0.0200 |
| `tempo_change` | 0.0425 | 0.0896 | 0.0471 |

## 4. Localization-Tolerance Curve

Shows detection recall across varying temporal IoU overlap thresholds:

| IoU Threshold | Total Ground Truth | Matched Events | Recall |
| :---: | :---: | :---: | :---: |
| 0.30 | 12 | 12 | 1.000 |
| 0.50 | 12 | 12 | 1.000 |
| 0.70 | 12 | 12 | 1.000 |
| 0.90 | 12 | 12 | 1.000 |

## 5. Event-Level Confusion Matrix

Rows represent Ground Truth categories (or `'none'` for spurious false alarms).
Columns represent Predicted categories (or `'missed'` for undetected events).

| GT \ Pred | `pause_insertion` | `tempo_change` | `energy_reduction` | `missed` |
| :--- | :---: | :---: | :---: | :---: |
| `pause_insertion` | 4 | 0 | 0 | 0 |
| `tempo_change` | 0 | 4 | 0 | 0 |
| `energy_reduction` | 0 | 0 | 4 | 0 |
| `none` | 0 | 0 | 0 | 0 |

## 6. Clean Negative Controls Analysis

- **Evaluated Identity Controls**: 4
- **Controls Triggering Predictions**: 0
- **Total Spurious Predictions**: 0
- **False Alarm Rate per Clean Control**: 0.000

---
*Report generated automatically by FlawBench Evaluation Engine (Phase 5).*