# FlawBench Evaluation Report: `heldout_baseline`

- **Generated At**: 2026-10-15T10:09:20.001083+00:00
- **FlawBench Version**: 0.1.0
- **Evaluation Split**: `all`
- **Cohort Size**: 24 samples across 8 sources
- **Audio Provenance**: Source types declared at registration as non-synthetic (not independently verified)

---

## 1. Executive Summary

| Metric | Micro-Averaged Value | Macro-Averaged Value |
| :--- | :---: | :---: |
| **Precision** | 0.615 | 0.650 |
| **Recall** | 0.667 | 0.667 |
| **F1-Score** | 0.640 | 0.778 |
| **Mean Matched IoU** | 0.986 | - |

## 2. Per-Category Classification Performance

| Category | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `energy_reduction` | 8 | 16 | 8 | 8 | 0 | 0.500 | 1.000 | 0.667 | 0.993 |
| `pause_insertion` | 8 | 10 | 8 | 2 | 0 | 0.800 | 1.000 | 0.889 | 0.979 |
| `tempo_change` | 8 | 0 | 0 | 0 | 8 | N/A | 0.000 | N/A | N/A |

## 3. Temporal Localization & Boundary Errors

| Category | Mean Start Err (s) | Mean End Err (s) | Mean Dur Err (s) |
| :--- | :---: | :---: | :---: |
| `energy_reduction` | 0.0113 | 0.0031 | 0.0100 |
| `pause_insertion` | 0.0088 | 0.0075 | 0.0163 |
| `tempo_change` | N/A | N/A | N/A |

## 4. Localization-Tolerance Curve

Shows detection recall across varying temporal IoU overlap thresholds:

| IoU Threshold | Total Ground Truth | Matched Events | Recall |
| :---: | :---: | :---: | :---: |
| 0.30 | 24 | 16 | 0.667 |
| 0.50 | 24 | 16 | 0.667 |
| 0.70 | 24 | 16 | 0.667 |
| 0.90 | 24 | 16 | 0.667 |

## 5. Event-Level Confusion Matrix

Rows represent Ground Truth categories (or `'none'` for spurious false alarms).
Columns represent Predicted categories (or `'missed'` for undetected events).

| GT \ Pred | `pause_insertion` | `tempo_change` | `energy_reduction` | `missed` |
| :--- | :---: | :---: | :---: | :---: |
| `pause_insertion` | 8 | 0 | 0 | 0 |
| `tempo_change` | 0 | 0 | 0 | 8 |
| `energy_reduction` | 0 | 0 | 8 | 0 |
| `none` | 2 | 0 | 8 | 0 |

## 6. Clean Negative Controls Analysis

- **Evaluated Identity Controls**: 8
- **Controls Triggering Predictions**: 0
- **Total Spurious Predictions**: 0
- **False Alarm Rate per Clean Control**: 0.000

---
*Report generated automatically by FlawBench Evaluation Engine (Phase 5).*