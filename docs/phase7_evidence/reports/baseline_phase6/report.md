# FlawBench Evaluation Report: `baseline_phase6`

- **Generated At**: 2026-10-15T10:03:01.381742+00:00
- **FlawBench Version**: 0.1.0
- **Evaluation Split**: `all`
- **Cohort Size**: 12 samples across 4 sources
- **Audio Provenance**: Source types declared at registration as non-synthetic (not independently verified)

---

## 1. Executive Summary

| Metric | Micro-Averaged Value | Macro-Averaged Value |
| :--- | :---: | :---: |
| **Precision** | 0.471 | 0.333 |
| **Recall** | 0.667 | 0.667 |
| **F1-Score** | 0.552 | 0.667 |
| **Mean Matched IoU** | 0.988 | - |

## 2. Per-Category Classification Performance

| Category | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `energy_reduction` | 4 | 8 | 4 | 4 | 0 | 0.500 | 1.000 | 0.667 | 0.991 |
| `pause_insertion` | 4 | 8 | 4 | 4 | 0 | 0.500 | 1.000 | 0.667 | 0.985 |
| `tempo_change` | 4 | 1 | 0 | 1 | 4 | 0.000 | 0.000 | N/A | N/A |

## 3. Temporal Localization & Boundary Errors

| Category | Mean Start Err (s) | Mean End Err (s) | Mean Dur Err (s) |
| :--- | :---: | :---: | :---: |
| `energy_reduction` | 0.0125 | 0.0063 | 0.0175 |
| `pause_insertion` | 0.0075 | 0.0075 | 0.0150 |
| `tempo_change` | N/A | N/A | N/A |

## 4. Localization-Tolerance Curve

Shows detection recall across varying temporal IoU overlap thresholds:

| IoU Threshold | Total Ground Truth | Matched Events | Recall |
| :---: | :---: | :---: | :---: |
| 0.30 | 12 | 8 | 0.667 |
| 0.50 | 12 | 8 | 0.667 |
| 0.70 | 12 | 8 | 0.667 |
| 0.90 | 12 | 8 | 0.667 |

## 5. Event-Level Confusion Matrix

Rows represent Ground Truth categories (or `'none'` for spurious false alarms).
Columns represent Predicted categories (or `'missed'` for undetected events).

| GT \ Pred | `pause_insertion` | `tempo_change` | `energy_reduction` | `missed` |
| :--- | :---: | :---: | :---: | :---: |
| `pause_insertion` | 4 | 0 | 0 | 0 |
| `tempo_change` | 0 | 0 | 0 | 4 |
| `energy_reduction` | 0 | 0 | 4 | 0 |
| `none` | 4 | 1 | 4 | 0 |

## 6. Clean Negative Controls Analysis

- **Evaluated Identity Controls**: 4
- **Controls Triggering Predictions**: 0
- **Total Spurious Predictions**: 0
- **False Alarm Rate per Clean Control**: 0.000

---
*Report generated automatically by FlawBench Evaluation Engine (Phase 5).*