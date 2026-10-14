# FlawBench Evaluation Schema Specification (Phase 5)

## 1. Schema Versioning & Storage Layout

- **Canonical Schema Version**: `v1.0.0`
- **Output Storage Directory**: `data/evaluations/<evaluation_run_id>/`
- **Primary Report**: `data/evaluations/<evaluation_run_id>/evaluation_report.json`
- **Event-Level Audit Table**: `data/evaluations/<evaluation_run_id>/event_matches.jsonl`
- **Human-Readable Summary**: `data/evaluations/<evaluation_run_id>/report.md`
- **Catalog Manifest**: `data/evaluations/evaluations_manifest.jsonl`

---

## 2. Canonical Timeline & Bound Conventions

1. **Variant Recording Timeline**: All temporal comparisons operate strictly on the **modified variant recording's timeline**.
2. **Half-Open Integer Sample Bounds**:
   $$[start\_sample, end\_sample) \quad \text{where } 0 \le start\_sample < end\_sample \le total\_samples$$
   Integer sample indices are authoritative for calculating Temporal Intersection over Union (IoU).
3. **Temporal Intersection over Union (IoU)**:
   For prediction interval $P = [s_P, e_P)$ and ground-truth interval $G = [s_G, e_G)$:
   $$\text{IoU}(P, G) = \frac{\max(0, \min(e_P, e_G) - \max(s_P, s_G))}{\max(e_P, e_G) - \min(s_P, s_G)}$$
4. **Boundary Errors (Seconds)**:
   $$\text{StartErrorSec} = \frac{start\_sample_P - start\_sample_G}{sample\_rate}$$
   $$\text{EndErrorSec} = \frac{end\_sample_P - end\_sample_G}{sample\_rate}$$
   $$\text{DurationErrorSec} = (end\_sec_P - start\_sec_P) - (end\_sec_G - start\_sec_G)$$

---

## 3. Data Structures

### 3.1 `EventMatch` Record (`event_matches.jsonl`)

Every evaluated prediction and ground-truth event is recorded in the deterministic event match audit table:

| Field | Type | Description |
|---|---|---|
| `match_id` | string | Unique UUID identifying the match outcome |
| `sample_id` | string | Evaluated variant audio identifier |
| `source_id` | string | Originating reference speech identifier |
| `prediction_id` | string \| null | ID of predicted interval, or `null` for FN |
| `gt_id` | string \| null | ID of ground-truth annotation, or `null` for FP |
| `pred_flaw_type` | string \| null | Inferred flaw category (`pause_insertion`, `tempo_change`, `energy_reduction`, or `null`) |
| `gt_flaw_type` | string \| null | True injected flaw category, or `null` |
| `pred_interval_sample` | [int, int] \| null | Authoritative prediction sample bounds $[s_P, e_P)$ |
| `gt_interval_sample` | [int, int] \| null | Authoritative ground-truth sample bounds $[s_G, e_G)$ |
| `pred_interval_sec` | [float, float] \| null | Prediction timestamps in seconds |
| `gt_interval_sec` | [float, float] \| null | Ground-truth timestamps in seconds |
| `iou` | float | Temporal Intersection over Union $\in [0.0, 1.0]$ |
| `match_status` | string | One of: `"true_positive"`, `"false_positive"`, `"false_negative"`, `"misclassified"` |
| `category_match` | boolean | `true` if `pred_flaw_type == gt_flaw_type` |
| `start_error_sec` | float \| null | Signed start boundary error (or `null`) |
| `end_error_sec` | float \| null | Signed end boundary error (or `null`) |
| `dur_error_sec` | float \| null | Signed duration error (or `null`) |
| `unmatched_reason` | string \| null | Explanation for FP/FN (e.g. `"below_iou_threshold"`, `"unmatched_prediction"`) |

### 3.2 `CategoryMetrics` Record

Computed per flaw category (`pause_insertion`, `tempo_change`, `energy_reduction`):

| Field | Type | Description |
|---|---|---|
| `category` | string | Flaw category name |
| `tp` | integer | Count of qualifying true positive event matches |
| `fp` | integer | Count of spurious or non-matching predictions |
| `fn` | integer | Count of unlocalised or missed ground-truth events |
| `gt_count` | integer | Total ground-truth events for this category |
| `pred_count` | integer | Total predictions generated for this category |
| `precision` | float \| null | $TP / (TP + FP)$ (null when denominator is 0) |
| `recall` | float \| null | $TP / (TP + FN)$ (null when denominator is 0) |
| `f1` | float \| null | $2 \cdot P \cdot R / (P + R)$ (null when undefined) |
| `mean_iou` | float \| null | Mean IoU across true positive matched events |
| `median_iou` | float \| null | Median IoU across true positive matched events |
| `mean_abs_start_error_sec` | float \| null | Mean absolute start error across matched events |
| `mean_abs_end_error_sec` | float \| null | Mean absolute end error across matched events |
| `mean_abs_dur_error_sec` | float \| null | Mean absolute duration error across matched events |

### 3.3 `AggregateMetrics` Record

| Field | Type | Description |
|---|---|---|
| `micro_precision` | float \| null | Pooled $\sum TP / (\sum TP + \sum FP)$ |
| `micro_recall` | float \| null | Pooled $\sum TP / (\sum TP + \sum FN)$ |
| `micro_f1` | float \| null | Harmonic mean of micro precision and recall |
| `macro_precision` | float \| null | Unweighted mean of valid per-category precisions |
| `macro_recall` | float \| null | Unweighted mean of valid per-category recalls |
| `macro_f1` | float \| null | Unweighted mean of valid per-category F1 scores |
| `mean_iou` | float \| null | Global average IoU over all matched events |
| `total_gt` | integer | Total ground-truth flaw events evaluated |
| `total_pred` | integer | Total predictions evaluated |
| `total_tp` | integer | Global true positive matches |
| `total_fp` | integer | Global false positive predictions |
| `total_fn` | integer | Global false negative missed events |

### 3.4 `LocalizationPoint` Record

Samples recall across configurable IoU thresholds:

| Field | Type | Description |
|---|---|---|
| `iou_threshold` | float | Evaluation threshold $\theta_{\text{IoU}}$ (e.g., 0.3, 0.5, 0.7, 0.9) |
| `gt_count` | integer | Total ground-truth events in cohort |
| `matched_count` | integer | Events matched with IoU $\ge \theta_{\text{IoU}}$ and category agreement |
| `recall` | float \| null | $\text{matched\_count} / \text{gt\_count}$ |

### 3.5 `ConfusionMatrix` Record

Event-level category attribution matrix:

| Field | Type | Description |
|---|---|---|
| `matrix` | dict[string, dict[string, int]] | Row: Ground-Truth, Column: Predicted |
| `classes` | list[string] | Canonical classes evaluated |
| `missed_events` | dict[string, int] | Ground-truth events with no prediction match |
| `spurious_detections`| dict[string, int] | Predictions with no ground-truth match |

### 3.6 `ControlMetrics` Record

Audits false-positive rate on clean reference self-comparisons:

| Field | Type | Description |
|---|---|---|
| `total_controls` | integer | Total negative controls evaluated |
| `clean_controls_evaluated` | integer | Controls successfully executed through detection engine |
| `controls_with_predictions` | integer | Controls that yielded $\ge 1$ spurious prediction |
| `total_false_positives` | integer | Total spurious candidates generated on clean audio |
| `false_positive_rate` | float \| null | Fraction of clean controls producing false alarms |

---

## 4. `EvaluationReport` Artifact Payload Structure

The top-level JSON artifact (`data/evaluations/<evaluation_run_id>/evaluation_report.json`) encapsulates the complete evaluation state:

```json
{
  "schema_version": "v1.0.0",
  "evaluation_run_id": "eval-20261015-010530-ab12cd",
  "created_at": "2026-10-15T01:05:30.123456+00:00",
  "flawbench_version": "0.5.0",
  "config_fingerprint": "a1b2c3d4e5f67890",
  "dataset_manifest_fingerprint": "d4e5f67890a1b2c3",
  "detections_manifest_fingerprint": "f67890a1b2c3d4e5",
  "split_name": "test",
  "evaluation_mode": "held_out",
  "sample_ids": ["smp-1", "smp-2"],
  "source_ids": ["src-1"],
  "matching_iou_threshold": 0.5,
  "aggregate": {
    "micro_precision": 0.857,
    "micro_recall": 1.0,
    "micro_f1": 0.923,
    "macro_precision": 0.889,
    "macro_recall": 1.0,
    "macro_f1": 0.933,
    "mean_iou": 0.784,
    "total_gt": 6,
    "total_pred": 7,
    "total_tp": 6,
    "total_fp": 1,
    "total_fn": 0
  },
  "per_category": {
    "pause_insertion": {
      "category": "pause_insertion",
      "tp": 2, "fp": 0, "fn": 0,
      "gt_count": 2, "pred_count": 2,
      "precision": 1.0, "recall": 1.0, "f1": 1.0,
      "mean_iou": 0.885, "median_iou": 0.885,
      "mean_abs_start_error_sec": 0.015,
      "mean_abs_end_error_sec": 0.020,
      "mean_abs_dur_error_sec": 0.005
    }
  },
  "localization_curve": [
    {"iou_threshold": 0.3, "gt_count": 6, "matched_count": 6, "recall": 1.0},
    {"iou_threshold": 0.5, "gt_count": 6, "matched_count": 6, "recall": 1.0},
    {"iou_threshold": 0.7, "gt_count": 6, "matched_count": 4, "recall": 0.667},
    {"iou_threshold": 0.9, "gt_count": 6, "matched_count": 1, "recall": 0.167}
  ],
  "confusion_matrix": {
    "matrix": {
      "pause_insertion": {"pause_insertion": 2, "tempo_change": 0, "energy_reduction": 0},
      "tempo_change": {"pause_insertion": 0, "tempo_change": 2, "energy_reduction": 0},
      "energy_reduction": {"pause_insertion": 0, "tempo_change": 0, "energy_reduction": 2}
    },
    "classes": ["pause_insertion", "tempo_change", "energy_reduction"],
    "missed_events": {"pause_insertion": 0, "tempo_change": 0, "energy_reduction": 0},
    "spurious_detections": {"pause_insertion": 0, "tempo_change": 1, "energy_reduction": 0}
  },
  "controls": {
    "total_controls": 2,
    "clean_controls_evaluated": 2,
    "controls_with_predictions": 0,
    "total_false_positives": 0,
    "false_positive_rate": 0.0
  },
  "bootstrap_ci": {
    "replicates": 100,
    "seed": 42,
    "metrics": {
      "micro_f1": {"ci_lower_2_5": 0.85, "ci_upper_97_5": 0.98}
    }
  },
  "sample_evaluations": [],
  "metadata": {
    "data_nature": "synthetic_fixtures",
    "notes": "Evaluation conducted strictly on synthetic tone fixtures. Generalization to natural speech is not claimed."
  }
}
```
