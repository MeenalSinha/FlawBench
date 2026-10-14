# FlawBench Detection Schema Specification (Phase 4)

## 1. Schema Versioning

- **Canonical Schema Version**: `v1.0.0`
- **Output Storage**: `data/detections/<sample_id>_detection.json`
- **Manifest File**: `data/detections/detections_manifest.jsonl`

---

## 2. Canonical Timeline & Bound Conventions

1. **Variant Timeline**: All detected flaw intervals are strictly positioned on the **modified variant recording's timeline**.
2. **Integer Sample Indices**: The authoritative temporal representation is given by half-open integer sample bounds:
   $$[start\_sample, end\_sample) \quad \text{where } 0 \le start\_sample < end\_sample \le total\_samples$$
3. **Continuous Timestamp Calculation**: Timestamps are computed directly from the true audio sample rate:
   $$start\_sec = \frac{start\_sample}{sample\_rate}, \quad end\_sec = \frac{end\_sample}{sample\_rate}, \quad duration\_sec = end\_sec - start\_sec$$
4. **Reference Interval Correspondence**: When unambiguous DTW alignment exists, the corresponding reference region is provided separately in `ref_start_sec` and `ref_end_sec`. If ambiguous or nonexistent, these fields remain `null`.

---

## 3. Data Structures

### 3.1 `FlawPrediction` Record

| Field | Type | Description |
|---|---|---|
| `prediction_id` | string | Unique UUID identifying the predicted interval |
| `flaw_type` | string | One of: `"pause_insertion"`, `"tempo_change"`, `"energy_reduction"` |
| `start_sample` | integer | Starting sample on variant timeline (inclusive) |
| `end_sample` | integer | Ending sample on variant timeline (exclusive) |
| `start_sec` | float | Starting time in seconds on variant timeline |
| `end_sec` | float | Ending time in seconds on variant timeline |
| `duration_sec` | float | Duration of the candidate region in seconds |
| `ref_start_sec` | float \| null | Aligned reference start time in seconds (or null) |
| `ref_end_sec` | float \| null | Aligned reference end time in seconds (or null) |
| `evidence` | object | Dictionary of measured acoustic feature values |
| `evidence_strength` | float | Normalized evidence strength $\in [0.0, 1.0]$ |
| `detector_id` | string | Algorithm identifier (e.g., `"pause_freeze_detector_v1"`) |
| `warnings` | list[string] | Diagnostic uncertainty indicators |

### 3.2 `DetectionResult` Artifact

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Version of detection schema (`"v1.0.0"`) |
| `run_id` | string | Execution run UUID |
| `sample_id` | string | Identifier of the evaluated variant recording |
| `source_id` | string | Identifier of the registered reference source |
| `ref_checksum` | string | SHA-256 hex digest of the reference audio |
| `variant_checksum` | string | SHA-256 hex digest of the variant audio |
| `detector_version` | string | Semantic version of detection engine |
| `config_fingerprint` | string | SHA-256 hash prefix of effective detector configuration |
| `sample_rate` | integer | Variant sample rate in Hertz |
| `total_samples` | integer | Total number of samples in variant audio |
| `duration_seconds` | float | Total duration of variant audio in seconds |
| `predictions` | list[FlawPrediction] | Chronologically sorted list of predictions |
| `raw_candidates_count`| integer | Number of pre-consolidation candidates |
| `created_at` | string | ISO 8601 UTC creation timestamp |
| `metadata` | object | Provenance and execution metadata (label-free) |

---

## 4. Illustrative Example Payload

*(Note: The following is an illustrative example demonstrating JSON structure; it does not represent actual test tones).*

```json
{
  "schema_version": "v1.0.0",
  "run_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "sample_id": "smp-489a690703c3",
  "source_id": "src-185d8ab5d57b",
  "ref_checksum": "a3b2c1d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8",
  "variant_checksum": "f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e7",
  "detector_version": "0.4.0",
  "config_fingerprint": "7d9b4f2c1a8e3d5b",
  "sample_rate": 16000,
  "total_samples": 80000,
  "duration_seconds": 5.0,
  "predictions": [
    {
      "prediction_id": "pred-4a8b2c1d-1",
      "flaw_type": "pause_insertion",
      "start_sample": 32000,
      "end_sample": 48000,
      "start_sec": 2.0,
      "end_sec": 3.0,
      "duration_sec": 1.0,
      "ref_start_sec": 2.0,
      "ref_end_sec": 2.01,
      "evidence": {
        "variant_pause_dur_sec": 1.0,
        "ref_pause_dur_sec": 0.01,
        "duration_diff_sec": 0.99,
        "min_variant_rms_db": -58.4,
        "mean_variant_rms_db": -55.1,
        "threshold_pause_dur_sec": 0.3
      },
      "evidence_strength": 0.95,
      "detector_id": "pause_freeze_detector_v1",
      "warnings": []
    }
  ],
  "raw_candidates_count": 1,
  "created_at": "2026-10-15T00:15:00.000000+00:00",
  "metadata": {
    "flawbench_version": "0.1.0",
    "platform": "Windows-11"
  }
}
```
