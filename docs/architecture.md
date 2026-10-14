# Architecture

## End-to-end conceptual architecture (planned beyond Phase 1)

```mermaid
flowchart LR
    R[Reference recording + transcript] --> G[Flaw injection engine - Phase 2]
    G --> V[Modified recording]
    G --> GT[Ground-truth metadata: flaw, severity, interval, seed]
    R --> F[Feature extraction - Phase 3]
    V --> F
    F --> D[Label-blind detector - Phase 4]
    D --> P[Predicted intervals]
    P --> E[Evaluation - Phase 5]
    GT --> E
    E --> U[Dashboard and exports - Phase 6]
```

Phase 1 implements the foundation underneath every box: configuration, storage, audio input,
logging, and provenance.

## Phase 1 modules

| Module | Responsibility |
|---|---|
| `config.py` | Single definition of every setting (`_FIELDS`), loading, precedence, validation, immutable `Config` |
| `paths.py` | Resolves directories from `Config`; input/output roles; real writability checks; traversal-safe `resolve_within` |
| `validation.py` | Pure checks: extension, size, stream properties, non-finite samples |
| `audio/loader.py` | Decoder-verified `probe_audio` (header only) and `load_audio` |
| `logging_config.py` | One managed handler under the `flawbench` namespace; `log_failure` |
| `exceptions.py` | Errors with `user_message` (UI-safe) and `detail` (logs) |
| `provenance.py` | Versions, platform, config hash, optional seed |
| `status.py` | Inspects the running system for the status panel and CLI |
| `roadmap.py` | Informational phase list |
| `ui/components.py` | Rendering only |

## Configuration flow

`defaults (_FIELDS)` -> `config file` -> `environment (.env under real env)` -> `overrides` ->
coercion and validation -> frozen `Config` -> `ProjectPaths(config)`. Every other module receives the
`Config` (or one of its sections) by argument; none reads the environment or defines paths itself.

## Audio-input flow

```mermaid
flowchart TD
    S[Path, bytes, or stream + filename] --> C1[Extension, existence, size: no file read]
    C1 --> H[Decoder header probe: format, rate, channels, frames]
    H --> C2[Content format allowed? rate and channel limits?]
    C2 --> D[Decode to float32 array shape channels x samples]
    D --> C3[Header/decoded length match; finite samples]
    C3 --> L[LoadedAudio: samples, sample_rate, metadata]
```

The extension is only a first filter; the decoder decides what the bytes are, and the detected
container must match an allowed format. Nothing is resampled, normalized, trimmed, or downmixed.

## Error-handling boundaries

- Library modules raise `FlawBenchError` subclasses and never print or show UI.
- The UI boundary (`app.py`) catches `AudioInputError` and shows `public_message()`; technical
  detail goes to logs. Only `diagnostics.expose_error_details` adds detail to the UI, and it is
  rejected when `app.environment` is `production`.
- An unexpected exception at the UI boundary is logged with a traceback and replaced by a generic
  message.

## Storage conventions

| Location | Role | Behavior |
|---|---|---|
| `storage.reference_dir` | input | read-only; never created or written |
| `storage.generated_dir`, `storage.metadata_dir` | output | created and write-tested |
| `storage.output_dir` (+ `reports/`, `figures/`) | output | created and write-tested |

Uploads are never written to disk.

## Planned interfaces for later phases (contracts, not code)

- Phase 2: a generator takes `LoadedAudio`, a flaw spec, and a seed; returns modified samples plus a
  ground-truth record (flaw type, severity, interval in seconds and samples in the output
  timeline), saved under `generated_dir` / `metadata_dir` with `build_provenance(config, seed=...)`.
- Phase 3 to 4: detectors consume `LoadedAudio` only, never the ground-truth record.
- Phase 5: evaluation joins predictions and records by recording id.


## Phase 2: dataset and ground-truth engine

```mermaid
flowchart LR
    CLI[CLI / Streamlit panel] --> G[DatasetGenerator]
    G --> R[SourceRegistry: sources.jsonl]
    G --> I[Injectors: pause, tempo, energy]
    I --> W[Verified temp WAV, then atomic commit]
    W --> M[manifest.jsonl: annotation + provenance]
    G --> RUN[runs.jsonl]
    M --> V[Validator: files, checksums, deep content]
    M --> S[Splits by source]
    M --> E[Export: permitted sources only]
```

Module responsibilities: `params` (flaw ids, parameter specs, bounds, profile validation; shared with
config), `schema` (record shapes and timeline helpers), `io` (checksums, atomic writes, JSONL),
`registry` (rights-gated registration, integrity checks), `injectors` (pure array transforms with measured
annotations), `plan` (plan parsing and expansion), `generator` (seed derivation, identity, transaction,
batch), `validator`, `splits`, `export`.

Boundaries: generation writes audio plus annotation; detectors receive audio and label-free features only; evaluation joins
predictions and annotations afterwards. Nothing in Phase 2 reads annotations to produce predictions.

Transaction per variant: compute in memory, write temp file, decode it back and compare, re-verify the source
checksum, rename into place, rewrite the manifest atomically; on manifest failure the new file is removed (or
the previous one restored). A crash between rename and manifest leaves an orphan that the validator reports and
that regeneration adopts only if byte-identical.

## Phase 3: feature extraction & time-series analysis

```mermaid
flowchart LR
    A[LoadedAudio: ref / var] --> EX[FeatureExtractor: canonical grid]
    EX --> F1[RMS & dB Energy]
    EX --> F2[Zero-Crossing Rate]
    EX --> F3[Spectral Centroid, Bandwidth, Rolloff]
    EX --> F4[MFCCs 13-dim]
    EX --> F5[F0 Pitch: pyin / yin]
    EX --> FS[FeatureStore: .npz + .json]
    FS --> DTW[Dynamic Time Warping Alignment: MFCC Euclidean]
    DTW --> CMP[FeatureComparisonResult: warp slope, delta RMS, pitch diff]
```

Canonical frame grid: 25 ms frame length, 10 ms hop interval, centered STFT. Time index for frame `i`: `t = i * hop_seconds`. Feature artifacts are cached with audio SHA-256 and configuration fingerprint.

## Phase 4: independent speech flaw detection & ground-truth isolation

Phase 4 introduces an independent, label-blind detection layer that inspects variant speech alongside the reference and produces timestamped flaw hypotheses with measured acoustic evidence.

### Architectural Ground-Truth Isolation

```mermaid
flowchart TD
    subgraph Phase 2 Dataset Generation
        REF_REC[Reference Audio WAV]
        VAR_REC[Variant Audio WAV]
        GT[manifest.jsonl: flaw_type, severity, output_interval, source_interval]
    end

    subgraph Phase 4 Independent Detection Pipeline
        LF_IN[DetectionInput: label-free audio, features & DTW comparison]
        REF_REC --> LF_IN
        VAR_REC --> LF_IN
        
        LF_IN --> DET_P[PauseDetector: DTW freeze / silence diff]
        LF_IN --> DET_T[TempoDetector: local warp slope ratio]
        LF_IN --> DET_E[EnergyReductionDetector: aligned delta RMS]
        
        DET_P --> PP[PostProcessor: gap merge, duration filter, boundary check]
        DET_T --> PP
        DET_E --> PP
        
        PP --> RES[DetectionResult: predictions, sample intervals, evidence]
        RES --> STORE[DetectionStore: detections/ + manifest.jsonl]
    end

    subgraph Phase 5 Formal Evaluation (Deferred)
        EVAL[Evaluation Engine]
        GT -.->|Labels joined ONLY at evaluation time| EVAL
        STORE -.->|Predictions joined by sample_id| EVAL
        EVAL --> METRICS[IoU, Precision, Recall, F1]
    end

    style GT fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    style LF_IN fill:#ccffcc,stroke:#009900,stroke-width:2px
```

## Phase 5: formal evaluation, temporal localization & benchmarking

Phase 5 introduces a reproducible evaluation engine that benchmarks independently generated predictions against ground-truth annotations post-inference.

### Architectural Evaluation Flow

```mermaid
flowchart TD
    subgraph Phase 4 Frozen Predictions
        DET_STORE[DetectionStore: data/detections/detections_manifest.jsonl]
        DET_STORE --> FREEZE[Freeze Eligible Predictions]
    end

    subgraph Phase 2 Ground Truth
        GT_MAN[data/metadata/manifest.jsonl]
        SPLITS[Source-Level Splits: test, validation, train]
    end

    subgraph Phase 5 Evaluation Engine
        FREEZE --> COHORT[EvaluationCohort: join by sample_id & verify audio checksums]
        GT_MAN --> COHORT
        SPLITS --> COHORT
        
        COHORT --> MATCHER[1-to-1 Bipartite Matcher: Half-Open Sample IoU]
        MATCHER --> MATCH_TABLE[event_matches.jsonl]
        
        MATCH_TABLE --> METRICS[Metrics Calculator]
        METRICS --> CAT_METRICS[Per-Category P/R/F1 & Boundary MAE]
        METRICS --> AGG_METRICS[Micro & Macro Averages]
        METRICS --> LOC_CURVE[Localization-Tolerance Curve: IoU 0.3..0.9]
        METRICS --> CONF_MAT[Event-Level Confusion Matrix & Error Attribution]
        
        COHORT --> CONTROLS[Clean Identity Controls: Ref vs Ref]
        CONTROLS --> CTRL_METRICS[False-Positive Rate Audit]
        
        COHORT --> BOOTSTRAP[Source-Level Cluster Bootstrap 95% CIs]
        
        CAT_METRICS --> RUNNER[EvaluationRunner & EvaluationStore]
        AGG_METRICS --> RUNNER
        LOC_CURVE --> RUNNER
        CONF_MAT --> RUNNER
        CTRL_METRICS --> RUNNER
        BOOTSTRAP --> RUNNER
        
        RUNNER --> ARTIFACTS[outputs: evaluation_report.json, event_matches.jsonl, report.md]
        RUNNER --> EVAL_MAN[evaluations_manifest.jsonl]
    end
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `schema.py` | Defines `EventMatch`, `CategoryMetrics`, `AggregateMetrics`, `LocalizationPoint`, `ConfusionMatrix`, `ControlMetrics`, `EvaluationReport` |
| `matching.py` | Half-open integer sample IoU $\frac{|P \cap G|}{|P \cup G|}$; deterministic bipartite greedy 1-to-1 matching |
| `metrics.py` | Computes precision, recall, F1, boundary error MAE, and localization-tolerance curves |
| `confusion.py` | Constructs event-level category attribution matrix (correct, confused, missed, spurious) |
| `controls.py` | Evaluates clean identity self-comparisons to audit false-positive susceptibility |
| `bootstrap.py` | Source-level cluster bootstrapping for empirical 95% confidence intervals |
| `inputs.py` | Validates split membership, joins predictions and annotations by `sample_id`, audits checksums |
| `store.py` | Atomic persistence under `data/evaluations/<run_id>/`, cataloging in `evaluations_manifest.jsonl` |
| `validator.py` | Audits evaluation reports for metric bounds $[0, 1]$, non-negative counts, and file integrity |
| `reports.py` | Generates auditable GitHub-flavored markdown benchmark summaries |
| `runner.py` | Coordinates the full evaluation pipeline and handles single-sample or batch cohorts |
