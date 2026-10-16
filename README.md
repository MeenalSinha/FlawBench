# FlawBench — Evidence-Based Speech Flaw Analysis

**v0.1.0 — Release Candidate 1**  
Multimodal AI Hackathon 2026, Track C: Contrastive Speech Analytics and Temporal Flaw Grounding.

## Concept

FlawBench evaluates speech-delivery analysis without relying on subjective scores. Clean reference
recordings are turned into controlled variations; every variation carries metadata describing the
flaw introduced, its severity, and the exact interval affected. An analyzer then examines the
recordings without access to those labels, and its predictions are scored against the truth.

**Core innovation: parametric flaw injection with exact ground truth.**

Planned flaw categories: excessive pauses, local pacing deviations, local energy reductions, and
(as a future extension) reduced pitch variation.

### Eventual workflow (planned)

1. Import a reference recording with its transcript and metadata.
2. Generate controlled variations by injecting configurable flaws.
3. Record exact transformation parameters and affected intervals.
4. Extract acoustic features from reference and modified recordings.
5. Detect candidate flaws independently of the labels.
6. Compare predicted intervals with ground truth.
7. Compute evaluation metrics.
8. Show audio comparisons, feature graphs, detections, and explanations.
9. Export reproducible results.

## Current scope: Phases 1 to 6 (implemented)

| Capability | Status |
|---|---|
| Centralized validated configuration (file, environment, overrides) | Implemented |
| Path management (read-only input vs writable output) | Implemented |
| Audio validation and loading (decoder-verified, no implicit transforms) | Implemented |
| Structured, idempotent logging | Implemented |
| Streamlit app: real status, upload inspection, roadmap | Implemented |
| Run provenance helpers (versions, config hash, seed field) | Implemented |
| Diagnostics CLI (`python -m flawbench`) | Implemented |
| Reference registry with transcript and rights metadata, SHA-256 integrity | Implemented (Phase 2) |
| Three injectors: pause insertion, local tempo change, local energy reduction | Implemented (Phase 2) |
| Exact ground truth on the generated timeline, JSONL manifest, run log, provenance | Implemented (Phase 2) |
| Deterministic seeded placement, collision-safe transactional generation | Implemented (Phase 2) |
| Dataset validator (deep content verification), source-level splits, safe export | Implemented (Phase 2) |
| Canonical frame grid acoustic features (RMS energy, ZCR, Spectral, MFCCs, F0 pitch) | Implemented (Phase 3) |
| Dynamic Time Warping (DTW) reference alignment and differential acoustic analysis | Implemented (Phase 3) |
| Feature persistence store, caching, validation, and JSONL manifest | Implemented (Phase 3) |
| Interactive Plotly time-series curves, MFCC heatmap, and comparison UI panel | Implemented (Phase 3) |
| Independent, label-blind speech flaw detection engine (`pause_insertion`, `tempo_change`, `energy_reduction`) | Implemented (Phase 4) |
| Temporal flaw localization on variant timeline with integer sample boundaries `[start_sample, end_sample)` | Implemented (Phase 4) |
| Measured acoustic evidence scoring and detector configuration fingerprints | Implemented (Phase 4) |
| Strict ground-truth isolation architecture & automated leakage prevention | Implemented (Phase 4) |
| Result persistence store, cache invalidation, detection validator, JSONL manifest | Implemented (Phase 4) |
| Interactive Streamlit flaw detection panel with variant audio playback & evidence drilldown | Implemented (Phase 4) |
| Deterministic 1-to-1 event-level temporal matching using half-open sample IoU | Implemented (Phase 5) |
| Quantitative classification metrics (per-category and micro/macro precision, recall, F1) | Implemented (Phase 5) |
| Temporal localization metrics (mean/median IoU, boundary start/end/duration MAE in seconds) | Implemented (Phase 5) |
| Multi-threshold localization-tolerance curves ($\theta_{\text{IoU}} \in [0.3, 0.9]$) | Implemented (Phase 5) |
| Event-level classification confusion matrix & error attribution (missed vs spurious) | Implemented (Phase 5) |
| Clean negative identity controls & false-alarm rate auditing | Implemented (Phase 5) |
| Source-level cluster bootstrap 95% confidence intervals | Implemented (Phase 5) |
| Immutable evaluation store, catalog manifest, and artifact validation | Implemented (Phase 5) |
| Interactive Streamlit Evaluation Lab panel with KPI cards, Plotly curves, and confusion heatmaps | Implemented (Phase 5) |
| Pitch-variation flattening | Planned extension, not implemented |
| Participant analysis: registered variant or in-memory upload compared with a registered reference | Implemented (Phase 6) |
| Deterministic evidence-linked explanation cards (pause, tempo, energy) with numeric traceability checks | Implemented (Phase 6) |
| Timeline of candidate intervals, reference alignment plots, and interval-specific audio excerpts | Implemented (Phase 6) |
| Exportable participant reports (JSON, Markdown, HTML, CSV) rendered from the saved analysis | Implemented (Phase 6) |
| Multi-page dashboard separating participant analysis, Evaluation Lab and technical panels | Implemented (Phase 6) |
| Detector v5 (windowed tempo slope, level-invariant alignment, silence/voicing guards), explicit F0-correlation policy, deterministic prediction IDs, QA test suites (**416 automated tests**) | Implemented (Phase 7, synthetic-fixture validation only) |
| Release-candidate verification CLI (`python -m flawbench release-check`), demo runbook & script, reproducible packaging, comprehensive documentation, licensing & AI disclosure | Implemented (Phase 8) |
| Optional LLM wording layer, pitch-related detection, word-level alignment, overall quality score | Not implemented |

FlawBench extracts real acoustic features, computes differential DTW alignments, independently detects speech delivery flaws without accessing ground-truth labels, and rigorously benchmarks predicted intervals against known truth using deterministic one-to-one temporal IoU matching.

## Quick Start

```powershell
# Windows PowerShell — create environment and install
py -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Verify the release candidate
python -m flawbench check
python -m flawbench release-check

# Launch the interactive dashboard
streamlit run app.py
```

## End-to-End Workflow (Phases 2–5)

```bash
# 1. Register a recording you are allowed to modify (rights must be confirmed)
python -m flawbench dataset register-reference data/reference/take1.wav \
    --transcript-file take1.txt --source-type self_recorded \
    --attribution "recorded by me" --license "all rights reserved, own recording" \
    --rights-confirmed --redistribution unknown

# 2. Generate variants from a plan (see configs/plan.example.yaml) or a single request
python -m flawbench dataset generate --plan configs/plan.example.yaml
python -m flawbench dataset generate --source src-... --flaw pause_insertion --severity medium --at-sec 3.0

# 3. Audit dataset files and build splits
python -m flawbench dataset validate --report outputs/reports/dataset.json
python -m flawbench dataset split

# 4. Extract time-indexed acoustic features (RMS, spectral, MFCC, pitch)
python -m flawbench features extract --all
python -m flawbench features list
python -m flawbench features validate

# 5. Differential reference comparison & candidate anomaly analysis
python -m flawbench features compare --sample smp-...
python -m flawbench features export --output outputs/reports/feature_summaries.json

# 6. Run independent flaw detection (label-blind inference)
python -m flawbench detect --all
python -m flawbench detect --sample smp-... --export outputs/reports/predictions.json
python -m flawbench detect list
python -m flawbench detect validate

# 7. Run formal evaluation & benchmark scoring against ground truth
python -m flawbench evaluate run --split test
python -m flawbench evaluate run --split all
python -m flawbench evaluate list
python -m flawbench evaluate report --run <run_id>
python -m flawbench evaluate validate --run <run_id>
```

Schema, timeline convention, seed policy: `docs/dataset_schema.md`. Evaluation methodology and schema: `docs/evaluation_methodology.md` and `docs/evaluation_schema.md`. Severity profiles are experimental manipulation strengths, not human-rated severities. No audio is shipped: you must supply a recording you have the rights to use. Tests use synthetic tones only.

## Participant analysis (Phase 6)

```bash
# Compare a recording with a registered reference (audio is read, never stored)
python -m flawbench explain upload --source <SOURCE_ID> --audio my_take.wav [--transcript t.txt] [--export]
# Analyze a registered variant as if it were a participant recording (labels are not read)
python -m flawbench explain registered --sample <SAMPLE_ID> [--export]
```

In the app, use Analyze Speech, then Flaw Timeline, Acoustic Evidence and Improvement Guidance. Each
candidate gets Observation, Evidence, Interpretation, Suggested review and Limitations sections whose
numbers come from validated detector evidence. Findings are candidates, not confirmed errors, and
the detector evidence strength is an uncalibrated heuristic, not a probability. A zero-candidate
result is shown as a completed analysis and does not mean the delivery is free of problems.
See `docs/explanation_methodology.md` and `docs/participant_report_schema.md`.

## Technology stack

Python 3.10+, NumPy, soundfile (libsndfile), audiotsm (WSOLA time-stretch), librosa (spectral, MFCC, YIN pitch, DTW), plotly (interactive visualization), PyYAML, Streamlit; pytest, ruff, mypy for development.


## Repository structure

```
FlawBench/
  app.py                      Streamlit entry point
  pyproject.toml              Package metadata and the single dependency definition
  requirements.txt            Convenience: installs the package with dev extras
  requirements-lock.txt       Reference snapshot of the verified environment
  .env.example                Documented environment variables
  .gitignore
  .streamlit/config.toml      Light theme and transport upload ceiling
  configs/config.example.yaml Example configuration file
  src/flawbench/
    config.py                 Defaults, loading, precedence, validation
    paths.py                  ProjectPaths, writable-directory checks, traversal-safe joins
    validation.py             Pure audio input checks
    audio/loader.py           probe_audio / load_audio, AudioMetadata, LoadedAudio
    logging_config.py         configure_logging, get_logger, log_failure
    exceptions.py             Exception hierarchy with user and detail messages
    provenance.py             Environment and config provenance records
    status.py                 Real component status
    roadmap.py                Phase list (informational)
    ui/components.py          Streamlit rendering helpers
    ui/dataset_panel.py       Dataset management and validation panel
    ui/feature_panel.py       Acoustic feature inspection and DTW comparison panel
    ui/detection_panel.py     Independent flaw detection and candidate drilldown panel
    ui/evaluation_panel.py    Formal Evaluation Lab panel (KPIs, curves, confusion matrix)
    dataset/                  Phase 2 engine: schema, registry, injectors, plan, splits, validator
    features/                 Phase 3 engine: frame grid, extractors, DTW comparison, store
    detection/                Phase 4 engine: label-free inputs, detectors, store, validator
    evaluation/               Phase 5 engine: 1-to-1 matching, metrics, confusion, controls, store
    explainability/           Phase 6 evidence facts, deterministic templates, validator, reports
    participant/              Phase 6 analysis workflow, playback excerpts, transcript check
    release.py                Phase 8 release-validation engine
    __main__.py               CLI: config, check, init, provenance, dataset, features,
                              detect, evaluate, explain, release-check
  tests/                      pytest suite (416 tests; 1 POSIX-only skip)
  data/reference/             Reference recordings (contents git-ignored)
  configs/plan.example.yaml   Example generation plan
  docs/                       architecture.md, development.md, dataset_schema.md,
                              detection_methodology.md, detection_schema.md,
                              evaluation_methodology.md, evaluation_schema.md,
                              explanation_methodology.md, quality_assurance.md,
                              phase_1_completion.md through phase_7_validation_report.md,
                              demo_runbook.md, release_notes.md,
                              licensing_and_disclosure.md, phase_8_release_report.md
```

`data/generated/`, `data/metadata/` (manifests), and `outputs/{reports,figures,exports}/` are created at runtime
(by the app or `python -m flawbench init`) and are git-ignored.

## Installation

```bash
# Unix-like
python3 -m venv .venv && source .venv/bin/activate
# Windows (PowerShell)
#   py -m venv .venv ; .venv\Scripts\Activate.ps1

pip install -e ".[dev]"      # equivalent: pip install -r requirements.txt
```

## Configuration

Precedence, highest first: runtime overrides, environment variables (real environment, then `.env`),
configuration file, application defaults. Defaults are defined once, in `src/flawbench/config.py`.

- Copy `configs/config.example.yaml` to `configs/config.yaml` (picked up automatically), or point
  `FLAWBENCH_CONFIG` at any file, or pass a path to `load_config` / `python -m flawbench --config`.
- Copy `.env.example` to `.env` for local environment settings. Never commit `.env`.
- Relative storage paths resolve against the project root, not the working directory.
- Inspect the effective result: `python -m flawbench config`.

## Run

```bash
# Unix-like
streamlit run app.py
python -m flawbench check           # real status of config, directories, audio decoder
python -m flawbench init            # create the writable directories
python -m flawbench release-check   # comprehensive release candidate verification
python -m flawbench release-check --run-tests   # include full test suite execution
```

```powershell
# Windows PowerShell
streamlit run app.py
python -m flawbench check
python -m flawbench release-check
```

## Test

```bash
pytest                          # full suite (416 tests; 1 POSIX-only skip)
pytest --cov=flawbench          # with coverage
ruff check src tests app.py     # lint
ruff format --check src tests app.py  # format
mypy                            # strict type-check (72 source files)
```

## Known limitations

- Only header-level and decode-level validation exist. libsndfile tolerates a truncated WAV by
  reporting a shorter length, so truncated WAV files may load as shorter audio; truncated FLAC is
  rejected.
- MP3 decoding depends on the installed libsndfile (1.1+).
- Streamlit's transport-level upload ceiling (`.streamlit/config.toml`, 200 MB) is separate from the
  configurable `audio.max_upload_mb`, which is enforced in code. Keep the former at or above the latter.
- Uploads are processed in memory and never persisted.

- Phase 2: the tempo injector uses WSOLA (`audiotsm`), which is pitch-preserving but is not a studio
  time-stretch; segments are joined without crossfades. Pause/tempo placements are uniform random within
  margins and are not aligned to speech boundaries. Manifests assume a single writer process.
- Output reproducibility is verified on the tested environment only; library or platform changes can alter
  tempo output (detected, never silently replaced). Windows and macOS were not exercised.
- Phase 3 & 4: DTW temporal alignment assumes that the spoken content order is preserved; non-linear phrase reorderings or severe background noise may degrade alignment reliability.
- DTW warp-slope thresholding estimates delivery rate anomalies relative to the reference recording; natural stylistic speech variations that are not flaw injections may trigger candidate warnings if thresholds are set too sensitively.
- Evidence scores reflect acoustic deviation magnitudes relative to reference features; they are not calibrated probabilities of real-world human speech competence.
- Detections and predictions are completely label-free; formal benchmark metrics against ground truth are evaluated strictly post-inference.
- Phase 5: Evaluation metrics computed against synthetic audio fixtures verify algorithmic execution, mathematical correctness, filter responses, and localization boundaries. They do not establish generalizability to naturally recorded human speech. Authentic benchmark claims require licensed or self-recorded human speech corpora.
- Source-level bootstrap confidence intervals require $\ge 3$ distinct source recordings; smaller cohorts are reported with documented sample-size caveats.

- Phase 6: Explanations are deterministic templates over detector evidence. Only pause, tempo and energy-reduction deviations are analyzed; pitch and other delivery aspects are not. No audio-text alignment backend exists, so no word-level references are given. Phase 7 investigated the Phase 6 tempo-recall and energy/pause false-positive weaknesses and changed the detectors (detector version 5.0); results are in `docs/phase_7_validation_report.md`. They were measured on synthetic harmonic-tone fixtures only: no natural-speech or manually annotated validation exists, so no accuracy claim about human speech is made. The mildest tempo setting (rate 0.85, equal to the configured 0.15 minimum deviation) is not reliably detected, and a tempo candidate can appear next to a long pause edge. Source types are declared at registration and are not verified; the registry has no synthetic type, so synthetic fixtures are not flagged as synthetic. See `docs/quality_assurance.md`. Source types are declared at registration and are not verified.

## Planned extensions

Pitch-variation detection, word-level audio-text alignment, pronunciation and intonation coaching, calibrated evidence scores, natural-speech validation corpus.

## Data and audio licensing

No audio is committed. Before using or sharing any recording, confirm its licence permits the
intended use and redistribution, and that speakers consented. Record source and licence beside each
reference file. Test tones are synthetic and generated inside pytest temporary directories only.
