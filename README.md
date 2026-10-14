# FlawBench

Evidence-based speech flaw analysis with exact ground truth.
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

## Current scope: Phases 1, 2, and 3 (implemented)

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
| Pitch-variation flattening | Planned extension, not implemented |
| Flaw detection, metrics, explanations | Planned (Phases 4 to 6) |

FlawBench extracts real acoustic features and performs differential reference comparison. Independent flaw detection and evaluation benchmarks will be introduced in Phases 4 & 5.

## Dataset and Feature workflow (Phases 2 & 3)

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
```

Schema, timeline convention, seed policy: `docs/dataset_schema.md`. Severity profiles are experimental
manipulation strengths, not human-rated severities. No audio is shipped: you must supply a recording you
have the rights to use. Tests use synthetic tones only.

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
    ui/dataset_panel.py       Real dataset counts and a validate button (calls the same engine)
    dataset/                  Phase 2 engine: params, schema, io, registry, injectors, plan,
                              generator, validator, splits, export
    __main__.py               CLI: config, check, init, provenance
  tests/                      pytest suite
  data/reference/             Reference recordings (contents git-ignored; see its README)
  configs/plan.example.yaml   Example generation plan
  docs/                       architecture.md, development.md, dataset_schema.md,
                              phase_1_completion.md, phase_2_completion.md
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
streamlit run app.py
python -m flawbench check      # real status of config, directories, audio decoder
python -m flawbench init       # create the writable directories
```

## Test

```bash
pytest                          # full suite
pytest --cov=flawbench          # with coverage
ruff check src tests app.py && mypy
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
- Generated metadata (`data/metadata`) is git-ignored by default; commit or archive it deliberately if wanted.

## Planned phases

3 Feature extraction; 4 Flaw detection; 5 Evaluation; 6 Explanations and dashboard. See `docs/architecture.md` for the planned interfaces.

## Data and audio licensing

No audio is committed. Before using or sharing any recording, confirm its licence permits the
intended use and redistribution, and that speakers consented. Record source and licence beside each
reference file. Test tones are synthetic and generated inside pytest temporary directories only.
