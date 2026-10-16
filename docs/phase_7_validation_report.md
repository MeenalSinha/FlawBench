# Phase 7 validation report

Scope: audit of Phase 6, repair of defects, detector-reliability work and QA suites. **Synthetic
fixtures only.** Phase 7 is marked *in progress* in the roadmap, not complete.

## Environment

Linux 6.18 (x86_64), 1 CPU, Python 3.12.3, numpy 2.5.4, librosa 1.0.0, scipy 1.18.1,
streamlit 1.66.0, pytest 9.1.1, ruff 0.17.0, mypy 2.4.0. Installed from `pyproject.toml`
(`pip install -e ".[dev]"`). A fresh-environment install from the pinned requirement files was not
separately tested.

## Phase 6 audit

| Check | Method | Outcome |
|---|---|---|
| Tests | `python -m pytest` on the unmodified ZIP | 376 passed, 1 skipped, 1 warning (matches Phase 6 report) |
| Bare `pytest` | run as documented | 7 collection errors - **defect**, fixed (`pythonpath = ["."]`) |
| Lint / format / types | ruff, ruff format --check, mypy | pass |
| CLI, provenance, health | `check`, `provenance`, help, headless Streamlit `/_stcore/health` | working (HTTP 200, `ok`) |
| Artifact validators | `dataset validate`, `features validate`, `detect --validate`, `evaluate validate --run` | VALID on benchmark runs |

## Defects found and fixed

| # | Defect | Fix | Pinned by |
|---|---|---|---|
| 1 | Bare `pytest` fails collection | pytest `pythonpath` | whole suite |
| 2 | Tempo detector missed all injected changes: slope taken over 5 DTW path points on a staircase path (quantized, direction flipping). DTW itself was accurate (endpoint slope 1.42-1.44 vs 1.429). | time-window slope (`tempo_slope_window_sec`), unusable windows = NaN | `test_tempo_change_detected...`, `test_windowed_slope_*` |
| 3 | Energy detector fired on inserted pauses | silent + unvoiced variant frames excluded | `test_inserted_pause_is_not_reported...` |
| 4 | Pause detector fired on 24 dB attenuated speech | voicing guard (`pause_max_voiced_fraction`) | `test_pause_voicing_guard...` |
| 5 | Level steps distorted DTW (MFCC c0), creating spurious tempo candidates | alignment excludes c0, Euclidean metric (configurable) | `test_strong_attenuation_is_not_reported...` |
| 6 | Undefined F0 correlation replaced by 0.0 and warned | `None` + status, explicit policy | `test_f0_correlation_*` |
| 7 | Prediction IDs random (`uuid4`) | deterministic hash | `test_prediction_ids_are_deterministic` |
| 8 | Detection fingerprint ignored detector version | version and new settings in fingerprint (detector 5.0) | `test_detector_fingerprint_...` |
| 9 | Evaluation report recorded package version as detector version and a path-dependent fingerprint | records detector version and the cache fingerprint; package version in metadata | `test_evaluation_report_identifies...` |
| 10 | Evaluation runner silently overwrote an existing run | refuses unless `--overwrite` | same test |
| 11 | Auto run IDs (1 s resolution) could collide once overwrite was refused (caught by a flaky isolation test) | unique suffix | `test_auto_generated_run_ids...` |
| 12 | Markdown export wrote uploaded filename raw | HTML/backtick escaping of untrusted strings | `test_upload_exports_neutralise_hostile_text` |
| 13 | 8 kHz (supported minimum) produced empty mel filters and a librosa warning | adaptive mel-band count | `test_mel_band_count...` |
| 14 | Report claimed sources were "non-synthetic" | wording now states origin is unverified and synthetic fixtures are not flagged | `test_evaluation_report_identifies...` |

## Detector results (synthetic, `evaluate run --split all`)

Preserved baseline runs use the unmodified Phase 6 code snapshot. The candidate is detector 5.0,
configuration fingerprint `ee4c7f85d2c2ad13`. No threshold was changed.

| Cohort | Detector | Micro P | Micro R | Micro F1 | Mean IoU |
|---|---|---|---|---|---|
| Dev: 4 sources, 12 variants (run `baseline_phase6` / `dev_rc`) | Phase 6 | 0.471 | 0.667 | 0.552 | - |
| Dev | v5 | 1.000 | 1.000 | 1.000 | 0.975 |
| Held-out: 8 new sources, 24 variants, plan fixed in advance (`heldout_baseline` / `heldout_rc`) | Phase 6 | 0.615 | 0.667 | 0.640 | 0.986 |
| Held-out | v5 | 0.957 | 0.917 | 0.936 | 0.970 |

Held-out v5 per category: energy 8/8, 0 FP; pause 8/8, 0 FP; tempo 6/8, 1 FP (precision 0.857,
recall 0.750). The dev baseline did not reproduce the historical 0.615 (its scratch parameters
were not recorded); the held-out baseline happens to match the historical micro figures but is a
different cohort. The dev result is not generalization evidence. Phase 6 tempo recall was 0/4 dev
and 0/8 held-out.

Held-out tempo misses are both rate 0.85, where the injected deviation equals the 0.15 threshold.
Dataset composition: both cohorts have 3 variants per source (pause, tempo, energy); with 4 dev
sources the validation split is empty, so no tuning-validation set existed.

A forced re-detection of all 24 held-out samples reproduced identical predictions and IDs.

## Commands and outcomes (final state)

`python -m pytest -W error::RuntimeWarning`: **415 passed, 1 skipped, 0 failed** (skip: POSIX
permission semantics when running as root). Run twice consecutively with identical counts at an
earlier 401-test state, which also exposed the flaky run-ID collision above. `ruff check`,
`ruff format --check`, `mypy`: pass. Streamlit headless health: HTTP 200.

## Not done / deferred

- Natural-speech validation (no licensed annotated speech available; none fabricated).
- Real-browser test; new AppTest journeys A-D; maximum-upload and long-audio stress tests;
  temporary-file cleanup re-test beyond existing tests.
- Synthetic-source marker in the registry (needs a schema decision).
- Word-level alignment (deferred by design).
- Phase 7 roadmap status: in progress, not complete.

## Phase 8 readiness

Not yet ready for final packaging: finish the items above, decide the synthetic-marker schema
change, and re-audit evidence-strength wording in the UI.
