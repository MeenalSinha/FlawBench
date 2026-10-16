# Phase 6 completion report

Phase 6 adds a participant-analysis workflow, an evidence-linked explanation engine, interval
playback, exportable reports and a multi-page dashboard on top of the unchanged Phase 1 to 5 core.
Figures below come from commands run while building this phase; historical counts were not copied.

## 1. Phase 5 audit
Details are in `docs/phase_5_audit.md`. Summary: the historical result (287 passed, 4 skipped)
reproduced only after declaring the missing `plotly` and `pandas` dependencies.

## 2. Changes to earlier phases
- `pyproject.toml`: declared `plotly` and `pandas`.
- `detection/detectors.py`: tempo and energy predictions report the reference span as `None` when it is degenerate (previously a zero-length span made the whole result fail validation).
- `detection/postprocess.py`: merge no longer drops a reference time of `0.0`.
- `detection/engine.py`: opt-in `tolerate_detector_failures` (default behavior unchanged); records `detectors_completed` and `detector_failures` in metadata.
- `evaluation/reports.py` and the Evaluation Lab: provenance wording no longer says "Natural Speech"; IoU slider now applied; splits come from configuration; emoji removed.
- `ui/feature_panel.py`: one mypy error fixed. `app.py`: single page replaced by navigation; the feature-only upload no longer claims detection ran.
- `config.py`: new validated `explanation` section; default `ui.layout` is now `wide`.
- Tests changed deliberately: `test_app_smoke.py` (pages moved, three stale skips re-enabled), `test_provenance_and_cli.py` (roadmap now shows Phase 6 implemented), `test_evaluation_store_runner_and_cli.py` (fixture used a wrong environment variable and wrote into the repository). Two regression tests were added for the detection repairs.

## 3. New modules
`explainability/` (schema, facts, templates, validator, builder, reports), `participant/` (analyzer, playback, transcript, benchmark), `ui/participant_panel.py`, `ui/participant_results.py`, CLI `flawbench explain`.

## 4. Workflows
- Registered variant: reads only whitelisted manifest fields (annotations are not loaded), reuses cached feature artifacts.
- Uploaded recording: decoded by the existing loader, analyzed in memory, never written; registration is a separate, rights-gated, configuration-controlled action.
- Statuses: `completed`, `completed_no_candidates`, `partial` (named failed detectors), `failed` (stage recorded, later stages skipped).
- Evidence linking, playback (sample-exact, clamped, reference anchor window) and export behavior are described in `docs/explanation_methodology.md` and `docs/participant_report_schema.md`.

## 5. Verification (actual commands and results)
| Command | Result |
|---|---|
| `python -m pytest -p no:cacheprovider -q` | 376 passed, 1 skipped (POSIX permission test; cannot run as root), 0 failed |
| `ruff check src tests app.py` | All checks passed |
| `ruff format --check src tests app.py` | 109 files already formatted |
| `mypy` | Success: no issues found in 71 source files |
| `python -m flawbench check / config / provenance` | OK |
| `dataset validate`, `features validate`, `detect --validate` | VALID; 0/0 on an empty repo, 12/12 on the scratch dataset below |
| `evaluate --help`, `explain --help` | OK |
| Streamlit headless launch, `GET /_stcore/health` | `ok`, root page HTTP 200 |

Scratch end-to-end run (4 synthetic references, 12 variants, created through the CLI in a temporary
project, not stored in the repository): `detect --all` processed 12/12 with 0 failures;
`evaluate run --split all` was VALID with micro precision 0.615, recall 0.667, F1 0.640, mean IoU
0.988. Participant analysis ran on pause, tempo and energy variants: the pause variant gave a pause
and an energy candidate, the energy variant gave an energy candidate, the tempo variant gave a
completed zero-candidate result; reports were exported in four formats.

Test coverage added: facts, templates and numeric traceability, grounding (source-level import and
field checks, invalid results, status/content agreement), playback bounds and rates, workflow
(upload, invalid input, zero candidates, partial failure, sample rates, persistence, rights, cleanup,
CLI), reports (all formats, consistency with saved JSON, escaping, path safety) and AppTest
integration (pages, empty states, end-to-end registered and uploaded flows, export on click,
Evaluation Lab not auto-running).

## 6. Known limitations
- All verification used synthetic fixtures; no natural-speech validation exists and source types are self-declared.
- On the scratch benchmark the tempo detector recalled 0 of 4 injected tempo changes (1 spurious prediction) and the energy detector produced 8 predictions for 4 events because it also fires on inserted pauses (precision 0.500). These are Phase 4 behaviors; they were not changed because that would alter benchmark results.
- Evidence strength is an uncalibrated heuristic. No overall quality score exists.
- Only pause, tempo and energy deviations are analyzed; pitch is not.
- No audio-text alignment backend: no word-level references; transcript comparison is text similarity only.
- Numeric RuntimeWarnings (invalid divide in a correlation) appear during feature extraction on some synthetic signals; they predate Phase 6 and were not investigated.
- Streamlit audio players cannot seek, so excerpts are generated instead.
- The dashboard was exercised with Streamlit's AppTest and a health check, not in a real browser.

## 7. Deferred
Optional language-model wording layer, user-supplied rubrics, PDF export, word-level alignment, pitch-related detection, natural-speech benchmark corpus, regenerating `requirements-lock.txt`.

## 8. Readiness
The participant workflow, explanations, playback, reports and Evaluation Lab separation work and are
tested on synthetic audio. The project is ready for a demonstration that states the limitations
above; it should not be presented as validated on natural speech or as detecting tempo reliably.
