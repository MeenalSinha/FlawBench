# FlawBench quality assurance (Phase 7)

All automated validation in this repository uses **synthetic fixtures** (a generated harmonic tone
with noise, `tests/conftest.py::synthetic_signal`). It checks that the pipeline behaves correctly
and consistently. It does **not** establish accuracy on natural speech.

## Test strategy

| Layer | Where | What it pins |
|---|---|---|
| Unit and component tests (Phases 1-6) | `tests/test_*.py` | Config, paths, audio loading, dataset, features, detectors, evaluation, explainability, participant workflow, UI smoke (Streamlit AppTest). |
| Detector regression tests | `tests/test_phase7_regressions.py` | Tempo detection with direction and localization, pause is not energy/tempo, attenuation is not pause/tempo, identical audio gives no predictions, deterministic prediction IDs, fingerprint sensitivity, slope estimator, F0-correlation policy, evaluation identity, no silent overwrite, unique auto run IDs. |
| End-to-end and security boundaries | `tests/test_phase7_end_to_end.py` | Register, generate three flaw types, validate, split, detect, evaluate, explain, export; ground-truth isolation (corrupting every annotation and force re-detecting leaves predictions identical); source and detection artifact integrity; hostile upload names/transcripts in exports; repeat-analysis idempotency. |
| Robustness | `tests/test_phase7_robustness.py` | Sample rates 8/16/22.05/44.1 kHz, short, silent, constant and very low-level audio, missing F0 never fabricated, mel-filter sanity. Warnings are errors. |

Regression tests were mutation-checked: removing the energy silence guard or the pause voicing
guard makes the matching test fail.

## Ground-truth isolation

Generation writes audio and annotations. Detection reads only audio, features and checksums.
Evaluation joins annotations after inference. Participant analysis reads saved predictions and
features only. The end-to-end test enforces this by rewriting every annotation interval and
re-running detection with `force=True`.

## Numerical-stability policy

- Undefined statistics are **unavailable**, not zero. `f0_voiced_correlation` is `null` with a
  `f0_voiced_correlation_status` of `unavailable_constant_f0`, `unavailable_insufficient_voiced_points`,
  `unavailable_non_finite_input`, `unavailable_shape_mismatch` or `unavailable_undefined`.
- Windows without usable reference progress or dominated by silence give no tempo slope (NaN),
  which detectors never treat as evidence.
- Mel-band count adapts only where 128 bands would produce empty filters (8 kHz uses 96); rates
  from 11.025 kHz up keep 128.
- No global warning suppression. The suite runs clean with `-W error::RuntimeWarning`.

## Security and privacy checks covered

Hostile upload names and transcripts (script tags, formula-leading text, path-traversal text) do
not produce raw inline HTML in Markdown/HTML, formula cells in CSV, or files outside the reports
directory. Markdown export now escapes untrusted strings. Existing Phase 6 tests cover upload
non-persistence. This is not a claim that the application is completely secure.

## Performance (diagnostic only)

One CPU, Linux, Python 3.12.3, numpy 2.5.4, librosa 1.0.0, synthetic 16 kHz audio, reference plus
variant of equal length: 6 s audio - features 1.83 s, DTW 0.02 s; 30 s - 1.99 s, 0.26 s; 120 s -
9.70 s, 4.74 s; detection under 0.03 s in all cases. Not a production guarantee.

## Known limitations

1. No natural-speech validation; no trusted manual annotations exist in the repository.
2. Detector thresholds and the alignment change were designed on a 4-source dev cohort; the
   8-source held-out check is still synthetic and small.
3. The mildest tempo setting (rate 0.85) equals the configured minimum deviation and is missed.
4. A short "slower" tempo candidate can appear beside the edge of a long pause.
5. Source types are declared, not verified; the registry has no synthetic type.
6. Evidence strength remains an uncalibrated heuristic. Wording across the UI was not re-audited.
7. No forced alignment or word-level timing (deferred).
8. Real-browser testing was not run. New UI journeys A-D were not re-executed beyond the
   existing AppTest smoke tests and a headless server health check.
