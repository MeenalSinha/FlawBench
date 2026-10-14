# Phase 1 completion report

Status: complete for the stated scope. Verification was run on Linux, Python 3.12.3,
streamlit 1.65.0, soundfile 0.14.0 (libsndfile 1.2.2), numpy 2.5.4, pytest 9.1.1.

## Implemented

Centralized validated configuration with documented precedence; path management with input/output
separation; decoder-verified audio probing and loading; idempotent logging; Streamlit app showing
real status and inspecting uploads; provenance helpers; diagnostics CLI; pytest suite; documentation.
No flaw injection, detection, metrics, or fabricated results exist.

## Files

Created (all): `app.py`, `pyproject.toml`, `requirements.txt`, `requirements-lock.txt`, `.gitignore`,
`.env.example`, `.streamlit/config.toml`, `configs/config.example.yaml`, `README.md`,
`data/reference/README.md`, `docs/*.md`, everything under `src/flawbench/`, and `tests/`.
The repository was empty at the start, so nothing pre-existing was modified.

Deviations from the suggested layout: no `data/samples/` (no purpose in Phase 1); `data/generated/`,
`data/metadata/`, `outputs/` are created at runtime rather than committed as empty directories;
extra modules `provenance.py`, `status.py`, `roadmap.py`, `__main__.py` and extra test files for
logging, provenance/CLI, and the app, each with real behavior and tests.

## Commands executed and actual outcomes

| Command | Outcome |
|---|---|
| `pip install -e ".[dev]"` | Succeeded |
| `ruff check src tests app.py` | All checks passed |
| `ruff format --check src tests app.py` | 23 files already formatted |
| `mypy` (strict, 15 source files) | No issues found |
| `pytest --cov=flawbench` | 89 passed, 1 skipped, 0 failed; total coverage 94% |
| `streamlit run app.py --server.headless true` | Server started; `/_stcore/health` returned 200 and `/` returned 200 |
| `python -m flawbench check` (fresh checkout) | Reports output directories missing until `init` is run; audio loader OK |

The one skipped test checks POSIX permission denial and is skipped because verification ran as root.

Streamlit validation limits: the HTTP check shows the server starts and serves the page; script
execution (status panel, uploads, configuration effects, error display) is verified headlessly with
Streamlit's `AppTest` in `tests/test_app_smoke.py`. No real browser session was run.

## Issues found and fixed during verification

- Logging tests counted pytest's capture handlers; they now count only the managed handler.
- A truncated-WAV test failed because libsndfile reports a shorter length instead of an error;
  the test now uses FLAC (which is rejected) and the WAV behavior is listed under limitations.

## Known issues and limitations

See README "Known limitations". Additionally: coverage gaps are mostly rarely reachable OS-error
branches; Windows and macOS were not exercised.

## Deferred (by design)

Everything in Phases 2 to 6.

## Prerequisites for Phase 2

- Licensed reference recordings with transcripts and recorded source/licence.
- A decision on the ground-truth record schema and on interval timeline convention (output timeline).
- A decision on the random-number policy (seed per recording and per flaw).
