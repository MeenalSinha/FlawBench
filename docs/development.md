# Development guide

## Environment

Python 3.10 or newer (verified on 3.12.3). Use an isolated virtual environment.

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: py -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"                                 # or: pip install -r requirements.txt
```

`pyproject.toml` is the only dependency definition. `requirements-lock.txt` is a reference snapshot.

## Configuration setup

```bash
cp configs/config.example.yaml configs/config.yaml     # optional, git-ignored
cp .env.example .env                                    # optional, git-ignored
python -m flawbench config                              # effective configuration
python -m flawbench init                                # create writable directories
python -m flawbench check                               # real status; exit code 1 on any failure
```

Precedence: overrides > environment > config file > defaults. See `.env.example` for every variable.

## Commands

| Task | Command |
|---|---|
| Run app | `streamlit run app.py` |
| Tests | `pytest` |
| Coverage | `pytest --cov=flawbench --cov-report=term-missing` |
| Lint / format check | `ruff check src tests app.py` / `ruff format --check src tests app.py` |
| Types | `mypy` |
| Dataset help | `python -m flawbench dataset --help` |
| Validate dataset | `python -m flawbench dataset validate` |

## Logging

Logs go to stderr under the `flawbench` namespace. Level: `app.log_level` /
`FLAWBENCH_LOG_LEVEL`. `diagnostics.detailed_logging` adds tracebacks. Raw audio is never logged.
`configure_logging` is idempotent.

## Common problems

- "The flawbench package is not installed": run `pip install -e .` in the active environment.
- Configuration error naming a setting: fix the value in the file, `.env`, or environment.
- Reference directory reported missing: create it or point `storage.reference_dir` elsewhere; it is
  never auto-created.
- MP3 rejected as undecodable: your libsndfile predates 1.1; use WAV or FLAC.
- Upload rejected above your limit but under 200 MB: that is `audio.max_upload_mb`, by design.
- Permission test skipped: it is skipped when running as root or on Windows.

## Phase 2 notes

- Dataset settings live in the `dataset` section (see `configs/config.example.yaml`); mapping-valued settings
  (`severity_profiles`, `parameter_bounds`, `split_ratios`) replace the default mapping wholesale and take JSON
  in environment variables.
- Tests use synthetic tones created in temporary directories; no recordings are committed.
- `register-reference` is the only operation that may write into `reference_dir` (only with `--copy` or
  `dataset.copy_reference`, and never over an existing file).
- Only one process should write the dataset metadata at a time (no file locking).
- Common problems: "Rights must be confirmed" (pass `--rights-confirmed`); "outside the reference directory"
  (move the file or use `--copy`); "already exists with different content" (library version changed or file
  edited; use `--overwrite`, history is kept); random placement without a seed (pass `--seed`).
