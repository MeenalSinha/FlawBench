# Phase 5 audit (performed before Phase 6 development)

| Requirement | Method | Result | Repair | Remaining limitation |
|---|---|---|---|---|
| Declared dependencies complete | Fresh `pip install -e ".[dev]"`, run suite | 4 UI smoke tests failed: `plotly` (and `pandas`) used by the UI but undeclared | Declared both in `pyproject.toml` | Lock file not regenerated |
| Historical 287 passed / 4 skipped | `pytest` after the fix | 287 passed, 4 skipped, 0 failed (reproduced, skips not counted as passes) | none | 3 uploader tests were hard-skipped as "Streamlit < 1.65" although 1.66 is installed; re-enabled in Phase 6 |
| Type check | `mypy` | 1 error in `feature_panel.py` | Narrowed the optional sample id | none |
| Lint and format | `ruff check`, `ruff format --check` | clean | none | none |
| Detection result always valid | Synthetic pause between two tones | Energy detector emitted a zero-length reference span (`ref_start >= ref_end`) and the Phase 4 validator rejected the whole result | Non-pause detectors report the reference span as `None` when degenerate; regression test | Reference correspondence is then unavailable for that candidate |
| Merge keeps reference times | Code review of `postprocess.py` | `a or b` discarded a legitimate `0.0` | Explicit `None` checks; regression test | none |
| Evaluation Lab inputs honored | Code review | IoU slider value discarded; split options hardcoded | Slider applied to a copied config; options from `VALID_EVALUATION_SPLITS` | none |
| Provenance wording | Ran a scratch evaluation | Non-synthetic declared types were labeled "Natural Speech" | Reworded to "declared, not independently verified" | Source types remain self-declared |
| Misleading app message | Code review | Feature-only upload displayed "Analysis complete ... flaw detection" | Section reworded and moved to Technical Details | none |
| Audio provenance | Inspected the repository | No recordings ship with the repository; all verification used synthetic fixtures | n/a | No natural-speech validation exists |
