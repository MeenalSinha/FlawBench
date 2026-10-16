# FlawBench Licensing, Attribution & Disclosures

## 1. Project License

FlawBench is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Meenal Sinha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Audio Data Policy & Licensing

- **No Proprietary Audio Committed**: In strict compliance with copyright and data privacy standards, no external audio files or proprietary voice recordings are distributed in this repository.
- **Controlled Synthetic Fixtures**: Automated tests and benchmark suites use synthetically generated acoustic waveforms (harmonic multitone audio created in-memory during testing). These fixtures are completely unencumbered by copyright or personal privacy restrictions.
- **User-Supplied Reference Audio**: Users of FlawBench who register reference recordings must supply audio that they have recorded themselves, that is in the public domain, or for which they hold explicit modification and processing rights. The dataset registry enforces explicit confirmation of rights (`--rights-confirmed`) and records licensing provenance for every registered source.

---

## 3. Third-Party Libraries & Dependencies

FlawBench relies on open-source libraries that are distributed under compatible permissive licenses:

| Library | Primary Use | License |
|:---|:---|:---|
| **NumPy** | Array manipulation & numerical computing | BSD 3-Clause |
| **soundfile / libsndfile** | Header validation & lossless audio I/O | BSD 3-Clause / LGPL |
| **audiotsm** | WSOLA pitch-preserving time stretching | MIT |
| **librosa** | Spectral features, MFCCs, DTW alignment, pitch | ISC |
| **plotly** | Interactive visualizations & heatmaps | MIT |
| **PyYAML** | Configuration & plan serialization | MIT |
| **Streamlit** | Interactive web application framework | Apache 2.0 |
| **pytest / pytest-cov** | Automated unit & regression testing | MIT |
| **ruff** | High-performance Python linting & formatting | MIT / Apache 2.0 |
| **mypy** | Static type checking | MIT |

---

## 4. AI-Tool Usage Disclosure

In compliance with hackathon regulations and scientific reproducibility guidelines:

- **AI Coding Assistants**: Google Antigravity / Claude 3.7 / Gemini coding assistants were utilized during the design, implementation, refactoring, test writing, and documentation of FlawBench.
- **Assisted Components**:
  - Drafting unit and integration test fixtures across Phases 1–8.
  - Designing parametric audio injector functions and boundary transformations.
  - Implementing Dynamic Time Warping (DTW) feature alignment and time-windowed warp slope algorithms.
  - Generating Streamlit interactive visualization panels and export templates.
  - Drafting architectural, evaluation schema, and API documentation.
- **Human Responsibility & Verification**:
  - All mathematical formulas, metric calculations, and signal processing pipelines were audited and validated.
  - Every line of code was verified by executing the automated test suite (416 passing tests), strict static type checking (`mypy --strict`), and linting checks (`ruff`).
  - All benchmark scores and evaluation results reported in this project were generated through actual execution of the software pipeline on reproducible inputs.
  - The human engineering team assumes full responsibility for the architectural integrity, safety, and functionality of the software deliverable.

---

## 5. Security & Privacy Audit

- **No Secrets or Credentials**: The repository contains no API keys, private tokens, passwords, or cloud credentials.
- **Safe Environment Defaults**: Environment configurations (`.env.example`) contain only safe placeholders. Real environment settings (`.env`) are excluded via `.gitignore`.
- **Ephemeral Upload Handling**: Audio uploaded via the participant analysis interface is processed strictly in-memory and never written to permanent disk storage without an explicit registration call.
- **Path Traversal Protection**: All filesystem inputs resolve through `ProjectPaths.resolve_within()`, which strictly prevents directory traversal (`../`) attacks.
- **Hostile Text Sanitization**: Transcripts and uploaded filenames are sanitized and backtick-escaped before rendering in exported Markdown or HTML reports.
