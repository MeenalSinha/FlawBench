"""Small dataset-management panel. It calls the same engine functions as the CLI and displays
real counts and real validation findings; it contains no dataset logic of its own."""

from __future__ import annotations

import streamlit as st

from flawbench.config import Config
from flawbench.dataset.generator import DatasetGenerator
from flawbench.dataset.registry import SourceRegistry
from flawbench.dataset.validator import validate_dataset
from flawbench.exceptions import FlawBenchError
from flawbench.paths import ProjectPaths

REGISTER_HINT = "python -m flawbench dataset register-reference --help"


def render_dataset_panel(config: Config, paths: ProjectPaths, *, expose_details: bool) -> None:
    st.subheader("Dataset")
    st.caption(
        "Generation and registration are command-line operations "
        "(`python -m flawbench dataset --help`). This panel shows real registry contents."
    )
    try:
        registry = SourceRegistry(config, paths)
        sources = registry.all()
        samples = DatasetGenerator(config, paths, registry).samples()
    except FlawBenchError as exc:
        st.error(f"Dataset registry problem: {exc.public_message(expose_details=expose_details)}")
        return
    col1, col2 = st.columns(2)
    col1.metric("Registered sources", len(sources))
    col2.metric("Generated samples", len(samples))
    if not sources:
        st.info(f"No reference recordings are registered yet. See `{REGISTER_HINT}`.")
    else:
        st.dataframe(
            [
                {
                    "source_id": s.source_id,
                    "duration_s": round(float(s.data["duration_seconds"]), 2),
                    "sample_rate_hz": s.data["sample_rate"],
                    "redistribution": s.redistribution,
                }
                for s in sources
            ],
            hide_index=True,
        )
    if st.button("Validate dataset"):
        report = validate_dataset(config, paths)
        if report.is_empty:
            st.info("The dataset is empty: nothing to validate.")
        elif report.ok:
            st.success(f"Valid: {len(sources)} source(s), {len(samples)} sample(s), no errors.")
        else:
            st.error(f"Invalid: {len(report.errors)} error(s).")
        for finding in report.findings:
            line = f"{finding.code}: {finding.message} ({finding.subject})"
            (st.error if finding.severity == "error" else st.warning)(line)
