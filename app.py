"""FlawBench Streamlit application (participant analysis, Evaluation Lab, technical panels).

Run with: ``streamlit run app.py`` (after ``pip install -e .``).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

try:
    from flawbench.audio.loader import load_audio
    from flawbench.config import Config, defaults, load_config
    from flawbench.exceptions import AudioInputError, ConfigurationError, FlawBenchError
    from flawbench.features.extractor import extract_features
    from flawbench.logging_config import configure_logging, get_logger, log_failure
    from flawbench.paths import ProjectPaths
    from flawbench.status import collect_status
    from flawbench.ui.components import render_metadata, render_status
    from flawbench.ui.dataset_panel import render_dataset_panel
    from flawbench.ui.detection_panel import render_detection_panel
    from flawbench.ui.evaluation_panel import render_evaluation_panel
    from flawbench.ui.feature_panel import (
        render_feature_result_plots,
        render_features_panel,
    )
    from flawbench.ui.participant_panel import render_analyze_page, render_overview_page
    from flawbench.ui.participant_results import (
        render_evidence_page,
        render_guidance_page,
        render_timeline_page,
    )

except ImportError as import_error:  # package not installed
    st.set_page_config(page_title="FlawBench")
    st.error("The flawbench package is not installed. Run `pip install -e .` in the project root.")
    st.caption(str(import_error))
    st.stop()


def _load_config_or_stop() -> Config:
    try:
        return load_config()
    except ConfigurationError as exc:
        fallback = defaults()
        st.set_page_config(page_title=fallback["ui"]["page_title"])
        st.title(fallback["app"]["name"])
        st.error(f"Configuration problem: {exc.public_message(expose_details=True)}")
        st.stop()


def main() -> None:
    config = _load_config_or_stop()
    st.set_page_config(page_title=config.ui.page_title, layout=config.ui.layout)  # type: ignore[arg-type]
    configure_logging(config.app.log_level)
    logger = get_logger("app")
    expose = config.diagnostics.expose_error_details
    detailed = config.diagnostics.detailed_logging

    paths = ProjectPaths(config)
    startup_error: str | None = None
    try:
        paths.ensure_output_directories()
    except FlawBenchError as exc:
        log_failure(logger, "Could not prepare output directories", exc, detailed=detailed)
        startup_error = exc.public_message(expose_details=expose)

    if startup_error:
        st.error(f"Storage problem: {startup_error}")

    pages = {
        "overview": "Overview",
        "analyze": "Analyze Speech",
        "timeline": "Flaw Timeline",
        "evidence": "Acoustic Evidence",
        "guidance": "Improvement Guidance",
        "evaluation": "Evaluation Lab",
        "technical": "Dataset and Technical Details",
    }
    if "flawbench_page" not in st.session_state:
        st.session_state["flawbench_page"] = config.explanation.default_page
    with st.sidebar:
        st.markdown(f"### {config.app.name}")
        page = st.radio(
            "Navigate",
            list(pages),
            format_func=lambda k: pages[k],
            key="flawbench_page",
        )
        st.caption("Participant analysis and the benchmark Evaluation Lab are separate workflows.")

    if page == "overview":
        render_overview_page(config, paths, expose_details=expose)
    elif page == "analyze":
        render_analyze_page(config, paths, expose_details=expose)
    elif page == "timeline":
        render_timeline_page(config)
    elif page == "evidence":
        render_evidence_page(config)
    elif page == "guidance":
        render_guidance_page(config)
    elif page == "evaluation":
        render_evaluation_panel(config, paths, expose_details=expose)
    else:
        _render_technical(config, paths, expose, logger, detailed)


def _render_technical(
    config: Config, paths: ProjectPaths, expose: bool, logger: Any, detailed: bool
) -> None:
    st.header("Dataset and Technical Details")
    render_status(collect_status(config, paths, expose_details=expose))
    render_dataset_panel(config, paths, expose_details=expose)
    render_features_panel(config, paths, expose_details=expose)
    render_detection_panel(config, paths, expose_details=expose)

    st.subheader("Audio file inspection")
    st.caption(
        "Upload a recording to validate it and inspect its acoustic features. This only extracts "
        "features: no flaw detection is run here. Files are processed in memory and not stored."
    )
    uploaded = st.file_uploader(
        "Audio file",
        type=config.audio.extension_names,
        help=f"Maximum size {config.audio.max_upload_mb} MB.",
    )
    if uploaded is None:
        st.info("No audio selected. No inspection has been performed.")
        return
    data = uploaded.getvalue()
    try:
        audio = load_audio(data, config.audio, filename=uploaded.name)
    except AudioInputError as exc:
        log_failure(logger, "Audio input rejected", exc, detailed=detailed)
        st.error(exc.public_message(expose_details=expose))
        return
    except Exception as exc:  # UI boundary: log everything, show a generic message
        log_failure(logger, "Unexpected error while loading audio", exc, detailed=True)
        st.error("An unexpected error occurred while reading the file. See the application log.")
        return
    render_metadata(audio.metadata)
    st.audio(data)
    with (
        st.expander("Time-indexed acoustic features", expanded=True),
        st.spinner("Extracting features from uploaded audio..."),
    ):
        try:
            feat_res = extract_features(audio, feat_cfg=config.features, config=config)
            render_feature_result_plots(feat_res, title_prefix=uploaded.name)
        except FlawBenchError as f_exc:
            st.error(f"Feature extraction failed: {f_exc.public_message(expose_details=expose)}")


main()
