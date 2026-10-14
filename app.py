"""FlawBench Phase 1 Streamlit application.

Run with: ``streamlit run app.py`` (after ``pip install -e .``).
"""

from __future__ import annotations

import streamlit as st

try:
    from flawbench.audio.loader import load_audio
    from flawbench.config import Config, defaults, load_config
    from flawbench.exceptions import AudioInputError, ConfigurationError, FlawBenchError
    from flawbench.logging_config import configure_logging, get_logger, log_failure
    from flawbench.paths import ProjectPaths
    from flawbench.roadmap import PHASES
    from flawbench.status import collect_status
    from flawbench.features.extractor import extract_features

    from flawbench.ui.components import (
        render_header,
        render_metadata,
        render_roadmap,
        render_status,
    )
    from flawbench.ui.dataset_panel import render_dataset_panel
    from flawbench.ui.feature_panel import (
        render_feature_result_plots,
        render_features_panel,
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

    render_header(config.app.name)
    if startup_error:
        st.error(f"Storage problem: {startup_error}")
    render_status(collect_status(config, paths, expose_details=expose))

    render_dataset_panel(config, paths, expose_details=expose)

    render_features_panel(config, paths, expose_details=expose)

    st.subheader("Reference audio")
    st.caption(
        "Upload a recording to validate and inspect it. Files are processed in memory and are "
        "not stored."
    )
    uploaded = st.file_uploader(
        "Audio file",
        type=config.audio.extension_names,
        help=f"Maximum size {config.audio.max_upload_mb} MB.",
    )
    if uploaded is not None:
        data = uploaded.getvalue()
        try:
            audio = load_audio(data, config.audio, filename=uploaded.name)
        except AudioInputError as exc:
            log_failure(logger, "Audio input rejected", exc, detailed=detailed)
            st.error(exc.public_message(expose_details=expose))
        except Exception as exc:  # UI boundary: log everything, show a generic message
            log_failure(logger, "Unexpected error while loading audio", exc, detailed=True)
            st.error(
                "An unexpected error occurred while reading the file. See the application log."
            )
        else:
            render_metadata(audio.metadata)
            st.audio(data)
            with st.expander("Extract and inspect time-indexed acoustic features", expanded=True):
                with st.spinner("Extracting features from uploaded audio..."):
                    try:
                        feat_res = extract_features(audio, feat_cfg=config.features, config=config)
                        render_feature_result_plots(feat_res, title_prefix=uploaded.name)
                    except FlawBenchError as f_exc:
                        st.error(f"Feature extraction failed: {f_exc.public_message(expose_details=expose)}")
            st.info("Analysis complete for acoustic features. Flaw detection is planned for Phase 4.")
    else:
        st.info("No audio selected. Analysis has not been performed.")



    if config.ui.show_roadmap:
        render_roadmap(PHASES)


main()
