"""Analyze Speech page and Overview page (Phase 6)."""

from __future__ import annotations

import streamlit as st

from flawbench.audio.loader import load_audio
from flawbench.config import Config
from flawbench.evaluation.store import EvaluationStore
from flawbench.exceptions import FlawBenchError
from flawbench.explainability.facts import SUPPORTED_CATEGORIES
from flawbench.participant.analyzer import (
    MODE_REGISTERED,
    MODE_UPLOAD,
    ParticipantAnalyzer,
)
from flawbench.participant.transcript import read_transcript_bytes
from flawbench.paths import ProjectPaths
from flawbench.status import StatusItem, collect_status
from flawbench.ui.components import render_header, render_status
from flawbench.ui.participant_results import SESSION_KEY, current_session, render_summary

MODE_LABELS = {
    MODE_REGISTERED: "Analyze a registered variant",
    MODE_UPLOAD: "Compare an uploaded participant recording with a reference",
}
STAGE_LABELS = {
    "validate_inputs": "Validate inputs",
    "load_audio": "Load audio",
    "extract_features": "Extract or load features",
    "compare_and_detect": "Compare with reference and detect",
    "validate_detection": "Validate detection result",
    "build_explanations": "Build explanations",
}


def render_analyze_page(config: Config, paths: ProjectPaths, *, expose_details: bool) -> None:
    st.header("Analyze Speech")
    st.caption(
        "Compare a participant recording with a reference recording. Uploaded audio is processed "
        "in memory and is not stored. Hidden benchmark annotations are never shown here."
    )
    analyzer = ParticipantAnalyzer(config, paths)
    try:
        references = analyzer.references()
    except FlawBenchError as exc:
        detail = exc.public_message(expose_details=expose_details)
        st.error(f"The source registry could not be read: {detail}")
        return
    if not references:
        st.info(
            "No reference recordings are registered. Register a reference in Dataset and "
            "Technical Details first; a participant comparison needs a reference."
        )
        return

    modes = list(config.explanation.analysis_modes)
    mode = st.radio("Input mode", modes, format_func=lambda m: MODE_LABELS[m], key="fb_mode")
    ref_labels = {
        r.source_id: f"{r.source_id}  ({r.data.get('source_type', 'unknown')})" for r in references
    }
    source_id = st.selectbox(
        "Reference recording", list(ref_labels), format_func=lambda k: ref_labels[k], key="fb_ref"
    )

    sample_id: str | None = None
    upload_name = ""
    upload_bytes = b""
    participant_text: str | None = None
    ready = False

    if mode == MODE_REGISTERED:
        variants = analyzer.registered_variants(source_id)
        if not variants:
            st.info("This reference has no generated variants. Generate one in the dataset panel.")
        else:
            labels = {v.sample_id: f"Variant {v.sample_id}" for v in variants}
            sample_id = st.selectbox(
                "Registered variant",
                list(labels),
                format_func=lambda k: labels[k],
                key="fb_variant",
            )
            st.caption(
                "This is a generated benchmark variant treated as a participant recording. Its "
                "hidden injection labels are not loaded or displayed."
            )
            ready = True
    else:
        uploaded = st.file_uploader(
            "Participant recording",
            type=config.audio.extension_names,
            key="fb_upload",
            help=f"Maximum size {config.audio.max_upload_mb} MB.",
        )
        if uploaded is not None:
            upload_bytes, upload_name = uploaded.getvalue(), uploaded.name
            try:
                meta = load_audio(upload_bytes, config.audio, filename=upload_name).metadata
            except FlawBenchError as exc:
                st.error(exc.public_message(expose_details=expose_details))
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Format", meta.container_format)
                c2.metric("Sample rate", f"{meta.sample_rate:,} Hz")
                c3.metric("Channels", meta.channels)
                c4.metric("Duration", f"{meta.duration_seconds:.2f} s")
                st.audio(upload_bytes)
                st.caption(
                    "The recording is analyzed exactly as uploaded: no resampling or trimming."
                )
                ready = True
        formats = [f.lstrip(".") for f in config.explanation.transcript_formats]
        with st.expander("Optional: participant transcript"):
            typed = st.text_area("Transcript text", key="fb_transcript_text")
            tfile = st.file_uploader("Or a transcript file", type=formats, key="fb_transcript_file")
            if tfile is not None:
                try:
                    participant_text = read_transcript_bytes(
                        tfile.getvalue(), tfile.name, config.explanation.transcript_formats
                    )
                except FlawBenchError as exc:
                    st.error(exc.public_message(expose_details=expose_details))
            elif typed.strip():
                participant_text = typed
            st.caption(
                "The transcript is compared to the reference transcript as text only. It cannot "
                "provide word-level timing."
            )

    if st.button("Run analysis", type="primary", disabled=not ready, key="fb_run"):
        with st.status("Running analysis", expanded=True) as box:
            if mode == MODE_REGISTERED and sample_id:
                fresh = analyzer.analyze_registered(sample_id)
            else:
                fresh = analyzer.analyze_upload(
                    source_id, upload_bytes, upload_name, participant_text
                )
            for stage in fresh.analysis.stages:
                st.write(f"{STAGE_LABELS.get(stage.name, stage.name)}: {stage.status}")
            box.update(
                label=f"Analysis {fresh.analysis.status}",
                state="error" if fresh.analysis.status == "failed" else "complete",
            )
        st.session_state[SESSION_KEY] = fresh
        st.session_state.pop("flawbench_written", None)

    session = current_session()
    if session is not None:
        render_summary(session.analysis)
        if session.analysis.cards:
            st.caption("Open Flaw Timeline, Acoustic Evidence or Improvement Guidance for details.")
        if (
            mode == MODE_UPLOAD
            and upload_bytes
            and config.explanation.upload_persistence == "explicit_registration"
        ):
            _render_registration(analyzer, config, upload_bytes, upload_name, expose_details)


def _render_registration(
    analyzer: ParticipantAnalyzer, config: Config, data: bytes, name: str, expose: bool
) -> None:
    with st.expander("Optional: permanently register this upload as a reference source"):
        st.warning(
            "Registering stores a copy of this recording in the project dataset. Do this only "
            "if you have the rights to do so. Nothing is stored unless you click Register."
        )
        source_type = st.selectbox(
            "Source type",
            ["self_recorded", "licensed", "public_domain", "other_permissioned"],
            key="fb_reg_type",
        )
        attribution = st.text_input("Attribution", key="fb_reg_attr")
        license_text = st.text_input("License", key="fb_reg_license")
        transcript = st.text_area("Transcript of the recording", key="fb_reg_transcript")
        confirmed = st.checkbox("I confirm I have the right to use this recording", key="fb_reg_ok")
        if st.button("Register recording", key="fb_reg_btn"):
            try:
                sid = analyzer.register_upload(
                    data,
                    name,
                    source_type=source_type,
                    attribution=attribution,
                    license=license_text,
                    rights_confirmed=confirmed,
                    transcript_text=transcript,
                )
            except FlawBenchError as exc:
                st.error(exc.public_message(expose_details=expose))
            else:
                st.success(f"Registered as source `{sid}`.")


def participant_status_items(config: Config, paths: ProjectPaths) -> list[StatusItem]:
    """Operational status derived from real checks and stored artifacts."""
    items: list[StatusItem] = []
    enabled = list(config.detection.enabled_detectors)
    items.append(
        StatusItem(
            "Detection engine",
            bool(enabled),
            "Enabled detectors: " + (", ".join(enabled) or "none"),
        )
    )
    try:
        runs = EvaluationStore(paths).list_runs()
        items.append(
            StatusItem(
                "Evaluation reports",
                bool(runs),
                f"{len(runs)} saved run(s)" if runs else "No saved evaluation runs",
            )
        )
    except Exception as exc:  # status must never crash the page
        items.append(StatusItem("Evaluation reports", False, f"Unreadable: {exc}"))
    rdir = config.explanation.reports_dir
    n_reports = len(list(rdir.glob("*.json"))) if rdir.is_dir() else 0
    items.append(StatusItem("Saved participant reports", True, f"{n_reports} report(s) on disk"))
    session = current_session()
    items.append(
        StatusItem(
            "Current analysis",
            session is not None and session.analysis.status != "failed",
            "None run in this session" if session is None else f"Status: {session.analysis.status}",
        )
    )
    return items


def render_overview_page(config: Config, paths: ProjectPaths, *, expose_details: bool) -> None:
    from flawbench.roadmap import PHASES
    from flawbench.ui.components import render_roadmap

    render_header(config.app.name)
    st.markdown(
        "FlawBench compares a participant recording with a reference recording, flags candidate "
        f"pause, tempo and energy-reduction deviations ({len(SUPPORTED_CATEGORIES)} supported "
        "categories), explains each one from the measured audio, and lets you listen to the "
        "corresponding excerpts and export a report. A separate Evaluation Lab measures the "
        "detector against benchmark recordings with known, injected flaws."
    )
    st.info(
        "Limits: candidates are produced by configured rules on acoustic differences from the "
        "reference; they are not confirmed errors. Pitch and other delivery aspects are not "
        "analyzed. Benchmark results come from controlled fixtures and do not by themselves show "
        "performance on natural speech."
    )
    items = collect_status(config, paths, expose_details=expose_details)
    items.extend(participant_status_items(config, paths))
    render_status(items)
    if config.ui.show_roadmap:
        render_roadmap(PHASES)


__all__ = ["render_analyze_page", "render_overview_page", "participant_status_items"]
