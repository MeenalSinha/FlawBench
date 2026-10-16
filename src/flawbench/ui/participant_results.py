"""Rendering of participant-analysis results: summary, timeline, evidence, guidance, export.

UI code only renders values from the validated :class:`ParticipantAnalysis` and the in-memory
analysis artifacts. It performs no detection or explanation logic of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from flawbench.config import Config
from flawbench.exceptions import FlawBenchError
from flawbench.explainability.reports import EXTENSIONS, export_report
from flawbench.explainability.schema import (
    REF_ANCHOR,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NO_CANDIDATES,
    STATUS_PARTIAL,
    ExplanationCard,
    ParticipantAnalysis,
)
from flawbench.explainability.templates import CATEGORY_LABELS
from flawbench.participant.analyzer import AnalysisSession
from flawbench.participant.playback import (
    participant_excerpt,
    reference_excerpt,
    slice_excerpt,
    to_wav_bytes,
)

SESSION_KEY = "flawbench_session"
SELECTED_KEY = "flawbench_selected_prediction"

# Okabe-Ito colour-blind-safe palette; categories also differ by dash style and text label.
CATEGORY_STYLE: dict[str, tuple[str, str, str]] = {
    "pause_insertion": ("230,159,0", "solid", "Pause"),
    "tempo_change": ("0,114,178", "dash", "Tempo"),
    "energy_reduction": ("204,121,167", "dot", "Energy"),
}
STATUS_LABEL = {
    STATUS_COMPLETED: "Completed (candidates found)",
    STATUS_NO_CANDIDATES: "Completed (no candidates)",
    STATUS_PARTIAL: "Partial analysis",
    STATUS_FAILED: "Failed",
}


def current_session() -> AnalysisSession | None:
    value = st.session_state.get(SESSION_KEY)
    return value if isinstance(value, AnalysisSession) else None


def _no_result() -> AnalysisSession | None:
    session = current_session()
    if session is None:
        st.info(
            "No analysis has been run in this session. Open Analyze Speech, choose a reference "
            "and a participant recording, and run the analysis."
        )
    elif session.analysis.status == STATUS_FAILED or session.artifacts is None:
        st.error(f"The analysis did not complete: {session.analysis.failure or 'unknown failure'}")
        return None
    return session if session and session.artifacts is not None else None


def render_summary(a: ParticipantAnalysis) -> None:
    st.subheader("Analysis summary")
    label = STATUS_LABEL.get(a.status, a.status)
    if a.status == STATUS_FAILED:
        st.error(f"Status: {label}. {a.failure}")
    elif a.status == STATUS_PARTIAL:
        st.warning(f"Status: {label}. Some detectors failed; the result below is incomplete.")
    elif a.status == STATUS_NO_CANDIDATES:
        st.success(f"Status: {label}.")
    else:
        st.success(f"Status: {label}.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates", a.prediction_count)
    c2.metric("Categories found", len(a.categories_detected))
    c3.metric(
        "Participant length",
        f"{a.participant.duration_seconds:.2f} s" if a.participant else "n/a",
    )
    c4.metric("Session", a.session_id.replace("session_", ""))
    if a.reference and a.participant:
        st.caption(
            f"Reference: {a.reference.label} ({a.reference.sample_rate} Hz). "
            f"Participant: {a.participant.label} ({a.participant.sample_rate} Hz, "
            f"{a.participant.channels} ch)."
        )
        if a.reference.is_synthetic_fixture:
            st.caption("The reference is declared as a synthetic fixture recording.")
    st.caption(
        "Categories analyzed: "
        + ", ".join(CATEGORY_LABELS.get(c, c) for c in a.categories_completed)
        + ". Categories that failed: "
        + (", ".join(a.categories_failed) or "none")
        + "."
    )
    for k, msg in a.categories_failed.items():
        st.warning(f"{CATEGORY_LABELS.get(k, k)} detector failed: {msg}")
    for w in a.warnings:
        st.warning(w)
    if a.status == STATUS_NO_CANDIDATES:
        st.info(
            "The analysis completed, and no candidate met the detector's decision rules under "
            "the current settings. This does not show that the delivery is free of problems: the "
            "result depends on the supported categories, the input quality, the alignment and the "
            "configured thresholds."
        )
    if a.transcript is not None:
        t = a.transcript
        if t.performed:
            st.caption(
                f"Transcript text comparison: word-sequence similarity {t.similarity:.3f}. "
                "No audio-text alignment is available, so no word-level references are shown."
            )
            if t.below_warning_threshold:
                st.warning("The transcripts differ noticeably; the recordings may not correspond.")
        else:
            st.caption(f"Transcript comparison not performed. {t.note}")
    with st.expander("Stage status"):
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in a.stages]), hide_index=True, width="stretch"
        )
    with st.expander("Limitations of this analysis"):
        for x in a.limitations:
            st.markdown(f"- {x}")


def _timeline_figure(session: AnalysisSession, selected: str | None) -> go.Figure:
    art = session.artifacts
    assert art is not None
    feat = art.participant_features
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=feat.center_times_sec,
            y=feat.rms_db,
            mode="lines",
            name="Participant RMS energy (dB)",
            line=dict(color="#52606d", width=1.5),
        )
    )
    for i, card in enumerate(session.analysis.cards, 1):
        rgb, dash, short = CATEGORY_STYLE.get(card.flaw_type, ("120,120,120", "solid", "Other"))
        v = card.bundle.variant
        is_sel = card.prediction_id == selected
        fig.add_vrect(
            x0=v.start_sec,
            x1=v.end_sec,
            fillcolor=f"rgba({rgb},{0.35 if is_sel else 0.2})",
            line=dict(color=f"rgb({rgb})", width=3 if is_sel else 1.5, dash=dash),
            annotation_text=f"#{i} {short} candidate",
            annotation_position="top left",
            layer="below",
        )
    fig.update_layout(
        title="Participant recording: measured RMS energy with candidate intervals",
        xaxis_title="Participant time (seconds)",
        yaxis_title="RMS level (dB)",
        height=360,
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


def _card_table(cards: list[ExplanationCard]) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(cards, 1):
        v = c.bundle.variant
        rows.append(
            {
                "#": i,
                "Type": CATEGORY_LABELS.get(c.flaw_type, c.flaw_type),
                "Start (s)": round(v.start_sec, 3),
                "End (s)": round(v.end_sec, 3),
                "Length (s)": round(v.duration_sec, 3),
                "Evidence strength": round(c.bundle.evidence_strength, 3),
                "Reference": c.bundle.reference.status,
            }
        )
    return pd.DataFrame(rows)


def _select_card(cards: list[ExplanationCard]) -> ExplanationCard:
    labels = {
        c.prediction_id: f"#{i} {CATEGORY_LABELS.get(c.flaw_type, c.flaw_type)} "
        f"{c.bundle.variant.start_sec:.2f} s to {c.bundle.variant.end_sec:.2f} s"
        for i, c in enumerate(cards, 1)
    }
    ids = list(labels)
    if st.session_state.get(SELECTED_KEY) not in ids:
        st.session_state[SELECTED_KEY] = ids[0]
    chosen = st.selectbox(
        "Select a candidate",
        ids,
        format_func=lambda k: labels[k],
        key=SELECTED_KEY,
    )
    return next(c for c in cards if c.prediction_id == chosen)


def _render_playback(session: AnalysisSession, card: ExplanationCard, config: Config) -> None:
    art = session.artifacts
    assert art is not None
    ex = config.explanation
    st.markdown("#### Listen")
    st.caption(
        "Excerpts are cut from the actual recordings with "
        f"{ex.playback_context_sec:g} s of context on each side, clamped to the recording. "
        "The player starts at the beginning of the excerpt; it does not seek automatically."
    )
    left, right = st.columns(2)
    pa = art.participant_audio
    pb = participant_excerpt(
        card.bundle,
        total_samples=pa.num_samples,
        margin_sec=ex.playback_context_sec,
        max_excerpt_sec=ex.max_excerpt_sec,
    )
    with left:
        st.markdown("**Participant excerpt**")
        st.audio(to_wav_bytes(slice_excerpt(pa.samples, pb), pa.sample_rate), format="audio/wav")
        offset = (pb.region_start_sample - pb.start_sample) / pb.sample_rate
        st.caption(
            f"{pb.start_sample / pb.sample_rate:.2f} s to {pb.end_sample / pb.sample_rate:.2f} s "
            f"of the participant recording; the flagged region begins {offset:.2f} s into the clip."
        )
    ra = art.reference_audio
    rb = reference_excerpt(
        card.bundle,
        reference_sample_rate=ra.sample_rate,
        reference_total_samples=ra.num_samples,
        margin_sec=ex.playback_context_sec,
        max_excerpt_sec=ex.max_excerpt_sec,
    )
    with right:
        st.markdown("**Reference excerpt**")
        if rb is None:
            st.info(
                "Reference alignment is unavailable for this candidate, so no reference excerpt "
                "is offered."
            )
        else:
            st.audio(
                to_wav_bytes(slice_excerpt(ra.samples, rb), ra.sample_rate), format="audio/wav"
            )
            what = (
                "context around the anchor point where the participant pause has no counterpart"
                if card.bundle.reference.status == REF_ANCHOR
                else "the DTW-matched reference region plus context"
            )
            st.caption(
                f"{rb.start_sample / rb.sample_rate:.2f} s to {rb.end_sample / rb.sample_rate:.2f} "
                f"s of the reference recording ({what}). The reference position is an estimate."
            )


def render_card(card: ExplanationCard, *, show_evidence: bool = True) -> None:
    """Render one explanation card with its five fixed sections."""
    st.markdown(f"### {card.title}")
    if card.status != "ok":
        st.warning(
            "Wording verification failed for this candidate, so only the recorded interval is "
            "stated."
        )
    st.markdown(f"**Observation.** {card.observation}")
    if show_evidence:
        st.markdown("**Evidence.**")
        for line in card.evidence_lines:
            st.markdown(f"- {line}")
    st.markdown(f"**Interpretation.** {card.interpretation}")
    st.markdown(f"**Suggested review.** {card.suggested_review}")
    st.markdown("**Limitations.**")
    for x in card.limitations:
        st.markdown(f"- {x}")
    for w in card.warnings:
        st.warning(w)
    for n in card.detector_notes:
        st.caption(f"Detector note: {n}")
    if card.missing_facts:
        st.caption("Unavailable measurements: " + ", ".join(card.missing_facts))


def render_timeline_page(config: Config) -> None:
    st.header("Flaw Timeline")
    session = _no_result()
    if session is None:
        return
    a = session.analysis
    if not a.cards:
        render_summary(a)
        return
    st.caption(
        "Shaded regions are candidates produced by configured detection rules. They are not "
        "confirmed flaws. Candidates are distinguished by label and outline style as well as color."
    )
    selected = st.session_state.get(SELECTED_KEY)
    st.plotly_chart(_timeline_figure(session, selected), use_container_width=True)
    st.dataframe(_card_table(a.cards), hide_index=True, width="stretch")
    card = _select_card(a.cards)
    v = card.bundle.variant
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Start", f"{v.start_sec:.2f} s")
    c2.metric("End", f"{v.end_sec:.2f} s")
    c3.metric("Length", f"{v.duration_sec:.2f} s")
    c4.metric("Evidence strength", f"{card.bundle.evidence_strength:.3f}")
    st.caption(
        "Evidence strength is an uncalibrated detector score from zero to one. It is not a "
        "probability or a confidence."
    )
    render_card(card, show_evidence=config.explanation.show_technical_evidence)
    _render_playback(session, card, config)


def _evidence_figures(session: AnalysisSession, card: ExplanationCard) -> list[go.Figure]:
    art = session.artifacts
    assert art is not None
    comp = art.detection_input.comparison
    v = card.bundle.variant
    ref = card.bundle.reference
    figs: list[go.Figure] = []

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=comp.var_times_sec, y=comp.aligned_var_rms_db, name="Participant"))
    f1.add_trace(
        go.Scatter(x=comp.var_times_sec, y=comp.aligned_ref_rms_db, name="Aligned reference")
    )
    f1.add_vrect(x0=v.start_sec, x1=v.end_sec, fillcolor="rgba(0,114,178,0.15)", line_width=1)
    f1.update_layout(
        title="RMS level on the participant timeline (reference warped onto it)",
        xaxis_title="Participant time (s)", yaxis_title="RMS level (dB)", height=300,
        margin=dict(l=50, r=20, t=50, b=40),
    )  # fmt: skip
    figs.append(f1)

    f2 = go.Figure(go.Scatter(x=comp.var_times_sec, y=comp.delta_rms_db, name="Difference"))
    f2.add_vrect(x0=v.start_sec, x1=v.end_sec, fillcolor="rgba(0,114,178,0.15)", line_width=1)
    f2.update_layout(
        title="RMS level difference (participant minus reference)",
        xaxis_title="Participant time (s)", yaxis_title="Difference (dB)", height=260,
        margin=dict(l=50, r=20, t=50, b=40),
    )  # fmt: skip
    figs.append(f2)

    f3 = go.Figure(go.Scatter(x=comp.var_times_sec, y=comp.local_slope, name="Warp slope"))
    f3.add_vrect(x0=v.start_sec, x1=v.end_sec, fillcolor="rgba(0,114,178,0.15)", line_width=1)
    f3.update_layout(
        title="Local warp slope (participant time per reference time)",
        xaxis_title="Participant time (s)", yaxis_title="Slope (ratio)", height=260,
        margin=dict(l=50, r=20, t=50, b=40),
    )  # fmt: skip
    figs.append(f3)

    f4 = go.Figure(
        go.Scatter(x=comp.var_times_sec, y=comp.ref_times_sec, mode="lines", name="Path")
    )
    f4.add_vrect(x0=v.start_sec, x1=v.end_sec, fillcolor="rgba(0,114,178,0.15)", line_width=1)
    if ref.start_sec is not None:
        f4.add_hrect(
            y0=ref.start_sec, y1=ref.end_sec if ref.end_sec is not None else ref.start_sec,
            line=dict(color="#d55e00", dash="dot", width=2),
        )  # fmt: skip
    f4.update_layout(
        title="DTW correspondence (reference time against participant time)",
        xaxis_title="Participant time (s)", yaxis_title="Reference time (s)", height=300,
        margin=dict(l=50, r=20, t=50, b=40),
    )  # fmt: skip
    figs.append(f4)
    return figs


def render_evidence_page(config: Config) -> None:
    st.header("Acoustic Evidence")
    session = _no_result()
    if session is None:
        return
    a = session.analysis
    if not a.cards:
        st.info("No candidates were detected, so there is no candidate-specific evidence to show.")
        return
    card = _select_card(a.cards)
    st.subheader("Measured values")
    show_defs = config.explanation.show_technical_evidence
    rows: list[dict[str, Any]] = []
    for f in card.bundle.facts:
        row: dict[str, Any] = {
            "Measurement": f.label,
            "Value": round(f.value, 4) if f.valid and f.value is not None else "unavailable",
            "Unit": f.unit,
            "Role": f.role,
            "Source": f.source,
        }
        if show_defs:
            row["Definition"] = f.definition
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if card.bundle.ignored_evidence_keys:
        st.caption(
            "Detector fields not interpreted: " + ", ".join(card.bundle.ignored_evidence_keys)
        )
    st.caption(
        "Participant interval (half-open): samples "
        f"[{card.bundle.variant.start_sample}, {card.bundle.variant.end_sample}) at "
        f"{card.bundle.variant.sample_rate} Hz. Reference correspondence: "
        f"{card.bundle.reference.status}."
    )
    st.subheader("Reference comparison plots")
    st.caption(
        "The reference is aligned to the participant by DTW; equal timestamps do not imply the "
        "same phrase. The shaded band marks the selected participant interval."
    )
    for fig in _evidence_figures(session, card):
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Full feature comparison (Phase 3 plots)"):
        from flawbench.ui.feature_panel import render_comparison_plots

        render_comparison_plots(session.artifacts.detection_input.comparison)  # type: ignore[union-attr]


def render_guidance_page(config: Config) -> None:
    st.header("Improvement Guidance")
    session = _no_result()
    if session is None:
        return
    a = session.analysis
    st.caption(
        "Suggestions describe what to review, based only on measured differences from the "
        "reference. They are not predictions of improvement and not judgments of quality."
    )
    if not a.cards:
        st.info(
            "No candidates were detected, so there is nothing specific to review. This does not "
            "show that the delivery is free of problems."
        )
    for i, c in enumerate(a.cards, 1):
        v = c.bundle.variant
        with st.expander(
            f"#{i} {c.title}: {v.start_sec:.2f} s to {v.end_sec:.2f} s", expanded=(i == 1)
        ):
            st.markdown(f"**Suggested review.** {c.suggested_review}")
            st.markdown("**Keep in mind.**")
            for x in c.limitations:
                st.markdown(f"- {x}")
    bm = a.benchmark
    if config.explanation.show_benchmark_context:
        st.subheader("Benchmark context (separate from this recording)")
        if bm and bm.available:
            st.caption(
                f"Saved evaluation run `{bm.run_id}`, split `{bm.split}`, "
                f"{bm.sample_count} samples."
            )
            if bm.is_synthetic_only:
                st.warning(
                    "This benchmark uses synthetic fixture audio only. It does not show "
                    "generalization to natural human speech."
                )
            st.caption(bm.note)
            rows: list[dict[str, Any]] = []
            for category, metrics in bm.per_category.items():
                row: dict[str, Any] = {"Category": category}
                for name, value in metrics.items():
                    if value is None:
                        row[name] = "N/A"
                    elif isinstance(value, float):
                        row[name] = round(value, 3)
                    else:
                        row[name] = value
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("Benchmark results are unavailable: no valid saved evaluation run was found.")
    render_export_controls(session, config)


def render_export_controls(session: AnalysisSession, config: Config) -> None:
    st.subheader("Export report")
    st.caption(
        "Reports contain measurements, explanations and provenance. Raw participant audio is "
        "never included. Files are written only when you click the button."
    )
    formats = list(config.explanation.export_formats)
    chosen = st.multiselect("Formats", formats, default=formats, key="flawbench_export_formats")
    if st.button("Write report files", key="flawbench_export"):
        if not chosen:
            st.error("Select at least one format.")
            return
        try:
            written = export_report(session.analysis, config.explanation, formats=tuple(chosen))
        except FlawBenchError as exc:
            st.error(f"Report export failed: {exc.public_message(expose_details=True)}")
            return
        st.session_state["flawbench_written"] = {k: str(v) for k, v in written.items()}
    written_paths = st.session_state.get("flawbench_written")
    if written_paths:
        for fmt, path in written_paths.items():
            p = Path(path)
            if p.is_file():
                st.download_button(
                    f"Download {fmt} ({EXTENSIONS[fmt]})",
                    p.read_bytes(),
                    file_name=p.name,
                    key=f"flawbench_dl_{fmt}",
                )
        st.caption(
            "Saved to: " + ", ".join(sorted({str(Path(q).parent) for q in written_paths.values()}))
        )
