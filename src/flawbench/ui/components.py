"""Streamlit rendering helpers. They render real data passed in; they compute nothing themselves."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from flawbench.audio.loader import AudioMetadata
from flawbench.roadmap import IMPLEMENTED, Phase
from flawbench.status import StatusItem

TAGLINE = "Evidence-Based Speech Flaw Analysis"
PURPOSE = (
    "FlawBench evaluates speech-delivery analysis against exact ground truth. Controlled flaws "
    "will be injected into clean reference recordings with recorded intervals, then detected "
    "without access to those labels and scored against them. This release provides only the "
    "project foundation and audio-input inspection; no speech analysis is performed yet."
)


def render_header(app_name: str) -> None:
    st.title(app_name)
    st.caption(f"*{TAGLINE}*")
    st.write(PURPOSE)


def render_status(items: Sequence[StatusItem]) -> None:
    st.subheader("Application status")
    for item in items:
        marker = "OK" if item.ok else "ATTENTION"
        (st.success if item.ok else st.error)(f"**{item.label}** [{marker}]: {item.message}")


def render_metadata(meta: AudioMetadata) -> None:
    st.subheader("Audio file details")
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{meta.duration_seconds:.2f} s")
    col2.metric("Sample rate", f"{meta.sample_rate:,} Hz")
    col3.metric("Channels", str(meta.channels))
    col4, col5, col6 = st.columns(3)
    col4.metric("File size", f"{meta.file_size_bytes / 1024:,.1f} KiB")
    col5.metric("Container", meta.container_format)
    col6.metric("Encoding", meta.subtype)
    st.caption(f"{meta.frames:,} frames decoded successfully. Samples were not modified.")


def render_roadmap(phases: Sequence[Phase]) -> None:
    st.subheader("Project roadmap")
    st.caption("Informational. Only phases marked Implemented exist in this repository.")
    for phase in phases:
        tag = "done" if phase.status == IMPLEMENTED else "planned"
        st.markdown(
            f"- **Phase {phase.number}: {phase.title}** ({tag}): {phase.summary} *{phase.status}*"
        )
