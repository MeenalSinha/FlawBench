"""Phase 3 Acoustic Feature Extraction & Comparison UI Panel.

Provides interactive visualization of time-indexed acoustic features:
- RMS energy trajectories and silence threshold
- Fundamental frequency (F0 / pitch) contour and voiced segments
- Spectral centroid, bandwidth, and rolloff curves
- MFCC heatmap
- Differential analysis: DTW warp path, delta RMS, and candidate flaw regions
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from flawbench.config import Config
from flawbench.exceptions import FlawBenchError
from flawbench.features.comparison import ComparisonResult
from flawbench.features.extractor import FeatureResult
from flawbench.features.pipeline import FeaturePipeline
from flawbench.paths import ProjectPaths

FEATURES_HINT = "python -m flawbench features extract --all"


def render_feature_result_plots(result: FeatureResult, title_prefix: str = "") -> None:
    """Render interactive Plotly figures for a single FeatureResult."""
    times = result.center_times_sec
    dur = result.metadata.duration_seconds
    prefix = f"{title_prefix} — " if title_prefix else ""

    st.markdown(f"#### {prefix}Acoustic Feature Trajectories")

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Duration", f"{dur:.2f} s")
    m2.metric("Frames", result.num_frames)
    m3.metric("Mean RMS", f"{result.summary_stats.get('mean_rms_db', 0):.1f} dB")
    m4.metric("Voiced ratio", f"{result.summary_stats.get('voiced_ratio', 0) * 100:.1f} %")
    m5.metric("Mean F0", f"{result.summary_stats.get('f0_mean_voiced_hz', 0):.1f} Hz")

    # 1. Energy & Silence
    fig_rms = go.Figure()
    fig_rms.add_trace(
        go.Scatter(x=times, y=result.rms_db, mode="lines", name="RMS (dB)", line=dict(color="#1f77b4"))
    )
    thresh = result.metadata.silence_threshold_db
    fig_rms.add_hline(
        y=thresh,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Silence threshold ({thresh:.0f} dB)",
    )
    fig_rms.update_layout(
        title=f"{prefix}RMS Energy (dB) over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Energy (dB)",
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig_rms, use_container_width=True)

    # 2. Pitch / F0 & Spectral features in 2 tabs
    tab_pitch, tab_spectral, tab_mfcc, tab_intervals = st.tabs(
        ["Pitch (F0)", "Spectral Features", "MFCC Matrix", "Activity Intervals"]
    )

    with tab_pitch:
        fig_pitch = go.Figure()
        # Plot only finite voiced points
        voiced_mask = result.voiced_flag & np.isfinite(result.f0_hz)
        fig_pitch.add_trace(
            go.Scatter(
                x=times[voiced_mask],
                y=result.f0_hz[voiced_mask],
                mode="markers+lines",
                name="Pitch (F0)",
                marker=dict(size=4, color="#2ca02c"),
                line=dict(color="#2ca02c", width=1),
            )
        )
        fig_pitch.update_layout(
            title=f"{prefix}Fundamental Frequency (F0 in Hz)",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            height=300,
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_pitch, use_container_width=True)

    with tab_spectral:
        fig_spec = go.Figure()
        fig_spec.add_trace(
            go.Scatter(
                x=times,
                y=result.spectral_centroid,
                mode="lines",
                name="Centroid (Hz)",
                line=dict(color="#ff7f0e"),
            )
        )
        fig_spec.add_trace(
            go.Scatter(
                x=times,
                y=result.spectral_rolloff,
                mode="lines",
                name="Rolloff (Hz)",
                line=dict(color="#9467bd", dash="dot"),
            )
        )
        fig_spec.update_layout(
            title=f"{prefix}Spectral Centroid & Rolloff",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            height=300,
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    with tab_mfcc:
        fig_mfcc = go.Figure(
            data=go.Heatmap(
                z=result.mfcc.T,
                x=times,
                y=[f"MFCC {i+1}" for i in range(result.mfcc.shape[1])],
                colorscale="Viridis",
            )
        )
        fig_mfcc.update_layout(
            title=f"{prefix}MFCC Heatmap",
            xaxis_title="Time (seconds)",
            yaxis_title="Coefficient",
            height=320,
            margin=dict(l=60, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_mfcc, use_container_width=True)

    with tab_intervals:
        if result.activity_intervals:
            st.dataframe(
                [
                    {
                        "start_sec": round(i.start_sec, 3),
                        "end_sec": round(i.end_sec, 3),
                        "duration_sec": round(i.end_sec - i.start_sec, 3),
                        "state": "Active speech" if i.is_active else "Silence / Pause",
                        "mean_rms_db": round(i.mean_rms_db, 1),
                    }
                    for i in result.activity_intervals
                ],
                hide_index=True,
            )
        else:
            st.caption("No activity intervals detected.")


def render_comparison_plots(comparison: ComparisonResult) -> None:
    """Render interactive differential comparison plots."""
    st.markdown("#### Reference vs Variant Differential Analysis")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DTW Normalized Cost", f"{comparison.normalized_dtw_cost:.3f}")
    m2.metric("Duration Ratio", f"{comparison.global_metrics.get('duration_ratio', 1.0):.2f}")
    m3.metric(
        "Mean Delta RMS", f"{comparison.global_metrics.get('mean_delta_rms_db', 0.0):.2f} dB"
    )
    m4.metric("Candidate Anomaly Regions", len(comparison.candidate_regions))

    # 1. DTW Warp Path & Differential Energy
    fig_dtw = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[
            "DTW Alignment Warp Path (Variant Timeline vs Reference Timeline)",
            "Differential RMS Energy: Δ RMS (dB) = Variant - Reference (Aligned)",
        ],
        vertical_spacing=0.15,
    )

    # Warp path
    fig_dtw.add_trace(
        go.Scatter(
            x=comparison.ref_times_sec,
            y=comparison.var_times_sec,
            mode="lines",
            name="Warp Path",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1,
        col=1,
    )
    # Ideal diagonal
    max_t = max(float(comparison.ref_times_sec[-1]), float(comparison.var_times_sec[-1]))
    fig_dtw.add_trace(
        go.Scatter(
            x=[0, max_t],
            y=[0, max_t],
            mode="lines",
            name="1:1 Timeline Diagonal",
            line=dict(color="#7f7f7f", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Delta RMS
    fig_dtw.add_trace(
        go.Scatter(
            x=comparison.var_times_sec,
            y=comparison.delta_rms_db,
            mode="lines",
            name="Δ RMS (dB)",
            line=dict(color="#d62728", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig_dtw.add_hline(
        y=0.0, line_dash="solid", line_color="#333333", row=2, col=1
    )
    fig_dtw.add_hline(
        y=-6.0,
        line_dash="dot",
        line_color="#e377c2",
        annotation_text="Significant attenuation (-6 dB)",
        row=2,
        col=1,
    )

    fig_dtw.update_xaxes(title_text="Reference Timeline (seconds)", row=1, col=1)
    fig_dtw.update_yaxes(title_text="Variant Timeline (seconds)", row=1, col=1)
    fig_dtw.update_xaxes(title_text="Variant Timeline (seconds)", row=2, col=1)
    fig_dtw.update_yaxes(title_text="Δ RMS (dB)", row=2, col=1)
    fig_dtw.update_layout(height=600, margin=dict(l=50, r=20, t=40, b=30))
    st.plotly_chart(fig_dtw, use_container_width=True)

    # Candidate regions table
    st.markdown("##### Heuristic Candidate Flaw Regions (Differential Detection)")
    if comparison.candidate_regions:
        st.dataframe(
            [
                {
                    "Hypothesis": r.get("flaw_hypothesis"),
                    "Variant start (s)": r.get("var_start_sec"),
                    "Variant end (s)": r.get("var_end_sec"),
                    "Duration (s)": r.get("duration_sec"),
                    "Key Metric": r.get("mean_delta_rms_db")
                    or r.get("mean_relative_rate")
                    or r.get("mean_var_rms_db"),
                }
                for r in comparison.candidate_regions
            ],
            hide_index=True,
        )
    else:
        st.info("No significant acoustic anomaly regions detected.")


def render_features_panel(config: Config, paths: ProjectPaths, *, expose_details: bool) -> None:
    """Streamlit UI panel for Phase 3 Feature Extraction & Time-Series Comparison."""
    st.subheader("Acoustic Features & Time-Series Analysis (Phase 3)")
    st.caption(
        "Extract canonical time-indexed acoustic features (RMS energy, F0 pitch, spectral features, MFCCs) "
        "and perform DTW differential comparison between variants and references."
    )

    try:
        pipeline = FeaturePipeline(config, paths)
        manifest_entries = pipeline.store.list_manifest()
    except FlawBenchError as exc:
        st.error(f"Features pipeline problem: {exc.public_message(expose_details=expose_details)}")
        return

    n_artifacts = len(manifest_entries)
    col1, col2 = st.columns(2)
    col1.metric("Feature artifacts in store", n_artifacts)
    col2.metric("Feature cache policy", config.features.cache_policy)

    if n_artifacts == 0:
        st.info(
            f"No features extracted yet. Run `{FEATURES_HINT}` in the terminal, "
            f"or click below to extract features for the registered dataset."
        )
        if st.button("Extract features for dataset"):
            with st.spinner("Extracting acoustic features..."):
                counts = pipeline.extract_dataset()
                st.success(
                    f"Extraction complete: {counts['references_extracted']} references, "
                    f"{counts['variants_extracted']} variants."
                )
                st.rerun()
        return

    mode = st.radio(
        "View mode",
        ["Inspect Extracted Features", "Reference vs Variant Comparison"],
        horizontal=True,
    )

    if mode == "Inspect Extracted Features":
        options = {
            f"[{e.role.upper()}] {e.artifact_id} (src: {e.source_id}"
            + (f", sample: {e.sample_id}" if e.sample_id else "")
            + f") — {e.duration_seconds:.2f}s": e.artifact_id
            for e in manifest_entries
        }
        chosen_label = st.selectbox("Select feature artifact", list(options.keys()))
        if chosen_label:
            art_id = options[chosen_label]
            try:
                feat = pipeline.store.load(art_id)
                render_feature_result_plots(feat, title_prefix=art_id)
            except FlawBenchError as exc:
                st.error(exc.public_message(expose_details=expose_details))

    elif mode == "Reference vs Variant Comparison":
        # Filter for variant entries
        variants = [e for e in manifest_entries if e.role == "variant" and e.sample_id]
        if not variants:
            st.info("No variant feature artifacts in store. Extract dataset variants first.")
            return

        var_options = {
            f"{v.sample_id} (derived from {v.source_id})": v.sample_id for v in variants
        }
        chosen_var_label = st.selectbox("Select variant sample to compare", list(var_options.keys()))
        if chosen_var_label:
            sid = var_options[chosen_var_label]
            if st.button("Compute Differential Comparison"):
                with st.spinner("Aligning time-series with Dynamic Time Warping (DTW)..."):
                    try:
                        comp = pipeline.compare_variant_to_reference(sid)
                        render_comparison_plots(comp)
                    except FlawBenchError as exc:
                        st.error(exc.public_message(expose_details=expose_details))
