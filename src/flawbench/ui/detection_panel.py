"""Phase 4 Speech Flaw Detection & Temporal Localization UI Panel.

Provides interactive visualization of independently predicted speech delivery flaws:
- Selection of registered reference and variant recordings
- Independent flaw detection with configurable algorithms
- Temporal interval localization on variant timeline with sample indices
- Deep acoustic evidence inspection (silence depth, warp slope, delta RMS)
- Zero ground-truth leakage
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from flawbench.config import Config
from flawbench.dataset.io import read_jsonl_lenient
from flawbench.detection.pipeline import DetectionPipeline
from flawbench.detection.schema import DetectionResult
from flawbench.exceptions import FlawBenchError
from flawbench.paths import ProjectPaths, resolve_within


def render_prediction_timeline_plot(
    result: DetectionResult, var_times: np.ndarray | None = None, var_rms: np.ndarray | None = None
) -> None:
    """Render Plotly visualization showing predicted flaw intervals on the variant timeline."""
    dur = result.duration_seconds
    fig = go.Figure()

    # If RMS curve is available, plot background energy curve
    if var_times is not None and var_rms is not None and len(var_times) == len(var_rms):
        fig.add_trace(
            go.Scatter(
                x=var_times,
                y=var_rms,
                mode="lines",
                name="Variant RMS (dB)",
                line=dict(color="#6c757d", width=1),
                opacity=0.7,
            )
        )

    # Color palette for predicted flaw categories
    colors = {
        "pause_insertion": "rgba(255, 165, 0, 0.4)",
        "tempo_change": "rgba(147, 112, 219, 0.4)",
        "energy_reduction": "rgba(0, 191, 255, 0.4)",
    }
    border_colors = {
        "pause_insertion": "#ff8c00",
        "tempo_change": "#8a2be2",
        "energy_reduction": "#00bfff",
    }

    for p in result.predictions:
        c_fill = colors.get(p.flaw_type, "rgba(255, 99, 71, 0.4)")
        c_border = border_colors.get(p.flaw_type, "#ff6347")

        fig.add_vrect(
            x0=p.start_sec,
            x1=p.end_sec,
            fillcolor=c_fill,
            line=dict(color=c_border, width=2),
            annotation_text=f"{p.flaw_type} ({p.duration_sec:.2f}s)",
            annotation_position="top left",
        )

    fig.update_layout(
        title=(
            f"Sample '{result.sample_id}' — Independent Flaw Detections "
            f"({len(result.predictions)} predicted)"
        ),
        xaxis_title="Variant Timeline (seconds)",
        yaxis_title="Acoustic Energy (dB)" if var_rms is not None else "Flaw Region Indicator",
        xaxis=dict(range=[0, dur]),
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_detection_panel(
    config: Config, paths: ProjectPaths, expose_details: bool = False
) -> None:
    """Render the Phase 4 flaw detection panel."""
    st.header("Flaw Detection & Temporal Localization")
    st.caption(
        "Independently detects speech delivery flaws (pauses, tempo changes, energy attenuation) "
        "and localizes them on the modified recording timeline without ground-truth labels."
    )

    manifest_file = paths.manifest_file
    if not manifest_file.is_file():
        st.info(
            "No generated variants found in manifest.jsonl. "
            "Generate variants first using the dataset panel."
        )
        return

    rows, problems = read_jsonl_lenient(manifest_file)
    if problems:
        st.warning(f"Manifest warning: {'; '.join(problems)}")

    samples = [r for _, r in rows if r.get("sample_id")]
    if not samples:
        st.info("No committed samples found in manifest.jsonl.")
        return

    sample_options = [s["sample_id"] for s in samples]
    selected_sample_id = st.selectbox(
        "Select Variant Sample for Flaw Detection",
        options=sample_options,
        key="det_select_sample",
    )

    selected_sample = next((s for s in samples if s["sample_id"] == selected_sample_id), None)
    if not selected_sample:
        return

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.write(f"**Sample ID:** `{selected_sample_id}`")
    col_info2.write(f"**Reference Source:** `{selected_sample.get('source_id')}`")
    gen_data = selected_sample.get("generated", {})
    dur_sec = gen_data.get("duration_seconds", 0.0)
    col_info3.write(f"**Duration:** {dur_sec:.2f} s")

    # Audio playback
    var_audio_rel = gen_data.get("path")
    if var_audio_rel:
        try:
            var_audio_path = resolve_within(paths.generated_dir, str(var_audio_rel))
            if var_audio_path.is_file():
                st.audio(str(var_audio_path))
        except Exception:
            pass

    col_btn1, col_btn2 = st.columns([1, 4])
    run_detection = col_btn1.button("Run Flaw Detection", type="primary", key="btn_run_detection")
    force_recompute = col_btn2.checkbox("Force recomputation (ignore cache)", key="det_force_chk")

    pipeline = DetectionPipeline(config=config, paths=paths)

    if run_detection:
        with st.spinner(f"Running flaw detection on sample '{selected_sample_id}'..."):
            try:
                result = pipeline.detect_sample(selected_sample_id, force=force_recompute)
                st.session_state[f"detection_result_{selected_sample_id}"] = result
            except FlawBenchError as exc:
                st.error(f"Detection failed: {exc.public_message(expose_details=expose_details)}")
                return
            except Exception as exc:
                st.error(f"Unexpected detection failure: {exc}")
                return

    stored_res: DetectionResult | None = st.session_state.get(
        f"detection_result_{selected_sample_id}"
    )
    if stored_res is None:
        stored_res = pipeline.store.get(selected_sample_id)

    if stored_res is not None:
        st.success(
            f"Detection completed. Predicted flaws: {len(stored_res.predictions)} "
            f"(from {stored_res.raw_candidates_count} raw candidates)."
        )

        # Plot timeline
        # Retrieve variant RMS if features are cached
        v_feat = pipeline.feature_pipeline.store.get_by_checksum(
            stored_res.variant_checksum, pipeline.feature_pipeline._config_fingerprint()
        )
        if v_feat is not None:
            render_prediction_timeline_plot(
                stored_res, var_times=v_feat.center_times_sec, var_rms=v_feat.rms_db
            )
        else:
            render_prediction_timeline_plot(stored_res)

        if len(stored_res.predictions) == 0:
            st.info(
                "No delivery flaws detected under current sensitivity settings. "
                "The signal acoustic differences did not exceed configured flaw thresholds."
            )
        else:
            st.markdown("### Predicted Flaw Intervals & Acoustic Evidence")
            for i, p in enumerate(stored_res.predictions, start=1):
                with st.expander(
                    f"Prediction #{i}: **{p.flaw_type}** | [{p.start_sec:.2f}s – {p.end_sec:.2f}s] "
                    f"(dur: {p.duration_sec:.2f}s, strength: {p.evidence_strength:.2f})",
                    expanded=True,
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Interval (samples)", f"[{p.start_sample} : {p.end_sample}]")
                    c1.caption(f"Sample Rate: {stored_res.sample_rate} Hz")
                    c2.metric("Interval (seconds)", f"{p.start_sec:.2f}s – {p.end_sec:.2f}s")
                    c2.caption(f"Duration: {p.duration_sec:.2f}s")
                    c3.metric("Evidence Strength", f"{p.evidence_strength:.2f}")
                    c3.caption(f"Detector: `{p.detector_id}`")

                    if p.ref_start_sec is not None and p.ref_end_sec is not None:
                        ref_dur = p.ref_end_sec - p.ref_start_sec
                        st.markdown(
                            f"**Aligned Reference Region:** "
                            f"`{p.ref_start_sec:.2f}s – {p.ref_end_sec:.2f}s` (dur: {ref_dur:.2f}s)"
                        )

                    st.markdown("**Acoustic Evidence Measurements:**")
                    st.json(p.evidence)

                    if p.warnings:
                        for w in p.warnings:
                            st.warning(f"Uncertainty warning: {w}")

        with st.expander("Detector Run Provenance & Effective Settings"):
            st.write(f"**Detector Version:** `{stored_res.detector_version}`")
            st.write(f"**Config Fingerprint:** `{stored_res.config_fingerprint}`")
            st.write(f"**Reference Checksum:** `{stored_res.ref_checksum}`")
            st.write(f"**Variant Checksum:** `{stored_res.variant_checksum}`")
            st.write(f"**Created At:** `{stored_res.created_at}`")
            st.json(config.detection.__dict__)
