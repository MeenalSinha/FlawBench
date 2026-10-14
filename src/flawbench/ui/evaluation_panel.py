"""Phase 5 Evaluation Lab & Benchmarking UI Panel.

Provides interactive visualization and inspection of formal evaluation metrics:
- Evaluation run configuration and trigger
- Classification performance (precision, recall, F1-score)
- Temporal localization (IoU, boundary error distributions, tolerance curves)
- Event-level confusion matrix & error categorization
- Clean negative controls & false-positive rates
- Provenance disclosure (synthetic audio vs natural human speech)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from flawbench.config import Config
from flawbench.evaluation.runner import EvaluationRunner
from flawbench.evaluation.store import EvaluationStore
from flawbench.exceptions import FlawBenchError
from flawbench.paths import ProjectPaths


def render_evaluation_panel(
    config: Config, paths: ProjectPaths, expose_details: bool = False
) -> None:
    """Render the Phase 5 Evaluation Lab panel."""
    st.header("Formal Evaluation & Benchmarking (Phase 5)")
    st.caption(
        "Evaluates independently generated speech flaw detections against ground-truth "
        "annotations using deterministic one-to-one temporal IoU matching."
    )

    store = EvaluationStore(paths)
    runs = store.list_runs()

    with st.expander("Evaluation Lab & Benchmark Run", expanded=True):
        col_setup1, col_setup2, col_setup3 = st.columns([2, 2, 2])
        with col_setup1:
            split_choice = st.selectbox(
                "Evaluation Split",
                options=["test", "validation", "train", "all"],
                index=0,
                help="Source-level partitioned evaluation cohort.",
            )
        with col_setup2:
            _ = st.slider(
                "Matching IoU Threshold",
                min_value=0.1,
                max_value=0.9,
                value=float(config.evaluation.matching_iou_threshold),
                step=0.05,
                help="Minimum temporal IoU required to count a prediction as correctly localized.",
            )
        with col_setup3:
            enable_bootstrap = st.checkbox(
                "Source-Level Bootstrap (95% CI)",
                value=config.evaluation.bootstrap_replicates > 0,
                help="Compute empirical confidence intervals by resampling source clusters.",
            )
            include_ctrls = st.checkbox(
                "Evaluate Clean Identity Controls",
                value=config.evaluation.include_identity_controls,
                help="Quantify false-positive rates using reference self-comparisons.",
            )

        if st.button("Run Formal Evaluation", type="primary"):
            with st.spinner("Executing independent evaluation and event matching..."):
                try:
                    runner = EvaluationRunner(config, paths)
                    report, run_dir = runner.run(
                        split=split_choice,
                        include_controls=include_ctrls,
                        enable_bootstrap=enable_bootstrap,
                    )
                    st.success(
                        f"Evaluation run `{report.run_id}` complete! Saved to `{run_dir.name}`."
                    )
                    # Refresh run list
                    runs = store.list_runs()
                except FlawBenchError as f_exc:
                    st.error(
                        f"Evaluation error: {f_exc.public_message(expose_details=expose_details)}"
                    )
                except Exception as exc:
                    st.error(f"Unexpected evaluation error: {exc}")

    if not runs:
        st.info(
            "No completed evaluation runs found. Configure settings above and click "
            "'Run Formal Evaluation'."
        )
        return

    # Select an evaluation run to view
    run_options = [r["run_id"] for r in runs if "run_id" in r]
    if not run_options:
        return
    selected_run_id = st.selectbox(
        "Inspect Completed Evaluation Run", options=run_options, index=len(run_options) - 1
    )

    try:
        report = store.load_evaluation(selected_run_id)
    except Exception as exc:
        st.error(f"Could not load evaluation run '{selected_run_id}': {exc}")
        return

    # 1. Provenance & Limitation Notice
    if report.is_synthetic_only:
        st.warning(
            "⚠️ **Synthetic Audio Limitation**: This evaluation was executed solely on "
            "synthetic audio signals. These metrics verify algorithmic detection logic and "
            "interval boundaries, but do NOT prove generalization to natural human speech.",
            icon="⚠️",
        )

    # 2. Headline Metrics
    agg = report.aggregate
    st.subheader(f"Run Overview: `{report.run_id}`")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        f1_val = f"{agg.micro_f1:.3f}" if agg.micro_f1 is not None else "N/A"
        st.metric("Micro F1-Score", f1_val)
    with m_col2:
        p_val = f"{agg.micro_precision:.3f}" if agg.micro_precision is not None else "N/A"
        st.metric("Precision", p_val)
    with m_col3:
        r_val = f"{agg.micro_recall:.3f}" if agg.micro_recall is not None else "N/A"
        st.metric("Recall", r_val)
    with m_col4:
        iou_val = f"{agg.mean_iou:.3f}" if agg.mean_iou is not None else "N/A"
        st.metric("Mean Matched IoU", iou_val)
    with m_col5:
        st.metric("Evaluated Samples", f"{report.sample_count} ({report.source_count} sources)")

    # 3. Per-Category Classification Breakdown
    st.markdown("### Per-Category Classification Performance")
    cat_rows = []
    for cat_name, m in sorted(report.per_category.items()):
        cat_rows.append(
            {
                "Category": cat_name,
                "GT Events": m.gt_count,
                "Pred Events": m.pred_count,
                "TP": m.tp,
                "FP": m.fp,
                "FN": m.fn,
                "Precision": f"{m.precision:.3f}" if m.precision is not None else "N/A",
                "Recall": f"{m.recall:.3f}" if m.recall is not None else "N/A",
                "F1-Score": f"{m.f1:.3f}" if m.f1 is not None else "N/A",
                "Mean IoU": f"{m.mean_iou:.3f}" if m.mean_iou is not None else "N/A",
            }
        )
    if cat_rows:
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

    # 4. Temporal Localization & Tolerance Curve
    st.markdown("### Temporal Localization Analysis")
    loc_col1, loc_col2 = st.columns([3, 2])
    with loc_col1:
        if report.localization_curve:
            thresholds = [p.iou_threshold for p in report.localization_curve]
            recalls = [p.recall if p.recall is not None else 0.0 for p in report.localization_curve]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=thresholds,
                    y=recalls,
                    mode="lines+markers",
                    name="Localization Recall",
                    line=dict(color="#1f77b4", width=3),
                    marker=dict(size=8),
                )
            )
            fig.update_layout(
                title="Localization-Tolerance Curve (Recall vs. IoU Threshold)",
                xaxis_title="IoU Overlap Threshold",
                yaxis_title="Detection Recall",
                yaxis=dict(range=[0, 1.05]),
                height=320,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
    with loc_col2:
        st.markdown("**Boundary Errors (seconds)**")
        b_rows = []
        for cat_name, m in sorted(report.per_category.items()):
            mse_s = (
                f"{m.mean_abs_start_error_sec:.3f}s"
                if m.mean_abs_start_error_sec is not None
                else "N/A"
            )
            mee_s = (
                f"{m.mean_abs_end_error_sec:.3f}s"
                if m.mean_abs_end_error_sec is not None
                else "N/A"
            )
            mde_s = (
                f"{m.mean_abs_dur_error_sec:.3f}s"
                if m.mean_abs_dur_error_sec is not None
                else "N/A"
            )
            b_rows.append(
                {
                    "Category": cat_name,
                    "Mean Start Err": mse_s,
                    "Mean End Err": mee_s,
                    "Mean Dur Err": mde_s,
                }
            )
        if b_rows:
            st.dataframe(pd.DataFrame(b_rows), use_container_width=True)

    # 5. Event-Level Confusion Matrix
    if report.confusion_matrix:
        st.markdown("### Event-Level Confusion Matrix")
        cats = report.confusion_matrix.categories
        cols = cats + ["missed"]
        rows = cats + ["none"]
        matrix_data = []
        for r_name in rows:
            row_dict: dict[str, Any] = {"Ground Truth \\ Pred": r_name}
            for c_name in cols:
                row_dict[c_name] = report.confusion_matrix.matrix.get(r_name, {}).get(c_name, 0)
            matrix_data.append(row_dict)
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)

    # 6. Clean Negative Controls & False-Positive Rate
    if report.controls:
        ctrl = report.controls
        st.markdown("### Clean Identity Controls (False-Alarm Evaluation)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Controls Evaluated", ctrl.clean_controls_evaluated)
        with c2:
            st.metric("Controls with Predictions", ctrl.controls_with_false_positives)
        with c3:
            st.metric("Spurious Predictions", ctrl.total_false_positive_events)
        with c4:
            fp_rate_str = (
                f"{ctrl.false_positive_rate_per_control:.3f}"
                if ctrl.false_positive_rate_per_control is not None
                else "N/A"
            )
            st.metric("False Alarm Rate", fp_rate_str)
