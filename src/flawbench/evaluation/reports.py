"""Markdown and human-readable report generation for FlawBench Phase 5."""

from __future__ import annotations

from flawbench.evaluation.schema import EvaluationReport


def generate_markdown_report(report: EvaluationReport) -> str:
    """Generate a structured Markdown report from an EvaluationReport."""
    prov_str = (
        "WARNING: Synthetic Signals Only"
        if report.is_synthetic_only
        else (
            "Audio origin is taken from registration metadata only and is not independently "
            "verified. The registry has no synthetic source type, so synthetic fixtures are NOT "
            "identified by this field; treat results as unvalidated for natural speech unless "
            "annotated, licensed speech was used."
        )
    )
    lines: list[str] = [
        f"# FlawBench Evaluation Report: `{report.run_id}`",
        "",
        f"- **Generated At**: {report.created_at}",
        f"- **Detector Version**: {report.detector_version}",
        f"- **Detector Config Fingerprint**: `{report.detector_config_fingerprint}`",
        f"- **FlawBench Version**: {report.metadata.get('flawbench_version', 'unknown')}",
        f"- **Evaluation Split**: `{report.split}`",
        f"- **Cohort Size**: {report.sample_count} samples across {report.source_count} sources",
        f"- **Audio Provenance**: {prov_str}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
    ]

    if report.is_synthetic_only:
        lines.extend(
            [
                "> [!WARNING]",
                "> **Synthetic Audio Limitation**: The evaluated cohort consists solely of "
                "synthetic signal fixtures. Metrics reflect algorithmic verification on "
                "controlled synthetic waveforms and must not be interpreted as verified "
                "generalization to naturally spoken speech.",
                "",
            ]
        )

    agg = report.aggregate
    p_str = f"{agg.micro_precision:.3f}" if agg.micro_precision is not None else "N/A"
    r_str = f"{agg.micro_recall:.3f}" if agg.micro_recall is not None else "N/A"
    f1_str = f"{agg.micro_f1:.3f}" if agg.micro_f1 is not None else "N/A"
    macro_p = f"{agg.macro_precision:.3f}" if agg.macro_precision is not None else "N/A"
    macro_r = f"{agg.macro_recall:.3f}" if agg.macro_recall is not None else "N/A"
    macro_f1_str = f"{agg.macro_f1:.3f}" if agg.macro_f1 is not None else "N/A"
    iou_str = f"{agg.mean_iou:.3f}" if agg.mean_iou is not None else "N/A"

    lines.extend(
        [
            "| Metric | Micro-Averaged Value | Macro-Averaged Value |",
            "| :--- | :---: | :---: |",
            f"| **Precision** | {p_str} | {macro_p} |",
            f"| **Recall** | {r_str} | {macro_r} |",
            f"| **F1-Score** | {f1_str} | {macro_f1_str} |",
            f"| **Mean Matched IoU** | {iou_str} | - |",
            "",
        ]
    )

    lines.extend(
        [
            "## 2. Per-Category Classification Performance",
            "",
            "| Category | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Mean IoU |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for cat_name, m in sorted(report.per_category.items()):
        p_val = f"{m.precision:.3f}" if m.precision is not None else "N/A"
        r_val = f"{m.recall:.3f}" if m.recall is not None else "N/A"
        f_val = f"{m.f1:.3f}" if m.f1 is not None else "N/A"
        iou_val = f"{m.mean_iou:.3f}" if m.mean_iou is not None else "N/A"
        lines.append(
            f"| `{cat_name}` | {m.gt_count} | {m.pred_count} | "
            f"{m.tp} | {m.fp} | {m.fn} | "
            f"{p_val} | {r_val} | {f_val} | {iou_val} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 3. Temporal Localization & Boundary Errors",
            "",
            "| Category | Mean Start Err (s) | Mean End Err (s) | Mean Dur Err (s) |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )

    for cat_name, m in sorted(report.per_category.items()):
        mse = (
            f"{m.mean_abs_start_error_sec:.4f}" if m.mean_abs_start_error_sec is not None else "N/A"
        )
        mee = f"{m.mean_abs_end_error_sec:.4f}" if m.mean_abs_end_error_sec is not None else "N/A"
        mde = f"{m.mean_abs_dur_error_sec:.4f}" if m.mean_abs_dur_error_sec is not None else "N/A"
        lines.append(f"| `{cat_name}` | {mse} | {mee} | {mde} |")
    lines.append("")

    lines.extend(
        [
            "## 4. Localization-Tolerance Curve",
            "",
            "Shows detection recall across varying temporal IoU overlap thresholds:",
            "",
            "| IoU Threshold | Total Ground Truth | Matched Events | Recall |",
            "| :---: | :---: | :---: | :---: |",
        ]
    )
    for pt in report.localization_curve:
        rec_str = f"{pt.recall:.3f}" if pt.recall is not None else "N/A"
        lines.append(f"| {pt.iou_threshold:.2f} | {pt.gt_count} | {pt.matched_count} | {rec_str} |")
    lines.append("")

    if report.confusion_matrix is not None:
        lines.extend(
            [
                "## 5. Event-Level Confusion Matrix",
                "",
                "Rows represent Ground Truth categories (or `'none'` for spurious false alarms).",
                "Columns represent Predicted categories (or `'missed'` for undetected events).",
                "",
            ]
        )
        cats = report.confusion_matrix.categories
        cols = cats + ["missed"]
        header = "| GT \\ Pred | " + " | ".join(f"`{c}`" for c in cols) + " |"
        sep = "| :--- | " + " | ".join(":---:" for _ in cols) + " |"
        lines.append(header)
        lines.append(sep)

        for gt in cats + ["none"]:
            row_vals = [
                str(report.confusion_matrix.matrix.get(gt, {}).get(pred, 0)) for pred in cols
            ]
            lines.append(f"| `{gt}` | " + " | ".join(row_vals) + " |")
        lines.append("")

    if report.controls is not None:
        ctrl = report.controls
        fp_rate_str = (
            f"{ctrl.false_positive_rate_per_control:.3f}"
            if ctrl.false_positive_rate_per_control is not None
            else "N/A"
        )
        lines.extend(
            [
                "## 6. Clean Negative Controls Analysis",
                "",
                f"- **Evaluated Identity Controls**: {ctrl.clean_controls_evaluated}",
                f"- **Controls Triggering Predictions**: {ctrl.controls_with_false_positives}",
                f"- **Total Spurious Predictions**: {ctrl.total_false_positive_events}",
                f"- **False Alarm Rate per Clean Control**: {fp_rate_str}",
                "",
            ]
        )

    if report.bootstrap_ci is not None:
        lines.extend(
            [
                "## 7. Source-Level Bootstrap Confidence Intervals (95% CI)",
                "",
                f"- **Status**: {report.bootstrap_ci.get('status', 'completed')}",
                f"- **Replicates**: {report.bootstrap_ci.get('replicates', 'N/A')}",
                f"- **Source Clusters**: {report.bootstrap_ci.get('source_count', 'N/A')}",
                "",
            ]
        )
        if "metrics" in report.bootstrap_ci:
            lines.extend(
                [
                    "| Metric | Point Estimate | 95% Confidence Interval |",
                    "| :--- | :---: | :---: |",
                ]
            )
            for metric_k in [
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "macro_f1",
                "mean_iou",
            ]:
                m_info = report.bootstrap_ci["metrics"].get(metric_k, {})
                lo = m_info.get("ci_lower_2_5")
                hi = m_info.get("ci_upper_97_5")
                ci_str = f"[{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else "N/A"
                val = getattr(report.aggregate, metric_k, None)
                v_str = f"{val:.3f}" if val is not None else "N/A"
                lines.append(f"| `{metric_k}` | {v_str} | {ci_str} |")
            lines.append("")

    lines.extend(
        [
            "---",
            "*Report generated automatically by FlawBench Evaluation Engine (Phase 5).*",
        ]
    )

    return "\n".join(lines)
