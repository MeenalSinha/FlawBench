# ruff: noqa: E501  (long report-text lines are clearer unwrapped)
"""Participant-analysis report export (JSON, Markdown, HTML, CSV).

The JSON file is the canonical saved analysis. Every other format is rendered from that JSON
(read back from disk), never from a fresh analysis, so exports cannot drift from the saved
result. Raw participant audio is never embedded.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
from pathlib import Path
from typing import Any

from flawbench.config import ExplanationSettings
from flawbench.dataset.io import atomic_write_text
from flawbench.exceptions import ReportError, StorageError
from flawbench.explainability.schema import ParticipantAnalysis
from flawbench.paths import resolve_within

EXTENSIONS = {"json": ".json", "markdown": ".md", "html": ".html", "csv": ".csv"}
_STATUS_TEXT = {
    "completed": "Completed. Candidates were found.",
    "completed_no_candidates": "Completed. No candidate met the detector's decision rules "
    "(this does not show that the delivery is free of problems).",
    "partial": "PARTIAL ANALYSIS. At least one detector failed.",
    "failed": "FAILED. No analysis result is available.",
}
CSV_COLUMNS = [
    "session_id", "prediction_id", "flaw_type", "start_sec", "end_sec", "duration_sec",
    "start_sample", "end_sample", "sample_rate", "reference_status", "reference_start_sec",
    "reference_end_sec", "evidence_strength", "card_status",
]  # fmt: skip


def _fmt(v: Any) -> str:
    return "unavailable" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _md(text: Any) -> str:
    """Neutralise untrusted text for Markdown: no inline HTML, no code-span breakout."""
    return html.escape(str(text), quote=True).replace("`", "&#96;")


def render_markdown(d: dict[str, Any]) -> str:
    out = ["# FlawBench participant analysis report", ""]
    out += [
        f"- Session: `{d['session_id']}`",
        f"- Created (UTC): {d['created_utc']}",
        f"- Mode: {d['mode']}",
        f"- Status: **{d['status']}**. {_STATUS_TEXT.get(d['status'], '')}",
    ]
    if d.get("failure"):
        out.append(f"- Failure: {_md(d['failure'])}")
    for key, title in (("reference", "Reference"), ("participant", "Participant")):
        r = d.get(key)
        if r:
            out.append(
                f"- {title}: {_md(r['label'])} ({r['origin']}), {r['duration_seconds']:.2f} s, "
                f"{r['sample_rate']} Hz, {r['channels']} ch, sha256 `{r['sha256']}`"
            )
    out += ["", "## Analysis stages", ""]
    out += [f"- {s['name']}: {s['status']}" + (f" ({_md(s['message'])})" if s["message"] else "")
            for s in d["stages"]]  # fmt: skip
    out += ["", "## Scope", ""]
    out.append("Supported categories: " + ", ".join(d["supported_categories"]) + ".")
    out.append("Completed categories: " + (", ".join(d["categories_completed"]) or "none") + ".")
    if d["categories_failed"]:
        out += [f"- FAILED {k}: {_md(v)}" for k, v in d["categories_failed"].items()]
    out += ["", f"## Detected candidates ({d['prediction_count']})", ""]
    if not d["cards"]:
        out.append("No candidates are reported for this analysis.")
    for i, c in enumerate(d["cards"], 1):
        b = c["evidence_bundle"]
        v, ref = b["variant_interval"], b["reference_correspondence"]
        out += [
            f"### {i}. {c['title']} ({c['flaw_type']})",
            "",
            f"- Prediction id: `{c['prediction_id']}`",
            f"- Participant interval: {v['start_sec']:.2f} s to {v['end_sec']:.2f} s "
            f"(samples [{v['start_sample']}, {v['end_sample']}) at {v['sample_rate']} Hz)",
            f"- Reference correspondence: {ref['status']}"
            + (
                f" ({_fmt(ref['start_sec'])} to {_fmt(ref['end_sec'])} s)"
                if ref["start_sec"] is not None
                else ""
            ),
            f"- Detector evidence strength: {b['evidence_strength']:.3f} (uncalibrated heuristic, not a probability)",
            f"- Explanation status: {c['status']}",
            "",
            f"**Observation.** {c['observation']}",
            "",
            "**Evidence.**",
            *[f"- {e}" for e in c["evidence"]],
            "",
            f"**Interpretation.** {c['interpretation']}",
            "",
            f"**Suggested review.** {c['suggested_review']}",
            "",
            "**Limitations.**",
            *[f"- {x}" for x in c["limitations"]],
        ]
        if c["warnings"] or c["detector_notes"]:
            out += ["", "**Warnings.**", *[f"- {w}" for w in c["warnings"]],
                    *[f"- Detector note: {w}" for w in c["detector_notes"]]]  # fmt: skip
        if c["missing_facts"]:
            out += ["", "Unavailable measurements: " + ", ".join(c["missing_facts"]) + "."]
        out += [
            "",
            "Measured values:",
            "",
            "| Measurement | Value | Unit | Source |",
            "|---|---|---|---|",
        ]
        for f in b["facts"]:
            out.append(
                f"| {f['label']} | {_fmt(f['value']) if f['valid'] else 'unavailable'} | {f['unit']} | {f['source']} |"
            )
        out.append("")
    if d.get("transcript"):
        t = d["transcript"]
        out += ["## Transcript check", "", t["note"] or "",
                f"- Performed: {t['performed']}", f"- Similarity: {_fmt(t['similarity'])}"]  # fmt: skip
    out += ["", "## Alignment context", ""]
    out += [f"- {k}: {_fmt(v)}" for k, v in d["alignment"].items()]
    if d["warnings"]:
        out += ["", "## Warnings", "", *[f"- {w}" for w in d["warnings"]]]
    out += ["", "## Limitations", "", *[f"- {x}" for x in d["limitations"]]]
    bm = d.get("benchmark")
    out += ["", "## Benchmark context (aggregate, separate from this analysis)", ""]
    if bm and bm["available"]:
        out.append(
            f"Run `{bm['run_id']}`, split {bm['split']}, {bm['sample_count']} samples. {bm['note']}"
        )
        if bm["is_synthetic_only"]:
            out.append("WARNING: this benchmark uses synthetic fixture audio only.")
        for cat, m in bm["per_category"].items():
            out.append(
                f"- {cat}: precision {_fmt(m['precision'])}, recall {_fmt(m['recall'])}, F1 {_fmt(m['f1'])}"
            )
    else:
        out.append("Benchmark results are unavailable.")
    out += [
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(
            {"detection": d["detection"], "provenance": d["provenance"]}, indent=2, sort_keys=True
        ),
        "```",
        "",
    ]
    return "\n".join(out)


def render_html(d: dict[str, Any]) -> str:
    md = render_markdown(d)
    body = html.escape(md)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>FlawBench report {html.escape(d['session_id'])}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;"
        "padding:0 1rem;line-height:1.5;color:#1f2933}pre{white-space:pre-wrap}</style></head>"
        f"<body><pre>{body}</pre></body></html>"
    )


def render_csv(d: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    for c in d["cards"]:
        b = c["evidence_bundle"]
        v, r = b["variant_interval"], b["reference_correspondence"]
        w.writerow({
            "session_id": d["session_id"], "prediction_id": c["prediction_id"],
            "flaw_type": c["flaw_type"], "start_sec": v["start_sec"], "end_sec": v["end_sec"],
            "duration_sec": v["duration_sec"], "start_sample": v["start_sample"],
            "end_sample": v["end_sample"], "sample_rate": v["sample_rate"],
            "reference_status": r["status"], "reference_start_sec": "" if r["start_sec"] is None else r["start_sec"],
            "reference_end_sec": "" if r["end_sec"] is None else r["end_sec"],
            "evidence_strength": b["evidence_strength"], "card_status": c["status"],
        })  # fmt: skip
    return buf.getvalue()


_RENDERERS = {"markdown": render_markdown, "html": render_html, "csv": render_csv}


def export_report(
    analysis: ParticipantAnalysis,
    settings: ExplanationSettings,
    *,
    formats: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Write the canonical JSON, then render the requested formats from the saved file."""
    wanted = tuple(formats) if formats is not None else settings.export_formats
    unknown = [f for f in wanted if f not in EXTENSIONS]
    disabled = [f for f in wanted if f in EXTENSIONS and f not in settings.export_formats]
    if unknown or disabled:
        raise ReportError(
            "Unsupported or disabled export format.", detail=", ".join(unknown + disabled)
        )
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", analysis.session_id)
    try:
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        json_path = resolve_within(settings.reports_dir, stem + ".json")
        atomic_write_text(json_path, json.dumps(analysis.to_dict(), indent=2, sort_keys=True))
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        written: dict[str, Path] = {"json": json_path}
        for fmt in wanted:
            if fmt == "json":
                continue
            path = resolve_within(settings.reports_dir, stem + EXTENSIONS[fmt])
            atomic_write_text(path, _RENDERERS[fmt](saved))
            written[fmt] = path
    except (OSError, StorageError) as exc:
        raise ReportError("The report could not be written.", detail=str(exc)) from exc
    return written
