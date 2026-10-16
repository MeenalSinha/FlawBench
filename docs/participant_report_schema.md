# Participant report schema (Phase 6)

Reports are written only when the user requests them (or `explanation.save_reports_automatically` is
true) to `explanation.reports_dir` (default `outputs/participant_reports`), confined with
`resolve_within`. The JSON file is the canonical saved analysis; Markdown, HTML and CSV are rendered
from that JSON after it is read back from disk. Export never reruns detection.

## JSON (`schema_version` 6.0)
Top level: `session_id`, `created_utc`, `mode`, `status` (`completed`, `completed_no_candidates`,
`partial`, `failed`), `failure`, `reference` and `participant` (label, origin, sha256, sample rate,
channels, duration, container, subtype), `stages`, `supported_categories`, `categories_completed`,
`categories_failed`, `prediction_count`, `categories_detected`, `sort_policy`, `cards`, `warnings`,
`limitations`, `transcript`, `alignment`, `detection` (run id, detector version, config
fingerprint, prediction ids), `provenance` (FlawBench version, config fingerprint, feature artifact
ids, software versions) and `benchmark` (aggregate, separate, may be null).

Each card holds the five text sections, `numeric_claims`, `missing_facts`, `status` and an
`evidence_bundle` with `variant_interval` (half-open samples plus seconds), `reference_correspondence`,
`facts` (value, unit, role, source, definition, validity), evidence strength with its meaning,
detector notes and provenance.

## Other formats
Markdown and HTML (HTML-escaped, self-contained) include summary, stages, scope, cards with measured
values, transcript check, alignment, warnings, limitations, benchmark context and provenance. CSV has
one row per candidate (`CSV_COLUMNS` in `explainability/reports.py`).

## Privacy
Raw participant audio is never embedded; reports hold metadata, checksums and measurements. Uploads
are processed in memory and stored only by the explicit, rights-gated registration action, which
copies the file into the dataset registry. Failed and partial analyses are labeled as such.
