# Dataset schema and timeline convention (schema_version 1.0)

## Files (names are configurable; all live in `storage.metadata_dir`)

| File | Content |
|---|---|
| `sources.jsonl` | one source record per line (registered reference recordings) |
| `manifest.jsonl` | one sample record per line (generated variants); a line exists only after the audio was verified and committed |
| `runs.jsonl` | one record per batch run: plan fingerprint, outcome per variant (including failures), provenance |
| `splits.json` | persisted source-level split assignments |

Generated audio: `storage.generated_dir/<dataset_id>/<sample_id>.wav`. The file name carries no flaw
information, and the WAV contains no annotation (no ground-truth leakage into the audio artifact).

## Timeline convention

* Every interval is half-open `[start, end)` and relative to the start of the recording it refers to.
* Integer sample indices (`start_sample`, `end_sample`) are authoritative; `start_sec = start_sample / sample_rate`.
* `output_interval` is on the **generated** recording's timeline. `source_interval` (when present) is on the
  **reference** timeline. They are stored separately and are never reused for each other.
* Explicit placement seconds are converted with `round(sec * sample_rate)`.

| Flaw | `source_interval` | `source_anchor` | `output_interval` |
|---|---|---|---|
| `pause_insertion` | null | sample/sec where silence was inserted | `[anchor, anchor + silence_samples)` |
| `tempo_change` | the stretched reference span | null | `[start, start + measured_segment_samples)` (measured from the produced signal) |
| `energy_reduction` | the attenuated span | null | identical to `source_interval` |

Audio after a pause or tempo change is shifted; its reference position is `output - shift`.

## Source record (`sources.jsonl`)

`schema_version, record_type="source", source_id, original_filename, audio_path` (relative to
`reference_dir`, POSIX), `transcript_text, transcript_file, sha256, sample_rate, channels, frames,
duration_seconds, container_format, subtype, source_type` (`self_recorded | public_domain | licensed |
other_permissioned), `attribution, license, rights` (`use_confirmed` must be true; `redistribution` is
`permitted | prohibited | unknown`; `cleared_for_public_redistribution` is true only for `permitted`),
`notes, registered_utc, flawbench_version`.

`source_id` defaults to `src-<first 12 hex of sha256>` unless an explicit id is given.

## Sample record (`manifest.jsonl`)

| Field | Meaning |
|---|---|
| `schema_version`, `record_type="sample"`, `status="committed"` | only committed records are written |
| `dataset_id`, `sample_id` | `sample_id = smp-<16 hex>` of a fingerprint of dataset id, source checksum, flaw, severity, parameters, resolved placement (integer samples), output subtype |
| `source_id`, `reference{path, sha256, frames, duration_seconds}` | the unchanged reference |
| `generated{path, sha256, frames, duration_seconds, sample_rate, channels, subtype}` | the produced file (path relative to `generated_dir`) |
| `transcript_ref{source_id, sha256}` | transcript lives in the source record |
| `annotation` | `flaw_type, severity` (profile id or `custom`), `base_severity`, `parameters` (validated inputs), `effective_parameters` (what actually happened), `source_interval`, `source_anchor`, `output_interval`, `timeline` |
| `seed` | `null` for deterministic operations; otherwise `{run_seed, sample_seed, used_for:"placement"}` |
| `generated_utc`, `software{flawbench_version, python, platform, libsndfile, packages{numpy, soundfile, audiotsm}}` | provenance |
| `config_fingerprint` | SHA-256 of the `dataset` configuration section (portable; excludes machine paths) |
| `plan_fingerprint` | SHA-256 of the canonical plan (null for single-variant generation) |
| `transformation{implementation, placement}` | implementation name/version; resolved integer placement |
| `history` | entries added when a sample is explicitly regenerated (previous checksum, config fingerprint, software) |

`effective_parameters` per flaw: pause `{silence_samples}`; tempo `{requested_rate, source_segment_samples,
output_segment_samples, effective_rate, output_peak_abs}`; energy `{requested_gain_db, amplitude_multiplier,
applied_gain_db}`.

Adding a flaw type means: a new entry in `PARAM_SPECS`, an injector returning an `InjectionResult`, and
profile/bounds configuration. The record shape does not change.

## Severity

A severity id names a configured *transformation strength* (an experimental manipulation). It is not a
claim about how a human listener would rate the recording. Overriding any profile value makes the severity
`custom` and records the profile it started from in `base_severity`.

## Randomness and reproducibility policy

1. Explicit positions win. Random placement requires an explicit seed; there is no silent default.
2. Per-sample seed = first 8 bytes of `sha256("<run_seed>|<variant_key>")`, used for a dedicated
   `numpy.random.default_rng` (never global state). Variant key = `source:flaw:severity:replicate`.
3. A seed is recorded only when it influenced placement.
4. Same source bytes, `dataset` configuration, plan and seed reproduce the same sample ids and, in the tested
   environment, byte-identical WAV files (the non-deterministic libsndfile PEAK chunk is disabled). Identity does
   not include library versions: a different `audiotsm`/`numpy` version may change tempo output; such a mismatch
   is detected (checksum differs for an existing id) and requires explicit overwrite. Cross-platform bit-exactness
   is not guaranteed.
5. Random positions are uniform within configured margins; they are **not** aligned to speech boundaries
   (no speech-activity detector exists yet).

## Validation findings

Errors invalidate the dataset (checksum mismatch, missing files, bad intervals, annotation that contradicts the
audio, duplicates, unsafe paths, missing rights or transcript, malformed lines, non-committed records). Warnings do
not (unknown/prohibited redistribution, orphan or interrupted-generation files, duplicate source audio, split gaps).
Deep mode re-derives ground truth from the audio: reference prefix/suffix must be bit-equal around the annotation,
pauses must be silent, attenuated spans must equal the reference times the recorded multiplier.
