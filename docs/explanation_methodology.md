# Explanation methodology (Phase 6)

## Evidence sources
Explanations read only: a `DetectionResult` that passed `DetectionValidator`, the Phase 3
comparison's `normalized_dtw_cost`, the lengths of the two recordings, and an optional text
comparison of transcripts. Ground-truth annotations are never read (enforced by a source-level test).

## Pipeline
`DetectionResult` -> `EvidenceBundle` (validated facts) -> template -> `ExplanationCard` -> validator.
An invalid detection result raises `ExplanationInputError`; no success-style card is produced.

## Evidence fields and meaning (taken from the Phase 4 implementation)
| Category | Fields used | Notes |
|---|---|---|
| pause_insertion | `variant_pause_duration_sec`, `reference_duration_sec`, `duration_difference_sec`, `mean_variant_rms_db`, `threshold_silence_db`, `min_duration_sec`, `ref_frozen_fraction` | A zero-length reference span is an anchor point, not an interval |
| tempo_change | `estimated_relative_rate` (reference time per participant time; below one is slower), `mean_warp_slope` (its reciprocal), `rate_deviation`, `threshold_min_deviation`, `duration_sec`, `direction` | A direction label that contradicts the rate is dropped |
| energy_reduction | `mean_delta_rms_db` (participant minus reference), `min_delta_rms_db`, `mean_ref_rms_db`, `mean_var_rms_db`, `threshold_gain_db`, `duration_sec` | Negative means the participant is quieter |

Unknown evidence keys are listed as ignored, never interpreted. Non-finite or wrong-typed values and
values that contradict each other (for example a difference that does not equal its components)
become unavailable facts and no number is printed for them.

## Reference correspondence
It comes from the detector's DTW alignment, not from equal timestamps. Statuses: `interval`,
`anchor`, `unavailable` (missing, invalid, past the end of the reference, or degenerate for a
non-pause event). Participant intervals are half-open sample ranges converted with the participant's
own sample rate.

## Wording rules
Template text contains no digits; every number is rendered by one formatter from a fact and recorded
as a `NumericClaim`. The validator rejects any number not matching a claim, any claim that does not
re-render from a valid fact, percentages, and a fixed list of causal or psychological wording.
Evidence strength is always described as an uncalibrated heuristic score from zero to one.
If verification fails, a facts-only fallback card is used and flagged.

## Why acoustic differences do not establish causes
The system measures differences from a reference under configured rules. It cannot know why a
difference occurred, whether a listener would object, or whether the reference is the right
target. Cards therefore say "candidate", describe measurements, suggest what to review, and list
limitations. No overall quality score exists.

## Alignment uncertainty
Direction wording for tempo is withheld when the recordings differ in length by more than a
configured factor, when a supplied transcript differs noticeably from the reference transcript, or
when the rate is unavailable. The DTW cost is reported as a measurement; no calibrated reliability
threshold exists.

## Merged predictions
Post-processing may merge adjacent detector regions. Numeric evidence then describes the first
region only; the card says so.

## Language model
Not implemented. The deterministic layer is the only explanation path and needs no external service.
