"""Deterministic explanation templates.

Templates are domain rules, not results: every number in the output is rendered from a validated
:class:`EvidenceFact` through :class:`Renderer`, which records it as a :class:`NumericClaim`.
Template text itself contains no digits, so the validator can prove that every number in a card
traces to a fact. Wording is deliberately restricted to what the measurements show: acoustic
differences from a reference under a configured rule, never inferred states or audience effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from flawbench.explainability.facts import AlignmentContext
from flawbench.explainability.schema import (
    REF_ANCHOR,
    REF_INTERVAL,
    STRENGTH_MEANING,
    EvidenceBundle,
    NumericClaim,
)

CATEGORY_LABELS = {
    "pause_insertion": "Pause",
    "tempo_change": "Tempo",
    "energy_reduction": "Energy reduction",
}


def format_value(value: float, unit: str) -> str:
    """Single formatting authority; the validator re-applies it to verify every claim."""
    if unit == "s":
        return f"{value:.2f} s"
    if unit == "dB":
        return f"{value:.2f} dB"
    return f"{value:.3f}"


class Renderer:
    """Renders facts into prose while recording each rendered number as a claim."""

    def __init__(self, bundle: EvidenceBundle) -> None:
        self.bundle = bundle
        self.claims: list[NumericClaim] = []
        self.missing: list[str] = []

    def num(self, key: str) -> str | None:
        fact = self.bundle.fact(key)
        if fact is None or not fact.valid or fact.value is None:
            if key not in self.missing:
                self.missing.append(key)
            return None
        text = format_value(fact.value, fact.unit)
        if not any(c.fact_key == key and c.rendered == text for c in self.claims):
            self.claims.append(NumericClaim(key, text, fact.value))
        return text

    def sentence(self, template: str, **placeholders: str) -> str | None:
        """Fill ``template`` from fact keys; return None when any required fact is unusable."""
        values: dict[str, str] = {}
        for name, key in placeholders.items():
            rendered = self.num(key)
            if rendered is None:
                return None
            values[name] = rendered
        return template.format(**values)


@dataclass(frozen=True)
class CardText:
    title: str
    observation: str
    evidence_lines: list[str]
    interpretation: str
    suggested_review: str
    limitations: list[str]
    warnings: list[str]
    claims: list[NumericClaim]
    missing: list[str]


def _interval_phrase(r: Renderer) -> tuple[str, str, str]:
    start = r.num("variant_start_sec") or ""
    end = r.num("variant_end_sec") or ""
    dur = r.num("variant_duration_sec") or ""
    return start, end, dur


def _reference_line(r: Renderer) -> str:
    ref = r.bundle.reference
    if ref.status == REF_ANCHOR:
        line = r.sentence(
            "In the reference recording this position maps to a single anchor point near "
            "{ref}; the aligned reference span has no length there.",
            ref="reference_start_sec",
        )
        return line or "A reference anchor point exists but its time was unavailable."
    if ref.status == REF_INTERVAL:
        line = r.sentence(
            "The DTW comparison matches this region to {a} through {b} in the reference recording.",
            a="reference_start_sec",
            b="reference_end_sec",
        )
        return line or "A reference interval exists but its times were unavailable."
    return (
        "No reliable reference correspondence is available for this interval, so no "
        "reference position is reported."
    )


def _strength_line(r: Renderer) -> str:
    return (
        r.sentence(
            "Detector evidence strength was {s} on a scale from zero to one. It is an "
            "uncalibrated heuristic score, not a calibrated probability.",
            s="evidence_strength",
        )
        or STRENGTH_MEANING
    )


def _common_warnings(r: Renderer, context: AlignmentContext) -> list[str]:
    out: list[str] = []
    if context.length_mismatch:
        line = r.sentence(
            "The longer recording is {ratio} times the length of the shorter one, so the "
            "alignment may not link equivalent speech.",
            ratio="recording_duration_ratio",
        )
        out.append(
            line
            or "The recordings differ substantially in length, so the alignment may not link "
            "equivalent speech."
        )
    if context.transcript_below_threshold:
        out.append(
            "The supplied transcript differs noticeably from the reference transcript, so the "
            "recordings may not contain corresponding speech."
        )
    if r.bundle.was_merged:
        out.append(
            "This candidate combines several adjacent detector regions. The interval covers "
            "all of them, but the numerical evidence describes the first region only."
        )
    out.extend(r.bundle.alignment_warnings)
    return out


def _definitions(r: Renderer) -> list[str]:
    lines: list[str] = []
    for f in r.bundle.facts:
        if f.valid and f.key not in {"variant_start_sec", "variant_end_sec"}:
            lines.append(f"Definition, {f.label}: {f.definition}")
    return lines


def _apply_verbosity(
    evidence: list[str], limitations: list[str], definitions: list[str], verbosity: str
) -> tuple[list[str], list[str]]:
    if verbosity == "concise":
        return evidence[:2], limitations[:2]
    if verbosity == "detailed":
        return evidence + definitions, limitations
    return evidence, limitations


def _pause(r: Renderer, context: AlignmentContext, verbosity: str) -> CardText:
    start, end, dur = _interval_phrase(r)
    observation = (
        f"The detector flagged a pause-related deviation from {start} to {end} on the "
        f"participant timeline (length {dur})."
    )
    evidence: list[str] = []
    for tmpl, kw in (
        (
            "Measured pause duration in the participant recording: {v}.",
            {"v": "variant_pause_duration_sec"},
        ),
        (
            "Duration difference (participant minus aligned reference): {d}.",
            {"d": "duration_difference_sec"},
        ),
        (
            "Mean participant RMS level over the region was {m}, against a silence threshold "
            "of {t}.",
            {"m": "mean_variant_rms_db", "t": "threshold_silence_db"},
        ),
    ):
        line = r.sentence(tmpl, **kw)
        if line:
            evidence.append(line)
    evidence.insert(1, _reference_line(r))
    detailed_extra = r.sentence(
        "Reference position did not advance in a fraction {f} of the alignment steps.",
        f="ref_frozen_fraction",
    )
    if detailed_extra:
        evidence.append(detailed_extra)
    evidence.append(_strength_line(r))
    minimum = r.sentence(
        "The measured pause satisfies the configured pause-deviation rule, which requires a "
        "pause of at least {m} that is also longer than the aligned reference.",
        m="min_duration_sec",
    )
    interpretation = (
        minimum
        or "The measurements satisfy the configured pause-deviation rule, although the "
        "configured minimum duration was unavailable."
    )
    limitations = [
        "Silence is estimated from a frame-level RMS threshold, so background noise can shorten "
        "or hide a pause.",
        "The reference position comes from a DTW alignment and is an estimate.",
        "A longer pause is a measured acoustic difference; it does not show that the pause was "
        "unintended or incorrect.",
    ]
    ev, lim = _apply_verbosity(evidence, limitations, _definitions(r), verbosity)
    return CardText(
        "Pause candidate",
        observation,
        ev,
        interpretation,
        "Listen to the participant excerpt around the flagged interval and compare it with the "
        "reference excerpt at the matching position. Review the transition into and out of the "
        "pause to decide whether it was intended.",
        lim,
        _common_warnings(r, context),
        r.claims,
        r.missing,
    )


def _tempo(r: Renderer, context: AlignmentContext, verbosity: str) -> CardText:
    start, end, dur = _interval_phrase(r)
    direction = r.bundle.categorical.get("direction")
    rate_ok = r.bundle.valid_value("estimated_relative_rate") is not None
    uncertain = context.uncertain or not rate_ok
    if direction in {"slower", "faster"} and not uncertain:
        phrase = f"{direction} than"
    else:
        phrase = "different from"
    observation = (
        f"Over {start} to {end} on the participant timeline (length {dur}), the temporal "
        f"comparison shows progression that is {phrase} the reference."
    )
    evidence: list[str] = []
    for tmpl, kw in (
        (
            "The estimated relative rate was {r} (below one means slower than the reference, "
            "above one means faster).",
            {"r": "estimated_relative_rate"},
        ),
        (
            "The mean warp slope was {s}, meaning participant time elapsed per unit of "
            "reference time.",
            {"s": "mean_warp_slope"},
        ),
        ("The detector measured this over a region of {d}.", {"d": "duration_sec"}),
    ):
        line = r.sentence(tmpl, **kw)
        if line:
            evidence.append(line)
    evidence.insert(min(1, len(evidence)), _reference_line(r))
    evidence.append(_strength_line(r))
    interp = r.sentence(
        "A deviation of {dev} from equal pace meets the configured minimum deviation of {thr}, "
        "which matches the tempo-deviation rule.",
        dev="rate_deviation",
        thr="threshold_min_deviation",
    )
    interpretation = (
        interp
        or "The measurements meet the configured tempo-deviation rule, although the deviation "
        "values were unavailable."
    )
    warnings = _common_warnings(r, context)
    if "direction_rejected" in r.bundle.categorical:
        warnings.append(
            "The detector's direction label contradicted its own rate measurement, so no "
            "direction is stated."
        )
    if uncertain:
        warnings.append(
            "Because the alignment or the rate measurement is uncertain, the direction and size "
            "of the change are uncertain."
        )
    limitations = [
        "The rate is derived from a DTW alignment of acoustic features; local alignment errors "
        "can exaggerate or hide a rate change.",
        "No word-level timing is available, so the interval cannot be tied to specific words.",
        "A different pace from the reference is a measured difference, not by itself an error.",
    ]
    ev, lim = _apply_verbosity(evidence, limitations, _definitions(r), verbosity)
    return CardText(
        "Tempo candidate",
        observation,
        ev,
        interpretation,
        "Listen to the participant and reference excerpts one after the other and compare the "
        "pacing of the flagged passage with the reference.",
        lim,
        warnings,
        r.claims,
        r.missing,
    )


def _energy(r: Renderer, context: AlignmentContext, verbosity: str) -> CardText:
    start, end, dur = _interval_phrase(r)
    delta = r.bundle.valid_value("mean_delta_rms_db")
    lower = delta is not None and delta < 0
    comparison = (
        "lower measured signal energy than" if lower else "measured signal energy that differs from"
    )
    observation = (
        f"From {start} to {end} on the participant timeline (length {dur}), the participant "
        f"recording shows {comparison} the aligned reference region."
    )
    evidence: list[str] = []
    for tmpl, kw in (
        (
            "The mean RMS level difference (participant minus reference) was {m}; the largest "
            "single-frame reduction was {x}.",
            {"m": "mean_delta_rms_db", "x": "min_delta_rms_db"},
        ),
        (
            "Mean RMS level was {r} in the aligned reference region and {v} in the participant "
            "region.",
            {"r": "mean_ref_rms_db", "v": "mean_var_rms_db"},
        ),
        ("The configured reduction threshold is {t}.", {"t": "threshold_gain_db"}),
    ):
        line = r.sentence(tmpl, **kw)
        if line:
            evidence.append(line)
    evidence.insert(min(1, len(evidence)), _reference_line(r))
    evidence.append(_strength_line(r))
    interpretation = (
        "A sustained RMS reduction at or beyond the configured threshold matches the "
        "energy-reduction rule."
    )
    warnings = _common_warnings(r, context)
    if delta is not None and not lower:
        warnings.append(
            "The mean level difference was not negative, which is unexpected for an "
            "energy-reduction candidate; treat this candidate with caution."
        )
    limitations = [
        "Signal energy is a measurable audio property that depends on recording level, "
        "microphone distance and gain; it is not a direct measure of confidence, clarity or "
        "rhetorical effectiveness.",
        "The reference region comes from a DTW alignment and is an estimate.",
        "A lower level than the reference is a measured difference and may reflect the "
        "recording setup rather than the delivery.",
    ]
    ev, lim = _apply_verbosity(evidence, limitations, _definitions(r), verbosity)
    return CardText(
        "Energy-reduction candidate",
        observation,
        ev,
        interpretation,
        "Listen to the participant and reference excerpts at the same level and check both "
        "the recording setup and the delivery around the flagged interval.",
        lim,
        warnings,
        r.claims,
        r.missing,
    )


_BUILDERS = {"pause_insertion": _pause, "tempo_change": _tempo, "energy_reduction": _energy}


def render_card_text(bundle: EvidenceBundle, context: AlignmentContext, verbosity: str) -> CardText:
    builder = _BUILDERS.get(bundle.flaw_type)
    if builder is None:
        raise ValueError(f"No explanation template for flaw type {bundle.flaw_type!r}")
    return builder(Renderer(bundle), context, verbosity)


def render_fallback_text(bundle: EvidenceBundle, problems: list[str]) -> CardText:
    """Facts-only text used if the standard template output fails verification."""
    r = Renderer(bundle)
    start, end, dur = _interval_phrase(r)
    label = CATEGORY_LABELS.get(bundle.flaw_type, "Detector")
    return CardText(
        f"{label} candidate (unverified wording)",
        f"The detector flagged a {label.lower()} candidate from {start} to {end} (length "
        f"{dur}). The explanatory wording could not be verified against the measurements, so "
        "only the recorded interval is stated.",
        [_strength_line(r)],
        "No interpretation is offered because the standard explanation failed verification.",
        "Listen to the participant excerpt around the flagged interval.",
        ["Explanation wording was withheld after a verification failure."],
        [],
        r.claims,
        r.missing,
    )
