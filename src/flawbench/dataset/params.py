"""Flaw identifiers, parameter specifications, bounds, and severity-profile validation.

Pure module (no I/O, no dependency on ``flawbench.config``) so configuration can validate profiles
at load time. The *values* (profiles, bounds) live in configuration; only the structural facts
(which parameters each flaw type needs) are defined here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from flawbench.exceptions import ConfigurationError, InjectionError

FLAW_PAUSE = "pause_insertion"
FLAW_TEMPO = "tempo_change"
FLAW_ENERGY = "energy_reduction"

#: Required numeric parameters per flaw type. New flaw types extend this table.
PARAM_SPECS: dict[str, tuple[str, ...]] = {
    FLAW_PAUSE: ("duration_sec",),
    FLAW_TEMPO: ("rate", "segment_sec"),
    FLAW_ENERGY: ("gain_db", "segment_sec"),
}

CUSTOM_SEVERITY = "custom"
_IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_identifier(value: str) -> bool:
    return bool(_IDENT.match(value))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_params(
    flaw_type: str, params: Mapping[str, Any], bounds: Mapping[str, Any], *, error: type[Exception]
) -> dict[str, float]:
    """Check keys, numeric type, and configured bounds; return a clean float dict."""
    if flaw_type not in PARAM_SPECS:
        raise error(f"Unknown flaw type '{flaw_type}'.")
    required = set(PARAM_SPECS[flaw_type])
    given = set(params)
    if given != required:
        raise error(
            f"Parameters for '{flaw_type}' must be exactly {sorted(required)}; got {sorted(given)}."
        )
    flaw_bounds = bounds.get(flaw_type, {})
    out: dict[str, float] = {}
    for key in PARAM_SPECS[flaw_type]:
        value = params[key]
        if not _number(value):
            raise error(f"Parameter '{flaw_type}.{key}' must be a finite number; got {value!r}.")
        limits = flaw_bounds.get(key)
        if limits is None:
            raise error(f"No bounds are configured for '{flaw_type}.{key}'.")
        if not (limits["min"] <= value <= limits["max"]):
            raise error(
                f"Parameter '{flaw_type}.{key}'={value} is outside the configured bounds "
                f"[{limits['min']}, {limits['max']}]."
            )
        out[key] = float(value)
    return out


def validate_bounds(bounds: Any) -> None:
    if not isinstance(bounds, Mapping):
        raise ConfigurationError("dataset.parameter_bounds must be a mapping.")
    for flaw, spec in PARAM_SPECS.items():
        entry = bounds.get(flaw)
        if not isinstance(entry, Mapping):
            raise ConfigurationError(f"dataset.parameter_bounds is missing '{flaw}'.")
        for key in spec:
            limits = entry.get(key)
            if (
                not isinstance(limits, Mapping)
                or set(limits) != {"min", "max"}
                or not _number(limits["min"])
                or not _number(limits["max"])
                or limits["min"] > limits["max"]
            ):
                raise ConfigurationError(
                    f"dataset.parameter_bounds.{flaw}.{key} must be {{min, max}} with min <= max."
                )
    unknown = set(bounds) - set(PARAM_SPECS)
    if unknown:
        raise ConfigurationError(f"Unknown flaw type(s) in parameter_bounds: {sorted(unknown)}.")
    if not (bounds[FLAW_TEMPO]["rate"]["min"] > 0):
        raise ConfigurationError("dataset.parameter_bounds.tempo_change.rate.min must be > 0.")
    if not (bounds[FLAW_ENERGY]["gain_db"]["max"] < 0):
        raise ConfigurationError(
            "dataset.parameter_bounds.energy_reduction.gain_db.max must be negative (a reduction)."
        )
    for flaw in (FLAW_PAUSE,):
        if not (bounds[flaw]["duration_sec"]["min"] > 0):
            raise ConfigurationError(
                f"dataset.parameter_bounds.{flaw}.duration_sec.min must be > 0."
            )
    for flaw in (FLAW_TEMPO, FLAW_ENERGY):
        if not (bounds[flaw]["segment_sec"]["min"] > 0):
            raise ConfigurationError(
                f"dataset.parameter_bounds.{flaw}.segment_sec.min must be > 0."
            )


def validate_profiles(profiles: Any, bounds: Mapping[str, Any]) -> None:
    """Validate severity profiles against parameter specs and bounds (called at config load)."""
    if not isinstance(profiles, Mapping) or not profiles:
        raise ConfigurationError("dataset.severity_profiles must be a non-empty mapping.")
    for flaw, severities in profiles.items():
        if flaw not in PARAM_SPECS:
            raise ConfigurationError(
                f"Unknown flaw type '{flaw}' in severity_profiles.",
                detail=f"known: {', '.join(PARAM_SPECS)}",
            )
        if not isinstance(severities, Mapping) or not severities:
            raise ConfigurationError(f"severity_profiles.{flaw} must be a non-empty mapping.")
        for severity, params in severities.items():
            if not isinstance(severity, str) or not is_identifier(severity):
                raise ConfigurationError(f"Invalid severity name {severity!r} for '{flaw}'.")
            if severity == CUSTOM_SEVERITY:
                raise ConfigurationError(
                    f"'{CUSTOM_SEVERITY}' is reserved and cannot be a profile."
                )
            if not isinstance(params, Mapping):
                raise ConfigurationError(f"severity_profiles.{flaw}.{severity} must be a mapping.")
            try:
                validate_params(flaw, params, bounds, error=InjectionError)
            except InjectionError as exc:
                raise ConfigurationError(
                    f"Invalid severity profile {flaw}.{severity}: {exc.user_message}"
                ) from exc


def resolve_parameters(
    flaw_type: str,
    severity: str | None,
    overrides: Mapping[str, Any] | None,
    profiles: Mapping[str, Any],
    bounds: Mapping[str, Any],
    *,
    error: type[Exception] = InjectionError,
) -> tuple[str, str | None, dict[str, float]]:
    """Resolve profile + overrides to ``(severity_id, base_severity, validated_params)``.

    If any override is supplied, the severity becomes ``custom`` and the profile it started from
    is returned as ``base_severity`` so the label never claims a profile that was not used.
    """
    if flaw_type not in profiles:
        raise error(f"Flaw type '{flaw_type}' has no configured severity profiles.")
    base: dict[str, Any] = {}
    if severity is not None:
        if severity not in profiles[flaw_type]:
            raise error(
                f"Unknown severity '{severity}' for '{flaw_type}'. "
                f"Configured: {', '.join(profiles[flaw_type])}."
            )
        base = dict(profiles[flaw_type][severity])
    overrides = dict(overrides or {})
    if severity is None and not overrides:
        raise error("Specify a severity profile or explicit parameters.")
    merged = {**base, **overrides}
    label = severity if (severity is not None and not overrides) else CUSTOM_SEVERITY
    return (
        label,
        (severity if label == CUSTOM_SEVERITY else None),
        validate_params(flaw_type, merged, bounds, error=error),
    )
