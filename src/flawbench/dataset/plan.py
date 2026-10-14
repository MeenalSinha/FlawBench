"""Generation plans: a YAML/JSON file describing which variants to produce.

Example::

    plan_version: 1
    name: first-pass
    seed: 42                      # run seed; required only if any entry uses random placement
    variants:
      - source: src-1a2b3c4d5e6f  # a registered id, or "*" for every registered source
        flaw: pause_insertion
        severity: medium
        placement: {mode: explicit, at_sec: 3.5}
      - source: "*"
        flaw: energy_reduction
        severity: low
        placement: {mode: random}
        count: 2                  # replicate; each replicate gets its own derived seed
        parameters: {segment_sec: 1.5}   # optional overrides (label becomes "custom")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from flawbench.dataset.io import fingerprint
from flawbench.dataset.params import PARAM_SPECS, is_identifier
from flawbench.exceptions import DatasetError

PLAN_VERSION = 1
ALL_SOURCES = "*"
_ENTRY_KEYS = {"source", "flaw", "severity", "placement", "parameters", "count"}
_PLAN_KEYS = {"plan_version", "name", "seed", "variants"}
PLACEMENT_EXPLICIT = "explicit"
PLACEMENT_RANDOM = "random"
_PLACEMENT_KEYS = {
    PLACEMENT_EXPLICIT: {"mode", "at_sec", "start_sec"},
    PLACEMENT_RANDOM: {"mode"},
}


@dataclass(frozen=True)
class VariantSpec:
    source: str
    flaw: str
    severity: str | None
    placement: dict[str, Any]
    parameters: dict[str, Any]
    index: int  # replicate index for count > 1

    @property
    def key(self) -> str:
        """Stable key used for seed derivation (independent of expansion order)."""
        return f"{self.source}:{self.flaw}:{self.severity or 'custom'}:{self.index}"


@dataclass(frozen=True)
class Plan:
    name: str
    seed: int | None
    entries: tuple[dict[str, Any], ...]
    fingerprint: str

    def expand(self, registered_ids: list[str]) -> list[VariantSpec]:
        specs: list[VariantSpec] = []
        for entry in self.entries:
            sources = registered_ids if entry["source"] == ALL_SOURCES else [entry["source"]]
            for source in sources:
                for index in range(entry["count"]):
                    specs.append(
                        VariantSpec(
                            source,
                            entry["flaw"],
                            entry["severity"],
                            entry["placement"],
                            entry["parameters"],
                            index,
                        )
                    )
        return specs

    @property
    def uses_random_placement(self) -> bool:
        return any(e["placement"]["mode"] == PLACEMENT_RANDOM for e in self.entries)


def validate_placement(flaw: str, placement: Any, where: str) -> dict[str, Any]:
    if not isinstance(placement, dict) or placement.get("mode") not in _PLACEMENT_KEYS:
        raise DatasetError(f"{where}: placement.mode must be 'explicit' or 'random'.")
    extra = set(placement) - _PLACEMENT_KEYS[placement["mode"]]
    if extra:
        raise DatasetError(f"{where}: unknown placement key(s): {sorted(extra)}.")
    if placement["mode"] == PLACEMENT_EXPLICIT:
        key = "at_sec" if flaw == "pause_insertion" else "start_sec"
        wrong = "start_sec" if key == "at_sec" else "at_sec"
        if key not in placement or wrong in placement:
            raise DatasetError(f"{where}: explicit placement for '{flaw}' needs '{key}' only.")
        value = placement[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise DatasetError(f"{where}: {key} must be a non-negative number of seconds.")
    return dict(placement)


def load_plan(
    path: Path | str, *, known_flaws: set[str] | None = None, seed_override: int | None = None
) -> Plan:
    try:
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)  # JSON is valid YAML
    except OSError as exc:
        raise DatasetError("The generation plan could not be read.", detail=str(exc)) from exc
    except yaml.YAMLError as exc:
        raise DatasetError("The generation plan is not valid YAML/JSON.", detail=str(exc)) from exc
    return parse_plan(data, known_flaws=known_flaws, seed_override=seed_override)


def parse_plan(
    data: Any, *, known_flaws: set[str] | None = None, seed_override: int | None = None
) -> Plan:
    if not isinstance(data, dict):
        raise DatasetError("A generation plan must be a mapping.")
    extra = set(data) - _PLAN_KEYS
    if extra:
        raise DatasetError(f"Unknown plan key(s): {sorted(extra)}.")
    if data.get("plan_version") != PLAN_VERSION:
        raise DatasetError(f"plan_version must be {PLAN_VERSION}.")
    name = data.get("name", "unnamed")
    if not isinstance(name, str) or not is_identifier(name):
        raise DatasetError("Plan name may contain only letters, digits, '.', '_' and '-'.")
    seed = seed_override if seed_override is not None else data.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise DatasetError("Plan seed must be an integer.")
    variants = data.get("variants")
    if not isinstance(variants, list) or not variants:
        raise DatasetError("A plan needs a non-empty 'variants' list.")
    entries: list[dict[str, Any]] = []
    valid_flaws = known_flaws if known_flaws is not None else set(PARAM_SPECS)
    for i, raw in enumerate(variants, start=1):
        where = f"variants[{i}]"
        if not isinstance(raw, dict):
            raise DatasetError(f"{where} must be a mapping.")
        extra = set(raw) - _ENTRY_KEYS
        if extra:
            raise DatasetError(f"{where}: unknown key(s): {sorted(extra)}.")
        for required in ("source", "flaw", "placement"):
            if required not in raw:
                raise DatasetError(f"{where}: missing '{required}'.")
        if raw["flaw"] not in valid_flaws:
            raise DatasetError(f"{where}: unknown flaw '{raw['flaw']}'.")
        count = raw.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise DatasetError(f"{where}: count must be a positive integer.")
        params = raw.get("parameters", {})
        if not isinstance(params, dict):
            raise DatasetError(f"{where}: parameters must be a mapping.")
        severity = raw.get("severity")
        if severity is None and not params:
            raise DatasetError(f"{where}: give a severity or explicit parameters.")
        entries.append(
            {
                "source": str(raw["source"]),
                "flaw": raw["flaw"],
                "severity": severity,
                "placement": validate_placement(raw["flaw"], raw["placement"], where),
                "parameters": dict(params),
                "count": count,
            }
        )
    plan = Plan(name=name, seed=seed, entries=tuple(entries), fingerprint="")
    if plan.uses_random_placement and seed is None:
        raise DatasetError("Random placement requires an explicit plan seed (or --seed).")
    canonical = {"plan_version": PLAN_VERSION, "name": name, "seed": seed, "variants": entries}
    return Plan(name=name, seed=seed, entries=tuple(entries), fingerprint=fingerprint(canonical))
