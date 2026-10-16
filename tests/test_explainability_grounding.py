"""Architecture-level enforcement that explanations cannot see ground truth."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import flawbench
from flawbench.config import Config
from flawbench.detection.schema import DetectionResult
from flawbench.exceptions import ExplanationInputError
from flawbench.explainability import explain_result
from flawbench.explainability.facts import AlignmentContext
from flawbench.explainability.validator import validate_analysis
from tests.explain_fixtures import PAUSE_EVIDENCE, ctx, make_pred

SRC = Path(flawbench.__file__).parent
PACKAGES = ("explainability", "participant")
FORBIDDEN_MODULES = (
    "flawbench.evaluation",
    "flawbench.dataset.generator",
    "flawbench.dataset.injectors",
    "flawbench.dataset.schema",
    "flawbench.dataset.splits",
)
FORBIDDEN_NAMES = {"annotation", "annotations", "ground_truth", "output_interval"}


def _modules() -> list[Path]:
    return [p for pkg in PACKAGES for p in (SRC / pkg).glob("*.py")]


def _imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            names.extend(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
    return names


def test_participant_and_explanation_code_never_import_ground_truth_modules() -> None:
    for path in _modules():
        for name in _imports(ast.parse(path.read_text())):
            for bad in FORBIDDEN_MODULES:
                if name == bad or name.startswith(bad + "."):
                    # benchmark.py reads saved aggregate evaluation runs, by design
                    assert path.name == "benchmark.py" and bad == "flawbench.evaluation", (
                        path,
                        name,
                    )


def test_no_code_references_annotation_fields() -> None:
    for path in _modules():
        tree = ast.parse(path.read_text())
        docstrings = {
            id(n.body[0])
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and id(node) in docstrings:
                continue
            if isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN_NAMES, (path, node.attr)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in FORBIDDEN_NAMES, (path, node.value)


def _result(preds: list, **kw: object) -> DetectionResult:  # type: ignore[type-arg]
    return DetectionResult(
        schema_version="1.0", run_id="run_x", sample_id="s", source_id="src",
        ref_artifact_id="r", variant_artifact_id="v", ref_checksum="b" * 64,
        variant_checksum="a" * 64, detector_version="t", config_fingerprint="f",
        created_at="2026-01-01T00:00:00+00:00", sample_rate=16000, total_samples=160000,
        duration_seconds=10.0, predictions=preds, raw_candidates_count=len(preds),
    )  # fmt: skip


def test_invalid_detection_result_cannot_be_explained(config: Config) -> None:
    bad = make_pred("pause_insertion", PAUSE_EVIDENCE, start=150000, end=170000)  # past the end
    with pytest.raises(ExplanationInputError):
        explain_result(_result([bad]), ctx(), config.explanation)


def test_non_finite_evidence_in_result_is_rejected(config: Config) -> None:
    bad = make_pred(
        "pause_insertion", dict(PAUSE_EVIDENCE, variant_pause_duration_sec=float("nan"))
    )
    with pytest.raises(ExplanationInputError):
        explain_result(_result([bad]), ctx(), config.explanation)


def test_zero_prediction_result_yields_no_cards(config: Config) -> None:
    assert explain_result(_result([]), AlignmentContext(), config.explanation) == []


def test_explanation_depends_only_on_prediction_and_context(config: Config) -> None:
    pred = make_pred("pause_insertion", PAUSE_EVIDENCE)
    a = explain_result(_result([pred]), ctx(), config.explanation)[0].to_dict()
    b = explain_result(_result([pred]), ctx(), replace(config.explanation))[0].to_dict()
    assert a == b


def test_failed_analysis_cannot_carry_success_style_cards(config: Config) -> None:
    from flawbench.explainability.schema import ParticipantAnalysis

    pred = make_pred("pause_insertion", PAUSE_EVIDENCE)
    cards = explain_result(_result([pred]), ctx(), config.explanation)
    base = dict(
        session_id="s", created_utc="t", mode="m", reference=None, participant=None, stages=[],
        supported_categories=[], categories_completed=[], categories_failed={}, warnings=[],
        limitations=[], transcript=None, alignment={}, detection={}, provenance={},
        sort_policy="chronological",
    )  # fmt: skip
    failed = ParticipantAnalysis(status="failed", cards=cards, failure="x", **base)  # type: ignore[arg-type]
    assert any("failed analysis must not carry" in p for p in validate_analysis(failed))
    silent = ParticipantAnalysis(status="failed", cards=[], failure=None, **base)  # type: ignore[arg-type]
    assert validate_analysis(silent)
    empty = ParticipantAnalysis(status="completed", cards=[], **base)  # type: ignore[arg-type]
    assert validate_analysis(empty)
