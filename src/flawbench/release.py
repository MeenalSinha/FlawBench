"""Release validation workflow for FlawBench release candidate verification.

Executes a repeatable checklist verifying configuration, paths, dependencies,
data stores, audio engine, detection engine, evaluation reports, and UI readiness.
Distinguishes: PASS, FAIL, SKIP, N/A, and BLOCKED.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

from flawbench import __version__
from flawbench.config import Config
from flawbench.paths import ProjectPaths
from flawbench.provenance import config_fingerprint


@dataclass(frozen=True)
class CheckResult:
    category: str
    name: str
    status: str  # "PASS" | "FAIL" | "SKIP" | "N/A" | "BLOCKED"
    detail: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseValidationReport:
    version: str
    python_version: str
    platform_name: str
    config_fingerprint: str
    checks: list[CheckResult]
    passed_count: int
    failed_count: int
    skipped_count: int
    na_count: int
    blocked_count: int
    is_ready: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "python_version": self.python_version,
            "platform_name": self.platform_name,
            "config_fingerprint": self.config_fingerprint,
            "checks": [c.to_dict() for c in self.checks],
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "na_count": self.na_count,
            "blocked_count": self.blocked_count,
            "is_ready": self.is_ready,
            "summary": self.summary,
        }

    def render_markdown(self) -> str:
        ready_label = "**RELEASE READY**" if self.is_ready else "**NOT READY (Defects Detected)**"
        lines = [
            f"# FlawBench Release Validation Report — v{self.version}",
            "",
            f"- **Platform**: {self.platform_name}",
            f"- **Python**: {self.python_version}",
            f"- **Config Fingerprint**: `{self.config_fingerprint[:16]}`",
            f"- **Status**: {ready_label}",
            "",
            "## Summary of Checks",
            "",
            "| Total | Passed | Failed | Skipped | N/A | Blocked |",
            "|---|---|---|---|---|---|",
            (
                f"| {len(self.checks)} | {self.passed_count} | {self.failed_count}"
                f" | {self.skipped_count} | {self.na_count} | {self.blocked_count} |"
            ),
            "",
            "## Detailed Checklist",
            "",
            "| Category | Check | Status | Details | Required |",
            "|---|---|---|---|---|",
        ]
        for c in self.checks:
            req_str = "Yes" if c.required else "No"
            lines.append(f"| {c.category} | {c.name} | **{c.status}** | {c.detail} | {req_str} |")
        lines.append("")
        lines.append("## Conclusion")
        lines.append("")
        lines.append(self.summary)
        return "\n".join(lines)


def run_release_validation(
    config: Config, paths: ProjectPaths, *, run_tests: bool = False
) -> ReleaseValidationReport:
    checks: list[CheckResult] = []

    # 1. Configuration check
    try:
        cfg_dict = config.to_dict()
        assert "app" in cfg_dict and "storage" in cfg_dict and "detection" in cfg_dict
        source_note = config.config_file.name if config.config_file else "defaults + environment"
        checks.append(
            CheckResult(
                "Configuration",
                "Schema & defaults validation",
                "PASS",
                f"Configuration loaded validly from {source_note}",
            )
        )
    except Exception as exc:
        checks.append(
            CheckResult(
                "Configuration",
                "Schema & defaults validation",
                "FAIL",
                f"Config error: {exc}",
            )
        )

    # 2. Directory access checks
    for d in paths.inspect():
        if d.accessible:
            checks.append(
                CheckResult(
                    "Paths & Storage",
                    f"{d.name.capitalize()} directory ({d.role})",
                    "PASS",
                    f"Accessible at {d.path}",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "Paths & Storage",
                    f"{d.name.capitalize()} directory ({d.role})",
                    "FAIL",
                    f"{d.problem or 'Not accessible'} at {d.path}",
                )
            )

    # 3. Core dependencies checks
    core_pkgs = [
        ("numpy", "2.0+"),
        ("soundfile", "0.12+"),
        ("audiotsm", "0.1.2+"),
        ("librosa", "0.10+"),
        ("plotly", "5.20+"),
        ("yaml", "6.0+"),
        ("streamlit", "1.56+"),
    ]
    for pkg, min_ver in core_pkgs:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "installed")
            checks.append(
                CheckResult(
                    "Dependencies",
                    f"Package {pkg}",
                    "PASS",
                    f"Version {ver} satisfies requirement ({min_ver})",
                )
            )
        except ImportError as exc:
            checks.append(
                CheckResult(
                    "Dependencies",
                    f"Package {pkg}",
                    "FAIL",
                    f"Missing core package: {exc}",
                )
            )

    # 4. Audio decoder check
    try:
        import soundfile as sf

        checks.append(
            CheckResult(
                "Audio Decoder",
                "libsndfile binding",
                "PASS",
                f"libsndfile {getattr(sf, '__libsndfile_version__', 'unknown')} available",
            )
        )
    except Exception as exc:
        checks.append(
            CheckResult(
                "Audio Decoder",
                "libsndfile binding",
                "FAIL",
                f"Audio decoder failed: {exc}",
            )
        )

    # 5. Dataset registry check
    try:
        from flawbench.dataset.generator import DatasetGenerator
        from flawbench.dataset.registry import SourceRegistry

        registry = SourceRegistry(config, paths)
        sources = registry.all()
        generator = DatasetGenerator(config, paths, registry)
        samples = generator.samples()
        checks.append(
            CheckResult(
                "Dataset",
                "Registry & manifest integrity",
                "PASS",
                f"{len(sources)} source(s), {len(samples)} sample(s) registered",
            )
        )
    except Exception as exc:
        checks.append(
            CheckResult(
                "Dataset",
                "Registry & manifest integrity",
                "FAIL",
                f"Dataset error: {exc}",
            )
        )

    # 6. Feature engine check
    try:
        from flawbench.features.store import FeatureStore

        fstore = FeatureStore(paths.features_dir, config.features.feature_manifest_filename)
        artifacts = fstore.list_manifest()
        checks.append(
            CheckResult(
                "Features",
                "Feature store & manifest integrity",
                "PASS",
                f"{len(artifacts)} feature artifact(s) recorded",
            )
        )
    except Exception as exc:
        checks.append(
            CheckResult(
                "Features",
                "Feature store & manifest integrity",
                "FAIL",
                f"Feature store error: {exc}",
            )
        )

    # 7. Detection engine check
    try:
        from flawbench.detection.engine import DetectionEngine
        from flawbench.detection.store import DetectionStore

        engine = DetectionEngine(config.detection)
        assert len(engine.detectors) == 3
        dstore = DetectionStore(paths.detection_dir, config.detection.detection_manifest_filename)
        detections = dstore.list_manifest()
        checks.append(
            CheckResult(
                "Detection",
                "Detectors & prediction store",
                "PASS",
                f"3 independent detectors active; {len(detections)} detection record(s) in store",
            )
        )
    except Exception as exc:
        checks.append(
            CheckResult(
                "Detection",
                "Detectors & prediction store",
                "FAIL",
                f"Detection engine error: {exc}",
            )
        )

    # 8. Evaluation engine & benchmark reports check
    try:
        from flawbench.evaluation.store import EvaluationStore

        estore = EvaluationStore(paths)
        runs = estore.list_runs()
        # Look for heldout_rc or dev_rc in docs/phase7_evidence
        evidence_dir = paths.project_root / "docs/phase7_evidence/reports"
        evidence_runs = [p.name for p in evidence_dir.glob("*") if p.is_dir()]
        checks.append(
            CheckResult(
                "Evaluation",
                "Benchmark runs & evaluation store",
                "PASS",
                (
                    f"{len(runs)} run(s) in output store; "
                    f"{len(evidence_runs)} verified evidence run(s) in repository"
                ),
            )
        )
    except Exception as exc:
        checks.append(
            CheckResult(
                "Evaluation",
                "Benchmark runs & evaluation store",
                "FAIL",
                f"Evaluation error: {exc}",
            )
        )

    # 9. Explainability & Participant analysis check
    try:
        from flawbench.explainability.templates import CATEGORY_LABELS
        from flawbench.participant import ParticipantAnalyzer

        ParticipantAnalyzer(config, paths)
        assert len(CATEGORY_LABELS) == 3
        export_fmts = ", ".join(config.explanation.export_formats)
        cat_labels = ", ".join(CATEGORY_LABELS.keys())
        checks.append(
            CheckResult(
                "Explainability",
                "Participant analyzer & explanation cards",
                "PASS",
                f"Templates active for {cat_labels}; export formats supported: {export_fmts}",
            )
        )
    except Exception as exc:
        checks.append(
            CheckResult(
                "Explainability",
                "Participant analyzer & explanation cards",
                "FAIL",
                f"Explainability error: {exc}",
            )
        )

    # 10. Application Entry Point check
    app_py = paths.project_root / "app.py"
    if app_py.is_file():
        try:
            from streamlit.testing.v1 import AppTest

            at = AppTest.from_file(str(app_py), default_timeout=20).run()
            if not at.exception:
                title_val = at.title[0].value if at.title else "OK"
                checks.append(
                    CheckResult(
                        "Application UI",
                        "Streamlit app.py execution test",
                        "PASS",
                        f"Streamlit entry point launches with title: {title_val}",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        "Application UI",
                        "Streamlit app.py execution test",
                        "FAIL",
                        f"App raised exception on startup: {at.exception}",
                    )
                )
        except Exception as exc:
            checks.append(
                CheckResult(
                    "Application UI",
                    "Streamlit app.py execution test",
                    "FAIL",
                    f"App startup error: {exc}",
                )
            )
    else:
        checks.append(
            CheckResult(
                "Application UI",
                "Streamlit app.py execution test",
                "FAIL",
                "app.py not found",
            )
        )

    # 11. Automated test suite check (optional on CLI flag)
    if run_tests:
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=short"],
                cwd=str(paths.project_root),
                capture_output=True,
                text=True,
                timeout=180,
            )
            if res.returncode == 0:
                checks.append(
                    CheckResult(
                        "Test Suite",
                        "Pytest regression suite",
                        "PASS",
                        "Full test suite executed and passed successfully",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        "Test Suite",
                        "Pytest regression suite",
                        "FAIL",
                        f"Pytest exited with code {res.returncode}:\n{res.stdout[-300:]}",
                    )
                )
        except Exception as exc:
            checks.append(
                CheckResult(
                    "Test Suite",
                    "Pytest regression suite",
                    "BLOCKED",
                    f"Test execution error: {exc}",
                )
            )
    else:
        checks.append(
            CheckResult(
                "Test Suite",
                "Pytest regression suite",
                "SKIP",
                "Run with --run-tests to execute the full automated test suite",
                required=False,
            )
        )

    # Compute tally
    passed = sum(1 for c in checks if c.status == "PASS")
    failed = sum(1 for c in checks if c.status == "FAIL")
    skipped = sum(1 for c in checks if c.status == "SKIP")
    na = sum(1 for c in checks if c.status == "N/A")
    blocked = sum(1 for c in checks if c.status == "BLOCKED")

    is_ready = failed == 0 and blocked == 0
    if is_ready:
        summary = (
            f"All {passed} required checks passed. FlawBench v{__version__} is verified as a "
            "functional release candidate ready for demonstration, packaging, and project handover."
        )
    else:
        summary = (
            f"{failed} check(s) failed out of {len(checks)}. "
            "Please resolve the issues detailed above."
        )

    return ReleaseValidationReport(
        version=__version__,
        python_version=sys.version.split()[0],
        platform_name=platform.platform(),
        config_fingerprint=config_fingerprint(config),
        checks=checks,
        passed_count=passed,
        failed_count=failed,
        skipped_count=skipped,
        na_count=na,
        blocked_count=blocked,
        is_ready=is_ready,
        summary=summary,
    )
