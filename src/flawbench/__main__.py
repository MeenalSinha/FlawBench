"""Command-line tools.

Phase 1 commands (unchanged): ``config``, ``check``, ``init``, ``provenance``.
Phase 2 commands: ``dataset {register-reference, list-sources, generate, validate, split, export}``.

``--config`` is accepted before or after the command name.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from flawbench.config import Config, load_config
from flawbench.exceptions import ConfigurationError, FlawBenchError
from flawbench.paths import ProjectPaths
from flawbench.provenance import build_provenance
from flawbench.status import collect_status

PHASE1_COMMANDS = ("config", "check", "init", "provenance")


def _add_config_option(parser: argparse.ArgumentParser, *, is_sub: bool) -> None:
    kwargs = {"default": argparse.SUPPRESS} if is_sub else {}
    parser.add_argument("--config", help="path to a YAML configuration file", **kwargs)  # type: ignore[arg-type]


def _parse_params(items: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--param expects KEY=VALUE, got {item!r}")
        try:
            out[key.strip()] = float(value)
        except ValueError:
            raise ValueError(f"--param {key}: {value!r} is not a number") from None
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flawbench", description="FlawBench command-line tools")
    _add_config_option(parser, is_sub=False)
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    helps = {
        "config": "print the effective configuration as JSON",
        "check": "report real status of configuration, directories and the audio decoder",
        "init": "create the writable output directories",
        "provenance": "print software versions and a configuration fingerprint",
    }
    for name in PHASE1_COMMANDS:
        _add_config_option(sub.add_parser(name, help=helps[name]), is_sub=True)

    ds = sub.add_parser("dataset", help="dataset and ground-truth engine")
    _add_config_option(ds, is_sub=True)
    dsub = ds.add_subparsers(dest="dataset_command", required=True, metavar="action")

    reg = dsub.add_parser("register-reference", help="register a reference recording")
    _add_config_option(reg, is_sub=True)
    reg.add_argument("audio", help="audio file (inside the reference directory, or use --copy)")
    group = reg.add_mutually_exclusive_group(required=True)
    group.add_argument("--transcript-text")
    group.add_argument("--transcript-file")
    reg.add_argument(
        "--source-type",
        required=True,
        choices=["self_recorded", "public_domain", "licensed", "other_permissioned"],
    )
    reg.add_argument("--attribution", required=True, help="origin / author of the recording")
    reg.add_argument("--license", required=True, help="licence or permission statement")
    reg.add_argument(
        "--rights-confirmed",
        action="store_true",
        help="confirm you may process and modify this recording (required)",
    )
    reg.add_argument(
        "--redistribution", choices=["permitted", "prohibited", "unknown"], default="unknown"
    )
    reg.add_argument("--source-id", help="explicit id (default derives from the checksum)")
    reg.add_argument(
        "--copy",
        action="store_true",
        default=None,
        help="copy the file into the reference directory (never overwrites)",
    )
    reg.add_argument("--notes")

    ls = dsub.add_parser("list-sources", help="list registered sources")
    _add_config_option(ls, is_sub=True)

    gen = dsub.add_parser("generate", help="generate variants from a plan or a single request")
    _add_config_option(gen, is_sub=True)
    gen.add_argument("--plan", help="generation plan file (YAML/JSON)")
    gen.add_argument("--source", help="registered source id (single variant)")
    gen.add_argument("--flaw", help="pause_insertion | tempo_change | energy_reduction")
    gen.add_argument("--severity", help="configured severity profile")
    gen.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="explicit parameter override (severity becomes 'custom')",
    )
    gen.add_argument("--at-sec", type=float, help="pause insertion point (explicit placement)")
    gen.add_argument("--start-sec", type=float, help="segment start (tempo/energy, explicit)")
    gen.add_argument("--random-placement", action="store_true", help="seeded random placement")
    gen.add_argument("--seed", type=int, help="run seed (required for random placement)")
    gen.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing sample (history is kept)",
    )

    val = dsub.add_parser("validate", help="audit the dataset against real files")
    _add_config_option(val, is_sub=True)
    val.add_argument("--report", help="write the JSON report to this file")
    val.add_argument(
        "--shallow", action="store_true", help="skip sample-level content verification"
    )

    spl = dsub.add_parser("split", help="assign sources to splits (source-level, persisted)")
    _add_config_option(spl, is_sub=True)
    spl.add_argument("--force", action="store_true", help="discard existing assignments")

    exp = dsub.add_parser("export", help="export redistributable samples to outputs/exports")
    _add_config_option(exp, is_sub=True)
    exp.add_argument("--name", required=True, help="export directory name")
    exp.add_argument("--sample-id", action="append", help="sample to export (default: all)")

    # ---- Phase 3: features engine ----------------------------------------------------
    feat = sub.add_parser("features", help="acoustic feature extraction & comparison engine")
    _add_config_option(feat, is_sub=True)
    fsub = feat.add_subparsers(dest="features_command", required=True, metavar="action")

    f_ext = fsub.add_parser("extract", help="extract acoustic features for reference or variant")
    _add_config_option(f_ext, is_sub=True)
    f_group = f_ext.add_mutually_exclusive_group(required=True)
    f_group.add_argument("--source", help="registered source id (extract reference features)")
    f_group.add_argument("--sample", help="committed sample id (extract variant features)")
    f_group.add_argument("--all", action="store_true", help="extract features for entire dataset")
    f_ext.add_argument("--force", action="store_true", help="force re-extraction ignoring cache")

    f_cmp = fsub.add_parser("compare", help="compare variant features against reference")
    _add_config_option(f_cmp, is_sub=True)
    c_group = f_cmp.add_mutually_exclusive_group(required=True)
    c_group.add_argument("--sample", help="committed sample id to compare against reference")
    c_group.add_argument("--all", action="store_true", help="compare all variants in dataset")
    f_cmp.add_argument("--force", action="store_true", help="force re-extraction before comparison")

    f_ls = fsub.add_parser("list", help="list extracted feature artifacts in the manifest")
    _add_config_option(f_ls, is_sub=True)

    f_val = fsub.add_parser("validate", help="validate feature store manifest and artifacts")
    _add_config_option(f_val, is_sub=True)

    f_exp = fsub.add_parser("export", help="export feature summary metrics to JSON")
    _add_config_option(f_exp, is_sub=True)
    f_exp.add_argument("--output", help="output JSON file path (default: stdout)")

    # ---- Phase 4: detection engine ---------------------------------------------------
    det = sub.add_parser("detect", help="flaw detection & temporal localization engine")
    _add_config_option(det, is_sub=True)
    det.add_argument(
        "action",
        nargs="?",
        choices=["list", "validate"],
        help="action to perform (list saved detections or validate artifacts)",
    )
    det.add_argument("--sample", help="committed sample id to run flaw detection on")
    det.add_argument(
        "--all", action="store_true", help="run flaw detection for all samples in dataset"
    )
    det.add_argument(
        "--list", action="store_true", help="list saved detection results from manifest"
    )
    det.add_argument("--validate", action="store_true", help="validate stored detection results")
    det.add_argument(
        "--force", action="store_true", help="force recomputation ignoring cached results"
    )
    det.add_argument("--export", help="export detection predictions summary to JSON file")

    # ---- Phase 5: evaluation engine --------------------------------------------------
    ev = sub.add_parser("evaluate", help="formal evaluation & benchmarking engine")
    _add_config_option(ev, is_sub=True)
    esub = ev.add_subparsers(dest="evaluate_command", required=True, metavar="action")

    ev_run = esub.add_parser("run", help="run evaluation against ground truth")
    _add_config_option(ev_run, is_sub=True)
    ev_run.add_argument("--split", default=None, help="evaluation split (test | val | train | all)")
    ev_run.add_argument("--sample", help="evaluate a single committed sample id")
    ev_run.add_argument("--run-id", help="custom identifier for this evaluation run")
    ev_run.add_argument(
        "--no-controls", action="store_true", help="skip negative identity controls"
    )
    ev_run.add_argument(
        "--bootstrap", action="store_true", default=None, help="enable source-level bootstrap CIs"
    )

    ev_ls = esub.add_parser("list", help="list saved evaluation runs from manifest")
    _add_config_option(ev_ls, is_sub=True)

    ev_rep = esub.add_parser("report", help="print or export markdown evaluation report")
    _add_config_option(ev_rep, is_sub=True)
    ev_rep.add_argument("--run", required=True, help="evaluation run identifier")
    ev_rep.add_argument("--output", help="output markdown file (default: stdout)")

    ev_val = esub.add_parser("validate", help="validate stored evaluation artifacts")
    _add_config_option(ev_val, is_sub=True)
    ev_val.add_argument("--run", required=True, help="evaluation run identifier")

    return parser


def _run_phase1(args: argparse.Namespace, config: Config) -> int:
    if args.command == "config":
        print(json.dumps(config.to_dict(), indent=2))
        return 0
    if args.command == "provenance":
        print(json.dumps(build_provenance(config), indent=2))
        return 0
    if args.command == "init":
        try:
            created = ProjectPaths(config).ensure_output_directories()
        except FlawBenchError as exc:
            print(f"Storage error: {exc.public_message(expose_details=True)}", file=sys.stderr)
            return 1
        for path in created:
            print(f"ready: {path}")
        return 0
    items = collect_status(config, ProjectPaths(config), expose_details=True)
    for item in items:
        print(f"[{'OK' if item.ok else 'FAIL'}] {item.label}: {item.message}")
    return 0 if all(i.ok for i in items) else 1


def _run_dataset(args: argparse.Namespace, config: Config) -> int:
    # Imported lazily so Phase 1 commands never pay for (or depend on) the dataset engine.
    from flawbench.dataset import plan as plan_mod
    from flawbench.dataset.export import export_samples
    from flawbench.dataset.generator import DatasetGenerator
    from flawbench.dataset.registry import SourceRegistry
    from flawbench.dataset.splits import build_splits
    from flawbench.dataset.validator import validate_dataset
    from flawbench.exceptions import DatasetError

    paths = ProjectPaths(config)
    cmd = args.dataset_command
    if cmd == "register-reference":
        rec = SourceRegistry(config, paths).register(
            args.audio,
            transcript_text=args.transcript_text,
            transcript_file=args.transcript_file,
            source_type=args.source_type,
            attribution=args.attribution,
            license=args.license,
            rights_confirmed=args.rights_confirmed,
            redistribution=args.redistribution,
            source_id=args.source_id,
            copy=args.copy,
            notes=args.notes,
        )
        print(
            f"registered {rec.source_id} sha256={rec.sha256[:16]}... "
            f"redistribution={rec.redistribution}"
        )
        return 0
    if cmd == "list-sources":
        records = SourceRegistry(config, paths).all()
        if not records:
            print("No sources are registered.")
        for r in records:
            print(
                f"{r.source_id}\t{r.data['duration_seconds']:.2f}s\t"
                f"{r.data['sample_rate']} Hz\t{r.redistribution}\t{r.data['original_filename']}"
            )
        return 0
    if cmd == "generate":
        gen = DatasetGenerator(config, paths)
        if args.plan:
            conflicting = [
                a
                for a in ("source", "flaw", "severity", "param", "at_sec", "start_sec")
                if getattr(args, a)
            ] + (["random_placement"] if args.random_placement else [])
            if conflicting:
                raise DatasetError(f"--plan cannot be combined with: {', '.join(conflicting)}.")
            plan = plan_mod.load_plan(args.plan, seed_override=args.seed)
            batch = gen.run_plan(plan, overwrite=True if args.overwrite else None)
            print(f"run {batch.run_id}: {json.dumps(batch.counts, sort_keys=True)}")
            for o in batch.outcomes:
                if o.status == "failed":
                    print(f"  FAILED {o.key}: {o.error}", file=sys.stderr)
            return 0 if batch.ok else 1
        if not (args.source and args.flaw):
            raise DatasetError("Provide --plan, or --source and --flaw.")
        if args.random_placement:
            if args.at_sec is not None or args.start_sec is not None:
                raise DatasetError("--random-placement conflicts with --at-sec/--start-sec.")
            placement: dict[str, object] = {"mode": "random"}
        else:
            key, value = (
                ("at_sec", args.at_sec)
                if args.flaw == "pause_insertion"
                else ("start_sec", args.start_sec)
            )
            if value is None:
                raise DatasetError(f"Give --{key.replace('_', '-')} or --random-placement.")
            placement = {"mode": "explicit", key: value}
        res = gen.generate_variant(
            args.source,
            args.flaw,
            severity=args.severity,
            parameters=_parse_params(args.param) or None,
            placement=placement,
            seed=args.seed,
            overwrite=True if args.overwrite else None,
        )
        iv = res.record.data["annotation"]["output_interval"]
        print(
            f"{res.status} {res.record.sample_id} output_interval="
            f"[{iv['start_sec']:.4f}, {iv['end_sec']:.4f}) s"
        )
        return 0
    if cmd == "validate":
        report = validate_dataset(config, paths, deep=not args.shallow)
        if args.report:
            Path(args.report).write_text(report.to_json() + "\n", encoding="utf-8")
        c = report.counts
        print(
            f"sources={c.get('sources', 0)} samples={c.get('samples', 0)} "
            f"errors={len(report.errors)} warnings={len(report.warnings)}"
        )
        if report.is_empty:
            print("The dataset is empty: no sources or samples are registered.")
        for f in report.findings:
            print(f"  [{f.severity.upper()}] {f.code}: {f.message} ({f.subject})")
        print("VALID" if report.ok else "INVALID")
        return 0 if report.ok else 1
    if cmd == "split":
        result = build_splits(config, paths, force=args.force)
        names = sorted(config.dataset.split_ratios)
        print(json.dumps(result.counts(names), sort_keys=True))
        for w in result.warnings:
            print(f"warning: {w}", file=sys.stderr)
        return 0
    if cmd == "export":
        exp = export_samples(config, args.name, args.sample_id, paths)
        print(f"exported {len(exp.exported_samples)} sample(s) to {exp.destination}")
        for sid, why in exp.refused.items():
            print(f"  refused {sid}: {why}", file=sys.stderr)
        return 0
    raise AssertionError(cmd)  # pragma: no cover


def _run_features(args: argparse.Namespace, config: Config) -> int:
    from flawbench.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(config)
    cmd = args.features_command

    if cmd == "extract":
        if args.source:
            res = pipeline.extract_reference(args.source, force=args.force)
            print(
                f"Extracted {res.metadata.artifact_id}: {res.num_frames} frames "
                f"({res.metadata.duration_seconds:.2f}s)"
            )
            return 0
        if args.sample:
            res = pipeline.extract_variant(args.sample, force=args.force)
            print(
                f"Extracted {res.metadata.artifact_id}: {res.num_frames} frames "
                f"({res.metadata.duration_seconds:.2f}s)"
            )
            return 0
        if args.all:
            counts = pipeline.extract_dataset(force=args.force)
            print(f"Dataset feature extraction complete: {json.dumps(counts, sort_keys=True)}")
            return 0

    if cmd == "compare":
        if args.sample:
            cmp_res = pipeline.compare_variant_to_reference(args.sample, force_extract=args.force)
            print(f"Compared {cmp_res.variant_artifact_id} vs {cmp_res.ref_artifact_id}:")
            print(
                f"  DTW cost: {cmp_res.dtw_cost:.3f} "
                f"(normalized: {cmp_res.normalized_dtw_cost:.4f})"
            )
            print(f"  Path length: {cmp_res.path_length} points")
            print(f"  Mean delta RMS: {cmp_res.global_metrics.get('mean_delta_rms_db', 0):.2f} dB")
            print(f"  Candidate flaw regions: {len(cmp_res.candidate_regions)}")
            for r in cmp_res.candidate_regions:
                print(
                    f"    - [{r.get('flaw_hypothesis')}] "
                    f"var [{r.get('var_start_sec')} - {r.get('var_end_sec')})s"
                )
            return 0
        if args.all:
            results = pipeline.compare_all(force_extract=args.force)
            print(f"Completed {len(results)} variant-to-reference comparisons.")
            return 0

    if cmd == "list":
        entries = pipeline.store.list_manifest()
        print(f"Feature artifacts: {len(entries)}")
        for e in entries:
            print(
                f"  [{e.role}] {e.artifact_id} (source: {e.source_id}, sample: {e.sample_id}) "
                f"{e.num_frames} frames"
            )
        return 0

    if cmd == "validate":
        val = pipeline.store.validate_store()
        print(
            f"Feature store valid: {val['is_valid']} "
            f"({val['valid_entries']}/{val['total_entries']} entries on disk)"
        )
        if val["missing_artifacts"]:
            for m in val["missing_artifacts"]:
                print(f"  Missing files for artifact: {m}", file=sys.stderr)
        return 0 if val["is_valid"] else 1

    if cmd == "export":
        out_p = Path(args.output) if args.output else None
        summaries = pipeline.export_feature_summaries(out_p)
        if out_p is None:
            print(json.dumps(summaries, indent=2))
        else:
            print(f"Exported {len(summaries)} feature summaries to {out_p}")
        return 0

    raise AssertionError(cmd)  # pragma: no cover


def _run_detect(args: argparse.Namespace, config: Config) -> int:
    from flawbench.detection.pipeline import DetectionPipeline

    pipeline = DetectionPipeline(config=config)
    action = getattr(args, "action", None)

    if action == "list" or getattr(args, "list", False):
        entries = pipeline.list_detections()
        print(f"Stored detections: {len(entries)}")
        for e in entries:
            print(
                f"  Sample: {e['sample_id']} (source: {e['source_id']}) -> "
                f"{e['predictions_count']} prediction(s) [candidates: {e['raw_candidates_count']}]"
            )
        return 0

    if action == "validate" or getattr(args, "validate", False):
        report = pipeline.validate_detections()
        total = len(report)
        valid = sum(1 for p in report.values() if not p)
        print(f"Detection results validation: {valid}/{total} valid files")
        has_problems = False
        for fname, problems in report.items():
            if problems:
                has_problems = True
                print(f"  [FAIL] {fname}:", file=sys.stderr)
                for prob in problems:
                    print(f"    - {prob}", file=sys.stderr)
        return 1 if has_problems else 0

    if getattr(args, "sample", None):
        sample_id = args.sample
        force = getattr(args, "force", False)
        result = pipeline.detect_sample(sample_id, force=force)
        print(
            f"Detection completed for sample '{sample_id}' "
            f"({len(result.predictions)} flaw(s) predicted, "
            f"{result.raw_candidates_count} raw candidates):"
        )
        for i, p in enumerate(result.predictions):
            ref_info = (
                f", ref: [{p.ref_start_sec:.2f}s - {p.ref_end_sec:.2f}s]"
                if p.ref_start_sec is not None
                else ""
            )
            print(
                f"  [{i + 1}] {p.flaw_type}: {p.start_sec:.2f}s - {p.end_sec:.2f}s "
                f"(samples [{p.start_sample}:{p.end_sample}], dur: {p.duration_sec:.2f}s"
                f"{ref_info}) strength={p.evidence_strength:.2f} detector={p.detector_id}"
            )
            for ek, ev in p.evidence.items():
                print(f"      {ek}: {ev}")
        if getattr(args, "export", None):
            export_path = Path(args.export).resolve()
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            print(f"Exported prediction to {export_path}")
        return 0

    if getattr(args, "all", False):
        force = getattr(args, "force", False)
        summary = pipeline.detect_all(force=force)
        print(
            f"Batch detection complete: "
            f"{summary['processed']}/{summary['total']} processed, "
            f"{summary['cached']} cached, {summary['failed']} failed | "
            f"{summary['with_candidates']} with flaws detected, "
            f"{summary['zero_candidates']} zero-flaw results."
        )
        if getattr(args, "export", None):
            export_path = Path(args.export).resolve()
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_data = [r.to_dict() for r in summary["results"]]
            export_path.write_text(json.dumps(export_data, indent=2), encoding="utf-8")
            print(f"Exported batch predictions to {export_path}")
        return 0 if summary["failed"] == 0 else 1

    print(
        "Please specify an action (list, validate) or detection target (--sample, --all). "
        "Use --help for usage.",
        file=sys.stderr,
    )
    return 2


def _run_evaluate(args: argparse.Namespace, config: Config) -> int:
    from flawbench.evaluation.reports import generate_markdown_report
    from flawbench.evaluation.runner import EvaluationRunner
    from flawbench.evaluation.store import EvaluationStore
    from flawbench.evaluation.validator import EvaluationValidator

    paths = ProjectPaths(config)
    cmd = args.evaluate_command

    if cmd == "run":
        split = args.split or config.evaluation.default_split
        runner = EvaluationRunner(config, paths)
        include_controls = False if args.no_controls else None
        report, run_dir = runner.run(
            split=split,
            sample_id=args.sample,
            run_id=args.run_id,
            include_controls=include_controls,
            enable_bootstrap=args.bootstrap,
        )
        agg = report.aggregate
        p_str = f"{agg.micro_precision:.3f}" if agg and agg.micro_precision is not None else "N/A"
        r_str = f"{agg.micro_recall:.3f}" if agg and agg.micro_recall is not None else "N/A"
        f1_str = f"{agg.micro_f1:.3f}" if agg and agg.micro_f1 is not None else "N/A"
        iou_str = f"{agg.mean_iou:.3f}" if agg and agg.mean_iou is not None else "N/A"

        print(f"Evaluation completed: run_id='{report.run_id}' (split='{report.split}')")
        print(
            f"  Cohort: {report.sample_count} samples across "
            f"{report.source_count} sources evaluated"
        )
        if report.is_synthetic_only:
            print("  Provenance: synthetic audio fixtures only (not natural human speech)")
        print(
            f"  Micro-average: Precision={p_str}, Recall={r_str}, F1={f1_str}, Mean IoU={iou_str}"
        )
        print(f"  Artifacts saved to: {run_dir}")
        return 0

    if cmd == "list":
        store = EvaluationStore(paths)
        runs = store.list_runs()
        print(f"Evaluation runs: {len(runs)}")
        for r in runs:
            f1 = f"{r['micro_f1']:.3f}" if r.get("micro_f1") is not None else "N/A"
            iou = f"{r['mean_iou']:.3f}" if r.get("mean_iou") is not None else "N/A"
            syn = " [synthetic]" if r.get("is_synthetic_only") else ""
            print(
                f"  {r['run_id']}\t(split: {r.get('split')}, "
                f"samples: {r.get('sample_count')}, "
                f"F1: {f1}, IoU: {iou}){syn}"
            )
        return 0

    if cmd == "report":
        store = EvaluationStore(paths)
        report = store.load_evaluation(args.run)
        run_dir = store.get_run_dir(args.run)
        md_file = run_dir / "report.md"
        if md_file.exists():
            md_content = md_file.read_text(encoding="utf-8")
        else:
            md_content = generate_markdown_report(report)

        if args.output:
            out_p = Path(args.output).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(md_content, encoding="utf-8")
            print(f"Report exported to: {out_p}")
        else:
            print(md_content)
        return 0

    if cmd == "validate":
        store = EvaluationStore(paths)
        run_dir = store.get_run_dir(args.run)
        validator = EvaluationValidator()
        report, warnings = validator.validate_run_directory(run_dir)
        total_matches = sum(len(s.matches) for s in report.sample_evaluations)
        print(
            f"Evaluation run '{args.run}' is VALID "
            f"({total_matches} matches, {len(report.per_category)} categories)."
        )
        for w in warnings:
            print(f"  Warning: {w}")
        return 0

    raise AssertionError(cmd)  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        config = load_config(getattr(args, "config", None))
    except ConfigurationError as exc:
        print(f"Configuration error: {exc.public_message(expose_details=True)}", file=sys.stderr)
        return 2
    try:
        if args.command == "dataset":
            return _run_dataset(args, config)
        if args.command == "features":
            return _run_features(args, config)
        if args.command == "detect":
            return _run_detect(args, config)
        if args.command == "evaluate":
            return _run_evaluate(args, config)
        return _run_phase1(args, config)
    except FlawBenchError as exc:
        print(
            f"Error: {exc.public_message(expose_details=config.diagnostics.expose_error_details)}",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
