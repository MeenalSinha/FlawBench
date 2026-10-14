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
            print(f"  DTW cost: {cmp_res.dtw_cost:.3f} (normalized: {cmp_res.normalized_dtw_cost:.4f})")
            print(f"  Path length: {cmp_res.path_length} points")
            print(f"  Mean delta RMS: {cmp_res.global_metrics.get('mean_delta_rms_db', 0):.2f} dB")
            print(f"  Candidate flaw regions: {len(cmp_res.candidate_regions)}")
            for r in cmp_res.candidate_regions:
                print(f"    - [{r.get('flaw_hypothesis')}] var [{r.get('var_start_sec')} - {r.get('var_end_sec')})s")
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
