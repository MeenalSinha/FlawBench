"""Analysis-only: joins saved predictions with annotations AFTER inference (never fed to detectors)."""
import json, sys, glob, pathlib
root = pathlib.Path(sys.argv[1])
gt = {}
for line in open(root/"data/metadata/manifest.jsonl"):
    r = json.loads(line)
    if r.get("record_type") == "sample":
        gt[r["sample_id"]] = r["annotation"]
for f in sorted(glob.glob(str(root/"data/detections/*_detection.json"))):
    d = json.load(open(f)); a = gt[d["sample_id"]]
    iv = a["output_interval"]
    p0 = a["parameters"]
    print(f'{d["sample_id"][:12]} {d["source_id"]} GT {a["flaw_type"]:17s} [{iv["start_sec"]:.2f},{iv["end_sec"]:.2f}] {p0}')
    for p in d["predictions"]:
        ev = {k: v for k, v in p["evidence"].items() if k in ("estimated_relative_rate","mean_delta_rms_db","mean_variant_rms_db","ref_frozen_fraction","direction")}
        print(f'      PRED {p["flaw_type"]:17s} [{p["start_sec"]:.2f},{p["end_sec"]:.2f}] s={p["evidence_strength"]} {ev}')
