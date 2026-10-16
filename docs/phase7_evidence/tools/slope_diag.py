"""Diagnostic only. Annotations are used here AFTER the fact to read the injected interval for analysis."""
import json, sys, numpy as np
from flawbench.config import load_config
from flawbench.features.pipeline import FeaturePipeline
from flawbench.detection.inputs import build_detection_input
cfg = load_config()
fp = FeaturePipeline(cfg)
ann = {}
for line in open(cfg.paths.manifest_file if hasattr(cfg,'paths') else "data/metadata/manifest.jsonl"):
    r = json.loads(line)
    if r.get("record_type") == "sample": ann[r["sample_id"]] = r["annotation"]
for sid, a in ann.items():
    if a["flaw_type"] != "tempo_change": continue
    inp = build_detection_input(sid, fp)
    c = inp.comparison
    iv = a["output_interval"]
    m = (c.var_times_sec >= iv["start_sec"]) & (c.var_times_sec < iv["end_sec"])
    s = c.local_slope[m]
    print(sid[:12], inp.source_id, "path pts in segment:", int(m.sum()),
          "| slope median %.2f mean %.2f  frac(|1/slope-1|>0.15)=%.2f" % (np.median(s), s.mean(), np.mean(np.abs(1/s-1) > 0.15)),
          "| frac slope>=9: %.2f" % np.mean(s >= 9),
          "| unique slopes:", np.unique(np.round(s,2))[:8])
    # whole-segment slope from endpoints (what a longer window would give)
    i0, i1 = np.where(m)[0][[0, -1]]
    d_var = c.warp_path[i1,1]-c.warp_path[i0,1]; d_ref = c.warp_path[i1,0]-c.warp_path[i0,0]
    print("     endpoint slope over injected segment: d_var/d_ref = %d/%d = %.3f (expected %.3f)" % (d_var, d_ref, d_var/max(d_ref,1), 1/a["parameters"]["rate"]))
