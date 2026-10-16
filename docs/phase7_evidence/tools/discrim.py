"""Diagnostic only (dev cohort). Looks at what separates inserted silence from strong attenuation."""
import json, numpy as np
from flawbench.config import load_config
from flawbench.features.pipeline import FeaturePipeline
from flawbench.detection.inputs import build_detection_input
cfg = load_config(); fp = FeaturePipeline(cfg)
ann = {}
for line in open("data/metadata/manifest.jsonl"):
    r = json.loads(line)
    if r.get("record_type") == "sample": ann[r["sample_id"]] = r["annotation"]
first = True
for sid, a in sorted(ann.items(), key=lambda kv: kv[1]["flaw_type"]):
    if a["flaw_type"] == "tempo_change": continue
    inp = build_detection_input(sid, fp); v = inp.var_features; c = inp.comparison
    if first: print([n for n in dir(v) if not n.startswith("_") and not callable(getattr(v, n))]); first = False
    iv = a["output_interval"]; t = v.center_times_sec
    m = (t >= iv["start_sec"]+0.1) & (t < iv["end_sec"]-0.1)
    rms = v.rms_db[m]
    mm = (c.var_times_sec >= iv["start_sec"]+0.1) & (c.var_times_sec < iv["end_sec"]-0.1)
    d = c.delta_rms_db[mm]
    f0 = v.f0_hz[m]
    print(f'{a["flaw_type"]:17s} {inp.source_id} var_rms mean {rms.mean():7.1f} std {rms.std():5.2f} | f0 finite frac {np.isfinite(f0).mean():.2f} | zcr mean {np.mean(v.zcr[m]):.3f}'
          f' | delta mean {d.mean():7.1f} std {d.std():5.2f} | MFCC0 std {v.mfcc[m,0].std():.2f} | mfcc1-12 std mean {v.mfcc[m,1:].std(axis=0).mean():.2f}')
