"""Experiment only: does excluding MFCC c0 (energy) from the DTW cost remove level-step alignment artifacts?"""
import json, numpy as np, librosa
import flawbench.features.comparison as cmp
from flawbench.config import load_config
from flawbench.features.pipeline import FeaturePipeline
from flawbench.detection.inputs import build_detection_input
from flawbench.detection.engine import DetectionEngine
cfg = load_config(); fp = FeaturePipeline(cfg); eng = DetectionEngine(cfg.detection)
orig = librosa.sequence.dtw
def patched(X=None, Y=None, **kw):
    kw["metric"] = "euclidean"
    return orig(X=X[1:], Y=Y[1:], **kw)
ann = {}
for line in open("data/metadata/manifest.jsonl"):
    r = json.loads(line)
    if r.get("record_type") == "sample": ann[r["sample_id"]] = r["annotation"]
for label, fn in (("with c0 (current)", orig), ("without c0", patched)):
    cmp.librosa.sequence.dtw = fn
    tot = {}
    for sid, a in sorted(ann.items(), key=lambda kv: kv[1]["flaw_type"]):
        res = eng.detect(build_detection_input(sid, fp))
        kinds = sorted(p.flaw_type for p in res.predictions)
        tot[(a["flaw_type"], tuple(kinds))] = tot.get((a["flaw_type"], tuple(kinds)), 0) + 1
    print(label)
    for k, n in sorted(tot.items()): print("   GT", k[0], "-> predicted", k[1], "x", n)
cmp.librosa.sequence.dtw = orig
