import time, sys, numpy as np
sys.path.insert(0, "/home/claude/fb/FlawBench-main")
from tests.conftest import synthetic_signal, SYN_RATE
from flawbench.config import load_config
from flawbench.features.extractor import extract_features
from flawbench.features.comparison import compare_features
from flawbench.dataset.injectors import insert_pause
from flawbench.detection.engine import DetectionEngine
from flawbench.detection.inputs import build_detection_input_from_features
cfg = load_config()
for dur in (6.0, 30.0, 120.0):
    ref = synthetic_signal(dur, 1); var = insert_pause(ref[np.newaxis,:], at_sample=int(dur/2*SYN_RATE), silence_samples=SYN_RATE).samples[0]
    t=time.perf_counter(); rf = extract_features(ref, sample_rate=SYN_RATE, config=cfg); vf = extract_features(var, sample_rate=SYN_RATE, config=cfg); t1=time.perf_counter()-t
    t=time.perf_counter(); inp = build_detection_input_from_features(rf, vf, settings=cfg.detection); t2=time.perf_counter()-t
    t=time.perf_counter(); DetectionEngine(cfg.detection).detect(inp); t3=time.perf_counter()-t
    print(f"{dur:6.0f}s audio: features(ref+var) {t1:6.2f}s | comparison/DTW {t2:6.2f}s | detection {t3:5.2f}s")
