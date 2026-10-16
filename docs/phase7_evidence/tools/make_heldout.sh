#!/bin/bash
# Usage: make_heldout.sh <run_dir> <run_id> <code_src_dir>
# Pre-registered held-out SYNTHETIC cohort. Parameters fixed before any detector was run on it.
set -e
source /home/claude/venv/bin/activate
RUN="$1"; RID="$2"; export PYTHONPATH="$3"
rm -rf "$RUN"; mkdir -p "$RUN/data/reference"; export FLAWBENCH_PROJECT_ROOT="$RUN"
python - "$RUN" <<'PY'
import sys, soundfile as sf
sys.path.insert(0, "/home/claude/fb/FlawBench-main")
from tests.conftest import synthetic_signal, SYN_RATE
for seed in range(10, 18):
    sf.write(f"{sys.argv[1]}/data/reference/syn_{seed}.wav", synthetic_signal(8.0, seed), SYN_RATE, subtype="PCM_16")
PY
python - "$RUN" <<'PY'
import sys, yaml
pause_dur = [0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0]; pause_at = [1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 5.0, 1.0]
rates     = [0.85, 1.3, 1.5, 0.6, 0.85, 1.3, 1.5, 0.6]; t_start = [1.0, 2.5, 3.0, 1.5, 4.0, 1.0, 2.0, 3.5]
gains     = [-6.0, -12.0, -24.0, -6.0, -12.0, -24.0, -6.0, -12.0]; e_start = [3.0, 1.0, 4.0, 2.0, 1.5, 3.5, 4.5, 0.5]
v = []
for k, seed in enumerate(range(10, 18)):
    src = f"syn_{seed}"
    v.append({"source": src, "flaw": "pause_insertion", "severity": "medium", "placement": {"mode": "explicit", "at_sec": pause_at[k]}, "parameters": {"duration_sec": pause_dur[k]}})
    v.append({"source": src, "flaw": "tempo_change", "severity": "medium", "placement": {"mode": "explicit", "start_sec": t_start[k]}, "parameters": {"rate": rates[k], "segment_sec": 2.0}})
    v.append({"source": src, "flaw": "energy_reduction", "severity": "medium", "placement": {"mode": "explicit", "start_sec": e_start[k]}, "parameters": {"gain_db": gains[k], "segment_sec": 2.0}})
yaml.safe_dump({"plan_version": 1, "name": "p7-heldout-synthetic", "seed": 11, "variants": v}, open(sys.argv[1] + "/plan.yaml", "w"))
PY
cp "$RUN/plan.yaml" /home/claude/p7/heldout_plan.yaml
python -m flawbench init >/dev/null
for s in 10 11 12 13 14 15 16 17; do
  python -m flawbench dataset register-reference "$RUN/data/reference/syn_$s.wav" --transcript-text "synthetic heldout $s" \
    --source-type self_recorded --attribution "synthetic generator" --license "CC0-1.0 (synthetic)" --rights-confirmed --redistribution permitted --source-id "syn_$s" >/dev/null
done
python -m flawbench dataset generate --plan "$RUN/plan.yaml" | tail -1
python -m flawbench dataset split 2>&1 | tail -1
python -m flawbench dataset validate | tail -1
python -m flawbench features extract --all 2>&1 | tail -1
python -m flawbench detect --all 2>&1 | tail -1
python -m flawbench evaluate run --split all --run-id "$RID" 2>&1 | grep -E "Micro|Cohort"
python -m flawbench evaluate validate --run "$RID" | tail -1
python -m flawbench evaluate report --run "$RID" | sed -n '/Per-Category/,/Localization/p' | head -12
