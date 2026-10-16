#!/bin/bash
# Usage: make_bench.sh <run_dir>   (synthetic functional benchmark, built only through the CLI)
set -e
source /home/claude/venv/bin/activate
RUN="$1"; rm -rf "$RUN"; mkdir -p "$RUN/data/reference"
export FLAWBENCH_PROJECT_ROOT="$RUN"
python - "$RUN" <<'PY'
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "/home/claude/fb/FlawBench-main")
from tests.conftest import synthetic_signal, SYN_RATE
for seed in range(4):
    sf.write(f"{sys.argv[1]}/data/reference/syn_{seed}.wav", synthetic_signal(6.0, seed), SYN_RATE, subtype="PCM_16")
PY
cat > "$RUN/plan.yaml" <<'YAML'
plan_version: 1
name: p7-synthetic-functional
seed: 7
variants:
  - {source: "*", flaw: pause_insertion, severity: medium, placement: {mode: explicit, at_sec: 3.0}}
  - {source: "*", flaw: tempo_change,   severity: medium, placement: {mode: explicit, start_sec: 2.0}}
  - {source: "*", flaw: energy_reduction, severity: high, placement: {mode: explicit, start_sec: 2.0}}
YAML
python -m flawbench init >/dev/null
for s in 0 1 2 3; do
  python -m flawbench dataset register-reference "$RUN/data/reference/syn_$s.wav" --transcript-text "synthetic fixture $s" \
    --source-type self_recorded --attribution "synthetic generator" --license "CC0-1.0 (synthetic)" --rights-confirmed --redistribution permitted --source-id "syn_$s" >/dev/null
done
python -m flawbench dataset generate --plan "$RUN/plan.yaml" | tail -3
python -m flawbench dataset split | tail -3
python -m flawbench dataset validate | tail -3
python -m flawbench features extract --all 2>&1 | tail -3
python -m flawbench features validate | tail -2
python -m flawbench detect --all 2>&1 | tail -3
python -m flawbench detect --validate | tail -2
python -m flawbench evaluate run --split all --run-id "${2:-run}" 2>&1 | tail -25
python -m flawbench evaluate validate | tail -3
