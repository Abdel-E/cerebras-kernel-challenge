#!/usr/bin/env bash
set -euo pipefail

mkdir -p out out/stats

compile_and_run() {
  local case_name="$1"
  local P="$2"
  local d_dim="$3"
  local rows_per_pe="$4"
  local K="$5"

  echo "=== ${case_name} (P=${P}, d=${d_dim}, rows_per_pe=${rows_per_pe}, K=${K}) ==="
  cslc --arch=wse2 layout.csl \
    --fabric-dims=11,6 \
    --fabric-offsets=4,1 \
    --params="P:${P},d_dim:${d_dim},rows_per_pe:${rows_per_pe},K:${K}" \
    --memcpy --channels=1 \
    -o "out/${case_name}"

  cs_python run.py --name "out/${case_name}" --case "${case_name}" --stats-dir out/stats
}

compile_and_run baseline   4 32 128 16
compile_and_run k_eq_1     2 32 256 1
compile_and_run k_large    2 16 256 256
compile_and_run uneven     4 32 64  16
compile_and_run all_equal  2 16 256 16
compile_and_run duplicates 2 16 256 8

python3 - <<'PY'
import json
from pathlib import Path

stats_dir = Path("out/stats")
rows = []
for path in sorted(stats_dir.glob("*.json")):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows.append((path.stem, int(data.get("cycle_count", -1)), float(data.get("sim_time", -1.0))))

print("\nCase performance summary:")
for case, cycles, sim_time in rows:
    print(f"  {case:10s}  cycles={cycles:9d}  sim_time={sim_time:9.3f}s")
PY
