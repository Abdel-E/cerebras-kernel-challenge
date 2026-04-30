#!/usr/bin/env bash
set -euo pipefail

# Baseline compile + run entrypoint expected by the challenge packet.
# Fill in layout.csl and run.py first, then use this script as your quick loop.

mkdir -p out
mkdir -p out/stats

cslc --arch=wse2 layout.csl \
  --fabric-dims=11,6 \
  --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:128,K:16 \
  --memcpy --channels=1 \
  -o out/baseline

cs_python run.py --name out/baseline --case baseline --stats-dir out/stats
