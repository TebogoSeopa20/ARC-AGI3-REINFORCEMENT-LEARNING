#!/usr/bin/env bash
# Ablations: each differs from its imp2 config by exactly one component. Evaluated separately per algorithm.
set -euo pipefail
SPLIT="${1:-dev}"
for CFG in configs/ablations/*.yaml; do
  echo "=== $CFG on $SPLIT ==="
  python scripts/evaluate.py --config "$CFG" --split "$SPLIT"
done
python scripts/aggregate.py
