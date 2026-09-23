#!/usr/bin/env bash
# Main iterative study: random reference + 3-4 versions per algorithm, >=3 seeds, dev then held-out.
# Held-out runs happen ONCE per final config, after development decisions are frozen (docs/PROTOCOL.md).
set -euo pipefail
SPLIT="${1:-dev}"
CONFIGS=(configs/random.yaml
         configs/dqn_baseline.yaml configs/dqn_imp1_explore.yaml configs/dqn_imp2_memory.yaml
         configs/ppo_baseline.yaml configs/ppo_imp1_explore.yaml configs/ppo_imp2_memory.yaml)
[[ "${WITH_IMP3:-0}" == "1" ]] && CONFIGS+=(configs/dqn_imp3_objclick.yaml configs/ppo_imp3_objclick.yaml)
for CFG in "${CONFIGS[@]}"; do
  echo "=== $CFG on $SPLIT ==="
  python scripts/evaluate.py --config "$CFG" --split "$SPLIT"
done
python scripts/aggregate.py
