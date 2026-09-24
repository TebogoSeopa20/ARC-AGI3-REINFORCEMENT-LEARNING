#!/usr/bin/env bash
# Main iterative study: random reference + 3-4 versions per algorithm, >=3 seeds, dev then held-out.
# Held-out runs happen ONCE per final config, after development decisions are frozen (docs/PROTOCOL.md).
set -euo pipefail
SPLIT="${1:-dev}"
CONFIGS=(configs/random.yaml
         configs/dqn_baseline.yaml configs/dqn_imp1_explore.yaml configs/dqn_imp2_clockmask.yaml
         configs/ppo_baseline.yaml configs/ppo_imp1_explore.yaml configs/ppo_imp2_clockmask.yaml)
[[ "${WITH_OPT:-0}" == "1" ]] && CONFIGS+=(configs/dqn_opt_memory.yaml configs/ppo_opt_memory.yaml
                                            configs/dqn_opt_objclick.yaml configs/ppo_opt_objclick.yaml)
# Improvement 3 uses pretraining and its own splits: see scripts/run_imp3.sh.
for CFG in "${CONFIGS[@]}"; do
  echo "=== $CFG on $SPLIT ==="
  python scripts/evaluate.py --config "$CFG" --split "$SPLIT"
done
python scripts/aggregate.py
