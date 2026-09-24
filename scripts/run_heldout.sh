#!/usr/bin/env bash
# One-time held-out evaluation (10 games, 3 seeds) of every version: bash scripts/run_heldout.sh [dqn|ppo|random|all]
# Nothing is tuned after this. Existing seeds are skipped, so a stopped run can be restarted with the same command.
set -euo pipefail
PART="${1:-all}"
run() { python scripts/evaluate.py --config "$1" --split heldout; }
if [[ "$PART" == "random" || "$PART" == "all" ]]; then
  for c in random random_imp4_actions random_imp5_graph; do run configs/$c.yaml; done
fi
if [[ "$PART" == "ppo" || "$PART" == "all" ]]; then
  for c in ppo_baseline ppo_imp1_explore ppo_imp2_clockmask ppo_imp4_efficient ppo_imp5_graph; do run configs/$c.yaml; done
  python scripts/pretrain.py --config configs/ppo_imp3_pretrain_final.yaml
  python scripts/evaluate.py --config configs/ppo_imp3_pretrain_final.yaml
fi
if [[ "$PART" == "dqn" || "$PART" == "all" ]]; then
  for c in dqn_baseline dqn_imp1_explore dqn_imp2_clockmask dqn_imp4_efficient dqn_imp5_graph; do run configs/$c.yaml; done
  python scripts/pretrain.py --config configs/dqn_imp3_pretrain_final.yaml
  python scripts/evaluate.py --config configs/dqn_imp3_pretrain_final.yaml
fi
python scripts/aggregate.py
