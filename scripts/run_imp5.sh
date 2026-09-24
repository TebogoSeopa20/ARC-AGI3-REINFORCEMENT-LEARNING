#!/usr/bin/env bash
# Improvement 5 on the dev split (3 seeds): bash scripts/run_imp5.sh [main|ablations|all]
# main: DQN, PPO and the random-with-graph control. ablations: each algorithm without the graph.
set -euo pipefail
PART="${1:-main}"
if [[ "$PART" == "main" || "$PART" == "all" ]]; then
  for c in random_imp5_graph ppo_imp5_graph dqn_imp5_graph; do python scripts/evaluate.py --config configs/$c.yaml; done
fi
if [[ "$PART" == "ablations" || "$PART" == "all" ]]; then
  for A in ppo dqn; do python scripts/evaluate.py --config configs/ablations/${A}_imp5_no_graph.yaml; done
fi
python scripts/aggregate.py
