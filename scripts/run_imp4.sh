#!/usr/bin/env bash
# Improvement 4 on the full dev split (3 seeds), then its ablations: bash scripts/run_imp4.sh [main|ablations|all]
set -euo pipefail
PART="${1:-main}"
if [[ "$PART" == "main" || "$PART" == "all" ]]; then
  for A in dqn ppo; do python scripts/evaluate.py --config configs/${A}_imp4_efficient.yaml; done
fi
if [[ "$PART" == "ablations" || "$PART" == "all" ]]; then
  python scripts/evaluate.py --config configs/random_imp4_actions.yaml
  for A in dqn ppo; do
    for X in no_effect no_prune no_objclick; do
      python scripts/evaluate.py --config configs/ablations/${A}_imp4_${X}.yaml
    done
  done
fi
python scripts/aggregate.py
