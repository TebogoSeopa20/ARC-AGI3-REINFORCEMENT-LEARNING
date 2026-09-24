#!/usr/bin/env bash
# Improvement 3 development check: pretrain on dev_a (3 seeds), evaluate on dev_b, plus the DQN schedule ablation.
# Improvement 2 on dev_b needs no rerun: the dev results already contain those 5 games at the same seeds.
set -euo pipefail
for A in dqn ppo; do
  python scripts/pretrain.py --config configs/${A}_imp3_pretrain.yaml
  python scripts/evaluate.py --config configs/${A}_imp3_pretrain.yaml
done
python scripts/evaluate.py --config configs/ablations/dqn_scratch_loweps.yaml
python scripts/aggregate.py
python scripts/compare_splits.py --split dev_b
