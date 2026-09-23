# Runbook — where each phase runs

Today: 23 Sept. Phase 1 (due 21 Sept) is late; Phase 2 baselines are due 5 Oct.

## Phase 1 — protocol (Mac, CPU, needs internet) · target 24 Sept
```bash
make setup
make test
make download-games          # caches every public game into environment_files/
make games                   # note count, ids, level counts -> docs/GAMES.md
python scripts/make_splits.py --force   # ONCE; then commit configs/splits.yaml
make smoke GAME=<a dev game> STEPS=200  # random/DQN/PPO on a real game
git add -A && git commit -m "Freeze protocol: splits, games" && git push
```
Fill in docs/PROTOCOL.md (date, commit, package versions from any run_meta.json).

## Phase 1b — size the budget (Kaggle T4) · 25 Sept
```bash
bash scripts/package_kaggle_dataset.sh create   # first time; later: make dataset-push
```
On Kaggle: new notebook -> File -> Import `notebooks/experiments.ipynb`, attach dataset `arcrl-code` and the
ARC-AGI-3 competition, GPU T4, internet off. Run with `CONFIGS = []` and `RUN_PROBE = True`.
If the projected study exceeds ~60 GPU-h (two weeks of quota), lower `max_actions_per_game` in
`configs/base.yaml` for ALL configs, log it in the PROTOCOL change table, re-push the dataset.

## Phase 2 — baselines · 26 Sept – 5 Oct
Split the configs across the three accounts so quota runs in parallel (same dataset version = same commit):
| Member | CONFIGS |
|---|---|
| Tebogo | `configs/dqn_baseline.yaml` |
| Mandisa | `configs/ppo_baseline.yaml` |
| Onkabetse | `configs/random.yaml`, then the Kaggle submission pipeline (`make submit` with the baseline) |

Download each `outputs.zip`, unzip into the repo's `outputs/`, run `make aggregate`, log rows in
docs/EXPERIMENT_LOG.md. Also push one early Kaggle submission of the DQN baseline to confirm the pipeline scores.

## Phase 2 decision point · 5 Oct
From baseline curves and per-game tables, write the "observed limitation -> hypothesis" block for
improvement 1 of each algorithm in docs/EXPERIMENT_LOG.md BEFORE running imp1. Expected (to be checked, not assumed):
- near-zero levels for both baselines, similar to random -> exploration (imp1) is the right first move;
- many no-op actions (frame unchanged) -> frame-change bonus;
- clicks on empty cells dominating -> object-aware clicks (imp3) may deserve to go earlier.
