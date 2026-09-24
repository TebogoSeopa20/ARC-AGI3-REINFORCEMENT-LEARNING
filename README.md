# DQN vs PPO on ARC-AGI-3

Wits COMS4061A/COMS7071A Reinforcement Learning assignment (ARC Prize 2026).
Kaggle team **General** — Tebogo Seopa (2563912), Mandisa Mashinini (2589970), Onkabetse Monamodi (2854004).
Moodle deadline: 27 October 2026, 23:59 SAST.

**Research question.** Do exploration shaping, short-horizon memory and an abstracted click action
space improve a DQN and a PPO agent's interaction efficiency and held-out generalisation on ARC-AGI-3,
relative to their own baselines and a random policy — and do improvements transfer across the two?

| Version | DQN config | PPO config | Change |
|---|---|---|---|
| Reference | `random.yaml` | `random.yaml` | Uniform over legal abstracted actions |
| Baseline | `dqn_baseline.yaml` | `ppo_baseline.yaml` | Sparse level reward, 8×8 click grid, single frame |
| Improvement 1 | `dqn_imp1_explore.yaml` | `ppo_imp1_explore.yaml` | Count-based novelty + frame-change shaping (Bellemare et al., 2016) |
| Improvement 2 | `dqn_imp2_clockmask.yaml` | `ppo_imp2_clockmask.yaml` | Novelty/change computed with clock-like pixels masked (ClockMask) |
| Improvement 3 | `dqn_imp3_pretrain.yaml` | `ppo_imp3_pretrain.yaml` | Cross-game pretraining on dev_a, online learning on unseen games (dev_b check; `*_final` = all dev → held-out) |
| Optional | `dqn_opt_memory.yaml` | `ppo_opt_memory.yaml` | + 4-frame stack (Hausknecht & Stone, 2015) |
| Optional | `dqn_opt_objclick.yaml` | `ppo_opt_objclick.yaml` | + 16×16 object-aware click cells |

The improvement order is a hypothesis, not a commitment: the brief requires each step to be motivated
by the previous stage's evidence. Reorder the configs if baseline failure modes point elsewhere, and log why.

---

## Repository layout

```
agent/my_agent.py        Kaggle submission agent (MyAgent), wraps the online controller
configs/                 One YAML per evaluated version; ablations/ differ by exactly one component
scripts/                 Entry points: splits, smoke test, evaluate, pretrain, aggregate, notebook build
src/arcrl/
  env/                   Frame processing, action abstraction, Gymnasium adapter, offline toy game
  models/                CNN with spatial click head (Q-values or policy logits + value)
  agents/                DQN, PPO, random, reward shaping, online controller
  evaluation/            Game runner, official scorecard extraction, seed aggregation + plots
tests/                   CPU-only unit tests (run before every experiment)
docs/                    Runbook, protocol, game disclosure, experiment log, Kaggle evidence, contributions
report/                  RLC-template report skeleton, bibliography, Paper Track write-up
notebooks/               experiments.ipynb (Kaggle T4 runs), kernel metadata, generated submission.ipynb
outputs/                 results/, checkpoints/, figures/ (results and checkpoints gitignored)
```

## Setup

```bash
make setup          # Python 3.12 venv, pip install -e ".[dev]", clones ARC-AGI-3-Agents framework
make test           # 14 CPU tests
make smoke-toy      # random/DQN/PPO end-to-end on the offline toy game
make smoke          # same on a real game (downloads it once into environment_files/)
```

With conda on the Intel Mac: `conda create -n arcrl python=3.12 && conda activate arcrl && pip install -e ".[dev]"`,
then `PYTHON=python make setup` or run the scripts directly. The Intel Mac pins torch 2.2 (last x86_64 macOS wheel);
full runs go on Kaggle T4.

## Experiment pipeline

See `docs/RUNBOOK.md` for which phase runs where (Mac vs Kaggle) and who runs what.

```bash
# Phase 1 — freeze the protocol (once)
python scripts/list_games.py
python scripts/make_splits.py              # writes configs/splits.yaml; commit it, never change it

# Phase 2 — baselines + reference, 3 seeds each, dev split
python scripts/evaluate.py --config configs/random.yaml
python scripts/evaluate.py --config configs/dqn_baseline.yaml
python scripts/evaluate.py --config configs/ppo_baseline.yaml

# Phases 3–4 — improvements and ablations
bash scripts/run_experiments.sh dev
bash scripts/run_ablations.sh dev

# Held-out generalisation — final configs only, run once
bash scripts/run_experiments.sh heldout

# Tables + curves -> outputs/figures/{summary.md,summary.tex,curve_*.png}
python scripts/aggregate.py
```

Every seed writes `outputs/results/<run>/<split>_seed<k>/games.jsonl` plus `run_meta.json` capturing
config, git commit, package versions, hardware and scorecard id. Log each run in `docs/EXPERIMENT_LOG.md`.

## Kaggle submission

```bash
mkdir -p .kaggle && echo "KGAT_..." > .kaggle/access_token && chmod 600 .kaggle/access_token
# point configs/submission.yaml at the final agent, then:
make play-local GAME=ls20     # same agent, through the framework loop
make submit                   # builds notebooks/submission.ipynb and pushes it
```

Then click **Submit to Competition** on the kernel page. See `docs/KAGGLE_SUBMISSION.md` for the
evidence the Moodle ZIP must contain.

## Design notes

- **Online learning.** Evaluation games are unseen, so each agent learns during play. Weights, replay/rollout
  and novelty counts persist across levels and GAME_OVER resets within a game and are reset between games.
  `scripts/pretrain.py` provides optional offline pretraining on dev games only (`init_checkpoint`).
- **No Stable-Baselines3.** SB3 owns the environment loop; the Kaggle framework calls `choose_action()` one
  step at a time. The learners are plain PyTorch so the same code runs locally and on Kaggle.
  `src/arcrl/env/arc_env.py` still provides a documented Gymnasium adapter.
- **Scores.** Reported ARC-AGI-3 scores come only from `arc_agi`'s scorecard. Shaped training return is
  logged separately as `shaped_return` and never presented as the score.
- **Reuse.** Kaggle plumbing (`build_notebook.py`, `play_local.py`, `slim_framework.py`, Makefile targets)
  is adapted from arcprize/ARC-AGI-3-Kaggle-Starter; the starter's per-game action branch was removed.
  All agent, learner, shaping and evaluation code is ours.
