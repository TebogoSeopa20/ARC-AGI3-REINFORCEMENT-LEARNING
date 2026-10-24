# Learning to Act Before Knowing the Rules: DQN and PPO on ARC-AGI-3

This is our group project for COMS4061A/COMS7071A Reinforcement Learning at Wits, and our entry to ARC Prize 2026
as team **DevRift**: Tebogo Seopa (2563912), Mandisa Mashinini (2589970) and Onkabetse Monamodi (2854004).

ARC-AGI-3 drops an agent into games it has never seen, with no instructions. It has to work out the rules and the
goal by playing. We wanted to know how far two standard reinforcement learners get when they have to learn inside
each game, from scratch, and what actually helps them.

## What we asked

When DQN (a value-based method from the course) and PPO (a policy-gradient method we learned for this project) must
learn online inside each unseen game, what holds them back, and which changes to exploration, action selection and
memory of the game make them more efficient and better at games they were never tuned on?

## What we found

Both baselines played no better than random. Looking closer, three things were going wrong. In most games no level
was ever completed, so the agents never saw a single reward. Most clicks landed on empty background. And many games
have a move counter that changes a few pixels on every action, so every frame looked new to the agent.

We tried five improvements, each one prompted by what the previous version got wrong:

| Version | What we changed | What happened |
|---|---|---|
| Baseline | DQN and PPO with a shared CNN, 6 simple actions plus an 8×8 click grid | Same as random play |
| I1 | Bonuses for reaching new frames and for actions that change the frame | More exploration, no more levels |
| I2 | ClockMask: learns which pixels act like a move counter and ignores them | Exploration improved where intended, levels within noise |
| I3 | Pretraining on other games before playing a new one | Exploration transferred, completing levels did not |
| I4 | An effect head that predicts which actions do something, pruning of actions seen to do nothing, clicks aimed at objects | **The one change that generalised:** held-out levels went from 0.33 to 3.33 (DQN) and 2.00 (PPO) |
| I5 | A state graph that returns to unexplored states after each game over (Go-Explore style) | Best on development games, but the gain came from one game and did not carry over |

Our Kaggle submission is DQN with I4. It scored 0.12 on the public leaderboard. The full story, with ablations,
learning curves and replayed games, is in the report (`report/main.tex`).

## How the repository is organised

```
agent/my_agent.py      The agent Kaggle runs (MyAgent), a thin wrapper around our controller
configs/               One YAML file per version we evaluated; configs/ablations/ remove one component each
src/arcrl/
  env/                 Frame processing and the action abstraction (clicks, object-aware clicks)
  models/              The shared CNN (Q-values or policy logits, value head, effect head)
  agents/              DQN, PPO, random policy, reward shaping, ClockMask, state graph, online controller
  evaluation/          Plays a game under a fixed budget, reads the official score, aggregates results
scripts/               Everything you run: splits, evaluation, pretraining, statistics, figures, notebook build
tests/                 24 CPU tests; we ran them before every experiment
docs/                  Protocol, game list, experiment log, Kaggle evidence and member contributions
report/                The report (RLC 2026 template), bibliography, figures and Paper Track write-up
notebooks/             Kaggle notebooks and their metadata
```

## Running it

You need Python 3.12. On Intel Macs the last available PyTorch is 2.2, which is what we used.

```bash
make setup        # virtual environment, the package, the Kaggle CLI and the ARC-AGI-3 agent framework
make test         # unit tests
make smoke-toy    # random, DQN and PPO end to end on a small offline toy game
make smoke        # the same on one real game
```

### Reproducing the experiments

The development/held-out split was fixed once, before any improvement, and is stored in `configs/splits.yaml`.
Everything was designed on the 15 development games; the 10 held-out games were only run once per version at the end.

```bash
python scripts/evaluate.py --config configs/random.yaml
python scripts/evaluate.py --config configs/dqn_baseline.yaml
python scripts/evaluate.py --config configs/ppo_baseline.yaml
bash scripts/run_experiments.sh dev
bash scripts/run_ablations.sh dev
bash scripts/run_imp3.sh
bash scripts/run_imp4.sh
bash scripts/run_imp5.sh
bash scripts/run_heldout.sh
python scripts/aggregate.py
python scripts/stats.py --split dev
python scripts/stats.py --split heldout
python scripts/report_figures.py
```

Each run uses 1000 actions per game and seeds 0, 1 and 2. Every seed writes its results and a `run_meta.json`
(config, git commit, package versions, hardware) under `outputs/results/`. On our 2016 MacBook one seed over the
development games took about 1 minute for random play, 21 to 29 minutes for DQN and 8 to 11 minutes for PPO.
Runs are deterministic on the same machine, so re-running reproduces the logged numbers exactly.
`docs/EXPERIMENT_LOG.md` records every run and every decision we made along the way, including the ones that
did not work.

To replay a logged run and see what the agent actually did:

```bash
python scripts/trajectories.py --config configs/dqn_imp4_efficient.yaml --split heldout --seed 0 --game vc33 --mode levels
```

### Submitting to Kaggle

```bash
make play-local GAME=ls20   # runs the submission agent through the official framework locally
make submit                 # builds notebooks/submission.ipynb and pushes it to Kaggle
```

Then press **Submit to Competition** on the notebook page. You need your own Kaggle token in
`.kaggle/access_token`; it is git-ignored and never ends up in the notebook.

## A few design choices worth knowing

- **The agents learn while they play.** Weights, replay memory and visit counts carry over between levels and lives
  within a game, and are reset when a new game starts. Nothing game-specific is hard-coded.
- **Shaped rewards are for training only.** Every score we report comes from the official `arc-agi` scorecard,
  never from our shaped training reward.
- **Plain PyTorch, no Stable-Baselines3.** The Kaggle framework asks for one action at a time, so we wrote the
  learners to work step by step, and the same code runs locally and on Kaggle.
- **What we reused.** The game engine, scorecard and agent framework come from the ARC Prize Foundation and are used
  unchanged. The Kaggle notebook build and local-play scripts are adapted from the official Kaggle starter, with its
  per-game action branch removed. The learners, improvements, evaluation and analysis were written for this project.
