# Evaluation protocol (freeze before any improvement work)

Fill in the date and commit, then do not change these values. Any unavoidable change gets a dated entry below.

Frozen on: ____ · commit: ____

## Environment
- Packages: arc-agi ____, arcengine ____, torch ____ (from `run_meta.json`).
- Interface: `arc_agi.Arcade(...).make(game_id, seed, scorecard_id)`; `env.reset()` sends RESET; `env.step(GameAction, data)`.
- Observation: last animation frame of each step, 64×64, 16 colours, one-hot encoded (16·k channels for a k-frame stack).
- Actions: indices 0–5 = ACTION1–5, ACTION7; indices 6.. = ACTION6 at the centre of each cell of a k×k grid
  (k=8 baseline). Illegal actions (not in `available_actions`) are masked in both learners.
- Termination: WIN ends the game. GAME_OVER is a terminal transition for learning, followed by RESET within the same game.
  Level completion is a non-terminal transition.

## MDP used for learning (training only)
- State: stack of the last k grids (k=1 baseline). Partially observable — rules must be inferred over time.
- Reward: `level_reward · Δlevels` (+ shaping terms for improvements: frame-change ±, novelty β/√N(s'), game-over penalty).
- γ = 0.95.

## Budgets and seeds
- Action budget: 1000 actions per game per seed (RESETs count). Identical for every agent.
- Seeds: 0, 1, 2 (agent seed = game seed). ≥3 seeds for every stochastic configuration.
- Compute: DQN one gradient step per 4 actions, batch 32. PPO update every 128 actions, 4 epochs, minibatch 32.
  Report wall time and hardware from `run_meta.json`. Training and evaluation cost are the same run in the
  online setting; any offline pretraining is reported separately (`outputs/checkpoints/*/run_meta.json`).
- Checkpoint selection: none. The final state of an online run is the result; no best-of-N reporting.

## Splits and leakage
- `configs/splits.yaml`, generated once by `scripts/make_splits.py` (seed 2026, 40% held out).
- All development, tuning and ablations use `dev` only. `heldout` is evaluated once per final config.
- No hidden game state, no game-specific code, no per-game hyperparameters.

## Metrics
- Primary: official ARC-AGI-3 score from the local `arc_agi` scorecard (identify the version); Kaggle public score reported separately.
- Levels completed, wins, actions, resets, game-overs per game — including failed runs.
- Levels completed vs. actions (learning curves), individual seeds + mean.
- Mean, std and 95% CI across seeds.

## Change log
| Date | Change | Reason |
|---|---|---|
