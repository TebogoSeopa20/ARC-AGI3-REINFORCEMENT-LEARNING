# Evaluation protocol (frozen before any improvement work)

Frozen on: 23 September 2026 · commit: `git rev-parse --short HEAD` after the splits commit → ____
Any unavoidable change gets a dated entry in the change log below.

## Environment
- Packages (Mac dev): arc-agi 0.9.9, arcengine 0.9.3, torch 2.2.2, numpy 1.26.4, gymnasium 1.3.0, Python 3.12.
  Kaggle runs record their own versions in `run_meta.json`.
- Public set: 25 games fetched from the ARC-AGI-3 API on 23 Sept 2026, cached in `environment_files/`.
- Interface: `arc_agi.Arcade(...).make(game_id, seed, scorecard_id)`; the wrapper resets on creation;
  `env.reset()` sends RESET; `env.step(GameAction, data)`.
- Observation: last animation frame of each step, 64×64, 16 colours, one-hot (16·k channels for a k-frame stack).
- Actions: indices 0–5 = ACTION1–5, ACTION7; indices 6.. = ACTION6 at the centre of each cell of a k×k grid
  (k=8 baseline, 70 actions). Actions absent from `available_actions` are masked in both learners.
- Termination: WIN ends the game. GAME_OVER is a terminal transition for learning, followed by RESET in the same game.
  Level completion is non-terminal.

## MDP used for learning (training only)
- State: stack of the last k grids (k=1 baseline); the problem is partially observable.
- Reward: `level_reward · Δlevels`; improvements add frame-change ±, novelty β/√N(s') and a game-over penalty.
- γ = 0.95.

## Budgets and seeds
- Action budget: 1000 actions per game per seed, RESETs included, identical for every agent.
  Provisional until the Kaggle T4 timing probe; Mac CPU smoke test (cn04, 300 actions): random ≈ instant,
  DQN ≈ 14 actions/s, PPO ≈ 39 actions/s.
- Seeds: 0, 1, 2 (agent seed = game seed).
- Compute: DQN one gradient step per 4 actions, batch 32, replay 5000. PPO update every 128 actions, 4 epochs, minibatch 32.
  Online setting: training and evaluation are the same run; offline pretraining, if used, is reported separately.
- Checkpoint selection: none. The final state of each online run is the result.

## Splits and leakage
- `configs/splits.yaml`, generated once with `scripts/make_splits.py` (seed 2026, 40% held out): 15 dev, 10 held-out.
- Development, tuning and ablations use `dev` only. `heldout` is evaluated once per final config.
- No hidden game state, no game-specific code, no per-game hyperparameters.

## Metrics
- Primary: official score from the local `arc_agi` 0.9.9 scorecard; Kaggle public score reported separately.
- Levels completed, wins, actions, resets and game-overs per game, including failed runs.
- Levels completed vs. actions (learning curves), individual seeds and mean.
- Mean, std and 95% CI across seeds.

## Change log
| Date | Change | Reason |
|---|---|---|
| 23 Sept 2026 | Protocol frozen; splits generated | Phase 1 |
