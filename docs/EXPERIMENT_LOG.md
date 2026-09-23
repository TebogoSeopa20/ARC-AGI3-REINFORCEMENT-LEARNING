# Experiment log

Every row traces to `outputs/results/<run>/<split>_seed<k>/run_meta.json`.
Hardware for all rows so far: MacBook Pro, Intel x86_64 CPU, no GPU, torch 2.2.2, arc-agi 0.9.9.

| Date | Run | Config | Split | Seeds | Official score per game (mean ± std over seeds) | Levels completed (of 112) | Wall time / seed | Notes |
|---|---|---|---|---|---|---|---|---|
| 23 Sept | random | random.yaml | dev | 0,1,2 | 0.010 ± 0.011 | 1.67 ± 0.58 | 50 s | sanity-check floor |
| 23 Sept | dqn_baseline | dqn_baseline.yaml | dev | 0,1,2 | 0.108 ± 0.180 | 1.67 ± 0.58 | 34 min | mean driven by seed 0 |
| 23 Sept | ppo_baseline | ppo_baseline.yaml | dev | 0,1,2 | 0.003 ± 0.002 | 1.33 ± 0.58 | 13 min | |

Budget: 1000 actions per game, 15 dev games, 15 000 actions per seed. No wins in any run.

## Baseline findings (23 Sept)

1. **Both learners are indistinguishable from random.** Levels completed overlap completely (1.3–1.7 of 112);
   the DQN score mean comes from one seed (seed 0: r11l in 32 actions, sp80 in 54 actions).
2. **Reward never reaches most games.** Across all 9 learner runs, a level was completed only in r11l (every run)
   and lp85 (some runs), plus sp80 once. 12 of 15 games produced zero reward in every run, so neither learner
   received a single non-zero learning signal there.
3. **The completions that did happen were luck, not learning.** Every completion is level 1, most occur while DQN
   ε is still high (32–580 actions into a game), and no run ever completed level 2 after being rewarded for level 1.
4. **Many games end on a fixed clock.** Game-over counts are identical across random, DQN and PPO in 8 games
   (cn04 13, ka59 9, ls20 7, re86 9, s5i5 19, tn36 16, tu93 19, wa30 4 per 1000 actions), so episodes end on an
   action limit regardless of behaviour. Each life is short (about 50–140 actions), so exploration must also carry
   information across lives within a game.
5. **Greedy DQN can be worse than random.** In r11l, DQN hits GAME_OVER 75.7 times per 1000 actions versus 46 for random:
   with no reward, the greedy action is arbitrary and gets repeated, and this game punishes repetition.
6. **PPO stays uniform.** Policy entropy stays near the maximum for 70 actions (4.25) and the clip fraction stays near 0;
   with zero advantages, there is nothing to update toward.
7. **Cost.** DQN ≈ 7 actions/s and PPO ≈ 19 actions/s on this CPU, well below the smoke test, because training
   dominates once the replay buffer fills. Improvement 2 (4-frame input) will be slower, so time it on the Kaggle T4.

## Improvement reasoning

### DQN — improvement 1 (`dqn_imp1_explore.yaml`)
- **Observed limitation:** zero extrinsic reward in 12/15 dev games; ε-greedy over arbitrary Q-values repeats actions
  (r11l game-overs 75.7 vs 46 for random); no level 2 completed after a level-1 reward.
- **Hypothesis:** a dense intrinsic signal gives the Q-network something to fit before the first level reward.
  Count-based novelty β/√N(s′) plus a ± bonus for changing the frame should (a) raise the share of actions that change
  the frame, (b) increase distinct states visited per game, and (c) through (a) and (b), raise levels completed above random.
- **Intervention:** novelty 0.1/√N over exact frame-hash counts (reset per game), frame-change ±0.05, game-over penalty 0.5.
  Shaping is training-only; scores stay official.
- **Evidence to check:** `change_rate` and `unique_states` (logged from now on) against the baseline; levels and score
  against baseline and random, 3 seeds.
- **Result:**

### PPO — improvement 1 (`ppo_imp1_explore.yaml`)
- **Observed limitation:** all-zero advantages in 12/15 games leave the policy uniform (entropy ≈ 4.25, clip fraction ≈ 0).
- **Hypothesis:** the same intrinsic reward gives non-zero advantages from the first rollout, so the policy should
  concentrate on frame-changing, state-expanding actions. PPO may gain more than DQN here, because GAE spreads the
  per-step bonus over the rollout, while DQN learns it one bootstrapped transition at a time.
- **Intervention:** identical shaping to DQN improvement 1, evaluated separately.
- **Evidence to check:** entropy and clip-fraction traces, `change_rate`, `unique_states`, levels and score, 3 seeds.
- **Result:**

### DQN — improvement 2
- Observed limitation:
- Hypothesis:
- Intervention:
- Result:

### PPO — improvement 2
- Observed limitation:
- Hypothesis:
- Intervention:
- Result:

## Failure cases to show in the report
| Run | Seed | Game | What happened | Evidence |
|---|---|---|---|---|
| dqn_baseline | 0–2 | r11l | 75.7 game-overs per 1000 actions vs 46 random; greedy repetition | games.jsonl `game_over_at` |
| all | all | cn04, ka59, ls20, … | fixed-clock game-overs, identical across agents | games.jsonl `game_overs` |
| dqn_baseline | 0 | sp80 | level 1 in 54 actions, then nothing in 946 more | games.jsonl `level_completed_at` |
