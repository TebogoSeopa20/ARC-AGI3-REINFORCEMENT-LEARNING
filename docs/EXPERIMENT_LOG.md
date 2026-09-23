# Experiment log

Every row traces to `outputs/results/<run>/<split>_seed<k>/run_meta.json`.
Hardware for all rows so far: MacBook Pro, Intel x86_64 CPU, no GPU, torch 2.2.2, arc-agi 0.9.9.

| Date | Run | Config | Split | Seeds | Official score per game (mean ± std over seeds) | Levels completed (of 112) | Wall time / seed | Notes |
|---|---|---|---|---|---|---|---|---|
| 23 Sept | random | random.yaml | dev | 0,1,2 | 0.010 ± 0.011 | 1.67 ± 0.58 | 50 s | sanity-check floor |
| 23 Sept | dqn_baseline | dqn_baseline.yaml | dev | 0,1,2 | 0.108 ± 0.180 | 1.67 ± 0.58 | 34 min | mean driven by seed 0 |
| 23 Sept | ppo_baseline | ppo_baseline.yaml | dev | 0,1,2 | 0.003 ± 0.002 | 1.33 ± 0.58 | 13 min | |
| 24 Sept | random (rerun) | random.yaml | dev | 0,1,2 | 0.010 ± 0.011 | 1.67 ± 0.58 | 55 s | identical to 23 Sept; change 0.676, unique 5922 |
| 24 Sept | dqn_baseline (rerun) | dqn_baseline.yaml | dev | 0,1,2 | 0.108 ± 0.180 | 1.67 ± 0.58 | 24 min | identical; change 0.671, unique 5738 |
| 24 Sept | ppo_baseline (rerun) | ppo_baseline.yaml | dev | 0,1,2 | 0.003 ± 0.002 | 1.33 ± 0.58 | 8 min | identical; change 0.675, unique 5911 |
| 24 Sept | dqn_imp1_explore | dqn_imp1_explore.yaml | dev | 0,1,2 | 0.112 ± 0.177 | 2.00 ± 1.00 | 24 min | change 0.742, unique 6577 |
| 24 Sept | ppo_imp1_explore | ppo_imp1_explore.yaml | dev | 0,1,2 | 0.003 ± 0.002 | 1.33 ± 0.58 | 17 min | change 0.712, unique 6356 |

Budget: 1000 actions per game, 15 dev games, 15 000 actions per seed. No wins in any run.
`change` = share of non-reset actions that changed the frame (mean over games); `unique` = distinct frames per seed
(summed over games). Reruns reproduced every level, score and game-over count exactly: CPU runs are deterministic
under fixed seeds. Wall time varies ±40% between runs on the same laptop, so compare it only within one session.

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
- **Result (24 Sept): the mechanism works where it can, but it does not reach levels.**
  - (a) Change rate rose from 0.671 to 0.742. The gain is concentrated in games where most actions are no-ops:
    sb26 0.17→0.51, sc25 0.25→0.49, sk48 0.08→0.21, su15 0.64→0.93.
  - (b) Distinct frames rose 15% (5738→6577), from below random (5922) to above it; sb26 96→296, sc25 231→414, sk48 68→178.
  - The repetition failure is fixed: r11l game-overs fell from 75.7 to 26.3 per 1000 actions (random 46).
  - (c) Not supported: levels 1.67→2.00 ± 1.00 and score 0.108→0.112 are within noise. The extra level is lp85 in
    seeds 1–2 (86 and 94 actions), which is suggestive but not established with 3 seeds.
  - In 7 games the change rate is already ≈1.0 for every agent, including random, whose actions are mostly clicks on
    empty cells (ls20, r11l, s5i5, sp80, tn36, tu93, re86), and distinct frames are close to the action count
    (e.g. ls20 ≈ 830 of 1000). There, both bonuses are nearly constant and carry no information.

### PPO — improvement 1 (`ppo_imp1_explore.yaml`)
- **Observed limitation:** all-zero advantages in 12/15 games leave the policy uniform (entropy ≈ 4.25, clip fraction ≈ 0).
- **Hypothesis:** the same intrinsic reward gives non-zero advantages from the first rollout, so the policy should
  concentrate on frame-changing, state-expanding actions. PPO may gain more than DQN here, because GAE spreads the
  per-step bonus over the rollout, while DQN learns it one bootstrapped transition at a time.
- **Intervention:** identical shaping to DQN improvement 1, evaluated separately.
- **Evidence to check:** entropy and clip-fraction traces, `change_rate`, `unique_states`, levels and score, 3 seeds.
- **Result (24 Sept): smaller effect than DQN, and no gain in levels.**
  - Change rate 0.675→0.712; distinct frames 5911→6356 (+7.5%, above random). Gains again sit in sb26
    (0.12→0.23), sc25 (0.13→0.29) and sk48 (0.07→0.24).
  - The policy now moves where the bonus is informative: seed 0 entropy in sb26/sc25/sk48 fell from 4.18–4.23 to
    3.95–3.97 (max ln 70 = 4.25), and clip fraction rose to 0.18–0.35. Elsewhere it stays near uniform
    (games restricted to 4–5 actions show entropy ≈ ln 4 = 1.39 or ln 5 = 1.61 in both versions).
  - Levels (1.33) and score (0.003) are unchanged.
  - The hypothesis that PPO would gain more than DQN is rejected: DQN moved more on every exploration metric.

### Diagnostic: what changes between frames (23 Sept, `scripts/diagnose_changes.py`, random policy, 300 actions/game, dev only)
- **No static HUD.** No pixel changes on more than half of the steps in any game (`px_changing_over_half` = 0 everywhere).
- **A per-action tick.** In 7 games almost every action, including random clicks on empty cells, changes exactly
  1–2 pixels (median changed pixels, with click change rate in brackets): s5i5 1 (1.00), tn36 1 (1.00), sp80 2 (1.00),
  su15 2 (0.87), ka59 1 (0.64), cn04 0–1 (0.46), tu93 2. Together with the identical game-over counts across agents, this
  matches a move-budget bar that fills one pixel per action and refills on RESET. Each tick pixel changes only once per
  life, so a frequency threshold cannot detect it; this is why `px_changing_over_half` stays 0.
- **Large, meaningful changes elsewhere.** ls20 52, re86 53, r11l 93 and wa30 32 pixels per step (moving sprites).
- **Rare changes.** lp85, sb26, sc25 and sk48 change on 4–13% of actions; these are the games where improvement 1 helped.
- **Coarse abstraction is not an option.** An 8×8 colour-mode grid collapses nearly every game to 1–3 states
  (`unique_coarse8`), destroying the information novelty needs.

### DQN and PPO — improvement 2 (`*_imp2_clockmask.yaml`)
- **Observed limitation:** in tick games, every action produces a new frame hash and a "changed" frame, so both
  improvement-1 bonuses are constant and carry no information about what the action did. Levels did not move with
  improvement 1 for either algorithm.
- **Hypothesis:** removing clock-like pixels before hashing and change detection makes the bonuses action-dependent in tick
  games. This should lower the masked change rate toward the share of actions with a real effect, raise the novelty
  signal's variance, and lead to more levels than improvement 1 in those games, without hurting lp85/sb26/sc25/sk48.
- **Intervention:** `ClockMask` (src/arcrl/agents/exploration.py). A pixel is masked once it has made the same colour
  transition at least twice in ordinary steps and never any other transition; level-change transitions are ignored. The mask
  is estimated online per game and applies only to the shaping. The network input, action choice and scores are unchanged.
  Known false positive: a pixel the agent changes identically once per life is also masked.
- **Check first (cheap):** rerun `scripts/diagnose_changes.py`. For tick games, `clock_px` should be > 0,
  `change_rate_clockmasked` should fall well below 1.0 and `unique_frames_clockmasked` should shrink. For sprite games
  (ls20, r11l), values should stay close to the unmasked ones.
- **Evaluation:** 3 seeds each for DQN and PPO against improvement 1. Report change rate and distinct frames masked and
  unmasked, levels, score, and per-game results for tick games vs. others.
- **Result:**

### Improvement 3 candidate (`*_imp3_memory.yaml`)
Frame-stack memory on top of improvement 2, justified if improvement 2 shows agents reaching new states without converting
them into level progress within a life. Decide after improvement 2 results.

## Failure cases to show in the report
| Run | Seed | Game | What happened | Evidence |
|---|---|---|---|---|
| dqn_baseline | 0–2 | r11l | 75.7 game-overs per 1000 actions vs 46 random; greedy repetition | games.jsonl `game_over_at` |
| all | all | cn04, ka59, ls20, … | fixed-clock game-overs, identical across agents | games.jsonl `game_overs` |
| dqn_baseline | 0 | sp80 | level 1 in 54 actions, then nothing in 946 more | games.jsonl `level_completed_at` |
