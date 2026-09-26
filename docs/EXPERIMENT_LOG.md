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
| 24 Sept | dqn_imp2_clockmask | dqn_imp2_clockmask.yaml | dev | 0,1,2 | 0.113 ± 0.176 | 2.67 ± 0.58 | 26 min | change 0.741, unique 6982; new: r11l s1@811, sp80 s2@587 |
| 24 Sept | ppo_imp2_clockmask | ppo_imp2_clockmask.yaml | dev | 0,1,2 | 0.003 ± 0.003 | 1.67 ± 0.58 | 9 min | change 0.711, unique 6678; new: sp80 s0@616 |
| 24 Sept | dqn_abl_no_novelty | ablations/dqn_final_no_novelty.yaml | dev | 0,1,2 | 0.110 | 2.33 ± 0.58 | 16 min | unique 6612 |
| 24 Sept | dqn_abl_no_change | ablations/dqn_final_no_change.yaml | dev | 0,1,2 | 0.110 | 2.33 ± 0.58 | 17 min | unique 6803 |
| 24 Sept | ppo_abl_no_novelty | ablations/ppo_final_no_novelty.yaml | dev | 0,1,2 | 0.003 | 2.00 ± 1.00 | 7 min | unique 6793; new: cn04 s2@971 |
| 24 Sept | ppo_abl_no_change | ablations/ppo_final_no_change.yaml | dev | 0,1,2 | 0.003 | 2.00 ± 1.00 | 7 min | unique 6427 |
| 24 Sept | dqn_imp4_efficient | dqn_imp4_efficient.yaml | dev | 0,1,2 | 0.138 ± 0.156 | 3.00 ± 1.00 | 29 min | unique 9962, effect rate 0.65 |
| 24 Sept | ppo_imp4_efficient | ppo_imp4_efficient.yaml | dev | 0,1,2 | 0.027 ± 0.029 | 4.00 ± 1.73 | 9 min | unique 9819, effect rate 0.63; first level 2 (tu93) |
| 24 Sept | random_imp4_actions | random_imp4_actions.yaml | dev | 0,1,2 | 0.235 ± 0.144 | 1.33 ± 0.58 | 1 min | control; score from r11l at 7/12 actions |
| 24 Sept | imp4 ablations (6 runs) | ablations/*_imp4_no_*.yaml | dev | 0,1,2 | see improvement 4 ablations | | | |
| 25 Sept | random_imp5_graph | random_imp5_graph.yaml | dev | 0,1,2 | 0.236 ± 0.142 | 2.67 ± 1.15 | 1 min | control; graph without learning |
| 25 Sept | dqn_imp5_graph | dqn_imp5_graph.yaml | dev | 0,1,2 | 0.141 ± 0.154 | 4.67 ± 1.53 | 21 min | 2 × tu93 level 2; determinism 98.6% |
| 25 Sept | ppo_imp5_graph | ppo_imp5_graph.yaml | dev | 0,1,2 | 0.070 ± 0.075 | 5.00 ± 1.00 | 11 min | 2 × tu93 level 2; determinism 98.6% |
| 25 Sept | dqn_abl_imp5_no_graph (rerun) | ablations/dqn_imp5_no_graph.yaml | dev | 0,1,2 | 0.142 | 3.00 ± 1.00 | 18 min | valid after fix; no tu93 |
| 25 Sept | ppo_abl_imp5_no_graph (rerun) | ablations/ppo_imp5_no_graph.yaml | dev | 0,1,2 | 0.014 | 3.67 ± 0.58 | 11 min | valid after fix; no tu93 |
| 25–26 Sept | 15 versions | see held-out section | heldout | 0,1,2 | see held-out section | | | run once |

Improvement 3 runs (dev_b = cn04, lp85, sk48, su15, tu93; levels out of 5 games' worth; imp1/imp2/baselines pooled from dev runs):

| Date | Run | Split | Seeds | Score | Levels on dev_b | Distinct frames | Eval time / seed | Pretrain time / seed |
|---|---|---|---|---|---|---|---|---|
| 24 Sept | random | dev_b | 0,1,2 | 0.001 | 0.67 ± 0.58 | 1116 | | |
| 24 Sept | dqn_imp2_clockmask | dev_b | 0,1,2 | 0.014 | 1.00 ± 0.00 | 1406 | | |
| 24 Sept | dqn_imp3_pretrain | dev_b | 0,1,2 | 0.005 | 1.33 ± 0.58 | 1146 | 5.6 min | 23 min (2 passes × 10 games) |
| 24 Sept | dqn_abl_scratch_loweps | dev_b | 0,1,2 | 0.009 | 0.33 ± 0.58 | 763 | 5.8 min | |
| 24 Sept | ppo_imp2_clockmask | dev_b | 0,1,2 | 0.001 | 0.33 ± 0.58 | 1312 | | |
| 24 Sept | ppo_imp3_pretrain | dev_b | 0,1,2 | 0.000 | 0.33 ± 0.58 | 1748 | 2.3 min | 9.6 min |

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

### Ablations of improvement 2 (24 Sept, dev, 3 seeds; each removes one shaping term, clock mask kept)

| Run | Levels (per seed) | Distinct frames | Score |
|---|---|---|---|
| random | 1.67 (2, 2, 1) | 5922 | 0.010 |
| dqn_imp2_clockmask | 2.67 (3, 2, 3) | 6982 | 0.113 |
| dqn_abl_no_novelty | 2.33 (3, 2, 2) | 6612 | 0.110 |
| dqn_abl_no_change | 2.33 (3, 2, 2) | 6803 | 0.110 |
| ppo_imp2_clockmask | 1.67 (2, 1, 2) | 6678 | 0.003 |
| ppo_abl_no_novelty | 2.00 (2, 1, 3) | 6793 | 0.003 |
| ppo_abl_no_change | 2.00 (1, 3, 2) | 6427 | 0.003 |

- **DQN: the two terms are complementary.** Removing either lowers exploration: −370 frames without novelty, −179 without
  the change term. The losses sit in different games. Without novelty: sb26 278 → 146. Without the change term:
  sp80 495 → 377. Without either: su15 355 → 213 / 196, so su15 needs both. s5i5 rises in both ablations
  (300 → 319 / 370), the one game where the combination explores less.
- **PPO: the change term carries the effect, novelty does not.** Without the change term, frames fall to 6427 (sc25
  280 → 191). Without novelty, frames rise to 6793, slightly above imp2. For PPO, novelty is neutral to mildly harmful.
- **Levels cannot separate the components.** All differences are within one level of imp2 and within seed noise. Two
  completions are new: DQN r11l seed 1 at 533 (both ablations), and PPO cn04 seed 2 at action 971 (no novelty), the first
  cn04 completion in any run.
- **Report framing:** attribution is supported on exploration metrics, not on levels or score. The terms work in different
  games for DQN and differently across algorithms. This is an instance of the brief's "evaluate an idea separately for each
  algorithm": the same shaping behaves differently for DQN and PPO.

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
- **Check result (24 Sept, random policy):** masked change rate fell from ≈1.0 to 0.16–0.25 in tick games
  (s5i5 0.19, sp80 0.18, su15 0.16, tn36 0.22, ka59 0.25, cn04 0.18) and to 0.51 in tu93; sprite games were almost
  untouched (ls20 1.00, re86 1.00, wa30 0.83); r11l fell to 0.60; the rare-change games were unchanged (sb26, sc25 and sk48
  had 0 masked pixels). Artefact: the mask grows during play, so one screen can hash differently before and after a pixel
  is masked. This inflates masked distinct-frame counts slightly (ls20 277→294, tn36 114→126) and splits some novelty counts.
- **Evaluation:** 3 seeds each for DQN and PPO against improvement 1. Report change rate and distinct frames masked and
  unmasked, levels, score, and per-game results for tick games vs. others.
- **Result, DQN (24 Sept): +2 levels over improvement 1, both late in a game; still no level 2.**
  - Levels 2.00 ± 1.00 → 2.67 ± 0.58 (per seed 3, 2, 3); random and baseline 1.67.
  - Common random numbers matter here. All configs share seeds, and DQN's ε is ≈1 early in each game, so the first
    actions are near-identical across versions. Completions at the same action index in several versions are that shared
    early luck, not an improvement effect: r11l seed 0 at action 32 and sp80 seed 0 at 54 (baseline, imp1, imp2), and lp85
    seeds 1–2 at 94/86 (imp1, imp2). Those two seed-0 completions also produce almost all of DQN's official score in every
    version, which is why the score is flat (0.108 → 0.112 → 0.113).
  - The completions unique to improvement 2 are **r11l seed 1 at action 811** and **sp80 seed 2 at 587**, both late
    (ε ≈ 0.25–0.45), when learned values drive most actions. sp80 is a tick game, the hypothesis's target case.
  - Exploration gains sit in tick games, as predicted (distinct frames, imp1 → imp2): s5i5 205 → 300, sp80 337 → 495,
    su15 191 → 355, tn36 344 → 363. Rare-change and sprite games are unchanged (sb26 296 → 278, sc25 414 → 417,
    ls20 830 → 825).
  - Total distinct frames 6577 → 6982 (+6%). Unmasked change rate unchanged (0.741), as expected.
  - With 3 seeds and 2 extra completions, the gain is suggestive, not established.
- **Result, PPO (24 Sept): one extra late completion.**
  - Levels 1.33 → 1.67. r11l (174/284/417) and lp85 seed 2 (246) repeat at the same actions in every PPO version, so
    they come from the shared early action stream. The new completion is **sp80 seed 0 at action 616**, again a tick game.
  - Tick-game exploration rose (sp80 308 → 422, tn36 339 → 377, tu93 421 → 515); total distinct frames 6356 → 6678.
    Score unchanged (0.003).
  - PPO remains at the random-policy level on levels; DQN benefits more from both exploration improvements.
- **Analysis rule adopted from here:** a completion counts as evidence for a change only if it does not occur at the same
  action index in the previous version with the same seed. Report early shared completions separately.
- **Open problem for any next step:** no run in any configuration has completed level 2. Every success is one level-1
  completion, and each game starts from freshly initialised weights, so each game relearns from zero which actions matter.

### DQN and PPO — improvement 3: cross-game pretraining (`*_imp3_pretrain.yaml`)
- **Observed limitation:** no run in any version has completed level 2. The few completions unique to improvement 2 arrive
  late in a game (actions 587–811), and every game starts from freshly initialised weights, so each game spends most of
  its 1000-action budget relearning, from zero, which kinds of action have effects.
- **Hypothesis:** weights pretrained across other games encode game-independent regularities (e.g. which action types tend
  to change the masked frame, or not repeating actions that trigger GAME_OVER). Starting from them, an agent on an unseen
  game should reach its first completion in fewer actions and complete more levels than improvement 2 from scratch,
  under the same budget.
- **Intervention:** `scripts/pretrain.py` trains one network per seed sequentially over the pretraining games
  (2 passes × 1000 actions per game), keeping weights (and DQN's replay) across games while restarting the exploration
  schedule and novelty counts per game. Evaluation loads the seed's checkpoint at the start of every unseen game and keeps
  learning online, exactly as before. DQN starts at ε = 0.3 (decay over 500 actions) so the pretrained values are used.
  PPO's policy uses the weights directly.
- **Leakage control:** the 15 dev games are split once (seeded, `configs/pretrain_splits.yaml`) into dev_a (10, pretraining)
  and dev_b (5, evaluation): cn04, lp85, sk48, su15, tu93. `evaluate.py` refuses to score a game that was used for
  pretraining, and `pretrain.py` refuses held-out games. The final version (`*_imp3_pretrain_final.yaml`) pretrains on all
  15 dev games and is scored once on the 10 held-out games.
- **Comparison:** improvement 2 on the same 5 dev_b games at the same seeds, taken from the existing dev runs
  (`scripts/compare_splits.py --split dev_b`). Random, baseline and imp1 are included the same way.
- **Attribution (DQN):** `ablations/dqn_scratch_loweps.yaml` is improvement 2 from scratch with the same ε schedule on dev_b,
  which separates the pretrained weights from the lower-exploration schedule. PPO has no such confound.
- **Evidence to check:** levels, and actions to first completion, on dev_b vs imp2 and vs the ε-ablation; distinct frames;
  completions judged by the rule above (not at the same action index as imp2 with the same seed). Pretraining cost is
  reported separately from evaluation cost (checkpoint `run_meta.json`, `pretrain_wall_time_s`).
- **Caveat known in advance:** dev_b has 5 games, 3 of which (sk48, su15, tu93) have never been completed by any agent.
  A null result on dev_b is therefore weak evidence either way; the held-out run is the real test.
- **Result, DQN (24 Sept): pretrained weights transfer an exploration prior; no clear gain in levels or efficiency.**
  - Levels on dev_b 1.00 → 1.33 ± 0.58. The extra level is **su15 seed 0 at action 841, the first su15 completion by
    any agent in any run**. One event, so suggestive only.
  - Actions to first completion did not improve: lp85 at 117/289/361 against 375/94/86 for imp2 (mean 256 vs 185).
  - Attribution: the ε-ablation (same ε = 0.3 schedule, no pretraining) collapses to 0.33 levels and 763 distinct frames.
    Pretrained weights under that schedule restore 1146 frames and 1.33 levels. The pretrained network therefore carries
    useful exploratory behaviour into unseen games; without it, low ε alone is harmful.
  - Against imp2 at ε = 1.0, exploration is lower (1146 vs 1406 frames), so pretraining plus low ε roughly substitutes for
    random exploration rather than improving on it.
- **Result, PPO (24 Sept): much broader exploration on unseen games, no gain in levels.**
  - Distinct frames 1312 → 1748 (+33%), the highest of any run on dev_b. Levels unchanged (0.33; lp85 seed 1 at 706, a
    new completion; the imp2 completion lp85 seed 2 at 246 disappeared).
  - The pretrained policy transfers an exploration behaviour, but not a level-completing one.
- **Pretraining itself did not improve across passes.** Per seed, level completions in pass 1 vs pass 2 were DQN 0/0, 1/1,
  1/1 and PPO 1/1, 3/1, 1/2, almost all r11l. The network sees a level reward about once per 10 000 actions, so what it
  learns across games is an exploration prior, not knowledge of goals.
- **Cost:** DQN pretraining ≈ 23 min per seed and PPO ≈ 9.6 min per seed on the Mac CPU, reported separately from
  evaluation (DQN 5.6 min, PPO 2.3 min per seed on dev_b).
- **Conclusion for the report:** improvement 3 answers the generalisation question with a clear, partly negative finding.
  Game-independent exploration behaviour transfers across ARC-AGI-3 games (strong for PPO, and it rescues low-ε DQN);
  level-completing behaviour does not, because level rewards are too rare during pretraining to learn from.

### Decision on improvement 4 (24 Sept, revised the same day)
The first decision was to stop, because pretraining did not give faster first completions. It was revised after
comparing against published ARC-AGI-3 agents, which surfaced a limitation the log had measured but not acted on.
- Context: frontier LLMs score below 1% on ARC-AGI-3. The preview winner StochasticGoose (a CNN that learns which actions
  change the frame) scored 12.58% on the preview set and 0.25% on the full benchmark. Second place, Blind Squirrel,
  prunes actions that do not change the state. Both spend their budget on actions with effects.
- Our agents do the opposite. The 23 Sept diagnostic shows click-hit rates of 4.4% (lp85), 9.7% (sb26), 5.1% (sc25)
  and 0% (sk48) under 8×8 centre clicks. Every agent re-tries actions already seen to do nothing in the same state,
  including after RESET.

### DQN and PPO — improvement 4: efficient action selection (`*_imp4_efficient.yaml`)
- **Observed limitation:** most actions in click games hit empty cells (hit rates 0–10%). Actions known to do nothing in a
  state are repeated whenever that state recurs. All learning signals are sparse: the level reward is rare, and the
  shaping reward is a single scalar per step.
- **Hypothesis:** spending actions on predicted-effective, not-yet-failed actions at the resolution of actual objects
  raises the share of actions that change the clock-masked frame (`effect_rate`), reaches more distinct states per action,
  and converts that into more and earlier level completions than improvement 2.
- **Intervention (three components, each ablated):**
  1. **Effect head** on the shared CNN: per-action logits of P(clock-masked frame changes), trained with BCE on every
     transition. This is a dense, supervised auxiliary task, as in UNREAL (Jaderberg et al., 2017) and StochasticGoose
     (Smit, 2025). DQN's ε-exploration samples actions in proportion to the predicted probability instead of uniformly.
     PPO adds `effect_bias · log σ(effect)` (detached) to its logits; the bias used at sampling time is stored and reused
     in the update so the importance ratio stays exact.
  2. **No-effect pruning:** per game, the controller records, per clock-masked state hash, the actions that left the frame
     unchanged, and excludes them when that state recurs (including after RESET) while one legal action remains
     (after Blind Squirrel).
  3. **Object-aware 16×16 clicks:** click cells of 4×4 pixels, restricted to cells containing non-background pixels,
     clicking an object pixel rather than the cell centre (`ActionSpace(object_clicks=True)`).
  All three are game-agnostic and use only the observation. Weights reset per game, as in improvement 2.
- **Reproducibility check:** after adding the code, dqn_baseline and ppo_baseline reproduce the previous code's results
  bit for bit (same losses, levels and game-overs on the toy harness), so earlier runs remain valid.
- **Evidence to check:** `effect_rate` (new metric: share of actions that changed the clock-masked frame), distinct frames,
  levels and actions to first completion, and official score against improvement 2 and random on all 15 dev games,
  3 seeds; then the three single-component ablations per algorithm (`ablations/*_imp4_no_*.yaml`).
  Completions are judged with the shared-luck rule.
- **Result (24 Sept, dev, 3 seeds): the largest change of the investigation, strongest for PPO.**

  | Run | Levels (per seed) | Distinct frames | Score | Effect rate |
  |---|---|---|---|---|
  | random | 1.67 (2, 2, 1) | 5922 | 0.010 | |
  | dqn_imp2_clockmask | 2.67 (3, 2, 3) | 6982 | 0.113 | |
  | **dqn_imp4_efficient** | **3.00 (4, 2, 3)** | **9962** | **0.138** | 0.65 |
  | ppo_imp2_clockmask | 1.67 (2, 1, 2) | 6678 | 0.003 | |
  | **ppo_imp4_efficient** | **4.00 (6, 3, 3)** | **9819** | **0.027** | 0.63 |

  - **Exploration:** distinct frames +43% for DQN and +47% for PPO, rising in 13 of 15 games for both. The largest gains
    are in click and tick games: s5i5 300 → 792, sc25 417 → 775, tn36 363 → 809, cn04 395 → 797 (DQN). ls20 and re86 were
    already near the action count.
  - **PPO levels:** every seed (6, 3, 3) is at or above the best imp2 seed (2). Non-overlapping across seeds, the clearest
    level effect in the study. Score 0.003 → 0.027.
  - **First level 2 of the investigation:** PPO seed 0 completed tu93 levels 1 and 2 at actions 652 and 702, the second
    level 50 actions after the first (human baseline 16).
  - **Games never completed before:** s5i5 (DQN seeds 0 and 2 at 291 and 652; PPO seeds 0 and 1 at 916 and 144),
    sc25 (DQN seed 1 at 656), tu93 (PPO). PPO completed sp80 in all 3 seeds (440–488).
  - **DQN:** levels 2.67 → 3.00 (within noise), score 0.113 → 0.138. r11l is now completed early in every seed (76, 52, 18);
    seed 2 at 18 actions beats the 22-action human baseline for that level (level score 4.76).
  - **Where it did not help:** lp85 completions fell for DQN (3 → 2) and su15 exploration dropped (355 → 290). lp85 has a
    0.13–0.27 effect rate and the most pruned choices (838–943), which suggests pruning or object clicks may cut off its
    working actions. The ablations will test this.
  - **Caveat on early completions:** the shared-luck rule cannot be applied, because the action space changed, so early
    action streams differ from imp2's. The r11l completions at 18–76 actions could be luck of the new action abstraction.
    `configs/random_imp4_actions.yaml` (random choice over the same abstraction, no learning) is the control for this.
  - **Cost:** wall time per seed similar to imp2 (DQN 29 min, PPO 9 min).

### Improvement 4 — ablations and random control (24 Sept, dev, 3 seeds)

| Run | Levels (per seed) | Levels excl. r11l | Distinct frames | Score | Score excl. r11l |
|---|---|---|---|---|---|
| random | 1.67 (2, 2, 1) | 0.67 | 5922 | 0.010 | 0.000 |
| random_imp4_actions (no learning) | 1.33 (1, 2, 1) | 0.33 | 8025 | **0.235** | 0.001 |
| dqn_imp4_efficient | 3.00 (4, 2, 3) | 2.00 | 9962 | 0.138 | 0.005 |
| dqn − effect head | 3.00 (4, 3, 2) | 2.00 | 9417 | 0.054 | 0.003 |
| dqn − pruning | 3.00 (4, 3, 2) | 2.00 | 9393 | 0.139 | 0.006 |
| dqn − object clicks | 2.67 (2, 3, 3) | 1.67 | 8425 | 0.056 | 0.005 |
| ppo_imp4_efficient | **4.00 (6, 3, 3)** | **3.00** | 9819 | 0.027 | **0.014** |
| ppo − effect head | 2.00 (2, 2, 2) | 1.00 | 8757 | 0.061 | 0.001 |
| ppo − pruning | 3.00 (3, 2, 4) | 2.00 | 9318 | 0.016 | 0.004 |
| ppo − object clicks | 3.33 (3, 3, 4) | 2.33 | 8214 | 0.200 | 0.004 |

- **Learning, not the abstraction, produces the level gains.** Random choice over the same action abstraction completes
  1.33 levels (0.33 excluding r11l). PPO imp4's worst seed (3) is above the control's best (2); DQN imp4 is above it in
  every seed but one.
- **The abstraction explains about half of the exploration gain:** distinct frames 5922 (random) → 8025 (random over
  imp4 actions) → about 9900 (learners).
- **PPO: the effect head is the key component.** Removing it halves levels (4.00 → 2.00, seeds non-overlapping) and
  removes all s5i5, tu93 and cn04 progress. Pruning and object clicks each account for about one level, within noise.
  tu93 level 2 appears only with all three components.
- **DQN: no single component changes levels** (2.67–3.00). Object clicks carry exploration (9962 → 8425; s5i5 792 → 404,
  tn36 809 → 367 without them) and every s5i5/sc25 completion. DQN without the effect head completed su15 in 2 seeds
  (713, 720), the first su15 completions from scratch. The lp85 drop noted above is noise: each ablation restores it to 3.
- **The official score cannot rank agents on dev.** The control scores highest (0.235) because it completed r11l level 1
  in 7 and 12 actions. Such a completion reaches the maximum score available from level 1 (100/Σweights = 4.76 per game),
  so one lucky early completion outweighs everything else. Excluding r11l, every run scores ≤ 0.014 and PPO imp4 is best.
  Report levels, levels excluding r11l, and score excluding r11l alongside the official score.
- **Submission choice (dev only):** PPO imp4. It has the most levels, the only level 2, is best excluding r11l on both
  measures, and runs about 3× faster than DQN (9 vs 29 min per seed), which matters for Kaggle's time limit.

### DQN and PPO — improvement 5: return-then-explore over a state graph (`*_imp5_graph.yaml`)
- **Observed limitation:** even with improvement 4, agents spend most of the budget re-deciding in states whose actions
  they have already tried, and lose all progress at each GAME_OVER: after RESET they must rediscover the path to where
  they were. In 8 dev games lives end on a fixed move count (fixed-clock game-overs, 4–19 per 1000 actions), so a life is
  too short to both return and explore by chance. No-effect pruning (imp4) only removes actions that did nothing.
- **Hypothesis:** if the agent never repeats an action whose outcome it already knows, and returns along the shortest known
  path to the nearest state with untried actions (including after RESET), it covers more of each game per action. It
  should complete more levels, and more second levels, than improvement 4 under the same budget.
- **Intervention:** a per-game `StateGraph` (src/arcrl/agents/graph.py). Nodes are (levels, clock-masked frame hash).
  It stores legal actions, tried actions and observed outcomes (next node or GAME_OVER). In a node with untried legal
  actions, DQN/PPO choose among the untried ones only. In an exhausted node, the controller takes the first action of
  the shortest known path to the nearest node with untried actions. This is Go-Explore's "first return, then explore"
  (Ecoffet et al., 2021), with return by replaying known transitions instead of restoring emulator state. The learner
  sees a semi-MDP: one transition per learner decision, with the undiscounted sum of shaped rewards until its next
  decision. PPO therefore stays on-policy, and DQN stores only its own choices.
- **Needed fix found in testing: timed clock mask.** The original ClockMask also masks progress markers the agent changes
  once per life (a documented false positive). For the graph this is fatal: different progress states merge into one
  node. The timed variant additionally requires each repeat to happen at the same action index within a life, which a
  per-action counter satisfies and agent-caused progress does not. Only improvement 5 uses it (`clock_mask_timed`).
- **Assumption measured, not assumed:** transitions are treated as deterministic. Every repeated (node, action) is
  logged as consistent or inconsistent (`transitions_consistent` / `transitions_inconsistent`). A GAME_OVER never
  overwrites a known live outcome, because with the counter masked, running out of moves looks like a fatal action.
- **Checks before real games:** on a deterministic combination-lock test (7 presses in a row, 12 moves per life, 400
  actions), plain random solved 1 of 6 locks; random with the graph solved 6 of 6 (37–114 actions), DQN with the graph
  6 of 6, PPO with the graph 5 of 6. DQN/PPO decision counts and stored transitions match exactly (tests/test_graph.py).
  Baseline, imp2 and imp4 configs reproduce the previous code bit for bit.
- **Controls:** `random_imp5_graph.yaml` (graph with random choice among untried actions, no learning) separates search
  from learning. `ablations/*_imp5_no_graph.yaml` (imp4 + timed mask, no graph) separates the graph from the mask change.
- **Reporting caveat:** improvement 5 is hybrid; the search structure does part of the work. The comparison that answers
  the course question is DQN vs PPO under the same graph, and each learner vs the random-with-graph control.
- **Decision rule, fixed before the results (24 Sept):** beats imp4 and random-with-graph on levels → full improvement,
  and the submission if it also beats PPO imp4; beats imp4 but random-with-graph matches it → report that the search does
  the work; no gain → short negative result.
- **Result (25 Sept, dev, 3 seeds): the largest level gain for DQN, and the most second levels of any run.**

  | Run | Levels (per seed) | Levels excl. r11l | Level-2 completions | Distinct frames | Score | Score excl. r11l |
  |---|---|---|---|---|---|---|
  | random_imp4_actions | 1.33 (1, 2, 1) | 0.33 | 0 | 8025 | 0.235 | 0.001 |
  | random_imp5_graph (no learning) | 2.67 (4, 2, 2) | 1.67 | 1 | 8462 | 0.236 | 0.002 |
  | dqn_imp4_efficient | 3.00 (4, 2, 3) | 2.00 | 0 | 9962 | 0.138 | 0.005 |
  | **dqn_imp5_graph** | **4.67 (6, 5, 3)** | **3.67** | **2** | 9954 | 0.141 | 0.007 |
  | ppo_imp4_efficient | 4.00 (6, 3, 3) | 3.00 | 1 | 9819 | 0.027 | 0.014 |
  | **ppo_imp5_graph** | **5.00 (4, 5, 6)** | **4.00** | **2** | 9787 | 0.070 | 0.003 |

  - **DQN: +1.67 levels over imp4** (3.00 → 4.67), levels excluding r11l 2.00 → 3.67. The worst imp5 seed (3) equals the
    imp4 mean. **PPO: +1.00** (4.00 → 5.00); its worst seed (4) is above the imp4 median (3).
  - **Learning still adds beyond the search.** Random with the graph reaches 2.67 levels (1.33 without it). DQN and PPO with
    the graph reach 4.67 and 5.00. Per seed, PPO's worst (4) equals the control's best (4), and DQN's worst (3) is below the
    control's best (4). The learner effect is consistent in the mean but overlaps at the extremes.
  - **Where it helped: tu93, the one game where return navigation is active.** tu93 levels, summed over seeds: imp4 DQN 0 /
    PPO 2 → imp5 DQN 4 / PPO 5 / random-with-graph 3. Level 2 was reached in 2 of 3 seeds for both DQN and PPO (e.g. PPO
    seed 1 at actions 403 and 599). 15% of tu93 actions were planned returns, against 1.6–1.8% over all games. DQN also
    gained lp85 (2 → 3), PPO lost none.
  - **Mechanism elsewhere is "never repeat a tried action", not navigation.** Over all games, only 1.6% (DQN), 1.8% (PPO)
    and 1.8% (random) of actions were planned returns. Click games have up to 262 legal actions per state, so states are
    rarely exhausted within 1000 actions, and the graph acts as a stricter version of imp4's no-effect pruning. The
    `*_imp5_no_graph` ablation separates the two.
  - **Determinism assumption holds:** 98.6% of repeated (state, action) pairs led to the same next state (DQN 1367/1386,
    PPO 794/805, random 1329/1346). Inconsistencies concentrate in tu93 (move-budget game-overs) and sk48.
  - **Fallback decisions** (exhausted state with no reachable untried state) occur only in tu93, for DQN (665 in total) and
    random (541), never for PPO. With the move counter masked, a life can run out before the return path completes.
  - **Official score is unchanged in substance.** Excluding r11l, all imp5 scores are ≤ 0.007: second-level completions
    take 200–490 actions against a 16-action human baseline, so they add almost nothing to the efficiency-weighted score.
    PPO imp5 is lower than PPO imp4 excluding r11l (0.003 vs 0.014) because its sp80 completions came later (416–956 vs
    440–488).
  - **Cost:** DQN 21 min, PPO 11 min per seed; the graph adds no measurable overhead (DQN is faster than imp4 because
    planned steps skip learner updates).
- **Decision under the fixed rule:** improvement 5 beats imp4 for both algorithms and beats random-with-graph in the mean,
  so it goes into the report as a full improvement. **Submission: PPO imp5** (most levels, 5.00; most second levels;
  3× faster than DQN). This is chosen on levels, not dev score, because the dev score is dominated by r11l luck.

### Improvement 5 — ablation without the graph (25 Sept)
- **First run invalid (bug found from the result).** `*_abl_imp5_no_graph` (imp4 + timed clock mask, no graph) reproduced
  imp4 exactly: same levels per seed (DQN 4/2/3, PPO 6/3/3), same completion times, same distinct frames in every game.
  Cause: the timed mask needs the action index within a life, which only the graph path passed to the shaper. Without it
  the mask fell back to the untimed rule, so the ablation was imp4 under another name.
- **Fix:** the non-graph path now also tracks the action index within a life and passes it to the shaper. Configs without
  `clock_mask_timed` are unaffected (bit-for-bit reproduction re-checked for baseline, imp2 and imp4). The graph path is
  unchanged, so imp5 results stand. Test added (`test_timed_clock_mask_is_active_without_graph`).
- **What the invalid run still shows:** imp4 is reproducible run to run on the same machine and seeds (all numbers identical).
- **Rerun (25–26 Sept), valid result: the graph, not the timed mask, produces improvement 5's gain.**

  | Run (dev) | Levels (per seed) | tu93 levels (sum over seeds) | Level-2 completions |
  |---|---|---|---|
  | dqn_imp4_efficient | 3.00 (4, 2, 3) | 0 | 0 |
  | dqn_abl_imp5_no_graph (imp4 + timed mask) | 3.00 (3, 2, 4) | 0 | 0 |
  | dqn_imp5_graph | 4.67 (6, 5, 3) | 4 | 2 |
  | ppo_imp4_efficient | 4.00 (6, 3, 3) | 2 | 1 |
  | ppo_abl_imp5_no_graph (imp4 + timed mask) | 3.67 (4, 3, 4) | 0 | 0 |
  | ppo_imp5_graph | 5.00 (4, 5, 6) | 5 | 2 |

  - The timed mask alone changes the runs (the ablation now differs from imp4 game by game) but not the level count:
    DQN 3.00 → 3.00, PPO 4.00 → 3.67. It completes no tu93 levels at all, where the graph gives 4 (DQN) and 5 (PPO).
  - So on dev, improvement 5's gain is attributable to the graph, and specifically to tu93, where return navigation is
    active (15% of actions).

## Held-out evaluation (25–26 Sept; 10 games never used in development; 3 seeds; run once, nothing tuned after)

Games: ar25, bp35, cd82, dc22, ft09, g50t, lf52, m0r0, tr87, vc33. Levels are summed over the 10 games.

| Run | Levels (per seed) | Games with ≥1 level | Level-2 completions | Distinct frames | Score |
|---|---|---|---|---|---|
| random | 0.33 (0, 1, 0) | 0.33 | 0 | 3980 | 0.002 |
| random_imp4_actions | 1.00 (1, 1, 1) | 1.00 | 0 | 4480 | 0.001 |
| random_imp5_graph | 1.67 (2, 1, 2) | 1.67 | 0 | 4630 | 0.001 |
| dqn_baseline / imp1 / imp2 | 0.00 / 0.33 / 0.33 | ≤ 0.33 | 0 | 4199 / 4442 / 4468 | ≤ 0.001 |
| dqn_imp3_pretrain_final | 0.00 (0, 0, 0) | 0 | 0 | 4003 | 0.000 |
| **dqn_imp4_efficient** | **3.33 (2, 4, 4)** | **2.33** | **3** | **6110** | **0.094** |
| dqn_imp5_graph | 2.67 (4, 1, 3) | 2.67 | 0 | 5681 | 0.005 |
| ppo_baseline / imp1 / imp2 | 0.67 / 0.67 / 0.33 | ≤ 0.67 | 0 | 3828 / 4275 / 4428 | ≤ 0.010 |
| ppo_imp3_pretrain_final | 0.00 (0, 0, 0) | 0 | 0 | 5127 | 0.000 |
| ppo_imp4_efficient | 2.00 (4, 1, 1) | 1.67 | 1 | 5794 | 0.003 |
| ppo_imp5_graph | 2.33 (3, 3, 1) | 2.33 | 0 | 5508 | 0.005 |

- **Improvements 1–3 do not generalise.** Baselines, shaping (imp1, imp2) and pretraining (imp3) complete 0–0.67 levels
  on held-out games, at or below plain random (0.33). Pretraining on all 15 dev games gave 0 levels for both algorithms
  (imp2 from scratch: 0.33 each). The dev_b finding (exploration transfers, goals do not) held, and on held-out games
  even the exploration benefit gave no levels.
- **Improvement 4 generalises; it is the main held-out result.** DQN imp4 3.33 and PPO imp4 2.00 levels, against 0.33
  for imp2. Both are above random over the same action abstraction (1.00). DQN imp4 also produced every held-out
  second level: vc33 (seed 0 at actions 681/713, seed 1 at 385/868) and ar25 (seed 2 at 624/788).
- **Within-game learning is visible in the second levels.** DQN seed 0 completed vc33 level 2 only 32 actions after
  level 1 (human baseline 18), against 681 actions for level 1. That single completion produces most of DQN imp4's
  held-out score (vc33 per-game score 2.26).
- **Improvement 5 does not generalise beyond improvement 4.** DQN 3.33 → 2.67 (lower), PPO 2.00 → 2.33; neither
  reached a held-out second level. Its dev gain came from tu93; no held-out game has tu93's combination of few actions
  and a short life where return navigation pays off. The one held-out game with heavy navigation (g50t, 230 planned
  steps per game) was never completed by any agent.
- **Determinism is lower on held-out games:** 87–90% of repeated (state, action) pairs were consistent (98.6% on dev).
  More inconsistency makes graph edges less reliable, which is consistent with improvement 5's weaker held-out result.
- **Normalised per game, held-out is not harder than dev overall:** levels per game dev → held-out: DQN imp4 0.20 →
  0.33, PPO imp4 0.27 → 0.20, DQN imp5 0.31 → 0.27, PPO imp5 0.33 → 0.23, random-with-graph 0.18 → 0.17. The
  generalisation failure is specific to improvements 1–3 and to improvement 5's extra gain.
- **Score:** outside vc33 and ar25 second levels, every held-out score is ≤ 0.03. ft09, which appears in the official
  starter material, was completed by almost every version including random, so it does not separate agents.

## Paired statistics (`scripts/stats.py`, levels per game, games as the unit, seeds averaged)

- **Design limit:** most games are never completed by any agent, so most per-game differences are exact ties. With
  3–4 non-tied games, the smallest possible two-sided sign-flip p-value is 0.125–0.25. **No comparison can reach
  p < 0.05 at this scale**, whatever the effect size. Bootstrap CIs over games are reported, but with so many ties
  they are optimistic. Treat both as descriptive and state this limitation in the report.
- **Comparisons whose 95% bootstrap CI excludes zero** (mean difference in levels per game [CI], games A>B / A<B / tie):
  - dev: PPO imp4 vs imp2 +0.156 [0.044, 0.289] 4/0/11; DQN imp4 vs random-imp4-actions +0.111 [0.022, 0.222] 4/0/11;
    PPO imp4 vs random-imp4-actions +0.178 [0.044, 0.356] 4/0/11; PPO imp4 vs its no-effect ablation +0.133
    [0.022, 0.267] 4/0/11; DQN imp5 vs random-with-graph +0.133 [0.022, 0.267] 4/0/11.
  - held-out: DQN imp4 vs imp2 +0.300 [0.067, 0.667] 4/0/6; DQN imp4 vs random-imp4-actions +0.233 [0.067, 0.467] 4/0/6.
- **Consistent direction, CI touching zero:** DQN/PPO imp5 vs imp4 on dev; PPO imp4 vs imp2 on held-out
  (+0.167 [0.000, 0.400]); all learner-vs-random-with-graph comparisons on held-out.
- **No difference:** every improvement 1–3 comparison, on dev and held-out; DQN vs PPO at matched versions.

## Submission decision after held-out (26 Sept): DQN imp4
Kaggle ranks by the official, efficiency-weighted score on unseen games, so the choice uses all 25 public games
(dev + held-out) and excludes r11l, where random over the same actions matches the learners by luck.

| Agent | Levels (25 games, per seed) | Level-2 completions | Games solved | Score | Score excl. r11l | Minutes per seed (Mac CPU) |
|---|---|---|---|---|---|---|
| **dqn_imp4_efficient** | 6.33 (6, 6, 7) | **3** | 9 | **0.121** | **0.042** | 41 |
| dqn_imp5_graph | 7.33 (10, 6, 6) | 2 | 11 | 0.086 | 0.006 | 33 |
| ppo_imp4_efficient | 6.00 (10, 4, 4) | 2 | 8 | 0.017 | 0.010 | 14 |
| ppo_imp5_graph | 7.33 (7, 8, 7) | 2 | 9 | 0.044 | 0.004 | 16 |
| random_imp5_graph | 4.33 (6, 3, 4) | 1 | 5 | 0.142 | 0.001 | 2 |

- **Chosen: DQN imp4.** Best score once r11l is excluded, and the best agent on held-out games (the closest analogue of
  Kaggle's hidden games). The imp5 agents complete about one more level in total, but late, so those levels earn almost
  no score. DQN imp4's second levels came quickly after the first (vc33 level 2 in 32 actions), which is what the metric
  rewards. It also does not depend on the determinism assumption, which was weaker on held-out games (87–90%).
- **Caveats, to be stated in the report:** (1) this replaces the pre-registered dev-only choice (PPO imp5) and was made
  after seeing held-out results, so DQN imp4's held-out numbers are an optimistic estimate of its performance on new
  games; (2) its score lead rests on few events: vc33 seed 0 alone contributes about 0.030 of its 0.042, and without it
  DQN imp4 (≈0.012) is roughly level with PPO imp4 (0.010); (3) DQN is about 2.6× slower than PPO, so the Kaggle rerun's
  time limit must be checked with the first submission.
- The pre-registered comparisons in the report (all dev and held-out tables) are unaffected by this choice.

### Optional, not in the main chain
Frame-stack memory (`*_opt_memory.yaml`) and object-aware clicks (`*_opt_objclick.yaml`) remain available as extra
experiments if time allows; current evidence does not make either the bottleneck.

## Failure cases to show in the report
| Run | Seed | Game | What happened | Evidence |
|---|---|---|---|---|
| dqn_baseline | 0–2 | r11l | 75.7 game-overs per 1000 actions vs 46 random; greedy repetition | games.jsonl `game_over_at` |
| all | all | cn04, ka59, ls20, … | fixed-clock game-overs, identical across agents | games.jsonl `game_overs` |
| dqn_baseline | 0 | sp80 | level 1 in 54 actions, then nothing in 946 more | games.jsonl `level_completed_at` |
