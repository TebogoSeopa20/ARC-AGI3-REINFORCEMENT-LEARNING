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
| 25 Sept | dqn_abl_no_novelty | ablations/dqn_final_no_novelty.yaml | dev | 0,1,2 | 0.110 | 2.33 ± 0.58 | 16 min | unique 6612 |
| 25 Sept | dqn_abl_no_change | ablations/dqn_final_no_change.yaml | dev | 0,1,2 | 0.110 | 2.33 ± 0.58 | 17 min | unique 6803 |
| 25 Sept | ppo_abl_no_novelty | ablations/ppo_final_no_novelty.yaml | dev | 0,1,2 | 0.003 | 2.00 ± 1.00 | 7 min | unique 6793; new: cn04 s2@971 |
| 25 Sept | ppo_abl_no_change | ablations/ppo_final_no_change.yaml | dev | 0,1,2 | 0.003 | 2.00 ± 1.00 | 7 min | unique 6427 |

Improvement 3 runs (dev_b = cn04, lp85, sk48, su15, tu93; levels out of 5 games' worth; imp1/imp2/baselines pooled from dev runs):

| Date | Run | Split | Seeds | Score | Levels on dev_b | Distinct frames | Eval time / seed | Pretrain time / seed |
|---|---|---|---|---|---|---|---|---|
| 25 Sept | random | dev_b | 0,1,2 | 0.001 | 0.67 ± 0.58 | 1116 | | |
| 25 Sept | dqn_imp2_clockmask | dev_b | 0,1,2 | 0.014 | 1.00 ± 0.00 | 1406 | | |
| 25 Sept | dqn_imp3_pretrain | dev_b | 0,1,2 | 0.005 | 1.33 ± 0.58 | 1146 | 5.6 min | 23 min (2 passes × 10 games) |
| 25 Sept | dqn_abl_scratch_loweps | dev_b | 0,1,2 | 0.009 | 0.33 ± 0.58 | 763 | 5.8 min | |
| 25 Sept | ppo_imp2_clockmask | dev_b | 0,1,2 | 0.001 | 0.33 ± 0.58 | 1312 | | |
| 25 Sept | ppo_imp3_pretrain | dev_b | 0,1,2 | 0.000 | 0.33 ± 0.58 | 1748 | 2.3 min | 9.6 min |

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

### Ablations of improvement 2 (25 Sept, dev, 3 seeds; each removes one shaping term, clock mask kept)

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
- **Result, DQN (25 Sept): pretrained weights transfer an exploration prior; no clear gain in levels or efficiency.**
  - Levels on dev_b 1.00 → 1.33 ± 0.58. The extra level is **su15 seed 0 at action 841, the first su15 completion by
    any agent in any run**. One event, so suggestive only.
  - Actions to first completion did not improve: lp85 at 117/289/361 against 375/94/86 for imp2 (mean 256 vs 185).
  - Attribution: the ε-ablation (same ε = 0.3 schedule, no pretraining) collapses to 0.33 levels and 763 distinct frames.
    Pretrained weights under that schedule restore 1146 frames and 1.33 levels. The pretrained network therefore carries
    useful exploratory behaviour into unseen games; without it, low ε alone is harmful.
  - Against imp2 at ε = 1.0, exploration is lower (1146 vs 1406 frames), so pretraining plus low ε roughly substitutes for
    random exploration rather than improving on it.
- **Result, PPO (25 Sept): much broader exploration on unseen games, no gain in levels.**
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

### Decision on improvement 4 (25 Sept, revised the same day)
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
- **Result:**

### Optional, not in the main chain
Frame-stack memory (`*_opt_memory.yaml`) and object-aware clicks (`*_opt_objclick.yaml`) remain available as extra
experiments if time allows; current evidence does not make either the bottleneck.

## Failure cases to show in the report
| Run | Seed | Game | What happened | Evidence |
|---|---|---|---|---|
| dqn_baseline | 0–2 | r11l | 75.7 game-overs per 1000 actions vs 46 random; greedy repetition | games.jsonl `game_over_at` |
| all | all | cn04, ka59, ls20, … | fixed-clock game-overs, identical across agents | games.jsonl `game_overs` |
| dqn_baseline | 0 | sp80 | level 1 in 54 actions, then nothing in 946 more | games.jsonl `level_completed_at` |
