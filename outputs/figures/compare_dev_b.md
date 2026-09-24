# Runs on dev_b: cn04, lp85, sk48, su15, tu93

| run_name               |   seeds |   score_mean |   score_ci95 |   levels_mean |   levels_std |   unique_states_mean |   game_overs_mean |
|:-----------------------|--------:|-------------:|-------------:|--------------:|-------------:|---------------------:|------------------:|
| dqn_abl_scratch_loweps |       3 |        0.009 |        0.018 |         0.333 |        0.577 |              763.333 |            64.333 |
| dqn_baseline           |       3 |        0     |        0.001 |         0.333 |        0.577 |             1083     |            51.667 |
| dqn_imp1_explore       |       3 |        0.014 |        0.012 |         1     |        0     |             1234.33  |            62.333 |
| dqn_imp2_clockmask     |       3 |        0.014 |        0.012 |         1     |        0     |             1405.67  |            61     |
| dqn_imp3_pretrain      |       3 |        0.005 |        0.007 |         1.333 |        0.577 |             1145.67  |            58.667 |
| ppo_baseline           |       3 |        0.001 |        0.002 |         0.333 |        0.577 |             1083.33  |            58.667 |
| ppo_imp1_explore       |       3 |        0.001 |        0.002 |         0.333 |        0.577 |             1181.33  |            61.333 |
| ppo_imp2_clockmask     |       3 |        0.001 |        0.002 |         0.333 |        0.577 |             1312     |            60     |
| ppo_imp3_pretrain      |       3 |        0     |        0     |         0.333 |        0.577 |             1747.67  |            60.667 |
| random                 |       3 |        0.001 |        0.001 |         0.667 |        0.577 |             1116     |            59     |

## Level completions
- lp85 dqn_abl_scratch_loweps seed 0: levels at actions [76]
- lp85 dqn_baseline seed 1: levels at actions [375]
- lp85 dqn_imp1_explore seed 0: levels at actions [359]
- lp85 dqn_imp1_explore seed 1: levels at actions [94]
- lp85 dqn_imp1_explore seed 2: levels at actions [86]
- lp85 dqn_imp2_clockmask seed 0: levels at actions [375]
- lp85 dqn_imp2_clockmask seed 1: levels at actions [94]
- lp85 dqn_imp2_clockmask seed 2: levels at actions [86]
- lp85 dqn_imp3_pretrain seed 0: levels at actions [117]
- lp85 dqn_imp3_pretrain seed 1: levels at actions [289]
- lp85 dqn_imp3_pretrain seed 2: levels at actions [361]
- lp85 ppo_baseline seed 2: levels at actions [246]
- lp85 ppo_imp1_explore seed 2: levels at actions [246]
- lp85 ppo_imp2_clockmask seed 2: levels at actions [246]
- lp85 ppo_imp3_pretrain seed 1: levels at actions [706]
- lp85 random seed 0: levels at actions [504]
- lp85 random seed 1: levels at actions [352]
- su15 dqn_imp3_pretrain seed 0: levels at actions [841]
