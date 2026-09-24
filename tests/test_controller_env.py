from pathlib import Path

import pytest

from arcrl.agents.controller import OnlineController
from arcrl.env.arc_env import ArcAgi3Env
from arcrl.env.toy import ToyArcEnv
from arcrl.evaluation.runner import play_game
from arcrl.utils import Config

ROOT = Path(__file__).resolve().parents[1]


def test_gym_adapter_semantics():
    env = ArcAgi3Env(ToyArcEnv(), click_grid=16)
    obs, info = env.reset()
    assert obs.shape == (64, 64) and info["action_mask"].sum() > 0
    gy, gx = env.wrapper.goal
    idx = 6 + (gy // 4) * 16 + gx // 4
    _, r, term, trunc, info = env.step(idx)
    assert r == 1.0 and info["levels_completed"] == 1 and not term


def test_runner_handles_reset_game_over_and_budget():
    cfg = Config(algo="random")
    r = play_game(ToyArcEnv(level_step_limit=10), OnlineController(cfg), "toy0", budget=60)
    assert r["actions"] <= 60 and r["resets"] >= 1
    assert r["game_overs"] == len(r["game_over_at"])


def test_every_config_parses():
    for p in list((ROOT / "configs").glob("*.yaml")) + list((ROOT / "configs" / "ablations").glob("*.yaml")):
        if p.name in ("splits.yaml", "pretrain_splits.yaml"):
            continue
        cfg = Config.from_yaml(p)
        assert cfg.algo in ("dqn", "ppo", "random"), p
        assert cfg.max_actions_per_game > 0 and not cfg.extra, (p, cfg.extra)


def test_ablation_differs_by_one_component():
    imp2 = Config.from_yaml(ROOT / "configs/dqn_imp2_clockmask.yaml")
    abl = Config.from_yaml(ROOT / "configs/ablations/dqn_final_no_novelty.yaml")
    assert imp2.clock_mask and abl.clock_mask and abl.novelty_bonus == 0 and imp2.novelty_bonus > 0
    imp3 = Config.from_yaml(ROOT / "configs/dqn_imp3_pretrain.yaml")
    ctrl = Config.from_yaml(ROOT / "configs/ablations/dqn_scratch_loweps.yaml")
    assert (imp3.eps_start, imp3.split) == (ctrl.eps_start, ctrl.split) and ctrl.init_checkpoint is None


def test_pretrain_splits_are_disjoint_from_evaluation():
    from arcrl.utils import check_no_leakage, load_split

    for name in ("dqn_imp3_pretrain", "ppo_imp3_pretrain", "dqn_imp3_pretrain_final", "ppo_imp3_pretrain_final"):
        cfg = Config.from_yaml(ROOT / f"configs/{name}.yaml")
        cfg.splits_file = str(ROOT / "configs/splits.yaml")
        check_no_leakage(cfg, load_split(cfg))
    cfg = Config(pretrain_split="dev_a", splits_file=str(ROOT / "configs/splits.yaml"))
    assert set(load_split(cfg, "dev_a")) | set(load_split(cfg, "dev_b")) == set(load_split(cfg, "dev"))
    with pytest.raises(SystemExit):
        check_no_leakage(cfg, ["ka59"])


def test_pretrained_checkpoint_loads_per_seed(tmp_path):
    import torch

    cfg = Config(algo="dqn")
    src = OnlineController(cfg, seed=3)
    (tmp_path / "seed3").mkdir()
    torch.save(src.state_dict(), tmp_path / "seed3" / "final.pt")
    ctrl = OnlineController(Config(algo="dqn", init_checkpoint=str(tmp_path / "seed{seed}" / "final.pt")), seed=3)
    for a, b in zip(src.learner.q.parameters(), ctrl.learner.q.parameters()):
        assert torch.equal(a, b)
    ctrl.cfg.reset_weights_per_game = False
    ctrl.learner.t = 50
    ctrl.new_game()
    assert ctrl.learner.t == 0


def test_clock_mask_ignores_counter_but_not_movement():
    import numpy as np

    from arcrl.agents.exploration import ClockMask

    cm = ClockMask()
    for life in range(2):
        g = np.zeros((64, 64), np.uint8)
        g[0, :] = 5
        for t in range(10):
            nxt = g.copy()
            nxt[0, t] = 9
            nxt[30:32, :] = 0
            nxt[30:32, t:t + 2] = 3
            cm.update(g, nxt)
            g = nxt
    assert cm.mask[0, :10].all() and not cm.mask[30:32, :9].any()
