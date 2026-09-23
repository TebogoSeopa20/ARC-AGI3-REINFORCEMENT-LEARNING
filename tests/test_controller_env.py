from pathlib import Path

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
        if p.name == "splits.yaml":
            continue
        cfg = Config.from_yaml(p)
        assert cfg.algo in ("dqn", "ppo", "random"), p
        assert cfg.max_actions_per_game > 0 and not cfg.extra, (p, cfg.extra)


def test_ablation_differs_by_one_component():
    imp2 = Config.from_yaml(ROOT / "configs/dqn_imp2_memory.yaml")
    abl = Config.from_yaml(ROOT / "configs/ablations/dqn_imp2_no_novelty.yaml")
    assert abl.frame_stack == imp2.frame_stack and abl.novelty_bonus == 0 and imp2.novelty_bonus > 0
