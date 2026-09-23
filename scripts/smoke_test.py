#!/usr/bin/env python
"""Minimal end-to-end check: random, DQN and PPO each play a real game (or the toy game with --toy)."""
import argparse

import _common  # noqa: F401

from arcrl.agents.controller import OnlineController
from arcrl.evaluation.runner import play_game
from arcrl.utils import Config, package_versions, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--toy", action="store_true")
    args = ap.parse_args()
    print(package_versions())
    arc = None if args.toy else _common.make_arcade()
    for algo in ["random", "dqn", "ppo"]:
        set_seed(0)
        cfg = Config(algo=algo, rollout_len=64, warmup_steps=32)
        if args.toy:
            from arcrl.env.toy import ToyArcEnv

            env = ToyArcEnv("toy0", seed=0)
        else:
            env = arc.make(args.game, seed=0)
        r = play_game(env, OnlineController(cfg, seed=0), args.game, args.steps)
        print(f"{algo:6} levels={r['levels_completed']} actions={r['actions']} "
              f"state={r['final_state']} resets={r['resets']} t={r['wall_time_s']}s stats={r['learner_stats'][-1:]}")
    if arc is not None:
        print("scorecard:", arc.get_scorecard().score)


if __name__ == "__main__":
    main()
