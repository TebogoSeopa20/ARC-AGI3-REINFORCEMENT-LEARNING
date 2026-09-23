#!/usr/bin/env python
"""Measure actions/second per config on one game and project the cost of the full study.

Run on the hardware you will use (Kaggle T4) before fixing the protocol budget.
"""
import argparse
import time

import _common

from arcrl.agents.controller import OnlineController
from arcrl.evaluation.runner import play_game
from arcrl.utils import Config, load_split, resolve_device, set_seed

CONFIGS = ["random", "dqn_baseline", "dqn_imp2_memory", "ppo_baseline", "ppo_imp2_memory"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=None, help="default: first dev game")
    ap.add_argument("--actions", type=int, default=300)
    ap.add_argument("--n-configs", type=int, default=13, help="configs in the full study (versions + ablations)")
    args = ap.parse_args()
    base = Config.from_yaml(_common.ROOT / "configs" / "base.yaml")
    games = load_split(base) + load_split(Config.from_dict({"split": "heldout"}))
    gid = args.game or load_split(base)[0]
    arc = _common.make_arcade()
    print(f"device={resolve_device()} game={gid} actions={args.actions}")
    rates = {}
    for name in CONFIGS:
        cfg = Config.from_yaml(_common.ROOT / "configs" / f"{name}.yaml")
        set_seed(0)
        t = time.time()
        r = play_game(arc.make(gid, seed=0), OnlineController(cfg, seed=0), gid, args.actions)
        rates[name] = r["actions"] / (time.time() - t)
        print(f"{name:18} {rates[name]:7.1f} actions/s  levels={r['levels_completed']}")
    slow = min(v for k, v in rates.items() if k != "random")
    per_run = base.max_actions_per_game * len(games) / slow / 3600
    total = per_run * len(base.seeds) * args.n_configs
    print(f"\nworst case: {per_run:.2f} h per config-seed over {len(games)} games at "
          f"{base.max_actions_per_game} actions; full study ~{total:.1f} GPU-h "
          f"({len(base.seeds)} seeds x {args.n_configs} configs). Kaggle GPU quota is ~30 h/week.")


if __name__ == "__main__":
    main()
