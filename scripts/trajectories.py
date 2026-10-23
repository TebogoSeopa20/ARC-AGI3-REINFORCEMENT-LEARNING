#!/usr/bin/env python
"""Replay one logged seed run exactly and save frames from one game as a figure strip.

The run is replayed over the same games in the same order as scripts/evaluate.py, because the torch random
state carries across games; runs on the same CPU reproduce the logged results exactly. The script checks the
replay against games.jsonl and stops after the target game.

  python scripts/trajectories.py --config configs/dqn_imp4_efficient.yaml --split heldout --seed 0 --game vc33 --mode levels
  python scripts/trajectories.py --config configs/ppo_imp5_graph.yaml --split dev --seed 1 --game tu93 --mode levels
  python scripts/trajectories.py --config configs/dqn_baseline.yaml --split dev --seed 0 --game r11l --mode gameovers
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import _common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from arcengine import GameAction
from matplotlib.colors import ListedColormap

from arcrl.agents.controller import OnlineController
from arcrl.env.obs import Obs
from arcrl.evaluation.runner import play_game
from arcrl.utils import Config, load_split, set_seed

COLORS = ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000", "#E53AA3", "#FF7BCC",
          "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00", "#FF851B", "#921231", "#4FCC30", "#A356D6"]


class Recorder:
    def __init__(self, env):
        self.env, self.grids, self.actions = env, [], []

    @property
    def observation_space(self):
        return self.env.observation_space

    def _keep(self, fd, label):
        if fd is not None:
            self.grids.append(Obs.from_frame_data(fd).grid.copy())
            self.actions.append(label)
        return fd

    def reset(self):
        return self._keep(self.env.reset(), "RESET")

    def step(self, action, data=None, reasoning=None):
        label = action.name + (f"({data['x']},{data['y']})" if data else "")
        return self._keep(self.env.step(action, data=data), label)


def logged(cfg, split, seed, game):
    f = Path(cfg.output_dir) / "results" / cfg.run_name / f"{split}_seed{seed}" / "games.jsonl"
    if f.exists():
        for line in f.open():
            r = json.loads(line)
            if r["game_id"] == game:
                return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--mode", choices=["levels", "gameovers"], default="levels")
    ap.add_argument("--frames", default=None, help="comma list of action indices (overrides --mode)")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.split = args.split
    games = load_split(cfg)
    assert args.game in games, f"{args.game} not in split {args.split}"
    set_seed(args.seed)
    ctrl = OnlineController(cfg, seed=args.seed)
    arc = _common.make_arcade()
    card = arc.create_scorecard(tags=["trajectory-replay"])
    rec = r = None
    for gid in games:
        env = arc.make(gid, seed=args.seed, scorecard_id=card)
        if gid == args.game:
            rec = Recorder(env)
            r = play_game(rec, ctrl, gid, cfg.max_actions_per_game)
            break
        play_game(env, ctrl, gid, cfg.max_actions_per_game)
        print("replayed", gid)

    ref = logged(cfg, args.split, args.seed, args.game)
    check = {k: (r[k], ref[k] if ref else None) for k in ("levels_completed", "level_completed_at", "game_overs")}
    print("replay vs log:", check)
    if ref and r["level_completed_at"] != ref["level_completed_at"]:
        print("WARNING: replay differs from the logged run (different machine or code?)")

    acts = rec.actions
    rep = np.mean([a == b for a, b in zip(acts[1:], acts[:-1]) if a != "RESET"]) if len(acts) > 1 else 0
    top = Counter(a for a in acts if a != "RESET").most_common(3)
    print(f"actions={len(acts)} repeated-consecutive-rate={rep:.2f} top={top}")

    if args.frames:
        idx = [int(i) for i in args.frames.split(",")]
    elif args.mode == "levels":
        idx = [1]
        for t in r["level_completed_at"]:
            idx += [t - 1, t]
    else:
        idx = [1] + [t - 1 for t in r["game_over_at"][:4]]
    idx = [i for i in dict.fromkeys(idx) if 1 <= i <= len(rec.grids)]

    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(1, len(idx), figsize=(1.25 * len(idx), 1.55))
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(rec.grids[i - 1], cmap=cmap, vmin=0, vmax=15, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={i}", fontsize=7)
        ax.set_xlabel(rec.actions[i - 1].replace("ACTION", "A"), fontsize=5.5)
    name = f"traj_{cfg.run_name}_{args.split}_s{args.seed}_{args.game}"
    fig.tight_layout(pad=0.3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    (out / f"{name}.json").write_text(json.dumps({"frames": idx, "actions": [rec.actions[i - 1] for i in idx],
                                                  "check": check, "repeat_rate": rep, "top_actions": top}, indent=1))
    print("wrote", out / f"{name}.pdf")


if __name__ == "__main__":
    main()
