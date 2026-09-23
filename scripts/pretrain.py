#!/usr/bin/env python
"""Optional offline phase: train one set of weights sequentially across dev games, save a checkpoint.

Evaluation configs can then set `init_checkpoint` so each held-out game starts from these weights
(online learning still continues during evaluation). Never run on held-out games.
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
import torch

from arcrl.agents.controller import OnlineController
from arcrl.evaluation.runner import play_game
from arcrl.utils import Config, append_jsonl, get_logger, load_split, set_seed, write_run_meta

log = get_logger("pretrain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    assert cfg.split != "heldout", "pretraining on held-out games leaks evaluation data"
    cfg.reset_weights_per_game = False
    set_seed(args.seed)
    games = load_split(cfg)
    ctrl = OnlineController(cfg, seed=args.seed)
    out = Path(cfg.output_dir) / "checkpoints" / cfg.run_name
    write_run_meta(cfg, out, seed=args.seed, games=games, passes=args.passes)
    arc = None if cfg.split == "toy" else _common.make_arcade()
    for p in range(args.passes):
        for gid in games:
            if arc is None:
                from arcrl.env.toy import ToyArcEnv

                env = ToyArcEnv(gid, seed=args.seed + p)
            else:
                env = arc.make(gid, seed=args.seed + p)
            r = play_game(env, ctrl, gid, cfg.max_actions_per_game)
            append_jsonl({"pass": p, **{k: r[k] for k in ("game_id", "levels_completed", "actions", "wall_time_s")}},
                         out / "pretrain_log.jsonl")
            log.info("pass %d %s levels=%d", p, gid, r["levels_completed"])
        torch.save(ctrl.state_dict(), out / f"pass{p}.pt")
    torch.save(ctrl.state_dict(), out / "final.pt")
    log.info("saved %s", out / "final.pt")


if __name__ == "__main__":
    main()
