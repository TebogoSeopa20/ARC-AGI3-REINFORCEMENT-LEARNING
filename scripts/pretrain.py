#!/usr/bin/env python
"""Offline phase of improvement 3: train one set of weights sequentially across the pretraining games.

Writes outputs/checkpoints/<run_name>/seed<k>/{final.pt, pass<p>.pt, pretrain_log.jsonl, run_meta.json}.
The evaluation config loads it through init_checkpoint (with {seed}) and keeps learning online on unseen games.
Refuses any pretraining game that is held out or in the config's own evaluation split.
"""
import argparse
import copy
import time
from pathlib import Path

import _common
import torch

from arcrl.agents.controller import OnlineController
from arcrl.evaluation.runner import play_game
from arcrl.utils import Config, append_jsonl, get_logger, load_split, set_seed, write_run_meta

log = get_logger("pretrain")


def pretrain_seed(cfg: Config, seed: int, games: list[str], out: Path) -> None:
    pcfg = copy.deepcopy(cfg)
    pcfg.init_checkpoint = None
    pcfg.reset_weights_per_game = False
    for k in ("eps_start", "eps_decay_steps"):
        base = Config()
        setattr(pcfg, k, getattr(base, k))
    set_seed(seed)
    ctrl = OnlineController(pcfg, seed=seed)
    arc = None if cfg.pretrain_split.startswith("toy") else _common.make_arcade()
    t0 = time.time()
    for p in range(cfg.pretrain_passes):
        for gid in games:
            if arc is None:
                from arcrl.env.toy import ToyArcEnv

                env = ToyArcEnv(gid, seed=seed + p)
            else:
                env = arc.make(gid, seed=seed + 100 * (p + 1))
            r = play_game(env, ctrl, gid, cfg.max_actions_per_game)
            append_jsonl({"pass": p, **{k: r[k] for k in ("game_id", "levels_completed", "actions", "unique_states",
                                                          "change_rate", "wall_time_s")}}, out / "pretrain_log.jsonl")
            log.info("seed=%d pass=%d %s levels=%d (%.0fs)", seed, p, gid, r["levels_completed"], r["wall_time_s"])
        torch.save(ctrl.state_dict(), out / f"pass{p}.pt")
    torch.save(ctrl.state_dict(), out / "final.pt")
    write_run_meta(pcfg, out, seed=seed, pretrain_games=games, passes=cfg.pretrain_passes,
                   pretrain_wall_time_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the evaluation config (its pretrain_split is used)")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    assert cfg.pretrain_split, "config has no pretrain_split"
    games = load_split(cfg, cfg.pretrain_split)
    forbidden = set(load_split(cfg, cfg.split))
    if not cfg.pretrain_split.startswith("toy"):
        forbidden |= set(load_split(cfg, "heldout"))
    bad = forbidden & set(games)
    if bad:
        raise SystemExit(f"refusing to pretrain on evaluation/held-out games: {sorted(bad)}")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg.seeds
    for seed in seeds:
        out = Path(cfg.output_dir) / "checkpoints" / cfg.run_name / f"seed{seed}"
        if (out / "final.pt").exists() and not args.overwrite:
            log.info("skip existing %s", out)
            continue
        out.mkdir(parents=True, exist_ok=True)
        pretrain_seed(cfg, seed, games, out)
        log.info("saved %s", out / "final.pt")


if __name__ == "__main__":
    main()
