#!/usr/bin/env python
"""Run one agent config over a game split for every seed, with online learning and the official scorecard.

Writes outputs/results/<run_name>/<split>_seed<k>/{games.jsonl, run_meta.json, final.pt}.
"""
import argparse
import shutil
import time
from pathlib import Path

import _common  # noqa: F401
import torch

from arcrl.agents.controller import OnlineController
from arcrl.evaluation.runner import play_game
from arcrl.evaluation.scoring import official_game_scores, reference_game_score
from arcrl.utils import Config, append_jsonl, get_logger, load_split, set_seed, write_run_meta

log = get_logger("evaluate")


def run_seed(cfg: Config, seed: int, games: list[str], out: Path) -> None:
    set_seed(seed)
    ctrl = OnlineController(cfg, seed=seed)
    toy = cfg.split == "toy"
    arc = card = None
    if not toy:
        arc = _common.make_arcade()
        card = arc.create_scorecard(tags=[cfg.run_name, cfg.split, f"seed{seed}"])
    results, t0 = [], time.time()
    for gid in games:
        if toy:
            from arcrl.env.toy import ToyArcEnv

            env = ToyArcEnv(gid, seed=seed)
        else:
            env = arc.make(gid, seed=seed, scorecard_id=card)
            if env is None:
                log.warning("could not create %s, skipping", gid)
                continue
        r = play_game(env, ctrl, gid, cfg.max_actions_per_game)
        if toy:
            lv = [(True, a, 10) for a in env.level_actions]
            lv += [(False, 0, 10)] * (env.win_levels - len(lv))
            r["official_score"] = reference_game_score(lv)
            r["score_source"] = "toy_reference"
        results.append(r)
        log.info("%s seed=%d %s: levels=%d/%s actions=%d state=%s (%.0fs)", cfg.run_name, seed, gid,
                 r["levels_completed"], r["win_levels"] or "?", r["actions"], r["final_state"], r["wall_time_s"])
    official = {} if toy else official_game_scores(arc, card)
    for r in results:
        if not toy:
            o = official.get(r["game_id"].split("-")[0], {})
            r["official_score"] = o.get("score", 0.0)
            r["official"] = o
            r["score_source"] = "arc_agi_scorecard"
            r["scorecard_id"] = card
        r.update(run_name=cfg.run_name, split=cfg.split, seed=seed, algo=cfg.algo)
        append_jsonl(r, out / "games.jsonl")
    write_run_meta(cfg, out, seed=seed, games=games, eval_wall_time_s=round(time.time() - t0, 1),
                   official_total=official.get("__total__"))
    if cfg.algo != "random":
        torch.save(ctrl.state_dict(), out / "final.pt")
    if arc is not None:
        arc.close_scorecard(card)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default=None, help="override config split: dev | heldout | toy")
    ap.add_argument("--seeds", default=None, help="comma list, overrides config")
    ap.add_argument("--games", default=None, help="comma list, overrides split")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.split:
        cfg.split = args.split
    if args.seeds:
        cfg.seeds = [int(s) for s in args.seeds.split(",")]
    if args.budget:
        cfg.max_actions_per_game = args.budget
    games = args.games.split(",") if args.games else load_split(cfg)

    for seed in cfg.seeds:
        out = Path(cfg.output_dir) / "results" / cfg.run_name / f"{cfg.split}_seed{seed}"
        if (out / "games.jsonl").exists():
            if not args.overwrite:
                log.info("skip existing %s", out)
                continue
            shutil.rmtree(out)
        run_seed(cfg, seed, games, out)


if __name__ == "__main__":
    main()
