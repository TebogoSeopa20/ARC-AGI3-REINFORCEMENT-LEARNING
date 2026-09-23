#!/usr/bin/env python
"""Kaggle-parity check: run agent/my_agent.py through the ARC-AGI-3-Agents framework loop locally.

Adapted from arcprize/ARC-AGI-3-Kaggle-Starter (scripts/play_local.py). Use scripts/evaluate.py
for experiments; use this only to confirm the submission agent behaves the same inside the framework.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

import _common

VENDOR = _common.ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))


def load_my_agent_class():
    spec = importlib.util.spec_from_file_location("user_agent_module", _common.ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default=None, help="comma list; default all")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--render", default=None, choices=[None, "terminal"])
    args = p.parse_args()
    arc = _common.make_arcade()
    envs = arc.get_environments()
    ids = [e.game_id.split("-")[0] for e in envs]
    if args.game:
        ids = [g for g in ids if g in {x.strip() for x in args.game.split(",")}]
    MyAgent = load_my_agent_class()
    if args.max_steps:
        MyAgent.MAX_ACTIONS = args.max_steps
    rows = []
    for gid in ids:
        env = arc.make(gid, render_mode=args.render)
        if env is None:
            continue
        agent = MyAgent(card_id="local-dev", game_id=gid, agent_name=f"MyAgent.local.{gid}",
                        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["local-dev"])
        agent.main()
        f = agent.frames[-1]
        rows.append((gid, f.state, f.levels_completed, agent.action_counter))
        print(f"{gid}: state={f.state} levels={f.levels_completed} actions={agent.action_counter}")
    print("scorecard score:", arc.get_scorecard().score)


if __name__ == "__main__":
    main()
