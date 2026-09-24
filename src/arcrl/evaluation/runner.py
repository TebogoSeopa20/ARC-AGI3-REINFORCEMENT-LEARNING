"""Play one game with online learning under a fixed action budget; log everything, including failures."""
from __future__ import annotations

import time

from arcengine import GameAction

from arcrl.agents.controller import OnlineController
from arcrl.env.obs import Obs, grid_hash


def play_game(env, ctrl: OnlineController, game_id: str, budget: int, log_every: int = 50) -> dict:
    ctrl.new_game()
    t0 = time.time()
    start = getattr(env, "observation_space", None)
    obs = Obs.from_frame_data(start) if start is not None else Obs(grid=None, state="NOT_PLAYED", levels=0)
    actions = resets = game_overs = 0
    level_at, game_over_at, curve, losses = [], [], [], []
    max_levels, win_levels = 0, 0
    moves = changed = 0
    seen: set[bytes] = set()
    while actions < budget:
        ga, data = ctrl.act(obs)
        if ga is None:
            break
        fd = env.reset() if ga is GameAction.RESET else env.step(ga, data=data)
        actions += 1
        resets += ga is GameAction.RESET
        if fd is None:
            break
        new = Obs.from_frame_data(fd)
        if ga is not GameAction.RESET and obs.grid is not None:
            moves += 1
            changed += not (new.grid == obs.grid).all()
        seen.add(grid_hash(new.grid))
        if new.levels > max_levels:
            level_at.extend([actions] * (new.levels - max_levels))
            max_levels = new.levels
        if new.state == "GAME_OVER" and obs.state != "GAME_OVER":
            game_overs += 1
            game_over_at.append(actions)
        win_levels = max(win_levels, new.win_levels)
        obs = new
        if actions % log_every == 0:
            curve.append((actions, max_levels))
            if ctrl.last_stats:
                losses.append({"t": actions, **ctrl.last_stats})
        if obs.state == "WIN":
            ctrl.act(obs)
            break
    curve.append((actions, max_levels))
    return {
        "game_id": game_id,
        "final_state": obs.state,
        "win": obs.state == "WIN",
        "levels_completed": max_levels,
        "win_levels": win_levels,
        "actions": actions,
        "resets": resets,
        "game_overs": game_overs,
        "level_completed_at": level_at,
        "game_over_at": game_over_at,
        "change_rate": round(changed / moves, 4) if moves else 0.0,
        "unique_states": len(seen),
        "curve": curve,
        "learner_stats": losses,
        "shaped_return": ctrl.shaped_return,
        "effect_rate": round(ctrl.effective_steps / ctrl.steps, 4) if ctrl.steps else 0.0,
        "pruned_choices": ctrl.pruned_choices,
        "wall_time_s": round(time.time() - t0, 2),
    }
