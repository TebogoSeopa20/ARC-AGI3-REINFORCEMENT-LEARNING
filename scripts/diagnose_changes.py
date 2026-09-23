#!/usr/bin/env python
"""Where do frames change under a random policy? Dev games only.

Tests whether per-step frame changes are driven by the agent or by a HUD/clock (e.g. a move counter),
which would make frame-hash novelty and the frame-change bonus uninformative.
Writes outputs/figures/change_heatmaps.png and outputs/figures/change_diagnostics.csv.
"""
import argparse
from pathlib import Path

import _common
import numpy as np
import pandas as pd

from arcrl.agents.exploration import ClockMask
from arcrl.env.actions import ActionSpace
from arcrl.env.obs import Obs, grid_hash
from arcrl.utils import Config, load_split


def coarse(grid: np.ndarray, k: int = 8) -> bytes:
    c = 64 // k
    cells = grid.reshape(k, c, k, c).transpose(0, 2, 1, 3).reshape(k, k, -1)
    modes = np.array([[np.bincount(v, minlength=16).argmax() for v in row] for row in cells], np.uint8)
    return modes.tobytes()


def run_game(arc, gid: str, steps: int, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    from arcengine import GameAction

    env = arc.make(gid, seed=0)
    sp = ActionSpace(8)
    obs = Obs.from_frame_data(env.observation_space)
    freq = np.zeros((64, 64))
    n_changed, click_changed, hashes, coarse_states = [], [], set(), set()
    clock, masked_hashes, masked_changed = ClockMask(), set(), []
    moves = 0
    for _ in range(steps):
        if obs.state in ("GAME_OVER", "NOT_PLAYED"):
            obs = Obs.from_frame_data(env.reset())
            continue
        if obs.state == "WIN":
            break
        idx = int(rng.choice(np.flatnonzero(sp.mask(obs.available, obs.grid))))
        ga, data = sp.decode(idx)
        new = Obs.from_frame_data(env.step(ga, data=data))
        if new.state == obs.state == "NOT_FINISHED" and new.levels == obs.levels:
            d = new.grid != obs.grid
            freq += d
            n_changed.append(int(d.sum()))
            if ga is GameAction.ACTION6:
                click_changed.append(int(d.sum()))
            clock.update(obs.grid, new.grid)
            masked_changed.append(int((d & ~clock.mask).sum()))
            moves += 1
        hashes.add(grid_hash(new.grid))
        coarse_states.add(coarse(new.grid))
        masked_hashes.add(grid_hash(clock.apply(new.grid)))
        obs = new
    n = np.asarray(n_changed) if n_changed else np.zeros(1)
    f = freq / max(1, moves)
    row = {
        "game_id": gid,
        "moves": moves,
        "change_rate": round(float((n > 0).mean()), 3),
        "median_px_changed": float(np.median(n)),
        "p90_px_changed": float(np.percentile(n, 90)),
        "min_px_changed_when_changed": int(n[n > 0].min()) if (n > 0).any() else 0,
        "click_change_rate": round(float((np.asarray(click_changed) > 0).mean()), 3) if click_changed else None,
        "px_changing_every_step": int((f > 0.95).sum()),
        "px_changing_over_half": int((f > 0.5).sum()),
        "rows_with_hot_px": sorted(set(np.nonzero(f > 0.5)[0].tolist()))[:12],
        "unique_frames": len(hashes),
        "unique_coarse8": len(coarse_states),
        "clock_px": int(clock.mask.sum()),
        "change_rate_clockmasked": round(float((np.asarray(masked_changed) > 0).mean()), 3) if masked_changed else 0.0,
        "unique_frames_clockmasked": len(masked_hashes),
    }
    return row, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    games = load_split(Config(split="dev"))
    arc = _common.make_arcade()
    rng = np.random.default_rng(0)
    rows, maps = [], []
    for g in games:
        r, f = run_game(arc, g, args.steps, rng)
        rows.append(r)
        maps.append(f)
        print(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "change_diagnostics.csv", index=False)
    print(df.drop(columns=["rows_with_hot_px"]).to_string(index=False))
    cols = 5
    fig, axes = plt.subplots(int(np.ceil(len(games) / cols)), cols, figsize=(2.2 * cols, 2.3 * np.ceil(len(games) / cols)))
    for ax, g, f in zip(axes.flat, games, maps):
        ax.imshow(f, cmap="magma", vmin=0, vmax=1)
        ax.set_title(g, fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(games):]:
        ax.axis("off")
    fig.suptitle("Per-pixel change frequency per step (random policy)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "change_heatmaps.png", dpi=150)
    print("wrote", out / "change_heatmaps.png")


if __name__ == "__main__":
    main()
