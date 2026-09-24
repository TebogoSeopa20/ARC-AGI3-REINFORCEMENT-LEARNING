"""Training-time reward shaping: level progress, frame-change bonus, count-based novelty.

Novelty follows Bellemare et al. (2016) with exact counts over frame hashes: beta / sqrt(N(s')).
Counts live per game and are reset between games (no cross-game leakage).

ClockMask (improvement 2) removes clock-like pixels before hashing and change detection. A pixel is
clock-like once it has made the same colour transition at least twice in ordinary steps and never any
other transition, as with a move-budget bar that fills one pixel per action and refills on RESET.
Level-change transitions are ignored. The estimate is online and per game; nothing is game-specific.
Known false positive: a pixel an agent changes identically once per life (e.g. the end cell of a repeated
path) is also masked.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from arcrl.env.obs import grid_hash
from arcrl.utils import Config

MASK_VALUE = 16


class ClockMask:
    """timed=True (improvement 5) also requires each repeat to happen at the same action index within a life,
    so a counter that ticks once per action is masked, but progress the agent causes at varying times is not."""

    def __init__(self, min_repeats: int = 2, timed: bool = False):
        self.min_repeats = min_repeats
        self.timed = timed
        self.src = np.full((64, 64), -1, np.int16)
        self.dst = np.full((64, 64), -1, np.int16)
        self.when = np.full((64, 64), -1, np.int32)
        self.count = np.zeros((64, 64), np.int32)
        self.bad = np.zeros((64, 64), bool)

    def update(self, prev: np.ndarray, grid: np.ndarray, t: int | None = None) -> None:
        d = prev != grid
        if not d.any():
            return
        new = d & (self.src < 0)
        self.src[new], self.dst[new] = prev[new], grid[new]
        same = d & (self.src == prev) & (self.dst == grid)
        if self.timed and t is not None:
            self.when[new] = t
            same &= self.when == t
        self.count[same] += 1
        self.bad |= d & ~same

    @property
    def mask(self) -> np.ndarray:
        return (self.count >= self.min_repeats) & ~self.bad

    def apply(self, grid: np.ndarray) -> np.ndarray:
        m = self.mask
        if not m.any():
            return grid
        g = grid.copy()
        g[m] = MASK_VALUE
        return g


class RewardShaper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.counts: Counter[bytes] = Counter()
        self.clock = ClockMask(timed=cfg.clock_mask_timed) if cfg.clock_mask else None
        self.last_changed = False

    def key(self, grid: np.ndarray) -> bytes:
        return grid_hash(self.clock.apply(grid) if self.clock is not None else grid)

    def reset_game(self) -> None:
        self.counts.clear()
        self.clock = ClockMask(timed=self.cfg.clock_mask_timed) if self.cfg.clock_mask else None

    def __call__(self, prev_grid: np.ndarray, grid: np.ndarray, d_levels: int, game_over: bool,
                 t: int | None = None) -> tuple[float, dict]:
        c = self.cfg
        parts = {"level": c.level_reward * d_levels}
        if game_over:
            parts["game_over"] = -c.game_over_penalty
        p, g = prev_grid, grid
        if self.clock is not None:
            if d_levels == 0:
                self.clock.update(prev_grid, grid, t)
            p, g = self.clock.apply(prev_grid), self.clock.apply(grid)
        self.last_changed = d_levels > 0 or not np.array_equal(p, g)
        if c.change_bonus:
            parts["change"] = c.change_bonus * (1.0 if self.last_changed else -1.0)
        if c.novelty_bonus:
            h = grid_hash(g)
            self.counts[h] += 1
            parts["novelty"] = c.novelty_bonus / np.sqrt(self.counts[h])
        return float(sum(parts.values())), parts
