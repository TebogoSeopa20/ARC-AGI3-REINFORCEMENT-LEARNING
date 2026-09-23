"""Training-time reward shaping: level progress, frame-change bonus, count-based novelty.

Novelty follows Bellemare et al. (2016) with exact counts over frame hashes: beta / sqrt(N(s')).
Counts live per game and are reset between games (no cross-game leakage).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from arcrl.env.obs import grid_hash
from arcrl.utils import Config


class RewardShaper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.counts: Counter[bytes] = Counter()

    def reset_game(self) -> None:
        self.counts.clear()

    def __call__(self, prev_grid: np.ndarray, grid: np.ndarray, d_levels: int, game_over: bool) -> tuple[float, dict]:
        c = self.cfg
        parts = {"level": c.level_reward * d_levels}
        if game_over:
            parts["game_over"] = -c.game_over_penalty
        if c.change_bonus:
            parts["change"] = c.change_bonus * (1.0 if not np.array_equal(prev_grid, grid) else -1.0)
        if c.novelty_bonus:
            h = grid_hash(grid)
            self.counts[h] += 1
            parts["novelty"] = c.novelty_bonus / np.sqrt(self.counts[h])
        return float(sum(parts.values())), parts
