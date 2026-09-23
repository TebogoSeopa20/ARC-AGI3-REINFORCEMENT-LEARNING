"""Discrete action abstraction over ARC-AGI-3's GameAction set.

Index layout: 0..5 -> ACTION1..5, ACTION7; 6.. -> ACTION6 clicks on a k x k grid of cells.
"""
from __future__ import annotations

import numpy as np

SIMPLE_IDS = [1, 2, 3, 4, 5, 7]
CLICK_ID = 6


def background(grid: np.ndarray) -> int:
    return int(np.bincount(grid.reshape(-1), minlength=16).argmax())


class ActionSpace:
    def __init__(self, click_grid: int = 8, object_clicks: bool = False):
        assert 64 % click_grid == 0
        self.k = click_grid
        self.cell = 64 // click_grid
        self.object_clicks = object_clicks
        self.n_simple = len(SIMPLE_IDS)
        self.n = self.n_simple + self.k * self.k
        self._click_xy = self._centres()

    def _centres(self) -> np.ndarray:
        c = self.cell
        r, q = np.divmod(np.arange(self.k * self.k), self.k)
        return np.stack([q * c + c // 2, r * c + c // 2], 1)

    def mask(self, available: list[int], grid: np.ndarray) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        avail = set(available) if available else set(SIMPLE_IDS + [CLICK_ID])
        for i, aid in enumerate(SIMPLE_IDS):
            m[i] = aid in avail
        self._click_xy = self._centres()
        if CLICK_ID in avail:
            m[self.n_simple:] = self._object_cells(grid) if self.object_clicks else True
        if not m.any():
            m[:] = True
        return m

    def _object_cells(self, grid: np.ndarray) -> np.ndarray:
        c, bg = self.cell, background(grid)
        fg = grid != bg
        cells = np.zeros(self.k * self.k, dtype=bool)
        for i in range(self.k * self.k):
            r, q = divmod(i, self.k)
            ys, xs = np.nonzero(fg[r * c:(r + 1) * c, q * c:(q + 1) * c])
            if len(ys):
                cells[i] = True
                j = len(ys) // 2
                self._click_xy[i] = (q * c + xs[j], r * c + ys[j])
        return cells if cells.any() else np.ones_like(cells)

    def decode(self, idx: int):
        from arcengine import GameAction

        if idx < self.n_simple:
            return GameAction.from_id(SIMPLE_IDS[idx]), None
        x, y = self._click_xy[idx - self.n_simple]
        return GameAction.ACTION6, {"x": int(x), "y": int(y)}
