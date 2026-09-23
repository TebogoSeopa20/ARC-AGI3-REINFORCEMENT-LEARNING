"""Offline toy game with the same reset/step/FrameDataRaw contract as arc_agi's wrappers.

Used only for unit tests and smoke runs (no internet, no ARC API). Never reported as results.
Player (colour 3) moves with ACTION1-4; reaching or clicking the goal (colour 4) completes a level.
"""
from __future__ import annotations

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

MOVES = {1: (-4, 0), 2: (4, 0), 3: (0, -4), 4: (0, 4)}


class ToyArcEnv:
    def __init__(self, game_id: str = "toy0", seed: int = 0, win_levels: int = 3, level_step_limit: int = 150):
        self.game_id = game_id
        self.rng = np.random.default_rng(seed)
        self.win_levels = win_levels
        self.level_step_limit = level_step_limit
        self.levels = 0
        self.state = GameState.NOT_PLAYED
        self.level_actions: list[int] = []
        self._new_level()
        self._last: FrameDataRaw = self._frame()

    @property
    def observation_space(self) -> FrameDataRaw | None:
        return self._last

    @property
    def action_space(self) -> list[GameAction]:
        return [GameAction.from_id(i) for i in (1, 2, 3, 4, 6)]

    def _new_level(self) -> None:
        self.player = np.array([32, 4])
        self.goal = self.rng.integers(1, 15, size=2) * 4
        self.steps_in_level = 0

    def _frame(self) -> FrameDataRaw:
        g = np.zeros((64, 64), dtype=np.int64)
        gy, gx = self.goal
        g[gy:gy + 4, gx:gx + 4] = 4
        py, px = self.player
        g[py:py + 4, px:px + 4] = 3
        fd = FrameDataRaw(
            game_id=self.game_id, state=self.state, levels_completed=self.levels,
            win_levels=self.win_levels, available_actions=[1, 2, 3, 4, 6],
        )
        fd.frame = [g]
        self._last = fd
        return fd

    def reset(self) -> FrameDataRaw:
        if self.state in (GameState.NOT_PLAYED, GameState.WIN):
            self.levels = 0
        self.state = GameState.NOT_FINISHED
        self._new_level()
        return self._frame()

    def step(self, action: GameAction, data: dict | None = None, reasoning=None) -> FrameDataRaw:
        if action is GameAction.RESET:
            return self.reset()
        if self.state is not GameState.NOT_FINISHED:
            return self._frame()
        self.steps_in_level += 1
        hit = False
        if action.value in MOVES:
            self.player = np.clip(self.player + MOVES[action.value], 0, 60)
            hit = bool(np.all(self.player == self.goal))
        elif action is GameAction.ACTION6 and data:
            gy, gx = self.goal
            hit = gx <= data["x"] < gx + 4 and gy <= data["y"] < gy + 4
        if hit:
            self.level_actions.append(self.steps_in_level)
            self.levels += 1
            if self.levels >= self.win_levels:
                self.state = GameState.WIN
            else:
                self._new_level()
        elif self.steps_in_level >= self.level_step_limit:
            self.state = GameState.GAME_OVER
        return self._frame()
