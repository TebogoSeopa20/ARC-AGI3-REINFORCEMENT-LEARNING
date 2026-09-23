"""Gymnasium adapter over an arc_agi EnvironmentWrapper (or ToyArcEnv).

Semantics (documented in docs/PROTOCOL.md):
  observation : uint8 (64, 64) grid, last animation frame of the step
  action      : Discrete(6 + k*k) via ActionSpace; illegal actions -> info["action_mask"]
  reward      : +1 per level completed on this step (unshaped; shaping lives in the controller)
  terminated  : state is WIN or GAME_OVER
  truncated   : max_steps actions taken since reset()
  reset()     : sends GameAction.RESET (counts as an action on the official scorecard)

The learners in arcrl.agents do not require this class (they drive arc_agi directly so the same
code runs inside the Kaggle framework); it exists for library interoperability and testing.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcrl.env.actions import ActionSpace
from arcrl.env.obs import Obs


class ArcAgi3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, wrapper, click_grid: int = 8, object_clicks: bool = False, max_steps: int = 1000):
        self.wrapper = wrapper
        self.actions = ActionSpace(click_grid, object_clicks)
        self.max_steps = max_steps
        self.observation_space = spaces.Box(0, 15, shape=(64, 64), dtype=np.uint8)
        self.action_space = spaces.Discrete(self.actions.n)
        self._obs: Obs | None = None
        self._t = 0

    def _info(self) -> dict:
        o = self._obs
        return {
            "action_mask": self.actions.mask(o.available, o.grid),
            "levels_completed": o.levels,
            "state": o.state,
            "win_levels": o.win_levels,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = Obs.from_frame_data(self.wrapper.reset())
        self._t = 0
        return self._obs.grid, self._info()

    def step(self, action: int):
        prev = self._obs
        ga, data = self.actions.decode(int(action))
        self._obs = Obs.from_frame_data(self.wrapper.step(ga, data=data))
        self._t += 1
        reward = float(max(0, self._obs.levels - prev.levels))
        terminated = self._obs.state in ("WIN", "GAME_OVER")
        truncated = self._t >= self.max_steps and not terminated
        return self._obs.grid, reward, terminated, truncated, self._info()
