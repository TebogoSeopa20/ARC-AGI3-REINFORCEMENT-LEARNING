"""Online controller: one object drives a learner through a game, locally or inside the Kaggle framework.

act(obs) is called with every new observation and returns the next (GameAction, data).
It closes the previous transition (shaped reward, done flag), lets the learner update, and
chooses the next action. Persistence: learner weights, replay/rollout and novelty counts persist
across levels and GAME_OVER resets within a game; new_game() resets them unless
cfg.reset_weights_per_game is False.
"""
from __future__ import annotations

import numpy as np
import torch
from arcengine import GameAction

from arcrl.agents.dqn import DQNLearner
from arcrl.agents.exploration import RewardShaper
from arcrl.agents.ppo import PPOLearner
from arcrl.agents.random_agent import RandomLearner
from arcrl.env.actions import ActionSpace
from arcrl.env.obs import FrameStack, Obs
from arcrl.utils import Config, resolve_device

LEARNERS = {"dqn": DQNLearner, "ppo": PPOLearner, "random": RandomLearner}


class OnlineController:
    def __init__(self, cfg: Config, seed: int = 0, device: str | None = None):
        self.cfg, self.seed = cfg, seed
        self.device = device or resolve_device(cfg.device)
        self.actions = ActionSpace(cfg.click_grid, cfg.object_clicks)
        self.init_state = None
        if cfg.init_checkpoint:
            self.init_state = torch.load(cfg.init_checkpoint, map_location=self.device)
        self.learner = None
        self.new_game(force=True)

    def new_game(self, force: bool = False) -> None:
        if force or self.cfg.reset_weights_per_game or self.learner is None:
            self.learner = LEARNERS[self.cfg.algo](self.cfg, self.actions.n, self.device, self.seed)
            if self.init_state is not None:
                self.learner.load_state_dict(self.init_state)
        self.shaper = RewardShaper(self.cfg)
        self.stack = FrameStack(self.cfg.frame_stack)
        self.prev: tuple | None = None
        self.last_stats: dict | None = None
        self.shaped_return = 0.0

    def act(self, obs: Obs):
        if obs.state in ("NOT_PLAYED", "GAME_OVER"):
            self._close(obs, done=True)
            self.stack.clear()
            self.prev = None
            return GameAction.RESET, None
        s = self.stack.push(obs.grid)
        mask = self.actions.mask(obs.available, obs.grid)
        self._close(obs, done=obs.state == "WIN", s2=s, m2=mask)
        if obs.state == "WIN":
            self.prev = None
            return None, None
        a = self.learner.select(s, mask)
        ga, data = self.actions.decode(a)
        self.prev = (s, mask, a, obs.grid, obs.levels)
        return ga, data

    def _close(self, obs: Obs, done: bool, s2=None, m2=None) -> None:
        if self.prev is None:
            return
        s, mask, a, pgrid, plev = self.prev
        if s2 is None:
            s2 = self.stack.push(obs.grid) if self.stack.buf else np.repeat(obs.grid[None], self.cfg.frame_stack, 0)
            m2 = np.ones(self.actions.n, bool)
        r, _ = self.shaper(pgrid, obs.grid, max(0, obs.levels - plev), obs.state == "GAME_OVER")
        self.shaped_return += r
        stats = self.learner.observe(s, mask, a, r, s2, m2, done)
        if stats:
            self.last_stats = stats

    def state_dict(self) -> dict:
        return self.learner.state_dict()
