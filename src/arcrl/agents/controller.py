"""Online controller: one object drives a learner through a game, locally or inside the Kaggle framework.

act(obs) is called with every new observation and returns the next (GameAction, data).
It closes the previous transition (shaped reward, done flag), lets the learner update, and
chooses the next action. Persistence: learner weights, replay/rollout and novelty counts persist
across levels and GAME_OVER resets within a game; new_game() resets them unless
cfg.reset_weights_per_game is False
(pretraining), in which case only the exploration schedule restarts and any partial PPO rollout is dropped.

Improvement 4 (cfg.prune_noeffect): per game, a table maps each clock-masked state to the actions already
seen to leave it unchanged; those actions are removed from the choice when that state recurs (including after
RESET), as long as one legal action remains (cf. Blind Squirrel's state-graph pruning). Per-step effect labels
(did the clock-masked frame change?) are passed to the learner for its effect head.

Improvement 5 (cfg.graph_explore): return-then-explore over a per-game StateGraph (Go-Explore; Ecoffet et al.,
2021). In a state with untried legal actions, the learner chooses among the untried ones only. In a state whose
actions are all tried, the controller follows the shortest known path to the nearest state with untried actions
(also after RESET, which keeps progress across lives). The learner sees this as a semi-MDP: each of its
decisions is one transition whose reward is the undiscounted sum of shaped rewards until its next decision,
so PPO stays on-policy and DQN learns only from its own choices.
"""
from __future__ import annotations

import numpy as np
import torch
from arcengine import GameAction

from arcrl.agents.dqn import DQNLearner
from arcrl.agents.exploration import RewardShaper
from arcrl.agents.graph import StateGraph
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
            path = cfg.init_checkpoint.format(seed=seed)
            self.init_state = torch.load(path, map_location=self.device)
        self.learner = None
        self.new_game(force=True)

    def new_game(self, force: bool = False) -> None:
        if force or self.cfg.reset_weights_per_game or self.learner is None:
            self.learner = LEARNERS[self.cfg.algo](self.cfg, self.actions.n, self.device, self.seed)
            if self.init_state is not None:
                self.learner.load_state_dict(self.init_state)
        else:
            self.learner.t = 0
            if hasattr(self.learner, "_clear"):
                self.learner._clear()
        self.shaper = RewardShaper(self.cfg)
        self.stack = FrameStack(self.cfg.frame_stack)
        self.prev: tuple | None = None
        self.last_stats: dict | None = None
        self.shaped_return = 0.0
        self.noeffect: dict[bytes, set[int]] = {}
        self.steps = 0
        self.effective_steps = 0
        self.pruned_choices = 0
        self.graph = StateGraph(np.random.default_rng(self.seed)) if self.cfg.graph_explore else None
        self.last: tuple | None = None
        self.acc_r = 0.0
        self.first_changed: bool | None = None
        self.planned_steps = 0
        self.fallback_decisions = 0
        self.life_t = 0

    def act(self, obs: Obs):
        if self.graph is not None:
            return self._act_graph(obs)
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
        key = self.shaper.key(obs.grid) if self.cfg.prune_noeffect else None
        act_mask = self._prune(mask, key)
        a = self.learner.select(s, act_mask)
        ga, data = self.actions.decode(a)
        self.prev = (s, act_mask, a, obs.grid, obs.levels, key)
        return ga, data

    def _prune(self, mask: np.ndarray, key: bytes | None) -> np.ndarray:
        tried = self.noeffect.get(key) if key is not None else None
        if not tried:
            return mask
        pruned = mask.copy()
        pruned[list(tried)] = False
        if not pruned.any():
            return mask
        self.pruned_choices += 1
        return pruned

    def _close(self, obs: Obs, done: bool, s2=None, m2=None) -> None:
        if self.prev is None:
            return
        s, mask, a, pgrid, plev, key = self.prev
        if s2 is None:
            s2 = self.stack.push(obs.grid) if self.stack.buf else np.repeat(obs.grid[None], self.cfg.frame_stack, 0)
            m2 = np.ones(self.actions.n, bool)
        r, _ = self.shaper(pgrid, obs.grid, max(0, obs.levels - plev), obs.state == "GAME_OVER")
        changed = self.shaper.last_changed
        self.shaped_return += r
        self.steps += 1
        self.effective_steps += changed
        if key is not None and not changed and obs.state != "GAME_OVER":
            self.noeffect.setdefault(key, set()).add(a)
        stats = self.learner.observe(s, mask, a, r, s2, m2, done, changed=changed)
        if stats:
            self.last_stats = stats

    def _key(self, obs: Obs) -> tuple:
        return obs.levels, self.shaper.key(obs.grid)

    def _act_graph(self, obs: Obs):
        game_over = obs.state == "GAME_OVER"
        if self.last is not None:
            lgrid, llev, lkey, la = self.last
            self.life_t += 1
            r, _ = self.shaper(lgrid, obs.grid, max(0, obs.levels - llev), game_over, t=self.life_t)
            changed = self.shaper.last_changed
            self.shaped_return += r
            self.steps += 1
            self.effective_steps += changed
            if self.prev is not None:
                self.acc_r += r
                if self.first_changed is None:
                    self.first_changed = changed
            self.graph.record(lkey, la, None if game_over else self._key(obs))
            self.last = None
        if obs.state in ("NOT_PLAYED", "GAME_OVER"):
            self._close_decision(True, np.repeat(obs.grid[None], self.cfg.frame_stack, 0) if obs.grid is not None
                                 else None, np.ones(self.actions.n, bool))
            self.stack.clear()
            self.life_t = 0
            return GameAction.RESET, None
        s = self.stack.push(obs.grid)
        legal = self.actions.mask(obs.available, obs.grid)
        if obs.state == "WIN":
            self._close_decision(True, s, legal)
            return None, None
        key = self._key(obs)
        self.graph.visit(key, legal)
        use = self.graph.untried(key)
        if not use.any():
            path = self.graph.frontier_path(key)
            if path:
                self.planned_steps += 1
                self.last = (obs.grid, obs.levels, key, path[0])
                return self.actions.decode(path[0])
            self.fallback_decisions += 1
            use = legal
        self._close_decision(False, s, use)
        a = self.learner.select(s, use)
        self.prev = (s, use, a)
        self.acc_r, self.first_changed = 0.0, None
        self.last = (obs.grid, obs.levels, key, a)
        return self.actions.decode(a)

    def _close_decision(self, done: bool, s2, m2) -> None:
        if self.prev is None:
            return
        s, mask, a = self.prev
        if s2 is None:
            s2 = s
        stats = self.learner.observe(s, mask, a, self.acc_r, s2, m2, done, changed=bool(self.first_changed))
        self.prev = None
        if stats:
            self.last_stats = stats

    def graph_stats(self) -> dict:
        if self.graph is None:
            return {}
        return {**self.graph.stats(), "planned_steps": self.planned_steps,
                "fallback_decisions": self.fallback_decisions}

    def state_dict(self) -> dict:
        return self.learner.state_dict()
