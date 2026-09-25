import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from arcrl.agents.controller import OnlineController
from arcrl.agents.graph import DEAD, StateGraph
from arcrl.evaluation.runner import play_game
from arcrl.utils import Config


class LockEnv:
    """Deterministic combination lock: `length` correct presses in a row; a life lasts `life` actions."""

    def __init__(self, length=7, life=12, seed=12345):
        self.code = np.random.default_rng(seed).integers(1, 5, size=length)
        self.length, self.life = length, life
        self.pos, self.used, self.state, self.levels = 0, 0, GameState.NOT_FINISHED, 0
        self._last = self._frame()

    @property
    def observation_space(self):
        return self._last

    def _frame(self):
        g = np.zeros((64, 64), np.int64)
        g[10, : self.pos * 4] = 3
        fd = FrameDataRaw(game_id="lock", state=self.state, levels_completed=self.levels, win_levels=1,
                          available_actions=[1, 2, 3, 4])
        fd.frame = [g]
        self._last = fd
        return fd

    def reset(self):
        self.pos, self.used, self.state = 0, 0, GameState.NOT_FINISHED
        return self._frame()

    def step(self, action, data=None, reasoning=None):
        if action is GameAction.RESET:
            return self.reset()
        self.used += 1
        if action.value == self.code[self.pos]:
            self.pos += 1
            if self.pos == self.length:
                self.levels, self.state = 1, GameState.WIN
        if self.state is not GameState.WIN and self.used >= self.life:
            self.state = GameState.GAME_OVER
        return self._frame()


def test_frontier_path_skips_dead_and_finds_nearest_untried():
    g = StateGraph(np.random.default_rng(0))
    full = np.ones(3, bool)
    for k in "abcd":
        g.visit(k, full)
    for a in range(3):
        g.record("a", a, "b" if a == 0 else ("c" if a == 1 else None))
    for a in range(3):
        g.record("b", a, "d" if a == 2 else "b")
    g.record("c", 0, DEAD and None)
    assert g.frontier_path("a") == [1]
    g.record("c", 1, "a")
    g.record("c", 2, "a")
    assert g.frontier_path("a") == [0, 2]


def test_graph_explore_solves_locks_that_random_cannot():
    wins = {False: 0, True: 0}
    for seed in range(6):
        for graph in (False, True):
            cfg = Config(algo="random", graph_explore=graph, clock_mask=True, clock_mask_timed=graph)
            r = play_game(LockEnv(seed=100 + seed), OnlineController(cfg, seed=seed), "lock", budget=400)
            wins[graph] += r["win"]
            if graph:
                assert r["planned_steps"] > 0
    assert wins[True] == 6 and wins[False] <= 2


def test_timed_clock_mask_keeps_agent_progress_pixels():
    from arcrl.agents.exploration import ClockMask

    cm = ClockMask(timed=True)
    for life, when in enumerate([3, 5, 2]):
        g = np.zeros((64, 64), np.uint8)
        for t in range(1, 8):
            nxt = g.copy()
            nxt[0, t] = 9
            if t == when:
                nxt[10, 0:4] = 3
            cm.update(g, nxt, t)
            g = nxt
    assert cm.mask[0, 1:8].all() and not cm.mask[10, 0:4].any()


def test_graph_mode_learners_run_and_update():
    """Decision accounting only. Whether a learner solves the lock depends on torch numerics, which differ
    across torch versions and platforms, so solving is tested with the deterministic random learner above."""
    from arcrl.utils import set_seed

    for algo in ("dqn", "ppo"):
        set_seed(0)
        cfg = Config(algo=algo, graph_explore=True, clock_mask=True, clock_mask_timed=True, effect_model=True,
                     warmup_steps=8, rollout_len=16, minibatch_size=8, batch_size=8, train_freq=1)
        ctrl = OnlineController(cfg, seed=1)
        r = play_game(LockEnv(seed=101), ctrl, "lock", budget=400)
        decisions = ctrl.learner.t
        assert decisions > 0 and decisions + ctrl.planned_steps + r["resets"] == r["actions"]
        if algo == "dqn":
            assert len(ctrl.learner.buf) in (decisions - 1, decisions)
        else:
            assert len(ctrl.learner.S) in ((decisions - 1) % 16, decisions % 16)


def test_timed_clock_mask_is_active_without_graph():
    cfg = Config(algo="random", clock_mask=True, clock_mask_timed=True, prune_noeffect=True)
    ctrl = OnlineController(cfg, seed=0)
    play_game(LockEnv(seed=7), ctrl, "lock", budget=60)
    when = ctrl.shaper.clock.when
    assert (when >= 1).any() and when.max() <= 12
