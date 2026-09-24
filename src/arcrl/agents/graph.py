"""Per-game state graph for improvement 5 (return-then-explore, after Go-Explore; Ecoffet et al., 2021).

Nodes are (levels completed, clock-masked frame hash). For each node it stores the legal abstracted actions,
the actions already tried, and the observed outcome of each tried action (next node, or DEAD for GAME_OVER).
It assumes transitions are deterministic and measures that assumption: a (node, action) seen twice either
leads to the same node (consistent) or not (inconsistent; the latest outcome is kept). One exception: a GAME_OVER
never overwrites a known live outcome, because with the move counter masked out of the key, running out of moves
looks like the action itself was fatal.
"""
from __future__ import annotations

from collections import deque

import numpy as np

DEAD = b"__dead__"


class StateGraph:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.legal: dict[tuple, np.ndarray] = {}
        self.edges: dict[tuple, dict[int, tuple | bytes]] = {}
        self.consistent = 0
        self.inconsistent = 0

    def visit(self, key: tuple, legal: np.ndarray) -> None:
        if key not in self.legal:
            self.legal[key] = legal.copy()
            self.edges[key] = {}

    def record(self, key: tuple, action: int, nxt: tuple | None) -> None:
        out = DEAD if nxt is None else nxt
        known = self.edges.setdefault(key, {})
        if action in known:
            if known[action] == out:
                self.consistent += 1
                return
            self.inconsistent += 1
            if out == DEAD:
                return
        known[action] = out

    def untried(self, key: tuple) -> np.ndarray:
        m = self.legal[key].copy()
        tried = list(self.edges.get(key, {}))
        if tried:
            m[tried] = False
        return m

    def frontier_path(self, start: tuple) -> list[int] | None:
        """Shortest known action sequence from start to a node with untried actions (random tie-break)."""
        parent: dict[tuple, tuple[tuple, int] | None] = {start: None}
        q = deque([start])
        while q:
            layer = list(q)
            q.clear()
            hits = [k for k in layer if k != start and self.untried(k).any()]
            if hits:
                node = hits[int(self.rng.integers(len(hits)))]
                path = []
                while parent[node] is not None:
                    node, a = parent[node]
                    path.append(a)
                return path[::-1]
            for k in layer:
                acts = list(self.edges.get(k, {}).items())
                self.rng.shuffle(acts)
                for a, nxt in acts:
                    if nxt == DEAD or nxt == k or nxt in parent or nxt not in self.legal:
                        continue
                    parent[nxt] = (k, a)
                    q.append(nxt)
        return None

    def stats(self) -> dict:
        return {
            "graph_nodes": len(self.legal),
            "graph_edges": sum(len(e) for e in self.edges.values()),
            "transitions_consistent": self.consistent,
            "transitions_inconsistent": self.inconsistent,
        }
