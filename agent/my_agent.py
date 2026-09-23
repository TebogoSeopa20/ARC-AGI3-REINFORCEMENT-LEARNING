"""Kaggle submission agent: wraps arcrl.OnlineController in the ARC-AGI-3-Agents `Agent` contract.

The class must be named MyAgent. scripts/build_notebook.py ships the arcrl package and the resolved
configs/submission.yaml into /tmp/arcrl_pkg on Kaggle; locally the repo's src/ is used.
No game-specific logic: every game gets the same learner, initialised identically.
"""
from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[1] / "src"
_PKG = str(_SRC) if (_SRC / "arcrl").exists() else "/tmp/arcrl_pkg"
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from agents.agent import Agent  # noqa: E402
from arcengine import FrameData, GameAction, GameState  # noqa: E402

from arcrl.agents.controller import OnlineController  # noqa: E402
from arcrl.env.obs import Obs  # noqa: E402
from arcrl.utils import Config  # noqa: E402


def _load_cfg() -> Config:
    for p in (os.getenv("ARCRL_CONFIG", ""), "/tmp/arcrl_pkg/submission.yaml",
              str(_HERE.parents[1] / "configs" / "submission.yaml")):
        if p and Path(p).exists():
            return Config.from_yaml(p)
    return Config()


CFG = _load_cfg()


class MyAgent(Agent):
    MAX_ACTIONS = CFG.max_actions_per_game

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ctrl = OnlineController(CFG, seed=zlib.crc32(self.game_id.split("-")[0].encode()) % 10_000)

    @property
    def name(self) -> str:
        return f"{super().name}.{CFG.algo}.{CFG.run_name}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        ga, data = self.ctrl.act(Obs.from_frame_data(latest_frame))
        if ga is None:
            return GameAction.RESET
        if data:
            ga.set_data(data)
        ga.reasoning = {"algo": CFG.algo}
        return ga
