"""Build notebooks/submission.ipynb from agent/my_agent.py + src/arcrl + configs/submission.yaml.

Adapted from arcprize/ARC-AGI-3-Kaggle-Starter (scripts/build_notebook.py). Changes: ships the
arcrl package and the resolved submission config to /tmp/arcrl_pkg so MyAgent can import them.

Cells:
  1. install arc-agi from the offline competition wheels
  2. write arcrl package + submission.yaml to /tmp/arcrl_pkg, my_agent.py to /tmp
  3. competition rerun: copy framework, register MyAgent, run main.py against the gateway
  4. commit run: write a dummy submission.parquet
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from textwrap import dedent

import yaml

ACCELERATOR = "t4"  # cpu | t4 | p100 | rtx6000

_ACCELERATORS = {
    "cpu": {"name": "none", "gpu": False},
    "t4": {"name": "nvidiaTeslaT4", "gpu": True},
    "p100": {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000", "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
AGENT_SRC = ROOT / "agent" / "my_agent.py"
PKG = ROOT / "src" / "arcrl"
CFG = ROOT / "configs" / "submission.yaml"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [],
            "execution_count": None, "source": source}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def package_files() -> dict[str, str]:
    from arcrl.utils import Config

    files = {str(p.relative_to(PKG.parent)): p.read_text() for p in sorted(PKG.rglob("*.py"))}
    cfg = asdict(Config.from_yaml(CFG))
    cfg.pop("extra", None)
    ckpt = cfg.get("init_checkpoint")
    if ckpt and not str(ckpt).startswith("/kaggle/input/"):
        raise SystemExit("init_checkpoint must point to an attached Kaggle dataset under /kaggle/input/")
    files["submission.yaml"] = yaml.safe_dump(cfg, sort_keys=False)
    return files


def build() -> dict:
    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )
    write_cell = code_cell(
        "import pathlib\n"
        f"FILES = {package_files()!r}\n"
        "root = pathlib.Path('/tmp/arcrl_pkg')\n"
        "for rel, src in FILES.items():\n"
        "    p = root / rel\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    p.write_text(src)\n"
        f"pathlib.Path('/tmp/my_agent.py').write_text({AGENT_SRC.read_text()!r})\n"
        "print('wrote', len(FILES), 'files')\n"
    )
    run_cell = code_cell(dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents
            !cp /tmp/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")
            !cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent myagent
        """
    ))
    dummy_cell = code_cell(dedent(
        """\
        import os
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            import pandas as pd
            submission = pd.DataFrame(data=[['1_0', '1', True, 1]],
                                      columns=['row_id', 'game_id', 'end_of_game', 'score'])
            submission.to_parquet('/kaggle/working/submission.parquet', index=False)
            submission.head()
        """
    ))
    accel = _ACCELERATORS[ACCELERATOR]
    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "mimetype": "text/x-python",
                              "file_extension": ".py", "pygments_lexer": "ipython3"},
            "kaggle": {"accelerator": accel["name"], "isInternetEnabled": False,
                       "isGpuEnabled": accel["gpu"], "language": "python", "sourceType": "notebook"},
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell("# ARC Prize 2026 — ARC-AGI-3 · Team General\n\n"
                          "DQN/PPO online-learning agent (COMS4061A/COMS7071A, Wits). "
                          "Generated by `scripts/build_notebook.py`; edit the repo, not this notebook."),
            install_cell, write_cell, run_cell, dummy_cell,
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] wrote {NOTEBOOK_PATH.relative_to(ROOT)} (accelerator: {ACCELERATOR})")
    meta = json.loads(METADATA_PATH.read_text())
    if meta.get("enable_gpu") != _ACCELERATORS[ACCELERATOR]["gpu"]:
        meta["enable_gpu"] = _ACCELERATORS[ACCELERATOR]["gpu"]
        METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
