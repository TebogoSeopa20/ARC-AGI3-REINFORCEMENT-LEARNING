"""Shared script helpers: sys.path setup and Arcade construction."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_arcade(mode: str | None = None):
    import arc_agi
    from arc_agi import OperationMode

    mode = mode or os.getenv("OPERATION_MODE", "normal")
    return arc_agi.Arcade(
        operation_mode=OperationMode(mode),
        environments_dir=os.getenv("ENVIRONMENTS_DIR", str(ROOT / "environment_files")),
    )
