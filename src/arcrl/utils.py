"""Config loading, reproducibility, provenance, logging, JSONL IO."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    run_name: str = "run"
    algo: str = "dqn"  # dqn | ppo | random
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    output_dir: str = "outputs"
    splits_file: str = "configs/splits.yaml"
    split: str = "dev"  # dev | heldout | toy
    device: str = "auto"

    # Interaction budget (per game, per seed)
    max_actions_per_game: int = 1000

    # Action abstraction
    click_grid: int = 8
    object_clicks: bool = False

    # Observation / memory
    frame_stack: int = 1

    # Training-time reward shaping (never reported as the ARC score)
    level_reward: float = 1.0
    game_over_penalty: float = 0.0
    change_bonus: float = 0.0
    novelty_bonus: float = 0.0
    clock_mask: bool = False

    # Improvement 4: efficient action selection
    effect_model: bool = False
    effect_coef: float = 1.0
    effect_bias: float = 1.0
    prune_noeffect: bool = False

    # Shared learner hparams
    gamma: float = 0.95
    lr: float = 2.5e-4
    grad_clip: float = 10.0

    # DQN
    replay_capacity: int = 5000
    batch_size: int = 32
    warmup_steps: int = 64
    train_freq: int = 4
    target_sync: int = 250
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 1000
    double_dqn: bool = False

    # PPO
    rollout_len: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 32
    clip_eps: float = 0.2
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01

    # Persistence
    init_checkpoint: str | None = None  # may contain {seed}
    reset_weights_per_game: bool = True
    pretrain_split: str | None = None
    pretrain_passes: int = 2

    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _resolve(path: Path) -> dict[str, Any]:
        raw = yaml.safe_load(path.read_text()) or {}
        parent = raw.pop("inherit", None)
        if not parent:
            return raw
        merged = Config._resolve((path.parent / parent).resolve())
        merged.update(raw)
        return merged

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls.from_dict(cls._resolve(Path(path).resolve()))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        cfg = cls(**{k: v for k, v in raw.items() if k in known})
        cfg.extra = {k: v for k, v in raw.items() if k not in known}
        return cfg

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for p in ["arc-agi", "arcengine", "torch", "numpy", "gymnasium"]:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "missing"
    return out


def hardware() -> dict[str, str]:
    info = {"platform": platform.platform(), "cpu": platform.processor() or platform.machine()}
    try:
        import torch

        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        info["gpu"] = "none"
    return info


def write_run_meta(cfg: Config, out_dir: str | Path, **extra: Any) -> None:
    meta = {
        "run_name": cfg.run_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "python": sys.version,
        "packages": package_versions(),
        "hardware": hardware(),
        "config": asdict(cfg),
        **extra,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def get_logger(name: str = "arcrl") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(record: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


TOY_SPLITS = {"toy": ["toy0", "toy1"], "toy_a": ["toy0"], "toy_b": ["toy1"]}


def load_split(cfg: Config, name: str | None = None) -> list[str]:
    name = name or cfg.split
    if name in TOY_SPLITS:
        return TOY_SPLITS[name]
    main = Path(cfg.splits_file)
    for f in (main, main.with_name("pretrain_splits.yaml")):
        if f.exists():
            splits = yaml.safe_load(f.read_text()) or {}
            if name in splits:
                return list(splits[name])
    raise KeyError(f"split '{name}' not found in {main} or pretrain_splits.yaml")


def check_no_leakage(cfg: Config, eval_games: list[str]) -> None:
    if not cfg.pretrain_split:
        return
    overlap = set(load_split(cfg, cfg.pretrain_split)) & set(eval_games)
    if overlap:
        raise SystemExit(f"leakage: evaluation games {sorted(overlap)} were used for pretraining ({cfg.pretrain_split})")
