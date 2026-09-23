"""Aggregate per-(run, seed, game) results into tables and learning curves."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from arcrl.utils import read_jsonl


def load_results(results_dir: str | Path) -> pd.DataFrame:
    rows = []
    for f in Path(results_dir).glob("*/*/games.jsonl"):
        rows.extend(read_jsonl(f))
    return pd.DataFrame(rows)


def ci95(x) -> float:
    x = np.asarray(x, float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    per_seed = df.groupby(["run_name", "split", "seed"]).agg(
        official_score=("official_score", "mean"),
        levels=("levels_completed", "sum"),
        wins=("win", "sum"),
        actions=("actions", "sum"),
        game_overs=("game_overs", "sum"),
        wall_time_s=("wall_time_s", "sum"),
    ).reset_index()
    g = per_seed.groupby(["run_name", "split"])
    out = g.agg(
        seeds=("seed", "nunique"),
        score_mean=("official_score", "mean"),
        score_std=("official_score", "std"),
        score_ci95=("official_score", ci95),
        levels_mean=("levels", "mean"),
        levels_std=("levels", "std"),
        wins_mean=("wins", "mean"),
        actions_mean=("actions", "mean"),
        wall_time_s_mean=("wall_time_s", "mean"),
    ).reset_index()
    return out.round(3)


def per_game_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["run_name", "split", "game_id"]).agg(
        score_mean=("official_score", "mean"),
        levels_mean=("levels_completed", "mean"),
        levels_min=("levels_completed", "min"),
        levels_max=("levels_completed", "max"),
        actions_mean=("actions", "mean"),
    ).reset_index().round(3)


def curve_matrix(df: pd.DataFrame, run: str, split: str, budget: int, step: int = 25):
    """Total levels completed (summed over games) vs actions-per-game, one row per seed."""
    xs = np.arange(0, budget + 1, step)
    rows = []
    for _, sdf in df[(df.run_name == run) & (df.split == split)].groupby("seed"):
        tot = np.zeros_like(xs, dtype=float)
        for at in sdf.level_completed_at:
            at = np.asarray(at)
            tot += np.array([(at <= x).sum() for x in xs])
        rows.append(tot)
    return xs, np.array(rows)


def plot_curves(df: pd.DataFrame, runs: list[str], split: str, budget: int, out: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for run in runs:
        xs, m = curve_matrix(df, run, split, budget)
        if not len(m):
            continue
        line, = ax.plot(xs, m.mean(0), label=run)
        for r in m:
            ax.plot(xs, r, color=line.get_color(), alpha=0.2, lw=0.8)
    ax.set_xlabel("actions per game")
    ax.set_ylabel(f"levels completed ({split}, summed)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
