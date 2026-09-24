#!/usr/bin/env python
"""Compare every run on one game subset (default dev_b), pooling rows from any split that contains those games.

Used for improvement 3: dev runs of random/baseline/imp1/imp2 already include the dev_b games at the same seeds.
Writes outputs/figures/compare_<split>.md.
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
import pandas as pd

from arcrl.evaluation.aggregate import ci95, load_results
from arcrl.utils import Config, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev_b")
    ap.add_argument("--results", default="outputs/results")
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()
    games = load_split(Config(), args.split)
    df = load_results(args.results)
    df = df[df.game_id.isin(games) & df.split.isin(["dev", args.split])]
    per_seed = df.groupby(["run_name", "seed"]).agg(
        score=("official_score", "mean"), levels=("levels_completed", "sum"),
        unique_states=("unique_states", "sum"), game_overs=("game_overs", "sum")).reset_index()
    table = per_seed.groupby("run_name").agg(
        seeds=("seed", "nunique"), score_mean=("score", "mean"), score_ci95=("score", ci95),
        levels_mean=("levels", "mean"), levels_std=("levels", "std"),
        unique_states_mean=("unique_states", "mean"), game_overs_mean=("game_overs", "mean")).round(3).reset_index()
    done = df[df.levels_completed > 0].sort_values(["game_id", "run_name", "seed"])
    events = [f"- {r.game_id} {r.run_name} seed {r.seed}: levels at actions {r.level_completed_at}"
              for r in done.itertuples()]
    text = (f"# Runs on {args.split}: {', '.join(games)}\n\n" + table.to_markdown(index=False)
            + "\n\n## Level completions\n" + ("\n".join(events) or "none") + "\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"compare_{args.split}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
