#!/usr/bin/env python
"""Build report tables (markdown + LaTeX) and learning-curve figures from outputs/results."""
import argparse
from pathlib import Path

import _common  # noqa: F401

from arcrl.evaluation.aggregate import load_results, per_game_table, plot_curves, summary_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--budget", type=int, default=1000)
    args = ap.parse_args()
    df = load_results(args.results)
    if df.empty:
        raise SystemExit("no results found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s, g = summary_table(df), per_game_table(df)
    s.to_csv(out / "summary.csv", index=False)
    g.to_csv(out / "per_game.csv", index=False)
    (out / "summary.md").write_text(s.to_markdown(index=False))
    (out / "summary.tex").write_text(s.to_latex(index=False, float_format="%.2f"))
    print(s.to_string(index=False))
    for split in df.split.unique():
        for algo in ["dqn", "ppo"]:
            runs = sorted(r for r in df[df.split == split].run_name.unique() if r.startswith(algo) or r == "random")
            if runs:
                plot_curves(df, runs, split, args.budget, out / f"curve_{algo}_{split}.png")


if __name__ == "__main__":
    main()
