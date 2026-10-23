#!/usr/bin/env python
"""Figures for the report, from outputs/results. Writes report/figures/{levels_by_version,curves}.pdf."""
import argparse
from pathlib import Path

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arcrl.evaluation.aggregate import load_results

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b6a65"
BLUE_RAMP = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
VERSIONS = [("baseline", "Base"), ("imp1_explore", "I1"), ("imp2_clockmask", "I2"),
            ("imp3_pretrain_final", "I3"), ("imp4_efficient", "I4"), ("imp5_graph", "I5")]
CONTROLS = [("random", "random", ":"), ("random_imp4_actions", "random, I4 actions", "--"),
            ("random_imp5_graph", "random, I5 graph", "-.")]

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2a27", "xtick.color": "#5b5a55",
                     "ytick.color": "#5b5a55", "legend.frameon": False, "pdf.fonttype": 42})


def per_seed_levels(df, run, split):
    d = df[(df.run_name == run) & (df.split == split)]
    return d.groupby("seed").levels_completed.sum().to_numpy() if len(d) else np.array([])


def levels_by_version(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3), sharey=False)
    for ax, split, title in zip(axes, ["dev", "heldout"], ["Development games (15)", "Held-out games (10)"]):
        x = np.arange(len(VERSIONS))
        for algo, color, off, marker in [("dqn", BLUE, -0.12, "o"), ("ppo", ORANGE, 0.12, "s")]:
            means = []
            for i, (suffix, _) in enumerate(VERSIONS):
                v = per_seed_levels(df, f"{algo}_{suffix}", split)
                if not len(v):
                    means.append(np.nan)
                    continue
                means.append(v.mean())
                ax.scatter(np.full(len(v), i + off), v, s=9, color=color, alpha=0.35, linewidths=0, zorder=2)
            ax.plot(x + off, means, color=color, lw=1.5, marker=marker, ms=4.5, zorder=3,
                    label=algo.upper(), markeredgecolor="white", markeredgewidth=0.6)
        for run, label, ls in CONTROLS:
            v = per_seed_levels(df, run, split)
            if len(v):
                ax.axhline(v.mean(), color=GREY, lw=0.9, ls=ls, zorder=1, label=label)
        ax.set_xticks(x, [lbl for _, lbl in VERSIONS])
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        if split == "dev":
            ax.text(3, 3.6, "not run\non dev", color=GREY, fontsize=6.5, ha="center", va="center")
    axes[0].set_ylabel("Levels completed (sum over games)")
    axes[0].legend(loc="upper left", fontsize=6.5, handlelength=2.2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def curve(df, run, splits, budget=1000, step=20):
    xs = np.arange(0, budget + 1, step)
    d = df[(df.run_name == run) & df.split.isin(splits)]
    rows = []
    for _, sd in d.groupby("seed"):
        tot = np.zeros(len(xs))
        for at in sd.level_completed_at:
            at = np.asarray(at)
            tot += np.array([(at <= x).sum() for x in xs])
        rows.append(tot)
    return xs, np.array(rows)


def curves(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2), sharey=True)
    for ax, algo in zip(axes, ["dqn", "ppo"]):
        for (suffix, label), color in zip([("baseline", "Base"), ("imp2_clockmask", "I2"),
                                           ("imp4_efficient", "I4"), ("imp5_graph", "I5")], BLUE_RAMP):
            xs, m = curve(df, f"{algo}_{suffix}", ["dev", "heldout"])
            if not len(m):
                continue
            for r in m:
                ax.plot(xs, r, color=color, lw=0.6, alpha=0.3)
            ax.plot(xs, m.mean(0), color=color, lw=2, label=label)
        xs, m = curve(df, "random_imp4_actions", ["dev", "heldout"])
        ax.plot(xs, m.mean(0), color=GREY, lw=1.2, ls="--", label="random, I4 actions")
        ax.set_title(f"{algo.upper()} (25 public games, 3 seeds)", fontsize=8.5, loc="left")
        ax.set_xlabel("Actions per game")
        ax.grid(color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Levels completed")
    axes[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    df = load_results(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    levels_by_version(df, out / "levels_by_version.pdf")
    curves(df, out / "curves.pdf")
    print("wrote", sorted(p.name for p in out.glob("*.pdf")))


if __name__ == "__main__":
    main()
