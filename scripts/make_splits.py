#!/usr/bin/env python
"""Freeze a dev / held-out split of the public games. Run once, commit configs/splits.yaml, never change it."""
import argparse
import random
from pathlib import Path

import yaml

import _common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(_common.ROOT / "configs" / "splits.yaml"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; the split is frozen. Use --force only before any experiment.")
    ids = sorted({e.game_id.split("-")[0] for e in _common.make_arcade().get_environments()})
    random.Random(args.seed).shuffle(ids)
    k = max(1, round(len(ids) * args.heldout_frac))
    split = {"seed": args.seed, "dev": sorted(ids[k:]), "heldout": sorted(ids[:k])}
    out.write_text(yaml.safe_dump(split, sort_keys=False))
    print(f"dev={len(split['dev'])} heldout={len(split['heldout'])} -> {out}")


if __name__ == "__main__":
    main()
