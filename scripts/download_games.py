#!/usr/bin/env python
"""Cache every public game into environment_files/ so later runs (and Kaggle) work offline."""
import _common


def main():
    arc = _common.make_arcade("normal")
    ids = sorted({e.game_id.split("-")[0] for e in arc.get_environments()})
    ok = [g for g in ids if arc.make(g) is not None]
    print(f"cached {len(ok)}/{len(ids)} games into environment_files/: {ok}")
    missing = sorted(set(ids) - set(ok))
    if missing:
        print("FAILED:", missing)


if __name__ == "__main__":
    main()
