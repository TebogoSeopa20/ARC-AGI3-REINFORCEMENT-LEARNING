#!/usr/bin/env python
"""List locally available ARC-AGI-3 games (downloads metadata on first run)."""
import _common  # noqa: F401


def main():
    arc = _common.make_arcade()
    for e in sorted(arc.get_environments(), key=lambda e: e.game_id):
        n = len(e.baseline_actions) if e.baseline_actions else "?"
        print(f"{e.game_id:24} levels={n:<3} title={e.title} tags={e.tags}")


if __name__ == "__main__":
    main()
