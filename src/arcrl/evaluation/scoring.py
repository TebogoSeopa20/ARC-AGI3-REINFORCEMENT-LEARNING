"""Official ARC-AGI-3 score extraction, plus a reference re-implementation for tests.

Reported numbers always come from `official_game_scores` (arc_agi's own scorecard,
arc_agi.scorecard.EnvironmentScorecard). `reference_game_score` mirrors
EnvironmentScoreCalculator in arc-agi 0.9.9 and is only used to sanity-check toy runs.
"""
from __future__ import annotations


def official_game_scores(arc, card_id: str) -> dict[str, dict]:
    sc = arc.get_scorecard(card_id)
    out: dict[str, dict] = {"__total__": {"score": float(sc.score) if sc else 0.0}}
    if sc is None:
        return out
    for env in sc.environments:
        best = max(env.runs, key=lambda r: r.score)
        out[env.id.split("-")[0]] = {
            "score": float(env.score),
            "levels_completed": int(env.levels_completed),
            "actions": int(env.actions),
            "resets": int(env.resets),
            "level_scores": best.level_scores,
            "level_actions": best.level_actions,
            "level_baseline_actions": best.level_baseline_actions,
            "number_of_levels": best.number_of_levels,
            "message": best.message,
        }
    return out


def reference_level_score(baseline_actions: int, actions_taken: int, completed: bool) -> float:
    if not completed or actions_taken <= 0:
        return 0.0
    return min((baseline_actions / actions_taken) ** 2 * 100.0, 115.0)


def reference_game_score(levels: list[tuple[bool, int, int]]) -> float:
    """levels: [(completed, actions_taken, baseline_actions)] in level order (1-indexed weights)."""
    if not levels:
        return 0.0
    tot = wsum = wmax = 0.0
    for i, (done, taken, base) in enumerate(levels, start=1):
        s = reference_level_score(base, taken, done)
        tot += s * i
        wsum += i
        wmax += i if s > 0 else 0
    return min(tot / wsum, wmax / wsum * 100.0)
