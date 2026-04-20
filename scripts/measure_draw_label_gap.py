"""
Quantify the gap between the trained label (P(equalize at any point))
and the true draw outcome (P(final score is level)) on the existing
backtest cache.

Mirrors the row-generation logic in src/Soccer/feature_builder.py so
the per-row counts match what the model actually trained on.
"""

import json
from pathlib import Path
from collections import defaultdict

MATCHES_FILE = Path("Data/soccer_backtest/matches.json")
ENTRY_MINUTES = [60, 65, 68, 70, 73, 75]


def score_at_minute(goals, minute):
    h, a = 0, 0
    for g in goals:
        if g["minute"] <= minute:
            h, a = g["home_after"], g["away_after"]
        else:
            break
    return h, a


def main():
    with open(MATCHES_FILE, "r", encoding="utf-8") as f:
        matches = json.load(f).get("matches", [])

    rows_total = 0
    eq_count = 0
    final_draw_count = 0

    # Conditional: of those that equalized, how many stayed level?
    stayed_level_given_eq = 0

    by_league = defaultdict(lambda: {"n": 0, "eq": 0, "fd": 0, "stay": 0})
    by_score_level = defaultdict(lambda: {"n": 0, "eq": 0, "fd": 0, "stay": 0})
    by_minute = defaultdict(lambda: {"n": 0, "eq": 0, "fd": 0, "stay": 0})

    for match in matches:
        goals = match.get("goals", []) or []
        league = match.get("league", "?")

        # Final score from last goal event (or 0-0 if no goals)
        if goals:
            final_h = goals[-1]["home_after"]
            final_a = goals[-1]["away_after"]
        else:
            final_h, final_a = 0, 0
        final_draw = (final_h == final_a)

        for entry_min in ENTRY_MINUTES:
            h, a = score_at_minute(goals, entry_min)
            if abs(h - a) != 1:
                continue
            losing_team = "home" if h < a else "away"

            equalized = False
            for g in goals:
                if g["minute"] > entry_min and g["team"] == losing_team:
                    if g["home_after"] == g["away_after"]:
                        equalized = True
                        break

            score_level = h + a

            rows_total += 1
            if equalized:
                eq_count += 1
                if final_draw:
                    stayed_level_given_eq += 1
            if final_draw:
                final_draw_count += 1

            for bucket in (by_league[league], by_score_level[score_level], by_minute[entry_min]):
                bucket["n"] += 1
                if equalized:
                    bucket["eq"] += 1
                if final_draw:
                    bucket["fd"] += 1
                if equalized and final_draw:
                    bucket["stay"] += 1

    def fmt(b):
        n = b["n"]
        if n == 0:
            return "n=0"
        eq = b["eq"] / n
        fd = b["fd"] / n
        stay = (b["stay"] / b["eq"]) if b["eq"] else 0.0
        return (f"n={n:5d}  P(eq)={eq:6.1%}  P(final_draw)={fd:6.1%}  "
                f"gap={eq-fd:+6.1%}  P(stay|eq)={stay:6.1%}")

    print("=" * 90)
    print("DRAW LABEL MISMATCH — full sample")
    print("=" * 90)
    print(fmt({"n": rows_total, "eq": eq_count, "fd": final_draw_count, "stay": stayed_level_given_eq}))
    print()

    print("By league:")
    for lg, b in sorted(by_league.items()):
        print(f"  {lg:20s} {fmt(b)}")
    print()

    print("By score level at entry (h+a):")
    for sl, b in sorted(by_score_level.items()):
        print(f"  level={sl}  {fmt(b)}")
    print()

    print("By entry minute:")
    for m, b in sorted(by_minute.items()):
        print(f"  min {m}  {fmt(b)}")


if __name__ == "__main__":
    main()
