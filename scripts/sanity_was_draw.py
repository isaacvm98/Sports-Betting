"""
Reconcile two ways of computing 'final score is a draw':
  A) match['was_draw'] field as stored in matches.json
  B) Last entry of match['goals'] array compared (home_after vs away_after)

If they disagree, one source is wrong and we need to know which.
"""

import json
from pathlib import Path
from collections import Counter

MATCHES_PATH = Path("Data/soccer_backtest/matches.json")


def main():
    with open(MATCHES_PATH, "r", encoding="utf-8") as f:
        matches = json.load(f).get("matches", [])

    n = len(matches)
    n_with_was = 0
    n_with_goals = 0
    n_was_true = 0
    n_goals_draw = 0
    agreement = Counter()

    sample_disagreements = []

    for m in matches:
        was = m.get("was_draw")
        goals = m.get("goals", []) or []
        if was is not None:
            n_with_was += 1
            if was:
                n_was_true += 1
        if goals:
            n_with_goals += 1
            final_h = goals[-1]["home_after"]
            final_a = goals[-1]["away_after"]
            from_goals = (final_h == final_a)
        else:
            # No goals = 0-0 draw
            from_goals = True

        if from_goals:
            n_goals_draw += 1

        if was is not None:
            agreement[(bool(was), from_goals)] += 1
            if bool(was) != from_goals and len(sample_disagreements) < 10:
                fs = m.get("final_score") or m.get("ft_score")
                sample_disagreements.append({
                    "match_id": m.get("match_id"),
                    "home": m.get("home_team"),
                    "away": m.get("away_team"),
                    "date": m.get("date"),
                    "was_draw_field": was,
                    "from_goals_draw": from_goals,
                    "n_goals": len(goals),
                    "last_goal": goals[-1] if goals else None,
                    "final_score_field": fs,
                })

    print(f"Total matches: {n}")
    print(f"With was_draw field:  {n_with_was}  (true: {n_was_true} = "
          f"{n_was_true/n_with_was*100:.1f}%)")
    print(f"With goals array:     {n_with_goals}")
    print(f"Goals -> draw:        {n_goals_draw}  ({n_goals_draw/n*100:.1f}%)")
    print()
    print("Agreement matrix (was_draw_field, from_goals_draw):")
    for k, v in sorted(agreement.items()):
        print(f"  {k}: {v}")
    print()

    if sample_disagreements:
        print("Sample disagreements:")
        for d in sample_disagreements:
            print(f"  {d}")

    # Inspect a couple of matches' raw goal arrays
    print()
    print("First 3 matches' goals arrays (sanity):")
    for m in matches[:3]:
        print(f"  {m.get('home_team')} vs {m.get('away_team')} "
              f"({m.get('date')})  was_draw={m.get('was_draw')}  "
              f"final_score={m.get('final_score') or m.get('ft_score')}")
        for g in (m.get("goals") or [])[:6]:
            print(f"    min {g.get('minute')}: {g.get('team')} -> "
                  f"{g.get('home_after')}-{g.get('away_after')}")
        if len(m.get("goals") or []) > 6:
            print(f"    ... ({len(m['goals'])} goals total, last="
                  f"{m['goals'][-1]})")


if __name__ == "__main__":
    main()
