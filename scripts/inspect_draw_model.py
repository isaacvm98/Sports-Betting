"""
Inspect Data/soccer_models/draw_xgb.pkl to figure out what target it
was trained on.

Heuristic: a model trained on `equalized` will predict a mean ~12% on
the historical sample. A model trained on `was_draw` will predict ~4-5%.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from src.Soccer.feature_builder import (
    build_features,
    get_feature_columns,
    _get_season,
)

MODEL_PATH = Path("Data/soccer_models/draw_xgb.pkl")
MATCHES_PATH = Path("Data/soccer_backtest/matches.json")


def main():
    with open(MODEL_PATH, "rb") as f:
        state = pickle.load(f)

    print("=" * 70)
    print(f"Loaded {MODEL_PATH}")
    print("=" * 70)
    print(f"State keys: {list(state.keys())}")
    for k, v in state.items():
        if k == "calibrated":
            print(f"  calibrated: {type(v).__name__}")
        elif k == "feature_cols":
            print(f"  feature_cols ({len(v)}): {v}")
        else:
            print(f"  {k}: {v!r}")

    cal = state["calibrated"]
    feature_cols = state.get("feature_cols")

    with open(MATCHES_PATH, "r", encoding="utf-8") as f:
        matches = json.load(f).get("matches", [])

    df = build_features(matches)

    # Add was_draw from match metadata
    match_lookup = {str(m["match_id"]): m for m in matches}
    df["was_draw"] = df["match_id"].apply(
        lambda mid: int(match_lookup.get(str(mid), {}).get("was_draw", False))
    )
    df["season"] = df["date"].apply(_get_season)

    if feature_cols is None:
        feature_cols = get_feature_columns(include_momentum=False, include_team=True)
        feature_cols = [c for c in feature_cols if c != "momentum"]
        print(f"\n(No feature_cols in pkl — falling back to: {feature_cols})")

    available = [c for c in feature_cols if c in df.columns]
    if len(available) != len(feature_cols):
        missing = set(feature_cols) - set(available)
        print(f"\n  WARNING: missing feature columns in df: {missing}")

    df_eval = df.dropna(subset=available).reset_index(drop=True)
    X = df_eval[available].values
    probs = cal.predict_proba(X)[:, 1]

    eq_rate = df_eval["equalized"].mean()
    draw_rate = df_eval["was_draw"].mean()
    mean_pred = probs.mean()

    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(f"Eval rows:                {len(df_eval)}")
    print(f"Mean predicted prob:      {mean_pred:.4f}  ({mean_pred*100:.1f}%)")
    print(f"Empirical P(equalized):   {eq_rate:.4f}  ({eq_rate*100:.1f}%)")
    print(f"Empirical P(was_draw):    {draw_rate:.4f}  ({draw_rate*100:.1f}%)")
    print()

    gap_to_eq = abs(mean_pred - eq_rate)
    gap_to_draw = abs(mean_pred - draw_rate)
    if gap_to_draw < gap_to_eq:
        verdict = "was_draw  [OK]  (final score is level - CORRECT for PM payout)"
    else:
        verdict = "equalized  [BAD]  (any equalize after entry - OVERESTIMATES draws)"
    print(f"Closest match -> target was likely: {verdict}")

    # Per-season breakdown
    print()
    print("By season (mean predicted vs actual draw rate):")
    df_eval["pred"] = probs
    for season, sub in df_eval.groupby("season"):
        if len(sub) < 50:
            continue
        print(f"  {season}: n={len(sub):5d}  pred={sub['pred'].mean():.3f}  "
              f"eq={sub['equalized'].mean():.3f}  draw={sub['was_draw'].mean():.3f}")


if __name__ == "__main__":
    main()
