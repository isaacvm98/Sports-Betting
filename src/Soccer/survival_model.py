"""
Survival Model for Equalization Timing

Cox Proportional Hazards model that predicts:
  - P(equalize by minute T | game state at entry minute)
  - Hazard curve: instantaneous equalization risk at each minute
  - Optimal entry timing by combining hazard with PM price trajectory

The XGBoost model says WHETHER to bet. This model says WHEN.

Usage:
    python -m src.Soccer.survival_model
    python -m src.Soccer.survival_model --entry-min 60
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

logger = logging.getLogger(__name__)

DATA_DIR = Path("Data/soccer_backtest")
MATCHES_FILE = DATA_DIR / "matches.json"
PM_FILE = DATA_DIR / "pm_1min_prices.json"

LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

# We'll build survival data from multiple entry minutes
# For each (match, entry_minute), the observation is:
#   duration = equalization_minute - entry_minute  (if equalized)
#   duration = 90 - entry_minute                   (if censored / no equalization)
#   event = 1 if equalized, 0 if censored


def load_data():
    with open(MATCHES_FILE, "r", encoding="utf-8") as f:
        matches = json.load(f).get("matches", [])
    pm_data = None
    if PM_FILE.exists():
        with open(PM_FILE, "r") as f:
            pm_data = json.load(f)
    return matches, pm_data


def _score_at_minute(goals, minute):
    h, a = 0, 0
    for g in goals:
        if g["minute"] <= minute:
            h, a = g["home_after"], g["away_after"]
        else:
            break
    return h, a


def _get_season(date_str):
    if not date_str or len(date_str) < 7:
        return "unknown"
    year = int(date_str[:4])
    month = int(date_str[5:7])
    if month >= 8:
        return f"{year}/{year+1}"
    return f"{year-1}/{year}"


def build_survival_data(matches: List[Dict], entry_minutes=None) -> pd.DataFrame:
    """Build survival dataset.

    Each row is a (match, entry_minute) observation with:
      - duration: minutes until equalization or censoring
      - event: 1 if equalized, 0 if censored
      - covariates: game state features at entry minute
    """
    if entry_minutes is None:
        entry_minutes = [55, 58, 60, 63, 65, 68, 70, 73, 75]

    rows = []
    for match in matches:
        goals = match.get("goals", [])
        league = match.get("league", "")
        xg_home = match.get("xg_home")
        xg_away = match.get("xg_away")
        date = match.get("date", "")
        was_draw = match.get("was_draw", False)

        for entry_min in entry_minutes:
            h, a = _score_at_minute(goals, entry_min)
            if abs(h - a) != 1:
                continue

            losing_team = "home" if h < a else "away"

            # Find equalization after entry
            eq_min = None
            for g in goals:
                if g["minute"] > entry_min and g["team"] == losing_team:
                    if g["home_after"] == g["away_after"]:
                        eq_min = g["minute"]
                        break

            if eq_min is not None:
                duration = eq_min - entry_min
                event = 1
            else:
                duration = 95 - entry_min  # ~end of match including stoppage
                event = 0

            # Ensure positive duration
            duration = max(1, duration)

            # Features
            losing_xg = xg_home if losing_team == "home" else xg_away
            winning_xg = xg_away if losing_team == "home" else xg_home
            total_xg = (losing_xg or 0) + (winning_xg or 0)
            xg_share = losing_xg / total_xg if total_xg > 0 else 0.5
            xg_diff = (losing_xg or 0) - (winning_xg or 0)

            goals_before = [g for g in goals if g["minute"] <= entry_min]
            losing_goals_before = sum(
                1 for g in goals_before if g["team"] == losing_team
            )
            total_goals_before = len(goals_before)

            if goals_before:
                mins_since_last_goal = entry_min - goals_before[-1]["minute"]
            else:
                mins_since_last_goal = entry_min

            score_level = h + a

            # Momentum at 70 (only available for that minute)
            momentum = match.get("momentum_at_70") if entry_min == 70 else None

            rows.append({
                "match_id": match.get("match_id"),
                "date": date,
                "league": league,
                "entry_minute": entry_min,
                "duration": duration,
                "event": event,
                "was_draw": int(was_draw),
                "losing_team": losing_team,
                "home_team": match.get("home_team", ""),
                "away_team": match.get("away_team", ""),
                # Covariates
                "xg_share": xg_share,
                "xg_diff": xg_diff,
                "total_xg": total_xg,
                "is_home_losing": 1 if losing_team == "home" else 0,
                "score_level": score_level,
                "total_goals_before": total_goals_before,
                "losing_goals_before": losing_goals_before,
                "mins_since_last_goal": mins_since_last_goal,
                "minutes_remaining": 90 - entry_min,
            })

    df = pd.DataFrame(rows)

    # League one-hots
    for lg in LEAGUES:
        df[f"league_{lg.lower().replace(' ', '_')}"] = (df["league"] == lg).astype(int)

    df["season"] = df["date"].apply(_get_season)
    return df


COVARIATES = [
    "xg_share", "xg_diff", "total_xg",
    "is_home_losing", "score_level",
    "total_goals_before", "losing_goals_before",
    "mins_since_last_goal", "minutes_remaining",
    "league_premier_league", "league_la_liga",
    "league_bundesliga", "league_serie_a", "league_ligue_1",
]


def train_cox(df: pd.DataFrame, covariates=None):
    """Train Cox PH model on survival data."""
    if covariates is None:
        covariates = COVARIATES

    cols = covariates + ["duration", "event"]
    df_clean = df[cols].dropna().copy()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_clean, duration_col="duration", event_col="event")
    return cph


def get_survival_curve(cph, features: Dict, times=None):
    """Get survival curve S(t) for a specific set of features.

    S(t) = P(no equalization by time t after entry)
    So P(equalize by t) = 1 - S(t)
    """
    if times is None:
        times = np.arange(1, 31)  # 1 to 30 minutes after entry

    df_feat = pd.DataFrame([features])
    # Ensure all covariates present
    for c in COVARIATES:
        if c not in df_feat.columns:
            df_feat[c] = 0

    surv = cph.predict_survival_function(df_feat[COVARIATES], times=times)
    return times, 1 - surv.values.flatten()  # P(equalize by t)


def optimal_entry(
    cph, match_features_by_minute: Dict[int, Dict],
    pm_prices_by_minute: Dict[int, float],
    hold_to_expiry: bool = True,
):
    """Find optimal entry minute by combining survival curve with PM prices.

    For each candidate entry minute:
      - P(draw) = P(equalize and draw holds) ≈ P(equalize) * P(draw | equalize)
      - For hold-to-expiry: EV = P(draw) * (1/entry_price - 1) - (1-P(draw))
      - Pick entry with highest EV

    Returns: (best_minute, best_ev, details_dict)
    """
    results = {}

    for entry_min, features in match_features_by_minute.items():
        pm_price = pm_prices_by_minute.get(entry_min)
        if pm_price is None or pm_price < 0.05 or pm_price > 0.50:
            continue

        # Get P(equalize by end of match) from survival curve
        minutes_left = 95 - entry_min
        times = np.arange(1, minutes_left + 1)
        _, cum_prob = get_survival_curve(cph, features, times)
        p_equalize = cum_prob[-1] if len(cum_prob) > 0 else 0

        # For hold-to-expiry draw bet:
        # P(final draw) ≈ p_equalize * 0.65  (rough: ~65% of equalizations hold as draws)
        # Actually let's use the was_draw rate from our data
        p_draw = p_equalize * 0.60  # conservative

        if hold_to_expiry:
            ev = p_draw * (1.0 / pm_price - 1) - (1 - p_draw)
        else:
            # Equalization exit at estimated ~0.42
            exit_price = 0.42
            ev = p_equalize * (exit_price / pm_price - 1) - (1 - p_equalize)

        # Hazard in next 5 minutes (urgency signal)
        if len(cum_prob) >= 5:
            near_hazard = cum_prob[4] - cum_prob[0]  # P(eq in next 5 min)
        else:
            near_hazard = 0

        results[entry_min] = {
            "entry_minute": entry_min,
            "pm_price": pm_price,
            "p_equalize": p_equalize,
            "p_draw": p_draw,
            "ev_per_unit": ev,
            "near_hazard_5min": near_hazard,
        }

    if not results:
        return None, None, {}

    best_min = max(results, key=lambda m: results[m]["ev_per_unit"])
    return best_min, results[best_min]["ev_per_unit"], results


def main():
    parser = argparse.ArgumentParser(description="Survival Model for Equalization Timing")
    parser.add_argument("--entry-min", type=int, default=None,
                        help="Specific entry minute to analyze")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Loading data...")
    matches, pm_data = load_data()
    print(f"Matches: {len(matches)}")

    print("\nBuilding survival dataset...")
    df = build_survival_data(matches)
    print(f"Survival observations: {len(df)}")
    print(f"Events (equalized): {df['event'].sum()} ({df['event'].mean():.1%})")
    print(f"Median duration: {df['duration'].median():.0f} min")
    print(f"Entry minutes: {sorted(df['entry_minute'].unique())}")

    # Season split for OOS
    s1, s2 = "2024/2025", "2025/2026"
    df_s1 = df[df["season"] == s1]
    df_s2 = df[df["season"] == s2]

    print(f"\nTrain ({s1}): {len(df_s1)} obs, {df_s1['event'].sum()} events")
    print(f"Test  ({s2}): {len(df_s2)} obs, {df_s2['event'].sum()} events")

    # Kaplan-Meier (non-parametric baseline)
    print(f"\n{'='*70}")
    print("KAPLAN-MEIER BASELINE")
    print(f"{'='*70}")

    kmf = KaplanMeierFitter()
    kmf.fit(df["duration"], df["event"])
    print("\nP(equalize by T minutes after entry):")
    for t in [5, 10, 15, 20, 25]:
        p = 1 - kmf.predict(t)
        print(f"  +{t:2d} min: {p:.1%}")

    # KM by losing_goals_before (the key split)
    print("\nKM by losing_goals_before:")
    for lg_val in [0, 1, 2]:
        sub = df[df["losing_goals_before"] == lg_val]
        if len(sub) < 10:
            continue
        kmf_sub = KaplanMeierFitter()
        kmf_sub.fit(sub["duration"], sub["event"])
        for t in [10, 20]:
            p = 1 - kmf_sub.predict(t)
            print(f"  goals_before={lg_val}, +{t}min: {p:.1%} (n={len(sub)})")

    # Cox PH — train on s1, evaluate on s2
    print(f"\n{'='*70}")
    print("COX PH MODEL")
    print(f"{'='*70}")

    print("\nTraining on 2024/2025...")
    cph = train_cox(df_s1)
    cph.print_summary(columns=["coef", "exp(coef)", "p"])

    # Concordance index on OOS
    df_s2_clean = df_s2[COVARIATES + ["duration", "event"]].dropna()
    ci = concordance_index(
        df_s2_clean["duration"],
        -cph.predict_partial_hazard(df_s2_clean[COVARIATES]),
        df_s2_clean["event"],
    )
    print(f"\nOOS Concordance Index: {ci:.3f}")

    # Show survival curves for different game states
    print(f"\n{'='*70}")
    print("SURVIVAL CURVES BY GAME STATE")
    print(f"{'='*70}")

    scenarios = [
        ("0-1 at min 60 (haven't scored)", {
            "xg_share": 0.4, "xg_diff": -0.5, "total_xg": 1.5,
            "is_home_losing": 1, "score_level": 1,
            "total_goals_before": 1, "losing_goals_before": 0,
            "mins_since_last_goal": 20, "minutes_remaining": 30,
        }),
        ("1-2 at min 60 (already scored)", {
            "xg_share": 0.5, "xg_diff": 0.0, "total_xg": 2.0,
            "is_home_losing": 1, "score_level": 3,
            "total_goals_before": 3, "losing_goals_before": 1,
            "mins_since_last_goal": 10, "minutes_remaining": 30,
        }),
        ("1-2 at min 70 (already scored)", {
            "xg_share": 0.5, "xg_diff": 0.0, "total_xg": 2.0,
            "is_home_losing": 1, "score_level": 3,
            "total_goals_before": 3, "losing_goals_before": 1,
            "mins_since_last_goal": 10, "minutes_remaining": 20,
        }),
        ("0-1 at min 75 (haven't scored, late)", {
            "xg_share": 0.35, "xg_diff": -0.8, "total_xg": 1.8,
            "is_home_losing": 1, "score_level": 1,
            "total_goals_before": 1, "losing_goals_before": 0,
            "mins_since_last_goal": 30, "minutes_remaining": 15,
        }),
    ]

    # Zero out league dummies for scenarios
    for _, feats in scenarios:
        for lg in LEAGUES:
            feats[f"league_{lg.lower().replace(' ', '_')}"] = 0
        feats["league_premier_league"] = 1  # default to PL

    for name, feats in scenarios:
        times, cum_prob = get_survival_curve(cph, feats, np.arange(1, 31))
        print(f"\n  {name}")
        for t_idx in [4, 9, 14, 19, 24, 29]:
            if t_idx < len(cum_prob):
                print(f"    +{times[t_idx]:2.0f} min: P(eq) = {cum_prob[t_idx]:.1%}")

    # Optimal entry analysis with PM prices
    if pm_data:
        print(f"\n{'='*70}")
        print("OPTIMAL ENTRY TIMING (Cox + PM prices)")
        print(f"{'='*70}")

        # Train final model on all data
        cph_full = train_cox(df)

        pm_by_date = {}
        for pm in pm_data:
            pm_by_date.setdefault(pm["start"][:10], []).append(pm)

        # For qualifying matches in test set with PM data, find optimal entry
        entry_choices = {m: 0 for m in [60, 65, 68, 70, 75]}
        ev_by_minute = {m: [] for m in [60, 65, 68, 70, 75]}

        qual_matches = [m for m in matches if m.get("qualifying") and m.get("date", "") >= "2026-03-01"]

        for match in qual_matches:
            h = match["home_team"].lower()[:8]
            a = match["away_team"].lower()[:8]
            best_pm = None
            for pm in pm_by_date.get(match["date"], []):
                if h in pm["title"].lower() and a in pm["title"].lower():
                    best_pm = pm
                    break
            if not best_pm:
                continue

            prices = {int(k): v for k, v in best_pm.get("prices", {}).items()}

            features_by_min = {}
            for entry_min in [60, 65, 68, 70, 75]:
                goals = match.get("goals", [])
                sh, sa = _score_at_minute(goals, entry_min)
                if abs(sh - sa) != 1:
                    continue
                losing = "home" if sh < sa else "away"
                losing_xg = match.get("xg_home") if losing == "home" else match.get("xg_away")
                winning_xg = match.get("xg_away") if losing == "home" else match.get("xg_home")
                total_xg = (losing_xg or 0) + (winning_xg or 0)
                goals_before = [g for g in goals if g["minute"] <= entry_min]

                features_by_min[entry_min] = {
                    "xg_share": (losing_xg or 0) / total_xg if total_xg > 0 else 0.5,
                    "xg_diff": (losing_xg or 0) - (winning_xg or 0),
                    "total_xg": total_xg,
                    "is_home_losing": 1 if losing == "home" else 0,
                    "score_level": sh + sa,
                    "total_goals_before": len(goals_before),
                    "losing_goals_before": sum(1 for g in goals_before if g["team"] == losing),
                    "mins_since_last_goal": entry_min - goals_before[-1]["minute"] if goals_before else entry_min,
                    "minutes_remaining": 90 - entry_min,
                }
                for lg in LEAGUES:
                    features_by_min[entry_min][f"league_{lg.lower().replace(' ', '_')}"] = (
                        1 if match.get("league") == lg else 0
                    )

            if not features_by_min:
                continue

            best_min, best_ev, details = optimal_entry(
                cph_full, features_by_min, prices, hold_to_expiry=True
            )
            if best_min is not None:
                entry_choices[best_min] = entry_choices.get(best_min, 0) + 1
                for m, d in details.items():
                    ev_by_minute[m].append(d["ev_per_unit"])

        print("\nOptimal entry distribution:")
        for m, count in sorted(entry_choices.items()):
            avg_ev = np.mean(ev_by_minute[m]) if ev_by_minute[m] else 0
            print(f"  Min {m}: chosen {count} times, avg EV {avg_ev:+.3f}")


if __name__ == "__main__":
    main()
