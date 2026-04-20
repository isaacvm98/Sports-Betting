"""
Feature builder for the soccer draw model.

Pure feature engineering — no model training, no I/O beyond reading
matches.json. Consumed by:
  - src/Soccer/train_models.py (production trainer)
  - src/Soccer/draw_scanner.py (live feature construction; builds its own
    feature dict but uses the same column ordering returned by
    get_feature_columns)
  - scripts/* (analysis & inspection)
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path("Data/soccer_backtest")
MATCHES_FILE = DATA_DIR / "matches.json"

LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

# Entry minutes to generate training rows from each match.
# Using multiple entry points per match boosts sample size.
ENTRY_MINUTES = [60, 65, 68, 70, 73, 75]

# Card types that send a player off (count toward red-card features).
SENDOFF_CARDS = {"red", "second_yellow"}


def _card_features_at(cards, entry_minute, losing_team):
    """Card-derived features at the given entry minute, from the
    losing team's perspective. Returns dict with:
      losing_red_count, winning_red_count, red_diff_losing,
      mins_since_red_card (or None if no red yet).
    """
    losing_red = winning_red = 0
    most_recent_red_min = None
    for c in cards or []:
        if c.get("card_type") not in SENDOFF_CARDS:
            continue
        m = c.get("minute", 0) + (c.get("added_time") or 0) / 60.0
        if m > entry_minute:
            continue
        team = c.get("team")
        if team == losing_team:
            losing_red += 1
        else:
            winning_red += 1
        if most_recent_red_min is None or m > most_recent_red_min:
            most_recent_red_min = m
    return {
        "losing_red_count": losing_red,
        "winning_red_count": winning_red,
        # Positive value = losing team has man advantage (opponent down)
        "red_diff_losing": winning_red - losing_red,
        "mins_since_red_card": (
            entry_minute - most_recent_red_min
            if most_recent_red_min is not None else None
        ),
    }


def _sub_features_at(subs, entry_minute, losing_team):
    """Substitution-derived features at the given entry minute."""
    losing_subs = winning_subs = 0
    for s in subs or []:
        m = s.get("minute", 0) + (s.get("added_time") or 0) / 60.0
        if m > entry_minute:
            continue
        if s.get("team") == losing_team:
            losing_subs += 1
        else:
            winning_subs += 1
    return {
        "losing_subs_used": losing_subs,
        "winning_subs_used": winning_subs,
    }


CARD_FEATURE_COLS = [
    "losing_red_count",
    "winning_red_count",
    "red_diff_losing",
    "mins_since_red_card",
]
SUB_FEATURE_COLS = [
    "losing_subs_used",
    "winning_subs_used",
]


def load_matches() -> List[Dict]:
    with open(MATCHES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("matches", [])


def _get_season(date_str: str) -> str:
    """Map a date string to its football season (e.g. '2025-01-15' -> '2024/2025')."""
    if not date_str or len(date_str) < 7:
        return "unknown"
    year = int(date_str[:4])
    month = int(date_str[5:7])
    if month >= 8:
        return f"{year}/{year+1}"
    return f"{year-1}/{year}"


def _score_at_minute(goals: List[Dict], minute: int) -> Tuple[int, int]:
    """Reconstruct score at a given minute from goal events."""
    h, a = 0, 0
    for g in goals:
        if g["minute"] <= minute:
            h, a = g["home_after"], g["away_after"]
        else:
            break
    return h, a


def _momentum_at_minute(
    momentum_raw: Optional[List[Dict]], minute: int, losing_team: str
) -> Optional[float]:
    """Average SofaScore momentum over a 10-min window ending at `minute`.

    Flipped so positive = losing team attacking, then normalized to [0, 1].
    """
    if not momentum_raw:
        return None

    window_start = max(1, minute - 10)
    window_end = minute

    values = []
    for dp in momentum_raw:
        if not isinstance(dp, dict):
            continue
        m = dp.get("minute", -1)
        v = dp.get("value", 0)
        if window_start <= m <= window_end:
            values.append(v)

    if not values:
        return None

    avg = sum(values) / len(values)
    if losing_team == "away":
        avg = -avg
    normalized = (avg + 100) / 200
    return round(max(0.0, min(1.0, normalized)), 3)


def _build_team_rolling_stats(matches: List[Dict]) -> Dict[str, Dict]:
    """Per-(team, league, season) rolling stats from prior matches only (no leakage).

    Returns: {match_id -> {home_goals_per_90, home_late_goals_pct,
                            home_conceded_per_90, away_*}}
    """
    sorted_matches = sorted(matches, key=lambda m: m.get("date", ""))
    team_history: Dict[str, List[Dict]] = defaultdict(list)
    match_stats: Dict[str, Dict] = {}

    for match in sorted_matches:
        date = match.get("date", "")
        league = match.get("league", "")
        home = match.get("home_team", "")
        away = match.get("away_team", "")
        goals = match.get("goals", [])
        match_id = str(match.get("match_id", ""))

        if not date or not home or not away:
            continue

        year = int(date[:4]) if date else 0
        month = int(date[5:7]) if date else 0
        season_year = year if month >= 8 else year - 1

        home_key = f"{home}_{league}_{season_year}"
        away_key = f"{away}_{league}_{season_year}"

        def _team_stats(prior_matches):
            if len(prior_matches) < 3:
                return None, None, None
            n = len(prior_matches)
            total_goals = sum(m["goals_scored"] for m in prior_matches)
            total_conceded = sum(m["goals_conceded"] for m in prior_matches)
            late_goals = sum(m["late_goals"] for m in prior_matches)
            return (
                total_goals / n,
                total_conceded / n,
                late_goals / total_goals if total_goals > 0 else 0,
            )

        home_g90, home_c90, home_late = _team_stats(team_history[home_key])
        away_g90, away_c90, away_late = _team_stats(team_history[away_key])

        match_stats[match_id] = {
            "home_goals_per_90": home_g90,
            "home_conceded_per_90": home_c90,
            "home_late_goals_pct": home_late,
            "away_goals_per_90": away_g90,
            "away_conceded_per_90": away_c90,
            "away_late_goals_pct": away_late,
        }

        home_goals = sum(1 for g in goals if g.get("team") == "home")
        away_goals = sum(1 for g in goals if g.get("team") == "away")
        home_late_g = sum(1 for g in goals if g.get("team") == "home" and g.get("minute", 0) >= 60)
        away_late_g = sum(1 for g in goals if g.get("team") == "away" and g.get("minute", 0) >= 60)

        team_history[home_key].append({
            "goals_scored": home_goals,
            "goals_conceded": away_goals,
            "late_goals": home_late_g,
        })
        team_history[away_key].append({
            "goals_scored": away_goals,
            "goals_conceded": home_goals,
            "late_goals": away_late_g,
        })

    return match_stats


def build_features(matches: List[Dict]) -> pd.DataFrame:
    """Build per-(match, entry_minute) feature rows.

    Each match contributes one row per entry minute where the score
    differential is exactly 1 goal. The `equalized` column captures
    whether the losing team scored to level the score after the entry
    minute (analytic only — the live model trains against `was_draw`,
    which is added downstream from match metadata).
    """
    team_stats = _build_team_rolling_stats(matches)
    rows = []

    for match in matches:
        goals = match.get("goals", [])
        if not goals and match.get("home_score", 0) == 0 and match.get("away_score", 0) == 0:
            continue

        league = match.get("league", "Unknown")
        xg_home = match.get("xg_home")
        xg_away = match.get("xg_away")
        date = match.get("date", "")
        match_id = str(match.get("match_id", ""))
        ts = team_stats.get(match_id, {})
        cards = match.get("cards") or []
        substitutions = match.get("substitutions") or []
        has_incidents = "cards" in match  # field present (even if empty)

        for entry_min in ENTRY_MINUTES:
            h, a = _score_at_minute(goals, entry_min)
            if abs(h - a) != 1:
                continue

            losing_team = "home" if h < a else "away"

            equalized = False
            eq_minute = None
            for g in goals:
                if g["minute"] > entry_min and g["team"] == losing_team:
                    if g["home_after"] == g["away_after"]:
                        equalized = True
                        eq_minute = g["minute"]
                        break

            momentum = match.get("momentum_at_70") if entry_min == 70 else None

            losing_xg = xg_home if losing_team == "home" else xg_away
            winning_xg = xg_away if losing_team == "home" else xg_home
            xg_share = xg_diff = total_xg = None
            if losing_xg is not None and winning_xg is not None:
                total = losing_xg + winning_xg
                xg_share = losing_xg / total if total > 0 else 0.5
                xg_diff = losing_xg - winning_xg
                total_xg = total

            goals_before = [g for g in goals if g["minute"] <= entry_min]
            losing_goals_before = sum(1 for g in goals_before if g["team"] == losing_team)
            mins_since_last_goal = (
                entry_min - goals_before[-1]["minute"]
                if goals_before else entry_min
            )

            if losing_team == "home":
                losing_goals_per_90 = ts.get("home_goals_per_90")
                losing_late_pct = ts.get("home_late_goals_pct")
                opp_conceded_per_90 = ts.get("away_conceded_per_90")
            else:
                losing_goals_per_90 = ts.get("away_goals_per_90")
                losing_late_pct = ts.get("away_late_goals_pct")
                opp_conceded_per_90 = ts.get("home_conceded_per_90")

            row = {
                "match_id": match_id,
                "date": date,
                "league": league,
                "home_team": match.get("home_team", ""),
                "away_team": match.get("away_team", ""),
                "losing_team": losing_team,
                "entry_minute": entry_min,
                "momentum": momentum,
                "xg_share": xg_share,
                "xg_diff": xg_diff,
                "total_xg": total_xg,
                "is_home_losing": 1 if losing_team == "home" else 0,
                "score_level": h + a,
                "total_goals_before": len(goals_before),
                "losing_goals_before": losing_goals_before,
                "mins_since_last_goal": mins_since_last_goal,
                "losing_goals_per_90": losing_goals_per_90,
                "losing_late_pct": losing_late_pct,
                "opp_conceded_per_90": opp_conceded_per_90,
                "equalized": int(equalized),
                "equalization_minute": eq_minute,
            }
            # Card / sub features — only present when the source has them.
            # XGBoost handles NaN natively; the trainer opts in via flags.
            if has_incidents:
                row.update(_card_features_at(cards, entry_min, losing_team))
                row.update(_sub_features_at(substitutions, entry_min, losing_team))
            rows.append(row)

    df = pd.DataFrame(rows)

    for lg in LEAGUES:
        df[f"league_{lg.lower().replace(' ', '_')}"] = (df["league"] == lg).astype(int)

    return df


def get_feature_columns(
    include_momentum: bool = True,
    include_team: bool = True,
    include_cards: bool = False,
    include_subs: bool = False,
) -> List[str]:
    """Canonical feature column ordering — must match what the live scanner builds.

    `include_cards` / `include_subs` default False so the existing trainer
    call sites keep producing the same column set. Once the data has been
    re-scraped with cards/subs populated, the trainer can opt in.
    """
    cols = [
        "xg_share", "xg_diff", "total_xg",
        "is_home_losing", "score_level", "total_goals_before",
        "losing_goals_before", "mins_since_last_goal", "entry_minute",
    ]
    if include_momentum:
        cols.insert(0, "momentum")
    if include_team:
        cols.extend(["losing_goals_per_90", "losing_late_pct", "opp_conceded_per_90"])
    if include_cards:
        cols.extend(CARD_FEATURE_COLS)
    if include_subs:
        cols.extend(SUB_FEATURE_COLS)
    for lg in LEAGUES:
        cols.append(f"league_{lg.lower().replace(' ', '_')}")
    return cols
