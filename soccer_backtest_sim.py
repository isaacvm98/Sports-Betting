"""
Soccer Draw Backtest – XGBoost + Cox, YES + NO (hold-to-expiry)

Walk-forward: train on 2024/25, test on 2025/26.
Trains XGBoost on was_draw target (not equalized) with no momentum.
Combines with Cox survival model for entry timing.
Uses actual PM 1-min prices.

YES trade: buy draw when model says underpriced  (model_prob > pm_price)
NO  trade: buy no-draw when model says overpriced (pm_price > model_prob)

Usage:
    python soccer_backtest_sim.py
    python soccer_backtest_sim.py --side yes
    python soccer_backtest_sim.py --side no
    python soccer_backtest_sim.py --min-edge 0.03
    python soccer_backtest_sim.py --entry-minutes 70
"""

import json
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from src.Soccer.feature_builder import (
    build_features, get_feature_columns, _get_season, _score_at_minute,
)
from src.Soccer.survival_model import (
    build_survival_data, train_cox, get_survival_curve, COVARIATES,
)

LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

# ── Config ───────────────────────────────────────────────────────────────
STARTING_BANKROLL = 1000.0
KELLY_FRACTION = 0.15
MAX_BET_PCT = 8.0
MIN_EDGE = 0.05
MIN_DRAW_PRICE = 0.05
MAX_DRAW_PRICE = 0.50
TRAIN_SEASON = "2024/2025"
TEST_SEASON = "2025/2026"


# ── Kelly ────────────────────────────────────────────────────────────────
def kelly_yes(p_draw, draw_price):
    b = (1.0 / draw_price) - 1.0
    q = 1.0 - p_draw
    k = (p_draw * b - q) / b
    return max(0.0, k)


def kelly_no(p_draw, draw_price):
    no_price = 1.0 - draw_price
    b = (1.0 / no_price) - 1.0
    p_no = 1.0 - p_draw
    k = (p_no * b - p_draw) / b
    return max(0.0, k)


# ── Train models (walk-forward) ─────────────────────────────────────────
def train_models(matches_raw):
    """Train XGBoost (was_draw target) and Cox on 2024/25 data."""
    # Build features from all matches (one row per entry minute)
    df = build_features(matches_raw)

    # Add was_draw target
    match_lookup = {str(m["match_id"]): m for m in matches_raw}
    df["was_draw"] = df["match_id"].apply(
        lambda mid: int(match_lookup.get(str(mid), {}).get("was_draw", False))
    )
    df["season"] = df["date"].apply(_get_season)

    # Feature columns: no momentum, with team stats
    feature_cols = get_feature_columns(include_momentum=False, include_team=True)
    feature_cols = [c for c in feature_cols if c != "momentum"]

    # Train on 2024/25
    df_train = df[df["season"] == TRAIN_SEASON].dropna(
        subset=[c for c in feature_cols if c in df.columns]
    )
    X_tr = df_train[feature_cols].values
    y_tr = df_train["was_draw"].values

    n_pos = y_tr.sum()
    scale = (len(y_tr) - n_pos) / n_pos if n_pos > 0 else 1.0

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_clf.fit(X_tr, y_tr, verbose=False)
    xgb_cal = CalibratedClassifierCV(xgb_clf, method="isotonic", cv=3)
    xgb_cal.fit(X_tr, y_tr)

    print(f"  XGBoost trained on {TRAIN_SEASON}: {len(X_tr)} rows, "
          f"{y_tr.sum()} draws ({y_tr.mean():.1%})")

    # Train Cox on 2024/25
    surv_df = build_survival_data(matches_raw)
    surv_train = surv_df[surv_df["season"] == TRAIN_SEASON]
    cox = train_cox(surv_train)
    print(f"  Cox trained on {TRAIN_SEASON}: {len(surv_train)} obs, "
          f"C-index={cox.concordance_index_:.3f}")

    return xgb_cal, feature_cols, cox


# ── Simulation ───────────────────────────────────────────────────────────
def run_simulation(side="both", min_edge=MIN_EDGE, entry_minutes=None):
    if entry_minutes is None:
        entry_minutes = [60, 65, 68, 70, 75]

    # Load data
    with open("Data/soccer_backtest/matches.json", encoding="utf-8") as f:
        matches_raw = json.load(f)["matches"]
    with open("Data/soccer_backtest/pm_1min_prices.json") as f:
        pm_data = json.load(f)

    pm_by_date = {}
    for pm in pm_data:
        pm_by_date.setdefault(pm["start"][:10], []).append(pm)

    match_lookup = {str(m["match_id"]): m for m in matches_raw}

    print("=" * 80)
    print("  SOCCER DRAW BACKTEST – XGBoost + Cox, YES + NO")
    print("=" * 80)
    print(f"  Train: {TRAIN_SEASON} | Test: {TEST_SEASON} (walk-forward OOS)")
    print(f"  Target: was_draw | Features: no momentum, with team stats")
    print(f"  Entry minutes: {entry_minutes}")
    print(f"  Min edge: {min_edge:.0%} | Side: {side.upper()}")
    print(f"  Kelly frac: {KELLY_FRACTION:.0%} | Max bet: {MAX_BET_PCT:.0f}%")
    print(f"  Exit: hold-to-expiry")
    print()

    # Train models
    print("Training models...")
    xgb_cal, feature_cols, cox = train_models(matches_raw)

    # Get test qualifying matches (25/26)
    test_matches = [
        m for m in matches_raw
        if m.get("qualifying") and _get_season(m.get("date", "")) == TEST_SEASON
    ]
    test_matches.sort(key=lambda m: m.get("date", ""))
    print(f"\n  Test qualifying matches ({TEST_SEASON}): {len(test_matches)}")
    print("=" * 80)

    bankroll = STARTING_BANKROLL
    peak = STARTING_BANKROLL
    trades = []

    for match in test_matches:
        home = match["home_team"]
        away = match["away_team"]
        date = match["date"]
        league = match.get("league", "")
        goals = match.get("goals", [])
        was_draw = match.get("was_draw", False)

        # Match to PM prices
        h_key = home.lower()[:8]
        a_key = away.lower()[:8]
        best_pm = None
        for pm in pm_by_date.get(date, []):
            if h_key in pm["title"].lower() and a_key in pm["title"].lower():
                best_pm = pm
                break
        if not best_pm:
            continue
        prices = {int(k): v for k, v in best_pm.get("prices", {}).items()}

        # Evaluate each entry minute, pick best EV
        best_ev_yes = -999
        best_entry_yes = None
        best_ev_no = -999
        best_entry_no = None

        for entry_min in entry_minutes:
            sh, sa = _score_at_minute(goals, entry_min)
            if abs(sh - sa) != 1:
                continue

            losing = "home" if sh < sa else "away"
            losing_xg = match.get("xg_home") if losing == "home" else match.get("xg_away")
            winning_xg = match.get("xg_away") if losing == "home" else match.get("xg_home")
            total_xg = (losing_xg or 0) + (winning_xg or 0)
            goals_before = [g for g in goals if g["minute"] <= entry_min]
            losing_goals = sum(1 for g in goals_before if g["team"] == losing)

            feats = {
                "xg_share": (losing_xg or 0) / total_xg if total_xg > 0 else 0.5,
                "xg_diff": (losing_xg or 0) - (winning_xg or 0),
                "total_xg": total_xg,
                "is_home_losing": 1 if losing == "home" else 0,
                "score_level": sh + sa,
                "total_goals_before": len(goals_before),
                "losing_goals_before": losing_goals,
                "mins_since_last_goal": (
                    entry_min - goals_before[-1]["minute"] if goals_before else entry_min
                ),
                "entry_minute": entry_min,
                "losing_goals_per_90": 0,
                "losing_late_pct": 0,
                "opp_conceded_per_90": 0,
            }
            for lg in LEAGUES:
                feats[f"league_{lg.lower().replace(' ', '_')}"] = (
                    1 if league == lg else 0
                )

            # XGBoost prediction
            X_pred = np.array([[feats.get(c, 0) for c in feature_cols]])
            draw_prob = float(xgb_cal.predict_proba(X_pred)[0, 1])

            # PM price
            pm_price = (
                prices.get(entry_min)
                or prices.get(entry_min - 1)
                or prices.get(entry_min + 1)
            )
            if pm_price is None or pm_price < MIN_DRAW_PRICE or pm_price > MAX_DRAW_PRICE:
                continue

            edge = draw_prob - pm_price

            # Survival: near-term hazard
            surv_feats = {k: feats.get(k, 0) for k in COVARIATES}
            surv_feats["minutes_remaining"] = 90 - entry_min
            _, cum_prob = get_survival_curve(cox, surv_feats, np.arange(1, 31))
            near_hazard = cum_prob[4] if len(cum_prob) > 4 else 0

            # EV for YES
            if edge > 0:
                ev_yes = draw_prob * (1.0 / pm_price - 1) - (1 - draw_prob)
                if ev_yes > best_ev_yes and edge >= min_edge:
                    best_ev_yes = ev_yes
                    best_entry_yes = {
                        "entry_min": entry_min, "draw_prob": draw_prob,
                        "pm_price": pm_price, "edge": edge,
                        "near_hazard": near_hazard, "score": f"{sh}-{sa}",
                    }

            # EV for NO
            if edge < 0:
                no_price = 1.0 - pm_price
                ev_no = (1 - draw_prob) * (1.0 / no_price - 1) - draw_prob
                if ev_no > best_ev_no and abs(edge) >= min_edge:
                    best_ev_no = ev_no
                    best_entry_no = {
                        "entry_min": entry_min, "draw_prob": draw_prob,
                        "pm_price": pm_price, "edge": abs(edge),
                        "near_hazard": near_hazard, "score": f"{sh}-{sa}",
                    }

        # Execute trades
        for bet_side, entry in [("YES", best_entry_yes), ("NO", best_entry_no)]:
            if entry is None:
                continue
            if side not in ("both", bet_side.lower()):
                continue

            dp = entry["draw_prob"]
            pp = entry["pm_price"]

            if bet_side == "YES":
                k = kelly_yes(dp, pp)
            else:
                k = kelly_no(dp, pp)

            if k <= 0:
                continue

            kelly_pct = min(k * KELLY_FRACTION * 100, MAX_BET_PCT)
            bet = round(bankroll * (kelly_pct / 100), 2)
            if bet < 1.0:
                continue

            # Settlement
            if bet_side == "YES":
                if was_draw:
                    pnl = round(bet / pp - bet, 2)
                    result = "WIN"
                else:
                    pnl = -bet
                    result = "LOSS"
            else:
                no_price = 1.0 - pp
                if not was_draw:
                    pnl = round(bet / no_price - bet, 2)
                    result = "WIN"
                else:
                    pnl = -bet
                    result = "LOSS"

            bankroll = round(bankroll + pnl, 2)
            peak = max(peak, bankroll)

            trades.append({
                "date": date, "league": league,
                "match": f"{home} vs {away}",
                "score": entry["score"],
                "min": entry["entry_min"],
                "side": bet_side, "dp": pp,
                "model": dp, "edge": entry["edge"],
                "hazard": entry["near_hazard"],
                "kelly": kelly_pct, "bet": bet,
                "result": result, "pnl": pnl,
                "bank": bankroll, "was_draw": was_draw,
            })

    # ── Trade log ────────────────────────────────────────────────────────
    hdr = (f"{'Date':<12}{'League':<14}{'Match':<30}{'Sc':<5}{'M':>3}"
           f"{'Side':<5}{'DP':>6}{'Mod':>6}{'Edge':>6}{'Bet':>7}"
           f" {'':>5}{'P&L':>8}{'Bank':>9}")
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for t in trades:
        print(
            f"{t['date']:<12}{t['league']:<14}{t['match']:<30}"
            f"{t['score']:<5}{t['min']:>3}"
            f"{t['side']:<5}{t['dp']:>6.3f}{t['model']:>6.3f}"
            f"{t['edge']:>5.1%}{t['bet']:>7.2f} {t['result']:<5}"
            f"${t['pnl']:>+7.2f}${t['bank']:>8.2f}"
        )

    if not trades:
        print("  No trades passed edge/Kelly filters.")
        return

    # ── Summary by side ──────────────────────────────────────────────────
    for label, subset in [("ALL", trades),
                          ("YES", [t for t in trades if t["side"] == "YES"]),
                          ("NO", [t for t in trades if t["side"] == "NO"])]:
        if not subset:
            continue
        wins = sum(1 for t in subset if t["result"] == "WIN")
        losses = len(subset) - wins
        total_pnl = sum(t["pnl"] for t in subset)
        total_wag = sum(t["bet"] for t in subset)

        run_peak = STARTING_BANKROLL
        max_dd = 0
        for t in trades:  # use full trade sequence for drawdown
            run_peak = max(run_peak, t["bank"])
            dd = (t["bank"] - run_peak) / run_peak
            max_dd = min(max_dd, dd)

        print(f"\n{'='*80}")
        print(f"  {label} TRADES")
        print(f"{'='*80}")
        print(f"  Trades: {len(subset)} ({wins}W-{losses}L)")
        print(f"  Win rate: {wins/len(subset)*100:.1f}%")
        print(f"  Total P&L: ${total_pnl:+.2f}")
        if total_wag:
            print(f"  ROI: {total_pnl/total_wag*100:+.1f}% on ${total_wag:.2f} wagered")
        if label == "ALL":
            print(f"  Final bankroll: ${bankroll:.2f}")
            print(f"  Max drawdown: {max_dd:.1%}")
            print(f"  Peak bankroll: ${peak:.2f}")

    # ── Monthly breakdown ────────────────────────────────────────────────
    monthly = defaultdict(lambda: {"t": 0, "w": 0, "pnl": 0.0, "wag": 0.0,
                                    "yes_t": 0, "yes_w": 0, "no_t": 0, "no_w": 0})
    for t in trades:
        mo = t["date"][:7]
        monthly[mo]["t"] += 1
        monthly[mo]["pnl"] += t["pnl"]
        monthly[mo]["wag"] += t["bet"]
        if t["result"] == "WIN":
            monthly[mo]["w"] += 1
        if t["side"] == "YES":
            monthly[mo]["yes_t"] += 1
            if t["result"] == "WIN":
                monthly[mo]["yes_w"] += 1
        else:
            monthly[mo]["no_t"] += 1
            if t["result"] == "WIN":
                monthly[mo]["no_w"] += 1

    print(f"\n  {'Month':<10}{'Trades':>7}{'W-L':>8}{'YES':>8}{'NO':>8}{'P&L':>10}{'ROI':>8}")
    print(f"  {'-'*59}")
    for mo in sorted(monthly):
        d = monthly[mo]
        roi = d["pnl"] / d["wag"] * 100 if d["wag"] else 0
        yes_s = f"{d['yes_w']}W-{d['yes_t']-d['yes_w']}L" if d["yes_t"] else "-"
        no_s = f"{d['no_w']}W-{d['no_t']-d['no_w']}L" if d["no_t"] else "-"
        print(f"  {mo:<10}{d['t']:>7}{d['w']}W-{d['t']-d['w']}L"
              f"{yes_s:>8}{no_s:>8}{d['pnl']:>+10.2f}{roi:>+7.1f}%")

    # ── League breakdown ─────────────────────────────────────────────────
    by_lg = defaultdict(lambda: {"t": 0, "w": 0, "pnl": 0.0,
                                  "yes_t": 0, "yes_w": 0, "no_t": 0, "no_w": 0})
    for t in trades:
        lg = t["league"]
        by_lg[lg]["t"] += 1
        by_lg[lg]["pnl"] += t["pnl"]
        if t["result"] == "WIN":
            by_lg[lg]["w"] += 1
        if t["side"] == "YES":
            by_lg[lg]["yes_t"] += 1
            if t["result"] == "WIN":
                by_lg[lg]["yes_w"] += 1
        else:
            by_lg[lg]["no_t"] += 1
            if t["result"] == "WIN":
                by_lg[lg]["no_w"] += 1

    print(f"\n  {'League':<16}{'Trades':>7}{'W-L':>8}{'YES':>8}{'NO':>8}{'P&L':>10}")
    print(f"  {'-'*57}")
    for lg in sorted(by_lg, key=lambda x: by_lg[x]["pnl"], reverse=True):
        d = by_lg[lg]
        yes_s = f"{d['yes_w']}W-{d['yes_t']-d['yes_w']}L" if d["yes_t"] else "-"
        no_s = f"{d['no_w']}W-{d['no_t']-d['no_w']}L" if d["no_t"] else "-"
        print(f"  {lg:<16}{d['t']:>7}{d['w']}W-{d['t']-d['w']}L"
              f"{yes_s:>8}{no_s:>8}{d['pnl']:>+10.2f}")

    # ── Edge bucket breakdown ────────────────────────────────────────────
    print(f"\n  {'Edge bucket':<14}{'Side':<6}{'Trades':>7}{'W-L':>8}{'P&L':>10}{'ROI':>8}")
    print(f"  {'-'*53}")
    edge_buckets = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]
    for side_label in ["YES", "NO"]:
        for lo, hi in edge_buckets:
            bucket = [t for t in trades if t["side"] == side_label
                      and lo <= t["edge"] < hi]
            if not bucket:
                continue
            w = sum(1 for t in bucket if t["result"] == "WIN")
            pnl = sum(t["pnl"] for t in bucket)
            wag = sum(t["bet"] for t in bucket)
            roi = pnl / wag * 100 if wag else 0
            print(f"  {lo:.0%}-{hi:.0%}{'':<8}{side_label:<6}{len(bucket):>7}"
                  f"{w}W-{len(bucket)-w}L{pnl:>+10.2f}{roi:>+7.1f}%")

    # ── Entry minute distribution ────────────────────────────────────────
    min_counts = Counter(t["min"] for t in trades)
    print(f"\n  Entry minute distribution: {dict(sorted(min_counts.items()))}")

    # ── Model calibration check ──────────────────────────────────────────
    print(f"\n  {'Model prob bucket':<20}{'N':>5}{'Actual draw%':>14}{'Avg model':>12}")
    print(f"  {'-'*51}")
    cal_df = pd.DataFrame([{"model": t["model"], "was_draw": t["was_draw"]} for t in trades])
    for lo, hi in [(0.0, 0.10), (0.10, 0.15), (0.15, 0.20),
                   (0.20, 0.30), (0.30, 0.50), (0.50, 1.0)]:
        b = cal_df[(cal_df["model"] >= lo) & (cal_df["model"] < hi)]
        if len(b) >= 3:
            print(f"  {lo:.0%}-{hi:.0%}{'':<14}{len(b):>5}"
                  f"{b['was_draw'].mean():>13.1%}{b['model'].mean():>12.3f}")

    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soccer draw backtest – XGB + Cox, YES + NO")
    parser.add_argument("--side", choices=["yes", "no", "both"], default="both")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE)
    parser.add_argument("--entry-minutes", type=int, nargs="+", default=None,
                        help="Entry minutes to evaluate (default: 60 65 68 70 75)")
    args = parser.parse_args()
    run_simulation(side=args.side, min_edge=args.min_edge,
                   entry_minutes=args.entry_minutes)
