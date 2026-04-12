"""
Equalizer Model — XGBoost classifier for P(equalize | trailing by 1 at entry minute)

Features engineered from backtest match data:
  - momentum_at_entry: SofaScore momentum graph (0-1, higher = losing team attacking)
  - losing_team_xg_share: xG share of losing team (0-1)
  - xg_differential: losing team xG minus winning team xG
  - total_xg: sum of both teams' xG (game openness proxy)
  - league: one-hot encoded (5 leagues)
  - is_home_losing: 1 if the home team is trailing
  - score_level: total goals at entry (1-0 → 1, 2-1 → 3, etc.)
  - minutes_since_last_goal: time since most recent goal before entry
  - losing_team_goals_before_entry: shows the trailing team can score
  - total_goals_before_entry: game openness

Training uses walk-forward by date to prevent leakage.
Calibrated via isotonic regression for well-calibrated probabilities.

Usage:
    # Train and evaluate
    python -m src.Soccer.equalizer_model

    # Train and save model for paper trading
    python -m src.Soccer.equalizer_model --save

    # Predict on a live signal
    from src.Soccer.equalizer_model import EqualizerModel
    model = EqualizerModel.load()
    prob = model.predict_proba(features_dict)
"""

import json
import logging
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

logger = logging.getLogger(__name__)

DATA_DIR = Path("Data/soccer_backtest")
MODEL_DIR = Path("Data/soccer_models")
MATCHES_FILE = DATA_DIR / "matches.json"
MODEL_FILE = MODEL_DIR / "equalizer_xgb.pkl"

LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

# Entry minutes to generate training rows from each match
# Using multiple entry points per match to boost sample size
ENTRY_MINUTES = [60, 65, 68, 70, 73, 75]


def load_matches() -> List[Dict]:
    """Load all matches from backtest cache."""
    with open(MATCHES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("matches", [])


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
    """Average momentum over a window centered on the entry minute.

    Uses 65-75 window for min 70, adapts proportionally for other minutes.
    """
    if not momentum_raw:
        return None

    # Use a 10-minute window ending at entry minute
    window_start = max(1, minute - 10)
    window_end = minute

    values = []
    for dp in momentum_raw:
        if isinstance(dp, dict):
            m = dp.get("minute", -1)
            v = dp.get("value", 0)
        else:
            continue
        if window_start <= m <= window_end:
            values.append(v)

    if not values:
        return None

    avg = sum(values) / len(values)

    # Flip so positive = losing team momentum
    if losing_team == "away":
        avg = -avg

    # Normalize from [-100, 100] to [0, 1]
    normalized = (avg + 100) / 200
    return round(max(0.0, min(1.0, normalized)), 3)


def _build_team_rolling_stats(matches: List[Dict]) -> Dict[str, Dict]:
    """Build rolling team stats from all matches (computed per-season).

    For each (team, league, season) computes rolling stats from all matches
    played BEFORE the current date (no leakage).

    Returns: {match_id -> {losing_team_goals_per_90, losing_team_late_goals_pct,
                           opponent_conceded_per_90}}
    """
    # Group matches by league + season (season = year from date)
    from collections import defaultdict

    # Sort all matches by date
    sorted_matches = sorted(matches, key=lambda m: m.get("date", ""))

    # Track per-team stats: {(team, league, season_year) -> list of match records}
    team_history: Dict[str, List[Dict]] = defaultdict(list)

    # For each match, compute rolling stats for both teams, then store
    match_stats = {}

    for match in sorted_matches:
        date = match.get("date", "")
        league = match.get("league", "")
        home = match.get("home_team", "")
        away = match.get("away_team", "")
        goals = match.get("goals", [])
        match_id = str(match.get("match_id", ""))

        if not date or not home or not away:
            continue

        # Season key: use year (Aug-Dec = first year of season, Jan-Jul = same season)
        year = int(date[:4]) if date else 0
        month = int(date[5:7]) if date else 0
        season_year = year if month >= 8 else year - 1

        home_key = f"{home}_{league}_{season_year}"
        away_key = f"{away}_{league}_{season_year}"

        # Compute stats from PRIOR matches (no leakage)
        home_prior = team_history[home_key]
        away_prior = team_history[away_key]

        def _team_stats(prior_matches):
            if len(prior_matches) < 3:
                return None, None, None
            n = len(prior_matches)
            total_goals = sum(m["goals_scored"] for m in prior_matches)
            total_conceded = sum(m["goals_conceded"] for m in prior_matches)
            late_goals = sum(m["late_goals"] for m in prior_matches)
            goals_per_90 = total_goals / n
            conceded_per_90 = total_conceded / n
            late_pct = late_goals / total_goals if total_goals > 0 else 0
            return goals_per_90, conceded_per_90, late_pct

        home_g90, home_c90, home_late = _team_stats(home_prior)
        away_g90, away_c90, away_late = _team_stats(away_prior)

        match_stats[match_id] = {
            "home_goals_per_90": home_g90,
            "home_conceded_per_90": home_c90,
            "home_late_goals_pct": home_late,
            "away_goals_per_90": away_g90,
            "away_conceded_per_90": away_c90,
            "away_late_goals_pct": away_late,
        }

        # Now add this match to team histories
        home_goals = sum(1 for g in goals if g.get("team") == "home")
        away_goals = sum(1 for g in goals if g.get("team") == "away")
        home_late = sum(1 for g in goals if g.get("team") == "home" and g.get("minute", 0) >= 60)
        away_late = sum(1 for g in goals if g.get("team") == "away" and g.get("minute", 0) >= 60)

        team_history[home_key].append({
            "goals_scored": home_goals,
            "goals_conceded": away_goals,
            "late_goals": home_late,
        })
        team_history[away_key].append({
            "goals_scored": away_goals,
            "goals_conceded": home_goals,
            "late_goals": away_late,
        })

    return match_stats


def build_features(matches: List[Dict]) -> pd.DataFrame:
    """Build feature matrix from match data.

    Each match can generate multiple rows (one per entry minute where
    the losing team is trailing by exactly 1 goal).
    """
    # Pre-compute rolling team stats
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

        # Team rolling stats for this match
        ts = team_stats.get(match_id, {})

        for entry_min in ENTRY_MINUTES:
            h, a = _score_at_minute(goals, entry_min)

            # Must be trailing by exactly 1
            if abs(h - a) != 1:
                continue

            losing_team = "home" if h < a else "away"

            # Check equalization after entry minute
            equalized = False
            eq_minute = None
            for g in goals:
                if g["minute"] > entry_min and g["team"] == losing_team:
                    if g["home_after"] == g["away_after"]:
                        equalized = True
                        eq_minute = g["minute"]
                        break

            # Momentum at entry
            momentum = None
            if entry_min == 70:
                momentum = match.get("momentum_at_70")

            # xG features
            losing_xg = xg_home if losing_team == "home" else xg_away
            winning_xg = xg_away if losing_team == "home" else xg_home
            xg_share = None
            xg_diff = None
            total_xg = None
            if losing_xg is not None and winning_xg is not None:
                total = losing_xg + winning_xg
                xg_share = losing_xg / total if total > 0 else 0.5
                xg_diff = losing_xg - winning_xg
                total_xg = total

            # Goal timing features
            goals_before = [g for g in goals if g["minute"] <= entry_min]
            total_goals_before = len(goals_before)
            losing_goals_before = sum(
                1 for g in goals_before if g["team"] == losing_team
            )

            # Minutes since last goal
            if goals_before:
                last_goal_min = goals_before[-1]["minute"]
                mins_since_last_goal = entry_min - last_goal_min
            else:
                mins_since_last_goal = entry_min

            # Score level (total goals at entry)
            score_level = h + a

            # Team rolling stats (map losing/winning to home/away)
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
                "score_level": score_level,
                "total_goals_before": total_goals_before,
                "losing_goals_before": losing_goals_before,
                "mins_since_last_goal": mins_since_last_goal,
                # Team rolling stats
                "losing_goals_per_90": losing_goals_per_90,
                "losing_late_pct": losing_late_pct,
                "opp_conceded_per_90": opp_conceded_per_90,
                # Target
                "equalized": int(equalized),
                "equalization_minute": eq_minute,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # One-hot encode league
    for lg in LEAGUES:
        df[f"league_{lg.lower().replace(' ', '_')}"] = (
            df["league"] == lg
        ).astype(int)

    return df


def get_feature_columns(include_momentum: bool = True, include_team: bool = True) -> List[str]:
    """Return the list of feature columns used by the model."""
    cols = [
        "xg_share",
        "xg_diff",
        "total_xg",
        "is_home_losing",
        "score_level",
        "total_goals_before",
        "losing_goals_before",
        "mins_since_last_goal",
        "entry_minute",
    ]
    if include_momentum:
        cols.insert(0, "momentum")

    if include_team:
        cols.extend([
            "losing_goals_per_90",
            "losing_late_pct",
            "opp_conceded_per_90",
        ])

    # League one-hots
    for lg in LEAGUES:
        cols.append(f"league_{lg.lower().replace(' ', '_')}")

    return cols


class EqualizerModel:
    """XGBoost classifier for predicting equalization probability."""

    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.feature_cols = []
        self.trained_at = None
        self.metrics = {}

    def train(self, df: pd.DataFrame, use_momentum: bool = True, include_team: bool = True) -> Dict:
        """Train XGBoost with time-series cross-validation and isotonic calibration.

        Returns dict of evaluation metrics.
        """
        self.feature_cols = get_feature_columns(
            include_momentum=use_momentum, include_team=include_team
        )

        # Sort by date for temporal ordering
        df = df.sort_values("date").reset_index(drop=True)

        # Determine which feature columns are available
        available_cols = [c for c in self.feature_cols if c in df.columns]

        # For rows without momentum (non-70 entry minutes), we have two options:
        # 1. Drop momentum feature entirely
        # 2. Only use rows with momentum
        # We'll train two models: one with momentum (min 70 only), one without
        if use_momentum and "momentum" in available_cols:
            # Filter to rows with momentum data
            df_train = df.dropna(subset=["momentum"]).copy()
            if len(df_train) < 50:
                logger.warning(
                    f"Only {len(df_train)} rows with momentum. "
                    "Falling back to no-momentum model."
                )
                return self.train(df, use_momentum=False)
        else:
            available_cols = [c for c in available_cols if c != "momentum"]
            df_train = df.copy()

        self.feature_cols = available_cols

        # Drop rows with any NaN in features
        df_train = df_train.dropna(subset=available_cols).reset_index(drop=True)

        X = df_train[available_cols].values
        y = df_train["equalized"].values

        print(f"\nTraining data: {len(df_train)} rows, {y.sum()} equalized "
              f"({y.mean():.1%} base rate)")
        print(f"Features ({len(available_cols)}): {available_cols}")

        # Time-series cross-validation
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_probs = np.zeros(len(y))
        cv_preds = np.zeros(len(y))
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Scale pos_weight for class imbalance
            n_pos = y_tr.sum()
            n_neg = len(y_tr) - n_pos
            scale = n_neg / n_pos if n_pos > 0 else 1.0

            clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            probs = clf.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)
            cv_probs[val_idx] = probs
            cv_preds[val_idx] = preds

            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, probs)
                brier = brier_score_loss(y_val, probs)
                fold_metrics.append({"fold": fold, "auc": auc, "brier": brier,
                                     "n": len(y_val), "eq_rate": y_val.mean()})
                print(f"  Fold {fold}: AUC={auc:.3f} Brier={brier:.3f} "
                      f"n={len(y_val)} eq_rate={y_val.mean():.1%}")

        # Overall CV metrics (only on validation folds)
        # First fold has no validation, so skip indices that were never in val
        val_mask = cv_probs > 0  # crude but works since probs are never exactly 0 for actual predictions
        # Better: collect all val indices
        all_val_idx = []
        for _, val_idx in tscv.split(X):
            all_val_idx.extend(val_idx)
        all_val_idx = sorted(set(all_val_idx))

        y_val_all = y[all_val_idx]
        probs_val_all = cv_probs[all_val_idx]

        if len(np.unique(y_val_all)) > 1:
            overall_auc = roc_auc_score(y_val_all, probs_val_all)
            overall_brier = brier_score_loss(y_val_all, probs_val_all)
            overall_logloss = log_loss(y_val_all, probs_val_all)
        else:
            overall_auc = overall_brier = overall_logloss = None

        print(f"\n  CV Overall: AUC={overall_auc:.3f} Brier={overall_brier:.3f} "
              f"LogLoss={overall_logloss:.3f}")

        # Train final model on all data
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale = n_neg / n_pos if n_pos > 0 else 1.0

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X, y, verbose=False)

        # Calibrate with isotonic regression (CV-based)
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method="isotonic", cv=3
        )
        self.calibrated_model.fit(X, y)

        # Calibrated probabilities on training data (for reference)
        cal_probs = self.calibrated_model.predict_proba(X)[:, 1]
        cal_brier = brier_score_loss(y, cal_probs)
        print(f"  Calibrated (in-sample) Brier: {cal_brier:.3f}")

        # Feature importance
        importances = self.model.feature_importances_
        importance_dict = dict(zip(available_cols, importances))
        sorted_imp = sorted(importance_dict.items(), key=lambda x: -x[1])
        print("\n  Feature Importance:")
        for feat, imp in sorted_imp:
            bar = "#" * int(imp * 50)
            print(f"    {feat:30s} {imp:.3f} {bar}")

        self.trained_at = datetime.utcnow().isoformat()
        self.metrics = {
            "cv_auc": overall_auc,
            "cv_brier": overall_brier,
            "cv_logloss": overall_logloss,
            "calibrated_brier": cal_brier,
            "n_train": len(df_train),
            "n_equalized": int(y.sum()),
            "base_rate": float(y.mean()),
            "n_features": len(available_cols),
            "fold_metrics": fold_metrics,
            "feature_importance": importance_dict,
        }

        return self.metrics

    def predict_proba(self, features: Dict) -> float:
        """Predict equalization probability from a feature dict.

        Args:
            features: dict with keys matching self.feature_cols

        Returns:
            Calibrated probability of equalization (0-1).
        """
        if self.calibrated_model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = np.array([[features.get(c, 0) for c in self.feature_cols]])
        return float(self.calibrated_model.predict_proba(X)[0, 1])

    def predict_from_signal(
        self,
        momentum: float,
        xg_home: float,
        xg_away: float,
        league: str,
        losing_team: str,
        score_at_entry: Tuple[int, int],
        entry_minute: int,
        goals: List[Dict],
    ) -> float:
        """Predict from raw match signal data (convenience wrapper).

        This is the interface used by paper_trader.py at runtime.
        """
        h, a = score_at_entry
        losing_xg = xg_home if losing_team == "home" else xg_away
        winning_xg = xg_away if losing_team == "home" else xg_home
        total_xg = (xg_home or 0) + (xg_away or 0)
        xg_share = losing_xg / total_xg if total_xg > 0 else 0.5
        xg_diff = (losing_xg or 0) - (winning_xg or 0)

        goals_before = [g for g in goals if g["minute"] <= entry_minute]
        losing_goals = sum(1 for g in goals_before if g["team"] == losing_team)
        mins_since = (
            entry_minute - goals_before[-1]["minute"]
            if goals_before else entry_minute
        )

        features = {
            "momentum": momentum,
            "xg_share": xg_share,
            "xg_diff": xg_diff,
            "total_xg": total_xg,
            "is_home_losing": 1 if losing_team == "home" else 0,
            "score_level": h + a,
            "total_goals_before": len(goals_before),
            "losing_goals_before": losing_goals,
            "mins_since_last_goal": mins_since,
            "entry_minute": entry_minute,
        }

        # League one-hots
        for lg in LEAGUES:
            key = f"league_{lg.lower().replace(' ', '_')}"
            features[key] = 1 if league == lg else 0

        return self.predict_proba(features)

    def save(self, path: Path = MODEL_FILE):
        """Save trained model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "feature_cols": self.feature_cols,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path = MODEL_FILE) -> "EqualizerModel":
        """Load a trained model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        m = cls()
        m.model = state["model"]
        m.calibrated_model = state["calibrated_model"]
        m.feature_cols = state["feature_cols"]
        m.trained_at = state["trained_at"]
        m.metrics = state["metrics"]
        return m


def backtest_with_model(
    df: pd.DataFrame, model: EqualizerModel,
    entry_minute: int = 70,
    exit_price: float = 0.42,
) -> pd.DataFrame:
    """Run a P&L backtest using model predictions vs draw price.

    For now uses assumed draw prices. When Polymarket price data is available,
    pass actual prices in a 'draw_price' column.

    Returns DataFrame with per-trade results.
    """
    # Filter to the target entry minute
    df_bt = df[df["entry_minute"] == entry_minute].copy()
    df_bt = df_bt.dropna(subset=model.feature_cols).reset_index(drop=True)

    if df_bt.empty:
        print("No data for backtest.")
        return pd.DataFrame()

    # Get model predictions
    X = df_bt[model.feature_cols].values
    df_bt["model_prob"] = model.calibrated_model.predict_proba(X)[:, 1]

    # Simulate at various entry prices if no actual PM price
    if "draw_price" not in df_bt.columns:
        results = []
        for assumed_price in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
            df_bt["draw_price"] = assumed_price
            df_bt["breakeven"] = assumed_price / exit_price
            df_bt["edge"] = df_bt["model_prob"] - df_bt["breakeven"]
            df_bt["would_bet"] = df_bt["edge"] > 0

            bets = df_bt[df_bt["would_bet"]]
            if bets.empty:
                continue

            wins = bets["equalized"].sum()
            losses = len(bets) - wins
            pnl_per_win = exit_price - assumed_price
            pnl_per_loss = -assumed_price
            total_pnl = wins * pnl_per_win + losses * pnl_per_loss
            roi = total_pnl / (assumed_price * len(bets)) * 100

            results.append({
                "entry_price": assumed_price,
                "bets": len(bets),
                "wins": int(wins),
                "losses": int(losses),
                "eq_rate": wins / len(bets),
                "avg_model_prob": bets["model_prob"].mean(),
                "avg_edge": bets["edge"].mean(),
                "total_pnl": round(total_pnl, 2),
                "roi_pct": round(roi, 1),
            })

        results_df = pd.DataFrame(results)
        print(f"\n{'='*70}")
        print(f"MODEL BACKTEST (entry min {entry_minute}, exit ${exit_price})")
        print(f"{'='*70}")
        print(f"Total qualifying: {len(df_bt)}")
        print(f"Base eq rate: {df_bt['equalized'].mean():.1%}")
        print(f"\n{results_df.to_string(index=False)}")

        return results_df
    else:
        # Use actual Polymarket prices
        df_bt["breakeven"] = df_bt["draw_price"] / exit_price
        df_bt["edge"] = df_bt["model_prob"] - df_bt["breakeven"]
        df_bt["would_bet"] = df_bt["edge"] > 0

        bets = df_bt[df_bt["would_bet"]].copy()
        if bets.empty:
            print("No bets would be placed.")
            return pd.DataFrame()

        bets["pnl"] = bets.apply(
            lambda r: (exit_price - r["draw_price"]) if r["equalized"]
                       else -r["draw_price"],
            axis=1,
        )

        print(f"\n{'='*70}")
        print(f"MODEL BACKTEST WITH POLYMARKET PRICES")
        print(f"{'='*70}")
        print(f"Total qualifying: {len(df_bt)}")
        print(f"Bets placed: {len(bets)} (skipped {len(df_bt) - len(bets)} no-edge)")
        print(f"Wins: {bets['equalized'].sum()} / {len(bets)} "
              f"({bets['equalized'].mean():.1%})")
        print(f"Total P&L: ${bets['pnl'].sum():.2f}")
        print(f"Avg edge: {bets['edge'].mean():.1%}")
        print(f"Avg model prob: {bets['model_prob'].mean():.1%}")
        print(f"Avg draw price: ${bets['draw_price'].mean():.3f}")

        return bets


def compare_vs_bucket_lookup(df: pd.DataFrame, model: EqualizerModel):
    """Compare model predictions vs the current bucket-lookup approach."""
    df_70 = df[df["entry_minute"] == 70].dropna(subset=model.feature_cols).copy()
    if df_70.empty:
        return

    X = df_70[model.feature_cols].values
    df_70["model_prob"] = model.calibrated_model.predict_proba(X)[:, 1]

    # Current bucket-lookup probabilities
    BUCKETS = {
        (0.0, 0.3): 0.000, (0.3, 0.4): 0.190, (0.4, 0.5): 0.333,
        (0.5, 0.6): 0.419, (0.6, 0.7): 0.452, (0.7, 1.0): 0.600,
    }
    LEAGUE_ADJ = {
        "Premier League": +0.05, "Bundesliga": +0.03,
        "La Liga": -0.02, "Serie A": -0.02, "Ligue 1": -0.04,
    }

    def bucket_prob(row):
        mom = row.get("momentum")
        if mom is None:
            return None
        for (lo, hi), rate in BUCKETS.items():
            if lo <= mom < hi or (hi == 1.0 and mom == 1.0):
                adj = LEAGUE_ADJ.get(row["league"], 0)
                return max(0.01, min(rate + adj, 0.65))
        return max(0.01, min(BUCKETS[(0.7, 1.0)] + LEAGUE_ADJ.get(row["league"], 0), 0.65))

    df_70["bucket_prob"] = df_70.apply(bucket_prob, axis=1)
    df_70 = df_70.dropna(subset=["bucket_prob"])

    y = df_70["equalized"].values
    model_probs = df_70["model_prob"].values
    bucket_probs = df_70["bucket_prob"].values

    print(f"\n{'='*70}")
    print("MODEL vs BUCKET LOOKUP COMPARISON")
    print(f"{'='*70}")
    print(f"Samples: {len(df_70)}")

    if len(np.unique(y)) > 1:
        print(f"\n{'Metric':<25s} {'XGBoost':>10s} {'Bucket':>10s}")
        print("-" * 47)
        print(f"{'AUC':.<25s} {roc_auc_score(y, model_probs):>10.3f} "
              f"{roc_auc_score(y, bucket_probs):>10.3f}")
        print(f"{'Brier Score':.<25s} {brier_score_loss(y, model_probs):>10.3f} "
              f"{brier_score_loss(y, bucket_probs):>10.3f}")
        print(f"{'Log Loss':.<25s} {log_loss(y, model_probs):>10.3f} "
              f"{log_loss(y, bucket_probs):>10.3f}")


def _get_season(date_str: str) -> str:
    """Map a date string to its season (e.g. '2025-01-15' -> '2024/2025')."""
    if not date_str or len(date_str) < 7:
        return "unknown"
    year = int(date_str[:4])
    month = int(date_str[5:7])
    if month >= 8:
        return f"{year}/{year+1}"
    else:
        return f"{year-1}/{year}"


def season_validation(
    df: pd.DataFrame,
    use_momentum: bool = True,
    include_team: bool = True,
):
    """3-fold season-based validation:
      Fold 1: train 24/25, test 25/26 (forward)
      Fold 2: train 25/26, test 24/25 (retrodiction)
      Fold 3: leave-one-season-out (average of both)

    Reports AUC, Brier, and classification metrics per fold.
    """
    df = df.copy()
    df["season"] = df["date"].apply(_get_season)
    seasons = sorted(df["season"].unique())

    print(f"\n{'='*70}")
    print("SEASON-BASED VALIDATION (3 folds)")
    print(f"{'='*70}")
    print(f"Seasons found: {seasons}")
    for s in seasons:
        n = len(df[df["season"] == s])
        eq = df[df["season"] == s]["equalized"].sum()
        print(f"  {s}: {n} rows, {eq} equalized ({eq/n*100:.1f}%)")

    feature_cols = get_feature_columns(
        include_momentum=use_momentum, include_team=include_team
    )

    # Filter to rows with all features available
    if use_momentum:
        df_valid = df.dropna(subset=["momentum"]).copy()
    else:
        feature_cols = [c for c in feature_cols if c != "momentum"]
        df_valid = df.copy()

    if include_team:
        df_valid = df_valid.dropna(
            subset=["losing_goals_per_90", "losing_late_pct", "opp_conceded_per_90"]
        )

    df_valid = df_valid.dropna(subset=[c for c in feature_cols if c in df_valid.columns])

    # Ensure we have at least 2 seasons
    valid_seasons = sorted(df_valid["season"].unique())
    if len(valid_seasons) < 2:
        print(f"Need at least 2 seasons, got {valid_seasons}. Skipping.")
        return

    s1, s2 = valid_seasons[0], valid_seasons[-1]  # earliest and latest
    df_s1 = df_valid[df_valid["season"] == s1]
    df_s2 = df_valid[df_valid["season"] == s2]

    folds = [
        (f"Train {s1} -> Test {s2} (forward)", df_s1, df_s2),
        (f"Train {s2} -> Test {s1} (retrodiction)", df_s2, df_s1),
    ]

    results = []

    for fold_name, df_train, df_test in folds:
        X_train = df_train[feature_cols].values
        y_train = df_train["equalized"].values
        X_test = df_test[feature_cols].values
        y_test = df_test["equalized"].values

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale = n_neg / n_pos if n_pos > 0 else 1.0

        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        clf.fit(X_train, y_train, verbose=False)

        # Calibrate
        cal_clf = CalibratedClassifierCV(clf, method="isotonic", cv=3)
        cal_clf.fit(X_train, y_train)

        probs = cal_clf.predict_proba(X_test)[:, 1]

        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, probs)
            brier = brier_score_loss(y_test, probs)
            ll = log_loss(y_test, probs)
        else:
            auc = brier = ll = None

        results.append({
            "fold": fold_name,
            "train_n": len(y_train),
            "test_n": len(y_test),
            "train_eq_rate": y_train.mean(),
            "test_eq_rate": y_test.mean(),
            "auc": auc,
            "brier": brier,
            "logloss": ll,
        })

        print(f"\n  {fold_name}")
        print(f"    Train: {len(y_train)} rows ({y_train.mean():.1%} eq)")
        print(f"    Test:  {len(y_test)} rows ({y_test.mean():.1%} eq)")
        if auc is not None:
            print(f"    AUC: {auc:.3f}  Brier: {brier:.3f}  LogLoss: {ll:.3f}")

    # Fold 3: leave-one-season-out average
    avg_auc = np.mean([r["auc"] for r in results if r["auc"] is not None])
    avg_brier = np.mean([r["brier"] for r in results if r["brier"] is not None])
    avg_ll = np.mean([r["logloss"] for r in results if r["logloss"] is not None])

    print(f"\n  Leave-one-season-out average:")
    print(f"    AUC: {avg_auc:.3f}  Brier: {avg_brier:.3f}  LogLoss: {avg_ll:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Equalizer XGBoost Model")
    parser.add_argument("--save", action="store_true", help="Save model after training")
    parser.add_argument(
        "--no-momentum", action="store_true",
        help="Train without momentum (uses all entry minutes)",
    )
    parser.add_argument(
        "--no-team", action="store_true",
        help="Train without team rolling stats",
    )
    parser.add_argument(
        "--entry-minute", type=int, default=70,
        help="Entry minute for backtest (default: 70)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Loading match data...")
    matches = load_matches()
    print(f"Loaded {len(matches)} matches")

    print("\nBuilding features...")
    df = build_features(matches)
    print(f"Built {len(df)} training rows from {df['match_id'].nunique()} matches")
    print(f"Equalization rate: {df['equalized'].mean():.1%}")
    print(f"Entry minutes: {sorted(df['entry_minute'].unique())}")

    # Show team stats coverage
    has_team = df["losing_goals_per_90"].notna().sum()
    print(f"Rows with team rolling stats: {has_team}/{len(df)} ({has_team/len(df)*100:.0f}%)")

    use_momentum = not args.no_momentum
    include_team = not args.no_team

    # Season-based validation (the main evaluation)
    season_validation(df, use_momentum=use_momentum, include_team=include_team)

    # Also run without team stats for comparison
    if include_team:
        print("\n\n--- Comparison: WITHOUT team stats ---")
        season_validation(df, use_momentum=use_momentum, include_team=False)

    # Train final model on all data
    model = EqualizerModel()
    print(f"\n\n{'='*70}")
    print("FINAL MODEL (trained on all data)")
    print(f"{'='*70}")

    if use_momentum:
        print("Training WITH momentum (min 70 only)")
    else:
        print("Training WITHOUT momentum (all entry minutes)")

    model.train(df, use_momentum=use_momentum, include_team=include_team)

    # Compare vs bucket lookup
    if use_momentum:
        compare_vs_bucket_lookup(df, model)

    # Backtest
    backtest_with_model(df, model, entry_minute=args.entry_minute)

    # Save
    if args.save:
        model.save()

    return model, df


if __name__ == "__main__":
    main()
