"""
Draw Scanner — XGBoost + Survival Model Signal Detection

Replaces the old momentum-only scanner with a two-stage approach:
  Stage 1 (XGBoost): Should we bet on this match? (P(draw) > PM price → +EV)
  Stage 2 (Survival): Should we enter now or wait? (hazard curve vs PM price trajectory)

Scans live matches every minute from min 55-75. For each qualifying match
(trailing by 1 goal), evaluates whether the draw bet has positive expected value
and whether now is the optimal entry time.

Usage:
    from src.Soccer.draw_scanner import DrawScanner

    scanner = DrawScanner(fotmob_provider, polymarket_provider)
    signals = scanner.scan()
"""

import logging
import pickle
from typing import Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path("Data/soccer_models")
XGB_MODEL_FILE = MODEL_DIR / "draw_xgb.pkl"
COX_MODEL_FILE = MODEL_DIR / "cox_model.pkl"

# ===================== CONFIGURATION =====================
# Wider window than old scanner — survival model picks optimal entry
MIN_MINUTE = 55
MAX_MINUTE = 75
GOAL_DIFFERENCE = 1
SCAN_INTERVAL = 30  # Every 30 seconds — need to catch the optimal entry minute

# XGBoost edge threshold: model P(draw) must exceed PM price by this much
MIN_EDGE = 0.05

# PM draw price bounds (relaxed upper for NO trades)
MIN_DRAW_PRICE = 0.05
MAX_DRAW_PRICE = 0.95

# Cooldown per match (don't re-signal within N minutes)
COOLDOWN_MINUTES = 10

# Survival: minimum near-term hazard to enter now vs wait
# If P(equalize in next 5 min) is low, waiting gives cheaper PM price
MIN_NEAR_HAZARD = 0.02

LEAGUES = [
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"
]
# ===========================================================


@dataclass
class DrawSignal:
    """A detected draw bet signal from the combined model."""
    match_id: int
    league: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    minute: int
    losing_team: str
    # Model outputs
    xgb_draw_prob: float
    pm_draw_price: float
    edge: float
    bet_side: str               # "YES" or "NO"
    # Survival outputs
    p_equalize_10min: float
    p_equalize_20min: float
    near_hazard_5min: float
    optimal_entry: bool         # True if survival says enter now
    # Market info
    draw_token_id: Optional[str]
    polymarket_event_id: Optional[str]
    # Raw features for logging
    momentum_value: Optional[float]
    xg_home: Optional[float]
    xg_away: Optional[float]
    score_level: int
    losing_goals_before: int
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


class DrawScanner:
    """Scans live matches for draw bet signals using XGBoost + survival model."""

    def __init__(
        self,
        fotmob_provider,
        polymarket_provider,
        ws_monitor=None,
        min_edge: float = MIN_EDGE,
        min_minute: int = MIN_MINUTE,
        max_minute: int = MAX_MINUTE,
        logger_override=None,
    ):
        self.fotmob = fotmob_provider
        self.polymarket = polymarket_provider
        self.ws_monitor = ws_monitor  # WebSocket price monitor (optional)
        self.min_edge = min_edge
        self.min_minute = min_minute
        self.max_minute = max_minute
        self.log = logger_override or logger

        # Load XGBoost model
        self.xgb_model = None
        self._load_xgb_model()

        # Load survival model (Cox PH)
        self.cox_model = None
        self._load_cox_model()

        # Cache of draw token IDs for WS subscription: {match_key: token_id}
        self._draw_tokens: Dict[str, str] = {}

        # Cooldowns
        self._cooldowns: Dict[int, datetime] = {}
        self.near_misses: List[Dict] = []

    def _load_xgb_model(self):
        """Load the trained XGBoost draw model."""
        try:
            with open(XGB_MODEL_FILE, "rb") as f:
                state = pickle.load(f)
            self._xgb_calibrated = state["calibrated"]
            self._xgb_feature_cols = state["feature_cols"]
            self.xgb_model = True  # flag that model is loaded
            self.log.info(
                f"XGBoost draw model loaded ({len(self._xgb_feature_cols)} features)"
            )
        except Exception as e:
            self.log.error(f"Failed to load XGBoost model: {e}")
            self._xgb_calibrated = None
            self._xgb_feature_cols = []

    def _load_cox_model(self):
        """Load the trained Cox PH survival model."""
        if COX_MODEL_FILE.exists():
            try:
                with open(COX_MODEL_FILE, "rb") as f:
                    self.cox_model = pickle.load(f)
                self.log.info("Cox survival model loaded")
            except Exception as e:
                self.log.warning(f"Failed to load Cox model: {e}")
        else:
            self.log.info("No Cox model found — survival timing disabled, will use XGBoost only")

    def scan(self) -> List[DrawSignal]:
        """Run one scan cycle across all live matches."""
        signals = []

        if not self.xgb_model:
            self.log.warning("No XGBoost model loaded, skipping scan")
            return signals

        live_matches = self.fotmob.get_live_matches()
        if not live_matches:
            self.log.info("No live matches in tracked leagues")
            return signals

        self.log.info(f"Scanning {len(live_matches)} live matches...")

        # Fetch PM markets via REST (for token IDs and fallback prices)
        polymarket_odds = self.polymarket.get_draw_odds()
        self.log.info(f"Found {len(polymarket_odds)} Polymarket draw markets")

        # Subscribe to draw tokens via WebSocket for real-time prices
        if self.ws_monitor:
            new_tokens = []
            for key, data in polymarket_odds.items():
                token = data.get("draw_yes_token_id")
                if token and token not in self._draw_tokens.values():
                    self._draw_tokens[key] = token
                    new_tokens.append(token)
            if new_tokens:
                self.ws_monitor.subscribe(new_tokens)
                self.log.info(f"WS subscribed to {len(new_tokens)} new draw tokens")

        for match in live_matches:
            signal = self._evaluate_match(match, polymarket_odds)
            if signal:
                signals.append(signal)

        if signals:
            self.log.info(f"SIGNALS DETECTED: {len(signals)}")
        else:
            self.log.info("No signals this cycle")

        return signals

    def _evaluate_match(
        self, match: Dict, polymarket_odds: Dict
    ) -> Optional[DrawSignal]:
        """Evaluate a single match through XGBoost + survival pipeline."""
        match_id = match["match_id"]
        minute = match.get("minute", 0)
        home_score = match.get("home_score", 0)
        away_score = match.get("away_score", 0)
        home_team = match.get("home_team", "")
        away_team = match.get("away_team", "")
        league = match.get("league", "")

        # Basic filters
        if minute < self.min_minute or minute > self.max_minute:
            return None
        if abs(home_score - away_score) != GOAL_DIFFERENCE:
            return None
        if self._is_on_cooldown(match_id):
            return None

        losing_team = "home" if home_score < away_score else "away"

        self.log.info(
            f"QUALIFYING: [{league}] {home_team} {home_score}-{away_score} "
            f"{away_team} (min {minute})"
        )

        # Fetch match details for momentum + xG
        details = self.fotmob.get_match_details(match_id)
        if not details:
            return None
        if details.get("is_extra_time"):
            return None

        # Extract features for XGBoost
        xg = details.get("xg", {})
        xg_home = xg.get("home")
        xg_away = xg.get("away")

        # Momentum
        momentum_result = self.fotmob.get_momentum_value(details)
        momentum_value = momentum_result["momentum_value"] if momentum_result else None

        # Build feature dict for XGBoost
        # We need: goals list to compute losing_goals_before, etc.
        # Approximate from current score since we don't have full goal timeline live
        if losing_team == "home":
            losing_goals = home_score
            winning_goals = away_score
            losing_xg = xg_home or 0
            winning_xg = xg_away or 0
        else:
            losing_goals = away_score
            winning_goals = home_score
            losing_xg = xg_away or 0
            winning_xg = xg_home or 0

        total_xg = losing_xg + winning_xg
        xg_share = losing_xg / total_xg if total_xg > 0 else 0.5
        xg_diff = losing_xg - winning_xg
        score_level = home_score + away_score
        total_goals = score_level
        # Approximate mins_since_last_goal (we don't have exact goal times live)
        mins_since_last_goal = 10  # rough default

        features = {
            "xg_share": xg_share,
            "xg_diff": xg_diff,
            "total_xg": total_xg,
            "is_home_losing": 1 if losing_team == "home" else 0,
            "score_level": score_level,
            "total_goals_before": total_goals,
            "losing_goals_before": losing_goals,
            "mins_since_last_goal": mins_since_last_goal,
            "entry_minute": minute,
        }

        # Add momentum if available
        if momentum_value is not None:
            features["momentum"] = momentum_value

        # Team rolling stats (not available live without pre-computation)
        # Set to None — model handles NaN via XGBoost's native support
        features["losing_goals_per_90"] = None
        features["losing_late_pct"] = None
        features["opp_conceded_per_90"] = None

        # League one-hots
        for lg in LEAGUES:
            key = f"league_{lg.lower().replace(' ', '_')}"
            features[key] = 1 if league == lg else 0

        # Card / substitution features from /incidents (if the loaded
        # model wasn't trained with these, the columns are simply not
        # in self._xgb_feature_cols and get ignored at predict time).
        try:
            from src.Soccer.feature_builder import (
                _card_features_at, _sub_features_at,
            )
            features.update(
                _card_features_at(details.get("cards"), minute, losing_team)
            )
            features.update(
                _sub_features_at(details.get("substitutions"), minute, losing_team)
            )
        except Exception as e:
            self.log.debug(f"Card/sub feature derivation skipped: {e}")

        # Stage 1: XGBoost prediction
        try:
            X_pred = np.array([[features.get(c, 0) if features.get(c) is not None else 0
                                for c in self._xgb_feature_cols]])
            draw_prob = float(self._xgb_calibrated.predict_proba(X_pred)[0, 1])
        except Exception as e:
            self.log.warning(f"XGBoost prediction failed: {e}")
            return None

        # Look up PM draw price
        market_key = self.polymarket.match_fotmob_to_polymarket(
            home_team, away_team, polymarket_odds
        )
        if not market_key:
            self._log_near_miss(match, "no_pm_market", draw_prob, momentum_value)
            return None

        market_data = polymarket_odds[market_key]
        draw_token_id = market_data["draw_yes_token_id"]

        # Prefer WebSocket price (real-time) over REST (stale)
        pm_draw_price = None
        price_source = "rest"
        if self.ws_monitor and draw_token_id:
            ws_price = self.ws_monitor.get_mid_price(draw_token_id)
            if ws_price and ws_price > 0:
                pm_draw_price = ws_price
                price_source = "ws"

        if pm_draw_price is None:
            pm_draw_price = market_data["draw_yes_price"]
            price_source = "rest"

        if pm_draw_price < MIN_DRAW_PRICE or pm_draw_price > MAX_DRAW_PRICE:
            self._log_near_miss(match, "pm_price_out_of_range", draw_prob, momentum_value)
            return None

        # Edge calculation (hold to expiry)
        edge = draw_prob - pm_draw_price

        self.log.info(
            f"  XGBoost P(draw)={draw_prob:.3f} | PM price={pm_draw_price:.3f} ({price_source}) | "
            f"Edge={edge:+.3f} | Score {score_level} | LG_before={losing_goals}"
        )

        # Determine bet side: YES if draw underpriced, NO if overpriced
        if edge >= self.min_edge:
            bet_side = "YES"
        elif edge <= -self.min_edge:
            bet_side = "NO"
        else:
            self._log_near_miss(match, "insufficient_edge", draw_prob, momentum_value)
            self.log.info(f"  SKIP: |edge| {abs(edge):.3f} < {self.min_edge}")
            return None

        # Stage 2: Survival model (timing)
        p_eq_10 = 0.0
        p_eq_20 = 0.0
        near_hazard = 0.0
        optimal_entry = True  # default: enter now if no Cox model

        if self.cox_model:
            try:
                from src.Soccer.survival_model import get_survival_curve, COVARIATES
                import pandas as pd

                surv_features = {k: features.get(k, 0) for k in COVARIATES}
                surv_features["minutes_remaining"] = 90 - minute

                times = np.arange(1, 31)
                _, cum_prob = get_survival_curve(self.cox_model, surv_features, times)

                p_eq_10 = cum_prob[9] if len(cum_prob) > 9 else 0
                p_eq_20 = cum_prob[19] if len(cum_prob) > 19 else 0
                near_hazard = cum_prob[4] if len(cum_prob) > 4 else 0

                # YES: enter when hazard high (equalization likely → price about to spike)
                # NO:  enter when hazard low (equalization unlikely → safe to sell draw)
                if bet_side == "YES":
                    optimal_entry = near_hazard >= MIN_NEAR_HAZARD
                else:
                    optimal_entry = True  # NO doesn't need hazard gating

                self.log.info(
                    f"  Survival: P(eq+10)={p_eq_10:.1%} P(eq+20)={p_eq_20:.1%} "
                    f"hazard_5min={near_hazard:.1%} → {'ENTER NOW' if optimal_entry else 'WAIT'}"
                    f" ({bet_side} side)"
                )
            except Exception as e:
                self.log.warning(f"Survival model error: {e}")

        if not optimal_entry:
            self._log_near_miss(match, "survival_says_wait", draw_prob, momentum_value)
            return None

        # ALL CRITERIA MET
        self._set_cooldown(match_id)

        signal = DrawSignal(
            match_id=match_id,
            league=league,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            minute=minute,
            losing_team=losing_team,
            xgb_draw_prob=draw_prob,
            pm_draw_price=pm_draw_price,
            edge=edge,
            bet_side=bet_side,
            p_equalize_10min=p_eq_10,
            p_equalize_20min=p_eq_20,
            near_hazard_5min=near_hazard,
            optimal_entry=optimal_entry,
            draw_token_id=market_data["draw_yes_token_id"],
            polymarket_event_id=market_data.get("event_id"),
            momentum_value=momentum_value,
            xg_home=xg_home,
            xg_away=xg_away,
            score_level=score_level,
            losing_goals_before=losing_goals,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if bet_side == "YES":
            self.log.info(
                f"  SIGNAL: Buy YES (Draw) @ {pm_draw_price:.1%} | "
                f"P(draw)={draw_prob:.3f} | Edge={edge:+.3f} | "
                f"Score={score_level} LG={losing_goals}"
            )
        else:
            self.log.info(
                f"  SIGNAL: Buy NO (No Draw) @ {1 - pm_draw_price:.1%} | "
                f"P(draw)={draw_prob:.3f} | Edge={abs(edge):+.3f} | "
                f"Score={score_level} LG={losing_goals}"
            )

        return signal

    def _is_on_cooldown(self, match_id: int) -> bool:
        if match_id not in self._cooldowns:
            return False
        elapsed = (datetime.now(timezone.utc) - self._cooldowns[match_id]).total_seconds()
        return elapsed < COOLDOWN_MINUTES * 60

    def _set_cooldown(self, match_id: int):
        self._cooldowns[match_id] = datetime.now(timezone.utc)

    def _log_near_miss(self, match, reason, draw_prob, momentum):
        self.near_misses.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "match_id": match["match_id"],
            "league": match.get("league", ""),
            "home_team": match.get("home_team", ""),
            "away_team": match.get("away_team", ""),
            "score": f"{match.get('home_score', 0)}-{match.get('away_score', 0)}",
            "minute": match.get("minute", 0),
            "reason": reason,
            "draw_prob": draw_prob,
            "momentum": momentum,
        })

    def get_near_misses(self) -> List[Dict]:
        return list(self.near_misses)
