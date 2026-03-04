"""
Momentum Scanner - Core Signal Detection

Scans live soccer matches for the momentum draw betting signal:
  1. Match is in minute 70-75
  2. One team is losing by exactly 1 goal
  3. The losing team shows momentum above threshold
  4. A Polymarket draw market exists for this match
  5. Draw price is within acceptable range

Usage:
    from src.Soccer.momentum_scanner import MomentumScanner

    scanner = MomentumScanner(fotmob_provider, polymarket_provider)
    signals = scanner.scan()
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# ===================== CONFIGURABLE THRESHOLDS =====================
# Calibrated from backtest: 874 matches, 162 qualifying, Aug 2025 - Feb 2026
# See Data/soccer_backtest/results.json for full analysis
#
# Backtest equalization rates by momentum bucket:
#   0.0-0.3:  0.0%  |  0.3-0.4: 19.0%  |  0.4-0.5: 33.3%
#   0.5-0.6: 41.9%  |  0.6-0.7: 45.2%  |  0.7-1.0: 60.0%
#
# Momentum >= 0.50 → 46% equalization, profitable at entry ≤ $0.18
# Momentum >= 0.65 → 51% equalization, but only 57 qualifying matches

MOMENTUM_THRESHOLD = 0.50       # Min momentum to trigger (backtest: 46% eq rate at 0.50+)
MIN_MINUTE = 70                 # Start scanning from minute 70
MAX_MINUTE = 75                 # Stop scanning after minute 75
GOAL_DIFFERENCE = 1             # Only trigger on exactly 1 goal difference
SCAN_INTERVAL = 60              # Seconds between scans

LEAGUES = [
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"
]

# Safety bounds on draw price
MIN_DRAW_PRICE = 0.05           # Don't buy shares cheaper than 5c (near impossible)
MAX_DRAW_PRICE = 0.45           # Upper bound — Kelly sizing handles edge at higher prices

# Cooldown: don't re-signal the same match within N minutes
COOLDOWN_MINUTES = 5
# ==================================================================


@dataclass
class MomentumSignal:
    """A detected momentum signal for a draw bet."""
    match_id: int
    league: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    minute: int
    losing_team: str              # "home" or "away"
    momentum_value: float
    momentum_source: str          # "fotmob_native" or "xg_derived"
    draw_price: Optional[float]
    draw_token_id: Optional[str]
    polymarket_event_id: Optional[str]
    timestamp: str
    xg_home: Optional[float]
    xg_away: Optional[float]

    def to_dict(self) -> Dict:
        return asdict(self)


class MomentumScanner:
    """Scans live matches for momentum draw signals."""

    def __init__(
        self,
        fotmob_provider,
        polymarket_provider,
        momentum_threshold: float = MOMENTUM_THRESHOLD,
        min_minute: int = MIN_MINUTE,
        max_minute: int = MAX_MINUTE,
        goal_difference: int = GOAL_DIFFERENCE,
        logger_override: logging.Logger = None,
    ):
        self.fotmob = fotmob_provider
        self.polymarket = polymarket_provider
        self.momentum_threshold = momentum_threshold
        self.min_minute = min_minute
        self.max_minute = max_minute
        self.goal_difference = goal_difference
        self.log = logger_override or logger

        # Cooldown tracking: {match_id: last_signal_time}
        self._cooldowns: Dict[int, datetime] = {}

        # Near-miss log for later analysis
        self.near_misses: List[Dict] = []

    def scan(self) -> List[MomentumSignal]:
        """Run one scan cycle across all live matches.

        Returns list of MomentumSignal objects for matches where all
        criteria are met.
        """
        signals = []

        # 1. Fetch live matches
        live_matches = self.fotmob.get_live_matches()
        if not live_matches:
            self.log.info("No live matches in tracked leagues")
            return signals

        self.log.info(f"Scanning {len(live_matches)} live matches...")

        # 2. Pre-fetch Polymarket draw odds (one call for all leagues)
        polymarket_odds = self.polymarket.get_draw_odds()
        self.log.info(f"Found {len(polymarket_odds)} Polymarket draw markets")

        # 3. Check each live match
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
    ) -> Optional[MomentumSignal]:
        """Evaluate a single live match for signal criteria.

        Returns MomentumSignal if all criteria met, None otherwise.
        """
        match_id = match["match_id"]
        minute = match.get("minute", 0)
        home_score = match.get("home_score", 0)
        away_score = match.get("away_score", 0)
        home_team = match.get("home_team", "")
        away_team = match.get("away_team", "")
        league = match.get("league", "")

        # Check basic criteria: minute range
        if minute < self.min_minute or minute > self.max_minute:
            return None

        # Check score: exactly 1 goal difference
        goal_diff = abs(home_score - away_score)
        if goal_diff != self.goal_difference:
            return None

        # Check cooldown
        if self._is_on_cooldown(match_id):
            self.log.debug(f"Match {match_id} on cooldown, skipping")
            return None

        self.log.info(
            f"QUALIFYING: [{league}] {home_team} {home_score}-{away_score} "
            f"{away_team} (min {minute})"
        )

        # Fetch detailed match data for momentum
        details = self.fotmob.get_match_details(match_id)
        if not details:
            self.log.warning(f"Could not fetch details for match {match_id}")
            return None

        # Check extra time (skip cup matches in ET)
        if details.get("is_extra_time"):
            self.log.info(f"Match {match_id} in extra time, skipping")
            return None

        # Check momentum
        momentum_result = self.fotmob.get_momentum_value(details)
        if not momentum_result:
            self._log_near_miss(match, "no_momentum_data", 0.0)
            self.log.warning(
                f"Could not determine momentum for {home_team} vs {away_team}"
            )
            return None

        momentum_value = momentum_result["momentum_value"]
        momentum_source = momentum_result["source"]
        losing_team = momentum_result["losing_team"]

        self.log.info(
            f"  Momentum: {momentum_value:.3f} ({momentum_source}) "
            f"for {losing_team} team [threshold: {self.momentum_threshold}]"
        )

        # Check momentum threshold
        if momentum_value < self.momentum_threshold:
            self._log_near_miss(match, "below_threshold", momentum_value)
            self.log.info(
                f"  NEAR MISS: momentum {momentum_value:.3f} < "
                f"threshold {self.momentum_threshold}"
            )
            return None

        # Look up Polymarket draw market
        market_key = self.polymarket.match_fotmob_to_polymarket(
            home_team, away_team, polymarket_odds
        )
        if not market_key:
            self._log_near_miss(match, "no_polymarket_market", momentum_value)
            self.log.warning(
                f"  No Polymarket draw market found for {home_team} vs {away_team}"
            )
            return None

        market_data = polymarket_odds[market_key]
        draw_price = market_data["draw_yes_price"]

        # Check draw price bounds
        if draw_price < MIN_DRAW_PRICE or draw_price > MAX_DRAW_PRICE:
            self._log_near_miss(match, "draw_price_out_of_range", momentum_value)
            self.log.info(
                f"  Draw price {draw_price:.1%} outside range "
                f"[{MIN_DRAW_PRICE:.1%}, {MAX_DRAW_PRICE:.1%}]"
            )
            return None

        # ALL CRITERIA MET - create signal
        self._set_cooldown(match_id)
        xg = details.get("xg", {})

        signal = MomentumSignal(
            match_id=match_id,
            league=league,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            minute=minute,
            losing_team=losing_team,
            momentum_value=momentum_value,
            momentum_source=momentum_source,
            draw_price=draw_price,
            draw_token_id=market_data["draw_yes_token_id"],
            polymarket_event_id=market_data.get("event_id"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            xg_home=xg.get("home"),
            xg_away=xg.get("away"),
        )

        self.log.info(
            f"  SIGNAL: Buy Draw @ {draw_price:.1%} | "
            f"Momentum {momentum_value:.3f} | {losing_team} team attacking"
        )

        return signal

    def _is_on_cooldown(self, match_id: int) -> bool:
        """Check if a match is on cooldown from recent signal."""
        if match_id not in self._cooldowns:
            return False
        elapsed = (datetime.now(timezone.utc) - self._cooldowns[match_id]).total_seconds()
        return elapsed < COOLDOWN_MINUTES * 60

    def _set_cooldown(self, match_id: int):
        """Set cooldown for a match after signal."""
        self._cooldowns[match_id] = datetime.now(timezone.utc)

    def _log_near_miss(self, match: Dict, reason: str, momentum_value: float):
        """Log a near-miss for later analysis of threshold tuning."""
        self.near_misses.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "match_id": match["match_id"],
            "league": match.get("league", ""),
            "home_team": match.get("home_team", ""),
            "away_team": match.get("away_team", ""),
            "score": f"{match.get('home_score', 0)}-{match.get('away_score', 0)}",
            "minute": match.get("minute", 0),
            "reason": reason,
            "momentum_value": momentum_value,
        })

    def get_near_misses(self) -> List[Dict]:
        """Get all logged near-misses for analysis."""
        return list(self.near_misses)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from src.DataProviders.FotMobProvider import FotMobProvider
    from src.DataProviders.PolymarketSoccerProvider import PolymarketSoccerProvider

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("SOCCER MOMENTUM SCANNER")
    print(f"Minute window: {MIN_MINUTE}-{MAX_MINUTE}")
    print(f"Momentum threshold: {MOMENTUM_THRESHOLD}")
    print(f"Draw price range: {MIN_DRAW_PRICE:.0%} - {MAX_DRAW_PRICE:.0%}")
    print("=" * 60)

    fotmob = FotMobProvider()
    polymarket = PolymarketSoccerProvider()
    scanner = MomentumScanner(fotmob, polymarket)

    signals = scanner.scan()

    if signals:
        print(f"\n{'='*60}")
        print(f"SIGNALS FOUND: {len(signals)}")
        print(f"{'='*60}")
        for s in signals:
            print(f"\n  [{s.league}] {s.home_team} {s.home_score}-{s.away_score} {s.away_team}")
            print(f"  Minute: {s.minute} | Losing: {s.losing_team}")
            print(f"  Momentum: {s.momentum_value:.3f} ({s.momentum_source})")
            print(f"  Draw price: {s.draw_price:.1%}")
            print(f"  xG: Home {s.xg_home} - Away {s.xg_away}")
    else:
        print("\nNo signals detected.")

    near = scanner.get_near_misses()
    if near:
        print(f"\n--- Near Misses ({len(near)}) ---")
        for nm in near:
            print(
                f"  [{nm['league']}] {nm['home_team']} {nm['score']} "
                f"{nm['away_team']} (min {nm['minute']}) - "
                f"reason: {nm['reason']}, momentum: {nm['momentum_value']:.3f}"
            )
