"""
Soccer Paper Trader

Simulates draw betting on Polymarket based on momentum signals.
No real money is placed — positions are tracked in JSON files
to validate the strategy before going live.

Usage:
    from src.Soccer.paper_trader import open_position, resolve_position, show_status

    # Open position from a momentum signal
    position_id = open_position(signal)

    # Resolve when match ends
    pnl = resolve_position(position_id, was_draw=True)

CLI:
    python -m src.Soccer.paper_trader --status
    python -m src.Soccer.paper_trader --dashboard
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List

from src.Soccer.momentum_scanner import MomentumSignal, MOMENTUM_THRESHOLD
from src.Utils.Kelly_Criterion import calculate_kelly_criterion
from src.Utils.DrawdownManager import DrawdownManager
from src.Utils.AlertManager import AlertManager, AlertType

logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================
DATA_DIR = Path("Data/soccer_paper_trading")
POSITIONS_FILE = DATA_DIR / "positions.json"
TRADES_LOG = DATA_DIR / "trades.json"
BANKROLL_FILE = DATA_DIR / "bankroll.json"

STARTING_BANKROLL = 1000.0        # Starting paper bankroll
KELLY_FRACTION = 0.15             # Conservative: 15% fractional Kelly
MAX_BET_PCT = 8.0                 # Max 8% of bankroll per bet
MAX_SIMULTANEOUS_POSITIONS = 4    # Max open positions at once

# Exit strategy: "equalization" = sell when score ties (backtest-validated)
#                "expiry" = hold until match end (draw = $1, no draw = $0)
EXIT_STRATEGY = "equalization"

# Exit price when selling on equalization (post-equalizer draw price)
# Draw typically trades at ~$0.40-0.45 right after an equalizer in min 70-85
EXIT_PRICE_ON_EQUALIZATION = 0.42

# Backtest-calibrated equalization rates by momentum bucket
# Source: 874 matches, Top 5 leagues, Aug 2025 - Feb 2026
# See Data/soccer_backtest/results.json
BACKTEST_EQUALIZATION_RATES = {
    (0.0, 0.3): 0.000,   # 0/5 matches
    (0.3, 0.4): 0.190,   # 4/21 matches
    (0.4, 0.5): 0.333,   # 12/36 matches
    (0.5, 0.6): 0.419,   # 18/43 matches
    (0.6, 0.7): 0.452,   # 19/42 matches
    (0.7, 1.0): 0.600,   # 9/15 matches
}

# League-specific adjustments (relative to base)
# PL and Bundesliga have higher equalization rates
LEAGUE_ADJUSTMENTS = {
    "Premier League": +0.05,   # 50% eq rate (base ~38%)
    "Bundesliga":     +0.03,   # 45% eq rate
    "La Liga":        -0.02,   # 32% eq rate
    "Serie A":        -0.02,   # 33% eq rate
    "Ligue 1":        -0.04,   # 29% eq rate
}

# Discord webhook (optional, set to enable)
DISCORD_WEBHOOK_URL = None
# =========================================================


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_bankroll() -> float:
    """Load current bankroll from file."""
    ensure_data_dir()
    if BANKROLL_FILE.exists():
        try:
            with open(BANKROLL_FILE, "r") as f:
                data = json.load(f)
                return float(data.get("bankroll", STARTING_BANKROLL))
        except (json.JSONDecodeError, IOError):
            pass
    return STARTING_BANKROLL


def save_bankroll(bankroll: float):
    """Save current bankroll to file."""
    ensure_data_dir()
    with open(BANKROLL_FILE, "w") as f:
        json.dump({
            "bankroll": round(bankroll, 2),
            "updated": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


def load_positions() -> Dict:
    """Load positions from file."""
    ensure_data_dir()
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_positions(positions: Dict):
    """Save positions to file."""
    ensure_data_dir()
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def log_trade(trade: Dict):
    """Append a trade entry to the trades log."""
    ensure_data_dir()
    trades = []
    if TRADES_LOG.exists():
        try:
            with open(TRADES_LOG, "r") as f:
                trades = json.load(f)
        except (json.JSONDecodeError, IOError):
            trades = []

    trades.append(trade)
    with open(TRADES_LOG, "w") as f:
        json.dump(trades, f, indent=2, default=str)


def _probability_to_american_odds(prob: float) -> int:
    """Convert probability to American odds for Kelly calculation."""
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    else:
        return int(round(100 * (1 - prob) / prob))


def _get_equalization_rate(momentum: float) -> float:
    """Look up backtest equalization rate for a momentum value."""
    for (low, high), rate in BACKTEST_EQUALIZATION_RATES.items():
        if low <= momentum < high:
            return rate
    # Above 0.7 or exact 1.0
    return BACKTEST_EQUALIZATION_RATES[(0.7, 1.0)]


def estimate_draw_probability(signal: MomentumSignal) -> float:
    """Estimate equalization probability using backtest-calibrated rates.

    Instead of a hypothetical edge formula, we use actual equalization rates
    from 874 backtested matches across Top 5 European leagues.

    For equalization strategy: probability = chance the losing team ties it up.
    For expiry strategy: probability = chance match ends as a draw (lower).

    Adjusted by league-specific factors (PL/Bundesliga higher, Ligue 1 lower).
    """
    if signal.draw_price is None:
        return 0.0

    # Base rate from backtest lookup
    base_rate = _get_equalization_rate(signal.momentum_value)

    # League adjustment
    league_adj = LEAGUE_ADJUSTMENTS.get(signal.league, 0.0)
    estimated = base_rate + league_adj

    if EXIT_STRATEGY == "expiry":
        # For hold-to-expiry, equalization doesn't mean final draw.
        # Roughly 60-70% of equalizations hold as draws (from backtest observation).
        # Apply a discount factor.
        estimated *= 0.65

    # Clamp to reasonable bounds
    return max(0.0, min(estimated, 0.65))


def open_position(signal: MomentumSignal) -> Optional[str]:
    """Open a paper trading position from a momentum signal.

    Returns position_id if opened, None if skipped.
    """
    positions = load_positions()
    bankroll = load_bankroll()

    # Check position limits
    open_count = sum(1 for p in positions.values() if p["status"] == "open")
    if open_count >= MAX_SIMULTANEOUS_POSITIONS:
        logger.info(f"Max positions reached ({MAX_SIMULTANEOUS_POSITIONS}), skipping")
        return None

    # Check if we already have a position on this match
    for p in positions.values():
        if p["match_id"] == signal.match_id and p["status"] == "open":
            logger.info(f"Already have position on match {signal.match_id}, skipping")
            return None

    # Check drawdown limits
    dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
    dm.sync_bankroll(bankroll)
    if not dm.can_trade():
        status = dm.get_status()
        logger.warning(f"Trading halted: {status['halt_reason']}")
        return None

    # Estimate equalization/draw probability
    estimated_prob = estimate_draw_probability(signal)

    if EXIT_STRATEGY == "equalization":
        # For equalization: buy at entry_price, sell at exit_price
        # Break-even rate = entry_price / exit_price
        breakeven_rate = signal.draw_price / EXIT_PRICE_ON_EQUALIZATION
        edge = estimated_prob - breakeven_rate

        if edge <= 0:
            logger.info(
                f"No edge: estimated {estimated_prob:.1%} <= "
                f"breakeven {breakeven_rate:.1%} (entry {signal.draw_price:.1%})"
            )
            return None

        # Kelly for equalization: odds are exit_price/entry_price - 1 (net win per $1 risked)
        # implied_prob from market = entry_price / exit_price
        american_odds = _probability_to_american_odds(breakeven_rate)
    else:
        # For expiry: binary bet, payout $1 if draw
        edge = estimated_prob - signal.draw_price
        if edge <= 0:
            logger.info(f"No edge: estimated {estimated_prob:.1%} <= market {signal.draw_price:.1%}")
            return None
        american_odds = _probability_to_american_odds(signal.draw_price)

    # Calculate Kelly bet size
    kelly_raw = calculate_kelly_criterion(american_odds, estimated_prob)

    if kelly_raw <= 0:
        logger.info("Kelly criterion says no bet")
        return None

    # Apply fractional Kelly and cap
    kelly_pct = min(kelly_raw * KELLY_FRACTION, MAX_BET_PCT)
    bet_amount = round(bankroll * (kelly_pct / 100), 2)

    if bet_amount < 1.0:
        logger.info(f"Bet amount too small: ${bet_amount:.2f}")
        return None

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create position
    position_id = (
        f"{signal.home_team}_{signal.away_team}_{today}_m{signal.minute}"
    ).replace(" ", "_")

    position = {
        "position_id": position_id,
        "match_id": signal.match_id,
        "league": signal.league,
        "home_team": signal.home_team,
        "away_team": signal.away_team,
        "home_score_at_entry": signal.home_score,
        "away_score_at_entry": signal.away_score,
        "minute_of_signal": signal.minute,
        "momentum_value": signal.momentum_value,
        "momentum_source": signal.momentum_source,
        "entry_draw_price": signal.draw_price,
        "estimated_draw_prob": round(estimated_prob, 4),
        "edge": round(edge, 4),
        "draw_yes_token_id": signal.draw_token_id,
        "polymarket_event_id": signal.polymarket_event_id,
        "bet_kelly": round(kelly_pct, 2),
        "bet_amount": bet_amount,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "xg_home": signal.xg_home,
        "xg_away": signal.xg_away,
        "exit_strategy": EXIT_STRATEGY,
        "status": "open",
        "exit_time": None,
        "exit_reason": None,
        "exit_price": None,
        "exit_minute": None,
        "score_at_exit": None,
        "final_score": None,
        "was_draw": None,
        "pnl": None,
    }

    positions[position_id] = position
    save_positions(positions)

    # Log trade
    log_trade({
        "type": "ENTRY",
        "time": position["entry_time"],
        "position_id": position_id,
        "match": f"{signal.home_team} vs {signal.away_team}",
        "league": signal.league,
        "minute": signal.minute,
        "score": f"{signal.home_score}-{signal.away_score}",
        "momentum": signal.momentum_value,
        "draw_price": signal.draw_price,
        "estimated_prob": round(estimated_prob, 4),
        "edge": round(edge, 4),
        "kelly": round(kelly_pct, 2),
        "amount": bet_amount,
    })

    # Alert
    am = AlertManager(data_dir=DATA_DIR, webhook_url=DISCORD_WEBHOOK_URL)
    am.entry(
        game=f"{signal.home_team} vs {signal.away_team}",
        bet_side="DRAW",
        entry_price=signal.draw_price,
        bet_amount=bet_amount,
        edge=edge * 100,
        data={
            "league": signal.league,
            "minute": signal.minute,
            "score": f"{signal.home_score}-{signal.away_score}",
            "momentum": f"{signal.momentum_value:.3f} ({signal.momentum_source})",
        },
    )

    logger.info(
        f"POSITION OPENED: {position_id} | "
        f"Draw @ {signal.draw_price:.1%} | "
        f"Edge {edge:.1%} | Kelly {kelly_pct:.1f}% | ${bet_amount:.2f}"
    )

    return position_id


def resolve_equalization(position_id: str, current_score: str, minute: int) -> float:
    """Resolve a position mid-match when the losing team equalizes.

    Equalization strategy: sell draw shares immediately when the score ties.
    Shares bought at entry_price are sold at EXIT_PRICE_ON_EQUALIZATION (~$0.42).

    P&L = (exit_price / entry_price - 1) * bet_amount

    Returns P&L amount.
    """
    positions = load_positions()
    if position_id not in positions:
        logger.error(f"Position {position_id} not found")
        return 0.0

    position = positions[position_id]
    if position["status"] != "open":
        logger.warning(f"Position {position_id} already resolved")
        return 0.0

    entry_price = position["entry_draw_price"]
    bet_amount = position["bet_amount"]
    exit_price = EXIT_PRICE_ON_EQUALIZATION

    # Shares bought at entry_price, sold at exit_price
    # Number of shares = bet_amount / entry_price
    # Sale proceeds = shares * exit_price
    # P&L = proceeds - cost
    shares = bet_amount / entry_price
    proceeds = shares * exit_price
    pnl = round(proceeds - bet_amount, 2)

    position["status"] = "resolved_win"
    position["exit_time"] = datetime.now(timezone.utc).isoformat()
    position["exit_reason"] = "equalization"
    position["exit_price"] = exit_price
    position["exit_minute"] = minute
    position["score_at_exit"] = current_score
    position["was_draw"] = None  # Match still in progress
    position["pnl"] = pnl

    # Update bankroll
    bankroll = load_bankroll()
    new_bankroll = round(bankroll + pnl, 2)
    save_bankroll(new_bankroll)

    # Save position
    positions[position_id] = position
    save_positions(positions)

    # Log trade
    log_trade({
        "type": "EXIT_EQUALIZATION",
        "time": position["exit_time"],
        "position_id": position_id,
        "match": f"{position['home_team']} vs {position['away_team']}",
        "score_at_exit": current_score,
        "exit_minute": minute,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
    })

    # Update drawdown manager
    dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
    dm.record_pnl(pnl, position_id, new_bankroll)

    # Alert
    am = AlertManager(data_dir=DATA_DIR, webhook_url=DISCORD_WEBHOOK_URL)
    am.resolution(
        game=f"{position['home_team']} vs {position['away_team']}",
        won=True,
        pnl=pnl,
        data={
            "exit_reason": "EQUALIZATION",
            "score": current_score,
            "minute": minute,
            "entry_price": f"{entry_price:.1%}",
            "exit_price": f"{exit_price:.1%}",
            "bankroll": f"${new_bankroll:.2f}",
        },
    )

    logger.info(
        f"EQUALIZATION EXIT: {position_id} | Score: {current_score} (min {minute}) | "
        f"Sold @ {exit_price:.1%} | P&L: ${pnl:+.2f} | Bankroll: ${new_bankroll:.2f}"
    )

    return pnl


def resolve_position(position_id: str, was_draw: bool, final_score: str = None) -> float:
    """Resolve a position when the match ends (expiry strategy or equalization miss).

    In Polymarket draw markets:
    - Draw (Yes wins): shares pay $1.00, profit = (1/price - 1) * bet_amount
    - No draw (Yes loses): shares pay $0.00, loss = -bet_amount

    Returns P&L amount.
    """
    positions = load_positions()
    if position_id not in positions:
        logger.error(f"Position {position_id} not found")
        return 0.0

    position = positions[position_id]
    if position["status"] != "open":
        logger.warning(f"Position {position_id} already resolved")
        return 0.0

    entry_price = position["entry_draw_price"]
    bet_amount = position["bet_amount"]

    if was_draw:
        # Shares bought at entry_price pay $1.00
        # Number of shares = bet_amount / entry_price
        # Payout = shares * $1.00 = bet_amount / entry_price
        # Profit = payout - bet_amount
        payout = bet_amount / entry_price
        pnl = round(payout - bet_amount, 2)
        position["status"] = "resolved_win"
    else:
        # Shares worth $0.00
        pnl = -bet_amount
        position["status"] = "resolved_loss"

    position["exit_time"] = datetime.now(timezone.utc).isoformat()
    position["exit_reason"] = "match_ended"
    position["final_score"] = final_score
    position["was_draw"] = was_draw
    position["pnl"] = pnl

    # Update bankroll
    bankroll = load_bankroll()
    new_bankroll = round(bankroll + pnl, 2)
    save_bankroll(new_bankroll)

    # Save position
    positions[position_id] = position
    save_positions(positions)

    # Log trade
    log_trade({
        "type": "RESOLVED",
        "time": position["exit_time"],
        "position_id": position_id,
        "match": f"{position['home_team']} vs {position['away_team']}",
        "final_score": final_score,
        "was_draw": was_draw,
        "pnl": pnl,
        "entry_price": entry_price,
    })

    # Update drawdown manager
    dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
    dm.record_pnl(pnl, position_id, new_bankroll)

    # Alert
    am = AlertManager(data_dir=DATA_DIR, webhook_url=DISCORD_WEBHOOK_URL)
    am.resolution(
        game=f"{position['home_team']} vs {position['away_team']}",
        won=was_draw,
        pnl=pnl,
        data={
            "final_score": final_score,
            "entry_price": f"{entry_price:.1%}",
            "bankroll": f"${new_bankroll:.2f}",
        },
    )

    result = "WIN" if was_draw else "LOSS"
    logger.info(
        f"RESOLVED: {position_id} | {result} | P&L: ${pnl:+.2f} | "
        f"Bankroll: ${new_bankroll:.2f}"
    )

    return pnl


def get_open_positions() -> List[Dict]:
    """Get all open positions."""
    positions = load_positions()
    return [p for p in positions.values() if p["status"] == "open"]


def get_resolved_positions(days: int = 7) -> List[Dict]:
    """Get resolved positions from the last N days."""
    from datetime import timedelta
    positions = load_positions()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return [
        p for p in positions.values()
        if p["status"] != "open" and (p.get("exit_time") or "") >= cutoff
    ]


def show_status():
    """Print current positions and P&L summary."""
    bankroll = load_bankroll()
    positions = load_positions()
    open_pos = [p for p in positions.values() if p["status"] == "open"]
    resolved = [p for p in positions.values() if p["status"] != "open"]

    print("=" * 60)
    print("SOCCER MOMENTUM DRAW BETTING - STATUS")
    print("=" * 60)
    print(f"Bankroll: ${bankroll:.2f} (started: ${STARTING_BANKROLL:.2f})")
    print(f"Total P&L: ${bankroll - STARTING_BANKROLL:+.2f}")
    print(f"Open positions: {len(open_pos)}/{MAX_SIMULTANEOUS_POSITIONS}")
    print()

    if open_pos:
        print("--- OPEN POSITIONS ---")
        for p in open_pos:
            print(
                f"  [{p['league']}] {p['home_team']} vs {p['away_team']}"
            )
            print(
                f"    Entry: Draw @ {p['entry_draw_price']:.1%} | "
                f"Edge: {p['edge']:.1%} | ${p['bet_amount']:.2f}"
            )
            print(
                f"    Signal: min {p['minute_of_signal']} | "
                f"Score {p['home_score_at_entry']}-{p['away_score_at_entry']} | "
                f"Momentum {p['momentum_value']:.3f}"
            )
            print()

    if resolved:
        wins = sum(1 for p in resolved if p.get("was_draw"))
        losses = len(resolved) - wins
        total_pnl = sum(p.get("pnl", 0) for p in resolved)
        win_rate = wins / len(resolved) * 100 if resolved else 0

        print("--- RESOLVED ---")
        print(f"  Record: {wins}W-{losses}L ({win_rate:.0f}%)")
        print(f"  Total P&L: ${total_pnl:+.2f}")
        print()

        # Show last 5 resolved
        recent = sorted(resolved, key=lambda p: p.get("exit_time", ""), reverse=True)[:5]
        for p in recent:
            result = "WIN" if p.get("was_draw") else "LOSS"
            print(
                f"  {result}: {p['home_team']} vs {p['away_team']} "
                f"({p.get('final_score', '?')}) -> ${p.get('pnl', 0):+.2f}"
            )

    # Drawdown status
    dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
    dm.sync_bankroll(bankroll)
    dd_status = dm.get_status()
    if dd_status["alerts"]:
        print("\n--- ALERTS ---")
        for alert in dd_status["alerts"]:
            print(f"  {alert}")


def show_dashboard():
    """Print comprehensive dashboard."""
    from src.Soccer.momentum_scanner import MIN_MINUTE, MAX_MINUTE

    show_status()
    print()
    print("--- CONFIGURATION ---")
    print(f"  Exit strategy: {EXIT_STRATEGY}")
    if EXIT_STRATEGY == "equalization":
        print(f"  Exit price on equalization: {EXIT_PRICE_ON_EQUALIZATION:.1%}")
    print(f"  Kelly fraction: {KELLY_FRACTION:.0%}")
    print(f"  Max bet: {MAX_BET_PCT:.0f}%")
    print(f"  Max positions: {MAX_SIMULTANEOUS_POSITIONS}")
    print(f"  Momentum threshold: {MOMENTUM_THRESHOLD}")
    print(f"  Minute window: {MIN_MINUTE}-{MAX_MINUTE}")
    print(f"  Edge source: backtest-calibrated (874 matches)")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Soccer Draw Betting Paper Trader")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--dashboard", action="store_true", help="Show full dashboard")
    parser.add_argument("--reset", action="store_true", help="Reset bankroll and positions")

    args = parser.parse_args()

    if args.reset:
        ensure_data_dir()
        save_bankroll(STARTING_BANKROLL)
        save_positions({})
        dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
        dm.reset_all(STARTING_BANKROLL)
        print(f"Reset complete. Bankroll: ${STARTING_BANKROLL:.2f}")
    elif args.dashboard:
        show_dashboard()
    else:
        show_status()


if __name__ == "__main__":
    main()
