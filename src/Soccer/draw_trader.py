"""
Draw Trader — Hold-to-Expiry Paper Trader

Strategy: Buy draw token when XGBoost + survival model signals +EV,
hold to match expiry. Draw = $1 payout, no draw = $0.

Replaces the old equalization-exit paper trader.

Usage:
    from src.Soccer.draw_trader import open_draw_position, resolve_draw_position
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List

from src.Soccer.draw_scanner import DrawSignal
from src.Utils.DrawdownManager import DrawdownManager
from src.Utils.AlertManager import AlertManager

logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================
DATA_DIR = Path("Data/soccer_draw_trading")
POSITIONS_FILE = DATA_DIR / "positions.json"
TRADES_LOG = DATA_DIR / "trades.json"
BANKROLL_FILE = DATA_DIR / "bankroll.json"

STARTING_BANKROLL = 1000.0
MAX_BET_PCT = 8.0                 # Max 8% of bankroll per bet
MAX_SIMULTANEOUS_POSITIONS = None  # No limit — bets are independent
KELLY_FRACTION = 0.15             # 15% fractional Kelly

DISCORD_WEBHOOK_URL = None
# =========================================================


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_bankroll() -> float:
    _ensure_dir()
    if BANKROLL_FILE.exists():
        try:
            with open(BANKROLL_FILE, "r") as f:
                return float(json.load(f).get("bankroll", STARTING_BANKROLL))
        except (json.JSONDecodeError, IOError):
            pass
    return STARTING_BANKROLL


def save_bankroll(bankroll: float):
    _ensure_dir()
    with open(BANKROLL_FILE, "w") as f:
        json.dump({
            "bankroll": round(bankroll, 2),
            "updated": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


def load_positions() -> Dict:
    _ensure_dir()
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_positions(positions: Dict):
    _ensure_dir()
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def log_trade(trade: Dict):
    _ensure_dir()
    trades = []
    if TRADES_LOG.exists():
        try:
            with open(TRADES_LOG, "r") as f:
                trades = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    trades.append(trade)
    with open(TRADES_LOG, "w") as f:
        json.dump(trades, f, indent=2, default=str)


def half_kelly_size(model_prob: float, entry_price: float, bankroll: float,
                    bet_side: str = "YES") -> float:
    """Half-Kelly sizing for binary bet.

    YES: K = (p*b - q) / b where b = 1/price - 1, p = P(draw)
    NO:  K = (p*b - q) / b where b = 1/(1-price) - 1, p = P(no draw)
    """
    if entry_price <= 0 or entry_price >= 1:
        return 0

    if bet_side == "YES":
        b = (1.0 / entry_price) - 1.0
        p = model_prob
    else:
        no_price = 1.0 - entry_price
        b = (1.0 / no_price) - 1.0
        p = 1.0 - model_prob

    q = 1.0 - p
    k = (p * b - q) / b
    if k <= 0:
        return 0
    half_k = k * KELLY_FRACTION
    return bankroll * min(half_k, MAX_BET_PCT / 100)


def open_draw_position(signal: DrawSignal) -> Optional[str]:
    """Open a hold-to-expiry draw position from a DrawSignal.

    Returns position_id if opened, None if skipped.
    """
    positions = load_positions()
    bankroll = load_bankroll()

    # No position limit — bets are independent

    # No duplicate match
    for p in positions.values():
        if p["match_id"] == signal.match_id and p["status"] == "open":
            logger.info(f"Already have position on match {signal.match_id}")
            return None

    # Drawdown check
    dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
    dm.sync_bankroll(bankroll)
    if not dm.can_trade():
        logger.warning(f"Trading halted: {dm.get_status()['halt_reason']}")
        return None

    # Size the bet
    bet_side = signal.bet_side
    bet_amount = half_kelly_size(
        signal.xgb_draw_prob, signal.pm_draw_price, bankroll, bet_side
    )
    if bet_amount < 1.0:
        logger.info(f"Bet too small: ${bet_amount:.2f}")
        return None

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    position_id = (
        f"{signal.home_team}_{signal.away_team}_{today}_m{signal.minute}_{bet_side}"
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
        "losing_team": signal.losing_team,
        # Bet direction
        "bet_side": bet_side,
        # Model data
        "xgb_draw_prob": round(signal.xgb_draw_prob, 4),
        "entry_draw_price": signal.pm_draw_price,
        "edge": round(abs(signal.edge), 4),
        "p_equalize_10min": round(signal.p_equalize_10min, 4),
        "p_equalize_20min": round(signal.p_equalize_20min, 4),
        "near_hazard_5min": round(signal.near_hazard_5min, 4),
        # Features
        "momentum_value": signal.momentum_value,
        "xg_home": signal.xg_home,
        "xg_away": signal.xg_away,
        "score_level": signal.score_level,
        "losing_goals_before": signal.losing_goals_before,
        # Market
        "draw_yes_token_id": signal.draw_token_id,
        "polymarket_event_id": signal.polymarket_event_id,
        # Sizing
        "bet_amount": round(bet_amount, 2),
        "bet_kelly_pct": round(bet_amount / bankroll * 100, 2),
        "entry_time": datetime.now(timezone.utc).isoformat(),
        # Exit (filled on resolution)
        "exit_strategy": "hold_to_expiry",
        "status": "open",
        "exit_time": None,
        "final_score": None,
        "was_draw": None,
        "pnl": None,
    }

    positions[position_id] = position
    save_positions(positions)

    log_trade({
        "type": "ENTRY",
        "time": position["entry_time"],
        "position_id": position_id,
        "bet_side": bet_side,
        "match": f"{signal.home_team} vs {signal.away_team}",
        "league": signal.league,
        "minute": signal.minute,
        "score": f"{signal.home_score}-{signal.away_score}",
        "draw_price": signal.pm_draw_price,
        "xgb_prob": round(signal.xgb_draw_prob, 4),
        "edge": round(abs(signal.edge), 4),
        "amount": round(bet_amount, 2),
        "losing_goals_before": signal.losing_goals_before,
        "score_level": signal.score_level,
    })

    # Alert
    if bet_side == "YES":
        entry_label = f"YES Draw @ {signal.pm_draw_price:.1%}"
    else:
        entry_label = f"NO Draw @ {1 - signal.pm_draw_price:.1%}"

    am = AlertManager(data_dir=DATA_DIR, webhook_url=DISCORD_WEBHOOK_URL)
    am.entry(
        game=f"{signal.home_team} vs {signal.away_team}",
        bet_side=f"{entry_label} (hold to expiry)",
        entry_price=signal.pm_draw_price if bet_side == "YES" else 1 - signal.pm_draw_price,
        bet_amount=bet_amount,
        edge=abs(signal.edge) * 100,
        data={
            "league": signal.league,
            "minute": signal.minute,
            "score": f"{signal.home_score}-{signal.away_score}",
            "xgb_prob": f"{signal.xgb_draw_prob:.3f}",
            "losing_goals_before": signal.losing_goals_before,
        },
    )

    logger.info(
        f"POSITION OPENED: {position_id} | "
        f"{bet_side} @ {signal.pm_draw_price:.1%} | "
        f"P(draw)={signal.xgb_draw_prob:.3f} | Edge={abs(signal.edge):+.3f} | "
        f"${bet_amount:.2f}"
    )

    return position_id


def resolve_draw_position(
    position_id: str, was_draw: bool, final_score: str = None
) -> float:
    """Resolve a position at match end.

    Draw: shares pay $1.00, profit = (1/price - 1) * bet_amount
    No draw: shares pay $0.00, loss = -bet_amount
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
    bet_side = position.get("bet_side", "YES")

    if bet_side == "YES":
        if was_draw:
            payout = bet_amount / entry_price  # shares * $1
            pnl = round(payout - bet_amount, 2)
            position["status"] = "resolved_win"
        else:
            pnl = -bet_amount
            position["status"] = "resolved_loss"
    else:  # NO
        if not was_draw:
            no_price = 1.0 - entry_price
            payout = bet_amount / no_price  # NO shares * $1
            pnl = round(payout - bet_amount, 2)
            position["status"] = "resolved_win"
        else:
            pnl = -bet_amount
            position["status"] = "resolved_loss"

    position["exit_time"] = datetime.now(timezone.utc).isoformat()
    position["final_score"] = final_score
    position["was_draw"] = was_draw
    position["pnl"] = pnl

    # Update bankroll
    bankroll = load_bankroll()
    new_bankroll = round(bankroll + pnl, 2)
    save_bankroll(new_bankroll)

    positions[position_id] = position
    save_positions(positions)

    # Log
    log_trade({
        "type": "RESOLVED",
        "time": position["exit_time"],
        "position_id": position_id,
        "match": f"{position['home_team']} vs {position['away_team']}",
        "final_score": final_score,
        "was_draw": was_draw,
        "entry_price": entry_price,
        "pnl": pnl,
        "bankroll": new_bankroll,
    })

    # Drawdown
    dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
    dm.record_pnl(pnl, position_id, new_bankroll)

    won = (bet_side == "YES" and was_draw) or (bet_side == "NO" and not was_draw)

    # Alert
    am = AlertManager(data_dir=DATA_DIR, webhook_url=DISCORD_WEBHOOK_URL)
    am.resolution(
        game=f"{position['home_team']} vs {position['away_team']}",
        won=won,
        pnl=pnl,
        data={
            "final_score": final_score,
            "entry_price": f"{entry_price:.1%}",
            "xgb_prob": f"{position['xgb_draw_prob']:.3f}",
            "bankroll": f"${new_bankroll:.2f}",
        },
    )

    result = f"WIN ({bet_side})" if won else f"LOSS ({bet_side})"
    logger.info(
        f"RESOLVED: {position_id} | {result} | "
        f"P&L: ${pnl:+.2f} | Bankroll: ${new_bankroll:.2f}"
    )

    return pnl


def get_open_positions() -> List[Dict]:
    return [p for p in load_positions().values() if p["status"] == "open"]


def show_status():
    """Print current state."""
    bankroll = load_bankroll()
    positions = load_positions()
    open_pos = [p for p in positions.values() if p["status"] == "open"]
    resolved = [p for p in positions.values() if p["status"] != "open"]

    print("=" * 60)
    print("SOCCER DRAW BETTING (XGBoost + Survival) — STATUS")
    print("=" * 60)
    print(f"Bankroll: ${bankroll:.2f} (started: ${STARTING_BANKROLL:.2f})")
    print(f"Total P&L: ${bankroll - STARTING_BANKROLL:+.2f}")
    print(f"Open positions: {len(open_pos)}")

    if resolved:
        wins = sum(1 for p in resolved if p.get("was_draw"))
        losses = len(resolved) - wins
        total_pnl = sum(p.get("pnl", 0) for p in resolved)
        wr = wins / len(resolved) * 100 if resolved else 0
        print(f"Record: {wins}W-{losses}L ({wr:.0f}%)")
        print(f"Resolved P&L: ${total_pnl:+.2f}")

    if open_pos:
        print(f"\n--- Open Positions ---")
        for p in open_pos:
            side = p.get('bet_side', 'YES')
            print(
                f"  [{p['league']}] {p['home_team']} vs {p['away_team']} ({side})"
            )
            print(
                f"    Draw @ {p['entry_draw_price']:.1%} | "
                f"P(draw)={p['xgb_draw_prob']:.3f} | "
                f"Edge={p['edge']:+.3f} | ${p['bet_amount']:.2f}"
            )
            print(
                f"    Min {p['minute_of_signal']} | "
                f"Score {p['home_score_at_entry']}-{p['away_score_at_entry']} | "
                f"LG_before={p['losing_goals_before']}"
            )

    if resolved:
        print(f"\n--- Last 5 Resolved ---")
        recent = sorted(resolved, key=lambda p: p.get("exit_time", ""), reverse=True)[:5]
        for p in recent:
            side = p.get('bet_side', 'YES')
            was_d = p.get("was_draw")
            won = (side == "YES" and was_d) or (side == "NO" and not was_d)
            result = "WIN" if won else "LOSS"
            print(
                f"  {result} ({side}): {p['home_team']} vs {p['away_team']} "
                f"({p.get('final_score', '?')}) | "
                f"Entry {p['entry_draw_price']:.1%} | ${p.get('pnl', 0):+.2f}"
            )


if __name__ == "__main__":
    show_status()
