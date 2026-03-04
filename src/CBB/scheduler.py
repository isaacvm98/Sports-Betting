"""
CBB Paper Trading Scheduler

Runs continuously:
- Places bets ~30 minutes before each game
- Monitors positions every 5 minutes
- Tracks results until resolution

Usage:
    python -m src.CBB.scheduler          # Run continuously
    python -m src.CBB.scheduler --status # Show current status
    python -m src.CBB.scheduler --once   # Run one cycle and exit
"""

import argparse
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Project imports
from src.CBB.paper_trader import (
    init_positions,
    monitor_positions,
    show_status,
    generate_report,
    resolve_positions,
    load_positions,
    save_positions,
    load_bankroll,
    load_predictions,
    fetch_polymarket_cbb_spreads,
    match_predictions_to_markets,
    determine_bet,
    calculate_ci_width_size,
    log_trade,
    can_open_position,
    DATA_DIR,
    STARTING_BANKROLL,
    DISCORD_WEBHOOK_URL,
)
from src.Utils.AlertManager import AlertManager

# Configuration
CHECK_INTERVAL_SECONDS = 300   # Check every 5 minutes
MINUTES_BEFORE_GAME = 30       # Place bets 30 min before game

ET = ZoneInfo("America/New_York")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "cbb_scheduler.log"


def setup_logging():
    """Set up logging to file and console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def place_bets_for_upcoming_games(predictions_df, logger, alert_mgr=None):
    """
    Place bets for games starting within MINUTES_BEFORE_GAME.

    Returns number of new bets placed.
    """
    # Fetch markets for games starting soon
    markets = fetch_polymarket_cbb_spreads(minutes_before_game=MINUTES_BEFORE_GAME)

    if not markets:
        return 0

    logger.info(f"Found {len(markets)} games starting within {MINUTES_BEFORE_GAME} min")

    # Match with predictions
    matches = match_predictions_to_markets(predictions_df, markets)

    if not matches:
        return 0

    positions = load_positions()
    bankroll = load_bankroll()
    bets_placed = 0

    for match in matches:
        game_key = match['game_key']
        position_id = f"{game_key}_{datetime.now().strftime('%Y%m%d')}"

        # Skip if already have position
        if position_id in positions:
            logger.info(f"  {match['away_team']} @ {match['home_team']} - already have position")
            continue

        # Check position limits
        can_bet, limit_reason = can_open_position(positions)
        if not can_bet:
            logger.info(f"  {match['away_team']} @ {match['home_team']} - {limit_reason}")
            continue

        # Determine bet for spread
        if match['has_spread']:
            bet_side, edge, reasoning = determine_bet(match)

            if bet_side:
                # bet_side is 'home' or 'away' from determine_bet
                entry_price = match['home_cover_price'] if bet_side == 'home' else match['away_cover_price']
                bet_size = calculate_ci_width_size(match['ci_width'], edge)
                bet_amount = bankroll * (bet_size / 100)

                position = {
                    "game_key": game_key,
                    "home_team": match['home_team'],
                    "away_team": match['away_team'],
                    "game_time": match['game_time'],
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "model_spread": match['model_spread'],
                    "away_spread": match.get('away_spread'),
                    "home_spread": match.get('home_spread'),
                    "is_home_spread": match.get('is_home_spread'),
                    "ci_width": match['ci_width'],
                    "bet_type": "spread",
                    "bet_side": bet_side,
                    "edge": edge,
                    "entry_price": entry_price,
                    "bet_size_pct": bet_size,
                    "bet_amount": round(bet_amount, 2),
                    "yes_token": match.get('yes_token'),
                    "no_token": match.get('no_token'),
                    "market_id": match['market_id'],
                    "status": "open",
                }

                positions[position_id] = position
                bets_placed += 1

                logger.info(f"  BET: {match['away_team']} @ {match['home_team']}")
                logger.info(f"       {bet_side.upper()} @ {entry_price:.1%} | {bet_size:.1f}% (${bet_amount:.2f})")
                logger.info(f"       Edge: {edge:+.1f} pts")

                log_trade({
                    "type": "ENTRY",
                    "time": datetime.now(timezone.utc).isoformat(),
                    "position_id": position_id,
                    "game": f"{match['away_team']} @ {match['home_team']}",
                    "bet_type": "spread",
                    "bet_side": bet_side,
                    "entry_price": entry_price,
                    "bet_size_pct": bet_size,
                    "bet_amount": round(bet_amount, 2),
                    "edge": edge,
                })

                # Send Discord alert for new bet
                if alert_mgr:
                    alert_mgr.entry(
                        game=f"CBB: {match['away_team']} @ {match['home_team']}",
                        bet_side=bet_side,
                        entry_price=entry_price,
                        bet_amount=bet_amount,
                        edge=edge,
                        data={"bet_type": "spread", "system": "CBB"}
                    )
            else:
                logger.info(f"  SKIP: {match['away_team']} @ {match['home_team']} - {reasoning}")

    save_positions(positions)
    return bets_placed


def check_for_resolutions(logger, alert_mgr=None):
    """Check if any games have ended, mark for resolution, and resolve them."""
    positions = load_positions()
    open_positions = {k: v for k, v in positions.items() if v['status'] == 'open'}

    now = datetime.now(timezone.utc)
    marked = 0

    # First, mark open positions whose games have ended
    for position_id, position in open_positions.items():
        game_time_str = position.get('game_time')
        if not game_time_str:
            continue

        try:
            game_time = datetime.fromisoformat(game_time_str)
        except:
            continue

        # If game ended more than 3 hours ago, mark for resolution
        if now > game_time + timedelta(hours=3):
            if position['status'] == 'open':
                position['status'] = 'pending_resolution'
                marked += 1
                logger.info(f"  Marked for resolution: {position['away_team']} @ {position['home_team']}")

    if marked > 0:
        save_positions(positions)
        logger.info(f"  {marked} games marked for resolution")

    # Now resolve any pending positions
    pending = {k: v for k, v in load_positions().items() if v['status'] == 'pending_resolution'}
    if pending:
        logger.info(f"  Attempting to resolve {len(pending)} pending positions...")
        resolved_count = resolve_positions()
        if resolved_count > 0:
            logger.info(f"  Resolved {resolved_count} positions")

            # Check for newly resolved positions and send alerts
            updated_positions = load_positions()
            for position_id, position in updated_positions.items():
                if position.get('status') == 'resolved' and position.get('pnl') is not None:
                    # Check if we already processed this (avoid duplicate alerts)
                    exit_reason = position.get('exit_reason', '')
                    if 'RESOLVED' in exit_reason:
                        pnl = position.get('pnl', 0)
                        won = pnl > 0
                        game = f"CBB: {position['away_team']} @ {position['home_team']}"

                        # Send alert
                        if alert_mgr:
                            alert_mgr.resolution(
                                game,
                                won=won,
                                pnl=pnl,
                                data={
                                    'bet_type': position.get('bet_type', 'spread'),
                                    'edge': f"{position.get('edge', 0):.1f}",
                                }
                            )


def run_scheduler(once=False):
    """Main scheduler loop."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("CBB PAPER TRADING SCHEDULER")
    logger.info(f"Started: {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}")
    logger.info("=" * 60)
    logger.info(f"Check interval: {CHECK_INTERVAL_SECONDS // 60} minutes")
    logger.info(f"Bet timing: {MINUTES_BEFORE_GAME} minutes before game")
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    # Initialize AlertManager with Discord webhook
    alert_mgr = AlertManager(
        data_dir=DATA_DIR,
        enable_console=True,
        enable_file=True,
        webhook_url=DISCORD_WEBHOOK_URL,
        webhook_platform="discord",
    )

    # Send startup alert
    alert_mgr.info("CBB Paper Trader started", {"system": "CBB", "status": "online"})

    # Load predictions once at startup
    logger.info("Loading predictions from GitHub...")
    try:
        predictions_df = load_predictions()
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        alert_mgr.error(f"Failed to load predictions: {e}", {"system": "CBB"})
        return

    iteration = 0

    while True:
        try:
            iteration += 1
            now_et = datetime.now(ET)

            logger.info("-" * 40)
            logger.info(f"[{now_et.strftime('%I:%M %p ET')}] Check #{iteration}")

            # Current status
            positions = load_positions()
            open_count = len([p for p in positions.values() if p['status'] == 'open'])
            pending_count = len([p for p in positions.values() if p['status'] == 'pending_resolution'])
            bankroll = load_bankroll()

            logger.info(f"Positions: {open_count} open, {pending_count} pending | Bankroll: ${bankroll:.2f}")

            # 1. Place bets for games starting soon
            logger.info("")
            logger.info("--- Checking for upcoming games ---")
            bets_placed = place_bets_for_upcoming_games(predictions_df, logger, alert_mgr)
            if bets_placed > 0:
                logger.info(f"Placed {bets_placed} new bet(s)")

            # 2. Monitor existing positions
            if open_count > 0:
                logger.info("")
                logger.info("--- Monitoring positions ---")
                monitor_positions()

            # 3. Check for resolutions
            check_for_resolutions(logger, alert_mgr)

            if once:
                logger.info("")
                logger.info("Single run complete")
                break

            # Sleep until next check
            logger.info("")
            logger.info(f"Next check in {CHECK_INTERVAL_SECONDS // 60} minutes...")
            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("")
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if once:
                break
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description='CBB Paper Trading Scheduler')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--once', action='store_true', help='Run one cycle and exit')
    parser.add_argument('--report', action='store_true', help='Generate report')

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.report:
        generate_report()
    elif args.once:
        run_scheduler(once=True)
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
