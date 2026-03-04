"""
Soccer Scheduler - Continuous Match Monitor

Runs continuously during European match hours, scanning for
momentum draw signals and managing paper trading positions.

Flow:
  1. Every SCAN_INTERVAL seconds: run momentum scanner
  2. For each signal: open paper trading position
  3. Every RESOLUTION_CHECK_INTERVAL minutes: check for finished matches
  4. End of day: generate daily report

Usage:
    python -m src.Soccer.scheduler
"""

import time
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.DataProviders.FotMobProvider import FotMobProvider
from src.DataProviders.PolymarketSoccerProvider import PolymarketSoccerProvider
from src.Soccer.momentum_scanner import MomentumScanner, SCAN_INTERVAL, LEAGUES
from src.Soccer.paper_trader import (
    open_position,
    resolve_position,
    resolve_equalization,
    load_positions,
    save_positions,
    load_bankroll,
    show_status,
    ensure_data_dir,
    DATA_DIR,
    STARTING_BANKROLL,
    DISCORD_WEBHOOK_URL,
    EXIT_STRATEGY,
)
from src.Utils.DrawdownManager import DrawdownManager
from src.Utils.AlertManager import AlertManager, AlertType
from src.Polymarket.websocket_monitor import WebSocketPriceMonitor

# ===================== CONFIGURATION =====================
LOG_DIR = PROJECT_ROOT / "logs"

# European match window (UTC)
# Weekday: ~17:00-22:30 UTC | Weekend: ~11:00-22:30 UTC
# We use a wide window to catch all possibilities
SCAN_START_HOUR_UTC = 11
SCAN_END_HOUR_UTC = 23

# How often to check if matches have finished (seconds)
RESOLUTION_CHECK_INTERVAL = 5 * 60  # 5 minutes

# How often to check for equalization (must be frequent to catch score changes)
EQUALIZATION_CHECK_INTERVAL = 60  # Every minute (same as scan interval)

# Sleep duration when outside match window (seconds)
OFF_WINDOW_SLEEP = 5 * 60  # 5 minutes
# =========================================================


def setup_logging() -> logging.Logger:
    """Set up logging to file and console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "soccer_scheduler.log"

    log = logging.getLogger("soccer_scheduler")
    log.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    ))
    log.addHandler(fh)

    return log


def is_within_match_window() -> bool:
    """Check if current time is within the European match scanning window."""
    now_utc = datetime.now(timezone.utc)
    return SCAN_START_HOUR_UTC <= now_utc.hour < SCAN_END_HOUR_UTC


def check_equalizations(fotmob: FotMobProvider, log: logging.Logger):
    """Check if any open positions' matches have equalized (score is now tied).

    For equalization strategy:
    1. Fetch live match data for each open position
    2. If the score is now tied (losing team scored), trigger equalization exit
    3. Sell draw shares at EXIT_PRICE_ON_EQUALIZATION

    Only runs when EXIT_STRATEGY == "equalization".
    """
    if EXIT_STRATEGY != "equalization":
        return

    positions = load_positions()
    open_positions = {
        pid: p for pid, p in positions.items() if p["status"] == "open"
    }

    if not open_positions:
        return

    log.info(f"Checking {len(open_positions)} open positions for equalization...")

    for pid, pos in open_positions.items():
        match_id = pos["match_id"]
        entry_home = pos["home_score_at_entry"]
        entry_away = pos["away_score_at_entry"]

        try:
            details = fotmob.get_match_details(match_id)
        except Exception as e:
            log.warning(f"Error fetching match {match_id}: {e}")
            continue

        if not details:
            log.debug(f"No details for match {match_id}")
            continue

        home_score = details.get("home_score", entry_home)
        away_score = details.get("away_score", entry_away)
        minute = details.get("minute", 0)
        current_score = f"{home_score}-{away_score}"

        status = str(details.get("status", "")).lower()
        is_finished = any(
            term in status for term in ["finished", "ft", "ended", "fulltime"]
        )

        # If match finished without equalizing, resolve as loss at expiry
        if is_finished:
            was_draw = home_score == away_score
            final_score = current_score
            log.info(
                f"Match FINISHED (no equalization exit): "
                f"{pos['home_team']} {final_score} {pos['away_team']} "
                f"-> {'DRAW' if was_draw else 'NO DRAW'}"
            )
            pnl = resolve_position(pid, was_draw, final_score)
            log.info(f"Position {pid} resolved at expiry: P&L ${pnl:+.2f}")
            continue

        # Check if score is now tied (equalization happened)
        if home_score == away_score and entry_home != entry_away:
            log.info(
                f"EQUALIZATION DETECTED: {pos['home_team']} {current_score} "
                f"{pos['away_team']} (min {minute}) | "
                f"Was {entry_home}-{entry_away} at entry"
            )
            pnl = resolve_equalization(pid, current_score, minute)
            log.info(f"Position {pid} equalization exit: P&L ${pnl:+.2f}")
        else:
            log.debug(
                f"Match {match_id}: {current_score} (min {minute}) - "
                f"no equalization yet"
            )


def check_resolutions(fotmob: FotMobProvider, log: logging.Logger):
    """Check if any open positions' matches have finished.

    For expiry strategy or as fallback for equalization strategy.
    Only handles positions that haven't been resolved by equalization.

    For each open position:
    1. Fetch match status from FotMob
    2. If match is finished, check final score
    3. Resolve position (draw or not)
    """
    positions = load_positions()
    open_positions = {
        pid: p for pid, p in positions.items() if p["status"] == "open"
    }

    if not open_positions:
        return

    log.info(f"Checking {len(open_positions)} open positions for resolution...")

    for pid, pos in open_positions.items():
        match_id = pos["match_id"]

        try:
            details = fotmob.get_match_details(match_id)
        except Exception as e:
            log.warning(f"Error fetching match {match_id}: {e}")
            continue

        if not details:
            log.debug(f"No details for match {match_id}")
            continue

        status = str(details.get("status", "")).lower()
        is_finished = any(
            term in status for term in ["finished", "ft", "ended", "fulltime"]
        )

        if not is_finished:
            log.debug(
                f"Match {match_id} ({pos['home_team']} vs {pos['away_team']}) "
                f"still in progress (status: {status})"
            )
            continue

        # Match is finished - resolve
        home_score = details["home_score"]
        away_score = details["away_score"]
        final_score = f"{home_score}-{away_score}"
        was_draw = home_score == away_score

        log.info(
            f"Match FINISHED: {pos['home_team']} {final_score} {pos['away_team']} "
            f"-> {'DRAW' if was_draw else 'NO DRAW'}"
        )

        pnl = resolve_position(pid, was_draw, final_score)
        log.info(f"Position {pid} resolved: P&L ${pnl:+.2f}")


def generate_daily_report(log: logging.Logger):
    """Generate end-of-day summary."""
    positions = load_positions()
    bankroll = load_bankroll()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Today's resolved positions
    today_resolved = [
        p for p in positions.values()
        if p["status"] != "open" and (p.get("exit_time") or "").startswith(today)
    ]

    # Today's opened positions
    today_opened = [
        p for p in positions.values()
        if (p.get("entry_time") or "").startswith(today)
    ]

    still_open = [p for p in positions.values() if p["status"] == "open"]

    if not today_resolved and not today_opened:
        log.info("No activity today - skipping report")
        return

    wins = sum(1 for p in today_resolved if p.get("was_draw"))
    losses = len(today_resolved) - wins
    total_pnl = sum(p.get("pnl", 0) for p in today_resolved)

    report_lines = [
        f"DAILY REPORT - {today}",
        "=" * 40,
        f"Bankroll: ${bankroll:.2f}",
        f"Positions opened: {len(today_opened)}",
        f"Positions resolved: {len(today_resolved)}",
        f"Record: {wins}W-{losses}L",
        f"Day P&L: ${total_pnl:+.2f}",
        f"Still open: {len(still_open)}",
        "",
    ]

    for p in today_resolved:
        result = "WIN" if p.get("was_draw") else "LOSS"
        report_lines.append(
            f"  {result}: [{p['league']}] {p['home_team']} vs {p['away_team']} "
            f"({p.get('final_score', '?')}) -> ${p.get('pnl', 0):+.2f}"
        )

    report_text = "\n".join(report_lines)

    # Save report
    ensure_data_dir()
    report_file = DATA_DIR / f"report_{today}.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    log.info(f"\n{report_text}")

    # Alert
    am = AlertManager(data_dir=DATA_DIR, webhook_url=DISCORD_WEBHOOK_URL)
    am.daily_summary(
        total_pnl=total_pnl,
        wins=wins,
        losses=losses,
        open_positions=len(still_open),
        data={"bankroll": bankroll},
    )


def run_scheduler():
    """Main scheduler loop."""
    log = setup_logging()

    log.info("=" * 60)
    log.info("SOCCER MOMENTUM DRAW BETTING - SCHEDULER")
    log.info("=" * 60)
    log.info(f"Exit strategy: {EXIT_STRATEGY}")
    log.info(f"Leagues: {', '.join(LEAGUES)}")
    log.info(f"Scan interval: {SCAN_INTERVAL}s")
    log.info(f"Match window: {SCAN_START_HOUR_UTC}:00 - {SCAN_END_HOUR_UTC}:00 UTC")
    log.info(f"Bankroll: ${load_bankroll():.2f}")
    log.info(f"Edge source: backtest-calibrated (874 matches)")
    log.info("=" * 60)

    # Initialize providers
    fotmob = FotMobProvider(leagues=LEAGUES)
    polymarket = PolymarketSoccerProvider(leagues=LEAGUES)
    scanner = MomentumScanner(fotmob, polymarket, logger_override=log)

    # Initialize WebSocket for real-time Polymarket prices
    def on_ws_price_update(asset_id, bid, ask):
        mid = (bid + ask) / 2
        log.debug(f"WS price: {asset_id[:20]}... bid={bid:.3f} ask={ask:.3f} mid={mid:.3f}")

    ws_monitor = WebSocketPriceMonitor(on_price_update=on_ws_price_update, logger=log)
    ws_monitor.start()
    log.info("WebSocket price monitor started")

    # Subscribe to any existing open positions
    positions = load_positions()
    open_token_ids = [
        p["draw_yes_token_id"]
        for p in positions.values()
        if p["status"] == "open" and p.get("draw_yes_token_id")
    ]
    if open_token_ids:
        ws_monitor.subscribe(open_token_ids)
        log.info(f"WebSocket subscribed to {len(open_token_ids)} open position tokens")

    last_resolution_check = 0
    last_equalization_check = 0
    last_report_date = None

    try:
        while True:
            now = datetime.now(timezone.utc)

            # Check if we should generate daily report (at end of window)
            today = now.strftime("%Y-%m-%d")
            if now.hour >= SCAN_END_HOUR_UTC and last_report_date != today:
                generate_daily_report(log)
                last_report_date = today

            # Check if within match window
            if not is_within_match_window():
                log.info("Outside match window, sleeping 5min...")
                time.sleep(OFF_WINDOW_SLEEP)
                continue

            # Run momentum scanner
            try:
                signals = scanner.scan()
            except Exception as e:
                log.error(f"Scanner error: {e}", exc_info=True)
                signals = []

            # Open positions for each signal
            for signal in signals:
                try:
                    # Use WebSocket price if available (more accurate than REST)
                    if signal.draw_token_id:
                        ws_price = ws_monitor.get_mid_price(signal.draw_token_id)
                        if ws_price and ws_price > 0:
                            log.info(
                                f"WS live price for draw: {ws_price:.3f} "
                                f"(REST: {signal.draw_price:.3f})"
                            )
                            signal.draw_price = ws_price

                    position_id = open_position(signal)
                    if position_id:
                        log.info(f"Position opened: {position_id}")
                        # Subscribe new position to WebSocket
                        if signal.draw_token_id:
                            ws_monitor.subscribe([signal.draw_token_id])
                except Exception as e:
                    log.error(f"Error opening position: {e}", exc_info=True)

            # Check for equalization exits (every EQUALIZATION_CHECK_INTERVAL)
            # This must be frequent to catch the equalizer quickly
            elapsed_since_eq_check = time.time() - last_equalization_check
            if elapsed_since_eq_check >= EQUALIZATION_CHECK_INTERVAL:
                try:
                    check_equalizations(fotmob, log)
                except Exception as e:
                    log.error(f"Equalization check error: {e}", exc_info=True)
                last_equalization_check = time.time()

            # Check for match resolutions (every RESOLUTION_CHECK_INTERVAL)
            # Handles expiry strategy and catches any positions missed by equalization
            elapsed_since_check = time.time() - last_resolution_check
            if elapsed_since_check >= RESOLUTION_CHECK_INTERVAL:
                try:
                    check_resolutions(fotmob, log)
                except Exception as e:
                    log.error(f"Resolution check error: {e}", exc_info=True)
                last_resolution_check = time.time()

            # Log near-misses periodically
            near_misses = scanner.get_near_misses()
            if near_misses:
                nm_file = DATA_DIR / "near_misses.json"
                ensure_data_dir()
                with open(nm_file, "w") as f:
                    json.dump(near_misses, f, indent=2)

            # Sleep until next scan
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        log.info("\nScheduler stopped by user")
        ws_monitor.stop()
        log.info("WebSocket monitor stopped")
        generate_daily_report(log)
        show_status()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Soccer Scheduler")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--report", action="store_true", help="Generate daily report and exit")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.report:
        log = setup_logging()
        generate_daily_report(log)
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
