"""
Draw Scheduler — Continuous Match Monitor (XGBoost + Survival)

Scans live matches every minute from min 55-75 for draw bets.
Uses XGBoost for go/no-go and Cox survival for entry timing.
Holds positions to match expiry (no equalization exit).

Usage:
    python -m src.Soccer.draw_scheduler
    python -m src.Soccer.draw_scheduler --status
"""

import time
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.DataProviders.FotMobProvider import FotMobProvider
from src.DataProviders.PolymarketSoccerProvider import PolymarketSoccerProvider
from src.Soccer.draw_scanner import DrawScanner, SCAN_INTERVAL, LEAGUES
from src.Soccer.draw_trader import (
    open_draw_position,
    resolve_draw_position,
    load_positions,
    save_positions,
    load_bankroll,
    show_status,
    DATA_DIR,
    STARTING_BANKROLL,
    DISCORD_WEBHOOK_URL,
)
from src.Utils.DrawdownManager import DrawdownManager
from src.Utils.AlertManager import AlertManager
from src.Polymarket.websocket_monitor import WebSocketPriceMonitor

# ===================== CONFIGURATION =====================
LOG_DIR = PROJECT_ROOT / "logs"

# Wide window: early PL kickoffs (12:30 UTC weekends) through late UCL/EL (~22:45+)
SCAN_START_HOUR_UTC = 10
SCAN_END_HOUR_UTC = 24  # midnight UTC

RESOLUTION_CHECK_INTERVAL = 2 * 60  # 2 minutes — check finished matches fast
OFF_WINDOW_SLEEP = 2 * 60           # 2 minutes when outside window
# =========================================================


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "draw_scheduler.log"

    log = logging.getLogger("draw_scheduler")
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    ))
    log.addHandler(fh)

    return log


def is_within_match_window() -> bool:
    now_utc = datetime.now(timezone.utc)
    if SCAN_END_HOUR_UTC >= 24:
        return now_utc.hour >= SCAN_START_HOUR_UTC  # runs until midnight
    return SCAN_START_HOUR_UTC <= now_utc.hour < SCAN_END_HOUR_UTC


def check_resolutions(fotmob: FotMobProvider, log: logging.Logger):
    """Check if any open positions' matches have finished. Resolve as draw or not."""
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
            continue

        status = str(details.get("status", "")).lower()
        is_finished = any(
            term in status for term in ["finished", "ft", "ended", "fulltime"]
        )

        if not is_finished:
            log.debug(
                f"Match {match_id} ({pos['home_team']} vs {pos['away_team']}) "
                f"still in progress"
            )
            continue

        home_score = details["home_score"]
        away_score = details["away_score"]
        final_score = f"{home_score}-{away_score}"
        was_draw = home_score == away_score

        log.info(
            f"Match FINISHED: {pos['home_team']} {final_score} {pos['away_team']} "
            f"-> {'DRAW' if was_draw else 'NO DRAW'}"
        )

        pnl = resolve_draw_position(pid, was_draw, final_score)
        log.info(f"Position {pid} resolved: P&L ${pnl:+.2f}")


def generate_daily_report(log: logging.Logger):
    positions = load_positions()
    bankroll = load_bankroll()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    today_resolved = [
        p for p in positions.values()
        if p["status"] != "open" and (p.get("exit_time") or "").startswith(today)
    ]
    today_opened = [
        p for p in positions.values()
        if (p.get("entry_time") or "").startswith(today)
    ]
    still_open = [p for p in positions.values() if p["status"] == "open"]

    if not today_resolved and not today_opened:
        log.info("No activity today")
        return

    wins = sum(1 for p in today_resolved if p.get("was_draw"))
    losses = len(today_resolved) - wins
    total_pnl = sum(p.get("pnl", 0) for p in today_resolved)

    report_lines = [
        f"DAILY REPORT (XGBoost + Survival Draw Model) - {today}",
        "=" * 50,
        f"Bankroll: ${bankroll:.2f}",
        f"Opened: {len(today_opened)} | Resolved: {len(today_resolved)}",
        f"Record: {wins}W-{losses}L | Day P&L: ${total_pnl:+.2f}",
        f"Still open: {len(still_open)}",
        "",
    ]

    for p in today_resolved:
        result = "WIN" if p.get("was_draw") else "LOSS"
        report_lines.append(
            f"  {result}: [{p['league']}] {p['home_team']} vs {p['away_team']} "
            f"({p.get('final_score', '?')}) | "
            f"Entry {p['entry_draw_price']:.1%} P(draw)={p['xgb_draw_prob']:.3f} | "
            f"${p.get('pnl', 0):+.2f}"
        )

    report_text = "\n".join(report_lines)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report_file = DATA_DIR / f"report_{today}.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    log.info(f"\n{report_text}")

    am = AlertManager(data_dir=DATA_DIR, webhook_url=DISCORD_WEBHOOK_URL)
    am.daily_summary(
        total_pnl=total_pnl,
        wins=wins,
        losses=losses,
        open_positions=len(still_open),
        data={"bankroll": bankroll, "model": "XGBoost + Survival"},
    )


def run_scheduler():
    log = setup_logging()

    log.info("=" * 60)
    log.info("SOCCER DRAW BETTING — XGBoost + Survival")
    log.info("=" * 60)
    from src.Soccer.draw_scanner import MIN_MINUTE, MAX_MINUTE, MIN_EDGE
    log.info(f"Strategy: Hold to expiry")
    log.info(f"Leagues: {', '.join(LEAGUES)}")
    log.info(f"Scan window: min {MIN_MINUTE}-{MAX_MINUTE} | Edge threshold: {MIN_EDGE:.0%}")
    log.info(f"Scan interval: {SCAN_INTERVAL}s")
    log.info(f"Match hours: {SCAN_START_HOUR_UTC}:00 - {SCAN_END_HOUR_UTC}:00 UTC")
    log.info(f"Bankroll: ${load_bankroll():.2f}")
    log.info("=" * 60)

    fotmob = FotMobProvider(leagues=LEAGUES)
    polymarket = PolymarketSoccerProvider(leagues=LEAGUES)

    # WebSocket for real-time PM prices
    def on_ws_price(asset_id, bid, ask):
        log.debug(f"WS: {asset_id[:20]}... bid={bid:.3f} ask={ask:.3f}")

    ws_monitor = WebSocketPriceMonitor(on_price_update=on_ws_price, logger=log)
    ws_monitor.start()
    log.info("WebSocket price monitor started")

    scanner = DrawScanner(fotmob, polymarket, ws_monitor=ws_monitor, logger_override=log)

    last_resolution_check = 0
    last_report_date = None

    try:
        while True:
            now = datetime.now(timezone.utc)

            # Daily report at end of day (after 23:30 UTC)
            today = now.strftime("%Y-%m-%d")
            if now.hour >= 23 and now.minute >= 30 and last_report_date != today:
                generate_daily_report(log)
                last_report_date = today

            if not is_within_match_window():
                log.debug("Outside match window")
                time.sleep(OFF_WINDOW_SLEEP)
                continue

            # Scan for signals
            try:
                signals = scanner.scan()
            except Exception as e:
                log.error(f"Scanner error: {e}", exc_info=True)
                signals = []

            # Open positions
            for signal in signals:
                try:
                    position_id = open_draw_position(signal)
                    if position_id:
                        log.info(f"Position opened: {position_id}")
                except Exception as e:
                    log.error(f"Error opening position: {e}", exc_info=True)

            # Check resolutions
            elapsed = time.time() - last_resolution_check
            if elapsed >= RESOLUTION_CHECK_INTERVAL:
                try:
                    check_resolutions(fotmob, log)
                except Exception as e:
                    log.error(f"Resolution check error: {e}", exc_info=True)
                last_resolution_check = time.time()

            # Save near-misses
            near_misses = scanner.get_near_misses()
            if near_misses:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                nm_file = DATA_DIR / "near_misses.json"
                with open(nm_file, "w") as f:
                    json.dump(near_misses, f, indent=2)

            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        log.info("\nScheduler stopped by user")
        ws_monitor.stop()
        generate_daily_report(log)
        show_status()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Draw Scheduler (XGBoost + Survival)")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--report", action="store_true", help="Generate daily report")

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
