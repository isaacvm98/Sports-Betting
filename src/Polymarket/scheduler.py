"""
Polymarket Paper Trading Scheduler

Smart scheduling - places bets 10 minutes before each game starts.
Monitors positions and closes after games end.

Usage:
    python -m src.Polymarket.scheduler

    # Or run in background (Windows):
    start /B python -m src.Polymarket.scheduler > logs/scheduler.log 2>&1
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import numpy as np
import pandas as pd
import xgboost as xgb

from src.DataProviders.PolymarketOddsProvider import PolymarketOddsProvider
from src.Utils.tools import get_json_data, to_data_frame
from src.Utils.Kelly_Criterion import calculate_kelly_criterion
from src.Utils.Dictionaries import team_index_current
from src.Polymarket.websocket_monitor import WebSocketPriceMonitor
from src.Polymarket.paper_trader import (
    load_positions,
    save_positions,
    log_trade,
    DATA_DIR,
    get_model_predictions,
    american_odds_to_probability,
    get_take_profit_threshold,
    get_underdog_take_profit_threshold,
    is_underdog_position,
    calculate_position_pnl,
    calculate_exit_pnl,
    load_bankroll,
    save_bankroll,
    STOP_LOSS_PCT,
    EARLY_EXIT_ENABLED,
    UNDERDOG_TAKE_PROFIT_ENABLED,
    UNDERDOG_TAKE_PROFIT_MAX_ENTRY,
    STARTING_BANKROLL,
    DISCORD_WEBHOOK_URL,
)
from src.Utils.DrawdownManager import DrawdownManager
from src.Utils.AlertManager import AlertManager, AlertType
from src.Polymarket.price_logger import get_price_logger
from src.DataProviders.espn_wp_logger import get_espn_wp_logger

try:
    from src.DataProviders.ESPNProvider import ESPNProvider
    _ESPN_AVAILABLE = True
except ImportError:
    _ESPN_AVAILABLE = False

# Configuration
MINUTES_BEFORE_GAME = 10   # Place bets X minutes before game
MONITOR_INTERVAL = 5    # Monitor every X minutes
LOCAL_TIMEZONE_OFFSET = -5  # EST = UTC-5 (adjust for your timezone)
MAX_KELLY_PCT = 15      # Maximum Kelly % per position (caps exposure to bad info)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "scheduler.log"

# Polymarket API
GAMMA_API_URL = "https://gamma-api.polymarket.com"
NBA_SERIES_ID = "10345"
GAMES_TAG_ID = "100639"


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


def get_todays_games_with_times():
    """Fetch today's games with their start times from Polymarket."""
    try:
        params = {
            "series_id": NBA_SERIES_ID,
            "tag_id": GAMES_TAG_ID,
            "active": "true",
            "closed": "false",
            "order": "startTime",
            "ascending": "true",
            "limit": 50
        }

        response = requests.get(f"{GAMMA_API_URL}/events", params=params, timeout=15)
        response.raise_for_status()
        events = response.json()

        now = datetime.now(timezone.utc)
        today = now.date()
        tomorrow = today + timedelta(days=1)

        games = []
        for event in events:
            end_date_str = event.get("endDate", "")
            if not end_date_str:
                continue

            try:
                # endDate is the game time
                game_time = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                game_date = game_time.date()

                # Only include games starting within the next 18 hours
                # This catches today's games without pulling in tomorrow's full slate
                hours_until_game = (game_time - now).total_seconds() / 3600
                if hours_until_game > 18 or hours_until_game < 0:
                    continue

                # Skip games that already started
                if game_time <= now:
                    continue

                title = event.get("title", "")
                games.append({
                    "title": title,
                    "event": event,
                    "game_time": game_time,
                    "bet_time": game_time - timedelta(minutes=MINUTES_BEFORE_GAME)
                })

            except (ValueError, TypeError):
                continue

        # Sort by game time
        games.sort(key=lambda x: x["game_time"])
        return games

    except Exception as e:
        logging.error(f"Error fetching games: {e}")
        return []


def init_single_game(event, game_time, logger, alert_mgr=None, ws_monitor=None):
    """Initialize position for a single game.

    Args:
        event: Polymarket event data
        game_time: Game start time (datetime)
        logger: Logger instance
        alert_mgr: AlertManager for Discord notifications
        ws_monitor: WebSocketPriceMonitor for real-time prices
    """
    title = event.get("title", "")
    logger.info(f"Initializing position for: {title}")

    # Get current odds for this game
    provider = PolymarketOddsProvider()
    all_odds = provider.get_odds()

    # Find matching game
    matching_key = None
    for game_key in all_odds.keys():
        # Match by team names in title
        home, away = game_key.split(":")
        if home.split()[-1] in title or away.split()[-1] in title:
            matching_key = game_key
            break

    if not matching_key:
        logger.warning(f"Could not find odds for: {title}")
        return None

    game_odds = all_odds[matching_key]
    home_team, away_team = matching_key.split(":")

    home_ml = game_odds[home_team]['money_line_odds']
    away_ml = game_odds[away_team]['money_line_odds']
    ou_line = game_odds['under_over_odds']

    # Get token IDs for WebSocket price monitoring
    home_token_id = game_odds.get('home_token_id')
    away_token_id = game_odds.get('away_token_id')

    if home_ml is None or away_ml is None:
        logger.warning(f"Invalid odds for: {title}")
        return None

    # Get model prediction
    predictions = get_model_predictions([matching_key], {matching_key: game_odds})

    if matching_key not in predictions:
        logger.warning(f"No model prediction for: {title}")
        return None

    pred = predictions[matching_key]
    model_home_prob = pred['home_prob']
    model_away_prob = pred['away_prob']

    # Market probabilities
    market_home_prob = american_odds_to_probability(home_ml)
    market_away_prob = american_odds_to_probability(away_ml)

    # Calculate edge and Kelly
    home_edge = model_home_prob - market_home_prob
    away_edge = model_away_prob - market_away_prob

    try:
        kelly_home = calculate_kelly_criterion(home_ml, model_home_prob)
        kelly_away = calculate_kelly_criterion(away_ml, model_away_prob)
    except:
        kelly_home = kelly_away = 0

    # Determine bet side and apply Kelly cap
    bet_side = None
    bet_kelly = 0
    kelly_raw = 0  # Store uncapped value for logging
    if kelly_home > 0 and kelly_home >= kelly_away:
        bet_side = "home"
        kelly_raw = kelly_home
        bet_kelly = min(kelly_home, MAX_KELLY_PCT)
    elif kelly_away > 0:
        bet_side = "away"
        kelly_raw = kelly_away
        bet_kelly = min(kelly_away, MAX_KELLY_PCT)

    # Create position
    positions = load_positions()
    position_id = f"{matching_key}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # Snapshot bet_amount at entry time for accurate P&L calculation
    entry_bankroll = load_bankroll()
    entry_bet_amount = (bet_kelly / 100) * entry_bankroll if bet_kelly and bet_side else 0

    position = {
        "game_key": matching_key,
        "home_team": home_team,
        "away_team": away_team,
        "home_token_id": home_token_id,
        "away_token_id": away_token_id,
        "game_time": game_time.isoformat(),
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "entry_home_prob": market_home_prob,
        "entry_away_prob": market_away_prob,
        "entry_home_odds": home_ml,
        "entry_away_odds": away_ml,
        "ou_line": ou_line,
        "model_home_prob": model_home_prob,
        "model_away_prob": model_away_prob,
        "home_edge": home_edge,
        "away_edge": away_edge,
        "kelly_home": kelly_home,
        "kelly_away": kelly_away,
        "bet_side": bet_side,
        "bet_kelly": bet_kelly,
        "kelly_raw": kelly_raw,
        "bet_amount": round(entry_bet_amount, 2),
        "entry_bankroll": round(entry_bankroll, 2),
        "status": "open",
        "exit_time": None,
        "exit_reason": None,
    }

    positions[position_id] = position
    save_positions(positions)

    # Log
    local_game_time = game_time + timedelta(hours=LOCAL_TIMEZONE_OFFSET)
    logger.info(f"  Game time: {local_game_time.strftime('%I:%M %p')} local")
    logger.info(f"  Market: Home {market_home_prob:.1%} | Away {market_away_prob:.1%}")
    logger.info(f"  Model:  Home {model_home_prob:.1%} | Away {model_away_prob:.1%}")
    logger.info(f"  Edge:   Home {home_edge:+.1%} | Away {away_edge:+.1%}")

    if bet_side:
        if kelly_raw > MAX_KELLY_PCT:
            logger.info(f"  >>> BET: {bet_side.upper()} ({bet_kelly:.1f}% Kelly, capped from {kelly_raw:.1f}%)")
        else:
            logger.info(f"  >>> BET: {bet_side.upper()} ({bet_kelly:.1f}% Kelly)")
    else:
        logger.info(f"  >>> NO BET (no edge)")

    log_trade({
        "type": "ENTRY",
        "time": datetime.now(timezone.utc).isoformat(),
        "position_id": position_id,
        "game": f"{away_team} @ {home_team}",
        "game_time": game_time.isoformat(),
        "model_home_prob": model_home_prob,
        "market_home_prob": market_home_prob,
        "bet_side": bet_side,
        "kelly": bet_kelly
    })

    # Send bet placement alert to Discord
    if alert_mgr and bet_side:
        game = f"{away_team} @ {home_team}"
        edge = home_edge if bet_side == "home" else away_edge
        alert_mgr.info(f"NEW BET: {game} - {bet_side.upper()}", {
            "kelly": f"{bet_kelly:.1f}%",
            "edge": f"{edge:+.1%}",
            "system": "NBA",
        })

    # Subscribe to WebSocket for real-time price monitoring
    if ws_monitor and bet_side:
        token_id = home_token_id if bet_side == "home" else away_token_id
        if token_id:
            ws_monitor.subscribe([token_id])
            logger.info(f"  Subscribed to WebSocket for {bet_side} token")

    return position_id


def get_live_prices(home_team, away_team, logger, game_time=None, market_id=None):
    """Fetch live in-game prices from Polymarket for a specific game.

    Args:
        home_team: Full home team name
        away_team: Full away team name
        logger: Logger instance
        game_time: ISO format game time string (used to match correct event)
        market_id: Optional market ID for direct lookup

    Returns:
        (home_prob, away_prob, is_closed, market_id) or (None, None, None, None) if not found
    """
    try:
        # Query active markets - NOTE: Don't use tag_id as it excludes in-progress games
        # Use closed=false to get games that haven't resolved yet
        params = {
            "series_id": NBA_SERIES_ID,
            "active": "true",
            "closed": "false",
            "limit": 100
        }
        response = requests.get(f"{GAMMA_API_URL}/events", params=params, timeout=15)
        response.raise_for_status()
        events = response.json()

        home_short = home_team.split()[-1]
        away_short = away_team.split()[-1]

        # Parse game_time for date matching
        game_date = None
        if game_time:
            try:
                from datetime import datetime
                if isinstance(game_time, str):
                    game_date = datetime.fromisoformat(game_time.replace('Z', '+00:00')).date()
            except:
                pass

        # Find matching events (may be multiple for same teams)
        matching_events = []
        for event in events:
            title = event.get("title", "")
            if home_short in title and away_short in title:
                matching_events.append(event)

        # If we have game_time, filter to event with matching end date
        if game_date and len(matching_events) > 1:
            for event in matching_events:
                end_date_str = event.get("endDate", "")
                if end_date_str:
                    try:
                        from datetime import datetime
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).date()
                        if end_date == game_date:
                            matching_events = [event]
                            break
                    except:
                        continue

        # Process the first (or only) matching event
        for event in matching_events[:1]:
            markets = event.get("markets", [])
            for market in markets:
                question = market.get("question", "")
                # Match full game moneyline only (exclude 1H, O/U, Spread)
                if " vs" in question.lower() and "O/U" not in question and "Spread" not in question and "1H" not in question:
                    # If we have a specific market_id, verify it matches
                    found_market_id = market.get("id")
                    if market_id and str(found_market_id) != str(market_id):
                        continue

                    is_closed = market.get("closed", False)
                    outcome_prices = market.get("outcomePrices", "")
                    outcomes = market.get("outcomes", "")

                    if isinstance(outcome_prices, str):
                        import json as json_module
                        try:
                            outcome_prices = json_module.loads(outcome_prices)
                        except:
                            continue

                    if isinstance(outcomes, str):
                        import json as json_module
                        try:
                            outcomes = json_module.loads(outcomes)
                        except:
                            outcomes = []

                    if len(outcome_prices) >= 2:
                        # Map prices to teams using outcomes array
                        # outcomes = ['Team1', 'Team2'], prices = [price1, price2]
                        home_prob = None
                        away_prob = None

                        # Match using both city and team name for robustness
                        # e.g., "Brooklyn Nets" → check "Nets" and "Brooklyn"
                        home_words = [w.lower() for w in home_team.split() if len(w) > 2]
                        away_words = [w.lower() for w in away_team.split() if len(w) > 2]

                        for i, outcome in enumerate(outcomes):
                            outcome_lower = outcome.lower()
                            if any(w in outcome_lower for w in home_words):
                                home_prob = float(outcome_prices[i])
                            elif any(w in outcome_lower for w in away_words):
                                away_prob = float(outcome_prices[i])

                        # Only use prices if BOTH teams were matched
                        if home_prob is None or away_prob is None:
                            logger.warning(f"Could not match outcomes {outcomes} to teams {home_team}/{away_team} - skipping")
                            continue

                        logger.debug(f"Live prices for {away_short}@{home_short}: away={away_prob:.3f}, home={home_prob:.3f}, closed={is_closed}, market_id={found_market_id}")
                        return home_prob, away_prob, is_closed, found_market_id
            break
    except Exception as e:
        logger.error(f"Error fetching live prices: {e}")

    return None, None, None, None


def check_market_resolved(game_key, home_team, away_team, logger, game_time=None):
    """Check if a market has resolved by querying the API for closed markets.

    Args:
        game_key: Game identifier string
        home_team: Full home team name
        away_team: Full away team name
        logger: Logger instance
        game_time: Optional ISO format game time for date matching

    Returns:
        (resolved, winner) - resolved is True if market is done, winner is 'home' or 'away' or None
    """
    try:
        # Query closed markets - resolved games are no longer "active"
        params = {
            "series_id": NBA_SERIES_ID,
            "closed": "true",
            "limit": 100,
            "order": "endDate",
            "ascending": "false"  # Most recent first
        }
        response = requests.get(f"{GAMMA_API_URL}/events", params=params, timeout=15)
        response.raise_for_status()
        events = response.json()

        home_short = home_team.split()[-1]
        away_short = away_team.split()[-1]
        logger.debug(f"Checking resolution for {away_short} @ {home_short} (API fallback)...")

        # Parse game_time for date matching
        game_date = None
        if game_time:
            try:
                if isinstance(game_time, str):
                    game_date = datetime.fromisoformat(game_time.replace('Z', '+00:00')).date()
            except:
                pass

        # Find matching events (filter by date if provided)
        matching_events = []
        for event in events:
            title = event.get("title", "")
            if home_short in title and away_short in title:
                matching_events.append(event)

        # If we have game_date, filter to event with matching end date
        if game_date and len(matching_events) > 1:
            for event in matching_events:
                end_date_str = event.get("endDate", "")
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).date()
                        if end_date == game_date:
                            matching_events = [event]
                            break
                    except:
                        continue

        for event in matching_events[:1]:
            title = event.get("title", "")
            markets = event.get("markets", [])
            for market in markets:
                question = market.get("question", "")
                # Match full game moneyline only (exclude 1H, O/U, Spread)
                if " vs" in question.lower() and "O/U" not in question and "Spread" not in question and "1H" not in question:
                    is_closed = market.get("closed", False)
                    if is_closed:
                        logger.info(f"Found closed market: {title}")
                        outcome_prices = market.get("outcomePrices", "")
                        outcomes = market.get("outcomes", "")

                        if isinstance(outcome_prices, str):
                            try:
                                outcome_prices = json.loads(outcome_prices)
                            except:
                                logger.warning(f"Failed to parse outcome prices: {outcome_prices}")
                                continue

                        if isinstance(outcomes, str):
                            try:
                                outcomes = json.loads(outcomes)
                            except:
                                outcomes = []

                        if len(outcome_prices) >= 2:
                            # Map prices to teams using outcomes array
                            home_prob = None
                            away_prob = None

                            for i, outcome in enumerate(outcomes):
                                if home_short in outcome:
                                    home_prob = float(outcome_prices[i])
                                elif away_short in outcome:
                                    away_prob = float(outcome_prices[i])

                            # Fallback if outcomes didn't match
                            if home_prob is None or away_prob is None:
                                away_prob = float(outcome_prices[0])
                                home_prob = float(outcome_prices[1])

                            logger.debug(f"Outcome prices - Away: {away_prob}, Home: {home_prob}")

                            # Resolved markets have ~1.0 for winner, ~0.0 for loser
                            if home_prob > 0.95:
                                logger.info(f"Market resolved: HOME won")
                                return True, 'home'
                            elif away_prob > 0.95:
                                logger.info(f"Market resolved: AWAY won")
                                return True, 'away'
                            else:
                                logger.warning(f"Market closed but no clear winner: home={home_prob}, away={away_prob}")
                    break
            break

    except Exception as e:
        logger.error(f"Error checking resolved markets: {e}")

    return False, None


def _record_resolution(position, position_id, winner, pnl, logger):
    """Record a position resolution and update drawdown manager.

    Args:
        position: The position dict (will be modified)
        position_id: Position identifier
        winner: 'home' or 'away'
        pnl: Dollar P&L amount
        logger: Logger instance
    """
    bet_side = position.get('bet_side')
    game = f"{position.get('away_team')} @ {position.get('home_team')}"
    won = (bet_side == winner) if bet_side else False

    # Update bankroll
    bankroll = load_bankroll()
    new_bankroll = bankroll + pnl
    save_bankroll(new_bankroll)

    # Send resolution alert (file + Discord)
    try:
        am = AlertManager(
            data_dir=DATA_DIR,
            enable_console=False,
            webhook_url=DISCORD_WEBHOOK_URL,
            webhook_platform="discord"
        )
        am.resolution(game, won, pnl, {
            "bet_side": bet_side,
            "winner": winner,
            "bankroll": new_bankroll,
        })
    except Exception as e:
        logger.warning(f"Alert error: {e}")

    # Record in drawdown manager (tracking only, no halt)
    try:
        dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
        dm.record_pnl(pnl, position_id, new_bankroll)
    except Exception as e:
        logger.error(f"Error recording to drawdown manager: {e}")


def monitor_positions(logger, ws_monitor=None):
    """Monitor open positions for market resolution.

    NOTE: Early exits (stop-loss/take-profit) are DISABLED by default.
    Analysis shows positions running to resolution outperform early exits.
    Set EARLY_EXIT_ENABLED = True in paper_trader.py to re-enable exit signals.

    Args:
        logger: Logger instance
        ws_monitor: Optional WebSocketPriceMonitor for real-time prices
    """
    positions = load_positions()
    open_positions = {k: v for k, v in positions.items() if v['status'] == 'open'}

    if not open_positions:
        return

    # Get current odds for active markets
    provider = PolymarketOddsProvider()
    current_odds = provider.get_odds()

    # Fetch ESPN win probabilities once per cycle for logging
    espn_games = {}
    if _ESPN_AVAILABLE:
        try:
            espn = ESPNProvider('nba')
            espn.CACHE_TTL = 0  # Fresh data for logging
            espn_games = espn.get_all_live_win_probabilities()
        except Exception as e:
            logger.debug(f"ESPN fetch skipped: {e}")

    for position_id, position in open_positions.items():
        game_key = position['game_key']
        home_team = position['home_team']
        away_team = position['away_team']
        bet_side = position.get('bet_side')

        # Check if market is still in pre-game odds
        if game_key not in current_odds:
            # Game has started - try WebSocket prices first, then fall back to REST API
            current_home_prob = None
            current_away_prob = None
            is_closed = False
            found_market_id = None
            stored_market_id = position.get('market_id')
            game_time = position.get('game_time')

            # Try WebSocket prices first (faster, real-time)
            ws_resolved = False
            ws_winner = None
            if ws_monitor and bet_side:
                token_id = position.get('home_token_id') if bet_side == 'home' else position.get('away_token_id')
                if token_id:
                    ws_price = ws_monitor.get_price(token_id)
                    if ws_price and ws_price.get('mid') is not None:
                        # WebSocket gives us the price for our bet side token
                        bet_price = ws_price.get('mid')

                        # Check if market resolved via WebSocket (price near 0 or 1)
                        if bet_price >= 0.95 or bet_price <= 0.05:
                            # Prices indicate resolution - verify with API that market is officially closed
                            logger.info(f"WebSocket prices at resolution level for {away_team}@{home_team} ({bet_side}={bet_price:.3f}) - verifying with API...")
                            api_resolved, api_winner = check_market_resolved(game_key, home_team, away_team, logger, game_time=game_time)

                            if api_resolved and api_winner:
                                ws_resolved = True
                                ws_winner = api_winner
                                logger.info(f"API confirmed resolution: {away_team}@{home_team} -> {api_winner.upper()} won")
                            else:
                                # Market not officially closed yet, just update prices
                                logger.debug(f"Market not officially closed yet, tracking prices")
                                if bet_side == 'home':
                                    current_home_prob = bet_price
                                    current_away_prob = 1 - bet_price
                                else:
                                    current_away_prob = bet_price
                                    current_home_prob = 1 - bet_price
                        else:
                            # Game still in progress
                            if bet_side == 'home':
                                current_home_prob = bet_price
                                current_away_prob = 1 - bet_price
                            else:
                                current_away_prob = bet_price
                                current_home_prob = 1 - bet_price
                            logger.debug(f"WebSocket prices: {away_team}@{home_team} -> {bet_side}={bet_price:.3f}")

            # Log price observation (WebSocket path)
            if current_home_prob is not None:
                game_date = position.get('game_time', '')[:10]
                get_price_logger().log(game_date, home_team, away_team, current_home_prob, current_away_prob, source="ws")

            # Log ESPN win probability alongside market price
            if espn_games:
                espn_key = f"{home_team}:{away_team}"
                espn_game = espn_games.get(espn_key)
                if espn_game and espn_game.get('home_win_prob') is not None:
                    game_date = position.get('game_time', '')[:10]
                    get_espn_wp_logger().log(
                        game_date, home_team, away_team,
                        espn_game['home_win_prob'], espn_game['away_win_prob'],
                        home_score=espn_game.get('home_score'),
                        away_score=espn_game.get('away_score'),
                        period=espn_game.get('period'),
                        clock=espn_game.get('clock'),
                        source="scheduler",
                    )

            # If WebSocket + API confirmed resolution, process it
            if ws_resolved and ws_winner:
                won = (bet_side == ws_winner) if bet_side else False
                pnl = calculate_position_pnl(position, won)

                position['status'] = 'resolved'
                position['exit_time'] = datetime.now(timezone.utc).isoformat()
                position['exit_reason'] = 'market_resolved'
                position['winner'] = ws_winner
                position['won'] = won
                position['pnl'] = pnl

                # Update bankroll and record in drawdown manager
                if pnl != 0:
                    _record_resolution(position, position_id, ws_winner, pnl, logger)

                result_str = "WON" if won else "LOST"
                logger.info(f"RESOLVED (WebSocket): {away_team} @ {home_team} - {ws_winner.upper()} won | We {result_str} ${abs(pnl):.2f}")

                log_trade({
                    "type": "RESOLVED",
                    "time": datetime.now(timezone.utc).isoformat(),
                    "position_id": position_id,
                    "game": f"{away_team} @ {home_team}",
                    "winner": ws_winner,
                    "bet_side": bet_side,
                    "won": won,
                    "pnl": pnl,
                    "source": "websocket",
                    "entry_edge": position.get(f"{bet_side}_edge", 0) if bet_side else 0,
                    "max_profit_pct": position.get('max_profit_pct', 0),
                    "max_drawdown_pct": position.get('max_drawdown_pct', 0),
                })
                continue

            # Fall back to REST API if WebSocket didn't have prices
            # But skip if WebSocket already provided a recent update (avoid overwriting good data)
            if current_home_prob is None:
                last_ws = position.get('last_ws_update')
                if last_ws:
                    try:
                        ws_age = (datetime.now(timezone.utc) - datetime.fromisoformat(last_ws.replace('Z', '+00:00'))).total_seconds()
                        if ws_age < 300:  # WebSocket updated within 5 minutes - trust it
                            logger.debug(f"Skipping REST fallback for {away_team} @ {home_team} - WebSocket data is {ws_age:.0f}s old")
                            continue
                    except (ValueError, TypeError):
                        pass

                logger.info(f"Game in progress: {away_team} @ {home_team} - fetching live prices via REST...")
                current_home_prob, current_away_prob, is_closed, found_market_id = get_live_prices(
                    home_team, away_team, logger, game_time=game_time, market_id=stored_market_id
                )
            else:
                logger.debug(f"Using WebSocket prices for {away_team} @ {home_team}")

            # Store market_id if we found it and don't have one
            if found_market_id and not stored_market_id:
                position['market_id'] = found_market_id

            if current_home_prob is None:
                # Couldn't get prices - check if resolved via API
                logger.info(f"Could not get live prices - checking if resolved via API...")
                resolved, winner = check_market_resolved(game_key, home_team, away_team, logger, game_time=game_time)

                if resolved and winner:
                    # Market resolved - calculate P&L
                    won = (bet_side == winner) if bet_side else False
                    pnl = calculate_position_pnl(position, won)

                    position['status'] = 'resolved'
                    position['exit_time'] = datetime.now(timezone.utc).isoformat()
                    position['exit_reason'] = 'market_resolved'
                    position['winner'] = winner
                    position['won'] = won
                    position['pnl'] = pnl

                    # Update bankroll and record in drawdown manager
                    if pnl != 0:
                        _record_resolution(position, position_id, winner, pnl, logger)

                    result_str = "WON" if won else "LOST"
                    logger.info(f"RESOLVED: {away_team} @ {home_team} - {winner.upper()} won | We {result_str} ${abs(pnl):.2f}")

                    log_trade({
                        "type": "RESOLVED",
                        "time": datetime.now(timezone.utc).isoformat(),
                        "position_id": position_id,
                        "game": f"{away_team} @ {home_team}",
                        "winner": winner,
                        "bet_side": bet_side,
                        "won": won,
                        "pnl": pnl,
                        "entry_edge": position.get(f"{bet_side}_edge", 0) if bet_side else 0,
                        "max_profit_pct": position.get('max_profit_pct', 0),
                        "max_drawdown_pct": position.get('max_drawdown_pct', 0),
                    })
                else:
                    logger.info(f"Market not yet resolved: {away_team} @ {home_team}")
                continue

            # Check if market is closed (game ended) with a clear winner
            if is_closed:
                if current_home_prob > 0.95:
                    winner = 'home'
                elif current_away_prob > 0.95:
                    winner = 'away'
                else:
                    winner = None

                if winner:
                    won = (bet_side == winner) if bet_side else False
                    pnl = calculate_position_pnl(position, won)

                    position['status'] = 'resolved'
                    position['exit_time'] = datetime.now(timezone.utc).isoformat()
                    position['exit_reason'] = 'market_resolved'
                    position['winner'] = winner
                    position['won'] = won
                    position['pnl'] = pnl

                    # Update bankroll and record in drawdown manager
                    if pnl != 0:
                        _record_resolution(position, position_id, winner, pnl, logger)

                    result_str = "WON" if won else "LOST"
                    logger.info(f"RESOLVED: {away_team} @ {home_team} - {winner.upper()} won | We {result_str} ${abs(pnl):.2f}")

                    log_trade({
                        "type": "RESOLVED",
                        "time": datetime.now(timezone.utc).isoformat(),
                        "position_id": position_id,
                        "game": f"{away_team} @ {home_team}",
                        "winner": winner,
                        "bet_side": bet_side,
                        "won": won,
                        "pnl": pnl,
                        "entry_edge": position.get(f"{bet_side}_edge", 0) if bet_side else 0,
                        "max_profit_pct": position.get('max_profit_pct', 0),
                        "max_drawdown_pct": position.get('max_drawdown_pct', 0),
                    })
                    continue

            # Got live prices - log them and track max profit/drawdown
            logger.info(f"Live odds: Home {current_home_prob:.1%} | Away {current_away_prob:.1%}")

            # Log price observation (REST path)
            game_date = position.get('game_time', '')[:10]
            get_price_logger().log(game_date, home_team, away_team, current_home_prob, current_away_prob, source="rest")

            # Track max profit/drawdown during position lifetime for analytics
            if bet_side and current_home_prob is not None:
                entry_prob = position['entry_home_prob'] if bet_side == 'home' else position['entry_away_prob']
                current_prob = current_home_prob if bet_side == 'home' else current_away_prob
                current_change = (current_prob - entry_prob) / entry_prob if entry_prob > 0 else 0

                position['current_price_change'] = current_change
                if 'max_profit_pct' not in position:
                    position['max_profit_pct'] = current_change
                    position['max_drawdown_pct'] = current_change
                else:
                    position['max_profit_pct'] = max(position['max_profit_pct'], current_change)
                    position['max_drawdown_pct'] = min(position['max_drawdown_pct'], current_change)
            continue

        # Market still in pre-game - use OddsProvider prices
        game_odds = current_odds[game_key]
        current_home_ml = game_odds[home_team]['money_line_odds']

        if current_home_ml is None:
            continue

        current_home_prob = american_odds_to_probability(current_home_ml)
        current_away_prob = 1 - current_home_prob

        # Log price observation (pre-game path)
        game_date = position.get('game_time', '')[:10]
        get_price_logger().log(game_date, home_team, away_team, current_home_prob, current_away_prob, source="pregame")

        entry_home_prob = position['entry_home_prob']
        entry_away_prob = position['entry_away_prob']

        # Calculate change
        home_change = (current_home_prob - entry_home_prob) / entry_home_prob if entry_home_prob > 0 else 0
        away_change = (current_away_prob - entry_away_prob) / entry_away_prob if entry_away_prob > 0 else 0

        # Track max profit/drawdown during position lifetime for analytics
        if bet_side:
            current_change = home_change if bet_side == 'home' else away_change
            position['current_price_change'] = current_change
            if 'max_profit_pct' not in position:
                position['max_profit_pct'] = current_change
                position['max_drawdown_pct'] = current_change
            else:
                position['max_profit_pct'] = max(position['max_profit_pct'], current_change)
                position['max_drawdown_pct'] = min(position['max_drawdown_pct'], current_change)

        # === EXIT LOGIC ===
        # Two independent systems:
        # 1. Legacy full early exits (stop-loss + take-profit on all bets) - EARLY_EXIT_ENABLED
        # 2. Underdog-only take-profit (no stop-loss) - UNDERDOG_TAKE_PROFIT_ENABLED

        exit_triggered = False
        exit_reason = None

        if EARLY_EXIT_ENABLED:
            # Legacy: stop-loss and take-profit on ALL positions
            entry_prob = entry_home_prob if bet_side == 'home' else entry_away_prob
            take_profit_pct = get_take_profit_threshold(None, entry_prob=entry_prob)

            if bet_side == 'home':
                if home_change <= -STOP_LOSS_PCT:
                    exit_triggered = True
                    exit_reason = f"STOP_LOSS (Home dropped {home_change:.1%})"
                elif take_profit_pct and home_change >= take_profit_pct:
                    exit_triggered = True
                    exit_reason = f"TAKE_PROFIT (Home up {home_change:.1%})"
            elif bet_side == 'away':
                if away_change <= -STOP_LOSS_PCT:
                    exit_triggered = True
                    exit_reason = f"STOP_LOSS (Away dropped {away_change:.1%})"
                elif take_profit_pct and away_change >= take_profit_pct:
                    exit_triggered = True
                    exit_reason = f"TAKE_PROFIT (Away up {away_change:.1%})"

        elif UNDERDOG_TAKE_PROFIT_ENABLED and bet_side:
            # Underdog-only take-profit (no stop-loss)
            entry_prob = entry_home_prob if bet_side == 'home' else entry_away_prob

            if entry_prob < UNDERDOG_TAKE_PROFIT_MAX_ENTRY:
                tp_threshold = get_underdog_take_profit_threshold(entry_prob)

                if tp_threshold is not None:
                    price_change_for_side = home_change if bet_side == 'home' else away_change

                    if price_change_for_side >= tp_threshold:
                        exit_triggered = True
                        exit_reason = (
                            f"UNDERDOG_TAKE_PROFIT ({bet_side.title()} up "
                            f"{price_change_for_side:.1%}, threshold {tp_threshold:.0%}, "
                            f"entry {entry_prob:.1%})"
                        )
                    else:
                        logger.debug(
                            f"Underdog TP tracking: {away_team} @ {home_team} | "
                            f"{bet_side} {price_change_for_side:+.1%} / {tp_threshold:.0%} threshold"
                        )
            else:
                # Favorites / moderate underdogs (>=35%): hold to resolution
                logger.debug(
                    f"Holding (not underdog TP eligible): {away_team} @ {home_team} | "
                    f"entry_prob={entry_prob:.1%}, Change: Home {home_change:+.1%}, Away {away_change:+.1%}"
                )

        if exit_triggered:
            price_change = home_change if bet_side == 'home' else away_change
            pnl = calculate_exit_pnl(position, price_change)

            # Update bankroll and record in drawdown manager
            bankroll = load_bankroll()
            new_bankroll = bankroll + pnl
            save_bankroll(new_bankroll)

            # Record in drawdown manager (tracking only, no halt)
            try:
                dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
                dm.record_pnl(pnl, position_id, new_bankroll)
            except Exception as e:
                logger.error(f"Error recording to drawdown manager: {e}")

            position['status'] = 'closed'
            position['exit_reason'] = exit_reason
            position['exit_time'] = datetime.now(timezone.utc).isoformat()
            position['exit_home_prob'] = current_home_prob
            position['exit_away_prob'] = current_away_prob
            position['pnl'] = pnl
            position['exit_price_change'] = price_change

            result_str = "PROFIT" if pnl >= 0 else "LOSS"
            logger.info(f"EXIT: {away_team} @ {home_team} - {exit_reason} | {result_str}: ${pnl:+.2f} | Bankroll: ${new_bankroll:.2f}")

            log_trade({
                "type": "EXIT",
                "time": datetime.now(timezone.utc).isoformat(),
                "position_id": position_id,
                "game": f"{away_team} @ {home_team}",
                "reason": exit_reason,
                "price_change": price_change,
                "pnl": pnl,
                "bankroll": new_bankroll
            })
        elif not EARLY_EXIT_ENABLED and not (UNDERDOG_TAKE_PROFIT_ENABLED and bet_side):
            logger.debug(f"Holding: {away_team} @ {home_team} | Change: Home {home_change:+.1%}, Away {away_change:+.1%}")

    save_positions(positions)


def generate_daily_report(logger, report_date=None):
    """Generate end-of-day summary with P&L.

    Args:
        logger: Logger instance
        report_date: Date string (YYYY-MM-DD) for the report. Defaults to today.
    """
    positions = load_positions()
    if report_date is None:
        report_date = datetime.now().strftime('%Y-%m-%d')

    report_file = DATA_DIR / f"report_{report_date}.txt"

    # Filter positions for this date (by entry_time or position_id)
    date_positions = {
        k: v for k, v in positions.items()
        if report_date in v.get('entry_time', '') or report_date.replace('-', '') in k
    }

    bets = [p for p in date_positions.values() if p.get('bet_side')]
    resolved = [p for p in date_positions.values() if p.get('status') == 'resolved']
    closed = [p for p in date_positions.values() if p.get('status') == 'closed']
    all_finished = resolved + closed

    # Calculate P&L from all finished positions
    total_pnl = sum(p.get('pnl', 0) for p in all_finished)
    wins = [p for p in all_finished if p.get('pnl', 0) > 0]
    losses = [p for p in all_finished if p.get('pnl', 0) < 0]

    current_bankroll = load_bankroll()

    report = []
    report.append("=" * 60)
    report.append(f"DAILY REPORT - {report_date}")
    report.append("=" * 60)
    report.append(f"\nGames tracked: {len(date_positions)}")
    report.append(f"Bets placed: {len(bets)}")
    report.append(f"Early exits (stop-loss/take-profit): {len(closed)}")
    report.append(f"Game resolutions: {len(resolved)}")

    if bets:
        total_kelly = sum(p.get('bet_kelly', 0) for p in bets)
        report.append(f"Total Kelly: {total_kelly:.1f}%")

    report.append(f"\n--- P&L SUMMARY ---")
    report.append(f"Wins: {len(wins)} | Losses: {len(losses)}")
    if len(all_finished) > 0:
        win_rate = len(wins) / len(all_finished) * 100
        report.append(f"Win Rate: {win_rate:.1f}%")
    report.append(f"Daily P&L: ${total_pnl:+.2f}")
    report.append(f"Current Bankroll: ${current_bankroll:.2f}")

    if closed:
        report.append("\n--- EARLY EXITS (REALIZED P&L) ---")
        for p in closed:
            game = f"{p['away_team']} @ {p['home_team']}"
            side = p.get('bet_side', '').upper()
            reason = p.get('exit_reason', '')
            pnl = p.get('pnl', 0)
            price_change = p.get('exit_price_change', 0)
            report.append(f"  {game}")
            report.append(f"    Bet: {side} | {reason} | P&L: ${pnl:+.2f} ({price_change:+.1%})")

    if resolved:
        report.append("\n--- GAME RESOLUTIONS ---")
        for p in resolved:
            game = f"{p['away_team']} @ {p['home_team']}"
            side = p.get('bet_side', '').upper()
            winner = p.get('winner', '').upper()
            result = "WIN" if p.get('won') else "LOSS"
            pnl = p.get('pnl', 0)
            report.append(f"  {game}")
            report.append(f"    Bet: {side} | Winner: {winner} | {result}: ${pnl:+.2f}")

    if bets:
        report.append("\n--- ALL BETS ---")
        for p in bets:
            game = f"{p['away_team']} @ {p['home_team']}"
            side = p.get('bet_side', '').upper()
            kelly = p.get('bet_kelly', 0)
            edge = p.get(f"{p.get('bet_side')}_edge", 0)
            status = p.get('status', 'unknown')
            report.append(f"  {game}: {side} ({kelly:.1f}%, edge: {edge:+.1%}) [{status}]")

    report.append("\n" + "=" * 60)

    report_text = "\n".join(report)

    with open(report_file, 'w') as f:
        f.write(report_text)

    logger.info(f"Report saved: {report_file}")
    logger.info(f"Daily P&L: ${total_pnl:+.2f} | Bankroll: ${current_bankroll:.2f}")
    print(report_text)


def run_scheduler():
    """Main scheduler loop - smart per-game timing."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("POLYMARKET SMART SCHEDULER")
    logger.info("=" * 60)
    logger.info(f"Bets placed: {MINUTES_BEFORE_GAME} minutes before each game")
    logger.info(f"Monitor interval: {MONITOR_INTERVAL} minutes")
    logger.info(f"Timezone offset: UTC{LOCAL_TIMEZONE_OFFSET:+d}")
    logger.info("")

    # Initialize AlertManager for Discord notifications
    alert_mgr = AlertManager(
        data_dir=DATA_DIR,
        enable_console=False,
        enable_file=True,
        webhook_url=DISCORD_WEBHOOK_URL,
        webhook_platform="discord",
    )

    # Initialize WebSocket monitor for real-time prices
    ws_monitor = WebSocketPriceMonitor(logger=logger)
    ws_started = ws_monitor.start()
    if ws_started:
        logger.info("WebSocket price monitor started")
    else:
        logger.warning("WebSocket monitor failed to start - falling back to REST API polling")
        ws_monitor = None

    # Subscribe to open positions' token IDs
    if ws_monitor:
        positions = load_positions()
        token_ids = []
        for pos in positions.values():
            if pos.get('status') == 'open' and pos.get('bet_side'):
                bet_side = pos.get('bet_side')
                token_id = pos.get('home_token_id') if bet_side == 'home' else pos.get('away_token_id')
                if token_id:
                    token_ids.append(token_id)
        if token_ids:
            ws_monitor.subscribe(token_ids)
            logger.info(f"Subscribed to {len(token_ids)} position tokens via WebSocket")

    # Send startup alert
    alert_mgr.info("NBA Paper Trader started", {"system": "NBA", "status": "online"})

    initialized_games = set()  # Track games we've already bet on
    last_monitor = None

    while True:
        try:
            now = datetime.now(timezone.utc)
            today = now.date()

            # Get today's games with times
            games = get_todays_games_with_times()

            # Check if we should generate a report for any date with all positions resolved
            positions = load_positions()

            # Group positions by entry date
            positions_by_date = {}
            for pos_id, pos in positions.items():
                entry_time = pos.get('entry_time', '')
                if entry_time:
                    entry_date = entry_time[:10]  # YYYY-MM-DD
                    if entry_date not in positions_by_date:
                        positions_by_date[entry_date] = []
                    positions_by_date[entry_date].append(pos)

            # Check each date - generate report if all positions are closed/resolved and we haven't reported yet
            for date, date_positions in positions_by_date.items():
                report_file = DATA_DIR / f"report_{date}.txt"
                # Positions are done if they're resolved (game ended) or closed (early exit)
                all_done = all(p.get('status') in ['resolved', 'closed'] for p in date_positions)
                has_pnl = any(p.get('pnl') is not None for p in date_positions)

                if all_done and has_pnl and not report_file.exists():
                    logger.info(f"All positions closed for {date}. Generating report...")
                    generate_daily_report(logger, report_date=date)

            # Track open positions for logging
            open_count = len([p for p in positions.values() if p['status'] == 'open'])

            if games:
                logger.debug(f"Found {len(games)} upcoming games")

                for game in games:
                    title = game['title']
                    game_time = game['game_time']
                    bet_time = game['bet_time']

                    # Skip if already initialized
                    game_id = f"{title}_{game_time.date()}"
                    if game_id in initialized_games:
                        continue

                    # Check if it's time to bet (within 5 min of bet_time)
                    time_until_bet = (bet_time - now).total_seconds()

                    if -300 <= time_until_bet <= 300:  # Within 5 min window
                        logger.info("-" * 40)
                        logger.info(f"TIME TO BET: {title}")
                        logger.info(f"Game starts in {MINUTES_BEFORE_GAME} minutes")
                        logger.info("-" * 40)

                        result = init_single_game(game['event'], game_time, logger, alert_mgr, ws_monitor)
                        initialized_games.add(game_id)

                    elif time_until_bet > 0:
                        hours = int(time_until_bet // 3600)
                        mins = int((time_until_bet % 3600) // 60)
                        logger.debug(f"  {title}: bet in {hours}h {mins}m")

            # Monitor existing positions
            if last_monitor is None or (now - last_monitor).total_seconds() >= MONITOR_INTERVAL * 60:
                positions = load_positions()
                open_count = len([p for p in positions.values() if p['status'] == 'open'])

                if open_count > 0:
                    logger.info(f"Monitoring {open_count} open positions...")
                    monitor_positions(logger, ws_monitor=ws_monitor)
                    last_monitor = now

            # Sleep before next check
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("\nScheduler stopping...")
            if ws_monitor:
                ws_monitor.stop()
                logger.info("WebSocket monitor stopped")
            logger.info("Scheduler stopped")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(60)


def show_schedule():
    """Show today's game schedule and bet times."""
    logger = setup_logging()
    games = get_todays_games_with_times()

    print("=" * 60)
    print("TODAY'S GAME SCHEDULE")
    print("=" * 60)
    print(f"\nBets will be placed {MINUTES_BEFORE_GAME} minutes before each game\n")

    if not games:
        print("No upcoming games found.")
        return

    for game in games:
        title = game['title']
        game_time = game['game_time'] + timedelta(hours=LOCAL_TIMEZONE_OFFSET)
        bet_time = game['bet_time'] + timedelta(hours=LOCAL_TIMEZONE_OFFSET)

        print(f"{title}")
        print(f"  Game: {game_time.strftime('%I:%M %p')} local")
        print(f"  Bet:  {bet_time.strftime('%I:%M %p')} local")
        print()


def show_status():
    """Show current positions and bankroll status."""
    from src.Polymarket.paper_trader import STARTING_BANKROLL

    positions = load_positions()
    bankroll = load_bankroll()

    print("=" * 60)
    print("PAPER TRADING STATUS")
    print("=" * 60)

    print(f"\nStarting Bankroll: ${STARTING_BANKROLL:.2f}")
    print(f"Current Bankroll:  ${bankroll:.2f}")
    print(f"Total P&L:         ${bankroll - STARTING_BANKROLL:+.2f}")

    open_pos = [p for p in positions.values() if p['status'] == 'open']
    resolved_pos = [p for p in positions.values() if p['status'] == 'resolved']
    closed_pos = [p for p in positions.values() if p['status'] == 'closed']
    all_finished = resolved_pos + closed_pos

    print(f"\nOpen positions: {len(open_pos)}")
    print(f"Closed (early exit): {len(closed_pos)}")
    print(f"Resolved (game ended): {len(resolved_pos)}")

    if all_finished:
        wins = len([p for p in all_finished if p.get('pnl', 0) > 0])
        losses = len([p for p in all_finished if p.get('pnl', 0) < 0])
        win_rate = wins / len(all_finished) * 100 if all_finished else 0
        print(f"\nWin/Loss: {wins}W - {losses}L ({win_rate:.1f}%)")

    if open_pos:
        print("\n--- OPEN POSITIONS ---")
        for p in open_pos:
            game = f"{p['away_team']} @ {p['home_team']}"
            side = p.get('bet_side', 'none').upper()
            kelly = p.get('bet_kelly', 0)
            print(f"  {game}: {side} ({kelly:.1f}%)")


def fix_exit_signals():
    """Retroactively process exit_signal positions to calculate realized P/L."""
    from src.Polymarket.paper_trader import calculate_exit_pnl, STARTING_BANKROLL

    positions = load_positions()
    exit_signals = {k: v for k, v in positions.items() if v.get('status') == 'exit_signal'}

    if not exit_signals:
        print("No exit_signal positions to fix.")
        return

    print(f"Found {len(exit_signals)} positions with exit_signal status to process...")

    # Reset bankroll to starting value, then process all exits chronologically
    bankroll = STARTING_BANKROLL
    total_pnl = 0

    # Sort by exit time
    sorted_exits = sorted(exit_signals.items(), key=lambda x: x[1].get('exit_time', ''))

    for position_id, position in sorted_exits:
        entry_prob = position['entry_home_prob'] if position['bet_side'] == 'home' else position['entry_away_prob']
        exit_prob = position.get('exit_home_prob') if position['bet_side'] == 'home' else position.get('exit_away_prob')

        if entry_prob and exit_prob:
            price_change = (exit_prob - entry_prob) / entry_prob

            # Calculate P/L
            kelly_pct = position.get('bet_kelly', 0) / 100
            position_size = bankroll * kelly_pct
            pnl = round(position_size * price_change, 2)

            # Update position
            position['status'] = 'closed'
            position['pnl'] = pnl
            position['exit_price_change'] = price_change

            bankroll += pnl
            total_pnl += pnl

            game = f"{position['away_team']} @ {position['home_team']}"
            result = "PROFIT" if pnl >= 0 else "LOSS"
            print(f"  {game}: {position['exit_reason']} | {result}: ${pnl:+.2f}")

    # Save updated positions and bankroll
    save_positions(positions)
    save_bankroll(bankroll)

    print(f"\nTotal P&L: ${total_pnl:+.2f}")
    print(f"Final Bankroll: ${bankroll:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Polymarket Smart Scheduler')
    parser.add_argument('--schedule', action='store_true', help='Show today\'s schedule')
    parser.add_argument('--status', action='store_true', help='Show current status and bankroll')
    parser.add_argument('--test', action='store_true', help='Run one init cycle')
    parser.add_argument('--fix-exits', action='store_true', help='Fix exit_signal positions with realized P/L')
    parser.add_argument('--report', action='store_true', help='Generate daily report')
    args = parser.parse_args()

    if args.schedule:
        show_schedule()
    elif args.status:
        show_status()
    elif args.fix_exits:
        fix_exit_signals()
    elif args.report:
        logger = setup_logging()
        generate_daily_report(logger)
    elif args.test:
        logger = setup_logging()
        games = get_todays_games_with_times()
        if games:
            init_single_game(games[0]['event'], games[0]['game_time'], logger)
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
