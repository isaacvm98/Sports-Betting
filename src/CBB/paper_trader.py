"""
CBB Paper Trading Bot - ACC Spread Betting

Uses confidence interval width for position sizing.
Narrower CI = more confidence = bigger bet.

Usage:
    python -m src.CBB.paper_trader --init       # Initialize positions for today's games
    python -m src.CBB.paper_trader --status     # View current positions and P&L
    python -m src.CBB.paper_trader --monitor    # Check for exits and resolutions
    python -m src.CBB.paper_trader --report     # Generate daily report
"""

import argparse
import json
import re
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# Configuration
DATA_DIR = Path("Data/cbb_paper_trading")
PREDICTIONS_URL = "https://raw.githubusercontent.com/isaacvm98/go_duke_triangle_comp/refs/heads/main/data/tsa_pi_midsters_2026.csv"
POINT_SPREAD_URL = "https://raw.githubusercontent.com/isaacvm98/go_duke_triangle_comp/refs/heads/main/data/tsa_pt_spread_midsters_2026.csv"

# Polymarket API
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

# Timezone
ET = ZoneInfo("America/New_York")

# Paper trading settings
STARTING_BANKROLL = 1000
BANKROLL_FILE = DATA_DIR / "bankroll.json"
POSITIONS_FILE = DATA_DIR / "positions.json"
TRADES_LOG = DATA_DIR / "trades.json"

# CI-Width Sizing parameters
BASE_UNITS = 2.0          # Base unit size (% of bankroll)
MIN_CI_WIDTH = 5.0        # Minimum CI width for normalization
MAX_CI_WIDTH = 30.0       # Maximum CI width (wider = less confident)
MIN_BET_UNITS = 0.5       # Minimum bet size
MAX_BET_UNITS = 5.0       # Maximum bet size (cap exposure)

# === Entry Criteria ===
MIN_EDGE_POINTS = 2.0     # Minimum edge (pts) to place a bet for spreads
MAX_EDGE_POINTS = 15.0    # Maximum edge (pts) - skip if too good (likely error)
MIN_EDGE_PROB = 0.08      # Minimum edge (8%) for ML bets (was 5%)
ENABLE_ML_BETTING = True  # Set False to disable ML bets entirely

# === ML Filters ===
MAX_ML_ENTRY_PRICE = 0.45  # Skip ML bets on heavy favorites (>45%)
MIN_ML_ENTRY_PRICE = 0.25  # Skip ML bets on extreme underdogs (<25%)

# === Position Limits ===
MAX_SIMULTANEOUS_POSITIONS = 5
MAX_DAILY_BETS = 8

# === Discord Alerts ===
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1471009722772623361/2FEqyrUTnUGOEKrLB-RVZOdcL1Fg_xKzACG1eVEGevHrJ2kFi7gwvDfDn1c7qgcr68Wq"

# ML Take-Profit settings (only for ML bets - spreads ride to resolution)
# Dynamic take-profit based on entry price (underdog vs favorite)
# Underdogs: take profit early since we're buying cheap shares
# Favorites: can wait longer since smaller upside anyway
ML_TAKE_PROFIT_BY_ENTRY = [
    # (max_entry_prob, take_profit_pct)
    (0.30, 0.12),   # Deep underdog (<30%): take profit at 12% move
    (0.40, 0.15),   # Underdog (30-40%): take profit at 15% move
    (0.50, 0.18),   # Slight underdog (40-50%): take profit at 18% move
    (0.60, 0.20),   # Slight favorite (50-60%): take profit at 20% move
    (0.70, 0.25),   # Favorite (60-70%): take profit at 25% move
    (1.00, 0.30),   # Heavy favorite (70%+): take profit at 30% move
]


def get_ml_take_profit_threshold(entry_prob):
    """
    Get take-profit threshold for ML bets based on entry price.

    Underdogs get tighter take-profits since we're buying cheap.
    Returns the percentage move at which we should take profit.
    """
    for max_prob, take_profit in ML_TAKE_PROFIT_BY_ENTRY:
        if entry_prob <= max_prob:
            return take_profit
    return 0.30  # Default fallback


# Spread to probability conversion factor
# Each point of spread ≈ 2.5% win probability change
SPREAD_TO_PROB_FACTOR = 0.025


def spread_to_win_prob(home_spread):
    """
    Convert home spread to home team win probability.

    Uses linear approximation: each point ≈ 2.5% change from 50%
    Capped between 1% and 99%.

    Examples:
        home_spread = 0 → 50% home win
        home_spread = -10 → 25% home win (home loses by 10)
        home_spread = +10 → 75% home win (home wins by 10)
    """
    prob = 0.50 + (home_spread * SPREAD_TO_PROB_FACTOR)
    return max(0.01, min(0.99, prob))


def determine_ml_bet(match):
    """
    Determine moneyline bet based on spread-derived probability.

    Returns: (bet_side, edge_prob, reasoning)
    - bet_side: 'home', 'away', or None
    - edge_prob: probability edge we have
    - reasoning: explanation string
    """
    model_spread = match['model_spread']

    # Convert model spread to win probabilities
    model_home_prob = spread_to_win_prob(model_spread)
    model_away_prob = 1 - model_home_prob

    # Market probabilities from ML prices
    market_home_prob = match.get('ml_home_price', 0.5)
    market_away_prob = match.get('ml_away_price', 0.5)

    # Calculate edges
    home_edge = model_home_prob - market_home_prob
    away_edge = model_away_prob - market_away_prob

    # Determine best bet
    if abs(home_edge) < MIN_EDGE_PROB and abs(away_edge) < MIN_EDGE_PROB:
        return None, max(abs(home_edge), abs(away_edge)), f"Edge too small ({home_edge:+.1%} home, {away_edge:+.1%} away)"

    if home_edge > away_edge and home_edge >= MIN_EDGE_PROB:
        return 'home', home_edge, f"Model {model_home_prob:.1%} > Market {market_home_prob:.1%} -> HOME"
    elif away_edge >= MIN_EDGE_PROB:
        return 'away', away_edge, f"Model {model_away_prob:.1%} > Market {market_away_prob:.1%} -> AWAY"
    else:
        return None, 0, "No significant edge"

# Scheduling parameters
MINUTES_BEFORE_GAME = 30  # Place bets this many minutes before game starts
MONITORING_INTERVAL = 300  # Check every 5 minutes (in seconds)

# ACC team name variations for matching
ACC_TEAMS = {
    "Virginia": ["Virginia", "UVA", "Cavaliers"],
    "Duke": ["Duke", "Blue Devils"],
    "North Carolina": ["North Carolina", "UNC", "Tar Heels"],
    "Wake Forest": ["Wake Forest", "Wake", "Demon Deacons"],
    "NC State": ["NC State", "North Carolina State", "Wolfpack"],
    "Louisville": ["Louisville", "Cardinals"],
    "Syracuse": ["Syracuse", "Orange"],
    "Clemson": ["Clemson", "Tigers"],
    "Florida State": ["Florida State", "FSU", "Seminoles"],
    "Georgia Tech": ["Georgia Tech", "GT", "Yellow Jackets"],
    "Miami": ["Miami", "Miami (FL)", "Hurricanes"],
    "Boston College": ["Boston College", "BC", "Eagles"],
    "Pittsburgh": ["Pittsburgh", "Pitt", "Panthers"],
    "Notre Dame": ["Notre Dame", "Irish", "Fighting Irish"],
    "Virginia Tech": ["Virginia Tech", "VT", "Hokies"],
    "California": ["California", "Cal", "Golden Bears"],
    "Stanford": ["Stanford", "Cardinal"],
    "SMU": ["SMU", "Southern Methodist", "Mustangs"],
}


def fetch_market_price(market_id):
    """
    Fetch current prices for a specific market by market_id.

    Returns dict with yes_price, no_price or None if not found.
    """
    try:
        response = requests.get(
            f"{GAMMA_API_URL}/markets/{market_id}",
            timeout=15
        )
        if response.status_code != 200:
            return None

        market = response.json()
        outcome_prices = market.get("outcomePrices", "")

        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices) if outcome_prices else []
            except json.JSONDecodeError:
                outcome_prices = []

        if len(outcome_prices) >= 2:
            return {
                "yes_price": float(outcome_prices[0]),
                "no_price": float(outcome_prices[1]),
                "closed": market.get("closed", False),
                "resolved": market.get("resolved", False),
            }
        return None
    except Exception as e:
        print(f"  Error fetching market {market_id}: {e}")
        return None


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_bankroll():
    """Load current bankroll from file."""
    if BANKROLL_FILE.exists():
        with open(BANKROLL_FILE, "r") as f:
            data = json.load(f)
            return data.get("bankroll", STARTING_BANKROLL)
    return STARTING_BANKROLL


def save_bankroll(bankroll):
    """Save bankroll to file."""
    ensure_data_dir()
    with open(BANKROLL_FILE, "w") as f:
        json.dump({
            "bankroll": bankroll,
            "updated": datetime.now(timezone.utc).isoformat()
        }, f, indent=2)


def load_positions():
    """Load current positions from file."""
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def count_open_positions(positions=None):
    """Count current open positions."""
    if positions is None:
        positions = load_positions()
    return sum(1 for p in positions.values() if p.get('status') == 'open')


def count_todays_bets(positions=None):
    """Count bets placed today."""
    if positions is None:
        positions = load_positions()
    today = datetime.now().strftime('%Y%m%d')
    return sum(1 for k in positions.keys() if today in k)


def can_open_position(positions=None):
    """Check if we can open a new position based on limits."""
    if positions is None:
        positions = load_positions()

    open_count = count_open_positions(positions)
    if open_count >= MAX_SIMULTANEOUS_POSITIONS:
        return False, f"At position limit ({open_count}/{MAX_SIMULTANEOUS_POSITIONS})"

    daily_count = count_todays_bets(positions)
    if daily_count >= MAX_DAILY_BETS:
        return False, f"At daily bet limit ({daily_count}/{MAX_DAILY_BETS})"

    return True, ""


def save_positions(positions):
    """Save positions to file."""
    ensure_data_dir()
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def log_trade(trade):
    """Append trade to trades log."""
    ensure_data_dir()
    trades = []
    if TRADES_LOG.exists():
        with open(TRADES_LOG, "r") as f:
            trades = json.load(f)
    trades.append(trade)
    with open(TRADES_LOG, "w") as f:
        json.dump(trades, f, indent=2, default=str)


def load_predictions():
    """Load model predictions from GitHub CSVs."""
    print("Loading predictions from GitHub...")

    # Load confidence intervals
    ci_df = pd.read_csv(PREDICTIONS_URL)
    ci_df['Date'] = pd.to_datetime(ci_df['Date'])

    # Load point spreads
    ps_df = pd.read_csv(POINT_SPREAD_URL)
    ps_df['Date'] = pd.to_datetime(ps_df['Date'])

    # Merge on Date, Away, Home
    df = pd.merge(
        ci_df[['Date', 'Away', 'Home', 'ci_lb', 'ci_ub']],
        ps_df[['Date', 'Away', 'Home', 'pt_spread']],
        on=['Date', 'Away', 'Home'],
        how='inner'
    )

    # Calculate CI width
    df['ci_width'] = df['ci_ub'] - df['ci_lb']

    print(f"Loaded {len(df)} game predictions")
    return df


def normalize_team_name(name):
    """
    Normalize team name for matching.

    Handles both short names (from predictions) and full names with mascots
    (from Polymarket, e.g., "Wake Forest Demon Deacons" -> "Wake Forest").
    """
    name = name.strip()

    # First check exact matches
    for canonical, variants in ACC_TEAMS.items():
        if name in variants or name == canonical:
            return canonical

    # Build list of all (pattern, canonical) pairs, sorted by pattern length (longest first)
    # This ensures "North Carolina State" matches NC State before "North Carolina" matches UNC
    all_patterns = []
    for canonical, variants in ACC_TEAMS.items():
        all_patterns.append((canonical, canonical))
        for variant in variants:
            all_patterns.append((variant, canonical))

    # Sort by pattern length, longest first
    all_patterns.sort(key=lambda x: len(x[0]), reverse=True)

    # Check if name starts with any pattern
    for pattern, canonical in all_patterns:
        if name.startswith(pattern + " ") or name == pattern:
            return canonical

    # Fallback: return the name before any common mascot words
    mascot_words = ["Cardinals", "Blue Devils", "Tar Heels", "Cavaliers", "Demon Deacons",
                    "Wolfpack", "Orange", "Tigers", "Seminoles", "Yellow Jackets",
                    "Hurricanes", "Eagles", "Panthers", "Irish", "Hokies",
                    "Golden Bears", "Cardinal", "Mustangs", "Bulldogs", "Wildcats"]
    for mascot in mascot_words:
        if f" {mascot}" in name:
            return name.split(f" {mascot}")[0].strip()

    return name


def calculate_ci_width_size(ci_width, edge_points):
    """
    Calculate bet size using CI-Width sizing with tiered edge multipliers.

    Narrower CI = more confidence = bigger bet.
    Larger edge = more confident bet = larger size (tiered).

    Capped between MIN_BET_UNITS and MAX_BET_UNITS.
    """
    # Clamp CI width to reasonable range
    ci_width = max(MIN_CI_WIDTH, min(MAX_CI_WIDTH, ci_width))

    # Base sizing: inverse of CI width (normalized)
    confidence_factor = MAX_CI_WIDTH / ci_width

    # Tiered edge multiplier (from plan)
    abs_edge = abs(edge_points)
    if abs_edge < 3.0:
        edge_mult = 0.5    # Small edge = half size
    elif abs_edge < 5.0:
        edge_mult = 0.75   # Medium edge
    elif abs_edge < 8.0:
        edge_mult = 1.0    # Good edge = full size
    else:
        edge_mult = 1.25   # Great edge = 25% bonus

    # Calculate size
    size = BASE_UNITS * confidence_factor * edge_mult

    # Apply caps
    size = max(MIN_BET_UNITS, min(MAX_BET_UNITS, size))

    return round(size, 2)


def fetch_polymarket_cbb_spreads(include_started=False, minutes_before_game=None):
    """
    Fetch CBB spread markets from Polymarket using series_id.

    Returns markets with home_spread from the HOME team's perspective.
    This matches our model output (positive = home favored).

    Polymarket bet: YES = home covers, NO = home doesn't cover

    Args:
        include_started: If True, include games that have already started (for monitoring positions)
        minutes_before_game: If set, only return games starting within this many minutes
    """
    print("Fetching Polymarket CBB spreads...")

    markets_dict = {}
    now = datetime.now(timezone.utc)
    today = datetime.now(ET).date()

    try:
        # Fetch all NCAA CBB events using series_id
        response = requests.get(
            f"{GAMMA_API_URL}/events",
            params={
                "series_id": "10470",  # NCAA CBB
                "active": "true",
                "closed": "false",
                "limit": 500,
            },
            timeout=30
        )
        response.raise_for_status()
        events = response.json()

        print(f"  Fetched {len(events)} NCAA CBB events total")

        # Filter for today's games
        for event in events:
            start_time_str = event.get("startTime") or event.get("endDate", "")
            if not start_time_str:
                continue

            try:
                game_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                game_time_et = game_time.astimezone(ET)
            except:
                continue

            # Only today's games
            if game_time_et.date() != today:
                continue

            # Skip games that already started (unless monitoring positions)
            if not include_started and game_time <= now:
                continue

            # If minutes_before_game is set, only include games starting within that window
            if minutes_before_game is not None:
                minutes_until_start = (game_time - now).total_seconds() / 60
                if minutes_until_start < 0 or minutes_until_start > minutes_before_game:
                    continue

            title = event.get("title", "")
            slug = event.get("slug", "")
            markets = event.get("markets", [])

            # Parse teams from title: "Away Team vs. Home Team"
            title_parts = re.split(r'\s+vs\.?\s+', title, flags=re.IGNORECASE)
            if len(title_parts) != 2:
                continue

            away_full = title_parts[0].strip()
            home_full = title_parts[1].strip()

            # Normalize for matching with predictions
            away = normalize_team_name(away_full)
            home = normalize_team_name(home_full)

            # Find spread and ML markets using sportsMarketType
            spread_market = None
            ml_market = None

            for market in markets:
                market_type = market.get("sportsMarketType", "")
                question = market.get("question", "")

                # Skip 1H markets
                if "1H" in question:
                    continue

                if market_type == "spreads":
                    # Parse outcomes to check if this is a home team spread
                    outcomes_str = market.get("outcomes", "[]")
                    try:
                        outcomes = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
                    except json.JSONDecodeError:
                        outcomes = []

                    if len(outcomes) >= 2:
                        spread_team = outcomes[0]
                        # Prefer home team spread market
                        is_home_spread = any(word.lower() in spread_team.lower()
                                            for word in home_full.split()[:2])
                        if is_home_spread:
                            spread_market = market
                        elif spread_market is None:
                            # Fall back to away spread if no home spread found
                            spread_market = market

                elif market_type == "moneyline":
                    ml_market = market

            game_key = f"{home}:{away}"

            if spread_market:
                # Use the line field directly - much cleaner than regex
                spread_line = spread_market.get("line", 0)

                # Parse outcomes to determine which team the spread is for
                outcomes_str = spread_market.get("outcomes", "[]")
                try:
                    outcomes = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
                except json.JSONDecodeError:
                    outcomes = []

                spread_team = outcomes[0] if outcomes else ""

                # Check if this spread is for the home team
                is_home_spread = any(word.lower() in spread_team.lower()
                                    for word in home_full.split()[:2])

                # Convert to home_spread (what our model outputs)
                if is_home_spread:
                    home_spread = spread_line  # Already from home perspective
                else:
                    home_spread = -spread_line  # Flip if it's away team's spread

                outcome_prices = spread_market.get("outcomePrices", "")
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices) if outcome_prices else []
                    except json.JSONDecodeError:
                        outcome_prices = []

                token_ids = spread_market.get("clobTokenIds", "")
                if isinstance(token_ids, str):
                    try:
                        token_ids = json.loads(token_ids)
                    except:
                        token_ids = []

                # For home spread markets: YES = home covers, NO = away covers
                # outcomes[0] price = home covers price
                # outcomes[1] price = away covers price
                if is_home_spread:
                    home_cover_price = float(outcome_prices[0]) if len(outcome_prices) > 0 else 0.5
                    away_cover_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5
                else:
                    # If away spread market, flip the prices
                    away_cover_price = float(outcome_prices[0]) if len(outcome_prices) > 0 else 0.5
                    home_cover_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5

                # Get ML data if available
                ml_home_price = None
                ml_away_price = None
                if ml_market:
                    ml_prices = ml_market.get("outcomePrices", "")
                    if isinstance(ml_prices, str):
                        try:
                            ml_prices = json.loads(ml_prices)
                        except:
                            ml_prices = []
                    if len(ml_prices) >= 2:
                        ml_away_price = float(ml_prices[0])
                        ml_home_price = float(ml_prices[1])

                markets_dict[game_key] = {
                    "home_team": home,
                    "away_team": away,
                    "home_spread": home_spread,  # From home perspective (matches model)
                    "is_home_spread": is_home_spread,  # True if market is for home team
                    "home_cover_price": home_cover_price,  # Price for home covering
                    "away_cover_price": away_cover_price,  # Price for away covering
                    "yes_token": token_ids[0] if len(token_ids) > 0 else None,
                    "no_token": token_ids[1] if len(token_ids) > 1 else None,
                    "ml_home_price": ml_home_price,
                    "ml_away_price": ml_away_price,
                    "has_spread": True,
                    "has_ml": ml_home_price is not None,
                    "game_time": game_time.isoformat(),
                    "market_id": spread_market.get("id"),
                    "event_slug": slug,
                    "title": title,
                }

                ml_str = f" | ML: {ml_away_price:.0%}/{ml_home_price:.0%}" if ml_home_price else ""
                print(f"  {away} @ {home} | Home {home_spread:+.1f} | Cover: {home_cover_price:.0%}/{away_cover_price:.0%}{ml_str}")

            elif ml_market:
                # ML only
                ml_prices = ml_market.get("outcomePrices", "")
                if isinstance(ml_prices, str):
                    try:
                        ml_prices = json.loads(ml_prices)
                    except:
                        ml_prices = []

                if len(ml_prices) >= 2:
                    ml_away_price = float(ml_prices[0])
                    ml_home_price = float(ml_prices[1])

                    ml_tokens = ml_market.get("clobTokenIds", "")
                    if isinstance(ml_tokens, str):
                        try:
                            ml_tokens = json.loads(ml_tokens)
                        except:
                            ml_tokens = []

                    markets_dict[game_key] = {
                        "home_team": home,
                        "away_team": away,
                        "home_spread": None,
                        "home_cover_price": None,
                        "away_cover_price": None,
                        "ml_home_price": ml_home_price,
                        "ml_away_price": ml_away_price,
                        "ml_home_token": ml_tokens[1] if len(ml_tokens) > 1 else None,
                        "ml_away_token": ml_tokens[0] if len(ml_tokens) > 0 else None,
                        "has_spread": False,
                        "has_ml": True,
                        "game_time": game_time.isoformat(),
                        "market_id": ml_market.get("id"),
                        "event_slug": slug,
                        "title": title,
                    }

                    print(f"  {away} @ {home} | ML only: {ml_away_price:.0%}/{ml_home_price:.0%}")

    except Exception as e:
        print(f"Error fetching markets: {e}")
        import traceback
        traceback.print_exc()

    print(f"Found {len(markets_dict)} CBB markets for today")
    return markets_dict


def match_predictions_to_markets(predictions_df, markets):
    """Match model predictions to Polymarket markets."""
    matches = []

    today = datetime.now(ET).date()

    for _, row in predictions_df.iterrows():
        pred_date = row['Date'].date()

        # Only look at today's games (or optionally future games)
        if pred_date != today:
            continue

        home = normalize_team_name(row['Home'])
        away = normalize_team_name(row['Away'])
        game_key = f"{home}:{away}"

        if game_key in markets:
            market = markets[game_key]
            matches.append({
                "date": pred_date,
                "home_team": home,
                "away_team": away,
                "game_key": game_key,
                "model_spread": row['pt_spread'],  # Predicted home spread
                "ci_lb": row['ci_lb'],
                "ci_ub": row['ci_ub'],
                "ci_width": row['ci_width'],
                # Spread market data - all from home perspective
                "home_spread": market.get('home_spread'),  # Market's home spread
                "home_cover_price": market.get('home_cover_price'),
                "away_cover_price": market.get('away_cover_price'),
                "is_home_spread": market.get('is_home_spread', True),
                "yes_token": market.get('yes_token'),
                "no_token": market.get('no_token'),
                "has_spread": market.get('has_spread', False),
                # ML market data
                "ml_home_price": market.get('ml_home_price'),
                "ml_away_price": market.get('ml_away_price'),
                "ml_home_token": market.get('ml_home_token'),
                "ml_away_token": market.get('ml_away_token'),
                "has_ml": market.get('has_ml', False),
                # Common
                "game_time": market['game_time'],
                "market_id": market['market_id'],
            })

    return matches


def determine_bet(match):
    """
    Determine if we should bet and which side.

    Both model and market spreads are from HOME perspective:
    - Negative spread = home is favorite (must win by X)
    - Positive spread = home is underdog (can lose by up to X)

    Home covers if: model_spread + home_spread > 0

    Example: home_spread = -2.5 (home favored by 2.5)
      - model_spread = +5 (home wins by 5): 5 + (-2.5) = 2.5 > 0 → home covers
      - model_spread = -1 (home loses by 1): -1 + (-2.5) = -3.5 < 0 → away covers

    Returns: (bet_on, edge_points, reasoning)
    - bet_on: 'home' (home covers) or 'away' (away covers) or None
    - edge_points: absolute edge in points
    - reasoning: explanation string
    """
    model_spread = match['model_spread']  # Model's predicted home margin
    market_spread = match['home_spread']  # Market's home spread line

    # Edge = model_spread + market_spread
    # Positive = home covers the spread
    # Negative = away covers (home doesn't cover)
    edge = model_spread + market_spread

    if abs(edge) < MIN_EDGE_POINTS:
        return None, abs(edge), f"Edge too small ({edge:+.1f} pts, need {MIN_EDGE_POINTS}+)"

    if abs(edge) > MAX_EDGE_POINTS:
        return None, abs(edge), f"Edge too large ({edge:+.1f} pts) - likely error, skipping"

    if edge > 0:
        # Home beats the spread
        return 'home', edge, f"Model {model_spread:+.1f} + Spread {market_spread:+.1f} = {edge:+.1f} -> HOME covers"
    else:
        # Away covers (home doesn't beat spread)
        return 'away', abs(edge), f"Model {model_spread:+.1f} + Spread {market_spread:+.1f} = {edge:+.1f} -> AWAY covers"


def init_positions(date_filter=None):
    """Initialize positions for games based on model predictions."""
    print("=" * 60)
    print("INITIALIZING CBB PAPER TRADING POSITIONS")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Load predictions
    predictions_df = load_predictions()

    # Fetch current Polymarket spreads
    markets = fetch_polymarket_cbb_spreads()

    if not markets:
        print("No Polymarket CBB markets found.")
        print("\nNote: Markets may not be available yet. Showing predictions anyway:\n")

        today = datetime.now(ET).date()
        todays_games = predictions_df[predictions_df['Date'].dt.date == today]

        for _, row in todays_games.iterrows():
            print(f"{row['Away']} @ {row['Home']}")
            print(f"  Model Spread: {row['pt_spread']:+.1f}")
            print(f"  CI: [{row['ci_lb']:.1f}, {row['ci_ub']:.1f}] (width: {row['ci_width']:.1f})")
            print()
        return

    # Match predictions to markets
    matches = match_predictions_to_markets(predictions_df, markets)

    if not matches:
        print("No matches found between predictions and markets for today.")
        return

    print(f"\nFound {len(matches)} matching games\n")

    # Load existing positions
    positions = load_positions()
    bankroll = load_bankroll()

    for match in matches:
        game_key = match['game_key']
        home = match['home_team']
        away = match['away_team']

        print(f"{away} @ {home}")
        print(f"  Model Spread: {match['model_spread']:+.1f}")
        print(f"  CI: [{match['ci_lb']:.1f}, {match['ci_ub']:.1f}] (width: {match['ci_width']:.1f})")

        # Try spread bet first, then ML as fallback
        bet_type = None
        bet_side = None
        edge = 0
        reasoning = ""
        entry_price = 0

        if match['has_spread']:
            print(f"  Market Spread: {match['home_spread']:+.1f} (home perspective)")
            print(f"  Cover Prices: Home {match['home_cover_price']:.1%} | Away {match['away_cover_price']:.1%}")

            bet_on, edge, reasoning = determine_bet(match)
            print(f"  Edge: {edge:+.1f} pts - {reasoning}")

            if bet_on:
                bet_type = 'spread'
                bet_side = bet_on  # 'home' or 'away'
                entry_price = match['home_cover_price'] if bet_on == 'home' else match['away_cover_price']

        # ML fallback ONLY if no spread market exists AND ML betting is enabled
        if not bet_side and not match['has_spread'] and match['has_ml'] and ENABLE_ML_BETTING:
            print(f"  No spread market - checking ML...")
            print(f"  ML Prices: Away {match['ml_away_price']:.1%} | Home {match['ml_home_price']:.1%}")

            # Bet on the team the MODEL thinks will win (not value betting)
            model_home_prob = spread_to_win_prob(match['model_spread'])
            model_away_prob = 1 - model_home_prob

            print(f"  Model Win Prob: Home {model_home_prob:.1%} | Away {model_away_prob:.1%}")

            # Bet on favorite according to model, if edge exists
            if model_home_prob > 0.5:
                # Model says home wins
                entry_price = match['ml_home_price']
                edge = model_home_prob - entry_price

                # Apply ML filters
                if entry_price > MAX_ML_ENTRY_PRICE:
                    print(f"  ML Skip: Entry {entry_price:.1%} > {MAX_ML_ENTRY_PRICE:.0%} (too expensive)")
                elif entry_price < MIN_ML_ENTRY_PRICE:
                    print(f"  ML Skip: Entry {entry_price:.1%} < {MIN_ML_ENTRY_PRICE:.0%} (too risky)")
                elif edge >= MIN_EDGE_PROB:
                    bet_side = 'home'
                    bet_type = 'ml'
                    reasoning = f"Model {model_home_prob:.1%} > Market {entry_price:.1%}"
                    print(f"  ML Edge: {edge:+.1%} - BET HOME")
                else:
                    print(f"  ML Edge: {edge:+.1%} - too small (need {MIN_EDGE_PROB:.0%}+)")
            else:
                # Model says away wins
                entry_price = match['ml_away_price']
                edge = model_away_prob - entry_price

                # Apply ML filters
                if entry_price > MAX_ML_ENTRY_PRICE:
                    print(f"  ML Skip: Entry {entry_price:.1%} > {MAX_ML_ENTRY_PRICE:.0%} (too expensive)")
                elif entry_price < MIN_ML_ENTRY_PRICE:
                    print(f"  ML Skip: Entry {entry_price:.1%} < {MIN_ML_ENTRY_PRICE:.0%} (too risky)")
                elif edge >= MIN_EDGE_PROB:
                    bet_side = 'away'
                    bet_type = 'ml'
                    reasoning = f"Model {model_away_prob:.1%} > Market {entry_price:.1%}"
                    print(f"  ML Edge: {edge:+.1%} - BET AWAY")
                else:
                    print(f"  ML Edge: {edge:+.1%} - too small (need {MIN_EDGE_PROB:.0%}+)")
        elif not bet_side and not match['has_spread'] and match['has_ml'] and not ENABLE_ML_BETTING:
            print(f"  ML betting disabled - skipping")

        if bet_side is None:
            print(f"  >>> NO BET")
            print()
            continue

        # Check position limits
        can_bet, limit_reason = can_open_position(positions)
        if not can_bet:
            print(f"  >>> SKIP: {limit_reason}")
            print()
            continue

        # Calculate position size using CI-Width sizing
        if bet_type == 'spread':
            bet_size = calculate_ci_width_size(match['ci_width'], edge)
        else:
            # ML: convert prob edge to equivalent points
            equiv_points = abs(edge) / SPREAD_TO_PROB_FACTOR
            bet_size = calculate_ci_width_size(match['ci_width'], equiv_points)

        bet_amount = bankroll * (bet_size / 100)

        print(f"  >>> BET: {bet_type.upper()} {bet_side.upper()} @ {entry_price:.1%} ({bet_size:.1f}% = ${bet_amount:.2f})")

        # Create position - use game_key + date only to prevent duplicates
        position_id = f"{game_key}_{datetime.now().strftime('%Y%m%d')}"

        # Check if we already have a position for this game today
        if position_id in positions:
            print(f"  (Position already exists)")
            print()
            continue

        # Also check for any existing position with same game_key today
        today_str = datetime.now().strftime('%Y%m%d')
        existing = [k for k in positions.keys() if game_key in k and today_str in k]
        if existing:
            print(f"  (Position already exists: {existing[0]})")
            print()
            continue

        position = {
            "game_key": game_key,
            "home_team": home,
            "away_team": away,
            "game_time": match['game_time'],
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "model_spread": match['model_spread'],
            "home_spread": match.get('home_spread'),
            "ci_lb": match['ci_lb'],
            "ci_ub": match['ci_ub'],
            "ci_width": match['ci_width'],
            "bet_type": bet_type,  # 'spread' or 'ml'
            "bet_side": bet_side,  # 'home' or 'away' (which team covers)
            "edge": edge,
            "entry_price": entry_price,
            "bet_size_pct": bet_size,
            "bet_amount": round(bet_amount, 2),
            # Spread prices for monitoring
            "home_cover_price": match.get('home_cover_price'),
            "away_cover_price": match.get('away_cover_price'),
            "is_home_spread": match.get('is_home_spread'),
            # Tokens for the spread market
            "yes_token": match.get('yes_token'),
            "no_token": match.get('no_token'),
            # ML tokens
            "ml_home_token": match.get('ml_home_token'),
            "ml_away_token": match.get('ml_away_token'),
            "ml_home_price": match.get('ml_home_price'),
            "ml_away_price": match.get('ml_away_price'),
            "market_id": match['market_id'],
            "status": "open",
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "pnl": None,
        }

        positions[position_id] = position

        log_trade({
            "type": "ENTRY",
            "time": datetime.now(timezone.utc).isoformat(),
            "position_id": position_id,
            "game": f"{away} @ {home}",
            "bet_type": bet_type,
            "bet_side": bet_side,  # 'home' or 'away'
            "entry_price": entry_price,
            "bet_size_pct": bet_size,
            "bet_amount": round(bet_amount, 2),
            "model_spread": match['model_spread'],
            "home_spread": match.get('home_spread'),
            "edge": edge,
            "ci_width": match['ci_width'],
        })

        print()

    save_positions(positions)

    # Summary
    open_positions = [p for p in positions.values() if p['status'] == 'open']
    bets = [p for p in open_positions if p.get('bet_side')]

    print("=" * 60)
    print(f"Total open positions: {len(open_positions)}")
    print(f"Positions with bets: {len(bets)}")
    if bets:
        total_exposure = sum(p['bet_size_pct'] for p in bets)
        total_amount = sum(p['bet_amount'] for p in bets)
        print(f"Total exposure: {total_exposure:.1f}% (${total_amount:.2f})")
    print(f"Bankroll: ${bankroll:.2f}")


def monitor_positions():
    """Monitor open positions and track current prices."""
    print("=" * 60)
    print("MONITORING CBB POSITIONS")
    print(f"Time: {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}")
    print("=" * 60)

    positions = load_positions()
    open_positions = {k: v for k, v in positions.items() if v['status'] == 'open'}

    if not open_positions:
        print("No open positions to monitor.")
        return

    # Fetch current markets (include started games for monitoring)
    markets = fetch_polymarket_cbb_spreads(include_started=True)

    for position_id, position in open_positions.items():
        game_key = position['game_key']
        home = position['home_team']
        away = position['away_team']
        bet_side = position['bet_side']
        bet_type = position.get('bet_type', 'spread')
        entry_price = position['entry_price']

        print(f"\n{away} @ {home}")
        print(f"  Bet: {bet_type.upper()} {bet_side.upper()} | Size: {position.get('bet_size_pct', 0):.1f}%")

        current_price = None

        # Try to get current price from bulk fetch
        if game_key in markets:
            market = markets[game_key]
            if bet_type == 'ml':
                current_price = market['ml_home_price'] if bet_side == 'home' else market['ml_away_price']
            else:
                # bet_side is 'home' or 'away' - which team we bet to cover
                current_price = market['home_cover_price'] if bet_side == 'home' else market['away_cover_price']
        else:
            # Fallback: try direct market lookup by ID
            market_id = position.get('market_id')
            market_data = fetch_market_price(market_id) if market_id else None

            if market_data:
                if market_data.get('resolved'):
                    print(f"  Market RESOLVED - awaiting result")
                    position['status'] = 'pending_resolution'
                    continue
                else:
                    # Need to map back to home/away using is_home_spread
                    is_home_spread = position.get('is_home_spread', True)
                    if bet_type == 'ml':
                        current_price = market_data['no_price'] if bet_side == 'home' else market_data['yes_price']
                    else:
                        # For spread bets, we need to know if market YES = home or away
                        if is_home_spread:
                            # YES = home covers
                            current_price = market_data['yes_price'] if bet_side == 'home' else market_data['no_price']
                        else:
                            # YES = away covers
                            current_price = market_data['no_price'] if bet_side == 'home' else market_data['yes_price']

        if current_price is not None:
            price_change = (current_price - entry_price) / entry_price
            position['current_price'] = current_price
            position['price_change'] = price_change
            position['last_update'] = datetime.now(timezone.utc).isoformat()

            # Color indicator
            if price_change > 0.05:
                indicator = "+++"
            elif price_change > 0:
                indicator = "+"
            elif price_change > -0.05:
                indicator = "-"
            else:
                indicator = "---"

            print(f"  Entry: {entry_price:.1%} | Current: {current_price:.1%} | Change: {price_change:+.1%} {indicator}")

            # Take-profit logic for ML bets only (spreads ride to resolution)
            if bet_type == 'ml' and price_change > 0:
                take_profit_threshold = get_ml_take_profit_threshold(entry_price)
                print(f"  Take-profit threshold: {take_profit_threshold:.0%}")

                if price_change >= take_profit_threshold:
                    # Calculate P&L
                    bet_amount = position.get('bet_amount', 0)
                    pnl = bet_amount * price_change

                    position['status'] = 'closed'
                    position['exit_time'] = datetime.now(timezone.utc).isoformat()
                    position['exit_price'] = current_price
                    position['exit_reason'] = f"TAKE_PROFIT ({price_change:+.1%} >= {take_profit_threshold:.0%})"
                    position['pnl'] = round(pnl, 2)

                    print(f"  *** TAKE PROFIT TRIGGERED! P&L: ${pnl:+.2f} ***")

                    # Update bankroll
                    bankroll = load_bankroll()
                    bankroll += pnl
                    save_bankroll(bankroll)

                    log_trade({
                        "type": "EXIT",
                        "time": datetime.now(timezone.utc).isoformat(),
                        "position_id": position_id,
                        "game": f"{away} @ {home}",
                        "bet_type": bet_type,
                        "bet_side": bet_side,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "price_change": price_change,
                        "reason": "TAKE_PROFIT",
                        "pnl": round(pnl, 2),
                    })
        else:
            # Check if game likely over
            game_time = datetime.fromisoformat(position['game_time'])
            if datetime.now(timezone.utc) > game_time + timedelta(hours=3):
                position['status'] = 'pending_resolution'
                print(f"  Game ended - awaiting resolution")
            else:
                print(f"  Price unavailable (game in progress)")

    save_positions(positions)


def show_status():
    """Show current positions and P&L summary."""
    print("=" * 60)
    print("CBB PAPER TRADING STATUS")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    positions = load_positions()
    bankroll = load_bankroll()

    print(f"\nStarting Bankroll: ${STARTING_BANKROLL:.2f}")
    print(f"Current Bankroll:  ${bankroll:.2f}")
    print(f"Total P&L:         ${bankroll - STARTING_BANKROLL:+.2f}")

    open_pos = [p for p in positions.values() if p['status'] == 'open']
    closed_pos = [p for p in positions.values() if p['status'] == 'closed']
    resolved_pos = [p for p in positions.values() if p['status'] == 'resolved']

    print(f"\nOpen: {len(open_pos)} | Closed: {len(closed_pos)} | Resolved: {len(resolved_pos)}")

    all_finished = closed_pos + resolved_pos
    if all_finished:
        wins = len([p for p in all_finished if p.get('pnl', 0) > 0])
        losses = len([p for p in all_finished if p.get('pnl', 0) < 0])
        win_rate = wins / len(all_finished) * 100 if all_finished else 0
        print(f"Record: {wins}W - {losses}L ({win_rate:.1f}%)")

    if open_pos:
        print("\n--- OPEN POSITIONS ---")
        for p in open_pos:
            game = f"{p['away_team']} @ {p['home_team']}"
            bet_type = p.get('bet_type', 'spread').upper()
            side = p.get('bet_side', 'none').upper()
            size = p.get('bet_size_pct', 0)
            edge = p.get('edge', p.get('edge_points', 0))
            ci_width = p.get('ci_width', 0)
            print(f"  {game}")
            if p.get('bet_type') == 'ml':
                print(f"    {bet_type} {side} @ {p['entry_price']:.1%} | Size: {size:.1f}% | Edge: {edge:+.1%}")
            else:
                print(f"    {bet_type} {side} @ {p['entry_price']:.1%} | Size: {size:.1f}% | Edge: {edge:+.1f} pts")

    if closed_pos or resolved_pos:
        print("\n--- RECENT RESULTS ---")
        for p in (closed_pos + resolved_pos)[-5:]:  # Last 5
            game = f"{p['away_team']} @ {p['home_team']}"
            pnl = p.get('pnl', 0)
            reason = p.get('exit_reason', 'unknown')
            print(f"  {game}: ${pnl:+.2f} ({reason})")


def generate_report():
    """Generate comprehensive daily report."""
    print("=" * 60)
    print("CBB PAPER TRADING - END OF DAY REPORT")
    print(f"Date: {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}")
    print("=" * 60)

    positions = load_positions()
    bankroll = load_bankroll()

    # Load trades for stats
    trades = []
    if TRADES_LOG.exists():
        with open(TRADES_LOG, "r") as f:
            trades = json.load(f)

    # Categorize positions
    open_pos = [p for p in positions.values() if p['status'] == 'open']
    closed_pos = [p for p in positions.values() if p['status'] == 'closed']
    resolved_pos = [p for p in positions.values() if p['status'] == 'resolved']
    all_finished = closed_pos + resolved_pos

    # Calculate stats
    entries = [t for t in trades if t['type'] == 'ENTRY']
    exits = [t for t in trades if t['type'] in ['EXIT', 'RESOLVED']]
    total_pnl = sum(t.get('pnl', 0) for t in exits)

    # --- BANKROLL SUMMARY ---
    print("\n--- BANKROLL ---")
    print(f"Starting:  ${STARTING_BANKROLL:.2f}")
    print(f"Current:   ${bankroll:.2f}")
    print(f"P&L:       ${bankroll - STARTING_BANKROLL:+.2f}")
    print(f"ROI:       {(bankroll - STARTING_BANKROLL) / STARTING_BANKROLL * 100:+.1f}%")

    # --- TODAY'S ACTIVITY ---
    today_str = datetime.now().strftime('%Y%m%d')
    todays_entries = [t for t in entries if today_str in t.get('position_id', '')]
    todays_exits = [t for t in exits if today_str in t.get('time', '')]

    print("\n--- TODAY'S ACTIVITY ---")
    print(f"Entries: {len(todays_entries)}")
    print(f"Exits:   {len(todays_exits)}")
    if todays_exits:
        todays_pnl = sum(t.get('pnl', 0) for t in todays_exits)
        print(f"P&L:     ${todays_pnl:+.2f}")

    # --- POSITION BREAKDOWN ---
    print("\n--- POSITIONS ---")
    print(f"Open:     {len(open_pos)}")
    print(f"Closed:   {len(closed_pos)}")
    print(f"Resolved: {len(resolved_pos)}")

    # --- WIN/LOSS RECORD ---
    if all_finished:
        wins = [p for p in all_finished if p.get('pnl', 0) > 0]
        losses = [p for p in all_finished if p.get('pnl', 0) < 0]
        pushes = [p for p in all_finished if p.get('pnl', 0) == 0]

        print("\n--- RECORD ---")
        print(f"Wins:   {len(wins)}")
        print(f"Losses: {len(losses)}")
        print(f"Pushes: {len(pushes)}")
        if wins or losses:
            win_rate = len(wins) / (len(wins) + len(losses)) * 100
            print(f"Win Rate: {win_rate:.1f}%")

        # Avg win/loss
        if wins:
            avg_win = sum(p.get('pnl', 0) for p in wins) / len(wins)
            print(f"Avg Win:  ${avg_win:+.2f}")
        if losses:
            avg_loss = sum(p.get('pnl', 0) for p in losses) / len(losses)
            print(f"Avg Loss: ${avg_loss:+.2f}")

    # --- BY BET TYPE ---
    spread_pos = [p for p in all_finished if p.get('bet_type') == 'spread']
    ml_pos = [p for p in all_finished if p.get('bet_type') == 'ml']

    if spread_pos or ml_pos:
        print("\n--- BY BET TYPE ---")
        if spread_pos:
            spread_wins = len([p for p in spread_pos if p.get('pnl', 0) > 0])
            spread_total = len(spread_pos)
            spread_pnl = sum(p.get('pnl', 0) for p in spread_pos)
            print(f"Spread: {spread_wins}/{spread_total} (${spread_pnl:+.2f})")
        if ml_pos:
            ml_wins = len([p for p in ml_pos if p.get('pnl', 0) > 0])
            ml_total = len(ml_pos)
            ml_pnl = sum(p.get('pnl', 0) for p in ml_pos)
            print(f"ML:     {ml_wins}/{ml_total} (${ml_pnl:+.2f})")

    # --- OPEN POSITIONS ---
    if open_pos:
        print("\n--- OPEN POSITIONS ---")
        for p in open_pos:
            game = f"{p['away_team']} @ {p['home_team']}"
            bet_type = p.get('bet_type', 'spread').upper()
            side = p.get('bet_side', '?').upper()
            entry = p.get('entry_price', 0)
            size = p.get('bet_size_pct', 0)
            print(f"  {game}")
            print(f"    {bet_type} {side} @ {entry:.1%} | Size: {size:.1f}%")

    # --- RECENT RESULTS ---
    if all_finished:
        print("\n--- RECENT RESULTS ---")
        for p in all_finished[-10:]:  # Last 10
            game = f"{p['away_team']} @ {p['home_team']}"
            pnl = p.get('pnl', 0)
            result = "W" if pnl > 0 else "L" if pnl < 0 else "P"
            reason = p.get('exit_reason', '')[:20]
            print(f"  [{result}] {game}: ${pnl:+.2f} ({reason})")

    # --- SAVE REPORT ---
    report_file = DATA_DIR / f"report_{datetime.now().strftime('%Y-%m-%d')}.txt"
    with open(report_file, 'w') as f:
        f.write(f"CBB Paper Trading Report - {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Bankroll: ${bankroll:.2f}\n")
        f.write(f"Starting: ${STARTING_BANKROLL:.2f}\n")
        f.write(f"P&L: ${bankroll - STARTING_BANKROLL:+.2f}\n")
        f.write(f"ROI: {(bankroll - STARTING_BANKROLL) / STARTING_BANKROLL * 100:+.1f}%\n\n")

        f.write(f"Positions: {len(open_pos)} open, {len(closed_pos)} closed, {len(resolved_pos)} resolved\n")

        if all_finished:
            wins = len([p for p in all_finished if p.get('pnl', 0) > 0])
            losses = len([p for p in all_finished if p.get('pnl', 0) < 0])
            f.write(f"Record: {wins}W - {losses}L\n")

        f.write(f"\n--- Open Positions ---\n")
        for p in open_pos:
            game = f"{p['away_team']} @ {p['home_team']}"
            bet_type = p.get('bet_type', 'spread').upper()
            side = p.get('bet_side', '?').upper()
            f.write(f"  {game}: {bet_type} {side}\n")

    print(f"\nReport saved: {report_file}")


def resolve_positions():
    """
    Resolve pending positions by checking if markets have settled.

    A market is considered resolved when the price is very close to 0 or 1:
    - Price >= 0.95: Position WON (bet amount * (1/entry_price - 1))
    - Price <= 0.05: Position LOST (lost bet amount)

    Updates bankroll and marks positions as resolved.
    """
    print("=" * 60)
    print("RESOLVING CBB POSITIONS")
    print(f"Time: {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}")
    print("=" * 60)

    positions = load_positions()
    pending = {k: v for k, v in positions.items() if v['status'] == 'pending_resolution'}

    if not pending:
        print("No positions pending resolution.")
        return 0

    print(f"Found {len(pending)} positions pending resolution\n")

    bankroll = load_bankroll()
    initial_bankroll = bankroll
    resolved_count = 0

    for position_id, position in pending.items():
        home = position['home_team']
        away = position['away_team']
        bet_side = position.get('bet_side', 'unknown')
        bet_type = position.get('bet_type', 'spread')
        entry_price = position.get('entry_price', 0.5)
        bet_amount = position.get('bet_amount', 0)
        current_price = position.get('current_price')

        print(f"{away} @ {home}")
        print(f"  Bet: {bet_type.upper()} {bet_side.upper()} @ {entry_price:.1%}")
        print(f"  Amount: ${bet_amount:.2f}")

        # Try to fetch current market price if we don't have it
        if current_price is None:
            market_id = position.get('market_id')
            if market_id:
                market_data = fetch_market_price(market_id)
                if market_data:
                    # Determine which price to use based on bet type and side
                    is_home_spread = position.get('is_home_spread', True)
                    if bet_type == 'ml':
                        # For ML bets, yes_price is typically the away team
                        current_price = market_data['no_price'] if bet_side == 'home' else market_data['yes_price']
                    else:
                        # For spread bets
                        if is_home_spread:
                            current_price = market_data['yes_price'] if bet_side == 'home' else market_data['no_price']
                        else:
                            current_price = market_data['no_price'] if bet_side == 'home' else market_data['yes_price']
                    position['current_price'] = current_price

        if current_price is None:
            print(f"  Cannot resolve - no current price available")
            print()
            continue

        print(f"  Current Price: {current_price:.4f}")

        # Check if market has settled
        pnl = None
        if current_price >= 0.95:
            # Position WON - we bought shares at entry_price that are now worth ~1
            # P&L = bet_amount * (1/entry_price - 1) = bet_amount * (1 - entry_price) / entry_price
            # Simplified: we get back bet_amount / entry_price, so profit = (1/entry_price - 1) * bet_amount
            payout = bet_amount / entry_price
            pnl = payout - bet_amount
            print(f"  >>> WON! Payout: ${payout:.2f} | P&L: ${pnl:+.2f}")
            position['exit_reason'] = "RESOLVED_WIN"
        elif current_price <= 0.05:
            # Position LOST - shares worth ~0
            pnl = -bet_amount
            print(f"  >>> LOST! P&L: ${pnl:+.2f}")
            position['exit_reason'] = "RESOLVED_LOSS"
        else:
            print(f"  Market not yet settled (price between 0.05 and 0.95)")
            print()
            continue

        # Update position
        position['status'] = 'resolved'
        position['exit_time'] = datetime.now(timezone.utc).isoformat()
        position['exit_price'] = current_price
        position['pnl'] = round(pnl, 2)

        # Update bankroll
        bankroll += pnl

        # Log the resolution
        log_trade({
            "type": "RESOLVED",
            "time": datetime.now(timezone.utc).isoformat(),
            "position_id": position_id,
            "game": f"{away} @ {home}",
            "bet_type": bet_type,
            "bet_side": bet_side,
            "entry_price": entry_price,
            "exit_price": current_price,
            "result": "WIN" if pnl > 0 else "LOSS",
            "pnl": round(pnl, 2),
        })

        resolved_count += 1
        print()

    # Save updated positions and bankroll
    save_positions(positions)
    save_bankroll(bankroll)

    # Summary
    print("=" * 60)
    print(f"Resolved: {resolved_count} positions")
    print(f"Bankroll: ${initial_bankroll:.2f} -> ${bankroll:.2f}")
    print(f"Session P&L: ${bankroll - initial_bankroll:+.2f}")
    print("=" * 60)

    return resolved_count


def run_continuous():
    """
    Run the paper trader continuously.

    - Places bets ~30 minutes before each game
    - Monitors all positions every 5 minutes
    - Runs until manually stopped (Ctrl+C)
    """
    print("=" * 60)
    print("CBB PAPER TRADER - CONTINUOUS MODE")
    print(f"Started: {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}")
    print(f"Bet timing: {MINUTES_BEFORE_GAME} minutes before game")
    print(f"Check interval: {MONITORING_INTERVAL // 60} minutes")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Load predictions once at startup
    print("\nLoading predictions...")
    try:
        predictions_df = load_predictions()
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return

    iteration = 0
    while True:
        iteration += 1
        now = datetime.now(ET)
        print(f"\n{'='*60}")
        print(f"[{now.strftime('%I:%M %p ET')}] Check #{iteration}")
        print("=" * 60)

        try:
            # 1. Check for games starting soon and place bets
            print(f"\n--- Checking for games starting within {MINUTES_BEFORE_GAME} min ---")
            markets = fetch_polymarket_cbb_spreads(minutes_before_game=MINUTES_BEFORE_GAME)

            if markets:
                print(f"Found {len(markets)} games starting soon")
                # Match with predictions and place bets
                matches = match_predictions_to_markets(predictions_df, markets)

                positions = load_positions()
                bankroll = load_bankroll()

                for match in matches:
                    game_key = match['game_key']
                    position_id = f"{game_key}_{datetime.now().strftime('%Y%m%d')}"

                    # Skip if already have position
                    if position_id in positions:
                        print(f"  {match['away_team']} @ {match['home_team']} - already have position")
                        continue

                    # Determine bet
                    if match['has_spread']:
                        bet_on, edge, reasoning = determine_bet(match)
                        if bet_on:
                            entry_price = match['home_cover_price'] if bet_on == 'home' else match['away_cover_price']
                            bet_size = calculate_ci_width_size(match['ci_width'], edge)
                            bet_amount = bankroll * (bet_size / 100)

                            position = {
                                "game_key": game_key,
                                "home_team": match['home_team'],
                                "away_team": match['away_team'],
                                "game_time": match['game_time'],
                                "entry_time": datetime.now(timezone.utc).isoformat(),
                                "model_spread": match['model_spread'],
                                "home_spread": match.get('home_spread'),
                                "ci_width": match['ci_width'],
                                "bet_type": "spread",
                                "bet_side": bet_on,  # 'home' or 'away'
                                "edge": edge,
                                "entry_price": entry_price,
                                "bet_size_pct": bet_size,
                                "bet_amount": round(bet_amount, 2),
                                "home_cover_price": match.get('home_cover_price'),
                                "away_cover_price": match.get('away_cover_price'),
                                "is_home_spread": match.get('is_home_spread'),
                                "yes_token": match.get('yes_token'),
                                "no_token": match.get('no_token'),
                                "market_id": match['market_id'],
                                "status": "open",
                            }

                            positions[position_id] = position
                            print(f"  BET PLACED: {match['away_team']} @ {match['home_team']}")
                            print(f"    {bet_on.upper()} COVERS @ {entry_price:.1%} | {bet_size:.1f}% (${bet_amount:.2f})")
                            print(f"    Edge: {edge:+.1f} pts | {reasoning}")

                            log_trade({
                                "type": "ENTRY",
                                "time": datetime.now(timezone.utc).isoformat(),
                                "position_id": position_id,
                                "game": f"{match['away_team']} @ {match['home_team']}",
                                "bet_type": "spread",
                                "bet_side": bet_on,  # 'home' or 'away'
                                "entry_price": entry_price,
                                "bet_size_pct": bet_size,
                                "bet_amount": round(bet_amount, 2),
                                "model_spread": match['model_spread'],
                                "home_spread": match.get('home_spread'),
                                "edge": edge,
                            })
                        else:
                            print(f"  {match['away_team']} @ {match['home_team']} - no edge ({reasoning})")

                save_positions(positions)
            else:
                print("No games starting soon")

            # 2. Monitor existing positions
            print(f"\n--- Monitoring open positions ---")
            monitor_positions()

            # 3. Show quick summary
            positions = load_positions()
            open_pos = [p for p in positions.values() if p['status'] == 'open']
            bankroll = load_bankroll()
            print(f"\n--- Summary ---")
            print(f"Open positions: {len(open_pos)}")
            print(f"Bankroll: ${bankroll:.2f}")

        except Exception as e:
            print(f"Error in iteration: {e}")
            import traceback
            traceback.print_exc()

        # Sleep until next check
        print(f"\nNext check in {MONITORING_INTERVAL // 60} minutes...")
        time.sleep(MONITORING_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description='CBB Paper Trading Bot')
    parser.add_argument('--init', action='store_true', help='Initialize positions for today (all games)')
    parser.add_argument('--run', action='store_true', help='Run continuously (place bets 30min before games)')
    parser.add_argument('--monitor', action='store_true', help='Monitor open positions once')
    parser.add_argument('--resolve', action='store_true', help='Resolve pending positions and update bankroll')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--report', action='store_true', help='Generate daily report')

    args = parser.parse_args()

    if args.run:
        run_continuous()
    elif args.init:
        init_positions()
    elif args.monitor:
        monitor_positions()
    elif args.resolve:
        resolve_positions()
    elif args.status:
        show_status()
    elif args.report:
        generate_report()
    else:
        # Default: show status
        show_status()


if __name__ == "__main__":
    main()
