"""
Unified Data Loader for NBA + CBB Paper Trading Dashboard

Loads and normalizes position data from both leagues into a unified schema.
Includes ESPN live win probability integration.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ESPN Provider for live win probability
try:
    from src.DataProviders.ESPNProvider import ESPNProvider
    from src.DataProviders.ESPNLiveTracker import get_live_tracker, ESPNLiveTracker
    from src.DataProviders.espn_wp_logger import get_espn_wp_logger
    ESPN_AVAILABLE = True
except ImportError:
    ESPN_AVAILABLE = False
    ESPNLiveTracker = None

# Data directories
NBA_DATA_DIR = Path("Data/paper_trading_v2")
CBB_DATA_DIR = Path("Data/cbb_paper_trading")
SOCCER_DATA_DIR = Path("Data/soccer_paper_trading")

STARTING_BANKROLL = 1000


def load_json_file(filepath: Path) -> Optional[Any]:
    """Load JSON file safely."""
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def load_jsonl_file(filepath: Path, limit: int = 100) -> List[Dict]:
    """Load JSONL file (last N lines)."""
    if not filepath.exists():
        return []

    lines = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return lines[-limit:]


def _get_data_dir(league: str) -> Path:
    """Get data directory for a league."""
    if league == "NBA":
        return NBA_DATA_DIR
    elif league == "CBB":
        return CBB_DATA_DIR
    elif league == "SOCCER":
        return SOCCER_DATA_DIR
    return NBA_DATA_DIR


def load_bankroll(league: str) -> float:
    """Load current bankroll for a league."""
    data = load_json_file(_get_data_dir(league) / "bankroll.json")
    if data:
        return data.get("bankroll", STARTING_BANKROLL)
    return STARTING_BANKROLL


def load_positions(league: str) -> Dict:
    """Load all positions for a league."""
    return load_json_file(_get_data_dir(league) / "positions.json") or {}


def load_trades(league: str) -> List[Dict]:
    """Load trade history for a league."""
    return load_json_file(_get_data_dir(league) / "trades.json") or []


def load_drawdown_state() -> Dict:
    """Load NBA drawdown manager state."""
    return load_json_file(NBA_DATA_DIR / "drawdown_state.json") or {}


def load_alerts(limit: int = 50) -> List[Dict]:
    """Load recent NBA alerts."""
    return load_jsonl_file(NBA_DATA_DIR / "alerts.jsonl", limit)


def normalize_nba_position(pos_id: str, pos: Dict, bankroll: float = None) -> Dict:
    """Normalize NBA V2 position to unified schema."""
    bet_side = pos.get('bet_side', '')
    bet_type = "ML"  # NBA is always moneyline
    if bankroll is None:
        bankroll = STARTING_BANKROLL

    # V2 fields: entry_price, bet_edge, model_prob, leg, is_favorite
    entry_price = pos.get('entry_price', 0)
    edge = pos.get('bet_edge', 0)
    amount = pos.get('bet_amount', 0)
    leg = pos.get('leg', '?')
    is_favorite = pos.get('is_favorite', False)

    # Fallback for V1 positions still in data
    if not entry_price:
        if bet_side == 'home':
            entry_price = pos.get('entry_home_prob', 0)
            edge = pos.get('home_edge', 0)
        elif bet_side == 'away':
            entry_price = pos.get('entry_away_prob', 0)
            edge = pos.get('away_edge', 0)

    # Current price from monitoring
    current_price = pos.get('current_pm_price', entry_price)
    if entry_price and current_price:
        price_change = (current_price - entry_price) / entry_price
    else:
        price_change = 0

    # P&L
    if pos.get('status') == 'open' and bet_side:
        unrealized_pnl = pos.get('unrealized_pnl', amount * price_change if price_change else 0)
    else:
        unrealized_pnl = pos.get('pnl', 0) if pos.get('pnl') is not None else 0

    # Leg display in bet_type
    leg_display = "FAV" if is_favorite else "DOG"

    return {
        "id": pos_id,
        "league": "NBA",
        "game": f"{pos.get('away_team', '?')} @ {pos.get('home_team', '?')}",
        "game_time": pos.get('game_time', ''),
        "bet_type": f"ML ({leg_display})",
        "bet_side": bet_side.upper() if bet_side else "-",
        "entry_price": entry_price,
        "current_price": current_price,
        "price_change_pct": price_change * 100 if price_change else 0,
        "size_pct": (amount / bankroll * 100) if bankroll and amount else 0,
        "amount": amount,
        "edge": edge,
        "edge_display": f"{edge*100:+.1f}%" if edge else "-",
        "status": pos.get('status', 'unknown').upper(),
        "pnl": unrealized_pnl,
        "won": pos.get('won'),
        "entry_time": pos.get('entry_time', ''),
        "leg": leg,
        "is_favorite": is_favorite,
    }


def normalize_cbb_position(pos_id: str, pos: Dict) -> Dict:
    """Normalize CBB position to unified schema."""
    bet_type = pos.get('bet_type', 'spread').upper()
    bet_side = pos.get('bet_side', '')

    entry_price = pos.get('entry_price', 0)
    current_price = pos.get('current_price', entry_price)
    price_change = pos.get('price_change', 0)

    # Edge display depends on bet type
    edge = pos.get('edge', 0)
    if bet_type == "SPREAD":
        edge_display = f"{edge:+.1f} pts"
    else:
        edge_display = f"{edge*100:+.1f}%"

    # Map bet_side for display
    if bet_side == 'home' or bet_side == 'no':
        side_display = "HOME"
    elif bet_side == 'away' or bet_side == 'yes':
        side_display = "AWAY"
    else:
        side_display = bet_side.upper() if bet_side else "-"

    return {
        "id": pos_id,
        "league": "CBB",
        "game": f"{pos.get('away_team', '?')} @ {pos.get('home_team', '?')}",
        "game_time": pos.get('game_time', ''),
        "bet_type": bet_type,
        "bet_side": side_display,
        "entry_price": entry_price,
        "current_price": current_price,
        "price_change_pct": price_change * 100 if price_change else 0,
        "size_pct": pos.get('bet_size_pct', 0),
        "amount": pos.get('bet_amount', 0),
        "edge": edge,
        "edge_display": edge_display,
        "status": pos.get('status', 'unknown').upper(),
        "pnl": pos.get('pnl', 0) if pos.get('status') in ['resolved', 'closed'] else (pos.get('bet_amount', 0) * price_change if price_change else 0),
        "won": True if pos.get('exit_reason', '').endswith('WIN') else (False if pos.get('exit_reason', '').endswith('LOSS') else None),
        "entry_time": pos.get('entry_time', ''),
    }


def normalize_soccer_position(pos_id: str, pos: Dict) -> Dict:
    """Normalize Soccer position to unified schema."""
    entry_price = pos.get('entry_draw_price', 0)
    exit_price = pos.get('exit_price')
    bet_amount = pos.get('bet_amount', 0)
    pnl = pos.get('pnl', 0) or 0

    # Current price: use exit_price if resolved, otherwise entry
    if pos.get('status') != 'open' and exit_price:
        current_price = exit_price
    else:
        current_price = entry_price

    price_change = ((current_price - entry_price) / entry_price * 100) if entry_price else 0

    # Determine win/loss
    status = pos.get('status', 'open')
    if status == 'open':
        status_display = 'OPEN'
        won = None
    elif 'win' in status:
        status_display = 'RESOLVED'
        won = True
    elif 'loss' in status:
        status_display = 'RESOLVED'
        won = False
    else:
        status_display = 'RESOLVED'
        won = pos.get('was_draw')

    return {
        "id": pos_id,
        "league": "SOCCER",
        "game": f"{pos.get('home_team', '?')} vs {pos.get('away_team', '?')}",
        "game_time": pos.get('entry_time', ''),
        "bet_type": "DRAW",
        "bet_side": "DRAW",
        "entry_price": entry_price,
        "current_price": current_price,
        "price_change_pct": price_change,
        "size_pct": pos.get('bet_kelly', 0),
        "amount": bet_amount,
        "edge": pos.get('edge', 0),
        "edge_display": f"{pos.get('edge', 0)*100:+.1f}%",
        "status": status_display,
        "pnl": pnl,
        "won": won,
        "entry_time": pos.get('entry_time', ''),
        "league_detail": pos.get('league', ''),
        "minute": pos.get('minute_of_signal', 0),
        "momentum": pos.get('momentum_value', 0),
        "score_at_entry": f"{pos.get('home_score_at_entry', 0)}-{pos.get('away_score_at_entry', 0)}",
        "exit_reason": pos.get('exit_reason', ''),
    }


def get_unified_positions() -> List[Dict]:
    """Get all positions from both leagues in unified format."""
    positions = []

    # Load NBA positions with actual bankroll for accurate P&L
    nba_bankroll = load_bankroll("NBA")
    nba_positions = load_positions("NBA")
    for pos_id, pos in nba_positions.items():
        if pos.get('bet_side'):  # Only include positions with actual bets
            positions.append(normalize_nba_position(pos_id, pos, bankroll=nba_bankroll))

    # Load CBB positions
    cbb_positions = load_positions("CBB")
    for pos_id, pos in cbb_positions.items():
        positions.append(normalize_cbb_position(pos_id, pos))

    # Load Soccer positions
    soccer_positions = load_positions("SOCCER")
    for pos_id, pos in soccer_positions.items():
        positions.append(normalize_soccer_position(pos_id, pos))

    return positions


def get_portfolio_summary() -> Dict:
    """Get portfolio summary with separate NBA, CBB, and Soccer metrics."""
    nba_bankroll = load_bankroll("NBA")
    cbb_bankroll = load_bankroll("CBB")
    soccer_bankroll = load_bankroll("SOCCER")

    nba_pnl = nba_bankroll - STARTING_BANKROLL
    cbb_pnl = cbb_bankroll - STARTING_BANKROLL
    soccer_pnl = soccer_bankroll - STARTING_BANKROLL
    nba_roi = (nba_pnl / STARTING_BANKROLL) * 100
    cbb_roi = (cbb_pnl / STARTING_BANKROLL) * 100
    soccer_roi = (soccer_pnl / STARTING_BANKROLL) * 100

    total_bankroll = nba_bankroll + cbb_bankroll + soccer_bankroll
    total_pnl = nba_pnl + cbb_pnl + soccer_pnl
    roi = (total_pnl / (3 * STARTING_BANKROLL)) * 100

    # Get position counts
    positions = get_unified_positions()
    open_positions = [p for p in positions if p['status'] == 'OPEN']
    resolved_positions = [p for p in positions if p['status'] == 'RESOLVED']

    nba_open = len([p for p in open_positions if p['league'] == 'NBA'])
    cbb_open = len([p for p in open_positions if p['league'] == 'CBB'])
    soccer_open = len([p for p in open_positions if p['league'] == 'SOCCER'])

    # Separate win rates
    nba_resolved = [p for p in resolved_positions if p['league'] == 'NBA']
    cbb_resolved = [p for p in resolved_positions if p['league'] == 'CBB']
    soccer_resolved = [p for p in resolved_positions if p['league'] == 'SOCCER']

    nba_wins = len([p for p in nba_resolved if p.get('won') is True])
    nba_losses = len([p for p in nba_resolved if p.get('won') is False])
    nba_total = nba_wins + nba_losses
    nba_win_rate = (nba_wins / nba_total * 100) if nba_total > 0 else 0

    cbb_wins = len([p for p in cbb_resolved if p.get('won') is True])
    cbb_losses = len([p for p in cbb_resolved if p.get('won') is False])
    cbb_total = cbb_wins + cbb_losses
    cbb_win_rate = (cbb_wins / cbb_total * 100) if cbb_total > 0 else 0

    soccer_wins = len([p for p in soccer_resolved if p.get('won') is True])
    soccer_losses = len([p for p in soccer_resolved if p.get('won') is False])
    soccer_total = soccer_wins + soccer_losses
    soccer_win_rate = (soccer_wins / soccer_total * 100) if soccer_total > 0 else 0

    # Overall win rate
    wins = nba_wins + cbb_wins + soccer_wins
    losses = nba_losses + cbb_losses + soccer_losses
    total_resolved = wins + losses
    win_rate = (wins / total_resolved * 100) if total_resolved > 0 else 0

    return {
        "total_bankroll": total_bankroll,
        "nba_bankroll": nba_bankroll,
        "cbb_bankroll": cbb_bankroll,
        "soccer_bankroll": soccer_bankroll,
        "nba_pnl": nba_pnl,
        "cbb_pnl": cbb_pnl,
        "soccer_pnl": soccer_pnl,
        "nba_roi": nba_roi,
        "cbb_roi": cbb_roi,
        "soccer_roi": soccer_roi,
        "total_pnl": total_pnl,
        "roi": roi,
        "open_positions": len(open_positions),
        "nba_open": nba_open,
        "cbb_open": cbb_open,
        "soccer_open": soccer_open,
        "nba_wins": nba_wins,
        "nba_losses": nba_losses,
        "nba_win_rate": nba_win_rate,
        "cbb_wins": cbb_wins,
        "cbb_losses": cbb_losses,
        "cbb_win_rate": cbb_win_rate,
        "soccer_wins": soccer_wins,
        "soccer_losses": soccer_losses,
        "soccer_win_rate": soccer_win_rate,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    }


def get_pnl_history() -> Dict[str, List[Dict]]:
    """Get P&L history for all leagues for charting."""
    result = {"NBA": [], "CBB": [], "SOCCER": []}

    # NBA trades
    nba_trades = load_trades("NBA")
    cumulative_nba = 0
    for t in nba_trades:
        if t.get('type') in ['RESOLVED', 'EXIT', 'CLOSE', 'PARTIAL_CLOSE'] and t.get('pnl') is not None:
            cumulative_nba += t.get('pnl', 0)
            time_str = t.get('time', '')
            if time_str:
                result["NBA"].append({
                    "time": time_str,
                    "pnl": t.get('pnl', 0),
                    "cumulative": cumulative_nba,
                    "game": t.get('game', ''),
                    "type": t.get('type', ''),
                })

    # CBB trades
    cbb_trades = load_trades("CBB")
    cumulative_cbb = 0
    for t in cbb_trades:
        if t.get('type') == 'RESOLVED' and t.get('pnl') is not None:
            cumulative_cbb += t.get('pnl', 0)
            time_str = t.get('time', '')
            if time_str:
                result["CBB"].append({
                    "time": time_str,
                    "pnl": t.get('pnl', 0),
                    "cumulative": cumulative_cbb,
                    "game": t.get('game', ''),
                    "result": t.get('result', ''),
                })

    # Soccer trades
    soccer_trades = load_trades("SOCCER")
    cumulative_soccer = 0
    for t in soccer_trades:
        if t.get('type') in ['RESOLVED', 'EXIT_EQUALIZATION'] and t.get('pnl') is not None:
            cumulative_soccer += t.get('pnl', 0)
            time_str = t.get('time', '')
            if time_str:
                result["SOCCER"].append({
                    "time": time_str,
                    "pnl": t.get('pnl', 0),
                    "cumulative": cumulative_soccer,
                    "game": t.get('match', t.get('game', '')),
                    "type": t.get('type', ''),
                })

    return result


def get_recent_activity(limit: int = 15) -> List[Dict]:
    """Get recent activity from both leagues."""
    activity = []

    # NBA trades
    nba_trades = load_trades("NBA")
    for t in nba_trades[-limit:]:
        trade_type = t.get('type', '')
        pnl = t.get('pnl')

        if trade_type == 'RESOLVED':
            badge = "WIN" if t.get('won') else "LOSS"
        elif trade_type == 'EXIT':
            badge = "EXIT"
        elif trade_type == 'ENTRY':
            badge = "ENTRY"
        else:
            badge = trade_type

        activity.append({
            "league": "NBA",
            "time": t.get('time', ''),
            "badge": badge,
            "game": t.get('game', ''),
            "pnl": pnl,
            "reason": t.get('reason', ''),
        })

    # CBB trades
    cbb_trades = load_trades("CBB")
    for t in cbb_trades[-limit:]:
        trade_type = t.get('type', '')
        pnl = t.get('pnl')

        if trade_type == 'RESOLVED':
            badge = "WIN" if t.get('result') == 'WIN' else "LOSS"
        elif trade_type == 'ENTRY':
            badge = "ENTRY"
        else:
            badge = trade_type

        activity.append({
            "league": "CBB",
            "time": t.get('time', ''),
            "badge": badge,
            "game": t.get('game', ''),
            "pnl": pnl,
        })

    # Soccer trades
    soccer_trades = load_trades("SOCCER")
    for t in soccer_trades[-limit:]:
        trade_type = t.get('type', '')
        pnl = t.get('pnl')

        if trade_type == 'EXIT_EQUALIZATION':
            badge = "WIN"
        elif trade_type == 'RESOLVED':
            badge = "WIN" if t.get('was_draw') else "LOSS"
        elif trade_type == 'ENTRY':
            badge = "ENTRY"
        else:
            badge = trade_type

        activity.append({
            "league": "SOCCER",
            "time": t.get('time', ''),
            "badge": badge,
            "game": t.get('match', t.get('game', '')),
            "pnl": pnl,
        })

    # Sort by time and return most recent
    activity.sort(key=lambda x: x.get('time', ''), reverse=True)
    return activity[:limit]


def get_edge_analysis() -> List[Dict]:
    """Analyze performance by edge bucket."""
    positions = get_unified_positions()
    resolved = [p for p in positions if p['status'] == 'RESOLVED' and p.get('won') is not None]

    buckets = {
        "0-5%": {"min": 0, "max": 0.05, "wins": 0, "total": 0, "pnl": 0},
        "5-10%": {"min": 0.05, "max": 0.10, "wins": 0, "total": 0, "pnl": 0},
        "10-15%": {"min": 0.10, "max": 0.15, "wins": 0, "total": 0, "pnl": 0},
        "15%+": {"min": 0.15, "max": 1.0, "wins": 0, "total": 0, "pnl": 0},
    }

    for p in resolved:
        edge = abs(p.get('edge', 0))
        for name, bucket in buckets.items():
            if bucket['min'] <= edge < bucket['max']:
                bucket['total'] += 1
                if p.get('won'):
                    bucket['wins'] += 1
                bucket['pnl'] += p.get('pnl', 0)
                break

    result = []
    for name, data in buckets.items():
        if data['total'] > 0:
            result.append({
                "bucket": name,
                "bets": data['total'],
                "wins": data['wins'],
                "win_rate": data['wins'] / data['total'] * 100,
                "pnl": data['pnl'],
            })

    return result


def get_live_game_updates(league: str = 'nba', event_ids: List[str] = None) -> Dict[str, Dict]:
    """
    Get real-time win probability updates using the live tracker.

    This polls ESPN only when there are new plays, minimizing API calls.
    Updates happen every ~15 seconds during live games.

    Args:
        league: 'nba' or 'cbb'
        event_ids: Optional list of specific event IDs to track

    Returns:
        Dict of event_id -> game state with latest probability
    """
    if not ESPN_AVAILABLE or ESPNLiveTracker is None:
        return {}

    try:
        tracker = get_live_tracker(league)

        # Add any new event_ids to track
        if event_ids:
            for eid in event_ids:
                tracker.track_game(eid)

        # Poll for updates (synchronous)
        updates = tracker.poll_once()

        # Return all current states (not just updates)
        return tracker.get_all_states()

    except Exception as e:
        print(f"Live tracker error: {e}")
        return {}


def get_espn_live_probabilities(league: str = 'nba') -> Dict[str, Dict]:
    """
    Get ESPN's live win probabilities for today's games.

    Returns dict keyed by "away @ home" with:
        - espn_home_prob: ESPN model's home win probability
        - espn_away_prob: ESPN model's away win probability
        - home_score, away_score: Current scores
        - period, clock: Game time
        - is_live: Whether game is in progress
        - plays_count: Number of probability updates
        - probability_history: List of (home_prob) for sparkline
    """
    if not ESPN_AVAILABLE:
        return {}

    try:
        provider = ESPNProvider(league)
        games = provider.get_all_live_win_probabilities()

        result = {}
        for game_key, data in games.items():
            # Normalize game key to "away @ home" format
            home = data.get('home_team', '')
            away = data.get('away_team', '')
            normalized_key = f"{away} @ {home}"

            # Get probability history for sparkline (sample every ~50 plays)
            prob_history = []
            if data.get('event_id') and data.get('plays_count', 0) > 1:
                full_prob = provider.get_live_win_probability(data['event_id'])
                if full_prob and full_prob.get('history'):
                    history = full_prob['history']
                    # Sample ~20 points for sparkline
                    step = max(1, len(history) // 20)
                    prob_history = [h.get('homeWinPercentage', 0.5) for h in history[::step]]

            # Only include games with valid probability data
            home_prob = data.get('home_win_prob')
            if home_prob is not None:
                result[normalized_key] = {
                    'event_id': data.get('event_id'),
                    'home_team': home,
                    'away_team': away,
                    'espn_home_prob': home_prob,
                    'espn_away_prob': data.get('away_win_prob', 1 - home_prob),
                    'home_score': data.get('home_score', 0),
                    'away_score': data.get('away_score', 0),
                    'period': data.get('period', 0),
                    'clock': data.get('clock', ''),
                    'is_live': data.get('is_live', False),
                    'is_final': data.get('is_final', False),
                    'plays_count': data.get('plays_count', 0),
                    'last_play': data.get('last_play', ''),
                    'probability_history': prob_history,
                }

                # Log ESPN WP for backtest analysis
                try:
                    from datetime import date as _date
                    get_espn_wp_logger().log(
                        _date.today().isoformat(), home, away,
                        home_prob, data.get('away_win_prob', 1 - home_prob),
                        home_score=data.get('home_score', 0),
                        away_score=data.get('away_score', 0),
                        period=data.get('period', 0),
                        clock=data.get('clock', ''),
                        source="dashboard",
                    )
                except Exception:
                    pass

        return result

    except Exception as e:
        print(f"ESPN error: {e}")
        return {}


# Long-term injury cache (persists for hours)
_injury_cache: Dict[str, Dict] = {}  # team -> {data, timestamp}
INJURY_CACHE_TTL = 3600  # 1 hour - injuries don't change often


def get_espn_injuries_for_teams(teams: List[str], league: str = 'nba', force_refresh: bool = False) -> Dict[str, List[Dict]]:
    """
    Get ESPN injury data for a list of teams.

    Injuries are cached for 1 hour since they rarely change during games.
    Use force_refresh=True to bypass cache (e.g., when placing a new bet).

    Returns dict keyed by team name with list of injuries.
    """
    global _injury_cache

    if not ESPN_AVAILABLE:
        return {}

    try:
        provider = ESPNProvider(league)
        results = {}
        now = datetime.now().timestamp()

        for team in teams:
            cache_key = f"{league}:{team}"

            # Check cache unless force refresh
            if not force_refresh and cache_key in _injury_cache:
                cached = _injury_cache[cache_key]
                if now - cached['timestamp'] < INJURY_CACHE_TTL:
                    results[team] = cached['data']
                    continue

            # Fetch fresh data
            injuries = provider.get_team_injuries(team)
            if injuries:
                parsed = []
                for inj in injuries:
                    details = inj.get('details', {})
                    parsed.append({
                        'status': inj.get('type', {}).get('description', inj.get('status', 'Unknown')),
                        'injury_type': details.get('type', 'Unknown'),
                        'detail': details.get('detail', ''),
                        'side': details.get('side', ''),
                        'return_date': details.get('returnDate'),
                        'news': inj.get('shortComment', ''),
                        'updated': inj.get('date', ''),
                    })
                results[team] = parsed

                # Cache it
                _injury_cache[cache_key] = {
                    'data': parsed,
                    'timestamp': now
                }
            else:
                results[team] = []
                _injury_cache[cache_key] = {'data': [], 'timestamp': now}

        return results

    except Exception as e:
        print(f"ESPN injury error: {e}")
        return {}


def fetch_injuries_for_position(home_team: str, away_team: str, league: str = 'nba') -> Dict[str, List[Dict]]:
    """
    Fetch injuries for both teams when placing a bet.

    This is called once when entering a position and the data is cached.
    """
    return get_espn_injuries_for_teams([home_team, away_team], league, force_refresh=True)


def close_position_partial(league: str, position_id: str, close_pct: float, reason: str = "MANUAL_CLOSE") -> bool:
    """
    Close a percentage of an open position.

    Args:
        league: 'NBA' or 'CBB'
        position_id: The position ID to close
        close_pct: Percentage to close (0.0 to 1.0)
        reason: Reason for closing

    Returns:
        True if successful, False otherwise
    """
    data_dir = NBA_DATA_DIR if league.upper() == "NBA" else CBB_DATA_DIR
    positions_file = data_dir / "positions.json"
    trades_file = data_dir / "trades.json"
    bankroll_file = data_dir / "bankroll.json"

    try:
        # Load positions
        positions = load_json_file(positions_file) or {}
        if position_id not in positions:
            return False

        pos = positions[position_id]
        if pos.get('status') != 'open':
            return False

        # Calculate close amount
        original_kelly = pos.get('bet_kelly', 0)
        close_kelly = original_kelly * close_pct
        remaining_kelly = original_kelly * (1 - close_pct)

        # Get current price for P&L calculation
        price_change = pos.get('current_price_change', 0)
        close_amount = (close_kelly / 100) * STARTING_BANKROLL
        pnl = close_amount * price_change

        # Update bankroll
        bankroll_data = load_json_file(bankroll_file) or {'bankroll': STARTING_BANKROLL}
        current_bankroll = bankroll_data.get('bankroll', STARTING_BANKROLL)
        new_bankroll = current_bankroll + close_amount + pnl
        bankroll_data['bankroll'] = new_bankroll

        # Record trade
        trades = load_json_file(trades_file) or []
        trade_record = {
            'type': 'PARTIAL_CLOSE',
            'time': datetime.now(ET).isoformat(),
            'position_id': position_id,
            'game': f"{pos.get('away_team', '?')} @ {pos.get('home_team', '?')}",
            'bet_side': pos.get('bet_side', ''),
            'close_pct': close_pct,
            'close_amount': close_amount,
            'pnl': pnl,
            'reason': reason,
            'remaining_kelly': remaining_kelly,
        }
        trades.append(trade_record)

        # Update position
        if close_pct >= 0.99:  # Full close
            pos['status'] = 'closed'
            pos['exit_reason'] = reason
            pos['exit_time'] = datetime.now(ET).isoformat()
            pos['pnl'] = pnl
            pos['bet_kelly'] = 0
        else:  # Partial close
            pos['bet_kelly'] = remaining_kelly
            pos['partial_closes'] = pos.get('partial_closes', [])
            pos['partial_closes'].append({
                'time': datetime.now(ET).isoformat(),
                'close_pct': close_pct,
                'pnl': pnl,
                'reason': reason,
            })

        positions[position_id] = pos

        # Save all files
        with open(positions_file, 'w') as f:
            json.dump(positions, f, indent=2)
        with open(trades_file, 'w') as f:
            json.dump(trades, f, indent=2)
        with open(bankroll_file, 'w') as f:
            json.dump(bankroll_data, f, indent=2)

        return True

    except Exception as e:
        print(f"Error closing position: {e}")
        return False


def match_positions_with_espn(positions: List[Dict], espn_data: Dict[str, Dict]) -> List[Dict]:
    """
    Match open positions with ESPN live probability data.

    Returns positions enriched with ESPN data where available.
    """
    enriched = []

    for pos in positions:
        pos_copy = pos.copy()
        game_str = pos.get('game', '')

        # Try to find matching ESPN data
        espn_match = espn_data.get(game_str)

        if espn_match:
            pos_copy['espn_home_prob'] = espn_match.get('espn_home_prob')
            pos_copy['espn_away_prob'] = espn_match.get('espn_away_prob')
            pos_copy['espn_score'] = f"{espn_match.get('away_score', 0)}-{espn_match.get('home_score', 0)}"
            pos_copy['espn_time'] = f"Q{espn_match.get('period', '?')} {espn_match.get('clock', '')}"
            pos_copy['espn_is_live'] = espn_match.get('is_live', False)
            pos_copy['espn_is_final'] = espn_match.get('is_final', False)
            pos_copy['probability_history'] = espn_match.get('probability_history', [])

            # Calculate divergence between Polymarket and ESPN
            bet_side = pos.get('bet_side', '').lower()
            current_price = pos.get('current_price', 0)

            if bet_side == 'home' and espn_match.get('espn_home_prob'):
                espn_prob = espn_match['espn_home_prob']
                pos_copy['espn_vs_market'] = espn_prob - current_price
            elif bet_side == 'away' and espn_match.get('espn_away_prob'):
                espn_prob = espn_match['espn_away_prob']
                pos_copy['espn_vs_market'] = espn_prob - current_price
            else:
                pos_copy['espn_vs_market'] = None
        else:
            pos_copy['espn_home_prob'] = None
            pos_copy['espn_away_prob'] = None
            pos_copy['espn_score'] = None
            pos_copy['espn_time'] = None
            pos_copy['espn_is_live'] = False
            pos_copy['espn_is_final'] = False
            pos_copy['espn_vs_market'] = None
            pos_copy['probability_history'] = []

        enriched.append(pos_copy)

    return enriched
