"""
SofaScore Data Provider (drop-in replacement for FotMob)

Uses SofaScore API via curl_cffi (TLS fingerprint impersonation) to bypass
Cloudflare protection. Provides live match listings, match details with
momentum graph data, and xG statistics for top 5 European leagues.

Class is still named FotMobProvider for backward compatibility with
momentum_scanner.py and the rest of the codebase.

Usage:
    from src.DataProviders.FotMobProvider import FotMobProvider

    provider = FotMobProvider()
    live = provider.get_live_matches()
    details = provider.get_match_details(match_id)
    momentum = provider.get_momentum_value(details)
"""

import logging
import math
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# SofaScore uniqueTournament IDs for top 5 European leagues
LEAGUE_IDS = {
    "Premier League": 17,
    "La Liga": 8,
    "Bundesliga": 35,
    "Serie A": 23,
    "Ligue 1": 34,
}

ALL_LEAGUE_IDS = set(LEAGUE_IDS.values())

SOFASCORE_BASE = "https://api.sofascore.com/api/v1"


class FotMobProvider:
    """SofaScore-backed provider for live soccer data (curl_cffi for anti-bot bypass).

    Maintains the same interface as the original FotMob-based provider so that
    momentum_scanner.py, paper_trader.py, and scheduler.py work unchanged.
    """

    def __init__(self, leagues: List[str] = None):
        """
        Args:
            leagues: List of league names to track.
                     Defaults to all top 5 European leagues.
        """
        if leagues:
            self.league_ids = {
                name: LEAGUE_IDS[name]
                for name in leagues
                if name in LEAGUE_IDS
            }
        else:
            self.league_ids = dict(LEAGUE_IDS)

        self._league_id_set = set(self.league_ids.values())
        self._league_id_to_name = {v: k for k, v in self.league_ids.items()}
        self._session = None

    # ---- HTTP layer ----

    def _get_session(self):
        """Lazy-init a curl_cffi Session with Chrome TLS fingerprint."""
        if self._session is None:
            from curl_cffi.requests import Session
            self._session = Session(impersonate="chrome")
        return self._session

    def _get(self, url: str, params: dict = None, retries: int = 2) -> Optional[Dict]:
        """Make a GET request via curl_cffi with Chrome impersonation."""
        session = self._get_session()
        for attempt in range(retries + 1):
            try:
                resp = session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt < retries:
                    time.sleep(0.5)
                    continue
                logger.error(f"SofaScore request failed after {retries + 1} attempts: {url} - {e}")
                return None

    # ---- Public API (same interface as old FotMob provider) ----

    def get_live_matches(self) -> List[Dict]:
        """Get all currently live matches from tracked leagues.

        Returns list of dicts:
        [
            {
                "match_id": int,
                "home_team": str,
                "away_team": str,
                "home_score": int,
                "away_score": int,
                "minute": int,
                "status": str,
                "league": str,
                "league_id": int,
            },
            ...
        ]
        """
        data = self._get(f"{SOFASCORE_BASE}/sport/football/events/live")
        if not data:
            return []

        matches = []
        events = data.get("events", [])

        for event in events:
            tournament = event.get("tournament", {})
            unique_tournament = tournament.get("uniqueTournament", {})
            tournament_id = unique_tournament.get("id")

            if tournament_id not in self._league_id_set:
                continue

            # Only include in-progress matches
            status = event.get("status", {})
            status_type = status.get("type", "")
            if status_type != "inprogress":
                continue

            league_name = self._league_id_to_name.get(
                tournament_id, unique_tournament.get("name", "Unknown")
            )

            parsed = self._parse_event(event, league_name, tournament_id)
            if parsed:
                matches.append(parsed)

        return matches

    def get_match_details(self, match_id: int) -> Optional[Dict]:
        """Get detailed match data including momentum graph and xG.

        Returns dict with:
        {
            "match_id": int,
            "home_team": str,
            "away_team": str,
            "home_score": int,
            "away_score": int,
            "minute": int,
            "status": str,
            "league": str,
            "momentum": list of {minute, value} dicts (or None),
            "xg": {"home": float, "away": float},
            "stats": full stats dict,
            "is_extra_time": bool,
            "raw": full raw response,
        }
        """
        # Fetch event details, momentum graph, and statistics
        event_data = self._get(f"{SOFASCORE_BASE}/event/{match_id}")
        graph_data = self._get(f"{SOFASCORE_BASE}/event/{match_id}/graph")
        stats_data = self._get(f"{SOFASCORE_BASE}/event/{match_id}/statistics")

        if not event_data:
            return None

        try:
            # SofaScore wraps in {"event": {...}} for detail endpoint
            event = event_data.get("event", event_data)

            home_team_data = event.get("homeTeam", {})
            away_team_data = event.get("awayTeam", {})

            home_team = home_team_data.get("name", "Unknown")
            away_team = away_team_data.get("name", "Unknown")

            home_score = event.get("homeScore", {}).get("current", 0)
            away_score = event.get("awayScore", {}).get("current", 0)

            minute = self._calc_minute(event)
            status = event.get("status", {}).get("description", "Unknown")

            # League info
            tournament = event.get("tournament", {})
            unique_tournament = tournament.get("uniqueTournament", {})
            league_name = self._league_id_to_name.get(
                unique_tournament.get("id"),
                unique_tournament.get("name", "Unknown"),
            )

            # Momentum from graph endpoint
            momentum = None
            if graph_data and "graphPoints" in graph_data:
                momentum = graph_data["graphPoints"]

            # xG from statistics
            xg = self._extract_xg(stats_data)

            # Extra time detection
            is_extra_time = self._detect_extra_time(event, minute)

            return {
                "match_id": match_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(home_score) if home_score is not None else 0,
                "away_score": int(away_score) if away_score is not None else 0,
                "minute": minute,
                "status": str(status),
                "league": league_name,
                "momentum": momentum,
                "xg": xg,
                "stats": stats_data if stats_data else {},
                "is_extra_time": is_extra_time,
                "raw": event_data,
            }
        except Exception as e:
            logger.error(f"Error parsing match details for {match_id}: {e}")
            return None

    def get_momentum_value(self, match_details: Dict) -> Optional[Dict]:
        """Extract or calculate momentum for the losing team.

        Strategy:
        1. Try SofaScore momentum graph (attackMomentum)
        2. Fallback: derive from xG differential

        Returns:
        {
            "losing_team": str ("home" or "away"),
            "momentum_value": float (0-1 scale, higher = more attacking momentum),
            "source": str ("sofascore_momentum" or "xg_derived"),
            "details": dict (raw underlying data),
        }
        or None if momentum cannot be determined.
        """
        if not match_details:
            return None

        home_score = match_details["home_score"]
        away_score = match_details["away_score"]

        # Must be losing by exactly 1
        if home_score == away_score:
            return None
        if abs(home_score - away_score) != 1:
            return None

        losing_team = "home" if home_score < away_score else "away"

        # Strategy 1: SofaScore momentum graph
        momentum_data = match_details.get("momentum")
        if momentum_data:
            value = self._parse_native_momentum(momentum_data, losing_team)
            if value is not None:
                return {
                    "losing_team": losing_team,
                    "momentum_value": value,
                    "source": "sofascore_momentum",
                    "details": {"raw_momentum": momentum_data[-10:]},
                }

        # Strategy 2: xG-derived momentum
        xg = match_details.get("xg", {})
        if xg.get("home") is not None and xg.get("away") is not None:
            value = self._derive_momentum_from_xg(
                xg, losing_team, match_details["minute"]
            )
            if value is not None:
                return {
                    "losing_team": losing_team,
                    "momentum_value": value,
                    "source": "xg_derived",
                    "details": {"xg_home": xg["home"], "xg_away": xg["away"]},
                }

        logger.warning(
            f"Could not determine momentum for match {match_details.get('match_id')}"
        )
        return None

    # ---- Internal helpers ----

    def _parse_event(
        self, event: Dict, league_name: str, league_id: int
    ) -> Optional[Dict]:
        """Parse a SofaScore live event into our standard format."""
        try:
            event_id = event.get("id")
            if not event_id:
                return None

            home_team_data = event.get("homeTeam", {})
            away_team_data = event.get("awayTeam", {})

            home_team = home_team_data.get("name", "Unknown")
            away_team = away_team_data.get("name", "Unknown")

            home_score = event.get("homeScore", {}).get("current", 0)
            away_score = event.get("awayScore", {}).get("current", 0)

            minute = self._calc_minute(event)

            status = event.get("status", {})
            status_desc = status.get("description", "inprogress")

            return {
                "match_id": int(event_id),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(home_score) if home_score is not None else 0,
                "away_score": int(away_score) if away_score is not None else 0,
                "minute": minute,
                "status": status_desc,
                "league": league_name,
                "league_id": int(league_id),
            }
        except Exception as e:
            logger.debug(f"Error parsing event: {e}")
            return None

    def _calc_minute(self, event: Dict) -> int:
        """Calculate current match minute from SofaScore event data.

        Uses time.currentPeriodStartTimestamp + time.initial for accurate
        live minute calculation.
        """
        status = event.get("status", {})
        code = status.get("code")

        # Handle halftime / breaks where clock is stopped
        if code == 31:  # Halftime
            return 45
        if code == 34:  # ET halftime
            return 105

        time_info = event.get("time", {})

        # Primary method: timestamp-based calculation
        period_start = time_info.get("currentPeriodStartTimestamp")
        initial = time_info.get("initial")  # seconds offset (0 for 1H, 2700 for 2H)

        if period_start and initial is not None:
            now = int(time.time())
            elapsed_since_period = now - period_start
            total_seconds = initial + elapsed_since_period
            minute = total_seconds // 60
            return max(0, min(int(minute), 130))

        # Fallback: status code estimation
        if code == 6:   # 1st half
            return 25
        elif code == 7:  # 2nd half
            return 70
        elif code in (41, 42):  # extra time
            return 105

        return 0

    def _parse_native_momentum(
        self, momentum_data: Any, losing_team: str
    ) -> Optional[float]:
        """Parse SofaScore momentum graph into 0-1 score for the losing team.

        SofaScore graph format: [{minute: N, value: V}, ...]
        Convention: positive value = home momentum, negative = away momentum.
        Values range roughly -10 to +10.
        """
        if not momentum_data:
            return None

        try:
            if isinstance(momentum_data, list):
                data_points = momentum_data
            elif isinstance(momentum_data, dict):
                data_points = momentum_data.get(
                    "graphPoints", momentum_data.get("data", [])
                )
            else:
                return None

            if not data_points:
                return None

            # Last 10 data points of momentum (recent form matters most)
            recent_window = min(10, len(data_points))
            recent = data_points[-recent_window:]

            total = 0
            count = 0
            for point in recent:
                if isinstance(point, (int, float)):
                    val = float(point)
                elif isinstance(point, dict):
                    val = float(point.get("value", 0))
                else:
                    continue
                total += val
                count += 1

            if count == 0:
                return None

            avg = total / count

            # Flip sign so positive = losing team's momentum
            if losing_team == "away":
                avg = -avg

            # Normalize to 0-1
            # SofaScore range ~-100 to +100; divisor=20 gives good sigmoid spread
            normalized = 1 / (1 + math.exp(-avg / 20))
            return round(normalized, 3)

        except Exception as e:
            logger.debug(f"Error parsing momentum: {e}")
            return None

    def _extract_xg(self, stats_data: Optional[Dict]) -> Dict:
        """Extract xG from SofaScore statistics response.

        SofaScore format:
        {"statistics": [{"period": "ALL", "groups": [{"statisticsItems": [...]}]}]}
        Each item: {"name": "Expected goals", "home": "1.27", "away": "0.48"}
        """
        xg = {"home": None, "away": None}
        if not stats_data:
            return xg

        try:
            statistics = stats_data.get("statistics", [])
            for period in statistics:
                groups = period.get("groups", [])
                for group in groups:
                    items = group.get("statisticsItems", [])
                    for item in items:
                        name = item.get("name", "").lower()
                        if "expected goals" in name or name == "expected_goals" or "xg" in name:
                            xg["home"] = float(item.get("home", 0))
                            xg["away"] = float(item.get("away", 0))
                            return xg
        except Exception as e:
            logger.debug(f"Error extracting xG: {e}")

        return xg

    def _detect_extra_time(self, event: Dict, minute: int) -> bool:
        """Detect if match is in extra time."""
        if minute > 100:
            return True
        status = event.get("status", {})
        desc = str(status.get("description", "")).lower()
        # SofaScore codes 41/42 = extra time halves
        if "extra" in desc or status.get("code") in (41, 42):
            return True
        return False

    def _derive_momentum_from_xg(
        self, xg: Dict, losing_team: str, minute: int
    ) -> Optional[float]:
        """Derive momentum from xG ratio.

        If the losing team has a higher xG relative to total,
        they're creating more chances — indicating attacking pressure.
        """
        home_xg = xg.get("home")
        away_xg = xg.get("away")

        if home_xg is None or away_xg is None:
            return None

        total_xg = home_xg + away_xg
        if total_xg <= 0:
            return 0.5  # No xG data yet, neutral

        losing_xg = home_xg if losing_team == "home" else away_xg
        xg_share = losing_xg / total_xg
        return round(xg_share, 3)


if __name__ == "__main__":
    # Test the provider
    logging.basicConfig(level=logging.INFO)
    provider = FotMobProvider()

    print("Fetching live matches from Top 5 European leagues (via SofaScore)...")
    matches = provider.get_live_matches()

    if not matches:
        print("No live matches right now.")
    else:
        print(f"\nFound {len(matches)} live matches:\n")
        for m in matches:
            print(
                f"  [{m['league']}] {m['home_team']} {m['home_score']}-{m['away_score']} "
                f"{m['away_team']} (min {m['minute']})"
            )

            # Fetch details for each live match
            details = provider.get_match_details(m["match_id"])
            if details:
                mom = provider.get_momentum_value(details)
                xg = details.get("xg", {})
                print(f"    xG: Home {xg.get('home', '?')} - Away {xg.get('away', '?')}")
                if mom:
                    print(
                        f"    Momentum: {mom['losing_team']} team = {mom['momentum_value']:.3f} "
                        f"({mom['source']})"
                    )
                else:
                    print(f"    Momentum: N/A (score may be tied or >1 goal gap)")
                print(f"    Has momentum data: {details.get('momentum') is not None}")
            print()
