"""
SofaScore Historical Data Collector

Fetches historical match data from SofaScore API for backtesting.
Uses curl_cffi for TLS fingerprint bypass.

Collects per match:
  - Basic info (teams, score, date, league)
  - Goal events with minutes
  - Momentum graph (graphPoints)
  - xG statistics

Usage:
    python -m src.Soccer.sofascore_collector                  # current season
    python -m src.Soccer.sofascore_collector --all-seasons    # 24/25 + 25/26
    python -m src.Soccer.sofascore_collector --summary        # show cached data
"""

import json
import logging
import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# SofaScore uniqueTournament IDs
LEAGUE_IDS = {
    "Premier League": 17,
    "La Liga": 8,
    "Bundesliga": 35,
    "Serie A": 23,
    "Ligue 1": 34,
}

# Season IDs (SofaScore internal)
SEASON_IDS = {
    "Premier League": {"2025/2026": 76986, "2024/2025": 61627},
    "La Liga":        {"2025/2026": 77559, "2024/2025": 61643},
    "Bundesliga":     {"2025/2026": 77333, "2024/2025": 63516},
    "Serie A":        {"2025/2026": 76457, "2024/2025": 63515},
    "Ligue 1":        {"2025/2026": 77356, "2024/2025": 61736},
}

ALL_SEASONS = ["2024/2025", "2025/2026"]
CURRENT_SEASON = "2025/2026"

BASE_URL = "https://api.sofascore.com/api/v1"
REQUEST_DELAY = 0.4  # seconds between requests (be polite)

DATA_DIR = Path("Data/soccer_backtest")


class SofaScoreCollector:
    """Collects historical match data from SofaScore for backtesting."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        matches_file: Optional[Path] = None,
    ):
        """
        Args:
            data_dir: directory for the cache (used when matches_file
                      is not given).
            matches_file: explicit path to read/write the cache. Takes
                          precedence over data_dir. Use this to scrape
                          to a versioned snapshot without touching the
                          canonical matches.json.
        """
        if matches_file is not None:
            self.matches_file = Path(matches_file)
            self.data_dir = self.matches_file.parent
        else:
            self.data_dir = Path(data_dir)
            self.matches_file = self.data_dir / "matches.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._session = None
        self._cached: Dict[str, Dict] = {}
        self._load_cache()

    def _get_session(self):
        if self._session is None:
            from curl_cffi.requests import Session
            self._session = Session(impersonate="chrome")
        return self._session

    def _get(self, url: str, retries: int = 2) -> Optional[Dict]:
        s = self._get_session()
        for attempt in range(retries + 1):
            try:
                resp = s.get(url, timeout=15)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt < retries:
                    time.sleep(1)
                    continue
                logger.debug(f"Request failed: {url} - {e}")
                return None

    def _load_cache(self):
        if self.matches_file.exists():
            with open(self.matches_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._cached = {
                    str(m["match_id"]): m for m in data.get("matches", [])
                }
            logger.info(f"Loaded {len(self._cached)} cached matches")

    def _save_cache(self):
        data = {
            "collected_at": datetime.utcnow().isoformat(),
            "total_matches": len(self._cached),
            "matches": list(self._cached.values()),
        }
        with open(self.matches_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_season_events(
        self, league: str, season: str
    ) -> List[Dict]:
        """Get all finished events for a league/season from SofaScore."""
        tid = LEAGUE_IDS.get(league)
        sid = SEASON_IDS.get(league, {}).get(season)
        if not tid or not sid:
            logger.error(f"Unknown league/season: {league} {season}")
            return []

        events = []
        page = 0
        while True:
            url = f"{BASE_URL}/unique-tournament/{tid}/season/{sid}/events/last/{page}"
            data = self._get(url)
            if not data:
                break
            batch = data.get("events", [])
            if not batch:
                break

            # Only finished matches
            for e in batch:
                status = e.get("status", {})
                if status.get("type") == "finished":
                    events.append(e)

            if not data.get("hasNextPage", False):
                break
            page += 1
            time.sleep(REQUEST_DELAY)

        return events

    def fetch_match_data(self, event_id: int) -> Optional[Dict]:
        """Fetch full match details: event info, momentum graph, statistics."""
        event_data = self._get(f"{BASE_URL}/event/{event_id}")
        if not event_data:
            return None

        graph_data = self._get(f"{BASE_URL}/event/{event_id}/graph")
        stats_data = self._get(f"{BASE_URL}/event/{event_id}/statistics")
        incidents = self._get(f"{BASE_URL}/event/{event_id}/incidents")

        time.sleep(REQUEST_DELAY)

        return {
            "event": event_data.get("event", event_data),
            "graph": graph_data,
            "statistics": stats_data,
            "incidents": incidents,
        }

    def parse_match(self, raw: Dict, league: str) -> Optional[Dict]:
        """Parse raw SofaScore data into our standard match format."""
        event = raw["event"]
        graph = raw.get("graph")
        stats = raw.get("statistics")
        incidents = raw.get("incidents")

        try:
            event_id = event.get("id")
            home_team = event.get("homeTeam", {}).get("name", "Unknown")
            away_team = event.get("awayTeam", {}).get("name", "Unknown")
            home_score = event.get("homeScore", {}).get("current", 0)
            away_score = event.get("awayScore", {}).get("current", 0)

            # Date
            start_ts = event.get("startTimestamp", 0)
            date = datetime.utcfromtimestamp(start_ts).strftime("%Y-%m-%d") if start_ts else ""

            # Goals + cards + substitutions from incidents
            goals = self._extract_goals(incidents)
            cards = self._extract_cards(incidents)
            substitutions = self._extract_substitutions(incidents)

            # Momentum from graph
            momentum_data = None
            if graph and "graphPoints" in graph:
                momentum_data = graph["graphPoints"]

            # xG from statistics
            xg_home, xg_away = self._extract_xg(stats)

            # Qualifying analysis at minute 70
            score_at_70, losing_team_at_70 = self._score_at_minute(goals, 70)
            qualifying = False
            if losing_team_at_70 and score_at_70:
                h, a = score_at_70
                if abs(h - a) == 1:
                    qualifying = True

            equalized_after_70 = False
            equalization_minute = None
            if qualifying and losing_team_at_70:
                equalized_after_70, equalization_minute = self._check_equalization(
                    goals, score_at_70, losing_team_at_70, 70
                )

            momentum_at_70 = None
            if momentum_data and losing_team_at_70:
                momentum_at_70 = self._get_momentum_at_minute(
                    momentum_data, 70, losing_team_at_70
                )

            losing_team_xg_share = None
            if qualifying and xg_home is not None and xg_away is not None:
                total_xg = xg_home + xg_away
                if total_xg > 0:
                    if losing_team_at_70 == "home":
                        losing_team_xg_share = round(xg_home / total_xg, 3)
                    else:
                        losing_team_xg_share = round(xg_away / total_xg, 3)

            return {
                "match_id": str(event_id),
                "date": date,
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "final_score": f"{home_score}-{away_score}",
                "home_score": int(home_score),
                "away_score": int(away_score),
                "was_draw": home_score == away_score,
                "xg_home": xg_home,
                "xg_away": xg_away,
                "goals": goals,
                "cards": cards,
                "substitutions": substitutions,
                "momentum_data": momentum_data is not None,
                "qualifying": qualifying,
                "losing_team_at_70": losing_team_at_70,
                "score_at_70": list(score_at_70) if score_at_70 else None,
                "losing_team_xg_share": losing_team_xg_share,
                "momentum_at_70": momentum_at_70,
                "equalized_after_70": equalized_after_70,
                "equalization_minute": equalization_minute,
            }
        except Exception as e:
            logger.error(f"Error parsing match: {e}")
            return None

    def _extract_goals(self, incidents: Optional[Dict]) -> List[Dict]:
        """Extract goals from SofaScore incidents endpoint.

        SofaScore /incidents returns events in reverse-chronological
        order. We must sort by minute BEFORE accumulating the running
        score, otherwise home_after/away_after are wrong.
        """
        if not incidents:
            return []

        raw = [
            inc for inc in incidents.get("incidents", [])
            if inc.get("incidentType") == "goal"
        ]
        raw.sort(key=lambda x: (x.get("time", 0), x.get("addedTime") or 0))

        goals = []
        home_score = away_score = 0
        for inc in raw:
            is_home = inc.get("isHome", False)
            player_name = (inc.get("player") or {}).get("name", "Unknown")

            if is_home:
                home_score += 1
            else:
                away_score += 1

            goals.append({
                "minute": inc.get("time", 0),
                "team": "home" if is_home else "away",
                "scorer": player_name,
                "score_after": f"{home_score}-{away_score}",
                "home_after": home_score,
                "away_after": away_score,
                "is_own_goal": inc.get("incidentClass", "") == "ownGoal",
            })

        return goals

    def _extract_cards(self, incidents: Optional[Dict]) -> List[Dict]:
        """Extract card events. Normalizes SofaScore incidentClass to
        {yellow, red, second_yellow}. Skips rescinded cards (VAR overturned)."""
        cards = []
        if not incidents:
            return cards
        for inc in incidents.get("incidents", []):
            if inc.get("incidentType") != "card":
                continue
            if inc.get("rescinded"):
                continue
            cls = (inc.get("incidentClass") or "").lower()
            if cls == "yellow":
                card_type = "yellow"
            elif cls == "red":
                card_type = "red"
            elif cls in ("yellowred", "secondyellow"):
                card_type = "second_yellow"
            else:
                continue
            minute = inc.get("time", 0)
            added = inc.get("addedTime") or 0
            cards.append({
                "minute": minute,
                "added_time": added,
                "team": "home" if inc.get("isHome") else "away",
                "card_type": card_type,
                "player": (inc.get("player") or {}).get("name", "Unknown"),
            })
        cards.sort(key=lambda c: (c["minute"], c["added_time"]))
        return cards

    def _extract_substitutions(self, incidents: Optional[Dict]) -> List[Dict]:
        """Extract substitution events from SofaScore incidents."""
        subs = []
        if not incidents:
            return subs
        for inc in incidents.get("incidents", []):
            if inc.get("incidentType") != "substitution":
                continue
            minute = inc.get("time", 0)
            added = inc.get("addedTime") or 0
            subs.append({
                "minute": minute,
                "added_time": added,
                "team": "home" if inc.get("isHome") else "away",
                "player_in": (inc.get("playerIn") or {}).get("name", "Unknown"),
                "player_out": (inc.get("playerOut") or {}).get("name", "Unknown"),
            })
        subs.sort(key=lambda s: (s["minute"], s["added_time"]))
        return subs

    def _extract_xg(self, stats: Optional[Dict]) -> Tuple[Optional[float], Optional[float]]:
        """Extract xG from SofaScore statistics."""
        if not stats:
            return None, None
        try:
            for period in stats.get("statistics", []):
                for group in period.get("groups", []):
                    for item in group.get("statisticsItems", []):
                        name = item.get("name", "").lower()
                        if "expected goals" in name or name == "expected_goals" or "xg" in name:
                            return float(item.get("home", 0)), float(item.get("away", 0))
        except Exception:
            pass
        return None, None

    def _score_at_minute(
        self, goals: List[Dict], minute: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        h, a = 0, 0
        for g in goals:
            if g["minute"] <= minute:
                h, a = g["home_after"], g["away_after"]
            else:
                break
        if h == a:
            return (h, a), None
        elif h < a:
            return (h, a), "home"
        else:
            return (h, a), "away"

    def _check_equalization(
        self, goals, score_at_entry, losing_team, entry_minute
    ) -> Tuple[bool, Optional[int]]:
        for g in goals:
            if g["minute"] <= entry_minute:
                continue
            if g["team"] == losing_team and g["home_after"] == g["away_after"]:
                return True, g["minute"]
        return False, None

    def _get_momentum_at_minute(
        self, momentum_data: List, minute: int, losing_team: str
    ) -> Optional[float]:
        """Get momentum from SofaScore graph data.

        SofaScore graphPoints: [{minute: N, value: V}, ...]
        Positive value = home momentum, negative = away momentum.
        Values range roughly -10 to +10.
        """
        window_start = max(1, minute - 10)
        window_end = minute

        values = []
        for point in momentum_data:
            if isinstance(point, dict):
                m = point.get("minute", -1)
                v = point.get("value", 0)
            else:
                continue
            if window_start <= m <= window_end:
                values.append(v)

        if not values:
            return None

        avg = sum(values) / len(values)

        # Flip so positive = losing team
        if losing_team == "away":
            avg = -avg

        # Sigmoid normalization (same as FotMobProvider)
        normalized = 1 / (1 + math.exp(-avg / 20))
        return round(normalized, 3)

    def collect(
        self, leagues: List[str] = None, seasons: List[str] = None,
        force: bool = False,
    ) -> List[Dict]:
        """Main collection method."""
        target_leagues = leagues or list(LEAGUE_IDS.keys())
        target_seasons = seasons or [CURRENT_SEASON]

        print("=" * 60)
        print("  SofaScore Backtest Data Collector")
        print(f"  Seasons: {', '.join(target_seasons)}")
        print(f"  Leagues: {', '.join(target_leagues)}")
        print(f"  Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print("=" * 60)

        # Step 1: Get all event IDs
        print("\nStep 1: Fetching event lists...")
        events_to_fetch = []

        for league in target_leagues:
            for season in target_seasons:
                events = self.get_season_events(league, season)
                new_events = []
                for e in events:
                    eid = str(e.get("id", ""))
                    if not force and eid in self._cached:
                        continue
                    new_events.append((eid, league))

                print(f"  {league} ({season}): {len(events)} finished, "
                      f"{len(new_events)} new")
                events_to_fetch.extend(new_events)

        already = len(self._cached)
        print(f"\nAlready cached: {already}")
        print(f"Events to fetch: {len(events_to_fetch)}")

        if not events_to_fetch:
            print("All matches already cached!")
            return list(self._cached.values())

        est_time = len(events_to_fetch) * REQUEST_DELAY * 4  # 4 requests per match
        print(f"Estimated time: ~{est_time/60:.0f} minutes")

        # Step 2: Fetch and parse
        print(f"\nStep 2: Fetching {len(events_to_fetch)} matches...")
        fetched = 0
        errors = 0

        for i, (event_id, league) in enumerate(events_to_fetch):
            pct = (i + 1) / len(events_to_fetch) * 100
            print(f"\r  [{i+1}/{len(events_to_fetch)}] ({pct:.0f}%) "
                  f"Fetching {event_id}...", end="", flush=True)

            raw = self.fetch_match_data(int(event_id))
            if not raw:
                errors += 1
                continue

            parsed = self.parse_match(raw, league)
            if not parsed:
                errors += 1
                continue

            self._cached[str(parsed["match_id"])] = parsed
            fetched += 1

            if fetched % 50 == 0:
                self._save_cache()
                print(f" (saved {len(self._cached)} total)")

        print(f"\n\nFetched: {fetched} | Errors: {errors}")

        self._save_cache()

        # Summary
        all_matches = list(self._cached.values())
        qualifying = [m for m in all_matches if m.get("qualifying")]
        equalized = [m for m in qualifying if m.get("equalized_after_70")]
        with_momentum = [m for m in qualifying if m.get("momentum_at_70") is not None]

        print(f"\n{'='*60}")
        print("  Collection Summary")
        print(f"{'='*60}")
        print(f"  Total matches: {len(all_matches)}")
        print(f"  Qualifying (losing by 1 at min 70): {len(qualifying)}")
        if qualifying:
            print(f"  With momentum: {len(with_momentum)}")
            print(f"  Equalized: {len(equalized)} ({len(equalized)/len(qualifying)*100:.1f}%)")

        by_league = {}
        for m in all_matches:
            by_league.setdefault(m.get("league", "?"), []).append(m)
        print("\n  Per league:")
        for league, matches in sorted(by_league.items()):
            q = sum(1 for m in matches if m.get("qualifying"))
            eq = sum(1 for m in matches if m.get("qualifying") and m.get("equalized_after_70"))
            print(f"    {league}: {len(matches)} matches, {q} qualifying, {eq} equalized")

        print(f"\n  Data saved to: {self.matches_file}")
        print("=" * 60)

        return all_matches


def main():
    parser = argparse.ArgumentParser(description="SofaScore Backtest Collector")
    parser.add_argument("--league", type=str, help="Single league to collect")
    parser.add_argument("--season", type=str, action="append", help="Season (repeatable)")
    parser.add_argument("--all-seasons", action="store_true", help="Collect 24/25 + 25/26")
    parser.add_argument("--force", action="store_true", help="Re-fetch all (ignore cache)")
    parser.add_argument("--summary", action="store_true", help="Show cached data summary")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to read/write the matches cache. Default: "
             "Data/soccer_backtest/matches.json (canonical). Use a "
             "date-stamped path to scrape into a versioned snapshot "
             "without overwriting the canonical file.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    out = Path(args.output) if args.output else None
    collector = SofaScoreCollector(matches_file=out)

    if args.summary:
        matches = list(collector._cached.values())
        if not matches:
            print("No cached data.")
            return
        qualifying = [m for m in matches if m.get("qualifying")]
        equalized = [m for m in qualifying if m.get("equalized_after_70")]
        print(f"Cached: {len(matches)} matches")
        print(f"Qualifying: {len(qualifying)}")
        if qualifying:
            print(f"Equalized: {len(equalized)} ({len(equalized)/len(qualifying)*100:.1f}%)")
        return

    leagues = [args.league] if args.league else None
    if args.all_seasons:
        seasons = ALL_SEASONS
    elif args.season:
        seasons = args.season
    else:
        seasons = None

    collector.collect(leagues=leagues, seasons=seasons, force=args.force)


if __name__ == "__main__":
    main()
