"""
Soccer Backtest Data Collector

Fetches historical match data from FotMob for backtesting the momentum-draw theory.
Uses FotMob match page HTML scraping to get goal events, momentum, and xG data.

Data flow:
1. get_league_fixtures() -> all match IDs and page URLs per league
2. Fetch each unique match page HTML -> extract __NEXT_DATA__ JSON
3. Parse goals (with minutes), momentum per minute, xG stats
4. Cache to Data/soccer_backtest/matches.json

Usage:
    python -m src.Soccer.backtest_collector
    python -m src.Soccer.backtest_collector --league "Premier League"
    python -m src.Soccer.backtest_collector --force  # re-fetch all
"""

import asyncio
import json
import re
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

# FotMob league IDs
LEAGUE_IDS = {
    "Premier League": 47,
    "La Liga": 87,
    "Bundesliga": 54,
    "Serie A": 55,
    "Ligue 1": 53,
}

# Season strings for FotMob API
CURRENT_SEASON = "2025/2026"
ALL_SEASONS = ["2024/2025", "2025/2026"]

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between page fetches

# Data directory
DATA_DIR = Path("Data/soccer_backtest")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>'
)


class BacktestCollector:
    """Collects historical match data from FotMob for backtesting."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.matches_file = self.data_dir / "matches.json"
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._cached_matches: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self):
        """Load previously collected matches from cache."""
        if self.matches_file.exists():
            with open(self.matches_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._cached_matches = {
                    str(m["match_id"]): m for m in data.get("matches", [])
                }
            logger.info(f"Loaded {len(self._cached_matches)} cached matches")

    def _save_cache(self):
        """Save collected matches to cache."""
        data = {
            "collected_at": datetime.utcnow().isoformat(),
            "season": CURRENT_SEASON,
            "total_matches": len(self._cached_matches),
            "matches": list(self._cached_matches.values()),
        }
        with open(self.matches_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self._cached_matches)} matches to cache")

    def get_all_fixtures(
        self, leagues: List[str] = None, seasons: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """Get all finished fixtures from FotMob for specified leagues and seasons.

        Returns dict of league_name -> list of fixture dicts.
        """
        target_leagues = leagues or list(LEAGUE_IDS.keys())
        target_seasons = seasons or [CURRENT_SEASON]
        all_fixtures = {}

        async def _fetch():
            from fotmob import FotMob
            async with FotMob() as fm:
                for league_name in target_leagues:
                    league_id = LEAGUE_IDS.get(league_name)
                    if not league_id:
                        continue

                    league_fixtures = []
                    for season in target_seasons:
                        try:
                            fixtures = await fm.get_league_fixtures(
                                league_id, season
                            )
                            finished = [
                                f
                                for f in fixtures
                                if f.get("status", {}).get("finished", False)
                            ]
                            league_fixtures.extend(finished)
                            print(
                                f"  {league_name} ({season}): {len(finished)} finished "
                                f"/ {len(fixtures)} total fixtures"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to get fixtures for {league_name} {season}: {e}"
                            )

                    all_fixtures[league_name] = league_fixtures

        asyncio.run(_fetch())
        return all_fixtures

    def _get_unique_page_slugs(
        self, fixtures: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Group fixtures by unique H2H page slug.

        FotMob page URLs use H2H slugs: /matches/team-a-vs-team-b/SLUG#matchId
        The page always returns the latest match for that H2H pair.

        Returns: slug -> list of fixtures with that slug
        """
        slug_groups: Dict[str, List[Dict]] = {}
        for f in fixtures:
            page_url = f.get("pageUrl", "")
            # Extract slug: /matches/seo-part/SLUG#matchId
            parts = page_url.split("/")
            if len(parts) >= 4:
                slug = parts[3].split("#")[0]  # Remove #matchId fragment
                seo = parts[2] if len(parts) >= 3 else ""
                key = f"{seo}/{slug}"
            else:
                key = page_url

            if key not in slug_groups:
                slug_groups[key] = []
            slug_groups[key].append(f)

        return slug_groups

    def fetch_match_page(self, page_url: str) -> Optional[Dict]:
        """Fetch a FotMob match page and extract __NEXT_DATA__ JSON.

        Args:
            page_url: Relative URL like /matches/team-vs-team/slug

        Returns:
            Parsed pageProps dict or None on failure.
        """
        # Remove fragment
        clean_url = page_url.split("#")[0]
        full_url = f"https://www.fotmob.com{clean_url}"

        try:
            r = self.session.get(full_url, timeout=20)
            if r.status_code != 200:
                logger.warning(f"HTTP {r.status_code} for {full_url}")
                return None

            match = NEXT_DATA_PATTERN.search(r.text)
            if not match:
                logger.warning(f"No __NEXT_DATA__ found in {full_url}")
                return None

            data = json.loads(match.group(1))
            return data.get("props", {}).get("pageProps", {})

        except Exception as e:
            logger.error(f"Error fetching {full_url}: {e}")
            return None

    def parse_match_data(
        self, page_props: Dict, league_name: str
    ) -> Optional[Dict]:
        """Parse match data from FotMob page props into our standard format.

        Extracts:
        - Basic info (teams, score, date)
        - Goal events with minutes
        - Momentum data per minute
        - xG stats
        """
        general = page_props.get("general", {})
        header = page_props.get("header", {})
        content = page_props.get("content", {})

        match_id = general.get("matchId")
        if not match_id:
            return None

        # Teams
        home_team_data = general.get("homeTeam", {})
        away_team_data = general.get("awayTeam", {})
        home_team = home_team_data.get("name", "Unknown")
        away_team = away_team_data.get("name", "Unknown")

        # Score from header
        teams = header.get("teams", [])
        home_score = teams[0].get("score", 0) if len(teams) > 0 else 0
        away_score = teams[1].get("score", 0) if len(teams) > 1 else 0

        # Date
        match_date = general.get("matchTimeUTCDate", "")[:10]  # YYYY-MM-DD

        # Goal events
        goals = self._extract_goals(content)

        # Momentum per minute
        momentum = self._extract_momentum(content)

        # xG
        xg_home, xg_away = self._extract_xg(content)

        # Determine if any team was losing by 1 at minute 70-75
        score_at_70, losing_team_at_70 = self._score_at_minute(goals, 70)
        qualifying = False
        if losing_team_at_70 and score_at_70:
            h, a = score_at_70
            if abs(h - a) == 1:
                qualifying = True

        # Check if equalizer happened after minute 70
        equalized_after_70 = False
        equalization_minute = None
        if qualifying and losing_team_at_70:
            equalized_after_70, equalization_minute = self._check_equalization(
                goals, score_at_70, losing_team_at_70, 70
            )

        # Momentum at minute 70 for the losing team
        momentum_at_70 = None
        if momentum and losing_team_at_70:
            momentum_at_70 = self._get_momentum_at_minute(
                momentum, 70, losing_team_at_70
            )

        # Losing team's xG share (momentum proxy)
        losing_team_xg_share = None
        if qualifying and xg_home is not None and xg_away is not None:
            total_xg = xg_home + xg_away
            if total_xg > 0:
                if losing_team_at_70 == "home":
                    losing_team_xg_share = round(xg_home / total_xg, 3)
                else:
                    losing_team_xg_share = round(xg_away / total_xg, 3)

        # Final score as string
        final_score = f"{home_score}-{away_score}"
        was_draw = home_score == away_score

        return {
            "match_id": match_id,
            "date": match_date,
            "league": league_name,
            "home_team": home_team,
            "away_team": away_team,
            "final_score": final_score,
            "home_score": home_score,
            "away_score": away_score,
            "was_draw": was_draw,
            "xg_home": xg_home,
            "xg_away": xg_away,
            "goals": goals,
            "momentum_data": momentum is not None,
            "qualifying": qualifying,
            "losing_team_at_70": losing_team_at_70,
            "score_at_70": list(score_at_70) if score_at_70 else None,
            "losing_team_xg_share": losing_team_xg_share,
            "momentum_at_70": momentum_at_70,
            "equalized_after_70": equalized_after_70,
            "equalization_minute": equalization_minute,
        }

    def _extract_goals(self, content: Dict) -> List[Dict]:
        """Extract goal events with minutes from match content."""
        goals = []
        match_facts = content.get("matchFacts", {})
        events_data = match_facts.get("events", {})
        events = events_data.get("events", [])

        for event in events:
            if event.get("type") != "Goal":
                continue

            minute = event.get("time", 0)
            overload = event.get("overloadTime")
            if overload:
                minute = minute  # Keep base minute, overload is added time

            is_home = event.get("isHome", False)
            # homeScore/awayScore in the event is the score BEFORE this goal
            home_before = event.get("homeScore", 0)
            away_before = event.get("awayScore", 0)

            if is_home:
                home_after = home_before + 1
                away_after = away_before
            else:
                home_after = home_before
                away_after = away_before + 1

            goals.append(
                {
                    "minute": minute,
                    "team": "home" if is_home else "away",
                    "scorer": event.get("player", {}).get("name", "Unknown"),
                    "score_after": f"{home_after}-{away_after}",
                    "home_after": home_after,
                    "away_after": away_after,
                    "is_own_goal": event.get("ownGoal") is not None,
                }
            )

        # Sort by minute
        goals.sort(key=lambda g: g["minute"])
        return goals

    def _extract_momentum(self, content: Dict) -> Optional[List[Dict]]:
        """Extract per-minute momentum data.

        Returns list of {minute: int, value: int} or None.
        Positive values = home momentum, negative = away momentum.
        """
        momentum = content.get("momentum", {})
        if not momentum:
            return None

        main = momentum.get("main", {})
        if not isinstance(main, dict):
            return None

        data_points = main.get("data", [])
        if not data_points:
            return None

        return data_points

    def _extract_xg(self, content: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Extract xG values for home and away teams."""
        stats = content.get("stats", {})
        if not stats:
            return None, None

        # Navigate: stats -> Periods -> All -> stats -> find expected_goals
        periods = stats.get("Periods", stats)
        if isinstance(periods, dict):
            all_stats = periods.get("All", {})
        else:
            return None, None

        stat_sections = all_stats.get("stats", []) if isinstance(all_stats, dict) else []

        for section in stat_sections:
            if not isinstance(section, dict):
                continue
            items = section.get("stats", [])
            for item in items:
                if not isinstance(item, dict):
                    continue
                key = item.get("key", "").lower()
                if "expected_goals" in key or "xg" in key:
                    vals = item.get("stats", [])
                    if len(vals) >= 2:
                        try:
                            return float(vals[0]), float(vals[1])
                        except (ValueError, TypeError):
                            pass

        return None, None

    def _score_at_minute(
        self, goals: List[Dict], minute: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        """Reconstruct the score at a given minute.

        Returns:
            (home_score, away_score), losing_team ("home"/"away"/None)
        """
        home, away = 0, 0
        for g in goals:
            if g["minute"] <= minute:
                home = g["home_after"]
                away = g["away_after"]
            else:
                break

        if home == away:
            return (home, away), None
        elif home < away:
            return (home, away), "home"
        else:
            return (home, away), "away"

    def _check_equalization(
        self,
        goals: List[Dict],
        score_at_entry: Tuple[int, int],
        losing_team: str,
        entry_minute: int,
    ) -> Tuple[bool, Optional[int]]:
        """Check if the losing team equalized after the entry minute.

        Returns (equalized: bool, equalization_minute: int or None).
        """
        h_at_entry, a_at_entry = score_at_entry

        for g in goals:
            if g["minute"] <= entry_minute:
                continue
            # Check if this goal was by the losing team and made it equal
            if g["team"] == losing_team:
                if g["home_after"] == g["away_after"]:
                    return True, g["minute"]

        return False, None

    def _get_momentum_at_minute(
        self, momentum_data: List[Dict], minute: int, losing_team: str
    ) -> Optional[float]:
        """Get momentum value at a specific minute for the losing team.

        FotMob momentum: positive = home, negative = away.
        We return a 0-1 normalized value where higher = more losing team momentum.
        """
        # Average momentum over minute 65-75 window
        values = []
        for dp in momentum_data:
            m = dp.get("minute", -1)
            if 65 <= m <= 75:
                values.append(dp.get("value", 0))

        if not values:
            return None

        avg = sum(values) / len(values)

        # If losing team is away, negative momentum = their momentum
        if losing_team == "away":
            avg = -avg

        # Normalize from [-100, 100] to [0, 1]
        normalized = (avg + 100) / 200
        return round(max(0, min(1, normalized)), 3)

    def collect(
        self, leagues: List[str] = None, seasons: List[str] = None,
        force: bool = False,
    ) -> List[Dict]:
        """Main collection method. Fetches all match data.

        Args:
            leagues: List of league names to collect. None = all 5.
            seasons: List of season strings. None = current season only.
            force: If True, re-fetch even cached matches.

        Returns:
            List of all collected match dicts.
        """
        target_seasons = seasons or [CURRENT_SEASON]
        print("=" * 60)
        print("  Soccer Backtest Data Collector")
        print(f"  Seasons: {', '.join(target_seasons)}")
        print(f"  Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print("=" * 60)

        # Step 1: Get all fixtures
        print("\nStep 1: Fetching fixtures from FotMob API...")
        all_fixtures = self.get_all_fixtures(leagues, target_seasons)

        total_fixtures = sum(len(f) for f in all_fixtures.values())
        print(f"\nTotal finished fixtures: {total_fixtures}")

        # Step 2: Build page URL list (one per fixture, keyed by match ID)
        print("\nStep 2: Building page URL list...")
        pages_to_fetch = []

        for league_name, fixtures in all_fixtures.items():
            for fixture in fixtures:
                match_id = str(fixture.get("id", ""))
                page_url = fixture.get("pageUrl", "")

                # Check cache
                if not force and match_id in self._cached_matches:
                    continue

                if page_url:
                    pages_to_fetch.append(
                        {
                            "page_url": page_url,
                            "league": league_name,
                            "expected_id": match_id,
                        }
                    )

        already_cached = len(self._cached_matches)
        print(f"  Already cached: {already_cached}")
        print(f"  Pages to fetch: {len(pages_to_fetch)}")

        if not pages_to_fetch:
            print("\nAll matches already cached!")
            return list(self._cached_matches.values())

        # Step 3: Fetch and parse match pages
        print(f"\nStep 3: Fetching {len(pages_to_fetch)} match pages...")
        print(f"  Estimated time: ~{len(pages_to_fetch) * REQUEST_DELAY:.0f}s")
        print()

        fetched = 0
        errors = 0
        no_data = 0

        for i, page_info in enumerate(pages_to_fetch):
            page_url = page_info["page_url"]
            league = page_info["league"]

            # Progress
            pct = (i + 1) / len(pages_to_fetch) * 100
            print(
                f"\r  [{i+1}/{len(pages_to_fetch)}] ({pct:.0f}%) "
                f"Fetching {page_url[:60]}...",
                end="",
                flush=True,
            )

            # Fetch page
            page_props = self.fetch_match_page(page_url)
            if not page_props:
                errors += 1
                time.sleep(REQUEST_DELAY)
                continue

            # Parse match data
            match_data = self.parse_match_data(page_props, league)
            if not match_data:
                no_data += 1
                time.sleep(REQUEST_DELAY)
                continue

            # Cache it
            self._cached_matches[str(match_data["match_id"])] = match_data
            fetched += 1

            # Save periodically
            if fetched % 25 == 0:
                self._save_cache()

            time.sleep(REQUEST_DELAY)

        print(f"\n\n  Fetched: {fetched} | Errors: {errors} | No data: {no_data}")

        # Final save
        self._save_cache()

        # Summary
        all_matches = list(self._cached_matches.values())
        qualifying = [m for m in all_matches if m.get("qualifying")]
        equalized = [m for m in qualifying if m.get("equalized_after_70")]
        with_momentum = [m for m in all_matches if m.get("momentum_data")]

        print("\n" + "=" * 60)
        print("  Collection Summary")
        print("=" * 60)
        print(f"  Total matches collected: {len(all_matches)}")
        print(f"  With momentum data: {len(with_momentum)}")
        print(f"  Qualifying (losing by 1 at min 70): {len(qualifying)}")
        print(
            f"  Equalized after min 70: {len(equalized)} "
            f"({len(equalized)/len(qualifying)*100:.1f}%)"
            if qualifying
            else "  Equalized after min 70: 0"
        )

        by_league = {}
        for m in all_matches:
            league = m.get("league", "Unknown")
            by_league.setdefault(league, []).append(m)

        print("\n  Per league:")
        for league, matches in sorted(by_league.items()):
            q = sum(1 for m in matches if m.get("qualifying"))
            eq = sum(
                1
                for m in matches
                if m.get("qualifying") and m.get("equalized_after_70")
            )
            print(
                f"    {league}: {len(matches)} matches, "
                f"{q} qualifying, {eq} equalized"
            )

        print(f"\n  Data saved to: {self.matches_file}")
        print("=" * 60)

        return all_matches


def main():
    parser = argparse.ArgumentParser(
        description="Soccer Backtest Data Collector"
    )
    parser.add_argument(
        "--league",
        type=str,
        help='Collect only this league (e.g. "Premier League")',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all matches (ignore cache)",
    )
    parser.add_argument(
        "--season",
        type=str,
        action="append",
        help='Season to collect (e.g. "2024/2025"). Can be repeated. '
             'Use --all-seasons for both 2024/2025 and 2025/2026.',
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Collect all supported seasons (2024/2025 + 2025/2026)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary of cached data without fetching",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    collector = BacktestCollector()

    if args.summary:
        matches = list(collector._cached_matches.values())
        if not matches:
            print("No cached data. Run without --summary to collect.")
            return

        qualifying = [m for m in matches if m.get("qualifying")]
        equalized = [m for m in qualifying if m.get("equalized_after_70")]

        print(f"Cached matches: {len(matches)}")
        print(f"Qualifying (losing by 1 at min 70): {len(qualifying)}")
        if qualifying:
            print(
                f"Equalized: {len(equalized)} ({len(equalized)/len(qualifying)*100:.1f}%)"
            )
        return

    leagues = [args.league] if args.league else None
    if args.all_seasons:
        seasons = ALL_SEASONS
    elif args.season:
        seasons = args.season
    else:
        seasons = None  # defaults to current season
    collector.collect(leagues=leagues, seasons=seasons, force=args.force)


if __name__ == "__main__":
    main()
