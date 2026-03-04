"""
Polymarket Soccer Provider

Fetches draw market odds from Polymarket for soccer matches.
Soccer events on Polymarket have 3 markets per match:
  1. Home team win (Yes/No)
  2. Draw (Yes/No)  <-- This is what we target
  3. Away team win (Yes/No)

Usage:
    from src.DataProviders.PolymarketSoccerProvider import PolymarketSoccerProvider

    provider = PolymarketSoccerProvider()
    odds = provider.get_draw_odds()
"""

import requests
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"

# Polymarket series IDs for top 5 European leagues
LEAGUE_SERIES_IDS = {
    "Premier League": "10188",
    "La Liga": "10193",
    "Bundesliga": "10194",
    "Serie A": "10203",
    "Ligue 1": "10195",
}

# Mapping data provider team names -> Polymarket team names
# Covers FotMob and SofaScore naming variants for top 5 leagues.
TEAM_NAME_MAP = {
    # Premier League
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa FC",
    "Bournemouth": "AFC Bournemouth",
    "AFC Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford FC",
    "Brighton": "Brighton & Hove Albion FC",
    "Brighton and Hove Albion": "Brighton & Hove Albion FC",
    "Brighton & Hove Albion": "Brighton & Hove Albion FC",
    "Leeds United": "Leeds United FC",
    "Leeds": "Leeds United FC",
    "Burnley": "Burnley FC",
    "Chelsea": "Chelsea FC",
    "Crystal Palace": "Crystal Palace FC",
    "Everton": "Everton FC",
    "Fulham": "Fulham FC",
    "Ipswich": "Ipswich Town FC",
    "Ipswich Town": "Ipswich Town FC",
    "Leicester": "Leicester City FC",
    "Leicester City": "Leicester City FC",
    "Liverpool": "Liverpool FC",
    "Manchester City": "Manchester City FC",
    "Manchester United": "Manchester United FC",
    "Newcastle": "Newcastle United FC",
    "Newcastle United": "Newcastle United FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "Southampton": "Southampton FC",
    "Tottenham": "Tottenham Hotspur FC",
    "Tottenham Hotspur": "Tottenham Hotspur FC",
    "West Ham": "West Ham United FC",
    "West Ham United": "West Ham United FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Wolverhampton": "Wolverhampton Wanderers FC",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers FC",
    # La Liga
    "Athletic Club": "Athletic Club",
    "Athletic Bilbao": "Athletic Club",
    "Atlético Madrid": "Atletico Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Atlético de Madrid": "Atletico Madrid",
    "Barcelona": "FC Barcelona",
    "FC Barcelona": "FC Barcelona",
    "Real Betis": "Real Betis",
    "Celta Vigo": "Celta Vigo",
    "Celta de Vigo": "Celta Vigo",
    "Espanyol": "RCD Espanyol",
    "Getafe": "Getafe CF",
    "Girona": "Girona FC",
    "Las Palmas": "UD Las Palmas",
    "Leganés": "CD Leganes",
    "Leganes": "CD Leganes",
    "Mallorca": "RCD Mallorca",
    "Osasuna": "CA Osasuna",
    "Rayo Vallecano": "Rayo Vallecano",
    "Real Madrid": "Real Madrid",
    "Real Sociedad": "Real Sociedad",
    "Sevilla": "Sevilla FC",
    "Valencia": "Valencia CF",
    "Valladolid": "Real Valladolid",
    "Real Valladolid": "Real Valladolid",
    "Villarreal": "Villarreal CF",
    "Alavés": "Deportivo Alaves",
    "Alaves": "Deportivo Alaves",
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Bayern München": "Bayern Munich",
    "FC Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "Dortmund": "Borussia Dortmund",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Leverkusen": "Bayer Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Frankfurt": "Eintracht Frankfurt",
    "VfB Stuttgart": "VfB Stuttgart",
    "Stuttgart": "VfB Stuttgart",
    "Wolfsburg": "VfL Wolfsburg",
    "VfL Wolfsburg": "VfL Wolfsburg",
    "Freiburg": "SC Freiburg",
    "SC Freiburg": "SC Freiburg",
    "Borussia M'gladbach": "Borussia Monchengladbach",
    "Borussia Mönchengladbach": "Borussia Monchengladbach",
    "Mönchengladbach": "Borussia Monchengladbach",
    "Werder Bremen": "Werder Bremen",
    "Bremen": "Werder Bremen",
    "Mainz": "1. FSV Mainz 05",
    "Mainz 05": "1. FSV Mainz 05",
    "Union Berlin": "1. FC Union Berlin",
    "Augsburg": "FC Augsburg",
    "FC Augsburg": "FC Augsburg",
    "Hoffenheim": "TSG Hoffenheim",
    "TSG Hoffenheim": "TSG Hoffenheim",
    "Heidenheim": "1. FC Heidenheim",
    "FC Heidenheim": "1. FC Heidenheim",
    "Bochum": "VfL Bochum",
    "VfL Bochum": "VfL Bochum",
    "Holstein Kiel": "Holstein Kiel",
    "St. Pauli": "FC St. Pauli",
    "FC St. Pauli": "FC St. Pauli",
    # Serie A
    "Napoli": "SSC Napoli",
    "SSC Napoli": "SSC Napoli",
    "Inter": "Inter Milan",
    "Inter Milan": "Inter Milan",
    "Internazionale": "Inter Milan",
    "Juventus": "Juventus FC",
    "AC Milan": "AC Milan",
    "Milan": "AC Milan",
    "Atalanta": "Atalanta BC",
    "Lazio": "SS Lazio",
    "SS Lazio": "SS Lazio",
    "Roma": "AS Roma",
    "AS Roma": "AS Roma",
    "Fiorentina": "ACF Fiorentina",
    "ACF Fiorentina": "ACF Fiorentina",
    "Bologna": "Bologna FC",
    "Torino": "Torino FC",
    "Udinese": "Udinese Calcio",
    "Genoa": "Genoa CFC",
    "Cagliari": "Cagliari Calcio",
    "Empoli": "Empoli FC",
    "Parma": "Parma Calcio",
    "Parma Calcio 1913": "Parma Calcio",
    "Como": "Como 1907",
    "Como 1907": "Como 1907",
    "Verona": "Hellas Verona",
    "Hellas Verona": "Hellas Verona",
    "Lecce": "US Lecce",
    "US Lecce": "US Lecce",
    "Monza": "AC Monza",
    "AC Monza": "AC Monza",
    "Venezia": "Venezia FC",
    # Ligue 1
    "Paris Saint-Germain": "Paris Saint-Germain FC",
    "PSG": "Paris Saint-Germain FC",
    "Marseille": "Olympique de Marseille",
    "Olympique de Marseille": "Olympique de Marseille",
    "Monaco": "AS Monaco",
    "AS Monaco": "AS Monaco",
    "Lille": "Lille OSC",
    "Lille OSC": "Lille OSC",
    "Lyon": "Olympique Lyonnais",
    "Olympique Lyonnais": "Olympique Lyonnais",
    "Nice": "OGC Nice",
    "OGC Nice": "OGC Nice",
    "Lens": "RC Lens",
    "RC Lens": "RC Lens",
    "Rennes": "Stade Rennais FC",
    "Stade Rennais": "Stade Rennais FC",
    "Strasbourg": "RC Strasbourg",
    "RC Strasbourg": "RC Strasbourg",
    "Toulouse": "Toulouse FC",
    "Toulouse FC": "Toulouse FC",
    "Nantes": "FC Nantes",
    "FC Nantes": "FC Nantes",
    "Montpellier": "Montpellier HSC",
    "Montpellier HSC": "Montpellier HSC",
    "Brest": "Stade Brestois 29",
    "Stade Brestois": "Stade Brestois 29",
    "Reims": "Stade de Reims",
    "Stade de Reims": "Stade de Reims",
    "Le Havre": "Le Havre AC",
    "Le Havre AC": "Le Havre AC",
    "Auxerre": "AJ Auxerre",
    "AJ Auxerre": "AJ Auxerre",
    "Angers": "Angers SCO",
    "Angers SCO": "Angers SCO",
    "Saint-Étienne": "AS Saint-Etienne",
    "Saint-Etienne": "AS Saint-Etienne",
    "AS Saint-Étienne": "AS Saint-Etienne",
}


def _normalize_name(name: str) -> str:
    """Lowercase, strip, remove FC/CF suffixes for fuzzy comparison."""
    n = name.lower().strip()
    for suffix in [" fc", " cf", " sc", " bc", " ac"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


class PolymarketSoccerProvider:
    """Fetches soccer draw market odds from Polymarket."""

    def __init__(self, leagues: List[str] = None):
        """
        Args:
            leagues: List of league names to fetch odds for.
                     Defaults to all top 5 European leagues.
        """
        if leagues:
            self.series_ids = {
                name: LEAGUE_SERIES_IDS[name]
                for name in leagues
                if name in LEAGUE_SERIES_IDS
            }
        else:
            self.series_ids = dict(LEAGUE_SERIES_IDS)

        self._events_cache: Dict[str, List[Dict]] = {}

    def _fetch_events(self, series_id: str) -> List[Dict]:
        """Fetch active events from Polymarket for a league series."""
        try:
            params = {
                "series_id": series_id,
                "active": "true",
                "closed": "false",
                "order": "startTime",
                "ascending": "true",
                "limit": 100,
            }
            response = requests.get(
                f"{GAMMA_API_URL}/events",
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching Polymarket events for series {series_id}: {e}")
            return []

    def _extract_draw_market(self, event: Dict, league: str) -> Optional[Dict]:
        """Extract the draw market from an event's markets list.

        Identifies the draw market by checking the question text for "draw".
        """
        markets = event.get("markets", [])
        title = event.get("title", "")

        draw_market = None
        home_win_market = None
        away_win_market = None

        for market in markets:
            question = market.get("question", "").lower()
            if "draw" in question:
                draw_market = market
            elif "win" in question:
                # Determine if this is home or away win market
                if home_win_market is None:
                    home_win_market = market
                else:
                    away_win_market = market

        if not draw_market:
            return None

        # Extract draw prices
        outcome_prices = draw_market.get("outcomePrices", "")
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, ValueError):
                return None

        if len(outcome_prices) < 2:
            return None

        try:
            draw_yes_price = float(outcome_prices[0])
            draw_no_price = float(outcome_prices[1])
        except (ValueError, TypeError):
            return None

        # Extract CLOB token IDs
        token_ids = draw_market.get("clobTokenIds", "")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except (json.JSONDecodeError, ValueError):
                return None

        if len(token_ids) < 2:
            return None

        draw_yes_token_id = token_ids[0]
        draw_no_token_id = token_ids[1]

        # Extract home/away win prices for reference
        home_win_price = self._get_yes_price(home_win_market)
        away_win_price = self._get_yes_price(away_win_market)

        # Parse team names from event title or market questions
        home_team, away_team = self._parse_teams_from_event(event)

        return {
            "event_title": title,
            "event_id": event.get("id", ""),
            "home_team": home_team,
            "away_team": away_team,
            "draw_question": draw_market.get("question", ""),
            "draw_yes_price": draw_yes_price,
            "draw_no_price": draw_no_price,
            "draw_yes_token_id": draw_yes_token_id,
            "draw_no_token_id": draw_no_token_id,
            "home_win_price": home_win_price,
            "away_win_price": away_win_price,
            "league": league,
            "start_time": event.get("startTime", ""),
            "condition_id": draw_market.get("conditionId", ""),
        }

    def _get_yes_price(self, market: Optional[Dict]) -> Optional[float]:
        """Get the Yes price from a market."""
        if not market:
            return None
        prices = market.get("outcomePrices", "")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except (json.JSONDecodeError, ValueError):
                return None
        if prices and len(prices) > 0:
            try:
                return float(prices[0])
            except (ValueError, TypeError):
                pass
        return None

    def _parse_teams_from_event(self, event: Dict) -> tuple:
        """Parse home and away team names from event data."""
        title = event.get("title", "")
        # Typical format: "Team A vs Team B" or "Team A vs. Team B"
        for sep in [" vs. ", " vs ", " v "]:
            if sep in title:
                parts = title.split(sep, 1)
                if len(parts) == 2:
                    home = parts[0].strip()
                    away = parts[1].strip()
                    # Remove date suffixes like "(Feb 18)"
                    for part in [home, away]:
                        if "(" in part:
                            part = part[: part.index("(")].strip()
                    return home.split("(")[0].strip(), away.split("(")[0].strip()

        # Fallback: try from market questions
        markets = event.get("markets", [])
        teams = []
        for market in markets:
            q = market.get("question", "")
            if "win" in q.lower() and "draw" not in q.lower():
                # "Will Arsenal FC win on Feb 18?" -> "Arsenal FC"
                q_clean = q.lower().replace("will ", "").split(" win")[0].strip()
                teams.append(q_clean.title())
        if len(teams) >= 2:
            return teams[0], teams[1]

        return title, ""

    def get_draw_odds(self) -> Dict[str, Dict]:
        """Get draw market odds for all tracked league matches.

        Returns:
        {
            "Home Team:Away Team": {
                "event_title": str,
                "event_id": str,
                "draw_yes_price": float,
                "draw_yes_token_id": str,
                "draw_no_token_id": str,
                "home_win_price": float,
                "away_win_price": float,
                "league": str,
                "start_time": str,
                ...
            },
            ...
        }
        """
        result = {}

        for league_name, series_id in self.series_ids.items():
            events = self._fetch_events(series_id)
            self._events_cache[league_name] = events

            for event in events:
                draw_data = self._extract_draw_market(event, league_name)
                if draw_data:
                    # Skip resolved/dead markets
                    if draw_data["draw_yes_price"] > 0.95 or draw_data["draw_yes_price"] < 0.02:
                        continue

                    key = f"{draw_data['home_team']}:{draw_data['away_team']}"
                    result[key] = draw_data

        return result

    def get_live_draw_price(self, draw_yes_token_id: str) -> Optional[float]:
        """Get current draw price via CLOB API for a specific token.

        Used during live monitoring for real-time prices.
        """
        try:
            url = f"https://clob.polymarket.com/price"
            params = {"token_id": draw_yes_token_id, "side": "buy"}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return float(data.get("price", 0))
        except Exception as e:
            logger.debug(f"CLOB price fetch failed for {draw_yes_token_id}: {e}")
            # Fallback: try Gamma API
            return self._get_price_from_gamma(draw_yes_token_id)

    def _get_price_from_gamma(self, token_id: str) -> Optional[float]:
        """Fallback price fetch from Gamma API."""
        try:
            response = requests.get(
                f"{GAMMA_API_URL}/markets",
                params={"clob_token_ids": token_id, "limit": 1},
                timeout=10,
            )
            response.raise_for_status()
            markets = response.json()
            if markets:
                prices = markets[0].get("outcomePrices", "")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if prices:
                    return float(prices[0])
        except Exception as e:
            logger.debug(f"Gamma price fetch failed: {e}")
        return None

    def match_fotmob_to_polymarket(
        self,
        fotmob_home: str,
        fotmob_away: str,
        polymarket_events: Dict[str, Dict],
    ) -> Optional[str]:
        """Match a FotMob match to its Polymarket event.

        Uses the TEAM_NAME_MAP for normalized matching, with fuzzy fallback.

        Returns matching market key or None.
        """
        # Strategy 1: Direct mapping
        poly_home = TEAM_NAME_MAP.get(fotmob_home, fotmob_home)
        poly_away = TEAM_NAME_MAP.get(fotmob_away, fotmob_away)

        # Try exact match
        for key in polymarket_events:
            pm_home, pm_away = key.split(":", 1) if ":" in key else (key, "")
            if (poly_home == pm_home and poly_away == pm_away) or \
               (poly_home == pm_away and poly_away == pm_home):
                return key

        # Strategy 2: Fuzzy match (normalized names)
        norm_home = _normalize_name(poly_home)
        norm_away = _normalize_name(poly_away)

        for key, data in polymarket_events.items():
            pm_home = _normalize_name(data.get("home_team", ""))
            pm_away = _normalize_name(data.get("away_team", ""))

            # Check both orderings
            if (norm_home in pm_home or pm_home in norm_home) and \
               (norm_away in pm_away or pm_away in norm_away):
                return key
            if (norm_home in pm_away or pm_away in norm_home) and \
               (norm_away in pm_home or pm_home in norm_away):
                return key

        # Strategy 3: Substring match on event titles
        search_terms = [
            _normalize_name(fotmob_home),
            _normalize_name(fotmob_away),
        ]
        for key, data in polymarket_events.items():
            title = _normalize_name(data.get("event_title", ""))
            if all(term in title for term in search_terms):
                return key

        return None

    @staticmethod
    def probability_to_american_odds(prob: float) -> Optional[int]:
        """Convert probability (0-1) to American odds."""
        if prob is None or prob <= 0 or prob >= 1:
            return None
        if prob >= 0.5:
            return int(round(-100 * prob / (1 - prob)))
        else:
            return int(round(100 * (1 - prob) / prob))


if __name__ == "__main__":
    # Test the provider
    logging.basicConfig(level=logging.INFO)
    provider = PolymarketSoccerProvider()

    print("Fetching draw odds from Polymarket for Top 5 leagues...\n")
    odds = provider.get_draw_odds()

    if not odds:
        print("No active draw markets found.")
    else:
        print(f"Found {len(odds)} matches with draw markets:\n")
        for key, data in odds.items():
            print(f"  [{data['league']}] {key}")
            print(f"    Draw Yes: {data['draw_yes_price']:.1%}")
            print(f"    Home Win: {data['home_win_price']:.1%}" if data['home_win_price'] else "    Home Win: N/A")
            print(f"    Away Win: {data['away_win_price']:.1%}" if data['away_win_price'] else "    Away Win: N/A")
            print(f"    Token ID: {data['draw_yes_token_id'][:20]}...")
            print()
