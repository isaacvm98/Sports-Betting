"""
Tests for ESPN API Provider

Tests the ESPNProvider class functionality including:
- Scoreboard data fetching
- Win probability retrieval
- Injury data (experimental)
- Caching behavior
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timezone

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.DataProviders.ESPNProvider import ESPNProvider


class TestESPNProviderInit:
    """Tests for ESPNProvider initialization."""

    def test_init_creates_session(self):
        """Provider should create a requests session on init."""
        provider = ESPNProvider()
        assert provider.session is not None

    def test_team_ids_mapping(self):
        """All 30 NBA teams should have ID mappings."""
        provider = ESPNProvider()
        assert len(provider.TEAM_IDS) == 30

    def test_team_abbr_mapping(self):
        """Team abbreviation mapping should cover common abbreviations."""
        provider = ESPNProvider()
        # Test some common abbreviations
        assert provider.TEAM_ABBR_TO_NAME.get('LAL') == 'Los Angeles Lakers'
        assert provider.TEAM_ABBR_TO_NAME.get('BOS') == 'Boston Celtics'
        assert provider.TEAM_ABBR_TO_NAME.get('GSW') == 'Golden State Warriors'


class TestESPNProviderCache:
    """Tests for caching behavior."""

    def test_cache_stores_data(self):
        """Data should be stored in cache."""
        provider = ESPNProvider()
        provider._set_cache('test_key', {'data': 'value'})

        assert 'test_key' in provider._cache
        assert provider._cache['test_key'][1] == {'data': 'value'}

    def test_cache_validity(self):
        """Cache should be valid within TTL."""
        provider = ESPNProvider()
        provider._set_cache('test_key', {'data': 'value'})

        assert provider._is_cache_valid('test_key') is True

    def test_cache_expiration(self):
        """Cache should expire after TTL."""
        provider = ESPNProvider()
        provider.CACHE_TTL = 0.1  # 100ms for testing
        provider._set_cache('test_key', {'data': 'value'})

        time.sleep(0.2)
        assert provider._is_cache_valid('test_key') is False

    def test_get_cached_returns_data(self):
        """get_cached should return cached data."""
        provider = ESPNProvider()
        provider._set_cache('test_key', {'data': 'value'})

        result = provider._get_cached('test_key')
        assert result == {'data': 'value'}

    def test_get_cached_returns_none_for_missing(self):
        """get_cached should return None for missing keys."""
        provider = ESPNProvider()
        result = provider._get_cached('nonexistent_key')
        assert result is None


class TestESPNProviderScoreboard:
    """Tests for scoreboard functionality."""

    @patch.object(ESPNProvider, '_make_request')
    def test_get_scoreboard_makes_request(self, mock_request):
        """get_scoreboard should make API request."""
        mock_request.return_value = {'events': []}
        provider = ESPNProvider()

        provider.get_scoreboard()

        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert 'scoreboard' in call_args[0][0]

    @patch.object(ESPNProvider, '_make_request')
    def test_get_scoreboard_with_date(self, mock_request):
        """get_scoreboard should pass date parameter."""
        mock_request.return_value = {'events': []}
        provider = ESPNProvider()

        provider.get_scoreboard(date='20260215')

        call_args = mock_request.call_args
        assert call_args[1]['params'] == {'dates': '20260215'}

    @patch.object(ESPNProvider, '_make_request')
    def test_get_todays_games_parses_events(self, mock_request):
        """get_todays_games should parse events correctly."""
        mock_request.return_value = {
            'events': [{
                'id': '401234567',
                'date': '2026-02-15T19:00:00Z',
                'competitions': [{
                    'id': '401234567',
                    'competitors': [
                        {
                            'homeAway': 'home',
                            'team': {'abbreviation': 'LAL', 'displayName': 'Lakers'},
                            'score': '105'
                        },
                        {
                            'homeAway': 'away',
                            'team': {'abbreviation': 'BOS', 'displayName': 'Celtics'},
                            'score': '102'
                        }
                    ],
                    'status': {
                        'type': {'name': 'STATUS_FINAL', 'description': 'Final'}
                    },
                    'venue': {'fullName': 'Crypto.com Arena'}
                }]
            }]
        }
        provider = ESPNProvider()

        games = provider.get_todays_games()

        assert len(games) == 1
        assert games[0]['home_team'] == 'Los Angeles Lakers'
        assert games[0]['away_team'] == 'Boston Celtics'
        assert games[0]['event_id'] == '401234567'


class TestESPNProviderWinProbability:
    """Tests for win probability functionality."""

    @patch.object(ESPNProvider, '_make_request')
    def test_get_win_probability_constructs_url(self, mock_request):
        """get_win_probability should construct correct URL."""
        mock_request.return_value = {'items': []}
        provider = ESPNProvider()

        provider.get_win_probability('401234567', '401234567')

        call_args = mock_request.call_args
        assert '401234567' in call_args[0][0]
        assert 'probabilities' in call_args[0][0]

    @patch.object(ESPNProvider, '_make_request')
    def test_get_current_win_probability_parses_data(self, mock_request):
        """get_current_win_probability should parse probability data."""
        mock_request.return_value = {
            'items': [
                {'homeWinPercentage': 0.5, 'timestamp': '2026-02-15T19:00:00Z'},
                {'homeWinPercentage': 0.65, 'timestamp': '2026-02-15T19:30:00Z'}
            ]
        }
        provider = ESPNProvider()

        result = provider.get_current_win_probability('401234567')

        assert result['home_team_prob'] == 0.65
        assert result['away_team_prob'] == 0.35
        assert result['is_live'] is True

    @patch.object(ESPNProvider, '_make_request')
    def test_get_current_win_probability_single_item(self, mock_request):
        """Single probability item means game hasn't started."""
        mock_request.return_value = {
            'items': [
                {'homeWinPercentage': 0.55, 'timestamp': '2026-02-15T19:00:00Z'}
            ]
        }
        provider = ESPNProvider()

        result = provider.get_current_win_probability('401234567')

        assert result['home_team_prob'] == 0.55
        assert result['is_live'] is False

    @patch.object(ESPNProvider, '_make_request')
    def test_get_current_win_probability_empty_items(self, mock_request):
        """Empty items should return None."""
        mock_request.return_value = {'items': []}
        provider = ESPNProvider()

        result = provider.get_current_win_probability('401234567')

        assert result is None


class TestESPNProviderInjuries:
    """Tests for injury data functionality."""

    def test_get_team_injuries_unknown_team(self):
        """Unknown team should return None."""
        provider = ESPNProvider()

        result = provider.get_team_injuries('Unknown Team')

        assert result is None

    @patch.object(ESPNProvider, '_make_request')
    def test_get_team_injuries_makes_request(self, mock_request):
        """get_team_injuries should make API request with correct team ID."""
        mock_request.return_value = {'items': []}
        provider = ESPNProvider()

        provider.get_team_injuries('Los Angeles Lakers')

        call_args = mock_request.call_args
        # Lakers team ID is 13
        assert '/teams/13/injuries' in call_args[0][0]

    @patch.object(ESPNProvider, '_make_request')
    def test_get_all_injuries_fetches_multiple_teams(self, mock_request):
        """get_all_injuries should fetch for all specified teams."""
        mock_request.return_value = {'items': []}
        provider = ESPNProvider()

        teams = ['Los Angeles Lakers', 'Boston Celtics']
        result = provider.get_all_injuries(teams)

        assert 'Los Angeles Lakers' in result
        assert 'Boston Celtics' in result


class TestESPNProviderTeams:
    """Tests for team data functionality."""

    @patch.object(ESPNProvider, '_make_request')
    def test_get_teams_makes_request(self, mock_request):
        """get_teams should make API request."""
        mock_request.return_value = {'sports': [{'leagues': [{'teams': []}]}]}
        provider = ESPNProvider()

        provider.get_teams()

        call_args = mock_request.call_args
        assert 'teams' in call_args[0][0]

    @patch.object(ESPNProvider, '_make_request')
    def test_get_team_roster_makes_request(self, mock_request):
        """get_team_roster should make API request with team ID."""
        mock_request.return_value = {'athletes': []}
        provider = ESPNProvider()

        provider.get_team_roster('Los Angeles Lakers')

        call_args = mock_request.call_args
        assert '/teams/13/roster' in call_args[0][0]


class TestESPNProviderIntegration:
    """Integration tests that make real API calls.

    These tests are marked slow and can be skipped in CI with:
    pytest -m "not integration"
    """

    @pytest.mark.integration
    def test_real_scoreboard_fetch(self):
        """Test real scoreboard API call."""
        provider = ESPNProvider()
        result = provider.get_scoreboard()

        # Should return dict with events key
        assert isinstance(result, dict)
        assert 'events' in result

    @pytest.mark.integration
    def test_real_teams_fetch(self):
        """Test real teams API call."""
        provider = ESPNProvider()
        result = provider.get_teams()

        assert isinstance(result, dict)
        # Should have sports/leagues/teams structure
        assert 'sports' in result

    @pytest.mark.integration
    def test_real_standings_fetch(self):
        """Test real standings API call."""
        provider = ESPNProvider()
        result = provider.get_standings()

        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
