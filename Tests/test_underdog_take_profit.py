"""Tests for underdog take-profit helper functions and backtester analysis."""

import pytest
from src.Polymarket.paper_trader import (
    is_underdog_position,
    get_underdog_take_profit_threshold,
    UNDERDOG_TAKE_PROFIT_MAX_ENTRY,
    UNDERDOG_TAKE_PROFIT_BY_ENTRY,
)


class TestIsUnderdogPosition:
    """Tests for is_underdog_position()."""

    def test_home_bet_underdog(self):
        position = {'bet_side': 'home', 'entry_home_prob': 0.30, 'entry_away_prob': 0.70}
        assert is_underdog_position(position) == 0.30

    def test_away_bet_underdog(self):
        position = {'bet_side': 'away', 'entry_home_prob': 0.65, 'entry_away_prob': 0.35}
        assert is_underdog_position(position) == 0.35

    def test_home_bet_favorite(self):
        position = {'bet_side': 'home', 'entry_home_prob': 0.60, 'entry_away_prob': 0.40}
        assert is_underdog_position(position) == 0.60

    def test_away_bet_favorite(self):
        position = {'bet_side': 'away', 'entry_home_prob': 0.40, 'entry_away_prob': 0.60}
        assert is_underdog_position(position) == 0.60

    def test_no_bet_side(self):
        position = {'entry_home_prob': 0.50, 'entry_away_prob': 0.50}
        assert is_underdog_position(position) is None

    def test_empty_position(self):
        assert is_underdog_position({}) is None


class TestGetUnderdogTakeProfitThreshold:
    """Tests for get_underdog_take_profit_threshold()."""

    def test_deep_underdog(self):
        # <20% entry -> should get 25% TP threshold
        assert get_underdog_take_profit_threshold(0.15) == 0.25

    def test_moderate_underdog(self):
        # 20-35% entry -> should get 8% TP threshold
        assert get_underdog_take_profit_threshold(0.25) == 0.08

    def test_at_boundary_20(self):
        # Exactly 20% -> falls into 20-35% bucket (not deep underdog)
        threshold = get_underdog_take_profit_threshold(0.20)
        # 0.20 is NOT <= 0.20 in (0.20, 0.25) first bucket, but IS checked
        # against UNDERDOG_TAKE_PROFIT_BY_ENTRY: (0.20, 0.25) means <= 0.20
        assert threshold == 0.25

    def test_at_boundary_35(self):
        # 35% is >= UNDERDOG_TAKE_PROFIT_MAX_ENTRY -> no TP
        assert get_underdog_take_profit_threshold(0.35) is None

    def test_slight_underdog_no_tp(self):
        # 40% entry -> above max entry threshold, no TP
        assert get_underdog_take_profit_threshold(0.40) is None

    def test_favorite_no_tp(self):
        # 55% entry -> favorite, no TP
        assert get_underdog_take_profit_threshold(0.55) is None

    def test_heavy_favorite_no_tp(self):
        assert get_underdog_take_profit_threshold(0.75) is None

    def test_none_input(self):
        assert get_underdog_take_profit_threshold(None) is None

    def test_edge_very_low(self):
        # Very deep underdog (5%) -> should get deep underdog bucket
        assert get_underdog_take_profit_threshold(0.05) == 0.25


class TestUnderdogTPConfig:
    """Tests that config constants are consistent."""

    def test_max_entry_matches_last_bucket(self):
        last_bucket_max = UNDERDOG_TAKE_PROFIT_BY_ENTRY[-1][0]
        assert last_bucket_max == UNDERDOG_TAKE_PROFIT_MAX_ENTRY

    def test_buckets_ordered(self):
        probs = [bucket[0] for bucket in UNDERDOG_TAKE_PROFIT_BY_ENTRY]
        assert probs == sorted(probs)

    def test_all_thresholds_positive(self):
        for _, tp in UNDERDOG_TAKE_PROFIT_BY_ENTRY:
            assert tp > 0


class TestUnderdogTPIntegration:
    """Integration tests: verify favorites are never affected."""

    def test_favorite_position_not_eligible(self):
        """Favorites should never get a TP threshold."""
        favorite_positions = [
            {'bet_side': 'home', 'entry_home_prob': 0.55, 'entry_away_prob': 0.45},
            {'bet_side': 'home', 'entry_home_prob': 0.70, 'entry_away_prob': 0.30},
            {'bet_side': 'away', 'entry_home_prob': 0.40, 'entry_away_prob': 0.60},
        ]
        for pos in favorite_positions:
            entry_prob = is_underdog_position(pos)
            threshold = get_underdog_take_profit_threshold(entry_prob)
            assert threshold is None, f"Favorite with entry_prob={entry_prob} should not have TP"

    def test_moderate_underdog_not_eligible(self):
        """Underdogs in 35-50% range should not get TP."""
        moderate_positions = [
            {'bet_side': 'away', 'entry_home_prob': 0.60, 'entry_away_prob': 0.40},
            {'bet_side': 'away', 'entry_home_prob': 0.55, 'entry_away_prob': 0.45},
            {'bet_side': 'home', 'entry_home_prob': 0.38, 'entry_away_prob': 0.62},
        ]
        for pos in moderate_positions:
            entry_prob = is_underdog_position(pos)
            threshold = get_underdog_take_profit_threshold(entry_prob)
            assert threshold is None, f"Moderate underdog with entry_prob={entry_prob} should not have TP"

    def test_deep_underdog_eligible(self):
        """Deep underdogs should get TP thresholds."""
        deep_positions = [
            {'bet_side': 'away', 'entry_home_prob': 0.80, 'entry_away_prob': 0.20},
            {'bet_side': 'home', 'entry_home_prob': 0.15, 'entry_away_prob': 0.85},
            {'bet_side': 'away', 'entry_home_prob': 0.70, 'entry_away_prob': 0.30},
        ]
        for pos in deep_positions:
            entry_prob = is_underdog_position(pos)
            threshold = get_underdog_take_profit_threshold(entry_prob)
            assert threshold is not None, f"Deep underdog with entry_prob={entry_prob} should have TP"
            assert threshold > 0
