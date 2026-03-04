"""
Tests for tiered Kelly criterion functions.
"""

import unittest
from src.Utils.Kelly_Criterion import (
    calculate_kelly_criterion,
    calculate_edge,
    calculate_tiered_kelly,
)


class TestCalculateEdge(unittest.TestCase):
    """Test cases for calculate_edge function."""

    def test_positive_edge(self):
        """Test edge calculation when model has higher probability."""
        edge = calculate_edge(0.60, 0.50)
        self.assertAlmostEqual(edge, 0.10, places=5)

    def test_negative_edge(self):
        """Test edge calculation when market has higher probability."""
        edge = calculate_edge(0.45, 0.55)
        self.assertAlmostEqual(edge, -0.10, places=5)

    def test_zero_edge(self):
        """Test edge calculation when probabilities match."""
        edge = calculate_edge(0.50, 0.50)
        self.assertEqual(edge, 0.0)


class TestTieredKelly(unittest.TestCase):
    """Test cases for calculate_tiered_kelly function."""

    def test_edge_below_minimum_returns_zero(self):
        """Test that edges below 5% return zero bet size."""
        # Model: 52%, Market: 50% -> Edge: 2% (below 5%)
        result = calculate_tiered_kelly(-110, 0.52, 0.50)
        self.assertEqual(result, 0)

    def test_edge_at_minimum_threshold(self):
        """Test that edge at exactly 5% returns a bet."""
        # Model: 55%, Market: 50% -> Edge: 5%
        result = calculate_tiered_kelly(-110, 0.55, 0.50)
        self.assertGreater(result, 0)

    def test_conservative_tier(self):
        """Test conservative tier (5-7% edge)."""
        # Model: 56%, Market: 50% -> Edge: 6%
        result = calculate_tiered_kelly(-110, 0.56, 0.50)
        # Should be positive but relatively small
        self.assertGreater(result, 0)
        self.assertLess(result, 5.0)  # Should be conservative

    def test_moderate_tier(self):
        """Test moderate tier (7-10% edge)."""
        # Model: 58%, Market: 50% -> Edge: 8%
        result = calculate_tiered_kelly(-110, 0.58, 0.50)
        self.assertGreater(result, 0)

    def test_aggressive_tier(self):
        """Test aggressive tier (10%+ edge)."""
        # Model: 62%, Market: 50% -> Edge: 12%
        result = calculate_tiered_kelly(-110, 0.62, 0.50)
        self.assertGreater(result, 0)

    def test_tiered_sizes_increase_with_edge(self):
        """Test that bet sizes increase with edge magnitude."""
        # Conservative: 6% edge
        conservative = calculate_tiered_kelly(-110, 0.56, 0.50)
        # Moderate: 8% edge
        moderate = calculate_tiered_kelly(-110, 0.58, 0.50)
        # Aggressive: 12% edge
        aggressive = calculate_tiered_kelly(-110, 0.62, 0.50)

        # Sizes should increase (accounting for base Kelly also increasing)
        # But the tier multiplier difference should be visible
        self.assertLess(conservative, aggressive)

    def test_max_bet_cap(self):
        """Test that bet size is capped at max_bet_pct."""
        # Very high edge should still be capped
        result = calculate_tiered_kelly(+300, 0.85, 0.30, max_bet_pct=5.0)
        self.assertLessEqual(result, 5.0)

    def test_custom_kelly_fraction(self):
        """Test custom kelly_fraction parameter."""
        base_result = calculate_tiered_kelly(-110, 0.60, 0.50, kelly_fraction=0.25)
        half_result = calculate_tiered_kelly(-110, 0.60, 0.50, kelly_fraction=0.125)

        # Half fraction should give roughly half the bet size
        self.assertAlmostEqual(half_result * 2, base_result, places=1)

    def test_negative_edge_returns_zero(self):
        """Test that negative edge returns zero."""
        result = calculate_tiered_kelly(-110, 0.45, 0.55)
        self.assertEqual(result, 0)

    def test_underdog_odds(self):
        """Test with underdog odds (+150)."""
        result = calculate_tiered_kelly(+150, 0.50, 0.40)  # 10% edge
        self.assertGreater(result, 0)

    def test_favorite_odds(self):
        """Test with favorite odds (-200)."""
        # Need enough edge for positive Kelly with heavy favorite
        result = calculate_tiered_kelly(-200, 0.75, 0.67)  # 8% edge
        self.assertGreater(result, 0)


class TestKellyConsistency(unittest.TestCase):
    """Test consistency between tiered and standard Kelly."""

    def test_tiered_kelly_uses_base_kelly(self):
        """Test that tiered Kelly is based on standard Kelly calculation."""
        # Both should be zero for no edge
        standard = calculate_kelly_criterion(-110, 0.45)
        tiered = calculate_tiered_kelly(-110, 0.45, 0.50)

        self.assertEqual(standard, 0)
        self.assertEqual(tiered, 0)

    def test_tiered_smaller_than_standard(self):
        """Test that tiered Kelly is more conservative than full Kelly."""
        model_prob = 0.60
        market_prob = 0.50

        standard = calculate_kelly_criterion(-110, model_prob)
        tiered = calculate_tiered_kelly(-110, model_prob, market_prob, kelly_fraction=1.0)

        # Even with 100% Kelly fraction, tiered should be smaller due to tier multiplier
        self.assertLess(tiered, standard)


if __name__ == '__main__':
    unittest.main()
