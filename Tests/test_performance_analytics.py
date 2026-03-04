"""
Tests for PerformanceAnalytics module.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.Utils.PerformanceAnalytics import PerformanceAnalytics


class TestPerformanceAnalytics(unittest.TestCase):
    """Test cases for PerformanceAnalytics."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.positions_file = Path(self.temp_dir) / "positions.json"
        self.trades_file = Path(self.temp_dir) / "trades.json"

        # Create sample positions data
        self.sample_positions = {
            "pos_1": {
                "status": "resolved",
                "bet_side": "home",
                "home_edge": 0.06,
                "away_edge": -0.06,
                "model_home_prob": 0.55,
                "adjusted_home_prob": 0.55,
                "won": True,
                "pnl": 50.0,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "max_profit_pct": 0.15,
                "max_drawdown_pct": -0.05,
            },
            "pos_2": {
                "status": "resolved",
                "bet_side": "away",
                "home_edge": -0.08,
                "away_edge": 0.08,
                "model_away_prob": 0.58,
                "adjusted_away_prob": 0.58,
                "won": False,
                "pnl": -40.0,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "max_profit_pct": 0.05,
                "max_drawdown_pct": -0.20,
            },
            "pos_3": {
                "status": "resolved",
                "bet_side": "home",
                "home_edge": 0.12,
                "away_edge": -0.12,
                "model_home_prob": 0.62,
                "adjusted_home_prob": 0.62,
                "won": True,
                "pnl": 80.0,
                "exit_time": datetime.now(timezone.utc).isoformat(),
            },
            "pos_4": {
                "status": "closed",  # Early exit
                "bet_side": "home",
                "home_edge": 0.06,
                "away_edge": -0.06,
                "won": False,
                "pnl": -25.0,
                "exit_time": datetime.now(timezone.utc).isoformat(),
            },
            "pos_5": {
                "status": "open",  # Still open, should be excluded
                "bet_side": "home",
                "home_edge": 0.07,
            },
        }

        with open(self.positions_file, 'w') as f:
            json.dump(self.sample_positions, f)

        self.analytics = PerformanceAnalytics(data_dir=Path(self.temp_dir))

    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_resolved_positions(self):
        """Test getting resolved positions."""
        resolved = self.analytics.get_resolved_positions()
        self.assertEqual(len(resolved), 3)

    def test_get_closed_positions(self):
        """Test getting closed (early exit) positions."""
        closed = self.analytics.get_closed_positions()
        self.assertEqual(len(closed), 1)

    def test_analyze_by_edge_bucket(self):
        """Test edge bucket analysis."""
        resolved = self.analytics.get_resolved_positions()
        bucket_stats = self.analytics.analyze_by_edge_bucket(resolved)

        # Should have entries for 5-7% and 10%+ buckets
        self.assertIn("5-7%", bucket_stats)
        self.assertIn("10%+", bucket_stats)

    def test_win_rate_by_bucket(self):
        """Test that win rate is calculated correctly per bucket."""
        resolved = self.analytics.get_resolved_positions()
        bucket_stats = self.analytics.analyze_by_edge_bucket(resolved)

        # 10%+ bucket has 1 position, 1 win = 100%
        if "10%+" in bucket_stats:
            self.assertEqual(bucket_stats["10%+"]["win_rate"], 1.0)

    def test_analyze_calibration(self):
        """Test calibration analysis."""
        resolved = self.analytics.get_resolved_positions()
        calibration = self.analytics.analyze_calibration(resolved)

        self.assertIn('overall_win_rate', calibration)
        self.assertIn('calibration_error', calibration)
        self.assertIn('bins', calibration)
        self.assertIn('alerts', calibration)

    def test_calibration_win_rate(self):
        """Test overall win rate in calibration."""
        resolved = self.analytics.get_resolved_positions()
        calibration = self.analytics.analyze_calibration(resolved)

        # 2 wins out of 3 = 66.7%
        expected_win_rate = 2 / 3
        self.assertAlmostEqual(calibration['overall_win_rate'], expected_win_rate, places=2)

    def test_generate_weekly_report(self):
        """Test report generation."""
        report = self.analytics.generate_weekly_report(days=7)

        self.assertIn("PERFORMANCE REPORT", report)
        self.assertIn("Resolved positions:", report)
        self.assertIn("Early exits:", report)

    def test_date_filtering(self):
        """Test that date filtering works."""
        # Create old position
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        self.sample_positions["pos_old"] = {
            "status": "resolved",
            "bet_side": "home",
            "home_edge": 0.10,
            "won": True,
            "pnl": 100.0,
            "exit_time": old_time,
        }
        with open(self.positions_file, 'w') as f:
            json.dump(self.sample_positions, f)

        # Reload analytics
        analytics = PerformanceAnalytics(data_dir=Path(self.temp_dir))

        # Get last 7 days only
        resolved_7d = analytics.get_resolved_positions(days=7)
        resolved_all = analytics.get_resolved_positions(days=None)

        # Old position should not be in 7-day view but should be in all
        self.assertLess(len(resolved_7d), len(resolved_all))


class TestEdgeBuckets(unittest.TestCase):
    """Test edge bucket classification."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.analytics = PerformanceAnalytics(data_dir=Path(self.temp_dir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_edge_bucket_classification(self):
        """Test that edges are classified into correct buckets."""
        # Test 5-7% range
        bucket = self.analytics._get_edge_bucket(0.05)
        self.assertEqual(bucket, "5-7%")

        bucket = self.analytics._get_edge_bucket(0.069)
        self.assertEqual(bucket, "5-7%")

        # Test 7-10% range
        bucket = self.analytics._get_edge_bucket(0.07)
        self.assertEqual(bucket, "7-10%")

        bucket = self.analytics._get_edge_bucket(0.099)
        self.assertEqual(bucket, "7-10%")

        # Test 10%+ range
        bucket = self.analytics._get_edge_bucket(0.10)
        self.assertEqual(bucket, "10%+")

        bucket = self.analytics._get_edge_bucket(0.25)
        self.assertEqual(bucket, "10%+")

    def test_edge_below_minimum_returns_none(self):
        """Test that edges below 5% return None."""
        bucket = self.analytics._get_edge_bucket(0.04)
        self.assertIsNone(bucket)

        bucket = self.analytics._get_edge_bucket(0.0)
        self.assertIsNone(bucket)


class TestCalibrationAlerts(unittest.TestCase):
    """Test calibration alert generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.positions_file = Path(self.temp_dir) / "positions.json"
        self.analytics = PerformanceAnalytics(data_dir=Path(self.temp_dir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_low_win_rate_alert(self):
        """Test alert for low win rate."""
        # Create positions with low win rate
        positions = {}
        for i in range(10):
            positions[f"pos_{i}"] = {
                "status": "resolved",
                "bet_side": "home",
                "home_edge": 0.06,
                "model_home_prob": 0.55,
                "adjusted_home_prob": 0.55,
                "won": i < 4,  # Only 4/10 wins = 40%
                "pnl": 50 if i < 4 else -50,
                "exit_time": datetime.now(timezone.utc).isoformat(),
            }

        with open(self.positions_file, 'w') as f:
            json.dump(positions, f)

        analytics = PerformanceAnalytics(data_dir=Path(self.temp_dir))
        resolved = analytics.get_resolved_positions()
        calibration = analytics.analyze_calibration(resolved)

        # Should have low win rate alert
        self.assertFalse(calibration['is_calibrated'])
        self.assertTrue(any('LOW WIN RATE' in alert for alert in calibration['alerts']))


if __name__ == '__main__':
    unittest.main()
