"""
Tests for DrawdownManager module.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.Utils.DrawdownManager import DrawdownManager


class TestDrawdownManager(unittest.TestCase):
    """Test cases for DrawdownManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dm = DrawdownManager(
            data_dir=Path(self.temp_dir),
            starting_bankroll=1000.0,
            max_daily_loss=0.05,
            max_weekly_loss=0.10,
            max_total_drawdown=0.20,
        )

    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initial_state(self):
        """Test initial state is correct."""
        status = self.dm.get_status()
        self.assertTrue(status['can_trade'])
        self.assertEqual(status['current_bankroll'], 1000.0)
        self.assertEqual(status['peak_bankroll'], 1000.0)
        self.assertEqual(status['daily_pnl'], 0.0)
        self.assertEqual(status['weekly_pnl'], 0.0)

    def test_can_trade_initially(self):
        """Test that trading is allowed initially."""
        self.assertTrue(self.dm.can_trade())

    def test_record_winning_pnl(self):
        """Test recording a winning trade."""
        status = self.dm.record_pnl(100.0, 'test_pos_1')
        self.assertEqual(status['current_bankroll'], 1100.0)
        self.assertEqual(status['peak_bankroll'], 1100.0)
        self.assertEqual(status['daily_pnl'], 100.0)
        self.assertTrue(status['can_trade'])

    def test_record_losing_pnl(self):
        """Test recording a losing trade."""
        status = self.dm.record_pnl(-30.0, 'test_pos_1')
        self.assertEqual(status['current_bankroll'], 970.0)
        self.assertEqual(status['peak_bankroll'], 1000.0)
        self.assertEqual(status['daily_pnl'], -30.0)
        self.assertTrue(status['can_trade'])

    def test_daily_limit_exceeded(self):
        """Test that daily limit stops trading."""
        # Lose 5% of bankroll (exactly at limit)
        status = self.dm.record_pnl(-50.0, 'test_pos_1')
        self.assertFalse(status['can_trade'])
        self.assertIn('DAILY_LIMIT', status['halt_reason'])

    def test_weekly_limit_exceeded(self):
        """Test that weekly limit stops trading."""
        # Create new manager with different limits
        dm = DrawdownManager(
            data_dir=Path(self.temp_dir),
            starting_bankroll=1000.0,
            max_daily_loss=0.20,  # Higher daily limit
            max_weekly_loss=0.10,
        )
        # Lose 10% over multiple days
        dm.record_pnl(-50.0, 'test_pos_1')
        status = dm.record_pnl(-50.0, 'test_pos_2')
        self.assertFalse(status['can_trade'])
        self.assertIn('WEEKLY_LIMIT', status['halt_reason'])

    def test_total_drawdown_limit(self):
        """Test that total drawdown limit stops trading."""
        dm = DrawdownManager(
            data_dir=Path(self.temp_dir),
            starting_bankroll=1000.0,
            max_daily_loss=0.50,  # Higher daily limit
            max_weekly_loss=0.50,  # Higher weekly limit
            max_total_drawdown=0.20,
        )
        # Lose 20% from peak
        status = dm.record_pnl(-200.0, 'test_pos_1')
        self.assertFalse(status['can_trade'])
        self.assertIn('TOTAL_DRAWDOWN', status['halt_reason'])

    def test_alert_threshold(self):
        """Test that alerts are generated at 80% of limits."""
        dm = DrawdownManager(
            data_dir=Path(self.temp_dir),
            starting_bankroll=1000.0,
            max_daily_loss=0.05,
            alert_threshold=0.80,
        )
        # Lose 4% (80% of 5% limit)
        status = dm.record_pnl(-40.0, 'test_pos_1')
        self.assertTrue(len(status['alerts']) > 0)
        self.assertIn('WARNING', status['alerts'][0])

    def test_sync_bankroll(self):
        """Test syncing bankroll from external source."""
        self.dm.sync_bankroll(1500.0)
        status = self.dm.get_status()
        self.assertEqual(status['current_bankroll'], 1500.0)
        self.assertEqual(status['peak_bankroll'], 1500.0)

    def test_reset_halt(self):
        """Test resetting halt status."""
        # Trigger halt with a smaller loss that only triggers daily limit
        self.dm.record_pnl(-50.0, 'test_pos_1')  # 5% daily limit hit
        self.assertFalse(self.dm.can_trade())

        # Reset halt, daily, and sync bankroll back up
        self.dm.reset_halt()
        self.dm.reset_daily()
        self.dm.sync_bankroll(1000.0)  # Reset bankroll to avoid drawdown limit
        self.assertTrue(self.dm.can_trade())

    def test_persistence(self):
        """Test that state is persisted to file."""
        self.dm.record_pnl(50.0, 'test_pos_1')

        # Create new instance pointing to same file
        dm2 = DrawdownManager(
            data_dir=Path(self.temp_dir),
            starting_bankroll=1000.0,
        )
        status = dm2.get_status()
        self.assertEqual(status['current_bankroll'], 1050.0)

    def test_get_history(self):
        """Test getting P&L history."""
        self.dm.record_pnl(50.0, 'pos_1')
        self.dm.record_pnl(-30.0, 'pos_2')
        self.dm.record_pnl(20.0, 'pos_3')

        history = self.dm.get_history(days=7)
        self.assertEqual(len(history), 3)


class TestDrawdownEdgeCases(unittest.TestCase):
    """Test edge cases for DrawdownManager."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_zero_bankroll(self):
        """Test handling of zero bankroll."""
        dm = DrawdownManager(
            data_dir=Path(self.temp_dir),
            starting_bankroll=0.0,
        )
        # Should not crash
        status = dm.get_status()
        self.assertEqual(status['current_bankroll'], 0.0)

    def test_multiple_wins_updates_peak(self):
        """Test that consecutive wins update peak correctly."""
        dm = DrawdownManager(
            data_dir=Path(self.temp_dir),
            starting_bankroll=1000.0,
        )
        dm.record_pnl(100.0, 'pos_1')
        dm.record_pnl(50.0, 'pos_2')

        status = dm.get_status()
        self.assertEqual(status['current_bankroll'], 1150.0)
        self.assertEqual(status['peak_bankroll'], 1150.0)

    def test_drawdown_calculation_after_peak(self):
        """Test drawdown is calculated correctly after establishing a peak."""
        dm = DrawdownManager(
            data_dir=Path(self.temp_dir),
            starting_bankroll=1000.0,
        )
        # Win then lose
        dm.record_pnl(200.0, 'pos_1')  # Peak at 1200
        dm.record_pnl(-100.0, 'pos_2')  # Now at 1100

        status = dm.get_status()
        self.assertEqual(status['peak_bankroll'], 1200.0)
        self.assertEqual(status['current_bankroll'], 1100.0)
        # Drawdown should be (1100 - 1200) / 1200 = -8.33%
        expected_dd = (1100 - 1200) / 1200
        self.assertAlmostEqual(status['total_drawdown'], expected_dd, places=3)


if __name__ == '__main__':
    unittest.main()
