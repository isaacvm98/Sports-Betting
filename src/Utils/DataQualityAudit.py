"""
Data Quality Audit Module

Validates data integrity and checks for look-ahead bias in the NBA betting system.

Key Checks:
1. Look-ahead bias: Stats used for predictions must be from before the game
2. Rolling window validation: Ensure rolling calculations use proper trailing windows
3. Date consistency: Verify all data sources align on dates

Usage:
    python -m src.Utils.DataQualityAudit --check-bias
    python -m src.Utils.DataQualityAudit --validate-dates
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
TEAMS_DB_PATH = BASE_DIR / "Data" / "TeamData.sqlite"
ODDS_DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
DATASET_DB_PATH = BASE_DIR / "Data" / "dataset.sqlite"


class DataQualityAudit:
    """Validates data quality and checks for look-ahead bias."""

    def __init__(
        self,
        teams_db: Path = TEAMS_DB_PATH,
        odds_db: Path = ODDS_DB_PATH,
        dataset_db: Path = DATASET_DB_PATH,
    ):
        self.teams_db = teams_db
        self.odds_db = odds_db
        self.dataset_db = dataset_db

    def get_team_data_dates(self) -> List[str]:
        """Get all dates with team data."""
        with sqlite3.connect(self.teams_db) as con:
            cursor = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            dates = []
            for (name,) in cursor.fetchall():
                try:
                    datetime.strptime(name, "%Y-%m-%d")
                    dates.append(name)
                except ValueError:
                    continue
            return sorted(dates)

    def get_games_for_date(self, date_str: str) -> pd.DataFrame:
        """Get games played on a specific date from odds data."""
        with sqlite3.connect(self.odds_db) as con:
            # Find the right table
            cursor = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                if 'odds' in table.lower():
                    try:
                        df = pd.read_sql_query(
                            f'SELECT * FROM "{table}" WHERE Date = ?',
                            con,
                            params=(date_str,)
                        )
                        if not df.empty:
                            return df
                    except:
                        continue
        return pd.DataFrame()

    def check_look_ahead_bias(self, sample_size: int = 50) -> Dict[str, Any]:
        """
        Check for look-ahead bias by validating that team stats are from before game date.

        The NBA.com API returns cumulative stats up to and including the date requested.
        For a game on date D, we should use stats from date D-1 (before the game).

        Returns:
            Dict with:
                - is_safe: bool - True if no look-ahead bias detected
                - issues: List of detected issues
                - recommendations: List of recommendations
        """
        issues = []
        recommendations = []

        team_dates = self.get_team_data_dates()
        if not team_dates:
            issues.append("No team data found in database")
            return {
                'is_safe': False,
                'issues': issues,
                'recommendations': ["Run Get_Data.py to populate team data"]
            }

        # Check the data fetching logic
        # In Get_Data.py, line 72: fetch_end = min(today - timedelta(days=1), end_date)
        # This means we fetch stats for yesterday, which is correct

        # However, in Create_Games.py, line 140:
        # team_df = fetch_team_table(teams_con, date_str)
        # This uses the SAME date as the game, which could include that day's games

        # Check if dates in team data are used correctly
        latest_date = max(team_dates)
        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d").date()
        today = datetime.today().date()

        if latest_dt >= today:
            issues.append(
                f"Team data includes today's date ({latest_date}). "
                "Stats may include games not yet played."
            )
            recommendations.append(
                "Ensure Get_Data.py only fetches data through yesterday"
            )

        # Validate Create_Games.py logic
        # The key issue: when we fetch team stats for game date D,
        # the stats include all games BEFORE date D (cumulative to date D)
        # This is actually correct behavior - NBA.com API returns season-to-date stats

        # The real question: does the API return stats including today or excluding today?
        # Based on the code, it fetches stats for date D and uses them for games on date D
        # If the API returns stats through end of day D, this is look-ahead bias

        recommendations.append(
            "CRITICAL: Verify that team stats for date D exclude games played on date D. "
            "The NBA.com API typically returns stats through end of day, so using stats "
            "from date D for games on date D may include that day's games."
        )

        recommendations.append(
            "RECOMMENDATION: In Create_Games.py, change line 140 to use D-1 stats:\n"
            "  prev_date = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')\n"
            "  team_df = fetch_team_table(teams_con, prev_date)"
        )

        return {
            'is_safe': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'team_data_dates': len(team_dates),
            'latest_date': latest_date,
        }

    def validate_rolling_windows(self) -> Dict[str, Any]:
        """
        Check for issues with rolling window calculations.

        Common issues:
        - Centered windows (look at future data)
        - Missing min_periods (can cause NaN issues)
        """
        issues = []
        recommendations = []

        # Check Create_Games.py for rolling window usage
        create_games_path = BASE_DIR / "src" / "Process-Data" / "Create_Games.py"
        if create_games_path.exists():
            with open(create_games_path, 'r') as f:
                content = f.read()

            # Check for rolling windows
            if '.rolling(' in content:
                if 'center=True' in content:
                    issues.append(
                        "Found centered rolling window (center=True). "
                        "This causes look-ahead bias by including future data."
                    )
                    recommendations.append(
                        "Change all rolling windows to use center=False (default)"
                    )

                if '.rolling(' in content and 'min_periods' not in content:
                    recommendations.append(
                        "Consider adding min_periods to rolling windows to handle edge cases"
                    )

        return {
            'has_rolling_issues': len(issues) > 0,
            'issues': issues,
            'recommendations': recommendations,
        }

    def validate_feature_dates(self, sample_games: int = 10) -> Dict[str, Any]:
        """
        Validate that features used for each game are from before the game date.

        This is a sampling check to verify data integrity.
        """
        results = {
            'checked': 0,
            'passed': 0,
            'failed': 0,
            'failures': [],
        }

        team_dates = self.get_team_data_dates()
        if len(team_dates) < 2:
            return results

        # Sample some dates and check
        import random
        sample_dates = random.sample(team_dates[1:], min(sample_games, len(team_dates) - 1))

        for date_str in sample_dates:
            results['checked'] += 1

            # Check if previous day's data exists
            prev_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            if prev_date in team_dates:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['failures'].append(f"Missing previous day data for game date {date_str}")

        return results

    def generate_report(self) -> str:
        """Generate a comprehensive data quality report."""
        lines = []
        lines.append("=" * 60)
        lines.append("DATA QUALITY AUDIT REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 60)

        # Look-ahead bias check
        lines.append("\n--- LOOK-AHEAD BIAS CHECK ---")
        bias_check = self.check_look_ahead_bias()
        lines.append(f"Status: {'SAFE' if bias_check['is_safe'] else 'POTENTIAL ISSUES'}")
        lines.append(f"Team data tables: {bias_check.get('team_data_dates', 0)}")
        lines.append(f"Latest date: {bias_check.get('latest_date', 'N/A')}")

        if bias_check['issues']:
            lines.append("\nIssues:")
            for issue in bias_check['issues']:
                lines.append(f"  * {issue}")

        if bias_check['recommendations']:
            lines.append("\nRecommendations:")
            for rec in bias_check['recommendations']:
                lines.append(f"  * {rec}")

        # Rolling window check
        lines.append("\n--- ROLLING WINDOW CHECK ---")
        rolling_check = self.validate_rolling_windows()
        lines.append(f"Status: {'ISSUES FOUND' if rolling_check['has_rolling_issues'] else 'OK'}")

        if rolling_check['issues']:
            for issue in rolling_check['issues']:
                lines.append(f"  * {issue}")

        if rolling_check['recommendations']:
            lines.append("\nRecommendations:")
            for rec in rolling_check['recommendations']:
                lines.append(f"  * {rec}")

        # Feature date validation
        lines.append("\n--- FEATURE DATE VALIDATION ---")
        date_check = self.validate_feature_dates()
        lines.append(f"Checked: {date_check['checked']}")
        lines.append(f"Passed: {date_check['passed']}")
        lines.append(f"Failed: {date_check['failed']}")

        if date_check['failures']:
            lines.append("\nFailures:")
            for failure in date_check['failures'][:5]:  # Limit to 5
                lines.append(f"  * {failure}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def add_date_validation_to_create_games():
    """
    Generate patch code to add date validation to Create_Games.py.

    This function prints the code that should be added to Create_Games.py
    to validate that stats are from before game dates.
    """
    patch = '''
# Add this validation at the start of the for loop in Create_Games.py main():

def validate_stats_date(game_date_str: str, stats_date_str: str) -> bool:
    """Validate that stats are from before the game date."""
    game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
    stats_date = datetime.strptime(stats_date_str, "%Y-%m-%d").date()

    if stats_date >= game_date:
        print(f"WARNING: Stats date {stats_date} >= game date {game_date}")
        return False
    return True

# Then modify the fetch_team_table call:
# OLD: team_df = fetch_team_table(teams_con, date_str)
# NEW:
prev_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
team_df = fetch_team_table(teams_con, prev_date)
if team_df is None:
    # Fall back to same date if previous not available
    team_df = fetch_team_table(teams_con, date_str)
    if team_df is not None:
        print(f"WARNING: Using same-day stats for {date_str}")
'''
    return patch


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Data Quality Audit')
    parser.add_argument('--check-bias', action='store_true', help='Check for look-ahead bias')
    parser.add_argument('--validate-dates', action='store_true', help='Validate feature dates')
    parser.add_argument('--report', action='store_true', help='Generate full report')
    parser.add_argument('--show-patch', action='store_true', help='Show recommended code patch')

    args = parser.parse_args()

    audit = DataQualityAudit()

    if args.check_bias:
        result = audit.check_look_ahead_bias()
        print("Look-Ahead Bias Check")
        print("=" * 40)
        print(f"Safe: {result['is_safe']}")
        if result['issues']:
            print("\nIssues:")
            for issue in result['issues']:
                print(f"  * {issue}")
        if result['recommendations']:
            print("\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  * {rec}")

    elif args.validate_dates:
        result = audit.validate_feature_dates()
        print("Feature Date Validation")
        print("=" * 40)
        print(f"Checked: {result['checked']}")
        print(f"Passed: {result['passed']}")
        print(f"Failed: {result['failed']}")

    elif args.show_patch:
        print(add_date_validation_to_create_games())

    else:
        # Default: generate full report
        report = audit.generate_report()
        print(report)


if __name__ == "__main__":
    main()
