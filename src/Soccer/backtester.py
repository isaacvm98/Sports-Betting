"""
Soccer Momentum Draw Backtester

Analyzes historical match data to validate the momentum-draw theory:
"Teams losing by 1 goal at minute 70-75 with high attacking momentum
equalize more often than the market implies."

Key analyses:
1. Base equalization rate for teams trailing by 1 at min 70-75
2. Equalization rate by momentum bucket
3. Equalization rate by xG momentum bucket
4. League breakdown
5. P&L simulation at various Polymarket draw price assumptions

Generates plotly charts and HTML report.

Usage:
    python -m src.Soccer.backtester
    python -m src.Soccer.backtester --min-minute 65 --max-minute 80
"""

import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

DATA_DIR = Path("Data/soccer_backtest")
CURRENT_SEASON = "2025/2026"


class SoccerBacktester:
    """Analyzes collected match data to validate the momentum-draw theory."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        min_minute: int = 70,
        max_minute: int = 75,
    ):
        self.data_dir = Path(data_dir)
        self.matches_file = self.data_dir / "matches.json"
        self.min_minute = min_minute
        self.max_minute = max_minute
        self.matches: List[Dict] = []
        self.qualifying: List[Dict] = []
        self._load_data()

    def _load_data(self):
        """Load collected match data."""
        if not self.matches_file.exists():
            raise FileNotFoundError(
                f"No data found at {self.matches_file}. "
                "Run backtest_collector first."
            )

        with open(self.matches_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.matches = data.get("matches", [])
        # Filter qualifying matches (team losing by 1 at target minutes)
        self.qualifying = [m for m in self.matches if m.get("qualifying")]

        logger.info(
            f"Loaded {len(self.matches)} matches, "
            f"{len(self.qualifying)} qualifying"
        )

    def base_equalization_rate(self) -> Dict:
        """Calculate the base rate: of all matches where a team was trailing
        by 1 at min 70, what % saw an equalizer?"""
        if not self.qualifying:
            return {"error": "No qualifying matches found"}

        equalized = [m for m in self.qualifying if m.get("equalized_after_70")]

        return {
            "total_qualifying": len(self.qualifying),
            "equalized": len(equalized),
            "rate": round(len(equalized) / len(self.qualifying), 3),
            "pct": round(len(equalized) / len(self.qualifying) * 100, 1),
        }

    def equalization_by_momentum_bucket(self) -> Dict:
        """Split qualifying matches by FotMob momentum bucket and show
        equalization rate per bucket.

        Momentum scale: 0 = all opponent momentum, 0.5 = neutral, 1 = all losing team momentum.
        """
        buckets = {
            "0.0-0.3 (opponent)": (0.0, 0.3),
            "0.3-0.4": (0.3, 0.4),
            "0.4-0.5": (0.4, 0.5),
            "0.5-0.6": (0.5, 0.6),
            "0.6-0.7": (0.6, 0.7),
            "0.7-1.0 (losing team)": (0.7, 1.0),
        }

        results = {}
        with_momentum = [
            m for m in self.qualifying if m.get("momentum_at_70") is not None
        ]

        if not with_momentum:
            return {"error": "No qualifying matches with momentum data"}

        for bucket_name, (low, high) in buckets.items():
            in_bucket = [
                m
                for m in with_momentum
                if low <= m["momentum_at_70"] < high
                or (high == 1.0 and m["momentum_at_70"] == 1.0)
            ]
            equalized = [m for m in in_bucket if m.get("equalized_after_70")]
            results[bucket_name] = {
                "count": len(in_bucket),
                "equalized": len(equalized),
                "rate": round(len(equalized) / len(in_bucket), 3)
                if in_bucket
                else 0,
                "pct": round(len(equalized) / len(in_bucket) * 100, 1)
                if in_bucket
                else 0,
            }

        return {
            "total_with_momentum": len(with_momentum),
            "buckets": results,
        }

    def equalization_by_xg_share(self) -> Dict:
        """Split qualifying matches by losing team's xG share and show
        equalization rate per bucket.

        Higher xG share = losing team was creating more chances = more "momentum".
        """
        buckets = {
            "0.0-0.25 (low)": (0.0, 0.25),
            "0.25-0.35": (0.25, 0.35),
            "0.35-0.45": (0.35, 0.45),
            "0.45-0.55": (0.45, 0.55),
            "0.55-0.65": (0.55, 0.65),
            "0.65-1.0 (high)": (0.65, 1.0),
        }

        results = {}
        with_xg = [
            m
            for m in self.qualifying
            if m.get("losing_team_xg_share") is not None
        ]

        if not with_xg:
            return {"error": "No qualifying matches with xG data"}

        for bucket_name, (low, high) in buckets.items():
            in_bucket = [
                m
                for m in with_xg
                if low <= m["losing_team_xg_share"] < high
                or (high == 1.0 and m["losing_team_xg_share"] == 1.0)
            ]
            equalized = [m for m in in_bucket if m.get("equalized_after_70")]
            results[bucket_name] = {
                "count": len(in_bucket),
                "equalized": len(equalized),
                "rate": round(len(equalized) / len(in_bucket), 3)
                if in_bucket
                else 0,
                "pct": round(len(equalized) / len(in_bucket) * 100, 1)
                if in_bucket
                else 0,
            }

        return {
            "total_with_xg": len(with_xg),
            "buckets": results,
        }

    def league_breakdown(self) -> Dict:
        """Equalization rates broken down by league."""
        by_league = defaultdict(list)
        for m in self.qualifying:
            by_league[m.get("league", "Unknown")].append(m)

        results = {}
        for league, matches in sorted(by_league.items()):
            equalized = [m for m in matches if m.get("equalized_after_70")]
            results[league] = {
                "qualifying": len(matches),
                "equalized": len(equalized),
                "rate": round(len(equalized) / len(matches), 3)
                if matches
                else 0,
                "pct": round(len(equalized) / len(matches) * 100, 1)
                if matches
                else 0,
            }

        return results

    def pnl_simulation(
        self,
        draw_prices: List[float] = None,
        momentum_threshold: float = 0.5,
        use_momentum: bool = True,
    ) -> Dict:
        """Simulate P&L for buying draw shares when signal triggers.

        Strategy: Buy "Yes Draw" shares at draw_price, sell at $1 if equalizer
        happens. If no equalizer, lose the entry price.

        Since we're selling when the equalizer happens (not holding to end),
        the payout depends on what draw shares are worth post-equalization.
        At equalization, draw price typically jumps to ~0.35-0.50.

        Simplified model:
        - Entry: buy at draw_price
        - If equalizer: sell at exit_price (estimated 0.40)
        - If no equalizer: shares expire near 0 (lose entry)
        """
        if draw_prices is None:
            draw_prices = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        # Filter by momentum threshold if using momentum
        if use_momentum:
            signal_matches = [
                m
                for m in self.qualifying
                if m.get("momentum_at_70") is not None
                and m["momentum_at_70"] >= momentum_threshold
            ]
        else:
            signal_matches = list(self.qualifying)

        if not signal_matches:
            return {"error": "No signal matches found"}

        equalized = [m for m in signal_matches if m.get("equalized_after_70")]
        eq_rate = len(equalized) / len(signal_matches)

        # Estimated exit price when equalizer happens
        # After equalization (1-1 or 2-2 at min ~75-85), draw price ~0.35-0.50
        exit_price_on_equalize = 0.42

        results = {}
        for entry_price in draw_prices:
            profit_per_win = exit_price_on_equalize - entry_price
            loss_per_loss = entry_price  # lose the entry cost

            n_wins = len(equalized)
            n_losses = len(signal_matches) - n_wins
            total_pnl = n_wins * profit_per_win - n_losses * loss_per_loss

            # Per-trade stats
            expected_value = eq_rate * profit_per_win - (1 - eq_rate) * loss_per_loss

            results[f"${entry_price:.2f}"] = {
                "entry_price": entry_price,
                "trades": len(signal_matches),
                "wins": n_wins,
                "losses": n_losses,
                "eq_rate": round(eq_rate, 3),
                "total_pnl": round(total_pnl, 2),
                "pnl_per_trade": round(total_pnl / len(signal_matches), 4),
                "expected_value": round(expected_value, 4),
                "roi_pct": round(
                    total_pnl / (entry_price * len(signal_matches)) * 100, 1
                ),
            }

        return {
            "momentum_threshold": momentum_threshold if use_momentum else "none",
            "signal_matches": len(signal_matches),
            "equalized": len(equalized),
            "eq_rate": round(eq_rate, 3),
            "exit_price": exit_price_on_equalize,
            "by_entry_price": results,
        }

    def generate_charts(self) -> str:
        """Generate plotly charts and save as HTML report.

        Returns path to the HTML report.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("plotly not installed. Install with: pip install plotly")
            return ""

        charts = []

        # Chart 1: Equalization rate by momentum bucket
        mom_data = self.equalization_by_momentum_bucket()
        if "buckets" in mom_data:
            buckets = mom_data["buckets"]
            names = list(buckets.keys())
            rates = [b["pct"] for b in buckets.values()]
            counts = [b["count"] for b in buckets.values()]

            fig1 = go.Figure()
            fig1.add_trace(
                go.Bar(
                    x=names,
                    y=rates,
                    text=[f"{r:.1f}%<br>(n={c})" for r, c in zip(rates, counts)],
                    textposition="auto",
                    marker_color=[
                        "#ef4444" if r < 25 else "#f59e0b" if r < 35 else "#22c55e"
                        for r in rates
                    ],
                )
            )
            fig1.update_layout(
                title="Equalization Rate by FotMob Momentum Bucket (min 65-75 avg)",
                xaxis_title="Momentum Bucket (losing team)",
                yaxis_title="Equalization Rate (%)",
                template="plotly_dark",
                height=500,
            )
            charts.append(("momentum_buckets", fig1))

        # Chart 2: Equalization rate by xG share
        xg_data = self.equalization_by_xg_share()
        if "buckets" in xg_data:
            buckets = xg_data["buckets"]
            names = list(buckets.keys())
            rates = [b["pct"] for b in buckets.values()]
            counts = [b["count"] for b in buckets.values()]

            fig2 = go.Figure()
            fig2.add_trace(
                go.Bar(
                    x=names,
                    y=rates,
                    text=[f"{r:.1f}%<br>(n={c})" for r, c in zip(rates, counts)],
                    textposition="auto",
                    marker_color=[
                        "#ef4444" if r < 25 else "#f59e0b" if r < 35 else "#22c55e"
                        for r in rates
                    ],
                )
            )
            fig2.update_layout(
                title="Equalization Rate by Losing Team xG Share",
                xaxis_title="xG Share Bucket",
                yaxis_title="Equalization Rate (%)",
                template="plotly_dark",
                height=500,
            )
            charts.append(("xg_buckets", fig2))

        # Chart 3: League breakdown
        league_data = self.league_breakdown()
        if league_data:
            leagues = list(league_data.keys())
            rates = [d["pct"] for d in league_data.values()]
            counts = [d["qualifying"] for d in league_data.values()]

            fig3 = go.Figure()
            fig3.add_trace(
                go.Bar(
                    x=leagues,
                    y=rates,
                    text=[f"{r:.1f}%<br>(n={c})" for r, c in zip(rates, counts)],
                    textposition="auto",
                    marker_color="#3b82f6",
                )
            )
            fig3.update_layout(
                title="Equalization Rate by League",
                xaxis_title="League",
                yaxis_title="Equalization Rate (%)",
                template="plotly_dark",
                height=500,
            )
            charts.append(("league_breakdown", fig3))

        # Chart 4: P&L simulation at different entry prices
        pnl_data = self.pnl_simulation(use_momentum=False)
        if "by_entry_price" in pnl_data:
            prices = list(pnl_data["by_entry_price"].keys())
            pnls = [d["total_pnl"] for d in pnl_data["by_entry_price"].values()]
            rois = [d["roi_pct"] for d in pnl_data["by_entry_price"].values()]

            fig4 = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Total P&L by Entry Price", "ROI % by Entry Price"),
            )
            fig4.add_trace(
                go.Bar(
                    x=prices,
                    y=pnls,
                    text=[f"${p:+.2f}" for p in pnls],
                    textposition="auto",
                    marker_color=[
                        "#22c55e" if p > 0 else "#ef4444" for p in pnls
                    ],
                    name="Total P&L",
                ),
                row=1, col=1,
            )
            fig4.add_trace(
                go.Bar(
                    x=prices,
                    y=rois,
                    text=[f"{r:+.1f}%" for r in rois],
                    textposition="auto",
                    marker_color=[
                        "#22c55e" if r > 0 else "#ef4444" for r in rois
                    ],
                    name="ROI %",
                ),
                row=1, col=2,
            )
            fig4.update_layout(
                title=f"P&L Simulation (all qualifying, eq rate={pnl_data['eq_rate']:.1%}, exit@${pnl_data['exit_price']:.2f})",
                template="plotly_dark",
                height=500,
                showlegend=False,
            )
            charts.append(("pnl_simulation", fig4))

        # Chart 5: Momentum scatter - momentum vs outcome
        momentum_matches = [
            m for m in self.qualifying if m.get("momentum_at_70") is not None
        ]
        if momentum_matches:
            x_vals = [m["momentum_at_70"] for m in momentum_matches]
            y_vals = [1 if m.get("equalized_after_70") else 0 for m in momentum_matches]
            colors = ["#22c55e" if y else "#ef4444" for y in y_vals]
            labels = [
                f"{m['home_team']} vs {m['away_team']}<br>"
                f"{m['final_score']} ({m['league']})<br>"
                f"Momentum: {m['momentum_at_70']:.2f}"
                for m in momentum_matches
            ]

            fig5 = go.Figure()
            fig5.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=[y + (hash(l) % 100) / 1000 - 0.05 for y, l in zip(y_vals, labels)],  # jitter
                    mode="markers",
                    marker=dict(color=colors, size=8, opacity=0.7),
                    text=labels,
                    hoverinfo="text",
                )
            )
            fig5.update_layout(
                title="Momentum at Min 70 vs Equalization Outcome",
                xaxis_title="Losing Team Momentum (0=opponent, 1=losing team)",
                yaxis_title="Equalized (1=Yes, 0=No)",
                template="plotly_dark",
                height=500,
            )
            charts.append(("momentum_scatter", fig5))

        # Chart 6: P&L by momentum threshold
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        threshold_results = []
        for t in thresholds:
            pnl = self.pnl_simulation(
                draw_prices=[0.15],
                momentum_threshold=t,
                use_momentum=True,
            )
            if "by_entry_price" in pnl:
                entry = list(pnl["by_entry_price"].values())[0]
                threshold_results.append({
                    "threshold": t,
                    "trades": entry["trades"],
                    "eq_rate": entry["eq_rate"],
                    "pnl": entry["total_pnl"],
                    "roi": entry["roi_pct"],
                })

        if threshold_results:
            fig6 = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "Equalization Rate by Momentum Threshold",
                    "ROI % by Momentum Threshold (entry=$0.15)",
                ),
            )
            fig6.add_trace(
                go.Scatter(
                    x=[r["threshold"] for r in threshold_results],
                    y=[r["eq_rate"] * 100 for r in threshold_results],
                    mode="lines+markers",
                    text=[f"n={r['trades']}" for r in threshold_results],
                    name="Eq Rate",
                    line=dict(color="#3b82f6"),
                ),
                row=1, col=1,
            )
            fig6.add_trace(
                go.Scatter(
                    x=[r["threshold"] for r in threshold_results],
                    y=[r["roi"] for r in threshold_results],
                    mode="lines+markers",
                    text=[f"n={r['trades']}" for r in threshold_results],
                    name="ROI %",
                    line=dict(color="#22c55e"),
                ),
                row=1, col=2,
            )
            fig6.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
            fig6.update_layout(
                title="Optimal Momentum Threshold Analysis",
                template="plotly_dark",
                height=500,
            )
            charts.append(("threshold_optimization", fig6))

        # Build HTML report
        report_path = self.data_dir / "report.html"
        html_parts = [
            "<!DOCTYPE html>",
            '<html><head><meta charset="utf-8">',
            "<title>Soccer Momentum Draw Backtest Report</title>",
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            "<style>",
            "body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; padding: 20px; max-width: 1200px; margin: 0 auto; }",
            "h1 { color: #3b82f6; } h2 { color: #8b5cf6; margin-top: 30px; }",
            ".stat-box { background: #16213e; border-radius: 8px; padding: 15px 20px; margin: 10px 0; display: inline-block; margin-right: 15px; }",
            ".stat-value { font-size: 28px; font-weight: bold; color: #22c55e; }",
            ".stat-label { font-size: 14px; color: #94a3b8; }",
            ".chart-container { margin: 20px 0; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }",
            "th { background: #1e293b; color: #94a3b8; }",
            "</style></head><body>",
            f"<h1>Soccer Momentum Draw Backtest</h1>",
            f"<p>Season: {CURRENT_SEASON} | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>",
        ]

        # Summary stats
        base = self.base_equalization_rate()
        html_parts.append("<h2>Key Findings</h2>")
        html_parts.append('<div>')
        html_parts.append(
            f'<div class="stat-box"><div class="stat-value">{len(self.matches)}</div>'
            f'<div class="stat-label">Total Matches</div></div>'
        )
        html_parts.append(
            f'<div class="stat-box"><div class="stat-value">{base.get("total_qualifying", 0)}</div>'
            f'<div class="stat-label">Qualifying (trailing by 1 at min 70)</div></div>'
        )
        html_parts.append(
            f'<div class="stat-box"><div class="stat-value">{base.get("pct", 0)}%</div>'
            f'<div class="stat-label">Base Equalization Rate</div></div>'
        )
        html_parts.append("</div>")

        # Charts
        for chart_name, fig in charts:
            html_parts.append(f'<div class="chart-container">')
            html_parts.append(
                fig.to_html(full_html=False, include_plotlyjs=False)
            )
            html_parts.append("</div>")

        # Qualifying matches table
        html_parts.append("<h2>Qualifying Matches Detail</h2>")
        html_parts.append("<table><tr>")
        html_parts.append(
            "<th>Date</th><th>League</th><th>Match</th><th>Score at 70'</th>"
            "<th>Final</th><th>Momentum</th><th>xG Share</th><th>Equalized?</th>"
            "<th>Eq Min</th>"
        )
        html_parts.append("</tr>")
        for m in sorted(self.qualifying, key=lambda x: x.get("date", ""), reverse=True):
            eq_str = "Yes" if m.get("equalized_after_70") else "No"
            eq_color = "#22c55e" if m.get("equalized_after_70") else "#ef4444"
            mom = f"{m.get('momentum_at_70', 0):.2f}" if m.get("momentum_at_70") is not None else "-"
            xg = f"{m.get('losing_team_xg_share', 0):.2f}" if m.get("losing_team_xg_share") is not None else "-"
            score70 = f"{m['score_at_70'][0]}-{m['score_at_70'][1]}" if m.get("score_at_70") else "-"
            eq_min = str(m.get("equalization_minute", "-"))
            html_parts.append(
                f'<tr><td>{m.get("date", "")}</td><td>{m.get("league", "")}</td>'
                f'<td>{m.get("home_team", "")} vs {m.get("away_team", "")}</td>'
                f'<td>{score70}</td><td>{m.get("final_score", "")}</td>'
                f'<td>{mom}</td><td>{xg}</td>'
                f'<td style="color:{eq_color}">{eq_str}</td>'
                f'<td>{eq_min}</td></tr>'
            )
        html_parts.append("</table>")

        html_parts.append("</body></html>")

        report_html = "\n".join(html_parts)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)

        return str(report_path)

    def generate_console_report(self) -> str:
        """Generate a text summary report for console output."""
        lines = []
        lines.append("=" * 65)
        lines.append("    SOCCER MOMENTUM DRAW BACKTEST REPORT")
        lines.append(f"    {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 65)

        # Base rate
        base = self.base_equalization_rate()
        lines.append(f"\nTotal matches analyzed: {len(self.matches)}")
        lines.append(
            f"Qualifying (trailing by 1 at min 70): {base.get('total_qualifying', 0)}"
        )
        lines.append(f"Base equalization rate: {base.get('pct', 0)}%")

        # Momentum analysis
        lines.append("\n--- EQUALIZATION BY MOMENTUM BUCKET ---")
        mom = self.equalization_by_momentum_bucket()
        if "buckets" in mom:
            lines.append(f"Matches with momentum data: {mom['total_with_momentum']}")
            for name, data in mom["buckets"].items():
                bar = "#" * int(data["pct"] / 2) if data["pct"] > 0 else ""
                lines.append(
                    f"  {name:25s}: {data['pct']:5.1f}% "
                    f"({data['equalized']}/{data['count']}) {bar}"
                )
        else:
            lines.append(f"  {mom.get('error', 'No data')}")

        # xG analysis
        lines.append("\n--- EQUALIZATION BY XG SHARE ---")
        xg = self.equalization_by_xg_share()
        if "buckets" in xg:
            lines.append(f"Matches with xG data: {xg['total_with_xg']}")
            for name, data in xg["buckets"].items():
                bar = "#" * int(data["pct"] / 2) if data["pct"] > 0 else ""
                lines.append(
                    f"  {name:25s}: {data['pct']:5.1f}% "
                    f"({data['equalized']}/{data['count']}) {bar}"
                )
        else:
            lines.append(f"  {xg.get('error', 'No data')}")

        # League breakdown
        lines.append("\n--- LEAGUE BREAKDOWN ---")
        leagues = self.league_breakdown()
        for league, data in leagues.items():
            lines.append(
                f"  {league:20s}: {data['pct']:5.1f}% "
                f"({data['equalized']}/{data['qualifying']})"
            )

        # P&L simulation
        lines.append("\n--- P&L SIMULATION (no momentum filter) ---")
        pnl = self.pnl_simulation(use_momentum=False)
        if "by_entry_price" in pnl:
            lines.append(
                f"Signal matches: {pnl['signal_matches']} | "
                f"Eq rate: {pnl['eq_rate']:.1%} | "
                f"Exit price: ${pnl['exit_price']:.2f}"
            )
            lines.append(
                f"{'Entry':>8s} {'Trades':>7s} {'Wins':>5s} "
                f"{'P&L':>10s} {'ROI':>8s} {'EV/trade':>10s}"
            )
            for name, data in pnl["by_entry_price"].items():
                lines.append(
                    f"  {name:>6s} {data['trades']:>7d} {data['wins']:>5d} "
                    f"${data['total_pnl']:>+8.2f} {data['roi_pct']:>+7.1f}% "
                    f"${data['expected_value']:>+8.4f}"
                )

        # P&L with momentum filter
        lines.append("\n--- P&L SIMULATION (momentum >= 0.5) ---")
        pnl_mom = self.pnl_simulation(
            momentum_threshold=0.5, use_momentum=True
        )
        if "by_entry_price" in pnl_mom:
            lines.append(
                f"Signal matches: {pnl_mom['signal_matches']} | "
                f"Eq rate: {pnl_mom['eq_rate']:.1%}"
            )
            lines.append(
                f"{'Entry':>8s} {'Trades':>7s} {'Wins':>5s} "
                f"{'P&L':>10s} {'ROI':>8s} {'EV/trade':>10s}"
            )
            for name, data in pnl_mom["by_entry_price"].items():
                lines.append(
                    f"  {name:>6s} {data['trades']:>7d} {data['wins']:>5d} "
                    f"${data['total_pnl']:>+8.2f} {data['roi_pct']:>+7.1f}% "
                    f"${data['expected_value']:>+8.4f}"
                )

        # Key takeaway
        lines.append("\n--- KEY TAKEAWAY ---")
        if "buckets" in mom:
            high_mom = mom["buckets"].get("0.7-1.0 (losing team)", {})
            low_mom = mom["buckets"].get("0.0-0.3 (opponent)", {})
            if high_mom.get("count", 0) > 0 and low_mom.get("count", 0) > 0:
                diff = high_mom["pct"] - low_mom["pct"]
                lines.append(
                    f"High momentum eq rate: {high_mom['pct']:.1f}% "
                    f"vs Low: {low_mom['pct']:.1f}% (diff: {diff:+.1f}pp)"
                )
                if diff > 10:
                    lines.append(
                        "-> STRONG signal: high momentum significantly increases equalization"
                    )
                elif diff > 5:
                    lines.append(
                        "-> MODERATE signal: momentum has some predictive power"
                    )
                else:
                    lines.append(
                        "-> WEAK signal: momentum doesn't strongly predict equalization"
                    )

        lines.append("\n" + "=" * 65)
        return "\n".join(lines)

    def save_results(self):
        """Save analysis results as JSON."""
        results = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_matches": len(self.matches),
            "base_equalization_rate": self.base_equalization_rate(),
            "by_momentum": self.equalization_by_momentum_bucket(),
            "by_xg_share": self.equalization_by_xg_share(),
            "by_league": self.league_breakdown(),
            "pnl_no_filter": self.pnl_simulation(use_momentum=False),
            "pnl_momentum_05": self.pnl_simulation(
                momentum_threshold=0.5, use_momentum=True
            ),
        }

        results_path = self.data_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return str(results_path)


def main():
    parser = argparse.ArgumentParser(description="Soccer Momentum Draw Backtester")
    parser.add_argument(
        "--min-minute", type=int, default=70, help="Min minute for qualifying"
    )
    parser.add_argument(
        "--max-minute", type=int, default=75, help="Max minute for qualifying"
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="Skip chart generation"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        bt = SoccerBacktester(
            min_minute=args.min_minute, max_minute=args.max_minute
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the collector first: python -m src.Soccer.backtest_collector")
        return

    # Console report
    report = bt.generate_console_report()
    print(report)

    # Save results JSON
    results_path = bt.save_results()
    print(f"\nResults saved to: {results_path}")

    # Generate charts
    if not args.no_charts:
        print("\nGenerating charts...")
        report_path = bt.generate_charts()
        if report_path:
            print(f"HTML report saved to: {report_path}")
            print("Open in browser to view interactive charts.")


if __name__ == "__main__":
    main()
