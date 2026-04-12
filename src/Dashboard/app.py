"""
Unified Bloomberg-Style Paper Trading Dashboard

A professional, dark-themed dashboard combining NBA and CBB paper trading
with live position tracking and interactive Plotly charts.

Usage:
    streamlit run src/Dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# Import data loader (relative import workaround for Streamlit)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.Dashboard.data_loader import (
    get_unified_positions,
    get_portfolio_summary,
    get_pnl_history,
    get_recent_activity,
    get_edge_analysis,
    load_drawdown_state,
    load_alerts,
    get_espn_live_probabilities,
    match_positions_with_espn,
    get_espn_injuries_for_teams,
    close_position_partial,
    STARTING_BANKROLL,
)

ET = ZoneInfo("America/New_York")

# Bloomberg-style CSS
BLOOMBERG_CSS = """
<style>
    /* Dark background */
    .stApp {
        background-color: #0d1117;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* Bloomberg orange headers */
    h1, h2, h3 {
        color: #ff9500 !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
    }

    /* Ticker bar styling */
    .ticker-bar {
        background: linear-gradient(90deg, #1a1f29 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 12px 20px;
        margin-bottom: 1rem;
        font-family: 'Consolas', 'Monaco', monospace;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
    }

    [data-testid="stMetricValue"] {
        color: #c9d1d9 !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
    }

    /* Green/Red delta colors */
    [data-testid="stMetricDelta"] svg {
        display: none;
    }

    /* DataFrames - monospace and dense */
    .stDataFrame {
        font-family: 'Consolas', 'Monaco', monospace !important;
    }

    [data-testid="stDataFrame"] {
        font-family: 'Consolas', 'Monaco', monospace !important;
    }

    /* Dividers */
    hr {
        border-color: #30363d !important;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #161b22;
        border-color: #30363d;
    }

    /* Badge styles */
    .badge-win {
        background-color: #238636;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }

    .badge-loss {
        background-color: #da3633;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }

    .badge-entry {
        background-color: #1f6feb;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }

    .badge-exit {
        background-color: #6e7681;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }

    /* Progress bar customization */
    .stProgress > div > div {
        background-color: #30363d;
    }

    /* Caption/small text */
    .stCaption, small {
        color: #8b949e !important;
    }
</style>
"""


def get_file_mtime() -> float:
    """Get max modification time of data files."""
    files = [
        Path("Data/paper_trading_v2/positions.json"),
        Path("Data/cbb_paper_trading/positions.json"),
        Path("Data/soccer_paper_trading/positions.json"),
    ]
    mtimes = []
    for f in files:
        if f.exists():
            mtimes.append(f.stat().st_mtime)
    return max(mtimes) if mtimes else 0


@st.fragment(run_every=5)
def auto_refresh_fragment():
    """Fragment that checks for file changes and triggers refresh."""
    current_mtime = get_file_mtime()

    if 'last_mtime' not in st.session_state:
        st.session_state.last_mtime = current_mtime
        return

    if current_mtime > st.session_state.last_mtime:
        st.session_state.last_mtime = current_mtime
        st.rerun()


def render_ticker_bar(summary: dict):
    """Render Bloomberg-style ticker bar with NBA, CBB, and Soccer portfolios."""
    now = datetime.now(ET)
    nba_color = "#00ff41" if summary['nba_pnl'] >= 0 else "#ff4444"
    cbb_color = "#00ff41" if summary['cbb_pnl'] >= 0 else "#ff4444"
    soccer_color = "#00ff41" if summary.get('soccer_pnl', 0) >= 0 else "#ff4444"
    status = "ACTIVE"

    ticker_html = f"""
    <div class="ticker-bar">
        <span style="color: #1f77b4; font-weight: bold;">NBA</span>
        <span style="color: #c9d1d9;"> ${summary['nba_bankroll']:,.2f}</span>
        <span style="color: {nba_color};"> ({summary['nba_pnl']:+,.2f} | {summary['nba_roi']:+.1f}%)</span>
        <span style="color: #8b949e; margin: 0 12px;">|</span>
        <span style="color: #ff7f0e; font-weight: bold;">CBB</span>
        <span style="color: #c9d1d9;"> ${summary['cbb_bankroll']:,.2f}</span>
        <span style="color: {cbb_color};"> ({summary['cbb_pnl']:+,.2f} | {summary['cbb_roi']:+.1f}%)</span>
        <span style="color: #8b949e; margin: 0 12px;">|</span>
        <span style="color: #2ca02c; font-weight: bold;">SOCCER</span>
        <span style="color: #c9d1d9;"> ${summary.get('soccer_bankroll', 1000):,.2f}</span>
        <span style="color: {soccer_color};"> ({summary.get('soccer_pnl', 0):+,.2f} | {summary.get('soccer_roi', 0):+.1f}%)</span>
        <span style="color: #8b949e; margin: 0 12px;">|</span>
        <span style="color: #238636; font-weight: bold;">{status}</span>
        <span style="color: #8b949e; margin: 0 8px;">|</span>
        <span style="color: #8b949e;">{now.strftime('%I:%M %p ET')}</span>
    </div>
    """
    st.markdown(ticker_html, unsafe_allow_html=True)


def render_portfolio_metrics(summary: dict):
    """Render portfolio summary metrics with separate NBA, CBB, and Soccer sections."""
    # NBA Portfolio
    st.markdown("<span style='color: #1f77b4; font-weight: bold; font-size: 1.1em;'>NBA PORTFOLIO</span>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Bankroll",
            f"${summary['nba_bankroll']:,.2f}",
            f"{summary['nba_pnl']:+,.2f}",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "ROI",
            f"{summary['nba_roi']:+.1f}%",
            None
        )

    with col3:
        st.metric(
            "Open Positions",
            f"{summary['nba_open']}",
            None
        )

    with col4:
        nba_record = f"{summary['nba_wins']}W-{summary['nba_losses']}L"
        st.metric(
            "Win Rate",
            f"{summary['nba_win_rate']:.1f}%",
            nba_record
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # CBB Portfolio
    st.markdown("<span style='color: #ff7f0e; font-weight: bold; font-size: 1.1em;'>CBB PORTFOLIO</span>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Bankroll",
            f"${summary['cbb_bankroll']:,.2f}",
            f"{summary['cbb_pnl']:+,.2f}",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "ROI",
            f"{summary['cbb_roi']:+.1f}%",
            None
        )

    with col3:
        st.metric(
            "Open Positions",
            f"{summary['cbb_open']}",
            None
        )

    with col4:
        cbb_record = f"{summary['cbb_wins']}W-{summary['cbb_losses']}L"
        st.metric(
            "Win Rate",
            f"{summary['cbb_win_rate']:.1f}%",
            cbb_record
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Soccer Portfolio
    st.markdown("<span style='color: #2ca02c; font-weight: bold; font-size: 1.1em;'>SOCCER PORTFOLIO</span> <span style='color: #8b949e; font-size: 0.9em;'>(Momentum Draw)</span>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Bankroll",
            f"${summary.get('soccer_bankroll', 1000):,.2f}",
            f"{summary.get('soccer_pnl', 0):+,.2f}",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "ROI",
            f"{summary.get('soccer_roi', 0):+.1f}%",
            None
        )

    with col3:
        st.metric(
            "Open Positions",
            f"{summary.get('soccer_open', 0)}",
            None
        )

    with col4:
        soccer_record = f"{summary.get('soccer_wins', 0)}W-{summary.get('soccer_losses', 0)}L"
        st.metric(
            "Win Rate",
            f"{summary.get('soccer_win_rate', 0):.1f}%",
            soccer_record
        )


def render_risk_status():
    """Render P&L tracking bars."""
    drawdown = load_drawdown_state()

    col1, col2, col3 = st.columns(3)

    # Daily P&L
    daily_pnl = sum(drawdown.get('daily_pnl', {}).values()) if drawdown else 0

    with col1:
        st.markdown("**Daily P&L**")
        color = "#00ff41" if daily_pnl >= 0 else "#ff4444"
        st.markdown(f"<span style='color:{color};font-size:1.2em;'>${daily_pnl:+.2f}</span>", unsafe_allow_html=True)

    # Weekly P&L
    weekly_pnl = sum(drawdown.get('weekly_pnl', {}).values()) if drawdown else 0

    with col2:
        st.markdown("**Weekly P&L**")
        color = "#00ff41" if weekly_pnl >= 0 else "#ff4444"
        st.markdown(f"<span style='color:{color};font-size:1.2em;'>${weekly_pnl:+.2f}</span>", unsafe_allow_html=True)

    # Total Drawdown from peak
    peak = drawdown.get('peak_bankroll', STARTING_BANKROLL) if drawdown else STARTING_BANKROLL
    current = drawdown.get('current_bankroll', STARTING_BANKROLL) if drawdown else STARTING_BANKROLL
    dd_pct = ((peak - current) / peak * 100) if peak > 0 else 0

    with col3:
        st.markdown("**Drawdown from Peak**")
        if dd_pct < 10:
            color = "#00ff41"
        elif dd_pct < 15:
            color = "#f0ad4e"
        else:
            color = "#ff4444"
        st.markdown(f"<span style='color:{color};font-size:1.2em;'>{dd_pct:.1f}%</span> <span style='color:#8b949e;'>(peak: ${peak:.2f})</span>", unsafe_allow_html=True)


def render_positions_grid(positions: list, league_filter: str):
    """Render live positions grid."""
    # Filter positions
    if league_filter != "All":
        positions = [p for p in positions if p['league'] == league_filter]

    # Separate by status
    open_pos = [p for p in positions if p['status'] == 'OPEN']
    pending_pos = [p for p in positions if p['status'] == 'PENDING_RESOLUTION']
    resolved_pos = [p for p in positions if p['status'] == 'RESOLVED']

    # Open Positions
    st.subheader("Open Positions")

    if open_pos:
        rows = []
        for p in open_pos:
            # Format price change with color
            change_pct = p['price_change_pct']
            change_color = "#00ff41" if change_pct >= 0 else "#ff4444"

            rows.append({
                "League": p['league'],
                "Game": p['game'],
                "Type": p['bet_type'],
                "Side": p['bet_side'],
                "Entry": f"{p['entry_price']:.1%}",
                "Current": f"{p['current_price']:.1%}",
                "Change": f"{change_pct:+.1f}%",
                "Size": f"{p['size_pct']:.1f}%",
                "P&L": f"${p['pnl']:+.2f}",
                "Edge": p['edge_display'],
            })

        df = pd.DataFrame(rows)

        # Apply styling
        def style_change(val):
            if '+' in str(val):
                return 'color: #00ff41'
            elif '-' in str(val):
                return 'color: #ff4444'
            return ''

        def style_pnl(val):
            if '+' in str(val) or (str(val).startswith('$') and not str(val).startswith('$-')):
                return 'color: #00ff41'
            elif '-' in str(val):
                return 'color: #ff4444'
            return ''

        styled_df = df.style.map(style_change, subset=['Change'])
        styled_df = styled_df.map(style_pnl, subset=['P&L'])

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Total exposure
        total_exposure = sum(p['size_pct'] for p in open_pos)
        total_amount = sum(p['amount'] for p in open_pos)
        st.caption(f"Total exposure: {total_exposure:.1f}% (${total_amount:.2f})")
    else:
        st.info("No open positions")

    # Recent Results
    if resolved_pos:
        with st.expander(f"Recent Results ({len(resolved_pos)} resolved)", expanded=False):
            rows = []
            for p in sorted(resolved_pos, key=lambda x: x.get('entry_time', ''), reverse=True)[:15]:
                result = "WIN" if p.get('won') else "LOSS" if p.get('won') is False else "-"
                rows.append({
                    "League": p['league'],
                    "Game": p['game'],
                    "Type": p['bet_type'],
                    "Side": p['bet_side'],
                    "Result": result,
                    "P&L": f"${p['pnl']:+.2f}" if p['pnl'] else "-",
                    "Edge": p['edge_display'],
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)


def render_pnl_chart():
    """Render interactive P&L chart with Plotly."""
    st.subheader("Cumulative P&L")

    pnl_history = get_pnl_history()

    fig = go.Figure()

    # NBA trace
    if pnl_history["NBA"]:
        nba_times = [datetime.fromisoformat(p['time'].replace('Z', '+00:00')) for p in pnl_history["NBA"]]
        nba_cumulative = [p['cumulative'] for p in pnl_history["NBA"]]
        nba_hover = [f"Game: {p['game']}<br>P&L: ${p['pnl']:+.2f}<br>Type: {p['type']}" for p in pnl_history["NBA"]]

        fig.add_trace(go.Scatter(
            x=nba_times,
            y=nba_cumulative,
            mode='lines+markers',
            name='NBA',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            hovertemplate='%{text}<br>Cumulative: $%{y:.2f}<extra></extra>',
            text=nba_hover,
        ))

    # CBB trace
    if pnl_history["CBB"]:
        cbb_times = [datetime.fromisoformat(p['time'].replace('Z', '+00:00')) for p in pnl_history["CBB"]]
        cbb_cumulative = [p['cumulative'] for p in pnl_history["CBB"]]
        cbb_hover = [f"Game: {p['game']}<br>P&L: ${p['pnl']:+.2f}<br>Result: {p['result']}" for p in pnl_history["CBB"]]

        fig.add_trace(go.Scatter(
            x=cbb_times,
            y=cbb_cumulative,
            mode='lines+markers',
            name='CBB',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=6),
            hovertemplate='%{text}<br>Cumulative: $%{y:.2f}<extra></extra>',
            text=cbb_hover,
        ))

    # Soccer trace
    if pnl_history.get("SOCCER"):
        soccer_times = [datetime.fromisoformat(p['time'].replace('Z', '+00:00')) for p in pnl_history["SOCCER"]]
        soccer_cumulative = [p['cumulative'] for p in pnl_history["SOCCER"]]
        soccer_hover = [f"Match: {p['game']}<br>P&L: ${p['pnl']:+.2f}<br>Type: {p.get('type', '')}" for p in pnl_history["SOCCER"]]

        fig.add_trace(go.Scatter(
            x=soccer_times,
            y=soccer_cumulative,
            mode='lines+markers',
            name='Soccer',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=6),
            hovertemplate='%{text}<br>Cumulative: $%{y:.2f}<extra></extra>',
            text=soccer_hover,
        ))

    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        margin=dict(l=50, r=20, t=30, b=50),
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Consolas, Monaco, monospace", size=12)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#30363d',
            title="",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#30363d',
            title="Cumulative P&L ($)",
            tickprefix="$",
        ),
        hovermode='x unified',
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5)

    st.plotly_chart(fig, use_container_width=True)


def render_espn_live_panel(positions: list, league_filter: str):
    """Render ESPN Live Win Probability panel with sparklines."""
    st.subheader("📊 ESPN Live Win Probability")

    # Tab selection: Positions vs All Games
    tab1, tab2 = st.tabs(["Open Positions", "All Live Games"])

    # Get ESPN data for both leagues
    espn_nba = get_espn_live_probabilities('nba') if league_filter in ['All', 'NBA'] else {}
    espn_cbb = get_espn_live_probabilities('cbb') if league_filter in ['All', 'CBB'] else {}
    espn_data = {**espn_nba, **espn_cbb}

    with tab1:
        _render_positions_espn(positions, league_filter, espn_data)

    with tab2:
        _render_all_live_games(espn_data, league_filter)


def _render_all_live_games(espn_data: dict, league_filter: str):
    """Render all live games from ESPN."""
    if not espn_data:
        st.info("No live games available")
        return

    # Filter by league if needed
    if league_filter == "NBA":
        espn_data = {k: v for k, v in espn_data.items() if 'Celtics' in k or 'Lakers' in k or any(
            team in k for team in ['Hawks', 'Nets', 'Hornets', 'Bulls', 'Cavaliers', 'Mavericks',
            'Nuggets', 'Pistons', 'Warriors', 'Rockets', 'Pacers', 'Clippers', 'Grizzlies',
            'Heat', 'Bucks', 'Timberwolves', 'Pelicans', 'Knicks', 'Thunder', 'Magic', '76ers',
            'Suns', 'Trail Blazers', 'Kings', 'Spurs', 'Raptors', 'Jazz', 'Wizards']
        )}

    for game_key, data in espn_data.items():
        home = data.get('home_team', '')
        away = data.get('away_team', '')
        home_prob = data.get('espn_home_prob')
        away_prob = data.get('espn_away_prob')

        if home_prob is None:
            continue

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Live indicator
            if data.get('is_live'):
                badge = "🔴 LIVE"
            elif data.get('is_final'):
                badge = "✅ FINAL"
            else:
                badge = "⏳ Pre"

            score = f"{data.get('away_score', 0)}-{data.get('home_score', 0)}"
            time_str = f"Q{data.get('period', '?')} {data.get('clock', '')}"

            st.markdown(f"**{away} @ {home}** {badge}")
            st.markdown(f"<span style='font-size:1.2em;font-weight:bold;'>{score}</span> {time_str}", unsafe_allow_html=True)

        with col2:
            # Win probability bars
            home_pct = int(home_prob * 100)
            away_pct = int(away_prob * 100)

            st.markdown(f"**{home[:15]}**: {home_prob:.1%}")
            st.progress(home_pct / 100)

            st.markdown(f"**{away[:15]}**: {away_prob:.1%}")
            st.progress(away_pct / 100)

        with col3:
            # Sparkline
            prob_history = data.get('probability_history', [])
            if len(prob_history) >= 3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=prob_history,
                    mode='lines',
                    line=dict(color='#1f77b4', width=2),
                    showlegend=False,
                ))
                fig.add_hline(y=0.5, line_dash="dot", line_color="#8b949e", opacity=0.3)
                fig.update_layout(
                    height=70,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False, range=[0, 1]),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"all_{data.get('event_id', game_key)}")

        st.markdown("---")


def _render_positions_espn(positions: list, league_filter: str, espn_data: dict):
    """Render ESPN data for open positions."""
    # Get open positions
    open_pos = [p for p in positions if p['status'] == 'OPEN']
    if league_filter != "All":
        open_pos = [p for p in open_pos if p['league'] == league_filter]

    if not open_pos:
        st.info("No open positions to track")
        return

    if not espn_data:
        st.warning("ESPN data unavailable")
        return

    # Enrich positions with ESPN data
    enriched_positions = match_positions_with_espn(open_pos, espn_data)

    # Filter to positions with ESPN data
    live_positions = [p for p in enriched_positions if p.get('espn_home_prob') is not None]

    if not live_positions:
        st.info("No live ESPN data for open positions (games may not have started)")
        return

    # Deduplicate by game: multiple positions can match the same ESPN game
    seen_games = set()
    unique_positions = []
    for pos in live_positions:
        game_key = pos['game']
        if game_key not in seen_games:
            seen_games.add(game_key)
            # Count how many positions reference this game
            pos['_position_count'] = sum(1 for p in live_positions if p['game'] == game_key)
            unique_positions.append(pos)

    # Render each unique game with ESPN data
    for pos in unique_positions:
        game = pos['game']
        bet_side = pos.get('bet_side', '')
        league = pos.get('league', '')

        # Determine which probability to show based on bet side
        if bet_side.upper() == 'HOME':
            espn_prob = pos.get('espn_home_prob', 0)
            market_prob = pos.get('current_price', 0)
            team_bet = pos.get('game', '').split(' @ ')[-1] if ' @ ' in pos.get('game', '') else ''
        else:
            espn_prob = pos.get('espn_away_prob', 0)
            market_prob = pos.get('current_price', 0)
            team_bet = pos.get('game', '').split(' @ ')[0] if ' @ ' in pos.get('game', '') else ''

        # Divergence
        divergence = pos.get('espn_vs_market', 0) or 0

        # Create columns for layout
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Game info with live indicator
            live_badge = "🔴 LIVE" if pos.get('espn_is_live') else ("✅ FINAL" if pos.get('espn_is_final') else "⏳")
            league_color = "#1f77b4" if league == "NBA" else "#ff7f0e"

            st.markdown(
                f"<span style='color:{league_color};font-weight:bold;'>[{league}]</span> "
                f"{game} <span style='color:#8b949e;font-size:0.9em;'>{live_badge}</span>",
                unsafe_allow_html=True
            )

            # Score and time
            if pos.get('espn_score'):
                st.markdown(
                    f"<span style='color:#c9d1d9;font-size:1.1em;font-weight:bold;'>{pos['espn_score']}</span> "
                    f"<span style='color:#8b949e;'>{pos.get('espn_time', '')}</span>",
                    unsafe_allow_html=True
                )

        with col2:
            # ESPN vs Market comparison
            bet_label = f"**Your Bet:** {bet_side}"
            if pos.get('_position_count', 1) > 1:
                bet_label += f" <span style='color:#8b949e;font-size:0.8em;'>({pos['_position_count']} positions)</span>"
            st.markdown(bet_label, unsafe_allow_html=True)
            st.markdown(f"ESPN: **{espn_prob:.1%}**")
            st.markdown(f"Market: **{market_prob:.1%}**")

            # Divergence indicator
            if divergence > 0.03:
                div_color = "#00ff41"
                div_text = "ESPN > Market ↑"
            elif divergence < -0.03:
                div_color = "#ff4444"
                div_text = "ESPN < Market ↓"
            else:
                div_color = "#8b949e"
                div_text = "Aligned"

            st.markdown(
                f"<span style='color:{div_color};font-size:0.9em;'>{div_text} ({divergence:+.1%})</span>",
                unsafe_allow_html=True
            )

        with col3:
            # Mini sparkline chart
            prob_history = pos.get('probability_history', [])
            if len(prob_history) >= 3:
                # Create mini sparkline with Plotly
                fig = go.Figure()

                # Determine color based on trend
                if len(prob_history) >= 2:
                    if bet_side.upper() == 'HOME':
                        trend_up = prob_history[-1] > prob_history[0]
                    else:
                        trend_up = prob_history[-1] < prob_history[0]  # For away bet, lower home prob is good
                    line_color = "#00ff41" if trend_up else "#ff4444"
                else:
                    line_color = "#8b949e"

                fig.add_trace(go.Scatter(
                    y=prob_history,
                    mode='lines',
                    line=dict(color=line_color, width=2),
                    fill='tozeroy',
                    fillcolor=f'rgba({",".join(str(int(line_color.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))},0.1)',
                    showlegend=False,
                ))

                # Reference line at 50%
                fig.add_hline(y=0.5, line_dash="dot", line_color="#8b949e", opacity=0.3)

                fig.update_layout(
                    height=80,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False, range=[0, 1]),
                )

                st.plotly_chart(fig, use_container_width=True, key=f"spark_{pos['id']}")
            else:
                st.markdown("<span style='color:#8b949e;'>No chart data</span>", unsafe_allow_html=True)

        st.markdown("---")


def render_injury_panel(positions: list, league_filter: str):
    """Render ESPN injury data for teams with open positions."""
    st.subheader("🏥 Team Injuries (ESPN)")

    # Get teams from open positions
    open_pos = [p for p in positions if p['status'] == 'OPEN']
    if league_filter != "All":
        open_pos = [p for p in open_pos if p['league'] == league_filter]

    if not open_pos:
        st.info("No open positions - add positions to see injury data")
        return

    # Extract unique teams
    teams = set()
    for pos in open_pos:
        game = pos.get('game', '')
        if ' @ ' in game:
            away, home = game.split(' @ ')
            teams.add(away)
            teams.add(home)

    if not teams:
        return

    # Get injuries for NBA and CBB teams
    nba_injuries = get_espn_injuries_for_teams(list(teams), 'nba')
    cbb_injuries = get_espn_injuries_for_teams(list(teams), 'cbb')
    all_injuries = {**nba_injuries, **cbb_injuries}

    if not all_injuries:
        st.success("No injuries reported for teams in your positions")
        return

    # Display injuries by team
    for team, injuries in all_injuries.items():
        if not injuries:
            continue

        with st.expander(f"**{team}** - {len(injuries)} injury report(s)", expanded=False):
            for inj in injuries:
                status = inj.get('status', 'Unknown').upper()
                injury_desc = f"{inj.get('side', '')} {inj.get('injury_type', 'Unknown')}".strip()
                detail = inj.get('detail', '')
                return_date = inj.get('return_date', 'TBD')
                news = inj.get('news', '')

                # Status color
                if status == 'OUT':
                    status_color = "#ff4444"
                elif status in ['DOUBTFUL', 'QUESTIONABLE']:
                    status_color = "#f0ad4e"
                else:
                    status_color = "#8b949e"

                st.markdown(
                    f"<span style='color:{status_color};font-weight:bold;'>{status}</span> - "
                    f"{injury_desc} ({detail})",
                    unsafe_allow_html=True
                )
                st.markdown(f"📅 Expected Return: **{return_date}**")
                if news:
                    st.caption(news[:150] + "..." if len(news) > 150 else news)
                st.markdown("---")


def render_manual_close_panel(positions: list, league_filter: str):
    """Render manual position close controls."""
    st.subheader("⚙️ Manual Position Management")

    # Get open positions
    open_pos = [p for p in positions if p['status'] == 'OPEN']
    if league_filter != "All":
        open_pos = [p for p in open_pos if p['league'] == league_filter]

    if not open_pos:
        st.info("No open positions to manage")
        return

    # Create selectbox for position selection
    position_options = {
        f"{p['league']} | {p['game']} | {p['bet_side']} ({p['size_pct']:.1f}%)": p
        for p in open_pos
    }

    selected_label = st.selectbox(
        "Select Position to Close",
        options=list(position_options.keys()),
        key="close_position_select"
    )

    if selected_label:
        selected_pos = position_options[selected_label]

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            # Show position details
            st.markdown(f"**Game:** {selected_pos['game']}")
            st.markdown(f"**Side:** {selected_pos['bet_side']}")
            st.markdown(f"**Size:** {selected_pos['size_pct']:.1f}%")
            st.markdown(f"**Current P&L:** ${selected_pos['pnl']:+.2f}")

        with col2:
            # Close percentage slider
            close_pct = st.slider(
                "Close Percentage",
                min_value=10,
                max_value=100,
                value=100,
                step=10,
                key="close_pct_slider",
                help="Choose what % of the position to close"
            )

            # Quick buttons
            quick_col1, quick_col2, quick_col3 = st.columns(3)
            with quick_col1:
                if st.button("25%", key="quick_25"):
                    st.session_state.close_pct_slider = 25
            with quick_col2:
                if st.button("50%", key="quick_50"):
                    st.session_state.close_pct_slider = 50
            with quick_col3:
                if st.button("100%", key="quick_100"):
                    st.session_state.close_pct_slider = 100

        with col3:
            # Close reason
            reason = st.selectbox(
                "Reason",
                ["TAKE_PROFIT", "STOP_LOSS", "HEDGE", "REBALANCE", "OTHER"],
                key="close_reason"
            )

            # Calculate preview
            close_amount = selected_pos['amount'] * (close_pct / 100)
            close_pnl = selected_pos['pnl'] * (close_pct / 100)

            st.markdown(f"**Closing:** ${close_amount:.2f}")
            pnl_color = "#00ff41" if close_pnl >= 0 else "#ff4444"
            st.markdown(f"**P&L:** <span style='color:{pnl_color}'>${close_pnl:+.2f}</span>", unsafe_allow_html=True)

            # Close button
            if st.button(f"🔒 Close {close_pct}%", type="primary", key="close_btn"):
                success = close_position_partial(
                    league=selected_pos['league'],
                    position_id=selected_pos['id'],
                    close_pct=close_pct / 100,
                    reason=reason
                )
                if success:
                    st.success(f"Closed {close_pct}% of position!")
                    st.rerun()
                else:
                    st.error("Failed to close position. Check logs.")


def render_bottom_panels():
    """Render bottom panels: activity feed and edge analysis."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recent Activity")

        activity = get_recent_activity(12)

        if activity:
            for item in activity:
                # Format time
                try:
                    dt = datetime.fromisoformat(item['time'].replace('Z', '+00:00'))
                    time_str = dt.astimezone(ET).strftime('%m/%d %H:%M')
                except:
                    time_str = ""

                badge = item['badge']
                if badge == "WIN":
                    badge_html = '<span class="badge-win">WIN</span>'
                elif badge == "LOSS":
                    badge_html = '<span class="badge-loss">LOSS</span>'
                elif badge == "ENTRY":
                    badge_html = '<span class="badge-entry">ENTRY</span>'
                elif badge == "EXIT":
                    badge_html = '<span class="badge-exit">EXIT</span>'
                else:
                    badge_html = f'<span class="badge-exit">{badge}</span>'

                pnl_str = ""
                if item.get('pnl') is not None:
                    pnl = item['pnl']
                    color = "#00ff41" if pnl >= 0 else "#ff4444"
                    pnl_str = f" <span style='color:{color}'>${pnl:+.2f}</span>"

                league_badge = f"<span style='color:#8b949e;'>[{item['league']}]</span>"

                st.markdown(
                    f"{league_badge} {badge_html} {item['game'][:35]}{pnl_str} <span style='color:#8b949e;font-size:0.8em;'>{time_str}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No recent activity")

    with col2:
        st.subheader("Performance by Edge")

        edge_data = get_edge_analysis()

        if edge_data:
            rows = []
            for e in edge_data:
                rows.append({
                    "Edge": e['bucket'],
                    "Bets": e['bets'],
                    "Wins": e['wins'],
                    "Win %": f"{e['win_rate']:.0f}%",
                    "P&L": f"${e['pnl']:+.2f}",
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data for analysis")


def main():
    st.set_page_config(
        page_title="Paper Trading Dashboard",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Inject CSS
    st.markdown(BLOOMBERG_CSS, unsafe_allow_html=True)

    # Auto-refresh watcher
    auto_refresh_fragment()

    # Load data
    summary = get_portfolio_summary()
    positions = get_unified_positions()

    # Ticker bar header
    render_ticker_bar(summary)

    # Portfolio metrics
    render_portfolio_metrics(summary)

    st.divider()

    # Risk status
    render_risk_status()

    st.divider()

    # League filter
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        league_filter = st.selectbox(
            "Filter by League",
            ["All", "NBA", "CBB", "SOCCER"],
            label_visibility="collapsed"
        )
    with col2:
        if st.button("Refresh"):
            st.rerun()

    # Positions grid
    render_positions_grid(positions, league_filter)

    st.divider()

    # ESPN Live Win Probability Panel
    render_espn_live_panel(positions, league_filter)

    st.divider()

    # Injury and Position Management side by side
    col_inj, col_mgmt = st.columns(2)

    with col_inj:
        render_injury_panel(positions, league_filter)

    with col_mgmt:
        render_manual_close_panel(positions, league_filter)

    st.divider()

    # P&L Chart
    render_pnl_chart()

    st.divider()

    # Bottom panels
    render_bottom_panels()

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #8b949e; font-size: 0.8em;'>"
        "NBA | CBB | Soccer Paper Trading Dashboard | Auto-refresh: 5s"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
