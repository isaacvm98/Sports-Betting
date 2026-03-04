"""
CBB Paper Trading Dashboard

Streamlit dashboard to track positions and performance.

Usage:
    streamlit run src/CBB/dashboard.py
"""

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd

# Configuration
DATA_DIR = Path("Data/cbb_paper_trading")
POSITIONS_FILE = DATA_DIR / "positions.json"
BANKROLL_FILE = DATA_DIR / "bankroll.json"
TRADES_LOG = DATA_DIR / "trades.json"
STARTING_BANKROLL = 1000

ET = ZoneInfo("America/New_York")


def load_positions():
    """Load positions from file."""
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def load_bankroll():
    """Load current bankroll."""
    if BANKROLL_FILE.exists():
        with open(BANKROLL_FILE, "r") as f:
            data = json.load(f)
            return data.get("bankroll", STARTING_BANKROLL)
    return STARTING_BANKROLL


def load_trades():
    """Load trades log."""
    if TRADES_LOG.exists():
        with open(TRADES_LOG, "r") as f:
            return json.load(f)
    return []


def get_file_mtime():
    """Get positions file modification time."""
    if POSITIONS_FILE.exists():
        return POSITIONS_FILE.stat().st_mtime
    return 0


@st.fragment(run_every=5)  # Check every 5 seconds
def auto_refresh_fragment():
    """Fragment that checks for file changes and triggers refresh."""
    current_mtime = get_file_mtime()

    if 'last_mtime' not in st.session_state:
        st.session_state.last_mtime = current_mtime
        return

    if current_mtime > st.session_state.last_mtime:
        st.session_state.last_mtime = current_mtime
        st.rerun()


def main():
    st.set_page_config(
        page_title="CBB Paper Trader",
        page_icon="🏀",
        layout="wide"
    )

    st.title("🏀 CBB Paper Trading Dashboard")
    st.caption(f"Last updated: {datetime.now(ET).strftime('%Y-%m-%d %I:%M:%S %p ET')}")

    # Auto-refresh watcher (runs in background)
    auto_refresh_fragment()

    # Manual refresh button
    if st.button("🔄 Refresh Now"):
        st.rerun()

    # Load data
    positions = load_positions()
    bankroll = load_bankroll()
    trades = load_trades()

    # Categorize positions
    open_pos = {k: v for k, v in positions.items() if v.get('status') == 'open'}
    pending_pos = {k: v for k, v in positions.items() if v.get('status') == 'pending_resolution'}
    closed_pos = {k: v for k, v in positions.items() if v.get('status') in ['closed', 'resolved']}

    # --- SUMMARY METRICS ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pnl = bankroll - STARTING_BANKROLL
        st.metric(
            "Bankroll",
            f"${bankroll:.2f}",
            f"{pnl:+.2f}" if pnl != 0 else None,
            delta_color="normal"
        )

    with col2:
        roi = (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL * 100
        st.metric("ROI", f"{roi:+.1f}%")

    with col3:
        st.metric("Open Positions", len(open_pos))

    with col4:
        # Win rate
        finished = [p for p in positions.values() if p.get('pnl') is not None]
        wins = len([p for p in finished if p.get('pnl', 0) > 0])
        losses = len([p for p in finished if p.get('pnl', 0) < 0])
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%" if (wins + losses) > 0 else "N/A", f"{wins}W-{losses}L")

    st.divider()

    # --- OPEN POSITIONS ---
    st.subheader("📊 Open Positions")

    if open_pos:
        rows = []
        for pos_id, pos in open_pos.items():
            game = f"{pos['away_team']} @ {pos['home_team']}"
            game_time = datetime.fromisoformat(pos['game_time']).astimezone(ET)

            # Calculate current P&L if we have current price
            current_price = pos.get('current_price', pos['entry_price'])
            price_change = pos.get('price_change', 0)
            unrealized_pnl = pos['bet_amount'] * price_change

            rows.append({
                "Game": game,
                "Time (ET)": game_time.strftime("%I:%M %p"),
                "Bet": f"{pos['bet_type'].upper()} {pos['bet_side'].upper()}",
                "Entry": f"{pos['entry_price']:.1%}",
                "Current": f"{current_price:.1%}",
                "Change": f"{price_change:+.1%}",
                "Size": f"{pos['bet_size_pct']:.1f}%",
                "Amount": f"${pos['bet_amount']:.2f}",
                "Unrealized": f"${unrealized_pnl:+.2f}",
                "Edge": f"{pos.get('edge', 0):+.1f} pts" if pos['bet_type'] == 'spread' else f"{pos.get('edge', 0):+.1%}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Total exposure
        total_exposure = sum(p['bet_size_pct'] for p in open_pos.values())
        total_amount = sum(p['bet_amount'] for p in open_pos.values())
        st.caption(f"Total exposure: {total_exposure:.1f}% (${total_amount:.2f})")
    else:
        st.info("No open positions")

    # --- PENDING RESOLUTION ---
    if pending_pos:
        st.subheader("⏳ Pending Resolution")

        rows = []
        for pos_id, pos in pending_pos.items():
            game = f"{pos['away_team']} @ {pos['home_team']}"
            rows.append({
                "Game": game,
                "Bet": f"{pos['bet_type'].upper()} {pos['bet_side'].upper()}",
                "Entry": f"{pos['entry_price']:.1%}",
                "Size": f"{pos['bet_size_pct']:.1f}%",
                "Amount": f"${pos['bet_amount']:.2f}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # --- RECENT RESULTS ---
    st.subheader("📈 Recent Results")

    if closed_pos:
        rows = []
        for pos_id, pos in list(closed_pos.items())[-10:]:  # Last 10
            game = f"{pos['away_team']} @ {pos['home_team']}"
            pnl = pos.get('pnl', 0)
            result = "✅ Win" if pnl > 0 else "❌ Loss" if pnl < 0 else "➖ Push"

            rows.append({
                "Game": game,
                "Result": result,
                "P&L": f"${pnl:+.2f}",
                "Entry": f"{pos['entry_price']:.1%}",
                "Exit": f"{pos.get('exit_price', 0):.1%}" if pos.get('exit_price') else "N/A",
                "Reason": pos.get('exit_reason', '')[:30],
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No completed trades yet")

    # --- TRADE LOG ---
    with st.expander("📝 Trade Log"):
        if trades:
            df = pd.DataFrame(trades[-20:])  # Last 20 trades
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades logged")

    # --- STATS BY BET TYPE ---
    st.subheader("📊 Stats by Bet Type")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Spread Bets**")
        spread_pos = [p for p in positions.values() if p.get('bet_type') == 'spread' and p.get('pnl') is not None]
        if spread_pos:
            spread_wins = len([p for p in spread_pos if p.get('pnl', 0) > 0])
            spread_losses = len([p for p in spread_pos if p.get('pnl', 0) < 0])
            spread_pnl = sum(p.get('pnl', 0) for p in spread_pos)
            st.write(f"Record: {spread_wins}W - {spread_losses}L")
            st.write(f"P&L: ${spread_pnl:+.2f}")
        else:
            st.write("No completed spread bets")

    with col2:
        st.write("**Moneyline Bets**")
        ml_pos = [p for p in positions.values() if p.get('bet_type') == 'ml' and p.get('pnl') is not None]
        if ml_pos:
            ml_wins = len([p for p in ml_pos if p.get('pnl', 0) > 0])
            ml_losses = len([p for p in ml_pos if p.get('pnl', 0) < 0])
            ml_pnl = sum(p.get('pnl', 0) for p in ml_pos)
            st.write(f"Record: {ml_wins}W - {ml_losses}L")
            st.write(f"P&L: ${ml_pnl:+.2f}")
        else:
            st.write("No completed ML bets")


if __name__ == "__main__":
    main()
