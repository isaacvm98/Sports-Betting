"""Analyze ESPN WP as a Bayesian proxy for live exit signals."""
import pandas as pd
import numpy as np

df = pd.read_csv('Data/backtest/nba_backtest_dataset.csv')
espn = pd.read_csv('Data/backtest/espn_wp_backtest.csv')

# Merge ESPN data into main dataset
merged = df.merge(
    espn[['game_date', 'home_team', 'away_team',
          'espn_pregame_home', 'espn_wp_min', 'espn_wp_max', 'espn_lead_changes',
          'espn_q1_end_wp', 'espn_q1_end_home_score', 'espn_q1_end_away_score',
          'espn_q2_end_wp', 'espn_q2_end_home_score', 'espn_q2_end_away_score',
          'espn_q3_end_wp', 'espn_q3_end_home_score', 'espn_q3_end_away_score',
          'espn_q4_end_wp']],
    on=['game_date', 'home_team', 'away_team'],
    how='left',
)

has_all = merged[
    merged['espn_pregame_home'].notna()
    & merged['model_home_prob'].notna()
    & merged['home_win'].notna()
].copy()

print(f"Games with ESPN + Model + Result: {len(has_all)}")

# =====================================================
# THREE-WAY ACCURACY COMPARISON
# =====================================================
print("\n" + "=" * 60)
print("THREE-WAY PREDICTOR COMPARISON")
print("=" * 60)

has_all['espn_correct'] = (
    ((has_all['espn_pregame_home'] >= 0.5) & (has_all['home_win'] == 1)) |
    ((has_all['espn_pregame_home'] < 0.5) & (has_all['home_win'] == 0))
)
has_all['model_correct'] = (
    ((has_all['model_home_prob'] >= 0.5) & (has_all['home_win'] == 1)) |
    ((has_all['model_home_prob'] < 0.5) & (has_all['home_win'] == 0))
)

print(f"\nESPN pregame:  {has_all['espn_correct'].mean():.1%} ({int(has_all['espn_correct'].sum())}/{len(has_all)})")
print(f"XGBoost model: {has_all['model_correct'].mean():.1%} ({int(has_all['model_correct'].sum())}/{len(has_all)})")

has_pm = has_all[has_all['pm_pregame_home'].notna()].copy()
if len(has_pm) > 0:
    has_pm['pm_correct'] = (
        ((has_pm['pm_pregame_home'] >= 0.5) & (has_pm['home_win'] == 1)) |
        ((has_pm['pm_pregame_home'] < 0.5) & (has_pm['home_win'] == 0))
    )
    print(f"Polymarket:    {has_pm['pm_correct'].mean():.1%} ({int(has_pm['pm_correct'].sum())}/{len(has_pm)})")

# By confidence bucket
print("\n--- ESPN Accuracy by Confidence ---")
has_all['espn_conf'] = has_all['espn_pregame_home'].apply(lambda x: max(x, 1-x))
bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0]
labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75%+']
has_all['espn_conf_bucket'] = pd.cut(has_all['espn_conf'], bins=bins, labels=labels, right=False)

stats = has_all.groupby('espn_conf_bucket', observed=True).agg(
    games=('espn_correct', 'size'),
    espn_acc=('espn_correct', 'mean'),
    model_acc=('model_correct', 'mean'),
)
stats['espn_acc'] = stats['espn_acc'].map('{:.1%}'.format)
stats['model_acc'] = stats['model_acc'].map('{:.1%}'.format)
print(stats.to_string())

# =====================================================
# CONSENSUS ANALYSIS
# =====================================================
print("\n" + "=" * 60)
print("CONSENSUS: WHEN DO ALL THREE AGREE?")
print("=" * 60)

has_all['espn_home'] = has_all['espn_pregame_home'] >= 0.5
has_all['model_home'] = has_all['model_home_prob'] >= 0.5

# ESPN vs Model agreement
has_all['espn_model_agree'] = has_all['espn_home'] == has_all['model_home']
agree_em = has_all[has_all['espn_model_agree']]
disagree_em = has_all[~has_all['espn_model_agree']]

print(f"\nESPN + Model agree: {len(agree_em)} games ({len(agree_em)/len(has_all):.0%})")
print(f"  Accuracy when agree: {agree_em['model_correct'].mean():.1%}")
print(f"  ESPN accuracy when agree: {agree_em['espn_correct'].mean():.1%}")
print(f"ESPN + Model disagree: {len(disagree_em)} games ({len(disagree_em)/len(has_all):.0%})")
print(f"  Model accuracy: {disagree_em['model_correct'].mean():.1%}")
print(f"  ESPN accuracy: {disagree_em['espn_correct'].mean():.1%}")

# Three-way on PM subset
if len(has_pm) > 0:
    has_pm['espn_home'] = has_pm['espn_pregame_home'] >= 0.5
    has_pm['model_home'] = has_pm['model_home_prob'] >= 0.5
    has_pm['pm_home'] = has_pm['pm_pregame_home'] >= 0.5

    has_pm['all_agree'] = (has_pm['espn_home'] == has_pm['model_home']) & (has_pm['model_home'] == has_pm['pm_home'])
    all_agree = has_pm[has_pm['all_agree']]
    not_all = has_pm[~has_pm['all_agree']]

    print(f"\nAll three agree: {len(all_agree)} games ({len(all_agree)/len(has_pm):.0%})")
    print(f"  Accuracy: {all_agree['model_correct'].mean():.1%}")
    print(f"Not all agree: {len(not_all)} games ({len(not_all)/len(has_pm):.0%})")
    print(f"  Model accuracy: {not_all['model_correct'].mean():.1%}")

    # Model + PM agree but ESPN disagrees (model finds edge ESPN misses)
    model_pm_agree_espn_not = has_pm[
        (has_pm['model_home'] == has_pm['pm_home']) & (has_pm['model_home'] != has_pm['espn_home'])
    ]
    if len(model_pm_agree_espn_not) > 0:
        print(f"\nModel+PM agree, ESPN disagrees: {len(model_pm_agree_espn_not)} games")
        print(f"  Model correct: {model_pm_agree_espn_not['model_correct'].mean():.1%}")
        print(f"  ESPN correct: {model_pm_agree_espn_not['espn_correct'].mean():.1%}")

    # Model disagrees with both ESPN+PM (contrarian model picks)
    model_contrarian = has_pm[
        (has_pm['model_home'] != has_pm['pm_home']) & (has_pm['model_home'] != has_pm['espn_home'])
    ]
    if len(model_contrarian) > 0:
        print(f"\nModel disagrees with BOTH ESPN+PM: {len(model_contrarian)} games")
        print(f"  Model correct: {model_contrarian['model_correct'].mean():.1%}")
        print(f"  ESPN/PM correct: {model_contrarian['espn_correct'].mean():.1%}")

# =====================================================
# ESPN WP AS LIVE EXIT SIGNAL (Bayesian proxy)
# =====================================================
print("\n" + "=" * 60)
print("ESPN WP AS LIVE EXIT SIGNAL (BAYESIAN PROXY)")
print("=" * 60)

# Focus on games where Kelly recommends a bet
bets = has_all[has_all['bet_side'] != 'none'].copy()
bets = bets[bets['pm_pregame_home'].notna()].copy()
print(f"\nKelly-recommended bets with ESPN data: {len(bets)}")

bets['bet_won'] = (
    ((bets['bet_side'] == 'home') & (bets['home_win'] == 1)) |
    ((bets['bet_side'] == 'away') & (bets['home_win'] == 0))
)
bets['entry_price'] = bets.apply(
    lambda r: r['pm_pregame_home'] if r['bet_side'] == 'home' else r['pm_pregame_away'], axis=1
)

# ESPN WP at each quarter for the bet side
bets['espn_q1_bet_wp'] = bets.apply(
    lambda r: r['espn_q1_end_wp'] if r['bet_side'] == 'home' else (1 - r['espn_q1_end_wp']) if pd.notna(r['espn_q1_end_wp']) else None, axis=1
)
bets['espn_q2_bet_wp'] = bets.apply(
    lambda r: r['espn_q2_end_wp'] if r['bet_side'] == 'home' else (1 - r['espn_q2_end_wp']) if pd.notna(r['espn_q2_end_wp']) else None, axis=1
)
bets['espn_q3_bet_wp'] = bets.apply(
    lambda r: r['espn_q3_end_wp'] if r['bet_side'] == 'home' else (1 - r['espn_q3_end_wp']) if pd.notna(r['espn_q3_end_wp']) else None, axis=1
)

# Score differential at each quarter
bets['q1_diff'] = bets.apply(
    lambda r: (r['espn_q1_end_home_score'] - r['espn_q1_end_away_score']) * (1 if r['bet_side'] == 'home' else -1)
    if pd.notna(r.get('espn_q1_end_home_score')) else None, axis=1
)
bets['q2_diff'] = bets.apply(
    lambda r: (r['espn_q2_end_home_score'] - r['espn_q2_end_away_score']) * (1 if r['bet_side'] == 'home' else -1)
    if pd.notna(r.get('espn_q2_end_home_score')) else None, axis=1
)
bets['q3_diff'] = bets.apply(
    lambda r: (r['espn_q3_end_home_score'] - r['espn_q3_end_away_score']) * (1 if r['bet_side'] == 'home' else -1)
    if pd.notna(r.get('espn_q3_end_home_score')) else None, axis=1
)

has_q = bets[bets['espn_q1_bet_wp'].notna()].copy()
print(f"Bets with quarter-level ESPN WP: {len(has_q)}")
print(f"Overall win rate: {has_q['bet_won'].mean():.1%}")

# Strategy: EXIT at halftime if ESPN WP drops below threshold
print("\n--- EXIT AT HALFTIME IF ESPN WP < THRESHOLD ---")
for thresh in [0.30, 0.25, 0.20, 0.15, 0.10]:
    sub = has_q.copy()
    sub['exit_at_half'] = sub['espn_q2_bet_wp'] < thresh
    stayed = sub[~sub['exit_at_half']]
    exited = sub[sub['exit_at_half']]

    if len(exited) == 0 or len(stayed) == 0:
        continue

    # Exited bets: lose entry price (we sell at ~ESPN WP which is low)
    # Approximation: sell at halftime ESPN WP price
    exited_pnl = exited.apply(
        lambda r: (r['espn_q2_bet_wp'] / r['entry_price'] - 1) if r['entry_price'] > 0 else -1, axis=1
    )

    # Stayed bets: hold to resolution
    stayed_pnl = stayed.apply(
        lambda r: (1.0 / r['entry_price'] - 1) if r['bet_won'] else -1.0, axis=1
    )

    total_pnl = (exited_pnl.sum() + stayed_pnl.sum()) / len(sub)
    stayed_wr = stayed['bet_won'].mean()
    exited_recovery = exited['bet_won'].mean()  # What % of exited bets would've won

    print(
        f"  WP < {thresh:.0%}: exit {len(exited)}/{len(sub)} bets, "
        f"stayed WR {stayed_wr:.1%}, "
        f"exited would've won {exited_recovery:.1%}, "
        f"avg P&L/bet {total_pnl:+.3f}"
    )

# Strategy: EXIT underdogs at Q1 if leading (lock in variance gains)
print("\n--- UNDERDOG EXIT: SELL AT Q1 END IF LEADING ---")
underdogs = has_q[has_q['entry_price'] < 0.45].copy()
print(f"Underdog bets (entry < 0.45): {len(underdogs)}")

if len(underdogs) > 0:
    underdogs['leading_q1'] = underdogs['q1_diff'] > 0
    leading_q1 = underdogs[underdogs['leading_q1']]
    trailing_q1 = underdogs[~underdogs['leading_q1']]

    if len(leading_q1) > 0:
        # Sell at Q1 ESPN WP (which is elevated because underdog is leading)
        leading_q1_pnl = leading_q1.apply(
            lambda r: (r['espn_q1_bet_wp'] / r['entry_price'] - 1), axis=1
        )
        hold_pnl = leading_q1.apply(
            lambda r: (1.0 / r['entry_price'] - 1) if r['bet_won'] else -1.0, axis=1
        )
        print(f"  Underdogs leading after Q1: {len(leading_q1)}/{len(underdogs)}")
        print(f"    Sell at Q1 ESPN WP: avg P&L {leading_q1_pnl.mean():+.3f}")
        print(f"    Hold to resolution: avg P&L {hold_pnl.mean():+.3f} (WR {leading_q1['bet_won'].mean():.1%})")

    if len(trailing_q1) > 0:
        trailing_pnl = trailing_q1.apply(
            lambda r: (1.0 / r['entry_price'] - 1) if r['bet_won'] else -1.0, axis=1
        )
        print(f"  Underdogs trailing after Q1: {len(trailing_q1)}/{len(underdogs)}")
        print(f"    Hold to resolution: avg P&L {trailing_pnl.mean():+.3f} (WR {trailing_q1['bet_won'].mean():.1%})")

# Strategy: COMBINED - exit at halftime if losing badly, take profit if up big
print("\n--- COMBINED: STOP-LOSS + TAKE-PROFIT AT HALFTIME ---")
for stop_thresh, tp_thresh in [(0.20, 0.70), (0.25, 0.65), (0.15, 0.75)]:
    sub = has_q.copy()
    sub['action'] = sub.apply(
        lambda r: 'stop' if r['espn_q2_bet_wp'] < stop_thresh
        else ('take_profit' if r['espn_q2_bet_wp'] > tp_thresh else 'hold'),
        axis=1,
    )

    stopped = sub[sub['action'] == 'stop']
    tp = sub[sub['action'] == 'take_profit']
    held = sub[sub['action'] == 'hold']

    # P&L for each group
    stopped_pnl = stopped.apply(
        lambda r: (r['espn_q2_bet_wp'] / r['entry_price'] - 1), axis=1
    ) if len(stopped) > 0 else pd.Series(dtype=float)

    tp_pnl = tp.apply(
        lambda r: (r['espn_q2_bet_wp'] / r['entry_price'] - 1), axis=1
    ) if len(tp) > 0 else pd.Series(dtype=float)

    held_pnl = held.apply(
        lambda r: (1.0 / r['entry_price'] - 1) if r['bet_won'] else -1.0, axis=1
    ) if len(held) > 0 else pd.Series(dtype=float)

    total = pd.concat([stopped_pnl, tp_pnl, held_pnl])
    avg_pnl = total.mean() if len(total) > 0 else 0

    # Compare to hold-all
    hold_all_pnl = sub.apply(
        lambda r: (1.0 / r['entry_price'] - 1) if r['bet_won'] else -1.0, axis=1
    ).mean()

    print(
        f"  SL<{stop_thresh:.0%} TP>{tp_thresh:.0%}: "
        f"stop {len(stopped)}, TP {len(tp)}, hold {len(held)} | "
        f"avg P&L {avg_pnl:+.3f} vs hold-all {hold_all_pnl:+.3f} "
        f"({'BETTER' if avg_pnl > hold_all_pnl else 'WORSE'})"
    )

# WP volatility as a game characteristic
print("\n--- GAME VOLATILITY (ESPN WP RANGE) ---")
has_q['wp_range'] = has_q['espn_wp_max'] - has_q['espn_wp_min']
volatile = has_q[has_q['wp_range'] > 0.6]
calm = has_q[has_q['wp_range'] <= 0.4]
mid = has_q[(has_q['wp_range'] > 0.4) & (has_q['wp_range'] <= 0.6)]

print(f"Low volatility (range<=0.4): {len(calm)} games, bet WR {calm['bet_won'].mean():.1%}")
print(f"Medium volatility (0.4-0.6): {len(mid)} games, bet WR {mid['bet_won'].mean():.1%}")
print(f"High volatility (range>0.6): {len(volatile)} games, bet WR {volatile['bet_won'].mean():.1%}")
