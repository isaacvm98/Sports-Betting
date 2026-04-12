"""Quick analysis: Polymarket favorite accuracy vs Model accuracy."""
import pandas as pd
import numpy as np

df = pd.read_csv('Data/backtest/nba_backtest_dataset.csv')
ph = pd.read_csv('Data/backtest/nba_price_history.csv')

has_both = df[df['pm_pregame_home'].notna() & df['home_win'].notna() & df['model_home_prob'].notna()].copy()
print(f"Games with PM prices + model + results: {len(has_both)}")
print()

# Predictions
has_both['pm_predicted_home'] = has_both['pm_pregame_home'] >= 0.5
has_both['pm_correct'] = (
    (has_both['pm_predicted_home'] & (has_both['home_win'] == 1)) |
    (~has_both['pm_predicted_home'] & (has_both['home_win'] == 0))
)
has_both['model_pred_home'] = has_both['model_home_prob'] >= 0.5
has_both['model_correct_bool'] = (
    (has_both['model_pred_home'] & (has_both['home_win'] == 1)) |
    (~has_both['model_pred_home'] & (has_both['home_win'] == 0))
)

print("=== PREGAME ACCURACY: POLYMARKET vs MODEL ===")
print(f"Polymarket favorites win rate: {has_both['pm_correct'].mean():.1%} ({int(has_both['pm_correct'].sum())}/{len(has_both)})")
print(f"Model predictions accuracy:    {has_both['model_correct_bool'].mean():.1%} ({int(has_both['model_correct_bool'].sum())}/{len(has_both)})")
print()

# By PM confidence bucket
has_both['pm_confidence'] = has_both.apply(
    lambda r: max(r['pm_pregame_home'], r['pm_pregame_away']), axis=1
)

bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%', '80%+']
has_both['pm_conf_bucket'] = pd.cut(has_both['pm_confidence'], bins=bins, labels=labels, right=False)

print("--- Polymarket Favorite Win Rate by Confidence ---")
pm_stats = has_both.groupby('pm_conf_bucket', observed=True).agg(
    games=('pm_correct', 'size'),
    pm_wins=('pm_correct', 'sum'),
    pm_accuracy=('pm_correct', 'mean'),
    model_accuracy=('model_correct_bool', 'mean'),
).reset_index()
pm_stats['pm_accuracy'] = pm_stats['pm_accuracy'].map('{:.1%}'.format)
pm_stats['model_accuracy'] = pm_stats['model_accuracy'].map('{:.1%}'.format)
print(pm_stats.to_string(index=False))

# When they DISAGREE
print()
print("=== WHEN MODEL AND POLYMARKET DISAGREE ===")
has_both['agree'] = has_both['pm_predicted_home'] == has_both['model_pred_home']

agree = has_both[has_both['agree']]
disagree = has_both[~has_both['agree']]

print(f"Agree on winner: {len(agree)} games ({len(agree)/len(has_both):.0%})")
print(f"  PM correct: {agree['pm_correct'].mean():.1%}, Model correct: {agree['model_correct_bool'].mean():.1%}")
print(f"Disagree: {len(disagree)} games ({len(disagree)/len(has_both):.0%})")
print(f"  PM correct: {disagree['pm_correct'].mean():.1%}, Model correct: {disagree['model_correct_bool'].mean():.1%}")

print()
print("When they disagree, who was right?")
print(f"  Polymarket was right: {int(disagree['pm_correct'].sum())}/{len(disagree)} ({disagree['pm_correct'].mean():.1%})")
print(f"  Model was right:      {int(disagree['model_correct_bool'].sum())}/{len(disagree)} ({disagree['model_correct_bool'].mean():.1%})")

# Underdog analysis
print()
print("=== UNDERDOG ANALYSIS ===")
has_both['underdog_won'] = (
    (has_both['pm_predicted_home'] & (has_both['home_win'] == 0)) |
    (~has_both['pm_predicted_home'] & (has_both['home_win'] == 1))
)
print(f"PM underdog upset rate: {has_both['underdog_won'].mean():.1%} ({int(has_both['underdog_won'].sum())}/{len(has_both)})")

fav_bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
fav_labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-80%', '80%+']
has_both['fav_bucket'] = pd.cut(has_both['pm_confidence'], bins=fav_bins, labels=fav_labels, right=False)

print()
print("--- Underdog Upset Rate by Favorite Strength ---")
upset_stats = has_both.groupby('fav_bucket', observed=True).agg(
    games=('underdog_won', 'size'),
    upsets=('underdog_won', 'sum'),
    upset_rate=('underdog_won', 'mean'),
).reset_index()
# Midpoint implied upset rate for each bucket
implied = {'50-55%': 0.475, '55-60%': 0.425, '60-65%': 0.375, '65-70%': 0.325, '70-80%': 0.25, '80%+': 0.10}
upset_stats['implied_upset'] = upset_stats['fav_bucket'].map(implied)
upset_stats['actual'] = upset_stats['upset_rate'].map('{:.1%}'.format)
upset_stats['implied'] = upset_stats['implied_upset'].map('{:.1%}'.format)
print(upset_stats[['fav_bucket', 'games', 'upsets', 'actual', 'implied']].to_string(index=False))

# 24h-before vs pregame accuracy
print()
print("=== 24H-BEFORE vs PREGAME ACCURACY ===")

ph_dt = ph.copy()
ph_dt['game_date'] = pd.to_datetime(ph_dt['game_date'])

# Get earliest available price (24h+ before) per game
early = ph_dt[ph_dt['minutes_to_start'] >= 1380].sort_values('minutes_to_start', ascending=False)
early_prices = early.groupby(['game_date', 'home_team', 'away_team']).first().reset_index()

has_both_dt = has_both.copy()
has_both_dt['game_date'] = pd.to_datetime(has_both_dt['game_date'])

merged = has_both_dt.merge(
    early_prices[['game_date', 'home_team', 'away_team', 'home_price']].rename(columns={'home_price': 'price_24h'}),
    on=['game_date', 'home_team', 'away_team'],
    how='inner'
)
print(f"Games with 24h-before price: {len(merged)}")

merged['pm_24h_pred_home'] = merged['price_24h'] >= 0.5
merged['pm_24h_correct'] = (
    (merged['pm_24h_pred_home'] & (merged['home_win'] == 1)) |
    (~merged['pm_24h_pred_home'] & (merged['home_win'] == 0))
)

print(f"24h-before PM accuracy:  {merged['pm_24h_correct'].mean():.1%} ({int(merged['pm_24h_correct'].sum())}/{len(merged)})")
print(f"Pregame PM accuracy:     {merged['pm_correct'].mean():.1%} ({int(merged['pm_correct'].sum())}/{len(merged)})")
print(f"Model accuracy:          {merged['model_correct_bool'].mean():.1%} ({int(merged['model_correct_bool'].sum())}/{len(merged)})")

# Model vs PM: calling upsets
print()
print("=== MODEL vs PM: CALLING UPSETS ===")
model_picks_underdog = disagree.copy()
print(f"Model picks the PM underdog: {len(model_picks_underdog)} times")
print(f"  Model was right (upset happened): {int(model_picks_underdog['model_correct_bool'].sum())} ({model_picks_underdog['model_correct_bool'].mean():.1%})")

# Flip side: when model agrees with PM favorite
model_agrees_fav = agree.copy()
print(f"Model agrees with PM favorite: {len(model_agrees_fav)} times")
print(f"  Both correct: {int(model_agrees_fav['pm_correct'].sum())} ({model_agrees_fav['pm_correct'].mean():.1%})")

# What if we ONLY bet when model disagrees AND has high confidence?
print()
print("=== STRATEGY: BET UNDERDOGS ONLY WHEN MODEL IS CONFIDENT ===")
for min_conf in [0.55, 0.60, 0.65]:
    sub = disagree.copy()
    sub['model_conf'] = sub.apply(
        lambda r: r['model_home_prob'] if r['model_pred_home'] else r['model_away_prob'], axis=1
    )
    sub = sub[sub['model_conf'] >= min_conf]
    if len(sub) == 0:
        continue
    wr = sub['model_correct_bool'].mean()
    # P&L: buy the underdog at PM price
    sub['entry_price'] = sub.apply(
        lambda r: r['pm_pregame_home'] if r['model_pred_home'] else r['pm_pregame_away'], axis=1
    )
    sub['pnl'] = sub.apply(
        lambda r: (1.0 / r['entry_price'] - 1) if r['model_correct_bool'] else -1.0, axis=1
    )
    avg_entry = sub['entry_price'].mean()
    avg_pnl = sub['pnl'].mean()
    print(f"  Model conf >= {min_conf:.0%}: {len(sub)} bets, win rate {wr:.1%}, avg entry {avg_entry:.3f}, avg P&L/bet {avg_pnl:+.3f}")

# What about COMBINING: only bet when model+PM agree on a side with high edge?
print()
print("=== STRATEGY: BET FAVORITES WHEN MODEL AGREES + HAS EDGE ===")
for min_edge in [0.0, 0.03, 0.05]:
    sub = agree.copy()
    sub['model_edge'] = sub.apply(
        lambda r: r['edge_home_pm'] if r['model_pred_home'] else r['edge_away_pm'], axis=1
    )
    sub = sub[sub['model_edge'].notna() & (sub['model_edge'] >= min_edge)]
    if len(sub) == 0:
        continue
    wr = sub['model_correct_bool'].mean()
    sub['entry_price'] = sub.apply(
        lambda r: r['pm_pregame_home'] if r['model_pred_home'] else r['pm_pregame_away'], axis=1
    )
    sub['pnl'] = sub.apply(
        lambda r: (1.0 / r['entry_price'] - 1) if r['model_correct_bool'] else -1.0, axis=1
    )
    avg_pnl = sub['pnl'].mean()
    print(f"  Agree + edge >= {min_edge:.0%}: {len(sub)} bets, win rate {wr:.1%}, avg P&L/bet {avg_pnl:+.3f}")
