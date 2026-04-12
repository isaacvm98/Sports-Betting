"""Backtest combined XGBoost + survival model with actual PM prices."""
import json, numpy as np, pandas as pd
from collections import Counter
from src.Soccer.equalizer_model import build_features, get_feature_columns, _get_season, _score_at_minute
from src.Soccer.survival_model import build_survival_data, train_cox, get_survival_curve, COVARIATES
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

with open('Data/soccer_backtest/matches.json', encoding='utf-8') as f:
    matches_raw = json.load(f)['matches']
with open('Data/soccer_backtest/pm_1min_prices.json') as f:
    pm_data = json.load(f)

match_lookup = {str(m['match_id']): m for m in matches_raw}
pm_by_date = {}
for pm in pm_data:
    pm_by_date.setdefault(pm['start'][:10], []).append(pm)

LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

# ---- Train XGBoost (draw target, no momentum) on 24/25 ----
df = build_features(matches_raw)
df_all = df.copy()
df_all['was_draw'] = df_all['match_id'].apply(
    lambda mid: int(match_lookup.get(str(mid), {}).get('was_draw', False))
)
df_all['season'] = df_all['date'].apply(_get_season)

feature_cols = get_feature_columns(include_momentum=False, include_team=True)
feature_cols = [c for c in feature_cols if c != 'momentum']

# Train on ALL entry minutes from 24/25 (more data for the model)
df_s1 = df_all[(df_all['season'] == '2024/2025')].dropna(
    subset=[c for c in feature_cols if c in df_all.columns]
)
X_tr, y_tr = df_s1[feature_cols].values, df_s1['was_draw'].values
n_pos = y_tr.sum()
scale = (len(y_tr) - n_pos) / n_pos if n_pos > 0 else 1.0

xgb_clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
    eval_metric='logloss', random_state=42, verbosity=0,
)
xgb_clf.fit(X_tr, y_tr, verbose=False)
xgb_cal = CalibratedClassifierCV(xgb_clf, method='isotonic', cv=3)
xgb_cal.fit(X_tr, y_tr)
print(f'XGBoost trained on 24/25: {len(X_tr)} rows, {y_tr.sum()} draws')

# ---- Train Cox on 24/25 ----
surv_df = build_survival_data(matches_raw)
surv_s1 = surv_df[surv_df['season'] == '2024/2025']
cox = train_cox(surv_s1)
print(f'Cox trained on 24/25: {len(surv_s1)} obs, C-index={cox.concordance_index_:.3f}')

# ---- OOS test on 25/26 ----
ENTRY_MINUTES = [60, 65, 68, 70, 75]
MIN_EDGE = 0.05

test_matches = [
    m for m in matches_raw
    if m.get('qualifying') and _get_season(m.get('date', '')) == '2025/2026'
]
print(f'Test qualifying matches (25/26): {len(test_matches)}')

# Strategy A: Combined (XGBoost + survival picks best entry minute)
# Strategy B: XGBoost only at fixed min 70
# Strategy C: XGBoost only at fixed min 60

results_combined = []
results_fixed_70 = []
results_fixed_60 = []

for match in test_matches:
    h = match['home_team'].lower()[:8]
    a = match['away_team'].lower()[:8]
    date = match['date']
    goals = match.get('goals', [])
    was_draw = match.get('was_draw', False)

    best_pm = None
    for pm in pm_by_date.get(date, []):
        if h in pm['title'].lower() and a in pm['title'].lower():
            best_pm = pm
            break
    if not best_pm:
        continue
    prices = {int(k): v for k, v in best_pm.get('prices', {}).items()}

    # Evaluate each entry minute
    best_ev = -999
    best_entry = None
    entries_by_min = {}

    for entry_min in ENTRY_MINUTES:
        sh, sa = _score_at_minute(goals, entry_min)
        if abs(sh - sa) != 1:
            continue

        losing = 'home' if sh < sa else 'away'
        losing_xg = match.get('xg_home') if losing == 'home' else match.get('xg_away')
        winning_xg = match.get('xg_away') if losing == 'home' else match.get('xg_home')
        total_xg = (losing_xg or 0) + (winning_xg or 0)
        goals_before = [g for g in goals if g['minute'] <= entry_min]
        losing_goals = sum(1 for g in goals_before if g['team'] == losing)

        feats = {
            'xg_share': (losing_xg or 0) / total_xg if total_xg > 0 else 0.5,
            'xg_diff': (losing_xg or 0) - (winning_xg or 0),
            'total_xg': total_xg,
            'is_home_losing': 1 if losing == 'home' else 0,
            'score_level': sh + sa,
            'total_goals_before': len(goals_before),
            'losing_goals_before': losing_goals,
            'mins_since_last_goal': entry_min - goals_before[-1]['minute'] if goals_before else entry_min,
            'entry_minute': entry_min,
            'losing_goals_per_90': 0,
            'losing_late_pct': 0,
            'opp_conceded_per_90': 0,
        }
        for lg in LEAGUES:
            feats[f"league_{lg.lower().replace(' ', '_')}"] = 1 if match['league'] == lg else 0

        X_pred = np.array([[feats.get(c, 0) for c in feature_cols]])
        draw_prob = float(xgb_cal.predict_proba(X_pred)[0, 1])

        pm_price = prices.get(entry_min) or prices.get(entry_min - 1) or prices.get(entry_min + 1)
        if pm_price is None or pm_price < 0.05 or pm_price > 0.50:
            continue

        edge = draw_prob - pm_price

        # Survival: near-term hazard
        surv_feats = {k: feats.get(k, 0) for k in COVARIATES}
        surv_feats['minutes_remaining'] = 90 - entry_min
        times, cum_prob = get_survival_curve(cox, surv_feats, np.arange(1, 31))
        near_hazard = cum_prob[4] if len(cum_prob) > 4 else 0

        # EV
        ev = draw_prob * (1.0 / pm_price - 1) - (1 - draw_prob)

        entry_data = {
            'entry_min': entry_min,
            'draw_prob': draw_prob,
            'pm_price': pm_price,
            'edge': edge,
            'ev': ev,
            'near_hazard': near_hazard,
            'losing_goals': losing_goals,
            'score_level': sh + sa,
        }
        entries_by_min[entry_min] = entry_data

        if edge >= MIN_EDGE and ev > best_ev:
            best_ev = ev
            best_entry = entry_data

    base = {
        'home': match['home_team'],
        'away': match['away_team'],
        'date': date,
        'league': match['league'],
        'was_draw': was_draw,
    }

    # Strategy A: Combined — pick best EV entry
    if best_entry and best_entry['ev'] > 0:
        results_combined.append({**base, **best_entry})

    # Strategy B: Fixed min 70
    if 70 in entries_by_min:
        e = entries_by_min[70]
        if e['edge'] >= MIN_EDGE and e['ev'] > 0:
            results_fixed_70.append({**base, **e})

    # Strategy C: Fixed min 60
    if 60 in entries_by_min:
        e = entries_by_min[60]
        if e['edge'] >= MIN_EDGE and e['ev'] > 0:
            results_fixed_60.append({**base, **e})


def print_results(name, res):
    print(f'\n--- {name} ---')
    if not res:
        print('  No bets')
        return
    n = len(res)
    wins = sum(1 for r in res if r['was_draw'])
    pnl = sum((1/r['pm_price'] - 1) if r['was_draw'] else -1 for r in res)
    avg_entry = np.mean([r['pm_price'] for r in res])
    avg_edge = np.mean([r['edge'] for r in res])
    print(f'  Bets: {n} | Wins: {wins} ({wins/n*100:.0f}%) | '
          f'P&L: {pnl:+.1f}u | ROI: {pnl/n*100:+.1f}% | '
          f'Avg entry: ${avg_entry:.3f} | Avg edge: {avg_edge:.3f}')

    # Entry min breakdown for combined
    if name == 'Combined (XGBoost + Survival)':
        mins = Counter(r['entry_min'] for r in res)
        print(f'  Entry minutes: {dict(sorted(mins.items()))}')
        for m in sorted(mins):
            sub = [r for r in res if r['entry_min'] == m]
            w = sum(1 for r in sub if r['was_draw'])
            p = sum((1/r['pm_price']-1) if r['was_draw'] else -1 for r in sub)
            ae = np.mean([r['pm_price'] for r in sub])
            print(f'    min {m}: {len(sub)} bets, {w}W, PnL {p:+.1f}u, avg entry ${ae:.3f}')


print(f'\n{"="*80}')
print('OOS BACKTEST: XGBoost + Survival vs XGBoost-only (train 24/25, test 25/26)')
print(f'{"="*80}')

print_results('Combined (XGBoost + Survival)', results_combined)
print_results('XGBoost only @ min 70', results_fixed_70)
print_results('XGBoost only @ min 60', results_fixed_60)

# Detail for combined
if results_combined:
    print(f'\n--- Combined detail ---')
    print(f'{"Match":<38s} {"Min":>3s} {"P(d)":>5s} {"PM":>6s} {"Edge":>6s} {"LG":>3s} {"Hz5":>5s} {"Result":>6s} {"PnL":>7s}')
    print('-' * 85)
    for r in sorted(results_combined, key=lambda x: x['date']):
        res = 'DRAW' if r['was_draw'] else ''
        pnl = (1/r['pm_price'] - 1) if r['was_draw'] else -1
        print(f'{r["home"][:17]:17s} v {r["away"][:17]:17s} '
              f'{r["entry_min"]:>3d} {r["draw_prob"]:>5.3f} {r["pm_price"]:>6.3f} '
              f'{r["edge"]:>+6.3f} {r["losing_goals"]:>3d} {r["near_hazard"]:>5.3f} '
              f'{res:>6s} {pnl:>+7.2f}')
