"""Backtest final draw and leading team wins models with actual PM prices."""
import json, numpy as np
from src.Soccer.equalizer_model import build_features, get_feature_columns, _get_season
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import xgboost as xgb

# Load data
with open('Data/soccer_backtest/matches.json', encoding='utf-8') as f:
    matches_raw = json.load(f)['matches']
with open('Data/soccer_backtest/pm_1min_prices.json') as f:
    pm_data = json.load(f)

df = build_features(matches_raw)
df_70 = df[df['entry_minute'] == 70].copy()

match_lookup = {str(m['match_id']): m for m in matches_raw}

# Add targets
df_70['was_draw'] = df_70['match_id'].apply(
    lambda mid: int(match_lookup.get(str(mid), {}).get('was_draw', False))
)

df_70['leader_won'] = 0
for idx, row in df_70.iterrows():
    m = match_lookup.get(str(row['match_id']), {})
    losing = m.get('losing_team_at_70')
    if not losing or m.get('was_draw'):
        continue
    if losing == 'home' and m.get('home_score', 0) > m.get('away_score', 0):
        continue
    if losing == 'away' and m.get('away_score', 0) > m.get('home_score', 0):
        continue
    df_70.at[idx, 'leader_won'] = 1

df_70['season'] = df_70['date'].apply(_get_season)

feature_cols = get_feature_columns(include_momentum=True, include_team=True)
df_valid = df_70.dropna(subset=['momentum']).copy()
df_valid = df_valid.dropna(subset=[c for c in feature_cols if c in df_valid.columns])

seasons = sorted(df_valid['season'].unique())
s1, s2 = seasons[0], seasons[-1]
df_s1 = df_valid[df_valid['season'] == s1]
df_s2 = df_valid[df_valid['season'] == s2]

print(f'Data: {len(df_valid)} rows, seasons {s1} and {s2}')
print(f'  Draw rate: {df_valid["was_draw"].mean():.1%}')
print(f'  Leader wins rate: {df_valid["leader_won"].mean():.1%}')


def train_and_eval(target_col, target_name):
    print(f'\n{"="*75}')
    print(f'{target_name.upper()} MODEL — SEASON VALIDATION')
    print(f'{"="*75}')

    folds = [
        (f'Train {s1} -> Test {s2} (forward)', df_s1, df_s2),
        (f'Train {s2} -> Test {s1} (retrodiction)', df_s2, df_s1),
    ]

    aucs, briers = [], []
    for fold_name, train, test in folds:
        X_tr = train[feature_cols].values
        y_tr = train[target_col].values
        X_te = test[feature_cols].values
        y_te = test[target_col].values

        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        scale = n_neg / n_pos if n_pos > 0 else 1.0

        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale, eval_metric='logloss',
            random_state=42, verbosity=0,
        )
        clf.fit(X_tr, y_tr, verbose=False)
        cal = CalibratedClassifierCV(clf, method='isotonic', cv=3)
        cal.fit(X_tr, y_tr)
        probs = cal.predict_proba(X_te)[:, 1]

        if len(np.unique(y_te)) > 1:
            auc = roc_auc_score(y_te, probs)
            brier = brier_score_loss(y_te, probs)
            ll = log_loss(y_te, probs)
            aucs.append(auc)
            briers.append(brier)
        else:
            auc = brier = ll = None

        print(f'  {fold_name}')
        print(f'    Train: {len(y_tr)} ({y_tr.mean():.1%} pos) | Test: {len(y_te)} ({y_te.mean():.1%} pos)')
        if auc is not None:
            print(f'    AUC: {auc:.3f}  Brier: {brier:.3f}  LogLoss: {ll:.3f}')

    if aucs:
        print(f'\n  LOSO Average: AUC={np.mean(aucs):.3f}  Brier={np.mean(briers):.3f}')

    # Train final on all data
    X_all = df_valid[feature_cols].values
    y_all = df_valid[target_col].values
    n_pos = y_all.sum()
    scale = (len(y_all) - n_pos) / n_pos if n_pos > 0 else 1.0

    clf_final = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, eval_metric='logloss',
        random_state=42, verbosity=0,
    )
    clf_final.fit(X_all, y_all, verbose=False)
    cal_final = CalibratedClassifierCV(clf_final, method='isotonic', cv=3)
    cal_final.fit(X_all, y_all)

    imps = sorted(zip(feature_cols, clf_final.feature_importances_), key=lambda x: -x[1])
    print(f'\n  Top features:')
    for f, imp in imps[:8]:
        print(f'    {f:30s} {imp:.3f} {"#" * int(imp * 50)}')

    return cal_final


draw_model = train_and_eval('was_draw', 'Final Draw')
leader_model = train_and_eval('leader_won', 'Leading Team Wins')

# ---- P&L BACKTEST WITH PM PRICES ----
print(f'\n{"="*75}')
print('P&L BACKTEST — HOLD TO EXPIRY (actual PM prices)')
print(f'{"="*75}')

pm_by_date = {}
for pm in pm_data:
    pm_by_date.setdefault(pm['start'][:10], []).append(pm)

X_all = df_valid[feature_cols].values
df_valid = df_valid.copy()
df_valid['draw_prob'] = draw_model.predict_proba(X_all)[:, 1]
df_valid['leader_prob'] = leader_model.predict_proba(X_all)[:, 1]

matched = []
for _, row in df_valid.iterrows():
    date = row['date']
    h = str(row.get('home_team', '')).lower()[:8]
    a = str(row.get('away_team', '')).lower()[:8]

    best_pm = None
    for pm in pm_by_date.get(date, []):
        if h in pm['title'].lower() and a in pm['title'].lower():
            best_pm = pm
            break
    if not best_pm:
        continue

    prices = {int(k): v for k, v in best_pm.get('prices', {}).items()}
    pm70 = prices.get(70) or prices.get(69) or prices.get(71)
    if pm70 is None:
        continue

    matched.append({
        'home': row.get('home_team', ''),
        'away': row.get('away_team', ''),
        'date': date,
        'was_draw': int(row['was_draw']),
        'leader_won': int(row['leader_won']),
        'draw_prob': row['draw_prob'],
        'leader_prob': row['leader_prob'],
        'pm_draw_70': pm70,
    })

print(f'Matched with PM: {len(matched)}')

# DRAW: buy draw token at pm_draw_70, pays $1 if draw
print(f'\n--- FINAL DRAW: buy draw @ min 70, hold to expiry ---')
print(f'Base rate: {sum(m["was_draw"] for m in matched)}/{len(matched)} '
      f'({sum(m["was_draw"] for m in matched)/len(matched)*100:.1f}%)')
print(f'{"Edge thresh":<15s} {"Bets":>5s} {"Wins":>5s} {"WR":>6s} {"P&L":>9s} {"ROI":>8s} {"Avg entry":>10s}')
print('-' * 62)

for min_edge in [0.0, 0.03, 0.05, 0.07, 0.10]:
    bets = [m for m in matched
            if (m['draw_prob'] - m['pm_draw_70']) > min_edge
            and 0.05 < m['pm_draw_70'] < 0.50]
    if not bets:
        print(f'  >{min_edge:.0%}          0')
        continue
    wins = sum(1 for b in bets if b['was_draw'])
    # $1 per bet: buy shares at pm_draw_70, get $1 if draw, $0 if not
    pnl = sum((1.0 / b['pm_draw_70'] - 1) if b['was_draw'] else -1.0 for b in bets)
    n = len(bets)
    avg_entry = np.mean([b['pm_draw_70'] for b in bets])
    print(f'  >{min_edge:.0%}          {n:5d} {wins:5d} {wins/n*100:5.0f}% ${pnl:+8.2f} {pnl/n*100:+7.1f}% ${avg_entry:9.3f}')

# LEADER: buy leader token at estimated price
# We don't have the actual leader token price from PM, so estimate from draw price
# At min 70 with 1-goal lead: leader price typically = 1 - draw - small_underdog
# We'll try a few assumptions for underdog price
print(f'\n--- LEADING TEAM WINS: buy leader @ min 70, hold to expiry ---')
print(f'Base rate: {sum(m["leader_won"] for m in matched)}/{len(matched)} '
      f'({sum(m["leader_won"] for m in matched)/len(matched)*100:.1f}%)')

# Estimate: underdog (comeback) probability ~ 8% based on our data
# So leader_price ~ 1 - draw_price - 0.08
# But Polymarket has vig, so these don't sum to exactly 1
UNDERDOG_EST = 0.08

print(f'(Estimating leader price = 1 - draw_price - {UNDERDOG_EST} underdog)')
print(f'{"Edge thresh":<15s} {"Bets":>5s} {"Wins":>5s} {"WR":>6s} {"P&L":>9s} {"ROI":>8s} {"Avg entry":>10s}')
print('-' * 62)

for min_edge in [0.0, 0.03, 0.05, 0.07, 0.10]:
    bets = []
    for m in matched:
        leader_price = 1.0 - m['pm_draw_70'] - UNDERDOG_EST
        leader_price = max(0.10, min(0.95, leader_price))
        edge = m['leader_prob'] - leader_price
        if edge > min_edge and 0.30 < leader_price < 0.95:
            bets.append({**m, 'leader_price': leader_price})
    if not bets:
        print(f'  >{min_edge:.0%}          0')
        continue
    wins = sum(1 for b in bets if b['leader_won'])
    pnl = sum((1.0 / b['leader_price'] - 1) if b['leader_won'] else -1.0 for b in bets)
    n = len(bets)
    avg_entry = np.mean([b['leader_price'] for b in bets])
    print(f'  >{min_edge:.0%}          {n:5d} {wins:5d} {wins/n*100:5.0f}% ${pnl:+8.2f} {pnl/n*100:+7.1f}% ${avg_entry:9.3f}')

# Show some example bets for best draw strategy
print(f'\n--- DRAW MODEL: sample bets (edge > 5%) ---')
bets = [m for m in matched
        if (m['draw_prob'] - m['pm_draw_70']) > 0.05
        and 0.05 < m['pm_draw_70'] < 0.50]
bets.sort(key=lambda b: -(b['draw_prob'] - b['pm_draw_70']))
print(f'{"Match":<38s} {"Model":>6s} {"PM":>6s} {"Edge":>6s} {"Result":>7s}')
for b in bets[:20]:
    edge = b['draw_prob'] - b['pm_draw_70']
    res = 'DRAW' if b['was_draw'] else ''
    print(f'{b["home"][:17]:17s} v {b["away"][:17]:17s} {b["draw_prob"]:>6.3f} {b["pm_draw_70"]:>6.3f} {edge:>+6.3f} {res:>7s}')
