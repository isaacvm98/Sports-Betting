"""Fetch full PM draw price histories for all closed soccer markets."""
import requests, json, time
from datetime import datetime

CLOB = 'https://clob.polymarket.com'
GAMMA = 'https://gamma-api.polymarket.com'
SERIES = {
    'Premier League': '10188', 'La Liga': '10193',
    'Bundesliga': '10194', 'Serie A': '10203', 'Ligue 1': '10195',
}

# Collect ALL closed draw markets
all_markets = []
for league, sid in SERIES.items():
    offset = 0
    while offset < 500:
        resp = requests.get(f'{GAMMA}/events', params={
            'series_id': sid, 'closed': 'true', 'limit': 50, 'offset': offset,
            'order': 'startTime', 'ascending': 'false',
        }, timeout=15)
        events = resp.json()
        if not events:
            break
        for e in events:
            for m in e.get('markets', []):
                q = m.get('question', '').lower()
                if 'draw' not in q or 'halftime' in q:
                    continue
                tokens = m.get('clobTokenIds', '[]')
                if isinstance(tokens, str):
                    tokens = json.loads(tokens)
                prices = m.get('outcomePrices', '[]')
                if isinstance(prices, str):
                    prices = json.loads(prices)
                was_draw = float(prices[0]) > 0.5 if prices else None
                all_markets.append({
                    'title': e.get('title', ''),
                    'start': e.get('startTime', ''),
                    'league': league,
                    'token': tokens[0] if tokens else None,
                    'was_draw': was_draw,
                })
        oldest = events[-1].get('startTime', '') if events else ''
        if not oldest or oldest < '2025-01-01':
            break
        offset += 50

print(f'Fetching price history for {len(all_markets)} markets...')

results = []
for i, mkt in enumerate(all_markets):
    if not mkt['token']:
        continue
    try:
        start_ts = int(datetime.fromisoformat(
            mkt['start'].replace('Z', '+00:00')
        ).timestamp())
    except Exception:
        continue

    resp = requests.get(f'{CLOB}/prices-history', params={
        'market': mkt['token'], 'interval': 'max', 'fidelity': 1,
    }, timeout=15)

    if resp.status_code != 200:
        continue

    history = resp.json().get('history', [])
    if not history:
        continue

    prices_at = {}
    for game_min in [0, 45, 60, 65, 70, 72, 75, 78, 80, 82, 85, 88, 90]:
        real_ts = start_ts + (game_min + 15) * 60
        best = min(history, key=lambda h: abs(h['t'] - real_ts))
        if abs(best['t'] - real_ts) < 600:
            prices_at[f'min_{game_min}'] = best['p']

    if 'min_70' not in prices_at:
        continue

    results.append({**mkt, **prices_at})

    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{len(all_markets)} processed ({len(results)} with data)...')

    time.sleep(0.15)

print(f'\nTotal with price data: {len(results)}')

with open('Data/soccer_backtest/pm_price_histories.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('Saved to Data/soccer_backtest/pm_price_histories.json')
