"""Fetch 1-minute PM draw prices during match windows for all closed soccer markets."""
import requests, json, time
from datetime import datetime, timezone

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

print(f'Total markets: {len(all_markets)}', flush=True)

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

    # Fetch 1-min candles for the 2nd half window (real min 55-120 from kickoff)
    window_start = start_ts + 55 * 60
    window_end = start_ts + 120 * 60

    try:
        resp = requests.get(f'{CLOB}/prices-history', params={
            'market': mkt['token'],
            'startTs': window_start,
            'endTs': window_end,
            'fidelity': 1,
        }, timeout=20)
    except Exception:
        time.sleep(2)
        continue

    if resp.status_code != 200:
        continue

    history = resp.json().get('history', [])
    if not history:
        continue

    # Extract prices at each game minute (real_min - 15 = game_min for 2nd half)
    prices_by_game_min = {}
    for h in history:
        real_min = (h['t'] - start_ts) / 60
        game_min = round(real_min - 15)  # subtract HT
        if 45 <= game_min <= 100:
            # Keep the last price for each game minute
            prices_by_game_min[game_min] = h['p']

    if not prices_by_game_min or 70 not in prices_by_game_min:
        # Try to find closest to min 70
        closest_70 = None
        for gm in range(68, 73):
            if gm in prices_by_game_min:
                closest_70 = gm
                break
        if closest_70:
            prices_by_game_min[70] = prices_by_game_min[closest_70]
        else:
            continue

    results.append({
        'title': mkt['title'],
        'start': mkt['start'],
        'league': mkt['league'],
        'token': mkt['token'],
        'was_draw': mkt['was_draw'],
        'prices': prices_by_game_min,
    })

    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{len(all_markets)} processed ({len(results)} with data)...', flush=True)
        # Incremental save
        with open('Data/soccer_backtest/pm_1min_prices.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    time.sleep(0.2)

print(f'\nTotal with 1-min price data: {len(results)}', flush=True)

with open('Data/soccer_backtest/pm_1min_prices.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('Saved to Data/soccer_backtest/pm_1min_prices.json', flush=True)
