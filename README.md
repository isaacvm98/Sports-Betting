# Sports Betting & Prediction Markets Research

End-to-end research pipeline for trading moneyline and draw markets on Polymarket using machine-learning models. Spans NBA, college basketball, and soccer (Premier League + other top-five leagues).

The project combines:
- Pre-trained XGBoost probability models for game outcomes
- Live data providers (Polymarket Gamma API, NBA Stats, ESPN, FotMob, SBR)
- Paper-trading engines that simulate execution with real-time prices
- A rich backtest + analysis suite for strategy design and validation
- A Streamlit dashboard for monitoring

---

## Current status (2026-04-17)

| System | State | Notes |
|--------|-------|-------|
| **NBA V2 (dual-leg)** | **Paused 2026-04-05** | Final bankroll $876.81 / $1000. Pause based on late-season regime shift — see `Data/backtest/analysis/retro_late_season_report.md` |
| **Soccer draws (XGBoost + Cox survival)** | Running | +$32 / $1000, Premier League + other top-5 leagues; bets live in-game via FotMob |
| CBB paper trader | Built, not running live | Infrastructure exists for CBB regular season & March Madness |
| NBA V1 (legacy Kelly) | Deprecated | Bankroll wiped; not a running system |

See [`NEXT_SEASON_CHECKLIST.md`](NEXT_SEASON_CHECKLIST.md) for the restart plan for Oct 2026.

---

## Project layout

```
├── main.py                        NBA predictions entry point (CLI)
├── NEXT_SEASON_CHECKLIST.md       2026-27 restart checklist
├── README.md                      (this file)
├── config.toml                    season date ranges
│
├── src/
│   ├── DataProviders/             PM Gamma, ESPN, FotMob, NBA, SBR, CLOB price history
│   ├── Process-Data/              Data ingestion & feature engineering
│   ├── Train-Models/              XGBoost + NN training scripts
│   ├── Predict/                   Model inference runners
│   ├── Utils/                     Kelly, EV, backtester, alerts, drawdown
│   ├── Dashboard/                 Streamlit UI
│   ├── Polymarket/                NBA paper traders (V1, V2)
│   ├── CBB/                       College basketball module
│   └── Soccer/                    Soccer draw + momentum systems
│
├── Models/                        Trained models (XGBoost + NN, ML + O/U)
├── Data/                          All state: DBs, schedules, paper-trading logs
└── Tests/                         Unit tests
```

---

## NBA: V2 dual-leg strategy (main live system)

**Entry filters:**
- Model probability ≥ 60% OR entry price ≥ $0.30 (underdog floor)
- Model edge ≥ 7% over Polymarket implied probability

**Two legs:**
1. **FAV (conf ≥ 60%):** hold to binary resolution — no intra-game exits
2. **DOG (conf < 60%, entry ≥ $0.30):** managed via ESPN Win Probability
   - Q1 underdog lead → sell at PM price
   - Halftime: WP < 25% stop-loss, WP > 65% take-profit

**Sizing:** Half-Kelly capped at 10% per bet.

Backtest result (Jan 28 → Mar 15 2026, n=76): **+40% return, Sharpe 5.50, 8.4% max drawdown** (Flat 2% sizing).

Live result (Mar 16 → Apr 5 2026, n=25): **−12% return**. Caveat: later discovered to be a regime-shift artifact. See the Key Findings section.

**Run it:**
```bash
python -m src.Polymarket.paper_trader_v2 --init      # place bets pre-game
python -m src.Polymarket.paper_trader_v2 --monitor   # ESPN WP exits for DOG leg
python -m src.Polymarket.paper_trader_v2 --resolve   # close finished games
python -m src.Polymarket.paper_trader_v2 --status    # positions + P&L
python -m src.Polymarket.paper_trader_v2 --reset     # reset bankroll/positions
```

---

## Soccer: draw system (second live system)

**Stage 1 (XGBoost):** given the live match state (minute, score, xG, momentum signals), is the draw mispriced by PM? Fire only when `P_model(draw) − price > threshold`.

**Stage 2 (Cox survival):** given the hazard rate for the next goal, is NOW the optimal entry window, or will price improve? Time the entry instead of just firing immediately.

Hold-to-expiry execution — no in-game exits. Also bets the NO side when PM overprices draws in 0-1 games.

Files: `src/Soccer/draw_scanner.py`, `src/Soccer/draw_trader.py`, `src/Soccer/survival_model.py`, `src/Soccer/equalizer_model.py`.

---

## Data pipeline

Build fresh NBA training datasets:
```bash
cd src/Process-Data
python -m Get_Data                  # team stats from NBA Stats
python -m Get_Odds_Data             # sportsbook odds
python -m Create_Games              # merge → training feature set
```

Use `--backfill --season 2025-26` to target a specific season.

---

## Model training

```bash
cd src/Train-Models
python -m XGBoost_Model_ML --dataset dataset_2012-26 --trials 100 --splits 5 --calibration sigmoid
python -m XGBoost_Model_UO --dataset dataset_2012-26 --trials 100 --splits 5 --calibration sigmoid
python -m NN_Model_ML
python -m NN_Model_UO
```

Current live model: `XGBoost_69.8%_ML_md4_eta0p033_*.json` (trained Jan 25 2026 on 2012-2025).

---

## Backtest & analysis scripts (project root)

### Dataset builders
- `build_backtest_dataset.py` — master pipeline: pulls PM events + CLOB price history + generates XGBoost predictions → `Data/backtest/nba_backtest_dataset.csv`
- `build_historical_backtest.py` — extended historical backtest across 2012-2026 using sportsbook ML as proxy for PM
- `build_price_history.py` — PM CLOB price history collection
- `build_espn_wp.py` — ESPN WP data for backtest
- `fetch_q1_scores.py` — Q1 score collection
- `scripts/fetch_pm_1min.py`, `scripts/fetch_pm_histories.py` — periodic PM snapshot fetches

### Strategy backtests
- `backtest_simulation.py` — dual-leg baseline backtest
- `sizing_comparison.py` — Flat / Half-Kelly / Edge-scaled comparison → `Data/backtest/analysis/sizing_*.csv`
- `flip_backtest.py` — Dr. Yang's flip hypothesis test
- `soccer_backtest_sim.py` — soccer draw backtest
- `backtest_analysis.py` — 3-part NBA study (accuracy decomposition, entry timing, bankroll sim)

### Strategy research (this session's findings)
- `leg_weight_optimization.py` — FAV-vs-DOG Kelly / risk-parity / mean-variance weights
- `mispricing_filter.py` — logistic-regression calibrated filter over raw `model_edge`
- `monthly_decomposition.py` — monthly FAV/DOG performance trends
- `tick_features.py` — microstructure features from 10-min PM tick snapshots
- `enriched_filter.py` — 12-feature mispricing filter using tick data
- `backfill_market_context.py` — retroactively enrich live trades with market-context features (within PM's 30-day window)
- `retro_backtest_late_season.py` — FAV-only retro-backtest for Apr 6-13 window (after V2 was paused)

All analysis outputs land in `Data/backtest/analysis/`.

---

## Data directory

| Path | Content |
|------|---------|
| `Data/OddsData.sqlite` | 18 seasons of sportsbook odds + outcomes (2007-2026) |
| `Data/TeamData.sqlite` | Per-date team stat snapshots (used by backtest predictions) |
| `Data/dataset.sqlite` | Training datasets |
| `Data/nba-2025-UTC.csv` | 2025-26 regular season schedule |
| `Data/backtest/` | Backtest inputs/outputs, PM tokens metadata, price history, ESPN WP data |
| `Data/backtest/analysis/` | Reports + scored trade CSVs from strategy research |
| `Data/paper_trading_v2/` | V2 live state: bankroll.json, positions.json, trades.json, alerts.jsonl |
| `Data/soccer_draw_trading/` | Soccer draw live state |
| `Data/soccer_backtest/` | Soccer backtest data |

---

## Setup

### 1. Python environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. API keys / credentials
**None required** for NBA, soccer-draw, and CBB live operation — all providers used (Polymarket Gamma API, NBA Stats, ESPN, FotMob, SBR) are public.

**Optional — Discord alerts for CBB**: the webhook URL is currently hardcoded in `src/CBB/paper_trader.py:65`. If you fork this repo or distribute it, rotate the webhook and move to an env var before committing.

### 3. Data & models

| What | State in repo | How to get it |
|------|---------------|---------------|
| NBA XGBoost ML model (69.8%) | **Committed** in `Models/XGBoost_Models/` | — |
| Soccer XGBoost + Cox pkls | **Committed** in `Data/soccer_models/` | — |
| NBA schedule CSV (2025-26) | **Committed** at `Data/nba-2025-UTC.csv` | Manually refresh per season |
| `Data/OddsData.sqlite` | **Gitignored** | `python -m src.Process-Data.Get_Odds_Data --backfill` |
| `Data/TeamData.sqlite` | **Gitignored** | `python -m src.Process-Data.Get_Data --backfill` |
| `Data/backtest/*` (PM tokens, price history, analyses) | **Gitignored** | `python build_backtest_dataset.py` |
| All `Data/*paper_trading*/` state | **Gitignored** | Created on first `--init` run |
| `nba_game_snapshots.parquet` (10-min ticks) | Not in repo | Fetched via `scripts/fetch_pm_1min.py` |

A fresh clone can generate everything — but expect a few hours of scraping/API work the first time. You only need the components for whatever you actually intend to run.

---

## Quick start

```bash
# NBA prediction (one-shot CLI — uses live NBA.com stats)
python main.py -xgb -odds=polymarket -kc

# V2 NBA paper trader — individual commands
python -m src.Polymarket.paper_trader_v2 --init      # place bets pre-game
python -m src.Polymarket.paper_trader_v2 --monitor   # ESPN WP exits for DOG leg
python -m src.Polymarket.paper_trader_v2 --resolve   # close finished games
python -m src.Polymarket.paper_trader_v2 --status    # positions + P&L

# V2 NBA paper trader — full scheduler loop
python -m src.Polymarket.scheduler

# Soccer draw system (XGBoost + Cox survival, live)
python -m src.Soccer.draw_scheduler

# CBB paper trader
python -m src.CBB.scheduler

# Streamlit dashboard
streamlit run src/Dashboard/app.py

# Utilities
python -m src.Utils.AlertManager --recent 20
python -m src.Utils.PerformanceAnalytics --report --days 7
python -m src.Utils.Backtester --report
```

Supported sportsbooks via `main.py -odds=`: `fanduel`, `draftkings`, `betmgm`, `pointsbet`, `caesars`, `wynn`, `bet_rivers_ny`, `polymarket`.

---

## Key findings from 2025-26 season

Full write-up: `Data/backtest/analysis/final_report.md` (the Dr. Yang submission).

**What worked:**
- Dual-leg structure (+40% backtest return, up from −25%+ losses on unfiltered Kelly)
- 7% edge + 60% confidence filter isolated the tradable subset
- ESPN WP Q1-exit on DOG leg was the biggest source of in-regime profit

**What broke in live:**
- FAV leg collapsed in late March (40% WR vs 67% in backtest)
- Monthly decomposition showed degradation started *within the backtest* in March (Feb FAV +71% → Mar FAV +34%)
- Mispricing filter trained on Oct-Mar data failed OOS because it concentrated into exactly the leg that was failing
- **Tentative explanation:** late-regular-season regime (star rest, tanking, seeding dynamics) is underrepresented in training data. Season-to-date stats don't reflect current rotations.

**Operating window that probably works:** ~Nov 20 → Mar 10. Early season = stats undercook. Late season = regime shift. Checklist for applying this next season is in `NEXT_SEASON_CHECKLIST.md`.

**Mispricing-filter design signal worth keeping:** coefficient analysis on tick features showed a *sign-dependent momentum* — slow 2-hour price grinds predict losses (edge already in price), sharp recent 30-min moves predict wins (fresh sharp money). This is microstructure logic that generalizes to any binary prediction market.

---

## Requirements

- Python 3.11+ (3.12 tested)
- Key packages: `xgboost`, `tensorflow`, `pandas`, `scikit-learn`, `streamlit`, `plotly`, `pyarrow` (for parquet tick data), `requests`

Install via `pip install -r requirements.txt`.

---

## Credits

- NBA outcome model: XGBoost trained on 2012-2025 seasons, 69.8% accuracy on ML
- Strategy research supervised by **Dr. Hanchao Yang** (Duke, MIDS; Kalshi trader)
- Soccer draw model: XGBoost classifier + Cox survival model
- Data: Polymarket Gamma + CLOB, NBA Stats, ESPN, FotMob, Sports-Betting-Refs
