# Sports Betting with Machine Learning

Predicts NBA game outcomes (moneyline and over/under) using XGBoost and neural network models trained on historical team stats and sportsbook odds from 2007 to present. Extends to college basketball (CBB), soccer, and Polymarket prediction markets.

## Features

- **NBA Predictions** — Moneyline and totals predictions with expected value and Kelly Criterion sizing
- **Multiple Models** — XGBoost (primary) and neural network models with calibrated probabilities
- **Odds Providers** — Automated odds ingestion from sportsbooks (FanDuel, DraftKings, BetMGM, etc.) and Polymarket
- **Paper Trading** — Simulated trading with drawdown management and performance analytics
- **Streamlit Dashboard** — Interactive dashboard for browsing predictions and tracking performance
- **Backtesting** — Historical backtesting engine to evaluate model performance
- **Multi-Sport** — CBB and soccer modules with dedicated data providers and paper traders
- **Alerts & Monitoring** — ESPN live game tracking, injury adjustments, and alert management

## Project Structure

```
├── main.py                  # Entry point — daily NBA predictions
├── config.toml              # Season date ranges for data pipeline
├── src/
│   ├── DataProviders/       # Odds and data providers (SBR, Polymarket, ESPN, FotMob)
│   ├── Process-Data/        # Data collection, odds scraping, game feature engineering
│   ├── Train-Models/        # XGBoost and NN model training scripts
│   ├── Predict/             # Model inference runners
│   ├── Utils/               # Kelly Criterion, expected value, backtester, drawdown, alerts
│   ├── Dashboard/           # Streamlit web dashboard
│   ├── Polymarket/          # Polymarket prediction market integration
│   ├── CBB/                 # College basketball module
│   └── Soccer/              # Soccer module (FotMob data, backtesting)
├── Models/                  # Trained model files (not tracked in git)
├── Data/                    # SQLite databases and schedules
└── Tests/                   # Unit tests
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run predictions

```bash
python main.py -xgb -odds=fanduel
```

Supported odds sources: `fanduel`, `draftkings`, `betmgm`, `pointsbet`, `caesars`, `wynn`, `bet_rivers_ny`

Flags:
- `-xgb` — XGBoost model
- `-nn` — Neural network model
- `-A` — All models
- `-kc` — Show Kelly Criterion bankroll sizing

### Launch dashboard

```bash
streamlit run src/Dashboard/app.py
```

## Data Pipeline

```bash
cd src/Process-Data

# Fetch team stats
python -m Get_Data

# Fetch odds data
python -m Get_Odds_Data

# Build training dataset
python -m Create_Games
```

Use `--backfill` to fetch historical data, optionally with `--season 2025-26` to target a specific season.

## Training Models

```bash
cd src/Train-Models

# XGBoost (primary)
python -m XGBoost_Model_ML --dataset dataset_2012-26 --trials 100 --splits 5 --calibration sigmoid
python -m XGBoost_Model_UO --dataset dataset_2012-26 --trials 100 --splits 5 --calibration sigmoid

# Neural network
python -m NN_Model_ML
python -m NN_Model_UO
```

## Requirements

- Python 3.11+
- Key packages: XGBoost, TensorFlow, Pandas, Scikit-learn, Streamlit, Plotly
