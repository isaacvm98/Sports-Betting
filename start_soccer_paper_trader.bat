@echo off
echo ==========================================
echo    Soccer Momentum Draw Paper Trading
echo ==========================================
echo.
echo Strategy:
echo - Momentum draw betting (backtest: 874 matches)
echo - Entry: Minute 70-75, losing by 1, momentum 0.50+
echo - Exit: Equalization (sell at ~42c) or expiry
echo - Draw price range: 5c - 18c
echo.
echo Leagues:
echo - Premier League, La Liga, Bundesliga
echo - Serie A, Ligue 1
echo.
echo Schedule:
echo - Scan window: 11:00-23:00 UTC
echo - Scan interval: 60s
echo - WebSocket: Real-time Polymarket prices
echo.
echo Logs: logs\soccer_scheduler.log
echo Data: Data\soccer_paper_trading\
echo.
echo Press Ctrl+C to stop
echo ==========================================
echo.

cd /d "%~dp0"
if not exist logs mkdir logs

call .venv\Scripts\activate.bat

python -m src.Soccer.scheduler

pause
