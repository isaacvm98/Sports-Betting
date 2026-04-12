@echo off
echo ==========================================
echo    NBA Polymarket Paper Trading System
echo ==========================================
echo.
echo Strategy: V2 Dual-Leg
echo - Leg 1 (FAV): conf ^>= 60%%, edge ^>= 7%%, hold to resolution
echo - Leg 2 (DOG): conf ^< 60%%, entry ^>= $0.30, edge ^>= 7%%, ESPN WP exits
echo - Sizing: Half-Kelly (capped 10%%)
echo - Max positions: 6 (3 per team)
echo.
echo Schedule:
echo - Bets placed: 10 minutes before each game
echo - Monitor: Every 5 minutes
echo - Resolution: When games complete
echo.
echo Logs: logs\scheduler.log
echo Alerts: Data\paper_trading_v2\alerts.jsonl
echo.
echo Press Ctrl+C to stop
echo ==========================================
echo.

cd /d "%~dp0"
if not exist logs mkdir logs

call .venv\Scripts\activate.bat

.venv\Scripts\python.exe -m src.Polymarket.scheduler

pause
