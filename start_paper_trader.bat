@echo off
echo ==========================================
echo    NBA Polymarket Paper Trading System
echo ==========================================
echo.
echo Configuration:
echo - Early exits: DISABLED (run to resolution)
echo - Tiered Kelly sizing: ENABLED
echo - Edge range: 5%% - 25%%
echo - Max positions: 6 (3 per team)
echo.
echo Schedule:
echo - Bets placed: 10 minutes before each game
echo - Monitor: Every 5 minutes
echo - Resolution: When games complete
echo.
echo Logs: logs\scheduler.log
echo Alerts: Data\paper_trading\alerts.jsonl
echo.
echo Press Ctrl+C to stop
echo ==========================================
echo.

cd /d "%~dp0"
if not exist logs mkdir logs

call .venv\Scripts\activate.bat

python -m src.Polymarket.scheduler

pause
