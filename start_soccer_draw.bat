@echo off
echo ==========================================
echo    Soccer Draw Betting (XGBoost + Survival)
echo ==========================================
echo.
echo Strategy:
echo - XGBoost draw model + Cox survival timing
echo - Entry: Min 55-75, trailing by 1, edge ^> 5%%
echo - Exit: Hold to expiry ($1 draw / $0 no draw)
echo - Sizing: Half-Kelly (8%% cap, no position limit)
echo.
echo Leagues:
echo - Premier League, La Liga, Bundesliga
echo - Serie A, Ligue 1
echo.
echo Schedule:
echo - Scan window: 10:00-24:00 UTC
echo - Scan interval: 30s
echo - WebSocket: Real-time Polymarket prices
echo.
echo Models:
echo - Data\soccer_models\draw_xgb.pkl
echo - Data\soccer_models\cox_model.pkl
echo.
echo Logs: logs\draw_scheduler.log
echo Data: Data\soccer_draw_trading\
echo.
echo Press Ctrl+C to stop
echo ==========================================
echo.

cd /d "%~dp0"
if not exist logs mkdir logs

.venv\Scripts\python.exe -m src.Soccer.draw_scheduler

pause
