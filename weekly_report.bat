@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo.
echo ========== PERFORMANCE REPORT ==========
python -m src.Utils.PerformanceAnalytics --report --days 7
echo.
echo ========== BACKTEST COMPARISON ==========
python -m src.Utils.Backtester --report
pause
