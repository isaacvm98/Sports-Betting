@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo.
echo ============================================
echo   CBB PAPER TRADER - College Basketball
echo ============================================
echo.
echo Features enabled:
echo   - DrawdownManager (5%% daily, 10%% weekly, 20%% total)
echo   - Discord alerts for resolutions
echo   - Position limits (max 5 simultaneous)
echo   - Edge thresholds (min 2 pts spread, 8%% ML)
echo   - ML filters (25-45%% entry price range)
echo.
python -m src.CBB.scheduler
pause
