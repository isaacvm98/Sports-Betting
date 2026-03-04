@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m src.Utils.AlertManager --recent 20
pause
