@echo off
chcp 65001 >nul
title FX Trading Bot

echo ============================================================
echo  FX Trading Bot
echo  Press Ctrl+C to stop
echo ============================================================
echo.

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run the trading script
python scripts\run_trading.py

echo.
echo Trading stopped.
pause
