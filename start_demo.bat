@echo off
chcp 65001 >nul
title FX Demo Trading

echo ============================================================
echo  FX Demo Trading System
echo  Press Ctrl+C to stop
echo ============================================================
echo.

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run the demo trading script
python scripts\run_demo.py

echo.
echo Trading stopped.
pause
