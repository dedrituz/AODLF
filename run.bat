@echo off
REM ==========================================================
REM  AODLF PROD Backend Launcher
REM ==========================================================

REM Ensure we are in the directory where this script lives
cd /d "%~dp0"

REM Set PYTHONPATH to include the current directory so 'app' is a reachable package
set PYTHONPATH=%CD%;%PYTHONPATH%

echo [INFO] Launching AODLF Backend from: %CD%
python app/main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] App crashed or failed to start!
    pause
) else (
    echo.
    echo [SUCCESS] App exited normally.
    pause
)
