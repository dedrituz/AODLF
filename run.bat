@echo off

cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%

echo [INFO] Launching AODLF Backend from: %CD%
python main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] App crashed or failed to start!
    pause
) else (
    echo.
    echo [SUCCESS] App exited normally.
    pause
)
