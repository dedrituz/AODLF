#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

export PYTHONPATH="$PWD:$PYTHONPATH"

echo "[INFO] Launching AODLF Backend from: $PWD"
python main.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[ERROR] App crashed or failed to start!"
else
    echo ""
    echo "[SUCCESS] App exited normally."
fi

read -p "Press Enter to continue..."