#!/bin/bash
# Launcher script for Cat Gesture Meme Tracker

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_PATH="${PYTHON_PATH:-python3}"

echo "Starting Gesture Meme Tracker..."
echo "Using Python: $PYTHON_PATH"
echo ""

cd "$SCRIPT_DIR"
exec "$PYTHON_PATH" gesture_meme_tracker.py
