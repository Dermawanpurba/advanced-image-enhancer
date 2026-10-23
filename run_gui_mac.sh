#!/usr/bin/env bash
# Script peluncur Desktop GUI Modern (CustomTkinter)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================================="
echo "  ⚡ Launching Modern Desktop GUI (CustomTkinter)..."
echo "=========================================================="

.venv_mac/bin/python gui.py
