#!/usr/bin/env bash
# Script peluncur Live Studio Web Server (Port 8501)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================================="
echo "  ⚡ Launching AI Image Enhancer Pro Live Studio..."
echo "  📍 URL: http://localhost:8501"
echo "=========================================================="

# Buka browser otomatis
sleep 1 && open "http://localhost:8501" 2>/dev/null &

# Jalankan server
.venv_mac/bin/python web_studio.py
