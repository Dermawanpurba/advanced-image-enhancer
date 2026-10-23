@echo off
title Advanced Image Enhancer GUI
cd /d "%~dp0"
echo Starting Advanced Image Enhancer GUI...
call .\.venv\Scripts\python.exe gui.py
pause
