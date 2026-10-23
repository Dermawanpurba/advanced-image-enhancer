@echo off
title AI Agent - Auto Batch Photo Enhancer
cd /d "%~dp0"
echo =================================================================
echo   Menjalankan AI Agent Automatic Batch Enhancer
echo   Folder Input : MD
echo   Folder Output: MD_ENHANCED
echo =================================================================
echo.
call .\.venv\Scripts\python.exe auto_process_all_photos.py %*
echo.
pause
