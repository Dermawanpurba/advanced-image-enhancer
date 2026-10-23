@echo off
title Advanced Image Enhancer CLI
cd /d "%~dp0"
.\.venv\Scripts\python.exe -m image_enhancer %*
