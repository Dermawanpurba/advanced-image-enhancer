@echo off
title PRO CAMERA CINEMATIC BATCH PROCESSOR
color 0B
cd /d "%~dp0"

echo ======================================================================
echo           PRO CAMERA CINEMATIC PROCESSOR (FULL BODY IN FOCUS)
echo ======================================================================
echo  - Input Folder  : MD_ENHANCED
echo  - Output Folder : MD_CINEMATIC
echo  - Kualitas Foto : 100%% Maksimal (Uncompressed 4:4:4)
echo  - Efek Kamera   : Kodak Portra Tone, Black Pro-Mist, Velvety Blacks,
echo                    Micro-Contrast Clarity, Prime Lens Vignette, Film Grain
echo  - Subjek Orang  : 100%% UTUH TAJAM (Wajah, Tubuh, Kaki, Sepatu)
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" create_cinematic_photos.py %*
) else (
    python create_cinematic_photos.py %*
)

echo.
echo Selesai! Tekan tombol apa saja untuk menutup jendela ini...
pause >nul
