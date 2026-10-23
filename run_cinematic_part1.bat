@echo off
title PRO CAMERA CINEMATIC PROCESSOR (MD PART 1)
color 0B
cd /d "%~dp0"

echo ======================================================================
echo    PRO CAMERA CINEMATIC & FACE RELIGHTING PROCESSOR (MD PART 1)
echo ======================================================================
echo  - Input Folder   : MD
echo  - Output Folder  : MD_CINEMATIC
echo  - Kualitas Foto  : 100%% Maksimal (Uncompressed 4:4:4)
echo  - Fitur AI       : Virtual Fill Light (Kulit Terikat Anatomi & Bebas Buram)
echo  - Efek Kamera    : Kodak Portra Tone, Black Pro-Mist, Velvety Blacks,
echo                    Micro-Contrast Clarity, Prime Lens Vignette, Film Grain
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" process_cinematic_part2.py --input-dir MD --output-dir MD_CINEMATIC --overwrite %*
) else (
    python process_cinematic_part2.py --input-dir MD --output-dir MD_CINEMATIC --overwrite %*
)

echo.
echo Selesai! Tekan tombol apa saja untuk menutup jendela ini...
pause >nul
