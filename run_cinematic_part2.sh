#!/bin/bash
cd "$(dirname "$0")"

echo "======================================================================"
echo "    PRO CAMERA CINEMATIC & FACE RELIGHTING PROCESSOR (MD PART 2)"
echo "======================================================================"
echo " - Input Folder   : MD/md part2"
echo " - Output Folder  : MD_CINEMATIC_PART2"
echo " - Kualitas Foto  : 100% Maksimal (Uncompressed 4:4:4)"
echo " - Fitur AI       : Virtual Fill Light (Mencerahkan Wajah Redup/Bayangan)"
echo " - Efek Kamera    : Kodak Portra Tone, Black Pro-Mist, Velvety Blacks,"
echo "                    Micro-Contrast Clarity, Prime Lens Vignette, Film Grain"
echo "======================================================================"
echo ""

if [ -f ".venv_mac/bin/python3" ]; then
    .venv_mac/bin/python3 process_cinematic_part2.py "$@"
elif [ -f ".venv/bin/python3" ]; then
    .venv/bin/python3 process_cinematic_part2.py "$@"
else
    python3 process_cinematic_part2.py "$@"
fi
