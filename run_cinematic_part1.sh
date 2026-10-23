#!/bin/bash
cd "$(dirname "$0")"

echo "======================================================================"
echo "    PRO CAMERA CINEMATIC & FACE RELIGHTING PROCESSOR (MD PART 1)"
echo "======================================================================"
echo " - Input Folder   : MD"
echo " - Output Folder  : MD_CINEMATIC"
echo " - Kualitas Foto  : 100% Maksimal (Uncompressed 4:4:4)"
echo " - Fitur AI       : Virtual Fill Light (Kulit Terikat Anatomi & Bebas Buram)"
echo " - Efek Kamera    : Kodak Portra Tone, Black Pro-Mist, Velvety Blacks,"
echo "                    Micro-Contrast Clarity, Prime Lens Vignette, Film Grain"
echo "======================================================================"
echo ""

if [ -f ".venv_mac/bin/python3" ]; then
    .venv_mac/bin/python3 process_cinematic_part2.py --input-dir MD --output-dir MD_CINEMATIC --overwrite "$@"
elif [ -f ".venv/bin/python3" ]; then
    .venv/bin/python3 process_cinematic_part2.py --input-dir MD --output-dir MD_CINEMATIC --overwrite "$@"
else
    python3 process_cinematic_part2.py --input-dir MD --output-dir MD_CINEMATIC --overwrite "$@"
fi
