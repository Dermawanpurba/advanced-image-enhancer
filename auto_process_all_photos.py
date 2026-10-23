from __future__ import annotations

import sys
import time
from pathlib import Path

from typing import Optional, Tuple

import cv2
import numpy as np

# Ensure image_enhancer package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_enhancer.ai_face import enhance_faces_ai, load_ai_models
from image_enhancer.enhancers import (
    adaptive_sharpening,
    enhance_contrast_local,
    enhance_saturation,
    save_image,
)


def auto_enhance_photo(
    img_bgr: np.ndarray,
    ai_strength: float = 0.85,
    enable_beauty: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Intelligently analyzes and enhances an image:
    1. Detects brightness & shadow levels to automatically determine optimal exposure lift.
    2. Protects highlights & flash glare (white decorations, lamps, bright shirts, shiny foreheads).
    3. Natural Beauty Retouching: softly evens out blemishes and oily flash reflections while keeping real skin pores.
    4. Restores all human faces using AI (YuNet + GFPGAN) with seamless blending.
    5. Applies high-pass halo-free crisp sharpening.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to LAB for luminance & color analysis
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    mean_l = float(np.mean(l))
    p10_l = float(np.percentile(l, 10))
    p90_l = float(np.percentile(l, 90))
    std_l = float(np.std(l))

    # Auto-exposure decision based on image luminance distribution:
    # If photo has direct flash or high dynamic range, keep contrast gentle to avoid harsh glare and blown-out highlights
    if p90_l > 185 or std_l > 55:
        contrast_boost = 1.03
        condition = "Flash / High Dynamic Range (Soft & Gentle)"
    elif mean_l < 95 and p90_l < 165:
        contrast_boost = 1.18
        condition = "Low-Light / Underexposed"
    elif mean_l < 120 and p90_l < 185:
        contrast_boost = 1.08
        condition = "Medium Indoor / Shadowed Subjects"
    else:
        contrast_boost = 1.04
        condition = "Well-Lit / Standard"

    mode_params = {
        "contrast_boost": contrast_boost,
        "saturation_boost": 1.06,
        "sharpening_strength": 1.15,
    }

    # Step 1: Intelligent local contrast & face illumination with anti-glare protection
    enhanced = enhance_contrast_local(img_rgb, mode_params)
    enhanced = enhance_saturation(enhanced, mode_params)

    # Step 2: High-pass crisp detail sharpening
    enhanced = adaptive_sharpening(enhanced, mode_params)

    # Step 3: AI Face Restoration & Natural Beauty Retouching (YuNet + GFPGAN)
    enhanced = enhance_faces_ai(enhanced, blend_strength=ai_strength, enable_beauty=enable_beauty)

    res_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    stats = {
        "mean_brightness": round(mean_l, 1),
        "condition": condition,
        "contrast_boost": contrast_boost,
    }
    return res_bgr, stats


def process_folder(
    input_dir: str = "MD",
    output_dir: str = "MD_ENHANCED",
    max_photos: Optional[int] = None,
    overwrite: bool = False,
):
    in_path = Path(input_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Error: Folder {in_path} tidak ditemukan!")
        return

    # Find all original JPG files (case-insensitive deduplication)
    seen = set()
    all_files = []
    for ext in ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.png"):
        for f in sorted(in_path.glob(ext)):
            if "enhanced" not in f.name.lower() and f.name.lower() not in seen:
                seen.add(f.name.lower())
                all_files.append(f)

    if max_photos:
        all_files = all_files[:max_photos]

    total = len(all_files)
    print("=" * 65)
    print(f"  AI AGENT AUTOMATIC BATCH ENHANCER")
    print(f"  Total Foto: {total} file")
    print(f"  Mode: {'REPROCESS ALL (Overwrite)' if overwrite else 'Incremental (Skip Completed)'}")
    print(f"  Input : {in_path}")
    print(f"  Output: {out_path}")
    print("=" * 65)

    # Preload AI models once
    load_ai_models()

    log_file = out_path / "batch_report.txt"
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n--- Batch Started at {time.strftime('%Y-%m-%d %H:%M:%S')} (overwrite={overwrite}) ---\n")

    success_count = 0
    start_all = time.time()

    for idx, fpath in enumerate(all_files, start=1):
        target_file = out_path / fpath.name

        # Skip if already processed in earlier run unless overwrite is requested
        if not overwrite and target_file.exists() and target_file.stat().st_size > 1000000:
            print(f"[{idx}/{total}] SKIP: {fpath.name} (sudah selesai sebelumnya)")
            continue

        t0 = time.time()
        print(f"[{idx}/{total}] Memproses: {fpath.name} ... ", end="", flush=True)

        try:
            img = cv2.imread(str(fpath))
            if img is None:
                print("GAGAL (File korup/tidak terbaca)")
                continue

            res_bgr, stats = auto_enhance_photo(img)

            # Save with 100% maximum quality & 4:4:4 sampling
            save_image(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB), target_file)

            elapsed = time.time() - t0
            success_count += 1
            print(f"SELESAI! ({elapsed:.1f}s | {stats['condition']})")

            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{idx}/{total}] {fpath.name} -> {stats['condition']} ({elapsed:.1f}s)\n")

        except Exception as e:
            print(f"ERROR: {e}")
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{idx}/{total}] {fpath.name} ERROR: {e}\n")

    total_time = time.time() - start_all
    print("=" * 65)
    print(f"SELESAI SEMUA! Berhasil: {success_count}/{total} foto.")
    print(f"Total waktu: {total_time / 60:.1f} menit.")
    print(f"Hasil tersimpan rapi di: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of photos (for testing)")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess all photos with the new soft beauty algorithm")
    args = parser.parse_args()

    process_folder(input_dir="MD", output_dir="MD_ENHANCED", max_photos=args.limit, overwrite=args.overwrite)
