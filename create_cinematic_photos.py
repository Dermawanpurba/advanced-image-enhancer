from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure root package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_enhancer.enhancers import correct_exif_orientation, save_image


def apply_cinematic_pro_camera(
    image_bgr: np.ndarray,
    cine_style: str = "kodak_portra",
    pro_mist_bloom: float = 0.22,
    vignette_strength: float = 0.22,
    film_grain_strength: float = 0.018,
    lift_blacks: float = 6.0,
    micro_contrast: float = 0.28,
) -> np.ndarray:
    """
    True Professional Cinema Camera & Film Emulsion Processing:
    - 100% SHARP ON EVERY PERSON: Menjaga seluruh tubuh manusia utuh tajam (rambut, wajah, tubuh, gaun, jas, kaki, sepatu).
    - Black Pro-Mist 1/8 Diffusion: Kilau hangat lembut pada lampu dekorasi, lilin, dan pantulan cahaya.
    - Filmic S-Curve Tone: Matte velvety darks dan soft highlight roll-off (bebas silau flash tajam).
    - Kodak Portra Color Science: Warna kulit hangat cerah bercahaya (golden-peach glow).
    - Pro Prime Lens Micro-Contrast: Ketajaman serat jas, kebaya, bordir renda, dan perhiasan.
    - Optical Vignette: Sudut lensa f/1.4 alami memusatkan fokus ke subjek.
    - 35mm Micro Film Grain: Tekstur film organik menghilangkan kesan digital steril.
    """
    h, w = image_bgr.shape[:2]
    img_float = image_bgr.astype(np.float32) / 255.0

    # 1. Pro-Mist / Cine Diffusion Filter (Glow around bright lights and reflections)
    if pro_mist_bloom > 0:
        gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
        bloom_thresh = np.maximum(0.0, gray - 0.78) / 0.22
        bloom_mask = (bloom_thresh ** 2)[..., None]

        bloom_k = int(min(w, h) * 0.05) | 1
        bloom_layer = cv2.GaussianBlur(img_float * bloom_mask, (bloom_k, bloom_k), 0)
        composite = img_float + bloom_layer * pro_mist_bloom * 0.85
        composite = np.clip(composite, 0.0, 1.0)
    else:
        composite = img_float

    # 2. Cinema S-Curve & Filmic Black Lift (Velvety darks & soft highlight roll-off)
    black_offset = lift_blacks / 255.0
    mid = 0.50
    x = composite
    curve = 1.0 / (1.0 + np.exp(-4.5 * (x - mid)))
    curve = (curve - curve.min()) / (curve.max() - curve.min())

    graded = 0.78 * x + 0.22 * curve
    graded = black_offset + graded * (0.98 - black_offset)
    graded = np.clip(graded, 0.0, 1.0)

    # 3. Cinematic Split-Toning & Kodak Portra Color Science
    b, g, r = graded[:, :, 0], graded[:, :, 1], graded[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    if cine_style == "kodak_portra":
        shadow_weight = np.clip(1.0 - 2.2 * lum, 0.0, 1.0)
        mid_weight = np.clip(1.0 - np.abs(lum - 0.5) * 2.2, 0.0, 1.0)
        hi_weight = np.clip(2.0 * lum - 1.0, 0.0, 1.0)

        # Shadows: Gentle cool teal tint (-R, +B)
        b_shift = shadow_weight * 0.020
        r_shadow = shadow_weight * -0.012

        # Midtones: Warm golden peach skin tones (+R, +G, -B)
        r_mid = mid_weight * 0.030
        g_mid = mid_weight * 0.010
        b_mid = mid_weight * -0.020

        # Highlights: Soft warm champagne
        r_hi = hi_weight * 0.012
        g_hi = hi_weight * 0.006

        graded[:, :, 0] = np.clip(graded[:, :, 0] + b_shift + b_mid, 0.0, 1.0)
        graded[:, :, 1] = np.clip(graded[:, :, 1] + g_mid + g_hi, 0.0, 1.0)
        graded[:, :, 2] = np.clip(graded[:, :, 2] + r_shadow + r_mid + r_hi, 0.0, 1.0)

    elif cine_style == "teal_orange":
        shadow_weight = np.clip(1.0 - 2.0 * lum, 0.0, 1.0)
        hi_weight = np.clip(2.0 * lum - 0.5, 0.0, 1.0)

        graded[:, :, 0] += shadow_weight * 0.035
        graded[:, :, 1] += shadow_weight * 0.015
        graded[:, :, 2] -= shadow_weight * 0.025

        graded[:, :, 2] += hi_weight * 0.035
        graded[:, :, 1] += hi_weight * 0.012
        graded[:, :, 0] -= hi_weight * 0.025
        graded = np.clip(graded, 0.0, 1.0)

    # 4. Pro Prime Lens Micro-Contrast & Clarity (Tack-Sharp 3D Pop on every person)
    if micro_contrast > 0:
        blur_detail = cv2.GaussianBlur(graded, (0, 0), 2.0)
        high_pass = graded - blur_detail
        graded = np.clip(graded + high_pass * micro_contrast, 0.0, 1.0)

    # 5. Optical Lens Vignette
    if vignette_strength > 0:
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w / 2.0, h / 2.0
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = np.sqrt(((X - center_x) * 1.0)**2 + ((Y - center_y) * 1.1)**2) / max_dist
        vig_factor = 1.0 - vignette_strength * np.clip((dist - 0.42) / 0.58, 0.0, 1.0) ** 1.8
        graded = graded * vig_factor[..., None]

    # 6. Organic 35mm Micro Film Grain
    if film_grain_strength > 0:
        np.random.seed(123)
        grain = np.random.normal(0.0, 1.0, (h, w)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (3, 3), 0.6)
        grain_mask = np.clip(1.0 - 4.0 * (lum - 0.5)**2, 0.05, 1.0)
        grain_layer = grain * grain_mask * film_grain_strength
        graded = np.clip(graded + grain_layer[..., None], 0.0, 1.0)

    return (graded * 255.0).astype(np.uint8)


def process_cinematic_batch(
    input_dir: str = "MD_ENHANCED",
    output_dir: str = "MD_CINEMATIC",
    limit: Optional[int] = None,
    overwrite: bool = False,
    cine_style: str = "kodak_portra",
):
    in_path = Path(input_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Error: Folder input {in_path} tidak ditemukan!")
        return

    # Find all JPG files
    seen = set()
    all_files = []
    for ext in ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.png"):
        for f in sorted(in_path.glob(ext)):
            if f.name.lower() not in seen:
                seen.add(f.name.lower())
                all_files.append(f)

    if limit:
        all_files = all_files[:limit]

    total = len(all_files)
    print("=" * 70)
    print("   PRO CAMERA CINEMATIC ENHANCER (KODAK PORTRA & BLACK PRO-MIST)")
    print(f"   Total Foto  : {total} file")
    print(f"   Input       : {in_path}")
    print(f"   Output      : {out_path}")
    print(f"   Style       : {cine_style}")
    print(f"   Fokus Subjek: 100% TAJAM UTUH 1 TUBUH (Tanpa Blur Pada Orang)")
    print(f"   Mode        : {'REPROCESS ALL (Overwrite)' if overwrite else 'Incremental (Skip Selesai)'}")
    print("=" * 70)

    log_file = out_path / "batch_report_cinematic.txt"
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n--- Cinematic Batch Started at {time.strftime('%Y-%m-%d %H:%M:%S')} (total={total}) ---\n")

    start_all = time.time()
    success_count = 0

    for idx, fpath in enumerate(all_files, start=1):
        target_file = out_path / fpath.name

        # Skip if already exists and complete
        if not overwrite and target_file.exists() and target_file.stat().st_size > 1000000:
            print(f"[{idx}/{total}] SKIP: {fpath.name} (sudah selesai sebelumnya)")
            continue

        t0 = time.time()
        print(f"[{idx}/{total}] Memproses Cinematic: {fpath.name} ... ", end="", flush=True)

        try:
            img = cv2.imread(str(fpath))
            if img is None:
                print("GAGAL (File korup/tidak terbaca)")
                continue

            res_bgr = apply_cinematic_pro_camera(img, cine_style=cine_style)

            # Save with 100% maximum quality & 4:4:4 sampling
            res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
            save_image(res_rgb, target_file)

            elapsed = time.time() - t0
            success_count += 1
            print(f"SELESAI! ({elapsed:.1f}s)")

            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{idx}/{total}] {fpath.name} -> SELESAI ({elapsed:.1f}s)\n")

        except Exception as e:
            print(f"ERROR: {e}")
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{idx}/{total}] {fpath.name} ERROR: {e}\n")

    total_time = time.time() - start_all
    print("=" * 70)
    print(f"SELESAI SEMUA! Berhasil memproses: {success_count}/{total} foto.")
    print(f"Total waktu : {total_time / 60:.1f} menit.")
    print(f"Hasil tersimpan rapi di: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Cinematic Pro Camera Processor")
    parser.add_argument("--input-dir", type=str, default="MD_ENHANCED", help="Folder sumber foto")
    parser.add_argument("--output-dir", type=str, default="MD_CINEMATIC", help="Folder hasil foto sinematik")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah foto untuk pengujian")
    parser.add_argument("--overwrite", action="store_true", help="Proses ulang semua file yang sudah ada")
    parser.add_argument("--style", type=str, default="kodak_portra", choices=["kodak_portra", "teal_orange"])
    args = parser.parse_args()

    process_cinematic_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        cine_style=args.style,
    )
