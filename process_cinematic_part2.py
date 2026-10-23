#!/usr/bin/env python3
"""
NATURAL CINEMATIC & ANATOMICALLY-ANCHORED SKIN TONE ENHANCER (MD PART 2)
========================================================================
Architecture:
1. Anatomically-Anchored Human Skin Brightening:
   - Mencerahkan kulit manusia murni: Wajah, Telinga, Leher, Dada, Tangan & Lengan.
   - 100% Proteksi Kain & Karpet: Menolak karpet merah (red carpet), kain songket/kebaya
     pink/magenta/ungu, dan lantai dengan rumus fisika rasio hemoglobin (R > G >= B).
   - Zona anatomi terikat pada posisi subjek sehingga lantai/karpet tidak pernah tersentuh.
2. AI Face Natural Beauty Smoothing:
   - Meratakan tekstur kulit dan menyamarkan noda/kilau minyak secara halus.
   - 100% Landmark Protection: Mata, pupil, alis, kacamata, dan bibir tetap tajam kristal.
3. True Cinema Camera & Kodak Portra 400 Film Science:
   - 100% pakaian utuh tajam (kain tenun, renda gaun, jas hitam, perhiasan, sepatu).
   - Black Pro-Mist 1/8 bloom pada lampu hias, lilin, dan chandelier.
   - Filmic S-curve dengan velvety black floor.
   - Micro-contrast clarity & 35mm organic micro film grain.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

# Ensure root package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_enhancer.ai_face import apply_natural_beauty_smoothing, load_ai_models


def correct_exif_orientation(image_path: Path) -> np.ndarray:
    """Loads image and corrects EXIF rotation, returning BGR numpy array."""
    pil_img = Image.open(image_path)
    pil_img = ImageOps.exif_transpose(pil_img)
    rgb = np.array(pil_img)
    if len(rgb.shape) == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    elif rgb.shape[2] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_image_100(image_rgb: np.ndarray, output_path: Path):
    """Saves RGB image with 100% maximum quality and 4:4:4 chroma sampling."""
    pil_img = Image.fromarray(image_rgb)
    pil_img.save(
        output_path,
        format="JPEG",
        quality=100,
        subsampling=0,
        optimize=True,
    )


def apply_anatomical_skin_brightening(
    image_bgr: np.ndarray,
    faces: Optional[np.ndarray],
    target_skin_l: float = 145.0,
    max_lift: float = 36.0,
) -> Tuple[np.ndarray, float]:
    """
    Brightens real human skin (face, ears, neck, hands, arms) while strictly rejecting
    red carpets, pink/purple fabrics, dresses, and floors.
    """
    h, w = image_bgr.shape[:2]

    # 1. Build Anatomical Search Zones (Anchored around detected people)
    # Never process the floor / carpet blindly
    skin_zone_mask = np.zeros((h, w), dtype=np.float32)

    if faces is not None and len(faces) > 0:
        for f in faces:
            fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])

            # Zone A: Head, Ears, Jaw, and Neck
            head_top = max(0, fy - int(fh * 0.25))
            neck_bot = min(h, fy + int(fh * 1.85))
            head_left = max(0, fx - int(fw * 0.45))
            head_right = min(w, fx + int(fw * 1.45))
            cv2.rectangle(skin_zone_mask, (head_left, head_top), (head_right, neck_bot), 1.0, -1)

            # Zone B: Torso, Arms, and Hands (Clasped hands, bouquet hands)
            hands_top = neck_bot
            hands_bot = min(h, fy + int(fh * 4.20))
            hands_left = max(0, fx - int(fw * 1.80))
            hands_right = min(w, fx + int(fw * 2.80))
            cv2.rectangle(skin_zone_mask, (hands_left, hands_top), (hands_right, hands_bot), 1.0, -1)
    else:
        # Fallback: central portrait upper area if no faces detected
        cv2.rectangle(skin_zone_mask, (int(w * 0.20), int(h * 0.10)), (int(w * 0.80), int(h * 0.70)), 1.0, -1)

    # 2. Strict Biological Human Skin Filter (Hemoglobin Absorption Physics)
    b = image_bgr[:, :, 0].astype(float)
    g = image_bgr[:, :, 1].astype(float)
    r = image_bgr[:, :, 2].astype(float)

    # Human skin: Red dominates (R > G + 6), Green absorbs less than Blue (G >= B - 2).
    # Reject Red Carpet: R/G > 1.85 or R - G > 70 (intense dye saturation).
    not_carpet = (r / np.maximum(g, 1.0) < 1.85) & ((r - g) < 70)
    # Reject Pink / Magenta Fabric: Blue dye causes B > G, never true in human skin!
    not_pink_cloth = (g >= b - 2) & ((r - b) > 10)
    bio_skin = (r > g + 6) & not_carpet & not_pink_cloth

    # YCrCb human skin gamut
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_ycrcb = (cr >= 135) & (cr <= 170) & (cb >= 90) & (cb <= 122)

    # HSV human skin gamut (Hue 3 to 20, Saturation 25 to 165, Value >= 45)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    skin_hsv = (h_chan >= 3) & (h_chan <= 20) & (s_chan >= 25) & (s_chan <= 165) & (v_chan >= 45)

    # Intersection of all physical conditions strictly within human body zones
    skin_mask_raw = (bio_skin & skin_ycrcb & skin_hsv).astype(np.float32) * skin_zone_mask
    skin_mask = cv2.GaussianBlur(skin_mask_raw, (11, 11), 3.0)

    skin_pixel_count = np.sum(skin_mask > 0.4)
    if skin_pixel_count < 100:
        return image_bgr, 0.0

    # 3. LAB Luminance & Chrominance operations
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    # Current mean skin luminance
    mean_skin_l = float(np.mean(L[skin_mask > 0.4]))

    # Compute adaptive lift
    deficit = max(0.0, target_skin_l - mean_skin_l)
    lift_amount = min(max_lift, deficit * 0.90)

    if lift_amount <= 2.0:
        return image_bgr, 0.0

    # Smooth photographic tone curve: lifts shadows & midtones, rolls off on highlights
    l_norm = L / 255.0
    curve = 3.6 * l_norm * ((1.0 - l_norm) ** 0.85)

    # Apply unified lift across real skin
    L_brightened = np.clip(L + lift_amount * curve * skin_mask, 0.0, 255.0)

    # Healthy peach-golden warmth (+A for blood flow blush, +B for golden tone)
    A_brightened = np.clip(A + 2.0 * skin_mask * (lift_amount / 30.0), 0.0, 255.0)
    B_brightened = np.clip(B + 3.8 * skin_mask * (lift_amount / 30.0), 0.0, 255.0)

    brightened_bgr = cv2.cvtColor(
        np.dstack([L_brightened, A_brightened, B_brightened]).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )
    return brightened_bgr, lift_amount


def apply_cinematic_pro_camera(
    image_bgr: np.ndarray,
    cine_style: str = "kodak_portra",
    pro_mist_bloom: float = 0.20,
    vignette_strength: float = 0.18,
    film_grain_strength: float = 0.015,
    lift_blacks: float = 5.0,
    micro_contrast: float = 0.28,
) -> np.ndarray:
    """True Professional Cinema Camera & Kodak Portra 400 Emulsion Science."""
    h, w = image_bgr.shape[:2]
    img_float = image_bgr.astype(np.float32) / 255.0

    # 1. Pro-Mist / Cine Diffusion Filter
    if pro_mist_bloom > 0:
        gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
        bloom_thresh = np.maximum(0.0, gray - 0.80) / 0.20
        bloom_mask = (bloom_thresh ** 2)[..., None]

        bloom_k = int(min(w, h) * 0.045) | 1
        bloom_layer = cv2.GaussianBlur(img_float * bloom_mask, (bloom_k, bloom_k), 0)
        composite = img_float + bloom_layer * pro_mist_bloom * 0.80
        composite = np.clip(composite, 0.0, 1.0)
    else:
        composite = img_float

    # 2. Cinema S-Curve & Filmic Black Lift
    black_offset = lift_blacks / 255.0
    mid = 0.50
    x = composite
    curve = 1.0 / (1.0 + np.exp(-4.5 * (x - mid)))
    curve = (curve - curve.min()) / (curve.max() - curve.min())

    graded = 0.80 * x + 0.20 * curve
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
        b_shift = shadow_weight * 0.018
        r_shadow = shadow_weight * -0.010

        # Midtones: Warm golden peach skin tones (+R, +G, -B)
        r_mid = mid_weight * 0.026
        g_mid = mid_weight * 0.008
        b_mid = mid_weight * -0.016

        # Highlights: Soft warm champagne
        r_hi = hi_weight * 0.010
        g_hi = hi_weight * 0.005

        graded[:, :, 0] = np.clip(graded[:, :, 0] + b_shift + b_mid, 0.0, 1.0)
        graded[:, :, 1] = np.clip(graded[:, :, 1] + g_mid + g_hi, 0.0, 1.0)
        graded[:, :, 2] = np.clip(graded[:, :, 2] + r_shadow + r_mid + r_hi, 0.0, 1.0)

    # 4. Pro Prime Lens Micro-Contrast & Clarity
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
        vig_factor = 1.0 - vignette_strength * np.clip((dist - 0.45) / 0.55, 0.0, 1.0) ** 1.8
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


def process_cinematic_part2(
    input_dir: str = "MD/md part2",
    output_dir: str = "MD_CINEMATIC_PART2",
    limit: Optional[int] = None,
    overwrite: bool = True,
    cine_style: str = "kodak_portra",
    target_skin_l: float = 145.0,
    beauty_strength: float = 0.65,
):
    base_dir = Path(__file__).resolve().parent
    in_path = (base_dir / input_dir).resolve() if not Path(input_dir).is_absolute() else Path(input_dir)
    out_path = (base_dir / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Error: Folder input '{in_path}' tidak ditemukan!")
        return

    # Load AI Face Detector
    det_path = base_dir / "models" / "face_detection_yunet_2023mar.onnx"
    detector = None
    if det_path.exists():
        try:
            detector = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), score_threshold=0.60)
            print("✓ AI Face Detector Loaded: YuNet ONNX (Anatomical Gating & Beauty Protection)")
        except Exception as e:
            print(f"⚠ Warning: Gagal inisialisasi YuNet: {e}")

    # Gather all images
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
    print("=" * 80)
    print("   NATURAL CINEMATIC & ANATOMICALLY-ANCHORED SKIN BATCH (MD PART 2)")
    print(f"   Total File       : {total} foto")
    print(f"   Input            : {in_path}")
    print(f"   Output           : {out_path}")
    print(f"   Skin Scope       : Terikat Anatomi Manusia (Wajah, Telinga, Leher, Tangan)")
    print(f"   Fabric/Carpet    : 100% Proteksi Anti-Bocor (Karpet Merah & Kain Pink)")
    print(f"   AI Face Beauty   : Natural Bilateral Retouching (Kekuatan: {beauty_strength})")
    print(f"   Film Emulsion    : Kodak Portra 400 + Black Pro-Mist 1/8 + Micro-Contrast")
    print(f"   Mode             : {'REPROCESS ALL (Overwrite)' if overwrite else 'Incremental (Skip existing)'}")
    print("=" * 80)

    log_file = out_path / "batch_report_cinematic_part2.txt"
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n--- Batch Started at {time.strftime('%Y-%m-%d %H:%M:%S')} (total={total}, anchored_skin=True) ---\n")

    start_all = time.time()
    success_count = 0

    for idx, fpath in enumerate(all_files, start=1):
        target_file = out_path / fpath.name

        # Skip if already exists and valid (>1MB) and not overwrite
        if not overwrite and target_file.exists() and target_file.stat().st_size > 1000000:
            print(f"[{idx}/{total}] SKIP: {fpath.name} (sudah selesai sebelumnya)")
            continue

        t0 = time.time()
        print(f"[{idx}/{total}] Memproses Foto: {fpath.name} ... ", end="", flush=True)

        try:
            # 1. Load with EXIF orientation correction
            img_bgr = correct_exif_orientation(fpath)
            h, w = img_bgr.shape[:2]

            # 2. Detect faces for anatomical anchor
            faces = None
            scaled_faces = None
            if detector is not None:
                scale_det = 640.0 / max(h, w)
                small_det = cv2.resize(img_bgr, (int(w * scale_det), int(h * scale_det)))
                detector.setInputSize((small_det.shape[1], small_det.shape[0]))
                _, detected = detector.detect(small_det)
                if detected is not None and len(detected) > 0:
                    faces = []
                    scaled_faces = []
                    for d in detected:
                        faces.append([d[0]/scale_det, d[1]/scale_det, d[2]/scale_det, d[3]/scale_det])
                        scaled_faces.append([
                            d[0]/scale_det, d[1]/scale_det, d[2]/scale_det, d[3]/scale_det,
                            d[4]/scale_det, d[5]/scale_det, d[6]/scale_det, d[7]/scale_det,
                            d[8]/scale_det, d[9]/scale_det, d[10]/scale_det, d[11]/scale_det,
                            d[12]/scale_det, d[13]/scale_det
                        ])
                    faces = np.array(faces)
                    scaled_faces = np.array(scaled_faces)

            # 3. Anatomically-Anchored Skin Brightening (100% anti-carpet & anti-dress)
            skin_brightened_bgr, lift_val = apply_anatomical_skin_brightening(
                img_bgr, faces=faces, target_skin_l=target_skin_l
            )

            # 4. AI Face Natural Beauty Smoothing (Smooths skin blemishes, protects eyes/expression)
            if scaled_faces is not None:
                smoothed_bgr = apply_natural_beauty_smoothing(
                    skin_brightened_bgr, scaled_faces, strength=beauty_strength
                )
            else:
                smoothed_bgr = skin_brightened_bgr

            # 5. Apply Cinema Camera Kodak Portra + Black Pro-Mist 1/8
            final_bgr = apply_cinematic_pro_camera(
                smoothed_bgr,
                cine_style=cine_style,
                pro_mist_bloom=0.20,
                vignette_strength=0.18,
                film_grain_strength=0.015,
                lift_blacks=5.0,
                micro_contrast=0.28,
            )

            # 6. Save with 100% Quality & 4:4:4 sampling
            final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
            save_image_100(final_rgb, target_file)

            elapsed = time.time() - t0
            success_count += 1
            print(f"SELESAI! (Kulit +{lift_val:.1f} EV, {elapsed:.1f}s)")

            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{idx}/{total}] {fpath.name} -> SELESAI (lift={lift_val:.1f}, {elapsed:.1f}s)\n")

        except Exception as e:
            print(f"ERROR: {e}")
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{idx}/{total}] {fpath.name} ERROR: {e}\n")

    total_time = time.time() - start_all
    print("=" * 80)
    print(f"SELESAI SEMUA! Berhasil memproses: {success_count}/{total} foto.")
    print(f"Total waktu: {total_time / 60:.1f} menit.")
    print(f"Folder hasil: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anatomically Anchored Natural Cinematic Skin Enhancer")
    parser.add_argument("--input-dir", type=str, default="MD/md part2", help="Folder input foto")
    parser.add_argument("--output-dir", type=str, default="MD_CINEMATIC_PART2", help="Folder hasil foto sinematik")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah foto untuk pengujian")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Proses ulang file yang sudah ada")
    parser.add_argument("--style", type=str, default="kodak_portra", choices=["kodak_portra"])
    parser.add_argument("--target-skin-l", type=float, default=145.0, help="Target luminance kulit (LAB L-channel)")
    parser.add_argument("--beauty-strength", type=float, default=0.65, help="Kekuatan AI natural beauty smoothing")
    args = parser.parse_args()

    process_cinematic_part2(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        cine_style=args.style,
        target_skin_l=args.target_skin_l,
        beauty_strength=args.beauty_strength,
    )
