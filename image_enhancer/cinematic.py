from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def build_full_body_subject_mask(image_bgr: np.ndarray, faces: Optional[np.ndarray]) -> np.ndarray:
    """
    Constructs a full-body subject protection mask:
    Ensures the entire body (hair, face, torso, dress/suit, legs, feet, shoes)
    and the ground plane where subjects stand remain 100% pin-sharp in the focal plane.
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    if faces is None or len(faces) == 0:
        # Fallback: central portrait focal plane
        cv2.rectangle(mask, (int(w * 0.15), int(h * 0.10)), (int(w * 0.85), h), 1.0, -1)
        ksize = int(min(w, h) * 0.08) | 1
        return cv2.GaussianBlur(mask, (ksize, ksize), 0)

    # Filter out tiny background wall specks (e.g. wall paintings or distant decorations)
    valid_faces = [
        f for f in faces
        if f[2] >= 45 and f[3] >= 45 and f[1] > h * 0.12
    ]

    if not valid_faces:
        valid_faces = [f for f in faces if f[2] >= 35]

    if not valid_faces:
        cv2.rectangle(mask, (int(w * 0.15), int(h * 0.10)), (int(w * 0.85), h), 1.0, -1)
        ksize = int(min(w, h) * 0.08) | 1
        return cv2.GaussianBlur(mask, (ksize, ksize), 0)

    min_x = w
    max_x = 0
    min_y = h

    for f in valid_faces:
        fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        head_top = max(0, int(fy - fh * 0.40))
        feet_bottom = min(h, int(fy + fh * 7.5))

        body_half_w = int(fw * 1.5)
        body_left = max(0, fx + fw // 2 - body_half_w)
        body_right = min(w, fx + fw // 2 + body_half_w)

        min_x = min(min_x, body_left)
        max_x = max(max_x, body_right)
        min_y = min(min_y, head_top)

        # Head zone
        center_head = (fx + fw // 2, fy + fh // 2)
        cv2.circle(mask, center_head, int(fw * 0.75), 1.0, -1)

        # Torso & Legs polygon extending to feet/floor
        poly = np.array([
            [body_left, int(fy + fh * 0.8)],
            [body_right, int(fy + fh * 0.8)],
            [min(w, body_right + int(fw * 0.4)), feet_bottom],
            [max(0, body_left - int(fw * 0.4)), feet_bottom]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 1.0)

    # Bridge subjects standing side-by-side in group shots
    if len(valid_faces) > 1:
        group_margin = int((max_x - min_x) * 0.04)
        g_left = max(0, min_x - group_margin)
        g_right = min(w, max_x + group_margin)
        cv2.rectangle(mask, (g_left, int(min_y + 80)), (g_right, h), 1.0, -1)

    # Slight dilation to cover hair flyaways, fingers, accessories
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Smooth optical transition (subtle depth falloff)
    ksize = int(min(w, h) * 0.05) | 1
    feathered_mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)

    return np.clip(feathered_mask, 0.0, 1.0)


def apply_cinematic_pro_camera(
    image_bgr: np.ndarray,
    cine_style: str = "kodak_portra",
    lens_bokeh_strength: float = 0.35,
    pro_mist_bloom: float = 0.22,
    vignette_strength: float = 0.22,
    film_grain_strength: float = 0.018,
    lift_blacks: float = 6.0,
    detector: Optional[cv2.FaceDetectorYN] = None
) -> np.ndarray:
    """
    Applies professional cinema camera optics & film emulation:
    1. Utuh 1 Tubuh (Full Body in Focus): Hair down to feet & shoes remain 100% sharp.
    2. Subtle Background Rolloff: Optical depth blur on distant background only.
    3. Black Pro-Mist 1/8 Diffusion: Romantic, gentle glow on lamps, chandeliers, candles.
    4. Filmic S-Curve Tone: Lifted velvety blacks and soft highlight compression.
    5. Kodak Portra / Cinema Color Grading: Radiant, warm golden-peach skin tones.
    6. Micro-Contrast Sharpness: Prime lens clarity on fabric weaves, lace, and jewelry.
    7. Optical Vignette: Natural fast-aperture corner falloff.
    8. 35mm Micro Film Grain: Organic texture eliminating digital sterility.
    """
    h, w = image_bgr.shape[:2]
    img_float = image_bgr.astype(np.float32) / 255.0

    # 1. Detect faces for full body protection
    faces = None
    if detector is not None:
        scale = 640.0 / max(h, w)
        small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        detector.setInputSize((small.shape[1], small.shape[0]))
        _, detected = detector.detect(small)
        if detected is not None and len(detected) > 0:
            faces = []
            for d in detected:
                faces.append([d[0] / scale, d[1] / scale, d[2] / scale, d[3] / scale])
            faces = np.array(faces)

    subject_mask = build_full_body_subject_mask(image_bgr, faces)
    subject_mask_3d = subject_mask[..., None]

    # 2. Optical Background Bokeh (Background only, subjects 100% sharp)
    if lens_bokeh_strength > 0:
        bokeh_k = int(min(w, h) * 0.016 * lens_bokeh_strength) | 1
        bokeh_k = max(5, bokeh_k)
        bg_blur = cv2.GaussianBlur(img_float, (bokeh_k, bokeh_k), bokeh_k / 2.5)
        composite = img_float * subject_mask_3d + bg_blur * (1.0 - subject_mask_3d)
    else:
        composite = img_float

    # 3. Pro-Mist / Cine Diffusion Filter
    if pro_mist_bloom > 0:
        gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
        bloom_thresh = np.maximum(0.0, gray - 0.78) / 0.22
        bloom_mask = (bloom_thresh ** 2)[..., None]

        bloom_k = int(min(w, h) * 0.05) | 1
        bloom_layer = cv2.GaussianBlur(composite * bloom_mask, (bloom_k, bloom_k), 0)
        composite = composite + bloom_layer * pro_mist_bloom * 0.85
        composite = np.clip(composite, 0.0, 1.0)

    # 4. Cinema S-Curve & Filmic Black Lift
    black_offset = lift_blacks / 255.0
    mid = 0.50
    x = composite
    curve = 1.0 / (1.0 + np.exp(-4.5 * (x - mid)))
    curve = (curve - curve.min()) / (curve.max() - curve.min())

    graded = 0.78 * x + 0.22 * curve
    graded = black_offset + graded * (0.98 - black_offset)
    graded = np.clip(graded, 0.0, 1.0)

    # 5. Cinematic Split-Toning & Color Science
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

    # 6. Pro Lens Micro-Contrast & Detail Clarity on Subjects
    blur_detail = cv2.GaussianBlur(graded, (0, 0), 2.0)
    high_pass = graded - blur_detail
    graded = np.clip(graded + high_pass * 0.25 * subject_mask_3d, 0.0, 1.0)

    # 7. Optical Lens Vignette
    if vignette_strength > 0:
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w / 2.0, h / 2.0
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = np.sqrt(((X - center_x) * 1.0)**2 + ((Y - center_y) * 1.1)**2) / max_dist
        vig_factor = 1.0 - vignette_strength * np.clip((dist - 0.42) / 0.58, 0.0, 1.0) ** 1.8
        graded = graded * vig_factor[..., None]

    # 8. Organic 35mm Micro Film Grain
    if film_grain_strength > 0:
        np.random.seed(123)
        grain = np.random.normal(0.0, 1.0, (h, w)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (3, 3), 0.6)
        grain_mask = np.clip(1.0 - 4.0 * (lum - 0.5)**2, 0.05, 1.0)
        grain_layer = grain * grain_mask * film_grain_strength
        graded = np.clip(graded + grain_layer[..., None], 0.0, 1.0)

    return (graded * 255.0).astype(np.uint8)
