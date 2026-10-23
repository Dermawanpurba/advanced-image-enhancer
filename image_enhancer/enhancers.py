from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from image_enhancer.config import EnhancementConfig


# ── EXIF Orientation ─────────────────────────────────────────────────────

def correct_exif_orientation(image: np.ndarray, image_path: Path) -> np.ndarray:
    try:
        from PIL import Image as PILImage
        with PILImage.open(image_path) as pil_img:
            exif = pil_img._getexif()
            if exif is not None:
                orientation = exif.get(0x0112, 1)
                if orientation == 3:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif orientation == 6:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif orientation == 8:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return image


# ── Load / Save ──────────────────────────────────────────────────────────

def load_image(image_path: Path) -> Optional[np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = correct_exif_orientation(image, image_path)
    return image


def save_image(image: np.ndarray, output_path: Path | str) -> bool:
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ext = output_path.suffix.lower()
        params: list[int] = []
        if ext in (".jpg", ".jpeg"):
            params = [
                cv2.IMWRITE_JPEG_QUALITY, 100,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            ]
        elif ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, 100]
        cv2.imwrite(str(output_path), image_bgr, params)
        return True
    except Exception:
        return False


# ── Resize ───────────────────────────────────────────────────────────────

def resize_to_target(
    image: np.ndarray,
    target_size: Tuple[int, int],
    fit_mode: str = "stretch",
) -> np.ndarray:
    height, width = image.shape[:2]
    target_width, target_height = target_size

    if (width, height) == (target_width, target_height):
        return image

    if fit_mode == "stretch":
        interp = cv2.INTER_LANCZOS4 if (width < target_width or height < target_height) else cv2.INTER_AREA
        return cv2.resize(image, (target_width, target_height), interpolation=interp)

    if fit_mode == "crop":
        scale = max(target_width / width, target_height / height)
        new_w, new_h = int(width * scale), int(height * scale)
        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        start_x = (new_w - target_width) // 2
        start_y = (new_h - target_height) // 2
        return resized[start_y:start_y + target_height, start_x:start_x + target_width]

    if fit_mode == "pad":
        scale = min(target_width / width, target_height / height)
        new_w, new_h = int(width * scale), int(height * scale)
        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        start_x = (target_width - new_w) // 2
        start_y = (target_height - new_h) // 2
        result[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        return result

    return image


# ── Denoising (Edge & Texture-Preserving) ─────────────────────────────────

def apply_non_local_means_denoising(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    strength = float(params.get("denoise_strength", 0.0))
    if strength <= 0.0:
        return image

    is_color = (image.ndim == 3 and image.shape[2] == 3)
    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        target_channel = l
    else:
        target_channel = image

    # Adaptive detail preservation mask (Sobel gradient magnitude)
    grad_x = cv2.Sobel(target_channel, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(target_channel, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    detail_mask = np.clip(mag / 20.0, 0.0, 1.0)

    # Controlled filter strength h to protect textures and facial details
    h = max(1.0, min(6.0, strength * 1.2))
    denoised_channel = cv2.fastNlMeansDenoising(target_channel, None, h=h, templateWindowSize=7, searchWindowSize=15)

    # In high-detail / edge regions, preserve original; denoise only flat regions
    clean_channel = (denoised_channel.astype(np.float32) * (1.0 - detail_mask) + target_channel.astype(np.float32) * detail_mask).astype(np.uint8)

    if is_color:
        # Chroma noise reduction: softly filter color channels without losing luminance sharpness
        a_clean = cv2.GaussianBlur(a, (3, 3), 0.5)
        b_clean = cv2.GaussianBlur(b, (3, 3), 0.5)
        lab_clean = cv2.merge([clean_channel, a_clean, b_clean])
        return cv2.cvtColor(lab_clean, cv2.COLOR_LAB2RGB)
    else:
        return clean_channel


# ── Contrast & Natural Face/Midtone Illumination ─────────────────────────

# ── Contrast & Natural Face/Midtone Illumination ─────────────────────────

def enhance_contrast_local(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    contrast_boost = float(params.get("contrast_boost", 1.0))
    if contrast_boost <= 1.0:
        return image

    is_color = (image.ndim == 3 and image.shape[2] == 3)
    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        target_channel = l
    else:
        target_channel = image

    # Adaptive CLAHE with responsive clip limit
    clip_limit = max(1.0, 1.0 + (contrast_boost - 1.0) * 5.0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_l = clahe.apply(target_channel)

    # Dynamic midtone S-curve lift
    u = target_channel.astype(np.float32) / 255.0
    lift = (contrast_boost - 1.0) * 0.35 * np.sin(np.pi * u)
    lifted = np.clip((u + lift) * 255.0, 0, 255)

    # Blend ratio proportional to slider
    weight = min(0.60, max(0.15, (contrast_boost - 1.0) * 1.5 + 0.15))
    final_l = np.clip(lifted * (1.0 - weight) + clahe_l.astype(np.float32) * weight, 0, 255).astype(np.uint8)

    if is_color:
        return cv2.cvtColor(cv2.merge([final_l, a, b]), cv2.COLOR_LAB2RGB)
    else:
        return final_l


# ── Saturation (Vibrance & Color Pop) ────────────────────────────────────

def enhance_saturation(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    saturation_factor = float(params.get("saturation_boost", 1.0))
    if saturation_factor <= 1.0 or image.ndim != 3 or image.shape[2] != 3:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Responsive vibrance curve: boost colors visibly
    s_boosted = np.clip(s * saturation_factor, 0, 255)
    hsv_enhanced = cv2.merge([h, s_boosted, v]).astype(np.uint8)
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)


# ── Sharpening (High-Pass Unsharp Masking with Crisp Micro-Contrast) ─────

def adaptive_sharpening(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    strength = float(params.get("sharpening_strength", 1.0))
    if strength <= 1.0:
        return image

    is_color = (image.ndim == 3 and image.shape[2] == 3)
    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        target_channel = l
    else:
        target_channel = image

    # High-pass unsharp mask
    blurred = cv2.GaussianBlur(target_channel, (0, 0), sigmaX=1.2)
    diff = target_channel.astype(np.float32) - blurred.astype(np.float32)

    # Proportional crispness factor: scales smoothly from 1.0 up to 2.5
    scale_factor = (strength - 1.0) * 3.5
    sharpened_channel = np.clip(target_channel.astype(np.float32) + diff * scale_factor, 0, 255).astype(np.uint8)

    if is_color:
        lab_sharp = cv2.merge([sharpened_channel, a, b])
        return cv2.cvtColor(lab_sharp, cv2.COLOR_LAB2RGB)
    else:
        return sharpened_channel



# ── White Balance (Balanced Auto Color Correction) ───────────────────────

def auto_white_balance(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        return image

    avg_r = float(np.mean(image[:, :, 0]))
    avg_g = float(np.mean(image[:, :, 1]))
    avg_b = float(np.mean(image[:, :, 2]))
    if min(avg_r, avg_g, avg_b) == 0:
        return image

    target = (avg_r + avg_g + avg_b) / 3.0
    scales = np.array([target / avg_r, target / avg_g, target / avg_b])
    balanced = image.astype(np.float32) * scales.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


# ── Low Light (Luminance Adaptive Curve) ──────────────────────────────────

def enhance_low_light(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    mean_brightness = float(np.mean(gray))
    gamma = max(0.3, min(2.0, 128.0 / max(mean_brightness, 1.0)))
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


# ── Super Resolution (Fine Detail Enhancement) ───────────────────────────

def apply_super_resolution_effect(image: np.ndarray, config: EnhancementConfig) -> np.ndarray:
    h, w = image.shape[:2]
    upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    # Subtle high-pass crispness at 2x before downsampling
    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.2)
    diff = np.clip(upscaled.astype(np.float32) - blurred.astype(np.float32), -20.0, 20.0)
    upscaled = np.clip(upscaled.astype(np.float32) + diff * 0.35, 0, 255).astype(np.uint8)
    downscaled = cv2.resize(upscaled, config.target_size, interpolation=cv2.INTER_AREA)
    return downscaled
