from __future__ import annotations

import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_REMBG_SESSION = None

def get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            import rembg
            _REMBG_SESSION = rembg.new_session("u2net")
        except Exception as e:
            logger.error(f"Failed to initialize rembg session: {e}")
            return None
    return _REMBG_SESSION

def remove_background(image_rgb: np.ndarray) -> np.ndarray:
    """
    Removes background using local AI (u2net / rembg).
    Input: RGB image (H, W, 3)
    Output: RGBA image (H, W, 4) with transparent alpha channel
    """
    try:
        import rembg
        session = get_rembg_session()
        if session:
            rgba = rembg.remove(image_rgb, session=session)
        else:
            rgba = rembg.remove(image_rgb)
        return rgba
    except Exception as e:
        logger.error(f"Error during background removal: {e}")
        # Fallback: return image with full opacity alpha
        h, w = image_rgb.shape[:2]
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        return np.concatenate([image_rgb, alpha], axis=2)

def replace_background_studio(
    image_rgb: np.ndarray,
    preset: str = "studio_gray",
    custom_color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Replaces background with professional studio backdrops:
    Presets:
    - 'studio_gray': Subtle radial gradient studio backdrop
    - 'clean_white': Pure e-commerce high-key white
    - 'executive_dark': Sleek charcoal dark mode gradient
    - 'passport_red': Official passport / ID red (#D32F2F)
    - 'passport_blue': Official passport / ID blue (#1976D2)
    """
    rgba = remove_background(image_rgb)
    h, w = image_rgb.shape[:2]
    rgb = rgba[:, :, :3]
    alpha = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

    # Generate background
    if preset == "clean_white":
        bg = np.full((h, w, 3), 255, dtype=np.uint8)
    elif preset == "passport_red":
        bg = np.full((h, w, 3), [211, 47, 47], dtype=np.uint8)
    elif preset == "passport_blue":
        bg = np.full((h, w, 3), [25, 118, 210], dtype=np.uint8)
    elif preset == "executive_dark":
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h * 0.4, w * 0.5
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        norm_dist = np.clip(dist / max_dist, 0.0, 1.0)
        # Gradient from #2C3440 (center) to #12151B (edges)
        val = (44 - norm_dist * 26).astype(np.uint8)
        bg = np.repeat(val[:, :, np.newaxis], 3, axis=2)
    elif custom_color is not None:
        bg = np.full((h, w, 3), custom_color, dtype=np.uint8)
    else: # studio_gray
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h * 0.45, w * 0.5
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        norm_dist = np.clip(dist / max_dist, 0.0, 1.0)
        # Gradient from #EAEAEA (center) to #B8B8B8 (edges)
        val = (235 - norm_dist * 50).astype(np.uint8)
        bg = np.repeat(val[:, :, np.newaxis], 3, axis=2)

    # Seamless alpha blending
    composite = (rgb.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return composite
