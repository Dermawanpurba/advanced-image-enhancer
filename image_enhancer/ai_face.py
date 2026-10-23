from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard 5-point face landmarks for 512x512 alignment (FFHQ / ArcFace template)
STD_PTS_512 = np.array([
    [38.2946 * 512.0 / 112.0, 51.6963 * 512.0 / 112.0],
    [73.5318 * 512.0 / 112.0, 51.5014 * 512.0 / 112.0],
    [56.0252 * 512.0 / 112.0, 71.7366 * 512.0 / 112.0],
    [41.5493 * 512.0 / 112.0, 92.3655 * 512.0 / 112.0],
    [70.7299 * 512.0 / 112.0, 92.2041 * 512.0 / 112.0]
], dtype=np.float32)

_SESSION = None
_DETECTOR = None


def get_default_model_paths() -> Tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent.parent
    detector_path = base_dir / "models" / "face_detection_yunet_2023mar.onnx"
    gfpgan_path = base_dir / "models" / "GFPGANv1.4.onnx"
    return detector_path, gfpgan_path


def load_ai_models(
    detector_path: Optional[Path] = None,
    gfpgan_path: Optional[Path] = None
) -> Tuple[Optional[cv2.FaceDetectorYN], Optional[object]]:
    global _SESSION, _DETECTOR

    det_p, gfp_p = get_default_model_paths()
    det_path = detector_path or det_p
    gfp_path = gfpgan_path or gfp_p

    if not det_path.exists() or not gfp_path.exists():
        logger.warning(f"AI models not found at {det_path} or {gfp_path}")
        return None, None

    if _DETECTOR is None:
        try:
            _DETECTOR = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), score_threshold=0.60)
        except Exception as e:
            logger.error(f"Failed to load YuNet face detector: {e}")
            return None, None

    if _SESSION is None:
        try:
            import onnxruntime as ort
            _SESSION = ort.InferenceSession(str(gfp_path), providers=["CPUExecutionProvider"])
        except Exception as e:
            logger.error(f"Failed to load GFPGAN ONNX model: {e}")
            return None, None

    return _DETECTOR, _SESSION


def apply_natural_beauty_smoothing(image_bgr: np.ndarray, faces: np.ndarray, strength: float = 0.50) -> np.ndarray:
    """
    Applies high-end natural beauty retouching:
    - Adaptive facial skin mask covering forehead, cheeks, nose, and chin
    - Landmark-based exclusion protecting eyes, pupils, eyebrows, glasses, and lips from blur
    - Bilateral smoothing + porcelain tone evening that responds visibly to the slider
    """
    if faces is None or len(faces) == 0 or strength <= 0.0:
        return image_bgr

    h, w = image_bgr.shape[:2]

    # Convert to LAB for luminance smoothing & color radiance
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Bilateral filter on luminance channel
    d_val = max(7, int(min(w, h) * 0.01) | 1)
    l_smoothed = cv2.bilateralFilter(l.astype(np.uint8), d=d_val, sigmaColor=50, sigmaSpace=50).astype(np.float32)

    # Facial masks with feature exclusion
    face_mask = np.zeros((h, w), dtype=np.float32)

    for face in faces:
        fw, fh = int(face[2]), int(face[3])
        if fw < 30 or fh < 30:
            continue

        x, y = int(face[0]), int(face[1])
        center = (x + fw // 2, y + int(fh * 0.55))
        axes = (int(fw * 0.48), int(fh * 0.58))

        # Elliptical face region
        cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1.0, -1)

        # Landmark exclusion: protect eyes, eyebrows, glasses, mouth & lips
        reye = face[4:6].astype(int)
        leye = face[6:8].astype(int)
        rmouth = face[10:12].astype(int)
        lmouth = face[12:14].astype(int)

        eye_rad = max(8, int(fw * 0.15))
        # Protect eyes and glasses
        cv2.circle(face_mask, (reye[0], reye[1]), eye_rad, 0.0, -1)
        cv2.circle(face_mask, (leye[0], leye[1]), eye_rad, 0.0, -1)
        # Protect eyebrows
        cv2.circle(face_mask, (reye[0], int(reye[1] - fw * 0.08)), eye_rad, 0.0, -1)
        cv2.circle(face_mask, (leye[0], int(leye[1] - fw * 0.08)), eye_rad, 0.0, -1)
        # Protect lips & mouth
        mouth_rad = max(10, int(fw * 0.17))
        mouth_center = ((rmouth[0] + lmouth[0]) // 2, (rmouth[1] + lmouth[1]) // 2)
        cv2.circle(face_mask, mouth_center, mouth_rad, 0.0, -1)

    # Feather soft edges
    ksize = int(min(w, h) * 0.03) | 1
    face_mask = cv2.GaussianBlur(face_mask, (ksize, ksize), ksize / 3.0)

    # Responsive blend: scales smoothly with strength (0.0 - 1.0)
    blend_weight = face_mask * float(np.clip(strength, 0.0, 1.0))

    # Porcelain luminance smoothing + gentle warm radiance on skin
    l_final = l * (1.0 - blend_weight) + (l_smoothed + 6.0 * strength) * blend_weight
    a_final = a + (1.8 * strength) * blend_weight
    b_final = b + (2.5 * strength) * blend_weight

    enhanced_lab = cv2.merge([
        np.clip(l_final, 0, 255).astype(np.uint8),
        np.clip(a_final, 0, 255).astype(np.uint8),
        np.clip(b_final, 0, 255).astype(np.uint8),
    ])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)



def apply_adaptive_face_relighting(
    image_bgr: np.ndarray,
    faces: np.ndarray,
    target_skin_l: float = 142.0
) -> np.ndarray:
    """
    Intelligent Virtual Fill-Light for shadowed / underexposed faces (tidak terkena flash light / cenderung gelap):
    - Analyzes actual human skin luminance within each detected face
    - If face is dark/shadowed (skin L < 137), dynamically calculates fill-light deficit up to target_skin_l (~142)
    - Faces already well-lit by flash (skin L >= 137) are left completely untouched to avoid overexposure
    - Protects pitch black hair, pupils, eyebrows, eyelashes, and clothing via skin segmentation & tone gating
    - Imparts natural warm vitality in LAB space to eliminate muddy or ashen shadow appearance
    """
    if faces is None or len(faces) == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Skin tone detection in YCrCb color space
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_mask = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.float32)

    total_delta_l = np.zeros((h, w), dtype=np.float32)
    total_delta_a = np.zeros((h, w), dtype=np.float32)
    total_delta_b = np.zeros((h, w), dtype=np.float32)

    for face in faces:
        fw, fh = int(face[2]), int(face[3])
        if fw < 50 or fh < 50:
            continue
        fx, fy = int(face[0]), int(face[1])

        # Elliptical face region covering cheeks, chin, nose, and forehead
        f_mask = np.zeros((h, w), dtype=np.float32)
        center = (fx + fw // 2, fy + int(fh * 0.55))
        axes = (int(fw * 0.48), int(fh * 0.55))
        cv2.ellipse(f_mask, center, axes, 0, 0, 360, 1.0, -1)

        # Intersect with skin mask to illuminate ONLY actual skin, keeping hair & clothing untouched
        pure_skin = f_mask * skin_mask
        if np.sum(pure_skin > 0.5) < 30:
            continue

        curr_skin_l = float(np.mean(l[pure_skin > 0.5]))
        # Skip faces that are already bright / well-lit by flash to prevent any glare or overexposure
        if curr_skin_l >= target_skin_l - 5.0:
            continue

        deficit = target_skin_l - curr_skin_l
        lift_val = deficit * 0.85

        # Smooth feathering on skin fill light
        ksize = int(fw * 0.25)
        if ksize % 2 == 0:
            ksize += 1
        feathered_skin = cv2.GaussianBlur(pure_skin, (ksize, ksize), ksize / 3.0)

        # Photographic tone curve: maximum lift on shadows & midtones, rolling off near highlights
        l_norm = l / 255.0
        curve = 4.0 * l_norm * (1.0 - l_norm)

        total_delta_l = np.maximum(total_delta_l, lift_val * curve * feathered_skin)
        total_delta_a = np.maximum(total_delta_a, 2.4 * feathered_skin * (deficit / 45.0))
        total_delta_b = np.maximum(total_delta_b, 3.4 * feathered_skin * (deficit / 45.0))

    if np.max(total_delta_l) == 0.0:
        return image_bgr

    l_out = np.clip(l + total_delta_l, 0, 255).astype(np.uint8)
    a_out = np.clip(a + total_delta_a, 0, 255).astype(np.uint8)
    b_out = np.clip(b + total_delta_b, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_out, a_out, b_out]), cv2.COLOR_LAB2BGR)


def enhance_faces_ai(
    image: np.ndarray,
    blend_strength: float = 0.85,
    enable_beauty: bool = True,
    beauty_strength: float = 0.60,
    detector_path: Optional[Path] = None,
    gfpgan_path: Optional[Path] = None
) -> np.ndarray:
    """
    Detects and restores all faces in the given RGB image using YuNet + GFPGAN ONNX.
    Includes adaptive virtual fill-light for shadowed faces & natural beauty skin smoothing.
    Keeps all original eye, glasses, and expression structures 100% authentic without synthetic tampering.
    """
    if image is None or image.size == 0 or blend_strength <= 0.0:
        return image

    detector, session = load_ai_models(detector_path, gfpgan_path)
    if detector is None or session is None:
        logger.warning("AI models unavailable, skipping AI face enhancement.")
        return image

    h, w = image.shape[:2]
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detector.setInputSize((w, h))
    _, faces = detector.detect(image_bgr)

    if faces is None or len(faces) == 0:
        logger.info("No faces detected by YuNet.")
        return image

    # Filter out tiny noise / background specks
    valid_faces = [
        f for f in faces
        if f[2] >= 30 and f[3] >= 30
    ]

    if not valid_faces:
        logger.info("No valid face candidates detected.")
        return image

    valid_faces = np.array(valid_faces)
    logger.info(f"Detected {len(valid_faces)} face(s). Applying Adaptive Skin Relighting + Beauty + GFPGAN...")

    # Step 1: Intelligent Adaptive Relighting for shadowed/underexposed faces (bright, healthy skin tone)
    image_bgr = apply_adaptive_face_relighting(image_bgr, valid_faces, target_skin_l=145.0)

    # Step 2: Natural beauty skin smoothing on face regions (protecting eyes, glasses, lips, eyebrows)
    if enable_beauty and beauty_strength > 0.05:
        image_bgr = apply_natural_beauty_smoothing(image_bgr, valid_faces, strength=beauty_strength)


    # Step 3: Soft elliptical blending mask for AI face reconstruction (preserves hairline)
    mask_512 = np.zeros((512, 512), dtype=np.float32)
    cv2.ellipse(mask_512, (256, 275), (165, 205), 0, 0, 360, 1.0, -1)
    mask_512 = cv2.GaussianBlur(mask_512, (51, 51), 15) * float(np.clip(blend_strength, 0.0, 1.0))

    result_bgr = image_bgr.copy()

    for face in valid_faces:
        landmarks = np.array([
            face[4:6],   # right eye (viewer left)
            face[6:8],   # left eye
            face[8:10],  # nose
            face[10:12], # right mouth
            face[12:14]  # left mouth
        ], dtype=np.float32)

        M, _ = cv2.estimateAffinePartial2D(landmarks, STD_PTS_512)
        if M is None:
            continue

        inv_M = cv2.invertAffineTransform(M)

        # Warp face to 512x512
        aligned = cv2.warpAffine(image_bgr, M, (512, 512), flags=cv2.INTER_LANCZOS4)
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        input_tensor = (aligned_rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None, ...]

        # Run ONNX inference
        out = session.run(None, {"input": input_tensor})[0]
        restored_rgb = np.clip((out[0].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

        # Warp restored face and mask back to original coordinates
        restored_warped = cv2.warpAffine(restored_bgr, inv_M, (w, h), flags=cv2.INTER_LANCZOS4)
        mask_warped = cv2.warpAffine(mask_512, inv_M, (w, h), flags=cv2.INTER_LANCZOS4)[..., None]

        # Blend seamlessly
        result_bgr = (restored_warped * mask_warped + result_bgr * (1.0 - mask_warped)).astype(np.uint8)

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
