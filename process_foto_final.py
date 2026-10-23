import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Import existing enhancement pipeline
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

from image_enhancer.enhancers import (
    load_image,
    apply_non_local_means_denoising,
    enhance_contrast_local,
    enhance_saturation,
    adaptive_sharpening,
)

def process_image(
    input_path: Path,
    output_path: Path,
    min_width: int = 2624,
    min_height: int = 3936,
    target_dpi: int = 300,
):
    print(f"Loading image from: {input_path}")
    img_rgb = load_image(input_path)
    if img_rgb is None:
        raise ValueError(f"Could not load image: {input_path}")

    orig_h, orig_w = img_rgb.shape[:2]
    print(f"Original resolution: {orig_w} x {orig_h}")

    # Proportional scaling to ensure both min_width and min_height are satisfied
    scale_w = min_width / orig_w
    scale_h = min_height / orig_h
    scale = max(scale_w, scale_h)

    target_w = int(round(orig_w * scale))
    target_h = int(round(orig_h * scale))
    print(f"Target resolution (proportional, no crop): {target_w} x {target_h} (scale: {scale:.4f}x)")

    # 1. High-quality Lanczos4 Upscale
    print("Applying Lanczos-4 Super-Resolution upscale...")
    upscaled = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # 2. Denoising to eliminate compression artifacts while protecting fine texture
    print("Applying edge & texture-preserving denoising...")
    denoised = apply_non_local_means_denoising(upscaled, {"denoise_strength": 1.2})

    # 3. Local contrast & natural illumination
    print("Applying subtle local contrast & illumination enhancement...")
    contrasted = enhance_contrast_local(denoised, {"contrast_boost": 1.05})

    # 4. Vibrance & natural color
    print("Applying natural saturation vibrance...")
    saturated = enhance_saturation(contrasted, {"saturation_boost": 1.04})

    # 5. Adaptive High-Pass Sharpening (de-halo)
    print("Applying adaptive high-pass sharpening...")
    sharpened = adaptive_sharpening(saturated, {"sharpening_strength": 1.15})

    # 6. Save as High-Res JPEG with 300 DPI
    print(f"Saving high-resolution JPEG with {target_dpi} DPI to: {output_path}")
    pil_img = Image.fromarray(sharpened)

    # Ensure file size is >= 1MB
    quality = 95
    pil_img.save(
        output_path,
        format="JPEG",
        dpi=(target_dpi, target_dpi),
        quality=quality,
        subsampling=0, # 4:4:4 high quality chroma
        optimize=True,
    )

    file_size = output_path.stat().st_size
    print(f"Saved: {output_path.name}")
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"Dimensions: {target_w} x {target_h}")
    print(f"DPI: {pil_img.info.get('dpi', (target_dpi, target_dpi))}")

    if file_size < 1024 * 1024:
        print("Warning: File size under 1MB. Increasing quality...")
        pil_img.save(
            output_path,
            format="JPEG",
            dpi=(target_dpi, target_dpi),
            quality=98,
            subsampling=0,
            optimize=False,
        )
        file_size = output_path.stat().st_size
        print(f"New file size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

    return {
        "output_path": str(output_path),
        "dimensions": (target_w, target_h),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024*1024), 2),
        "dpi": target_dpi,
    }

if __name__ == "__main__":
    input_file = Path("/Users/mac/Desktop/PROJECT/htdocs/foto final/IMG_1051.WEBP")
    output_file = Path("/Users/mac/Desktop/PROJECT/htdocs/foto final/IMG_1051.JPG")

    result = process_image(
        input_path=input_file,
        output_path=output_file,
        min_width=2624,
        min_height=3936,
        target_dpi=300,
    )
    print("Process completed successfully:", result)
