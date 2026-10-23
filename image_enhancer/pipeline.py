from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from image_enhancer.config import (
    EnhancementConfig,
    EnhancementMode,
    ProcessingResult,
    ProcessingStep,
    QualityMetrics,
)
from image_enhancer.enhancers import (
    adaptive_sharpening,
    apply_non_local_means_denoising,
    apply_super_resolution_effect,
    auto_white_balance,
    enhance_contrast_local,
    enhance_low_light,
    enhance_saturation,
    load_image,
    resize_to_target,
    save_image,
)


class AdvancedImageEnhancer:
    def __init__(self, config: Optional[EnhancementConfig] = None) -> None:
        self.config = config or EnhancementConfig()
        self.logger = logging.getLogger(__name__)

    def _start_timer(self, step: ProcessingStep) -> float:
        return time.time()

    def _end_timer(self, start_time: float, step: ProcessingStep, step_times: Dict[str, float]) -> float:
        duration = time.time() - start_time
        step_times[step.value] = duration
        return duration

    def _calculate_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> QualityMetrics:
        return QualityMetrics.from_images(original, enhanced)

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            image = load_image(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
            return image
        except Exception as e:
            self.logger.error(f"Error loading {image_path}: {e}")
            return None

    def _resize_to_target(self, image: np.ndarray) -> np.ndarray:
        return resize_to_target(image, self.config.target_size, self.config.fit_mode)

    def _apply_non_local_means_denoising(self, image: np.ndarray) -> np.ndarray:
        return apply_non_local_means_denoising(image, self.config.get_mode_params())

    def _enhance_contrast_local(self, image: np.ndarray) -> np.ndarray:
        return enhance_contrast_local(image, self.config.get_mode_params())

    def _enhance_saturation(self, image: np.ndarray) -> np.ndarray:
        return enhance_saturation(image, self.config.get_mode_params())

    def _adaptive_sharpening(self, image: np.ndarray) -> np.ndarray:
        return adaptive_sharpening(image, self.config.get_mode_params())

    def _auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        return auto_white_balance(image)

    def _enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        return enhance_low_light(image)

    def _correct_exif_orientation(self, image: np.ndarray, image_path: Path) -> np.ndarray:
        from image_enhancer.enhancers import correct_exif_orientation
        return correct_exif_orientation(image, image_path)

    def _apply_super_resolution_effect(self, image: np.ndarray) -> np.ndarray:
        return apply_super_resolution_effect(image, self.config)

    def _save_image(self, image: np.ndarray, output_path: Path) -> bool:
        return save_image(image, output_path)

    def enhance_image(self, input_path: Path, output_path: Optional[Path] = None) -> ProcessingResult:
        start_time = time.time()
        step_times: Dict[str, float] = {}

        try:
            t = time.time()
            original = self._load_image(input_path)
            if original is None:
                raise ValueError(f"Failed to load image: {input_path}")
            step_times[ProcessingStep.LOADING.value] = time.time() - t

            if output_path is None:
                output_path = self._generate_output_path(input_path)

            original_size_kb = input_path.stat().st_size / 1024

            t = time.time()
            resized = self._resize_to_target(original)
            step_times[ProcessingStep.RESIZING.value] = time.time() - t

            enhanced = resized.copy()
            if self.config.low_light_correction:
                enhanced = self._enhance_low_light(enhanced)
            if self.config.auto_wb:
                enhanced = self._auto_white_balance(enhanced)

            t = time.time()
            enhanced = self._apply_non_local_means_denoising(enhanced)
            step_times[ProcessingStep.DENOISING.value] = time.time() - t

            if self.config.mode in (EnhancementMode.SHARP, EnhancementMode.LANDSCAPE):
                enhanced = self._apply_super_resolution_effect(enhanced)

            t = time.time()
            enhanced = self._enhance_contrast_local(enhanced)
            enhanced = self._enhance_saturation(enhanced)
            step_times[ProcessingStep.CONTRAST_ENHANCEMENT.value] = time.time() - t

            t = time.time()
            enhanced = self._adaptive_sharpening(enhanced)
            step_times[ProcessingStep.SHARPENING.value] = time.time() - t

            if getattr(self.config, "ai_face", False):
                try:
                    from image_enhancer.ai_face import enhance_faces_ai
                    t = time.time()
                    enhanced = enhance_faces_ai(enhanced)
                    step_times["ai_face"] = time.time() - t
                except Exception as e:
                    self.logger.warning(f"AI face enhancement error: {e}")

            t = time.time()
            if not self._save_image(enhanced, output_path):
                raise RuntimeError(f"Failed to save image to {output_path}")
            step_times[ProcessingStep.SAVING.value] = time.time() - t

            if not self.config.preserve_original and output_path.resolve() != input_path.resolve():
                try:
                    input_path.unlink()
                except Exception:
                    pass

            processing_time = time.time() - start_time
            metrics = self._calculate_metrics(resized, enhanced)
            output_size_kb = output_path.stat().st_size / 1024
            file_size_change = ((output_size_kb - original_size_kb) / max(original_size_kb, 1)) * 100

            result = ProcessingResult(
                input_path=input_path,
                output_path=output_path,
                success=True,
                processing_time=processing_time,
                quality_metrics=metrics,
                file_size_change=file_size_change,
                step_times=step_times,
            )
            self._log_success(result)
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to process {input_path}: {e}")
            return ProcessingResult(
                input_path=input_path,
                output_path=output_path or Path(),
                success=False,
                processing_time=processing_time,
                error_message=str(e),
                step_times=step_times,
            )

    def _generate_output_path(self, input_path: Path) -> Path:
        output_dir = input_path.parent / "enhanced"
        output_dir.mkdir(exist_ok=True)
        stem = input_path.stem
        fmt = self.config.output_format
        if fmt:
            ext = f".{fmt}"
        else:
            ext = input_path.suffix if input_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp') else '.png'
        suffix = f"_enhanced_{self.config.mode.value}{ext}"
        return output_dir / f"{stem}{suffix}"

    def _log_success(self, result: ProcessingResult) -> None:
        metrics = result.quality_metrics
        self.logger.info(f"Success: {result.input_path.name} -> {result.output_path.name}")
        self.logger.info(f"  Time: {result.processing_time:.2f}s")
        if metrics:
            self.logger.info(f"  Sharpness: {metrics.sharpness:+.1f}")
            self.logger.info(f"  Contrast: {metrics.contrast:+.1f}")
            self.logger.info(f"  SNR: {metrics.snr:+.2f}")
        self.logger.info(f"  File size change: {result.file_size_change:+.1f}%")

    def process_batch(self, input_dir: Path, patterns: Optional[List[str]] = None) -> List[ProcessingResult]:
        if patterns is None:
            patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        files: List[Path] = []
        for pattern in patterns:
            files.extend(sorted(input_dir.glob(pattern)))
        if not files:
            self.logger.warning(f"No matching files found in {input_dir}")
            return []

        self.logger.info(f"Batch processing {len(files)} images with {self.config.workers} workers")
        results: List[ProcessingResult] = []

        if self.config.workers <= 1:
            for idx, file_path in enumerate(files, 1):
                self.logger.info(f"[{idx}/{len(files)}] {file_path.name}")
                results.append(self.enhance_image(file_path))
        else:
            with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                future_to_file = {executor.submit(self.enhance_image, f): f for f in files}
                for future in as_completed(future_to_file):
                    results.append(future.result())

        successful = [r for r in results if r.success]
        if successful:
            avg_time = sum(r.processing_time for r in successful) / len(successful)
            self.logger.info(f"Batch complete: {len(successful)}/{len(files)} success, avg {avg_time:.2f}s")
        return results
