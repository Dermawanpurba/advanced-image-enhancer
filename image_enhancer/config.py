from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class EnhancementMode(Enum):
    NATURAL = "natural"
    SHARP = "sharp"
    VIBRANT = "vibrant"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class ProcessingStep(Enum):
    LOADING = "loading"
    RESIZING = "resizing"
    DENOISING = "denoising"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"
    SHARPENING = "sharpening"
    SAVING = "saving"


@dataclass
class QualityMetrics:
    sharpness: float
    contrast: float
    snr: float
    brightness: float
    color_variance: float

    @classmethod
    def from_images(cls, before: np.ndarray, after: np.ndarray) -> QualityMetrics:
        if len(before.shape) == 3:
            gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
            gray_after = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        else:
            gray_before = before
            gray_after = after

        sharpness = cv2.Laplacian(gray_after, cv2.CV_64F).var() - cv2.Laplacian(gray_before, cv2.CV_64F).var()
        contrast = float(np.std(gray_after) - np.std(gray_before))

        std_before = np.std(gray_before)
        std_after = np.std(gray_after)
        snr_before = float(np.mean(gray_before) / std_before) if std_before > 0 else 0.0
        snr_after = float(np.mean(gray_after) / std_after) if std_after > 0 else 0.0
        snr = snr_after - snr_before

        brightness = float(np.mean(gray_after) - np.mean(gray_before))

        if before.ndim == 3:
            color_var_before = float(np.mean([np.std(before[:, :, i]) for i in range(3)]))
            color_var_after = float(np.mean([np.std(after[:, :, i]) for i in range(3)]))
        else:
            color_var_before = 0.0
            color_var_after = 0.0
        color_variance = color_var_after - color_var_before

        return cls(sharpness=sharpness, contrast=contrast, snr=snr, brightness=brightness, color_variance=color_variance)


@dataclass
class ProcessingResult:
    input_path: Path
    output_path: Path
    success: bool
    processing_time: float
    quality_metrics: Optional[QualityMetrics] = None
    file_size_change: float = 0.0
    error_message: Optional[str] = None
    step_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnhancementConfig:
    target_size: Tuple[int, int] = (1920, 1080)
    mode: EnhancementMode = EnhancementMode.NATURAL
    denoise_strength: float = 3.0
    sharpening_strength: float = 1.2
    contrast_boost: float = 1.1
    saturation_boost: float = 1.1
    preserve_original: bool = True
    output_format: Optional[str] = None
    fit_mode: str = "stretch"
    auto_wb: bool = False
    low_light_correction: bool = False
    workers: int = 1
    ai_face: bool = False

    def get_mode_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "denoise_strength": self.denoise_strength,
            "sharpening_strength": self.sharpening_strength,
            "contrast_boost": self.contrast_boost,
            "saturation_boost": self.saturation_boost,
        }
        if self.mode == EnhancementMode.SHARP:
            params.update(sharpening_strength=1.5, denoise_strength=2.0)
        elif self.mode == EnhancementMode.VIBRANT:
            params.update(contrast_boost=1.3, saturation_boost=1.3)
        elif self.mode == EnhancementMode.PORTRAIT:
            params.update(denoise_strength=4.0, sharpening_strength=1.1, contrast_boost=1.05)
        elif self.mode == EnhancementMode.LANDSCAPE:
            params.update(sharpening_strength=1.4, saturation_boost=1.2)
        return params



