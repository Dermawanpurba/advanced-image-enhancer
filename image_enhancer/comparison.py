from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_comparison_grid(
    original_path: Path,
    enhanced_path: Path,
    output_path: Path,
    labels: tuple = ("Original", "Enhanced"),
) -> None:
    original = cv2.imread(str(original_path))
    enhanced = cv2.imread(str(enhanced_path))
    if original is None or enhanced is None:
        raise FileNotFoundError("Could not load one or both input images")

    height = min(original.shape[0], enhanced.shape[0])
    width1 = int(original.shape[1] * height / original.shape[0])
    width2 = int(enhanced.shape[1] * height / enhanced.shape[0])
    original = cv2.resize(original, (width1, height))
    enhanced = cv2.resize(enhanced, (width2, height))

    comparison = np.hstack([original, enhanced])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, labels[0], (20, 50), font, 1.5, (255, 255, 255), 3)
    cv2.putText(comparison, labels[1], (width1 + 20, 50), font, 1.5, (255, 255, 255), 3)
    cv2.line(comparison, (width1, 0), (width1, height), (255, 255, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), comparison)
