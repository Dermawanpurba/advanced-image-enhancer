"""
Test suite for Advanced Image Enhancer V2.
Run with: python -m pytest tests.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Import the module under test
from image_enhancer import (
    AdvancedImageEnhancer,
    EnhancementConfig,
    EnhancementMode,
    ProcessingResult,
    ProcessingStep,
    QualityMetrics,
)
from image_enhancer import get_target_size, parse_arguments, main


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rgb_image():
    """Synthetic RGB image (100x100) with gradient."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        img[:, i] = [int(i * 2.55), int(255 - i * 2.55), 128]
    return img


@pytest.fixture
def grayscale_image():
    """Synthetic grayscale image (50x50)."""
    return np.random.randint(0, 256, (50, 50), dtype=np.uint8)


@pytest.fixture
def temp_image(tmp_path):
    """Write a synthetic image to a temp file and return its path."""
    path = tmp_path / "input.jpg"
    cv2.imwrite(str(path), np.ones((100, 100, 3), dtype=np.uint8) * 128)
    return path


@pytest.fixture
def default_enhancer():
    return AdvancedImageEnhancer()


# ============================================================================
# Test: Enums and Data Classes
# ============================================================================

class TestEnhancementMode:
    def test_values(self):
        assert EnhancementMode.NATURAL.value == "natural"
        assert EnhancementMode.SHARP.value == "sharp"
        assert EnhancementMode.VIBRANT.value == "vibrant"
        assert EnhancementMode.PORTRAIT.value == "portrait"
        assert EnhancementMode.LANDSCAPE.value == "landscape"

    def test_from_string(self):
        assert EnhancementMode("natural") == EnhancementMode.NATURAL
        assert EnhancementMode("vibrant") == EnhancementMode.VIBRANT

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            EnhancementMode("invalid")


class TestProcessingStep:
    def test_members(self):
        assert len(ProcessingStep) == 6
        assert ProcessingStep.LOADING.value == "loading"


class TestQualityMetrics:
    def test_construction(self):
        m = QualityMetrics(sharpness=1.0, contrast=2.0, snr=0.5, brightness=-0.1, color_variance=3.0)
        assert m.sharpness == 1.0
        assert m.contrast == 2.0

    def test_from_images_identical(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        m = QualityMetrics.from_images(img, img)
        assert m.sharpness == 0.0
        assert m.contrast == 0.0
        assert m.snr == 0.0
        assert m.brightness == 0.0
        assert m.color_variance == 0.0

    def test_from_images_grayscale(self):
        img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        m = QualityMetrics.from_images(img, img)
        assert m.color_variance == 0.0

    def test_from_images_snr_zero_std(self):
        img = np.ones((50, 50), dtype=np.uint8) * 128
        m = QualityMetrics.from_images(img, img)
        assert m.snr == 0.0


class TestProcessingResult:
    def test_default_step_times(self):
        r = ProcessingResult(
            input_path=Path("in.jpg"),
            output_path=Path("out.jpg"),
            success=True,
            processing_time=1.0,
        )
        assert r.step_times == {}

    def test_with_metrics(self):
        r = ProcessingResult(
            input_path=Path("in.jpg"),
            output_path=Path("out.jpg"),
            success=True,
            processing_time=2.0,
            quality_metrics=QualityMetrics(1, 2, 3, 4, 5),
        )
        assert r.quality_metrics is not None
        assert r.success

    def test_error_result(self):
        r = ProcessingResult(
            input_path=Path("in.jpg"),
            output_path=Path("out.jpg"),
            success=False,
            processing_time=0.0,
            error_message="failed",
        )
        assert not r.success
        assert r.error_message == "failed"


class TestEnhancementConfig:
    def test_defaults(self):
        c = EnhancementConfig()
        assert c.target_size == (1920, 1080)
        assert c.mode == EnhancementMode.NATURAL

    def test_custom(self):
        c = EnhancementConfig(
            target_size=(3840, 2160),
            mode=EnhancementMode.VIBRANT,
            denoise_strength=5.0,
            sharpening_strength=1.5,
            output_format="jpeg",
            fit_mode="crop",
            auto_wb=True,
            low_light_correction=True,
            workers=4,
        )
        assert c.target_size == (3840, 2160)
        assert c.output_format == "jpeg"
        assert c.fit_mode == "crop"
        assert c.auto_wb
        assert c.workers == 4

    def test_mode_params_natural(self):
        p = EnhancementConfig().get_mode_params()
        assert p["denoise_strength"] == 3.0

    def test_mode_params_sharp(self):
        p = EnhancementConfig(mode=EnhancementMode.SHARP).get_mode_params()
        assert p["sharpening_strength"] == 1.5
        assert p["denoise_strength"] == 2.0

    def test_mode_params_vibrant(self):
        p = EnhancementConfig(mode=EnhancementMode.VIBRANT).get_mode_params()
        assert p["contrast_boost"] == 1.3

    def test_mode_params_portrait(self):
        p = EnhancementConfig(mode=EnhancementMode.PORTRAIT).get_mode_params()
        assert p["denoise_strength"] == 4.0
        assert p["sharpening_strength"] == 1.1

    def test_mode_params_landscape(self):
        p = EnhancementConfig(mode=EnhancementMode.LANDSCAPE).get_mode_params()
        assert p["sharpening_strength"] == 1.4


# ============================================================================
# Test: Enhancement Pipeline
# ============================================================================

class TestPipeline:
    def test_load_image_valid(self, temp_image):
        enhancer = AdvancedImageEnhancer()
        img = enhancer._load_image(temp_image)
        assert img is not None
        assert img.shape[2] == 3

    def test_load_image_invalid(self):
        enhancer = AdvancedImageEnhancer()
        img = enhancer._load_image(Path("/nonexistent.jpg"))
        assert img is None

    def test_resize_same_size(self, rgb_image):
        enhancer = AdvancedImageEnhancer(EnhancementConfig(target_size=(100, 100)))
        result = enhancer._resize_to_target(rgb_image)
        assert result.shape == (100, 100, 3)

    def test_resize_upscale(self, rgb_image):
        enhancer = AdvancedImageEnhancer(EnhancementConfig(target_size=(200, 200)))
        result = enhancer._resize_to_target(rgb_image)
        assert result.shape == (200, 200, 3)

    def test_resize_downscale(self, rgb_image):
        enhancer = AdvancedImageEnhancer(EnhancementConfig(target_size=(50, 50)))
        result = enhancer._resize_to_target(rgb_image)
        assert result.shape == (50, 50, 3)

    def test_resize_fit_crop(self, rgb_image):
        enhancer = AdvancedImageEnhancer(EnhancementConfig(target_size=(50, 100), fit_mode="crop"))
        result = enhancer._resize_to_target(rgb_image)
        assert result.shape == (100, 50, 3)

    def test_resize_fit_pad(self, rgb_image):
        enhancer = AdvancedImageEnhancer(EnhancementConfig(target_size=(200, 100), fit_mode="pad"))
        result = enhancer._resize_to_target(rgb_image)
        assert result.shape == (100, 200, 3)
        # Left border should be black (image padded horizontally)
        assert float(np.mean(result[:, 0, :])) == 0

    def test_resize_same_size_no_copy(self, rgb_image):
        enhancer = AdvancedImageEnhancer(EnhancementConfig(target_size=(100, 100)))
        result = enhancer._resize_to_target(rgb_image)
        assert result is rgb_image

    def test_denoising(self, rgb_image):
        enhancer = AdvancedImageEnhancer()
        result = enhancer._apply_non_local_means_denoising(rgb_image)
        assert result.shape == rgb_image.shape
        assert result.dtype == np.uint8

    def test_contrast_enhancement(self, rgb_image):
        enhancer = AdvancedImageEnhancer()
        result = enhancer._enhance_contrast_local(rgb_image)
        assert result.shape == rgb_image.shape

    def test_saturation_boost(self, rgb_image):
        enhancer = AdvancedImageEnhancer(EnhancementConfig(saturation_boost=1.5))
        result = enhancer._enhance_saturation(rgb_image)
        assert result.shape == rgb_image.shape

    def test_saturation_no_change(self, rgb_image):
        enhancer = AdvancedImageEnhancer(EnhancementConfig(saturation_boost=1.0))
        result = enhancer._enhance_saturation(rgb_image)
        assert result is rgb_image

    def test_sharpening(self, rgb_image):
        enhancer = AdvancedImageEnhancer()
        result = enhancer._adaptive_sharpening(rgb_image)
        assert result.shape == rgb_image.shape

    def test_sharpening_no_change(self, rgb_image):
        enhancer = AdvancedImageEnhancer(EnhancementConfig(sharpening_strength=1.0))
        result = enhancer._adaptive_sharpening(rgb_image)
        assert result is rgb_image

    def test_sharpening_grayscale(self, grayscale_image):
        enhancer = AdvancedImageEnhancer()
        result = enhancer._adaptive_sharpening(grayscale_image)
        assert result.ndim == 2

    def test_exif_correction_no_exif(self, rgb_image, temp_image):
        enhancer = AdvancedImageEnhancer()
        result = enhancer._correct_exif_orientation(rgb_image, temp_image)
        assert np.array_equal(result, rgb_image)

    def test_auto_white_balance(self):
        enhancer = AdvancedImageEnhancer()
        blue_img = np.zeros((50, 50, 3), dtype=np.uint8)
        blue_img[:, :, 0] = 200
        blue_img[:, :, 1] = 50
        blue_img[:, :, 2] = 50
        result = enhancer._auto_white_balance(blue_img)
        means = [float(np.mean(result[:, :, i])) for i in range(3)]
        assert max(means) - min(means) < 40

    def test_auto_white_balance_zero_channel(self):
        enhancer = AdvancedImageEnhancer()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = enhancer._auto_white_balance(img)
        assert np.array_equal(result, img)

    def test_enhance_low_light(self):
        enhancer = AdvancedImageEnhancer()
        dark = np.ones((50, 50, 3), dtype=np.uint8) * 30
        result = enhancer._enhance_low_light(dark)
        assert float(np.mean(result)) > float(np.mean(dark))

    def test_super_resolution_sharp(self, rgb_image):
        config = EnhancementConfig(target_size=(200, 200), mode=EnhancementMode.SHARP)
        enhancer = AdvancedImageEnhancer(config)
        result = enhancer._apply_super_resolution_effect(rgb_image)
        assert result.shape[:2] == (200, 200)

    def test_super_resolution_landscape(self, rgb_image):
        config = EnhancementConfig(target_size=(200, 200), mode=EnhancementMode.LANDSCAPE)
        enhancer = AdvancedImageEnhancer(config)
        result = enhancer._apply_super_resolution_effect(rgb_image)
        assert result.shape[:2] == (200, 200)

    def test_save_jpeg(self, rgb_image, tmp_path):
        path = tmp_path / "out.jpg"
        enhancer = AdvancedImageEnhancer()
        assert enhancer._save_image(rgb_image, path)
        assert path.exists()

    def test_save_png(self, rgb_image, tmp_path):
        path = tmp_path / "out.png"
        enhancer = AdvancedImageEnhancer()
        assert enhancer._save_image(rgb_image, path)
        assert path.exists()

    def test_save_creates_dir(self, rgb_image, tmp_path):
        path = tmp_path / "sub" / "out.jpg"
        enhancer = AdvancedImageEnhancer()
        assert enhancer._save_image(rgb_image, path)
        assert path.exists()

    def test_generate_output_path_default(self):
        enhancer = AdvancedImageEnhancer()
        path = enhancer._generate_output_path(Path("photo.jpg"))
        assert "enhanced" in str(path)
        assert path.suffix == ".jpg"  # mantém extensão original

    def test_generate_output_path_with_format(self):
        config = EnhancementConfig(output_format="jpeg")
        enhancer = AdvancedImageEnhancer(config)
        path = enhancer._generate_output_path(Path("photo.jpg"))
        assert path.suffix == ".jpeg"


class TestEnhanceImage:
    def test_success(self, temp_image, tmp_path):
        enhancer = AdvancedImageEnhancer()
        out = tmp_path / "result.png"
        result = enhancer.enhance_image(temp_image, out)
        assert result.success
        assert out.exists()

    def test_failure_invalid(self):
        enhancer = AdvancedImageEnhancer()
        result = enhancer.enhance_image(Path("/nonexistent.jpg"))
        assert not result.success
        assert result.error_message is not None

    def test_auto_output_path(self, temp_image):
        enhancer = AdvancedImageEnhancer()
        result = enhancer.enhance_image(temp_image)
        assert result.success
        assert "enhanced" in str(result.output_path)

    def test_all_modes(self, temp_image, tmp_path):
        for mode in EnhancementMode:
            config = EnhancementConfig(mode=mode)
            enhancer = AdvancedImageEnhancer(config)
            out = tmp_path / f"{mode.value}.png"
            result = enhancer.enhance_image(temp_image, out)
            assert result.success, f"Mode {mode} failed"

    def test_step_times_populated(self, temp_image, tmp_path):
        enhancer = AdvancedImageEnhancer()
        result = enhancer.enhance_image(temp_image, tmp_path / "out.png")
        assert len(result.step_times) > 0

    def test_preserve_original_false(self, temp_image, tmp_path):
        config = EnhancementConfig(preserve_original=False, output_format="jpg")
        enhancer = AdvancedImageEnhancer(config)
        out = tmp_path / "out.jpg"
        result = enhancer.enhance_image(temp_image, out)
        assert result.success

    def test_with_auto_wb(self, temp_image, tmp_path):
        config = EnhancementConfig(auto_wb=True)
        enhancer = AdvancedImageEnhancer(config)
        result = enhancer.enhance_image(temp_image, tmp_path / "out.png")
        assert result.success

    def test_with_low_light(self, temp_image, tmp_path):
        config = EnhancementConfig(low_light_correction=True)
        enhancer = AdvancedImageEnhancer(config)
        result = enhancer.enhance_image(temp_image, tmp_path / "out.png")
        assert result.success

    def test_with_fit_crop(self, temp_image, tmp_path):
        config = EnhancementConfig(fit_mode="crop")
        enhancer = AdvancedImageEnhancer(config)
        result = enhancer.enhance_image(temp_image, tmp_path / "out.png")
        assert result.success

    def test_with_fit_pad(self, temp_image, tmp_path):
        config = EnhancementConfig(fit_mode="pad")
        enhancer = AdvancedImageEnhancer(config)
        result = enhancer.enhance_image(temp_image, tmp_path / "out.png")
        assert result.success


class TestProcessBatch:
    def test_no_files(self, tmp_path):
        enhancer = AdvancedImageEnhancer()
        results = enhancer.process_batch(tmp_path)
        assert results == []

    def test_multiple_files(self, tmp_path):
        for i in range(3):
            cv2.imwrite(str(tmp_path / f"img_{i}.jpg"),
                        np.ones((50, 50, 3), dtype=np.uint8) * (i * 50))
        enhancer = AdvancedImageEnhancer()
        results = enhancer.process_batch(tmp_path)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_custom_patterns(self, tmp_path):
        cv2.imwrite(str(tmp_path / "a.png"), np.ones((10, 10, 3), dtype=np.uint8))
        (tmp_path / "b.txt").write_text("not an image")
        enhancer = AdvancedImageEnhancer()
        results = enhancer.process_batch(tmp_path, patterns=["*.png"])
        assert len(results) == 1

    def test_parallel_workers(self, tmp_path):
        for i in range(3):
            cv2.imwrite(str(tmp_path / f"img_{i}.jpg"),
                        np.ones((50, 50, 3), dtype=np.uint8) * (i * 50))
        config = EnhancementConfig(workers=2)
        enhancer = AdvancedImageEnhancer(config)
        results = enhancer.process_batch(tmp_path)
        assert len(results) == 3
        assert all(r.success for r in results)


# ============================================================================
# Test: CLI
# ============================================================================

class TestGetTargetSize:
    @pytest.mark.parametrize("s,expected", [
        ("720p", (1280, 720)),
        ("1080p", (1920, 1080)),
        ("1440p", (2560, 1440)),
        ("4k", (3840, 2160)),
        ("4K", (3840, 2160)),
        ("invalid", (1920, 1080)),
    ])
    def test_resolutions(self, s, expected):
        assert get_target_size(s) == expected

    def test_original_with_size(self):
        assert get_target_size("original", (800, 600)) == (800, 600)


class TestParseArguments:
    def test_defaults(self):
        args = parse_arguments([])
        assert args.mode == "natural"
        assert args.size == "1080p"

    def test_positional(self):
        args = parse_arguments(["photo.jpg"])
        assert args.input == "photo.jpg"

    def test_all_flags_batch(self):
        args = parse_arguments([
            "--input-dir", "./photos", "--batch",
            "--mode", "vibrant", "--size", "4k",
            "--format", "jpeg",
        ])
        assert args.batch
        assert args.mode == "vibrant"
        assert args.format == "jpeg"

    def test_new_cli_flags(self):
        args = parse_arguments([
            "photo.jpg",
            "--fit", "crop",
            "--auto-wb",
            "--low-light",
            "--workers", "4",
        ])
        assert args.fit == "crop"
        assert args.auto_wb
        assert args.low_light
        assert args.workers == 4


class TestMain:
    def test_single_file(self, temp_image):
        rc = main([str(temp_image)])
        assert rc == 0

    def test_file_not_found(self):
        rc = main(["/nonexistent.jpg"])
        assert rc == 1

    def test_batch(self, tmp_path):
        cv2.imwrite(str(tmp_path / "test.jpg"), np.ones((50, 50, 3), dtype=np.uint8))
        rc = main(["--input-dir", str(tmp_path), "--batch"])
        assert rc == 0

    def test_batch_dir_not_found(self):
        rc = main(["--input-dir", "/nonexistent", "--batch"])
        assert rc == 1

    def test_batch_format_flag(self, tmp_path):
        cv2.imwrite(str(tmp_path / "test.jpg"), np.ones((50, 50, 3), dtype=np.uint8))
        rc = main(["--input-dir", str(tmp_path), "--batch", "--format", "jpeg"])
        assert rc == 0
