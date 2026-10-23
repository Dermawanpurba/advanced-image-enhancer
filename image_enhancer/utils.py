from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("image_enhancer.log", mode="w", encoding="utf-8"),
        ],
    )


def print_welcome_banner() -> None:
    print(f"""
{'#' * 70}
{'ADVANCED IMAGE ENHANCER'.center(70)}
{'#' * 70}
Professional-grade image processing pipeline
""")


def get_target_size(size_str: str, original_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    resolutions = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4k": (3840, 2160),
    }
    if size_str.lower() == "original" and original_size is not None:
        return original_size
    return resolutions.get(size_str.lower(), (1920, 1080))


def guess_output_format(input_path: Path, forced_format: Optional[str] = None) -> str:
    if forced_format:
        return forced_format.lower()
    ext = input_path.suffix.lower()
    return {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "tiff": "tiff"}.get(ext, "png")


def load_yaml_config(config_path: str) -> dict:
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("Warning: PyYAML not installed. Install with: pip install pyyaml")
        return {}
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return {}
