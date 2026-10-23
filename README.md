# Advanced Image Enhancer Pro ⚡

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV Version](https://img.shields.io/badge/opencv-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional-grade on-device AI image enhancement studio with both modern Desktop GUI and Web Live Studio interfaces. Features local neural network inference (GFPGAN v1.4, YuNet, U2Net Rembg, Depth Anything v2) without external API costs or cloud privacy risks.

## ✨ All Local AI Skills Included

- ⚡ **Local AI Face Restoration** — GFPGAN v1.4 + YuNet 5-point face alignment to restore realistic facial textures.
- 🧴 **Natural Skin Beauty Retouching** — Bilateral pore-preserving smoothing with eye, brow, glasses, and lip protection.
- 🖨️ **300 DPI Print Super-Resolution** — Lanczos-4 high-precision upscale (min 2624 × 3936 px, >1MB guaranteed) for official photo prints.
- ✂️ **AI Background Removal (Rembg)** — 1-click transparent PNG or professional studio backdrops (Studio Gray, Clean White, Passport ID).
- 🎬 **Cinematic Kodak Portra Bokeh** — Optical depth simulation with creamy bokeh falloff, subtle Pro Mist bloom, and organic film grain.
- 🧹 **Edge-Preserving Denoising** — Non-local means denoising with Sobel gradient texture gating.
- 🎯 **Interactive Comparison Slider** — Side-by-side interactive split slider in both Desktop GUI and Web Live Studio.

## 🚀 Quick Launch

### 🌐 1. Web Live Studio (Browser Interface)
Jalankan live server web studio dengan tampilan browser modern (port 8501):
```bash
./run_live_studio.sh
# Atau: .venv_mac/bin/python web_studio.py
# Buka di browser: http://localhost:8501
```

### 🖥️ 2. Modern Desktop GUI (CustomTkinter)
Jalankan aplikasi desktop native dengan dark-theme premium:
```bash
./run_gui_mac.sh
# Atau: .venv_mac/bin/python gui.py
```

```bash
# Enhance a single image
python -m image_enhancer photo.jpg

# Batch process a directory
python -m image_enhancer --input-dir ./photos --batch

# Vibrant 4K with crop
python -m image_enhancer photo.jpg --mode vibrant --size 4k --fit crop

# Portrait with white balance
python -m image_enhancer photo.jpg --mode portrait --auto-wb

# Low-light correction
python -m image_enhancer photo.jpg --low-light

# Parallel batch with 4 workers
python -m image_enhancer --input-dir ./photos --batch --workers 4

# Use YAML config profile
python -m image_enhancer photo.jpg --config config.yaml --mode vibrant
```

### Install as a System Command

```bash
pip install -e .
image-enhancer photo.jpg          # CLI
image-enhancer-gui                 # GUI
```

## 🎨 Enhancement Modes

| Mode | Best For | Effect |
|------|----------|--------|
| Natural | General purpose | Balanced, preserves character |
| Sharp | Text, architecture | Enhanced edges + super-resolution |
| Vibrant | Nature, travel | Boosted colors and contrast |
| Portrait | People photos | Strong denoising, subtle skin smoothing |
| Landscape | Scenery, outdoors | Depth + super-resolution + saturation |

## 🖥️ GUI Overview

The graphical interface provides:

- **Original / Preview** panels side by side
- **Magic Enhance** button — runs the full 8-step pipeline
- **Individual operation buttons** — Resize, Denoise, Sharpen, Contrast, Saturation, White Balance, Low Light
- **Sliders** for denoise strength, sharpening, contrast boost, saturation boost
- **Mode, Size, Fit, Format** dropdowns
- **Auto WB** and **Low Light** toggle checkboxes
- **Save / Save As** with format selection

All processing runs in background threads — the UI stays responsive.

## 🛠️ Advanced Usage (Python API)

```python
from image_enhancer import AdvancedImageEnhancer, EnhancementConfig, EnhancementMode

config = EnhancementConfig(
    target_size=(3840, 2160),
    mode=EnhancementMode.VIBRANT,
    denoise_strength=3.5,
    sharpening_strength=1.4,
    contrast_boost=1.3,
    saturation_boost=1.2,
)

enhancer = AdvancedImageEnhancer(config)
result = enhancer.enhance_image("input.jpg", "output.jpg")

print(f"Sharpness: {result.quality_metrics.sharpness:+.1f}")
print(f"Time: {result.processing_time:.2f}s")
```

### Individual Pipeline Steps

```python
from image_enhancer.enhancers import (
    resize_to_target,
    apply_non_local_means_denoising,
    enhance_contrast_local,
    enhance_saturation,
    adaptive_sharpening,
    auto_white_balance,
    enhance_low_light,
)
```

## 📊 Quality Metrics

| Metric | What It Measures | Interpretation |
|--------|-----------------|----------------|
| Sharpness | Laplacian variance | Higher = more edge definition |
| Contrast | Pixel std deviation | Moderate increase is ideal |
| SNR | Mean / std ratio | Higher = cleaner image |
| Brightness | Mean luminance | Should change minimally |
| Color Variance | Channel std deviation | Slight increase = more vibrant |

## 🏗️ Project Structure

```
advanced-image-enhancer/
├── gui.py                  # Graphical interface (tkinter)
├── pyproject.toml          # Package configuration
├── config.yaml             # YAML enhancement profiles
├── requirements.txt        # Dependencies
├── tests.py                # Test suite (74 tests)
├── README.md               # This file
├── LICENSE                 # MIT License
└── image_enhancer/         # Core package
    ├── __init__.py         # Public API exports
    ├── config.py           # Data classes & enums
    ├── enhancers.py        # Individual image operations
    ├── pipeline.py         # Enhancement pipeline orchestrator
    ├── cli.py              # Command-line interface
    ├── utils.py            # Logging, target size, YAML loading
    └── comparison.py       # Side-by-side comparison grid
```

## ⚙️ Processing Pipeline

1. **Load** — Read image with EXIF orientation correction
2. **Resize** — Stretch, crop, or pad to target dimensions (Lanczos upscale)
3. **Pre-process** — Optional low-light gamma correction + white balance
4. **Denoise** — Non-local means denoising on LAB luminance channel
5. **Super-Resolution** — 2× upscale → process → downscale (for Sharp/Landscape modes)
6. **Contrast** — CLAHE on LAB luminance + bilateral filter
7. **Saturation** — HSV saturation channel boost
8. **Sharpen** — Edge-aware unsharp masking via Canny mask

## 📋 CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Input image file | — |
| `-i, --input-dir` | Input directory for batch | `.` |
| `-o, --output` | Output file path | auto-generated |
| `-m, --mode` | Enhancement mode | `natural` |
| `-s, --size` | Target resolution | `1080p` |
| `-d, --denoise` | Denoising strength (1–10) | `3.0` |
| `-sh, --sharpen` | Sharpening strength (1–2) | `1.2` |
| `-c, --contrast` | Contrast boost (1–2) | `1.1` |
| `-sat, --saturation` | Saturation boost (1–2) | `1.1` |
| `--fit` | Aspect ratio handling | `stretch` |
| `--auto-wb` | Auto white balance | off |
| `--low-light` | Low-light correction | off |
| `--format` | Output format (png/jpeg/webp) | original ext |
| `-w, --workers` | Parallel batch workers | `1` |
| `--config` | YAML config file | none |
| `-b, --batch` | Enable batch mode | off |
| `-v, --verbose` | Verbose logging | off |

## 🧪 Tests

```bash
pip install pytest
python -m pytest tests.py -v
```

## 📝 License

MIT License — see [LICENSE](LICENSE) file.

## ⚠️ Disclaimer

Always keep backups of original images. While the enhancer is designed to be safe, unexpected results can occur with certain image types. Test on copies before processing important files.
