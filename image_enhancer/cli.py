from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2

from image_enhancer.config import EnhancementConfig, EnhancementMode
from image_enhancer.pipeline import AdvancedImageEnhancer
from image_enhancer.utils import get_target_size, load_yaml_config, print_welcome_banner, setup_logging


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced Image Enhancer - Professional image quality enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  image-enhancer image.jpg
  image-enhancer --input-dir ./photos --batch
  image-enhancer --mode vibrant --size 4k
  image-enhancer --mode portrait --denoise 4.0 --fit crop
        """,
    )
    parser.add_argument("input", nargs="?", help="Input image file")
    parser.add_argument("--input-dir", "-i", default=".", help="Input directory")
    parser.add_argument("--output", "-o", help="Output file or directory")
    parser.add_argument("--mode", "-m", choices=[m.value for m in EnhancementMode], default="natural")
    parser.add_argument("--size", "-s", default="1080p", choices=["720p", "1080p", "1440p", "4k", "original"])
    parser.add_argument("--denoise", "-d", type=float, default=3.0)
    parser.add_argument("--sharpen", "-sh", type=float, default=1.2)
    parser.add_argument("--contrast", "-c", type=float, default=1.1)
    parser.add_argument("--saturation", "-sat", type=float, default=1.1)
    parser.add_argument("--batch", "-b", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--preserve-original", action="store_true", default=True)
    parser.add_argument("--format", "-f", choices=["png", "jpeg", "jpg", "webp"], default=None)
    parser.add_argument("--fit", choices=["stretch", "crop", "pad"], default="stretch")
    parser.add_argument("--auto-wb", action="store_true")
    parser.add_argument("--low-light", action="store_true")
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--config", "-cfg", default=None)
    parser.add_argument("--ai-face", "--ai", action="store_true", help="Enable AI face restoration (GFPGAN)")
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        print_welcome_banner()
        mode = EnhancementMode(args.mode)

        config_kwargs = dict(
            mode=mode,
            denoise_strength=args.denoise,
            sharpening_strength=args.sharpen,
            contrast_boost=args.contrast,
            saturation_boost=args.saturation,
            preserve_original=args.preserve_original,
            output_format=args.format,
            fit_mode=args.fit,
            auto_wb=args.auto_wb,
            low_light_correction=args.low_light,
            workers=args.workers,
            ai_face=getattr(args, "ai_face", False),
        )

        if args.config:
            profiles = load_yaml_config(args.config)
            if profiles and args.mode in profiles:
                profile = profiles[args.mode]
                for key, value in profile.items():
                    if key == 'mode':
                        config_kwargs['mode'] = EnhancementMode(value)
                    elif key == 'target_size':
                        config_kwargs['target_size'] = tuple(value)
                    else:
                        config_kwargs[key] = value
                logger.info(f"Applied YAML profile '{args.mode}' from {args.config}")

        config = EnhancementConfig(**config_kwargs)
        enhancer = AdvancedImageEnhancer(config)

        if args.batch or args.input is None:
            input_dir = Path(args.input_dir).resolve()
            if not input_dir.exists():
                logger.error(f"Directory not found: {input_dir}")
                return 1
            if args.size != "original":
                config.target_size = get_target_size(args.size)
            else:
                logger.warning("--size original em batch usa target_size default (1920x1080)")

            logger.info(f"Batch processing: {input_dir}")
            results = enhancer.process_batch(input_dir)

            if results:
                successful = [r for r in results if r.success]
                if successful:
                    summary = {
                        "total_processed": len(results),
                        "successful": len(successful),
                        "failed": len(results) - len(successful),
                        "enhancement_mode": mode.value,
                        "timestamp": datetime.now().isoformat(),
                        "results": [
                            {
                                "input": str(r.input_path),
                                "output": str(r.output_path),
                                "success": r.success,
                                "processing_time": r.processing_time,
                                "file_size_change": r.file_size_change,
                                "sharpness_improvement": r.quality_metrics.sharpness if r.quality_metrics else 0,
                            }
                            for r in results
                        ],
                    }
                    report_path = input_dir / "enhancement_report.json"
                    with open(report_path, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    logger.info(f"Report: {report_path}")
        else:
            input_path = Path(args.input).resolve()
            if not input_path.exists():
                logger.error(f"File not found: {input_path}")
                return 1

            if args.size == "original":
                img = cv2.imread(str(input_path))
                if img is not None:
                    config.target_size = (img.shape[1], img.shape[0])
            else:
                config.target_size = get_target_size(args.size)

            output_path = Path(args.output).resolve() if args.output else None
            result = enhancer.enhance_image(input_path, output_path)

            print(f"\n{'=' * 60}")
            if result.success:
                print(f"ENHANCEMENT COMPLETE")
                print(f"{'-' * 60}")
                print(f"Input:  {result.input_path.name}")
                print(f"Output: {result.output_path.name}")
                print(f"Time:   {result.processing_time:.2f}s")
                if result.quality_metrics:
                    print(f"\nQUALITY IMPROVEMENT:")
                    print(f"  Sharpness:    {result.quality_metrics.sharpness:+.1f}")
                    print(f"  Contrast:     {result.quality_metrics.contrast:+.1f}")
                    print(f"  Color:        {result.quality_metrics.color_variance:+.1f}")
                orig_kb = result.input_path.stat().st_size / 1024
                enh_kb = result.output_path.stat().st_size / 1024
                print(f"\nSize: {orig_kb:.1f}KB -> {enh_kb:.1f}KB ({result.file_size_change:+.1f}%)")
            else:
                print(f"ENHANCEMENT FAILED: {result.error_message}")
            print(f"{'=' * 60}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=args.verbose if 'args' in dir() else False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
