"""Preprocess raw images before annotation validation and YOLO dataset build."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils.common import ensure_dir, list_image_files, relative_posix, setup_logger, write_csv, write_json
from utils.config import load_yaml
from utils.image_ops import (
    compute_blur_score,
    compute_brightness_metrics,
    image_size,
    resize_long_edge,
    safe_read_image,
    save_image,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Preprocess raw images for YOLO training.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Input raw image directory.")
    parser.add_argument("--output", type=Path, default=Path("data/cleaned"), help="Output cleaned image directory.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/thresholds.yaml"),
        help="Threshold configuration YAML.",
    )
    return parser.parse_args()


def build_output_path(image_path: Path, input_root: Path, output_root: Path, extension: str) -> Path:
    """Map an input image path to the normalized output path."""
    relative_path = image_path.relative_to(input_root)
    return output_root / relative_path.with_suffix(extension)


def main() -> None:
    """Run preprocessing and save accepted images plus rejection logs."""
    args = parse_args()
    logger = setup_logger("preprocess_images")
    config = load_yaml(args.config)

    image_cfg = config.get("image", {})
    blur_cfg = config.get("blur", {})
    brightness_cfg = config.get("brightness", {})

    min_width = int(image_cfg.get("min_width", 384))
    min_height = int(image_cfg.get("min_height", 384))
    max_long_edge = image_cfg.get("max_long_edge", 1600)
    normalize_extension = str(image_cfg.get("normalize_extension", ".jpg"))
    jpeg_quality = int(image_cfg.get("jpeg_quality", 95))
    blur_threshold = float(blur_cfg.get("variance_of_laplacian_min", 90.0))
    min_mean = float(brightness_cfg.get("min_mean", 35.0))
    max_mean = float(brightness_cfg.get("max_mean", 225.0))
    dark_ratio_max = float(brightness_cfg.get("dark_pixel_ratio_max", 0.75))
    bright_ratio_max = float(brightness_cfg.get("bright_pixel_ratio_max", 0.75))

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for image_path in list_image_files(args.input):
        relative_path = relative_posix(image_path, args.input)
        image_rgb, error = safe_read_image(image_path)
        if error or image_rgb is None:
            rejected_rows.append({"image_path": relative_path, "reason": "corrupted", "detail": error or ""})
            logger.warning("Rejected corrupted image: %s", relative_path)
            continue

        width, height = image_size(image_rgb)
        if width < min_width or height < min_height:
            rejected_rows.append(
                {
                    "image_path": relative_path,
                    "reason": "too_small",
                    "detail": f"{width}x{height} < {min_width}x{min_height}",
                }
            )
            continue

        blur_score = compute_blur_score(image_rgb)
        if blur_score < blur_threshold:
            rejected_rows.append(
                {
                    "image_path": relative_path,
                    "reason": "too_blurry",
                    "detail": f"blur_score={blur_score:.4f}",
                }
            )
            continue

        brightness = compute_brightness_metrics(image_rgb)
        if brightness["mean_brightness"] < min_mean or brightness["dark_pixel_ratio"] > dark_ratio_max:
            rejected_rows.append(
                {
                    "image_path": relative_path,
                    "reason": "too_dark",
                    "detail": str(brightness),
                }
            )
            continue

        if brightness["mean_brightness"] > max_mean or brightness["bright_pixel_ratio"] > bright_ratio_max:
            rejected_rows.append(
                {
                    "image_path": relative_path,
                    "reason": "too_bright",
                    "detail": str(brightness),
                }
            )
            continue

        resized_image, scale = resize_long_edge(image_rgb, max_long_edge)
        output_path = build_output_path(image_path, args.input, args.output, normalize_extension)
        save_image(output_path, resized_image, jpeg_quality=jpeg_quality)

        accepted_rows.append(
            {
                "source_image": relative_path,
                "output_image": relative_posix(output_path, args.output),
                "original_width": width,
                "original_height": height,
                "saved_width": resized_image.shape[1],
                "saved_height": resized_image.shape[0],
                "scale": round(scale, 6),
                "blur_score": round(blur_score, 4),
                "mean_brightness": round(brightness["mean_brightness"], 4),
            }
        )

    report_dir = ensure_dir("outputs/reports")
    write_csv(report_dir / "preprocess_manifest.csv", accepted_rows)
    write_json(report_dir / "preprocess_manifest.json", {"accepted": accepted_rows})
    write_csv(report_dir / "preprocess_rejected.csv", rejected_rows)
    write_json(report_dir / "preprocess_rejected.json", {"rejected": rejected_rows})
    logger.info(
        "Preprocess finished. Accepted=%s | Rejected=%s | Output=%s",
        len(accepted_rows),
        len(rejected_rows),
        args.output,
    )


if __name__ == "__main__":
    main()

