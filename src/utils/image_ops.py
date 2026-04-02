"""Image reading, analysis, and writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from utils.common import ensure_dir


def safe_read_image(path: str | Path) -> tuple[np.ndarray | None, str | None]:
    """Read an image into RGB numpy format while honoring EXIF orientation."""
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            rgb_image = image.convert("RGB")
            return np.asarray(rgb_image), None
    except Exception as exc:  # pragma: no cover - defensive in data pipeline
        return None, str(exc)


def save_image(path: str | Path, image_rgb: np.ndarray, jpeg_quality: int = 95) -> Path:
    """Save an RGB numpy image while preserving directory structure."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    image = Image.fromarray(image_rgb.astype(np.uint8))
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=jpeg_quality, optimize=True)
    else:
        image.save(output_path)
    return output_path


def resize_long_edge(image_rgb: np.ndarray, max_long_edge: int | None) -> tuple[np.ndarray, float]:
    """Resize the image when its longest side is above the configured limit."""
    if not max_long_edge:
        return image_rgb, 1.0
    height, width = image_rgb.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= max_long_edge:
        return image_rgb, 1.0

    scale = max_long_edge / float(longest_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=interpolation)
    return resized, scale


def compute_blur_score(image_rgb: np.ndarray) -> float:
    """Estimate blur using variance of the Laplacian."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness_metrics(image_rgb: np.ndarray) -> dict[str, float]:
    """Compute brightness summary metrics used by the pipeline thresholds."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    return {
        "mean_brightness": float(gray.mean()),
        "dark_pixel_ratio": float((normalized < 0.15).mean()),
        "bright_pixel_ratio": float((normalized > 0.92).mean()),
    }


def image_size(image_rgb: np.ndarray) -> tuple[int, int]:
    """Return image width and height."""
    height, width = image_rgb.shape[:2]
    return width, height


def build_binary_mask(shape: tuple[int, int], polygon: list[tuple[float, float]]) -> np.ndarray:
    """Rasterize a polygon into a binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def lighten_background(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Apply a gentle background smoothing effect outside the mask."""
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=3.0)
    mask_3c = np.repeat((mask > 0)[:, :, None], 3, axis=2)
    mixed = np.where(mask_3c, image_rgb, cv2.addWeighted(image_rgb, 1.0 - alpha, blurred, alpha, 0.0))
    return mixed


def crop_bbox_pixels(
    image_rgb: np.ndarray,
    bbox_xywh: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a normalized YOLO bbox from an image and return the crop plus pixel bounds."""
    height, width = image_rgb.shape[:2]
    x_center, y_center, box_width, box_height = bbox_xywh
    x1 = max(0, int(round((x_center - box_width / 2.0) * width)))
    y1 = max(0, int(round((y_center - box_height / 2.0) * height)))
    x2 = min(width, int(round((x_center + box_width / 2.0) * width)))
    y2 = min(height, int(round((y_center + box_height / 2.0) * height)))
    return image_rgb[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def paste_bbox_pixels(
    image_rgb: np.ndarray,
    patch_rgb: np.ndarray,
    coords: tuple[int, int, int, int],
) -> np.ndarray:
    """Paste a patch back into the original image at the provided coordinates."""
    x1, y1, x2, y2 = coords
    output = image_rgb.copy()
    if patch_rgb.size == 0 or x1 >= x2 or y1 >= y2:
        return output
    patch_resized = cv2.resize(patch_rgb, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    output[y1:y2, x1:x2] = patch_resized
    return output


def clamp_uint8(image_rgb: np.ndarray) -> np.ndarray:
    """Clip image values to the uint8 range."""
    return np.clip(image_rgb, 0, 255).astype(np.uint8)

