"""Generate lightweight synthetic images for minority classes without external APIs."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.annotations import (
    build_label_index,
    infer_task_from_labels,
    parse_label_file,
    resolve_label_path,
    serialize_label,
)
from utils.common import ensure_dir, list_image_files, relative_posix, setup_logger, write_csv, write_json
from utils.config import load_yaml
from utils.image_ops import (
    build_binary_mask,
    clamp_uint8,
    crop_bbox_pixels,
    lighten_background,
    paste_bbox_pixels,
    safe_read_image,
    save_image,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate lightweight synthetic minority samples.")
    parser.add_argument("--input", type=Path, required=True, help="Input deduplicated image directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output synthesized dataset root.")
    parser.add_argument("--config", type=Path, default=Path("configs/thresholds.yaml"), help="Threshold config YAML.")
    parser.add_argument("--labels", type=Path, default=Path("data/raw_labels"), help="Label directory.")
    parser.add_argument("--task", choices=["auto", "detect", "segment"], default="auto", help="YOLO task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def mild_photometric_transform(image_rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply a subtle photometric transform that preserves lesion morphology."""
    alpha = rng.uniform(0.92, 1.08)
    beta = rng.uniform(-10.0, 10.0)
    transformed = image_rgb.astype(np.float32) * alpha + beta

    if rng.random() < 0.3:
        hsv = cv2.cvtColor(clamp_uint8(transformed), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(0.92, 1.08)
        hsv[:, :, 2] *= rng.uniform(0.95, 1.05)
        transformed = cv2.cvtColor(clamp_uint8(hsv), cv2.COLOR_HSV2RGB)

    return clamp_uint8(transformed)


def synthesize_segment(
    image_rgb: np.ndarray,
    polygons: list[list[tuple[float, float]]],
    rng: random.Random,
) -> tuple[np.ndarray, str] | None:
    """Create a gentle segmentation-aware synthetic sample."""
    if not polygons:
        return None

    height, width = image_rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        pixel_points = [(int(round(x_coord * width)), int(round(y_coord * height))) for x_coord, y_coord in polygon]
        mask = np.maximum(mask, build_binary_mask((height, width), pixel_points))

    lightened = lighten_background(image_rgb, mask, alpha=0.18)
    return mild_photometric_transform(lightened, rng), "segment_foreground_blend"


def synthesize_detect(
    image_rgb: np.ndarray,
    bboxes: list[tuple[float, float, float, float]],
    rng: random.Random,
) -> tuple[np.ndarray, str] | None:
    """Create a gentle detection-aware synthetic sample."""
    if not bboxes:
        return None

    largest_bbox = max(bboxes, key=lambda bbox: bbox[2] * bbox[3])
    patch, coords = crop_bbox_pixels(image_rgb, largest_bbox)
    if patch.size == 0:
        return None

    enhanced_patch = mild_photometric_transform(patch, rng)
    blended = paste_bbox_pixels(image_rgb, enhanced_patch, coords)
    return mild_photometric_transform(blended, rng), "bbox_focus_blend"


def copy_original_samples(
    input_root: Path,
    output_root: Path,
    label_root: Path,
    logger,
) -> list[dict[str, Any]]:
    """Copy original images and labels into the synthesized dataset root."""
    image_output_root = ensure_dir(output_root / "images")
    label_output_root = ensure_dir(output_root / "labels")
    label_index = build_label_index([label_root] if label_root.exists() else [])
    manifest_rows: list[dict[str, Any]] = []

    for image_path in list_image_files(input_root):
        relative_path = Path(relative_posix(image_path, input_root))
        output_image_path = image_output_root / relative_path
        ensure_dir(output_image_path.parent)
        shutil.copy2(image_path, output_image_path)

        label_path = resolve_label_path(image_path, input_root, label_index) if label_root.exists() else None
        output_label_path = None
        if label_path is not None and label_path.exists():
            output_label_path = label_output_root / relative_path.with_suffix(".txt")
            ensure_dir(output_label_path.parent)
            shutil.copy2(label_path, output_label_path)

        manifest_rows.append(
            {
                "image_path": relative_posix(output_image_path, output_root),
                "label_path": relative_posix(output_label_path, output_root) if output_label_path else "",
                "is_synthetic": False,
                "parent_image": "",
                "method": "original_copy",
            }
        )
    logger.info("Copied %s original images into %s", len(manifest_rows), output_root)
    return manifest_rows


def main() -> None:
    """Generate synthetic minority samples and copy originals into one output root."""
    args = parse_args()
    logger = setup_logger("synthesize_images")
    rng = random.Random(args.seed)
    config = load_yaml(args.config)
    synthesis_cfg = config.get("synthesis", {})
    image_cfg = config.get("image", {})
    target_per_class_cfg = synthesis_cfg.get("target_per_class", {"default": 1200})
    max_multiplier = float(synthesis_cfg.get("max_multiplier", 2.0))
    max_per_parent = int(synthesis_cfg.get("max_per_parent", 2))
    methods_cfg = synthesis_cfg.get("methods", {})
    output_extension = str(image_cfg.get("normalize_extension", ".jpg"))
    jpeg_quality = int(image_cfg.get("jpeg_quality", 95))

    manifest_rows = copy_original_samples(args.input, args.output, args.labels, logger)
    image_output_root = args.output / "images"
    label_output_root = args.output / "labels"
    candidate_label_roots = [label_output_root]
    if args.labels.exists():
        candidate_label_roots.append(args.labels)
    label_index = build_label_index(candidate_label_roots)
    label_paths = sorted(label_output_root.rglob("*.txt"))
    task = args.task if args.task != "auto" else infer_task_from_labels(label_paths)

    class_counts: Counter[int] = Counter()
    class_to_records: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    parent_generate_count: Counter[str] = Counter()
    original_image_paths = list_image_files(image_output_root)

    for image_path in original_image_paths:
        label_path = resolve_label_path(image_path, args.output, label_index)
        if label_path is None or not label_path.exists():
            continue
        try:
            parsed = parse_label_file(label_path, task=task)
        except Exception as exc:
            logger.warning("Skipping synthesis source with invalid label %s: %s", label_path, exc)
            continue

        class_ids = sorted({obj.class_id for obj in parsed.objects})
        class_counts.update(obj.class_id for obj in parsed.objects)
        record = {
            "image_path": image_path,
            "label_path": label_path,
            "parsed": parsed,
            "class_ids": class_ids,
        }
        for class_id in class_ids:
            class_to_records[class_id].append(record)

    def target_for_class(class_id: int) -> int:
        if str(class_id) in target_per_class_cfg:
            return int(target_per_class_cfg[str(class_id)])
        return int(target_per_class_cfg.get("default", 1200))

    plan: dict[int, int] = {}
    for class_id, current_count in class_counts.items():
        target = target_for_class(class_id)
        max_allowed = max(current_count, int(round(current_count * max_multiplier)))
        desired_total = min(target, max_allowed)
        needed = max(0, desired_total - current_count)
        if needed > 0 and class_to_records[class_id]:
            plan[class_id] = needed

    logger.info("Synthesis plan: %s", plan)

    synthetic_rows: list[dict[str, Any]] = []
    synthetic_counter = 0

    for class_id, needed in plan.items():
        candidates = class_to_records[class_id]
        candidate_index = 0
        attempts = 0
        max_attempts = max(needed * max(len(candidates), 1) * 4, 20)
        generated = 0

        while generated < needed and attempts < max_attempts:
            record = candidates[candidate_index % len(candidates)]
            candidate_index += 1
            attempts += 1

            parent_rel = relative_posix(record["image_path"], args.output)
            if parent_generate_count[parent_rel] >= max_per_parent:
                continue

            image_rgb, error = safe_read_image(record["image_path"])
            if error or image_rgb is None:
                continue

            parsed = record["parsed"]
            synthetic_result = None
            if task == "segment" and methods_cfg.get("segment_foreground_blend", True):
                polygons = [obj.polygon for obj in parsed.objects if obj.polygon is not None]
                synthetic_result = synthesize_segment(image_rgb, polygons, rng)
            elif task == "detect" and methods_cfg.get("bbox_focus_blend", True):
                bboxes = [obj.bbox for obj in parsed.objects if obj.bbox is not None]
                synthetic_result = synthesize_detect(image_rgb, bboxes, rng)

            if synthetic_result is None and methods_cfg.get("photometric_light", True):
                synthetic_result = (mild_photometric_transform(image_rgb, rng), "photometric_light")
            if synthetic_result is None:
                continue

            synthetic_image, method = synthetic_result
            synthetic_counter += 1
            generated += 1
            parent_generate_count[parent_rel] += 1

            stem = record["image_path"].stem
            relative_image_path = Path("synthetic") / f"class_{class_id}" / f"{stem}__syn_{synthetic_counter:04d}{output_extension}"
            relative_label_path = relative_image_path.with_suffix(".txt")
            output_image_path = image_output_root / relative_image_path
            output_label_path = label_output_root / relative_label_path

            save_image(output_image_path, synthetic_image, jpeg_quality=jpeg_quality)
            ensure_dir(output_label_path.parent)
            output_label_path.write_text(serialize_label(parsed), encoding="utf-8")

            synthetic_rows.append(
                {
                    "image_path": relative_posix(output_image_path, args.output),
                    "label_path": relative_posix(output_label_path, args.output),
                    "is_synthetic": True,
                    "parent_image": parent_rel,
                    "method": method,
                }
            )

    full_manifest = manifest_rows + synthetic_rows
    summary = {
        "task": task,
        "original_images": len(manifest_rows),
        "synthetic_images": len(synthetic_rows),
        "class_counts_before": {str(key): value for key, value in class_counts.items()},
        "generation_plan": {str(key): value for key, value in plan.items()},
    }
    write_json(args.output / "synthesis_manifest.json", {"summary": summary, "items": full_manifest})
    write_csv(args.output / "synthesis_manifest.csv", full_manifest)
    logger.info("Synthesis complete. Added %s synthetic images to %s", len(synthetic_rows), args.output)


if __name__ == "__main__":
    main()

