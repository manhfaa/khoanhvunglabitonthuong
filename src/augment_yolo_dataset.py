"""Offline augmentation utility for YOLO train split only."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import albumentations as A

from utils.annotations import infer_task_from_labels, parse_label_file
from utils.common import ensure_dir, list_image_files, relative_posix, setup_logger, write_csv, write_json
from utils.config import load_yaml
from utils.image_ops import safe_read_image, save_image


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Offline augmentation for YOLO train split.")
    parser.add_argument("--input", type=Path, default=Path("data/yolo"), help="Input YOLO dataset root.")
    parser.add_argument("--output", type=Path, default=Path("data/yolo_augmented"), help="Output dataset root.")
    parser.add_argument("--config", type=Path, default=Path("configs/thresholds.yaml"), help="Threshold config YAML.")
    parser.add_argument("--task", choices=["auto", "detect", "segment"], default="auto", help="YOLO task.")
    return parser.parse_args()


def build_augmentations(config: dict[str, Any], task: str) -> A.Compose:
    """Build an Albumentations pipeline from YAML config."""
    aug_cfg = config.get("augmentation", {})
    transforms: list[Any] = []

    def enabled(name: str) -> bool:
        return bool(aug_cfg.get(name, {}).get("enabled", False))

    if enabled("horizontal_flip"):
        transforms.append(A.HorizontalFlip(p=float(aug_cfg["horizontal_flip"].get("p", 0.5))))
    if enabled("vertical_flip"):
        transforms.append(A.VerticalFlip(p=float(aug_cfg["vertical_flip"].get("p", 0.1))))
    if enabled("rotate"):
        transforms.append(
            A.Rotate(
                limit=float(aug_cfg["rotate"].get("limit", 10)),
                border_mode=0,
                p=float(aug_cfg["rotate"].get("p", 0.25)),
            )
        )
    if enabled("shift_scale_rotate"):
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=float(aug_cfg["shift_scale_rotate"].get("shift_limit", 0.05)),
                scale_limit=float(aug_cfg["shift_scale_rotate"].get("scale_limit", 0.08)),
                rotate_limit=float(aug_cfg["shift_scale_rotate"].get("rotate_limit", 8)),
                border_mode=0,
                p=float(aug_cfg["shift_scale_rotate"].get("p", 0.2)),
            )
        )
    if enabled("random_brightness_contrast"):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(aug_cfg["random_brightness_contrast"].get("brightness_limit", 0.15)),
                contrast_limit=float(aug_cfg["random_brightness_contrast"].get("contrast_limit", 0.15)),
                p=float(aug_cfg["random_brightness_contrast"].get("p", 0.3)),
            )
        )
    if enabled("hue_saturation_value"):
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=int(aug_cfg["hue_saturation_value"].get("hue_shift_limit", 6)),
                sat_shift_limit=int(aug_cfg["hue_saturation_value"].get("sat_shift_limit", 10)),
                val_shift_limit=int(aug_cfg["hue_saturation_value"].get("val_shift_limit", 8)),
                p=float(aug_cfg["hue_saturation_value"].get("p", 0.2)),
            )
        )
    if enabled("blur"):
        transforms.append(
            A.Blur(
                blur_limit=int(aug_cfg["blur"].get("blur_limit", 3)),
                p=float(aug_cfg["blur"].get("p", 0.1)),
            )
        )
    if enabled("gauss_noise"):
        std_range = aug_cfg["gauss_noise"].get("std_range", [0.01, 0.03])
        transforms.append(
            A.GaussNoise(
                std_range=(float(std_range[0]), float(std_range[1])),
                p=float(aug_cfg["gauss_noise"].get("p", 0.08)),
            )
        )
    if enabled("clahe"):
        tile_grid_size = aug_cfg["clahe"].get("tile_grid_size", [8, 8])
        transforms.append(
            A.CLAHE(
                clip_limit=float(aug_cfg["clahe"].get("clip_limit", 2.0)),
                tile_grid_size=(int(tile_grid_size[0]), int(tile_grid_size[1])),
                p=float(aug_cfg["clahe"].get("p", 0.08)),
            )
        )

    if task == "detect":
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
        )
    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def copy_split(source_root: Path, destination_root: Path, split_name: str) -> None:
    """Copy an existing split without augmentation."""
    image_root = source_root / "images" / split_name
    label_root = source_root / "labels" / split_name
    if not image_root.exists():
        return
    for image_path in list_image_files(image_root):
        relative_path = image_path.relative_to(image_root)
        label_path = label_root / relative_path.with_suffix(".txt")
        output_image = destination_root / "images" / split_name / relative_path
        output_label = destination_root / "labels" / split_name / relative_path.with_suffix(".txt")
        ensure_dir(output_image.parent)
        ensure_dir(output_label.parent)
        shutil.copy2(image_path, output_image)
        if label_path.exists():
            shutil.copy2(label_path, output_label)


def clamp_norm(value: float) -> float:
    """Clamp a normalized coordinate into [0, 1]."""
    return max(0.0, min(1.0, value))


def main() -> None:
    """Augment only the train split and keep val/test untouched."""
    args = parse_args()
    logger = setup_logger("augment_yolo_dataset")
    config = load_yaml(args.config)
    copies_per_image = int(config.get("augmentation", {}).get("copies_per_image", 1))

    train_label_paths = sorted((args.input / "labels" / "train").rglob("*.txt")) if (args.input / "labels" / "train").exists() else []
    task = args.task if args.task != "auto" else infer_task_from_labels(train_label_paths)
    augmenter = build_augmentations(config, task)

    copy_split(args.input, args.output, "val")
    copy_split(args.input, args.output, "test")
    copy_split(args.input, args.output, "train")

    train_image_root = args.input / "images" / "train"
    train_label_root = args.input / "labels" / "train"
    output_image_root = ensure_dir(args.output / "images" / "train")
    output_label_root = ensure_dir(args.output / "labels" / "train")

    manifest_rows: list[dict[str, Any]] = []
    for image_path in list_image_files(train_image_root):
        relative_path = image_path.relative_to(train_image_root)
        label_path = train_label_root / relative_path.with_suffix(".txt")
        if not label_path.exists():
            continue

        image_rgb, error = safe_read_image(image_path)
        if error or image_rgb is None:
            logger.warning("Skipping unreadable train image during augmentation: %s", relative_path)
            continue

        parsed = parse_label_file(label_path, task=task)
        output_base = relative_path.with_suffix(".jpg")
        for copy_index in range(1, copies_per_image + 1):
            if task == "detect":
                bboxes = [obj.bbox for obj in parsed.objects if obj.bbox is not None]
                class_labels = [obj.class_id for obj in parsed.objects if obj.bbox is not None]
                transformed = augmenter(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
                if not transformed["bboxes"]:
                    continue
                label_lines = []
                for class_id, bbox in zip(transformed["class_labels"], transformed["bboxes"]):
                    label_lines.append(
                        f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                    )
            else:
                height, width = image_rgb.shape[:2]
                keypoints: list[tuple[float, float]] = []
                point_counts: list[int] = []
                class_ids: list[int] = []
                for obj in parsed.objects:
                    if obj.polygon is None:
                        continue
                    class_ids.append(obj.class_id)
                    point_counts.append(len(obj.polygon))
                    keypoints.extend([(x_coord * width, y_coord * height) for x_coord, y_coord in obj.polygon])
                transformed = augmenter(image=image_rgb, keypoints=keypoints)
                transformed_keypoints = transformed["keypoints"]
                cursor = 0
                label_lines = []
                for class_id, point_count in zip(class_ids, point_counts):
                    polygon = []
                    for _ in range(point_count):
                        x_coord, y_coord = transformed_keypoints[cursor]
                        cursor += 1
                        polygon.extend([f"{clamp_norm(x_coord / width):.6f}", f"{clamp_norm(y_coord / height):.6f}"])
                    if len(polygon) >= 6:
                        label_lines.append(f"{class_id} {' '.join(polygon)}")

            if not label_lines:
                continue

            aug_image_rel = output_base.with_name(f"{output_base.stem}__aug_{copy_index:02d}.jpg")
            aug_label_rel = aug_image_rel.with_suffix(".txt")
            aug_image_path = output_image_root / aug_image_rel
            aug_label_path = output_label_root / aug_label_rel
            ensure_dir(aug_image_path.parent)
            ensure_dir(aug_label_path.parent)
            save_image(aug_image_path, transformed["image"])
            aug_label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            manifest_rows.append(
                {
                    "source_image": relative_posix(image_path, train_image_root),
                    "augmented_image": relative_posix(aug_image_path, args.output),
                    "task": task,
                }
            )

    write_json(args.output / "augmentation_manifest.json", {"task": task, "items": manifest_rows})
    write_csv(args.output / "augmentation_manifest.csv", manifest_rows)
    logger.info("Offline augmentation complete. Added %s augmented train images.", len(manifest_rows))


if __name__ == "__main__":
    main()

