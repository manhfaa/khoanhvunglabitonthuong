"""Validate YOLO detection or segmentation annotations before dataset build."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from utils.annotations import build_label_index, infer_task_from_labels, parse_label_file, resolve_label_path
from utils.common import flatten_counter, list_image_files, relative_posix, setup_logger, write_csv, write_json
from utils.image_ops import image_size, safe_read_image


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Validate YOLO annotations.")
    parser.add_argument("--images", type=Path, required=True, help="Image directory.")
    parser.add_argument("--labels", type=Path, required=True, help="Label directory.")
    parser.add_argument("--task", choices=["auto", "detect", "segment"], default="auto", help="YOLO task type.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Report output directory.")
    parser.add_argument("--num-classes", type=int, default=None, help="Optional number of classes for validation.")
    return parser.parse_args()


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Compute polygon area using the shoelace formula."""
    area = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def main() -> None:
    """Validate annotations and write detailed reports."""
    args = parse_args()
    logger = setup_logger("validate_annotations")
    image_paths = list_image_files(args.images)
    label_paths = sorted(args.labels.rglob("*.txt"))
    label_index = build_label_index([args.labels])
    task = args.task if args.task != "auto" else infer_task_from_labels(label_paths)

    error_rows: list[dict[str, Any]] = []
    summary_counter: Counter[str] = Counter()
    matched_labels: set[Path] = set()

    for image_path in image_paths:
        image_rgb, read_error = safe_read_image(image_path)
        image_rel = relative_posix(image_path, args.images)
        if read_error or image_rgb is None:
            summary_counter["image_read_error"] += 1
            error_rows.append({"image_path": image_rel, "label_path": "", "error": read_error or "read_error"})
            continue

        width, height = image_size(image_rgb)
        label_path = resolve_label_path(image_path, args.images, label_index)
        if label_path is None:
            summary_counter["missing_label"] += 1
            error_rows.append({"image_path": image_rel, "label_path": "", "error": "missing_label"})
            continue

        matched_labels.add(label_path)
        try:
            parsed = parse_label_file(label_path, task=task)
        except Exception as exc:
            summary_counter["parse_error"] += 1
            error_rows.append(
                {
                    "image_path": image_rel,
                    "label_path": relative_posix(label_path, args.labels),
                    "error": f"parse_error: {exc}",
                }
            )
            continue

        for index, obj in enumerate(parsed.objects):
            if obj.class_id < 0:
                summary_counter["invalid_class_id"] += 1
                error_rows.append(
                    {
                        "image_path": image_rel,
                        "label_path": relative_posix(label_path, args.labels),
                        "error": f"negative_class_id at object {index}",
                    }
                )

            if args.num_classes is not None and obj.class_id >= args.num_classes:
                summary_counter["invalid_class_id"] += 1
                error_rows.append(
                    {
                        "image_path": image_rel,
                        "label_path": relative_posix(label_path, args.labels),
                        "error": f"class_id {obj.class_id} >= num_classes {args.num_classes}",
                    }
                )

            if task == "detect" and obj.bbox is not None:
                x_center, y_center, box_width, box_height = obj.bbox
                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                    summary_counter["bbox_center_out_of_range"] += 1
                    error_rows.append(
                        {
                            "image_path": image_rel,
                            "label_path": relative_posix(label_path, args.labels),
                            "error": f"bbox_center_out_of_range at object {index}",
                        }
                    )
                if box_width <= 0.0 or box_height <= 0.0:
                    summary_counter["bbox_non_positive"] += 1
                    error_rows.append(
                        {
                            "image_path": image_rel,
                            "label_path": relative_posix(label_path, args.labels),
                            "error": f"bbox_non_positive at object {index}",
                        }
                    )
                x1 = x_center - box_width / 2.0
                y1 = y_center - box_height / 2.0
                x2 = x_center + box_width / 2.0
                y2 = y_center + box_height / 2.0
                if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0:
                    summary_counter["bbox_out_of_bounds"] += 1
                    error_rows.append(
                        {
                            "image_path": image_rel,
                            "label_path": relative_posix(label_path, args.labels),
                            "error": f"bbox_out_of_bounds at object {index}",
                        }
                    )

            if task == "segment" and obj.polygon is not None:
                if len(obj.polygon) < 3:
                    summary_counter["polygon_too_few_points"] += 1
                    error_rows.append(
                        {
                            "image_path": image_rel,
                            "label_path": relative_posix(label_path, args.labels),
                            "error": f"polygon_too_few_points at object {index}",
                        }
                    )
                if any(x_coord < 0.0 or x_coord > 1.0 or y_coord < 0.0 or y_coord > 1.0 for x_coord, y_coord in obj.polygon):
                    summary_counter["polygon_out_of_bounds"] += 1
                    error_rows.append(
                        {
                            "image_path": image_rel,
                            "label_path": relative_posix(label_path, args.labels),
                            "error": f"polygon_out_of_bounds at object {index}",
                        }
                    )
                if polygon_area(obj.polygon) <= 1e-6:
                    summary_counter["polygon_zero_area"] += 1
                    error_rows.append(
                        {
                            "image_path": image_rel,
                            "label_path": relative_posix(label_path, args.labels),
                            "error": f"polygon_zero_area at object {index}",
                        }
                    )

    unmatched_labels = [label_path for label_path in label_paths if label_path not in matched_labels]
    for label_path in unmatched_labels:
        summary_counter["label_without_image"] += 1
        error_rows.append({"image_path": "", "label_path": relative_posix(label_path, args.labels), "error": "label_without_image"})

    summary = {
        "task": task,
        "total_images": len(image_paths),
        "total_labels": len(label_paths),
        "images_with_errors": len({row["image_path"] for row in error_rows if row["image_path"]}),
        "error_distribution": flatten_counter(summary_counter),
        "status": "passed" if not error_rows else "failed",
    }

    write_json(args.output / "annotation_validation_report.json", {"summary": summary, "errors": error_rows})
    write_csv(args.output / "annotation_validation_report.csv", error_rows)
    logger.info("Annotation validation complete. Status=%s", summary["status"])


if __name__ == "__main__":
    main()

