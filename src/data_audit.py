"""Audit raw image data before building a YOLO dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from utils.annotations import build_label_index, compute_class_distribution, resolve_label_path
from utils.common import flatten_counter, list_image_files, relative_posix, setup_logger, write_csv, write_json
from utils.image_ops import compute_blur_score, image_size, safe_read_image


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Audit raw image data.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Input image directory.")
    parser.add_argument("--labels", type=Path, default=Path("data/raw_labels"), help="Optional label directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory for reports.")
    parser.add_argument("--min-size", type=int, default=384, help="Minimum accepted width/height for audit flagging.")
    parser.add_argument("--blur-threshold", type=float, default=90.0, help="Blur threshold for audit statistics.")
    return parser.parse_args()


def summarize_dimensions(widths: list[int], heights: list[int]) -> dict[str, Any]:
    """Compute min/max/avg dimension statistics."""
    if not widths or not heights:
        return {
            "min_width": 0,
            "max_width": 0,
            "avg_width": 0.0,
            "min_height": 0,
            "max_height": 0,
            "avg_height": 0.0,
        }

    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "avg_width": round(mean(widths), 2),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_height": round(mean(heights), 2),
    }


def source_name(image_path: Path, input_root: Path) -> str:
    """Infer a simple source bucket from the top-level relative directory."""
    rel = Path(relative_posix(image_path, input_root))
    if len(rel.parts) <= 1:
        return "root"
    return rel.parts[0]


def main() -> None:
    """Run the audit and export JSON/CSV summaries."""
    args = parse_args()
    logger = setup_logger("data_audit")
    image_paths = list_image_files(args.input)
    label_roots = [args.labels] if args.labels.exists() else []
    label_index = build_label_index(label_roots)

    logger.info("Scanning %s images under %s", len(image_paths), args.input)

    per_image_rows: list[dict[str, Any]] = []
    widths: list[int] = []
    heights: list[int] = []
    valid_blur_scores: list[float] = []
    source_distribution: Counter[str] = Counter()
    corrupted_count = 0
    too_small_count = 0

    for image_path in image_paths:
        image_rgb, error = safe_read_image(image_path)
        row: dict[str, Any] = {
            "image_path": relative_posix(image_path, args.input),
            "source": source_name(image_path, args.input),
            "status": "ok",
            "error": "",
            "width": 0,
            "height": 0,
            "blur_score": 0.0,
            "too_small": False,
            "label_found": False,
            "class_ids": "",
            "object_count": 0,
        }
        source_distribution.update([row["source"]])

        if error or image_rgb is None:
            corrupted_count += 1
            row["status"] = "corrupted"
            row["error"] = error or "unknown_read_error"
            per_image_rows.append(row)
            continue

        width, height = image_size(image_rgb)
        blur_score = compute_blur_score(image_rgb)
        too_small = width < args.min_size or height < args.min_size

        row["width"] = width
        row["height"] = height
        row["blur_score"] = round(blur_score, 4)
        row["too_small"] = too_small

        if too_small:
            too_small_count += 1
            row["status"] = "too_small"

        widths.append(width)
        heights.append(height)
        valid_blur_scores.append(blur_score)

        label_path = resolve_label_path(image_path, args.input, label_index) if label_roots else None
        if label_path and label_path.exists():
            row["label_found"] = True
            try:
                class_distribution = compute_class_distribution([label_path])
                row["class_ids"] = ",".join(str(class_id) for class_id in sorted(class_distribution))
                row["object_count"] = int(sum(class_distribution.values()))
            except Exception as exc:
                row["error"] = f"label_parse_error: {exc}"
                row["status"] = "label_warning" if row["status"] == "ok" else row["status"]

        per_image_rows.append(row)

    all_label_paths = list(args.labels.rglob("*.txt")) if args.labels.exists() else []
    class_distribution = compute_class_distribution(all_label_paths) if all_label_paths else Counter()
    blurred_ratio = (
        round(sum(score < args.blur_threshold for score in valid_blur_scores) / len(valid_blur_scores), 4)
        if valid_blur_scores
        else 0.0
    )

    summary = {
        "input_dir": args.input.as_posix(),
        "total_images": len(image_paths),
        "corrupted_images": corrupted_count,
        "too_small_images": too_small_count,
        "blurred_ratio": blurred_ratio,
        "dimension_summary": summarize_dimensions(widths, heights),
        "class_distribution": flatten_counter(class_distribution),
        "source_distribution": flatten_counter(source_distribution),
    }

    json_payload = {"summary": summary, "images": per_image_rows}
    write_json(args.output / "audit_report.json", json_payload)
    write_csv(args.output / "audit_report.csv", per_image_rows)
    logger.info("Audit completed. JSON and CSV reports saved to %s", args.output)


if __name__ == "__main__":
    main()

