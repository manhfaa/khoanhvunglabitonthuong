"""Build a split YOLO dataset from cleaned, deduplicated, and synthesized assets."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from utils.annotations import (
    build_label_index,
    infer_task_from_labels,
    parse_label_file,
    primary_class_id,
    resolve_label_path,
)
from utils.common import ensure_dir, list_image_files, relative_posix, setup_logger, write_csv, write_json
from utils.config import save_yaml


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build a YOLO dataset with grouped splits.")
    parser.add_argument("--images", type=Path, required=True, help="Image root.")
    parser.add_argument("--labels", type=Path, required=True, help="Primary label root.")
    parser.add_argument("--output", type=Path, required=True, help="YOLO dataset output root.")
    parser.add_argument("--task", choices=["auto", "detect", "segment"], default="auto", help="YOLO task.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dedup-report",
        type=Path,
        default=Path("outputs/duplicate_report.json"),
        help="Optional duplicate report for group-aware splits.",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("outputs/annotation_validation_report.json"),
        help="Optional annotation validation report.",
    )
    parser.add_argument("--names", type=str, default=None, help="Comma-separated class names.")
    return parser.parse_args()


def load_duplicate_group_map(report_path: Path) -> dict[str, str]:
    """Load image-to-group mapping from the duplicate report JSON."""
    if not report_path.exists():
        return {}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    group_map: dict[str, str] = {}
    for group in payload.get("groups", []):
        for member in group.get("members", []):
            group_map[member["image_path"]] = group["group_id"]
    return group_map


def load_invalid_images(report_path: Path) -> set[str]:
    """Load invalid image paths from the validation report."""
    if not report_path.exists():
        return set()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        row["image_path"]
        for row in payload.get("errors", [])
        if row.get("image_path")
    }


def load_synthesis_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load synthesis metadata keyed by relative image path."""
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {item["image_path"]: item for item in payload.get("items", [])}


def candidate_keys(relative_image_path: str, parent_image: str | None = None) -> list[str]:
    """Generate matching keys for dedup and validation lookups."""
    keys = [relative_image_path]
    if relative_image_path.startswith("images/"):
        keys.append(relative_image_path[len("images/") :])
    if parent_image:
        keys.append(parent_image)
        if parent_image.startswith("images/"):
            keys.append(parent_image[len("images/") :])
    return list(dict.fromkeys(keys))


def can_stratify(labels: list[int | None]) -> bool:
    """Return True when group labels are sufficient for stratification."""
    filtered = [label for label in labels if label is not None]
    if len(set(filtered)) < 2:
        return False
    counts = Counter(filtered)
    return counts and min(counts.values()) >= 2


def split_group_ids(
    group_ids: list[str],
    group_labels: list[int | None],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Split group ids into train/val/test with best-effort stratification."""
    if not group_ids:
        return set(), set(), set()
    if len(group_ids) == 1:
        return set(group_ids), set(), set()

    effective_stratify = group_labels if can_stratify(group_labels) else None
    train_ids, temp_ids, _, temp_labels = train_test_split(
        group_ids,
        group_labels,
        train_size=train_ratio,
        random_state=seed,
        stratify=effective_stratify,
    )

    if not temp_ids or val_ratio + test_ratio <= 0:
        return set(train_ids), set(), set()
    if len(temp_ids) == 1:
        return set(train_ids), set(temp_ids), set()

    val_share = val_ratio / (val_ratio + test_ratio)
    temp_stratify = temp_labels if can_stratify(temp_labels) else None
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_share,
        random_state=seed,
        stratify=temp_stratify,
    )
    return set(train_ids), set(val_ids), set(test_ids)


def build_class_names(max_class_id: int, names_arg: str | None) -> dict[int, str]:
    """Build YOLO names mapping."""
    if names_arg:
        provided = [name.strip() for name in names_arg.split(",") if name.strip()]
        return {index: provided[index] if index < len(provided) else f"class_{index}" for index in range(max_class_id + 1)}
    if max_class_id == 0:
        return {0: "lesion"}
    return {index: f"class_{index}" for index in range(max_class_id + 1)}


def destination_relative(relative_image_path: str) -> Path:
    """Normalize image-relative paths for YOLO split folders."""
    path = Path(relative_image_path)
    if path.parts and path.parts[0] == "images":
        path = Path(*path.parts[1:])
    return path


def main() -> None:
    """Build the grouped YOLO dataset."""
    args = parse_args()
    logger = setup_logger("build_yolo_dataset")
    image_paths = list_image_files(args.images)

    label_roots = [args.labels]
    internal_label_root = args.images / "labels"
    if internal_label_root.exists():
        label_roots.insert(0, internal_label_root)
    label_index = build_label_index(label_roots)

    task = args.task
    all_label_paths: list[Path] = []
    for label_root in label_roots:
        if label_root.exists():
            all_label_paths.extend(sorted(label_root.rglob("*.txt")))
    if task == "auto":
        task = infer_task_from_labels(all_label_paths)

    duplicate_group_map = load_duplicate_group_map(args.dedup_report)
    invalid_images = load_invalid_images(args.validation_report)
    synthesis_manifest = load_synthesis_manifest(args.images / "synthesis_manifest.json")

    records: list[dict[str, Any]] = []
    max_class_id = 0

    for image_path in image_paths:
        relative_image_path = relative_posix(image_path, args.images)
        manifest_item = synthesis_manifest.get(relative_image_path, {})
        if any(key in invalid_images for key in candidate_keys(relative_image_path, manifest_item.get("parent_image"))):
            logger.warning("Skipping invalid item from validation report: %s", relative_image_path)
            continue

        label_path = resolve_label_path(image_path, args.images, label_index)
        if label_path is None or not label_path.exists():
            logger.warning("Skipping image without label: %s", relative_image_path)
            continue

        try:
            parsed = parse_label_file(label_path, task=task)
        except Exception as exc:
            logger.warning("Skipping image with invalid label %s: %s", label_path, exc)
            continue

        class_ids = sorted({obj.class_id for obj in parsed.objects})
        if not class_ids:
            continue
        max_class_id = max(max_class_id, max(class_ids))
        primary_class = primary_class_id(label_path)
        parent_image = manifest_item.get("parent_image")

        group_id = None
        for key in candidate_keys(relative_image_path, parent_image):
            if key in duplicate_group_map:
                group_id = duplicate_group_map[key]
                break
        if group_id is None:
            group_id = parent_image or relative_image_path

        records.append(
            {
                "image_path": image_path,
                "relative_image_path": relative_image_path,
                "label_path": label_path,
                "class_ids": class_ids,
                "primary_class": primary_class,
                "group_id": group_id,
                "is_synthetic": bool(manifest_item.get("is_synthetic", False)),
            }
        )

    if not records:
        raise RuntimeError("No valid image/label pairs were found to build the YOLO dataset.")

    forced_train_groups = {record["group_id"] for record in records if record["is_synthetic"]}
    group_to_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_to_records[record["group_id"]].append(record)

    split_candidates = {
        group_id: group_records
        for group_id, group_records in group_to_records.items()
        if group_id not in forced_train_groups
    }
    group_ids = list(split_candidates)
    group_labels = [
        Counter(record["primary_class"] for record in split_candidates[group_id] if record["primary_class"] is not None).most_common(1)[0][0]
        if any(record["primary_class"] is not None for record in split_candidates[group_id])
        else None
        for group_id in group_ids
    ]

    train_groups, val_groups, test_groups = split_group_ids(
        group_ids=group_ids,
        group_labels=group_labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_groups |= forced_train_groups

    split_map: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for group_id, group_records in group_to_records.items():
        if group_id in train_groups:
            split_map["train"].extend(group_records)
        elif group_id in val_groups:
            split_map["val"].extend(group_records)
        else:
            split_map["test"].extend(group_records)

    for split_name in ("train", "val", "test"):
        ensure_dir(args.output / "images" / split_name)
        ensure_dir(args.output / "labels" / split_name)
        for record in split_map[split_name]:
            destination_rel = destination_relative(record["relative_image_path"])
            image_destination = args.output / "images" / split_name / destination_rel
            label_destination = args.output / "labels" / split_name / destination_rel.with_suffix(".txt")
            ensure_dir(image_destination.parent)
            ensure_dir(label_destination.parent)
            shutil.copy2(record["image_path"], image_destination)
            shutil.copy2(record["label_path"], label_destination)

    class_names = build_class_names(max_class_id=max_class_id, names_arg=args.names)
    dataset_yaml = {
        "path": relative_posix(args.output, Path.cwd()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    save_yaml(args.output / "dataset.yaml", dataset_yaml, header_comment=f"task: {task}")

    manifest_rows = []
    for split_name, split_records in split_map.items():
        for record in split_records:
            manifest_rows.append(
                {
                    "split": split_name,
                    "image_path": record["relative_image_path"],
                    "label_path": record["label_path"].name,
                    "group_id": record["group_id"],
                    "primary_class": record["primary_class"],
                    "is_synthetic": record["is_synthetic"],
                }
            )

    summary = {
        "task": task,
        "total_items": len(records),
        "train_items": len(split_map["train"]),
        "val_items": len(split_map["val"]),
        "test_items": len(split_map["test"]),
        "forced_train_groups": len(forced_train_groups),
        "class_names": {str(key): value for key, value in class_names.items()},
    }
    write_json(args.output / "split_manifest.json", {"summary": summary, "items": manifest_rows})
    write_csv(args.output / "split_manifest.csv", manifest_rows)
    logger.info("YOLO dataset built at %s with task=%s", args.output, task)


if __name__ == "__main__":
    main()

