"""Detect exact and near-duplicate images before YOLO dataset build."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image

from utils.annotations import build_label_index, label_quality_score, resolve_label_path
from utils.common import ensure_dir, list_image_files, relative_posix, setup_logger, sha256_file, write_csv, write_json
from utils.config import load_yaml
from utils.image_ops import compute_blur_score, image_size, safe_read_image


@dataclass(slots=True)
class ImageFingerprint:
    """Fingerprint and metadata for duplicate comparison."""

    image_path: Path
    relative_path: str
    sha256: str
    phash: imagehash.ImageHash
    width: int
    height: int
    blur_score: float
    label_quality: float
    label_path: Path | None

    @property
    def resolution_score(self) -> int:
        """Return the pixel-count score for prioritization."""
        return self.width * self.height


class UnionFind:
    """Minimal union-find implementation for grouping near-duplicates."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        """Find the representative for an index."""
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        """Union two indices."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Detect exact and near-duplicate images.")
    parser.add_argument("--input", type=Path, required=True, help="Input image directory.")
    parser.add_argument("--mode", choices=["report-only", "apply"], required=True, help="Reporting or apply mode.")
    parser.add_argument("--output", type=Path, required=True, help="Report directory or deduped image directory.")
    parser.add_argument("--labels", type=Path, default=Path("data/raw_labels"), help="Optional label directory.")
    parser.add_argument("--config", type=Path, default=Path("configs/thresholds.yaml"), help="Threshold YAML.")
    parser.add_argument("--report-dir", type=Path, default=Path("outputs"), help="Where to save duplicate reports.")
    return parser.parse_args()


def compute_fingerprints(input_root: Path, label_root: Path | None, logger) -> list[ImageFingerprint]:
    """Compute hashes and ranking metadata for all images."""
    image_paths = list_image_files(input_root)
    label_index = build_label_index([label_root] if label_root and label_root.exists() else [])
    fingerprints: list[ImageFingerprint] = []

    for image_path in image_paths:
        relative_path = relative_posix(image_path, input_root)
        image_rgb, error = safe_read_image(image_path)
        if error or image_rgb is None:
            logger.warning("Skipping unreadable image during dedup: %s", relative_path)
            continue

        label_path = resolve_label_path(image_path, input_root, label_index) if label_root and label_root.exists() else None
        with Image.fromarray(image_rgb) as pil_image:
            phash = imagehash.phash(pil_image)

        width, height = image_size(image_rgb)
        fingerprints.append(
            ImageFingerprint(
                image_path=image_path,
                relative_path=relative_path,
                sha256=sha256_file(image_path),
                phash=phash,
                width=width,
                height=height,
                blur_score=compute_blur_score(image_rgb),
                label_quality=label_quality_score(label_path),
                label_path=label_path,
            )
        )

    return fingerprints


def choose_keeper(group_members: list[ImageFingerprint]) -> ImageFingerprint:
    """Choose the best image to keep from a duplicate group."""
    ranked = sorted(
        group_members,
        key=lambda item: (item.resolution_score, item.label_quality, item.blur_score, -len(item.relative_path)),
        reverse=True,
    )
    return ranked[0]


def group_duplicates(fingerprints: list[ImageFingerprint], threshold: int) -> list[list[ImageFingerprint]]:
    """Group exact and near-duplicates using perceptual hash distance."""
    union_find = UnionFind(len(fingerprints))
    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            if fingerprints[left].sha256 == fingerprints[right].sha256:
                union_find.union(left, right)
                continue
            if fingerprints[left].phash - fingerprints[right].phash <= threshold:
                union_find.union(left, right)

    grouped: defaultdict[int, list[ImageFingerprint]] = defaultdict(list)
    for index, fingerprint in enumerate(fingerprints):
        grouped[union_find.find(index)].append(fingerprint)
    return [members for members in grouped.values() if len(members) > 1]


def apply_deduplication(
    groups: list[list[ImageFingerprint]],
    input_root: Path,
    output_root: Path,
    logger,
) -> list[dict[str, Any]]:
    """Copy keepers to the output root and duplicates to quarantine."""
    quarantine_root = ensure_dir(output_root / "quarantine")
    all_rows: list[dict[str, Any]] = []

    keepers = {choose_keeper(group).relative_path for group in groups}
    remove_set = {
        member.relative_path
        for group in groups
        for member in group
        if member.relative_path not in keepers
    }

    for image_path in list_image_files(input_root):
        relative_path = relative_posix(image_path, input_root)
        destination = output_root / relative_path
        if relative_path in remove_set:
            destination = quarantine_root / relative_path
        ensure_dir(destination.parent)
        shutil.copy2(image_path, destination)
        all_rows.append(
            {
                "image_path": relative_path,
                "copied_to": relative_posix(destination, output_root),
                "status": "removed_to_quarantine" if relative_path in remove_set else "kept",
            }
        )

    logger.info("Deduplicated data copied to %s", output_root)
    return all_rows


def main() -> None:
    """Detect duplicates and optionally apply deduplication."""
    args = parse_args()
    logger = setup_logger("deduplicate_images")
    config = load_yaml(args.config)
    threshold = int(config.get("duplicates", {}).get("perceptual_hash_distance_max", 6))

    fingerprints = compute_fingerprints(args.input, args.labels if args.labels.exists() else None, logger)
    duplicate_groups = group_duplicates(fingerprints, threshold)

    report_rows: list[dict[str, Any]] = []
    report_groups: list[dict[str, Any]] = []

    for index, group in enumerate(duplicate_groups, start=1):
        keeper = choose_keeper(group)
        duplicate_type = "exact" if len({member.sha256 for member in group}) == 1 else "near"
        group_id = f"dup_{index:04d}"
        members_payload: list[dict[str, Any]] = []

        for member in sorted(group, key=lambda item: item.relative_path):
            hamming_to_keeper = int(member.phash - keeper.phash)
            status = "keep" if member.relative_path == keeper.relative_path else "remove"
            row = {
                "group_id": group_id,
                "duplicate_type": duplicate_type,
                "image_path": member.relative_path,
                "status": status,
                "keeper_image": keeper.relative_path,
                "width": member.width,
                "height": member.height,
                "blur_score": round(member.blur_score, 4),
                "label_quality": round(member.label_quality, 4),
                "hamming_to_keeper": hamming_to_keeper,
            }
            report_rows.append(row)
            members_payload.append(row)

        report_groups.append(
            {
                "group_id": group_id,
                "duplicate_type": duplicate_type,
                "keeper_image": keeper.relative_path,
                "members": members_payload,
            }
        )

    summary = {
        "input_dir": args.input.as_posix(),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_images": sum(len(group) for group in duplicate_groups),
        "exact_groups": sum(group["duplicate_type"] == "exact" for group in report_groups),
        "near_groups": sum(group["duplicate_type"] == "near" for group in report_groups),
        "mode": args.mode,
    }
    report_payload = {"summary": summary, "groups": report_groups}

    report_dir = args.output if args.mode == "report-only" else args.report_dir
    write_json(report_dir / "duplicate_report.json", report_payload)
    write_csv(report_dir / "duplicate_report.csv", report_rows)

    if args.mode == "apply":
        apply_rows = apply_deduplication(duplicate_groups, args.input, args.output, logger)
        write_csv(args.output / "dedup_apply_manifest.csv", apply_rows)
        write_json(args.output / "dedup_apply_manifest.json", {"items": apply_rows})
    else:
        logger.info("Report-only mode finished. No files were moved.")


if __name__ == "__main__":
    main()

