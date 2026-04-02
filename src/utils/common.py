"""Common helpers for filesystem, logging, and report export."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_image_file(path: str | Path) -> bool:
    """Return True when a path looks like a supported image file."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(root: str | Path) -> list[Path]:
    """Recursively list supported image files under a root directory."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.rglob("*") if path.is_file() and is_image_file(path))


def setup_logger(
    name: str,
    log_dir: str | Path = "outputs/logs",
    filename: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a console + file logger without duplicating handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    ensure_dir(log_dir)
    log_filename = filename or f"{name}.log"
    log_path = Path(log_dir) / log_filename

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Write JSON with UTF-8 encoding."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    return output_path


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    """Write rows to CSV, inferring field names when needed."""
    output_path = Path(path)
    ensure_dir(output_path.parent)

    if not rows and not fieldnames:
        fieldnames = ["message"]
        rows = [{"message": "no_rows"}]
    elif fieldnames is None:
        ordered_fields: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in ordered_fields:
                    ordered_fields.append(key)
        fieldnames = ordered_fields

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_counter(counter_like: dict[Any, Any]) -> dict[str, Any]:
    """Convert dict keys to strings for JSON/YAML serialization."""
    return {str(key): value for key, value in counter_like.items()}


def relative_posix(path: str | Path, root: str | Path | None = None) -> str:
    """Return a normalized POSIX-like relative path when possible."""
    target = Path(path)
    if root is not None:
        try:
            target = target.relative_to(root)
        except ValueError:
            pass
    return target.as_posix()


def safe_stem(path: str | Path) -> str:
    """Return a filename stem that is safe to use as an identifier."""
    return Path(path).stem.replace(" ", "_")


def chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield fixed-size chunks from a sequence."""
    for index in range(0, len(items), size):
        yield items[index : index + size]

