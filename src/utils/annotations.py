"""Helpers for YOLO detection and segmentation annotations."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from utils.common import relative_posix


@dataclass(slots=True)
class AnnotationObject:
    """Single YOLO annotation object."""

    class_id: int
    bbox: tuple[float, float, float, float] | None = None
    polygon: list[tuple[float, float]] | None = None


@dataclass(slots=True)
class ParsedLabel:
    """Parsed label file payload."""

    task: str
    objects: list[AnnotationObject]


def infer_task_from_line(line: str) -> str:
    """Infer YOLO task type from a single label line."""
    tokens = line.strip().split()
    if len(tokens) == 5:
        return "detect"
    if len(tokens) >= 7 and (len(tokens) - 1) % 2 == 0:
        return "segment"
    raise ValueError(f"Cannot infer task from line: {line!r}")


def parse_annotation_line(line: str, task: str = "auto") -> AnnotationObject:
    """Parse one YOLO annotation line."""
    stripped = line.strip()
    if not stripped:
        raise ValueError("Annotation line is empty.")

    tokens = stripped.split()
    line_task = infer_task_from_line(stripped) if task == "auto" else task
    class_id = int(float(tokens[0]))

    if line_task == "detect":
        if len(tokens) != 5:
            raise ValueError("Detection label must contain 5 values.")
        bbox = tuple(float(value) for value in tokens[1:5])
        return AnnotationObject(class_id=class_id, bbox=bbox)

    if (len(tokens) - 1) % 2 != 0 or len(tokens) < 7:
        raise ValueError("Segmentation label must contain at least 3 polygon points.")

    coords = [float(value) for value in tokens[1:]]
    polygon = [(coords[index], coords[index + 1]) for index in range(0, len(coords), 2)]
    return AnnotationObject(class_id=class_id, polygon=polygon)


def parse_label_file(label_path: str | Path, task: str = "auto") -> ParsedLabel:
    """Parse a YOLO label file while inferring the task if needed."""
    path = Path(label_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Label file is empty: {path}")
    inferred_task = infer_task_from_line(lines[0]) if task == "auto" else task
    objects = [parse_annotation_line(line, inferred_task) for line in lines]
    return ParsedLabel(task=inferred_task, objects=objects)


def infer_task_from_labels(label_paths: Iterable[Path]) -> str:
    """Infer dataset task from the first valid label file."""
    for label_path in label_paths:
        try:
            parsed = parse_label_file(label_path, task="auto")
            return parsed.task
        except Exception:
            continue
    return "detect"


def build_label_index(label_roots: list[Path]) -> dict[str, dict[str, list[Path] | Path]]:
    """Index labels by relative path and stem to match images flexibly."""
    relative_index: dict[str, Path] = {}
    stem_index: defaultdict[str, list[Path]] = defaultdict(list)

    for label_root in label_roots:
        if not label_root.exists():
            continue
        for label_path in label_root.rglob("*.txt"):
            relative_index[relative_posix(label_path, label_root)] = label_path
            stem_index[label_path.stem].append(label_path)

    return {"relative": relative_index, "stem": dict(stem_index)}


def resolve_label_path(
    image_path: Path,
    image_root: Path,
    label_index: dict[str, dict[str, list[Path] | Path]],
) -> Path | None:
    """Resolve a label path for an image using relative-path and stem fallbacks."""
    relative_index = label_index["relative"]
    stem_index = label_index["stem"]

    try:
        rel = image_path.relative_to(image_root)
    except ValueError:
        rel = Path(image_path.name)

    candidates = [rel.with_suffix(".txt")]
    if rel.parts and rel.parts[0] == "images":
        candidates.append(Path(*rel.parts[1:]).with_suffix(".txt"))
    if len(rel.parts) >= 2 and rel.parts[0] in {"train", "val", "test"}:
        candidates.append(rel.with_suffix(".txt"))

    for candidate in candidates:
        key = candidate.as_posix()
        if key in relative_index:
            return relative_index[key]  # type: ignore[return-value]

    stem_matches = stem_index.get(image_path.stem, [])
    if len(stem_matches) == 1:
        return stem_matches[0]
    return None


def compute_class_distribution(label_paths: Iterable[Path], task: str = "auto") -> Counter[int]:
    """Count class occurrences across YOLO label files."""
    counter: Counter[int] = Counter()
    for label_path in label_paths:
        try:
            parsed = parse_label_file(label_path, task=task)
        except Exception:
            continue
        counter.update(obj.class_id for obj in parsed.objects)
    return counter


def label_quality_score(label_path: Path | None) -> float:
    """Assign a simple quality score to a label file for dedup prioritization."""
    if label_path is None or not label_path.exists():
        return 0.0
    try:
        parsed = parse_label_file(label_path)
    except Exception:
        return 0.0
    score = float(len(parsed.objects) * 10)
    for obj in parsed.objects:
        if obj.polygon:
            score += len(obj.polygon)
        if obj.bbox:
            score += max(obj.bbox[2], obj.bbox[3]) * 10.0
    return score


def primary_class_id(label_path: Path | None) -> int | None:
    """Return the most common class id in a label file."""
    if label_path is None or not label_path.exists():
        return None
    try:
        parsed = parse_label_file(label_path)
    except Exception:
        return None
    counts = Counter(obj.class_id for obj in parsed.objects)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def serialize_label(parsed: ParsedLabel) -> str:
    """Serialize parsed YOLO annotations back to text."""
    lines: list[str] = []
    for obj in parsed.objects:
        if parsed.task == "detect" and obj.bbox is not None:
            payload = [obj.class_id, *obj.bbox]
            lines.append(" ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in payload))
        elif parsed.task == "segment" and obj.polygon is not None:
            values = [str(obj.class_id)]
            for x_coord, y_coord in obj.polygon:
                values.extend([f"{x_coord:.6f}", f"{y_coord:.6f}"])
            lines.append(" ".join(values))
    return "\n".join(lines) + "\n"

