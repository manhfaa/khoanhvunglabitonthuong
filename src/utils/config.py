"""YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from utils.common import ensure_dir


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and always return a dictionary."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(payload)!r}")
    return payload


def save_yaml(path: str | Path, payload: dict[str, Any], header_comment: str | None = None) -> Path:
    """Save a YAML dictionary with optional header comment."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if header_comment:
        body = f"# {header_comment}\n{body}"
    output_path.write_text(body, encoding="utf-8")
    return output_path

