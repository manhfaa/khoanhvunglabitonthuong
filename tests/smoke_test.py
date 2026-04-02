"""Lightweight smoke test for the training-prep pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from select_yolo_config import choose_training_config  # noqa: E402
from utils.annotations import infer_task_from_labels, parse_label_file  # noqa: E402
from utils.image_ops import compute_blur_score  # noqa: E402


def main() -> None:
    """Run a minimal smoke test without training."""
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        label_path = root / "sample.txt"
        label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

        parsed = parse_label_file(label_path, task="detect")
        assert parsed.task == "detect"
        assert parsed.objects[0].class_id == 0

        task = infer_task_from_labels([label_path])
        assert task == "detect"

        config = choose_training_config(gpu_vram_gb=8, ram_gb=16, task="detect", prefer_family="yolo11")
        assert config["batch"] in {2, 4, 6, 8}
        assert config["imgsz"] in {512, 640}

        dummy_blur = compute_blur_score(__import__("numpy").zeros((32, 32, 3), dtype="uint8"))
        assert dummy_blur >= 0.0

    print("Smoke test passed.")


if __name__ == "__main__":
    main()

