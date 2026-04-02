"""Ultralytics YOLO training entrypoint kept idle until explicitly executed."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils.common import setup_logger
from utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO from a saved config.")
    parser.add_argument("--config", type=Path, default=Path("configs/yolo_train.yaml"), help="Training YAML config.")
    parser.add_argument("--data", type=Path, default=None, help="YOLO dataset.yaml path override.")
    parser.add_argument("--model", type=str, default=None, help="Model name override.")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size override.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size override.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override.")
    parser.add_argument("--workers", type=int, default=None, help="Worker count override.")
    parser.add_argument("--device", type=str, default=None, help="Device override, for example 0 or cpu.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
    return parser.parse_args()


def merge_runtime_config(base_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI overrides into the YAML config."""
    merged = dict(base_config)
    for key in ("model", "imgsz", "batch", "epochs", "workers", "device"):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    merged["data"] = str(args.data) if args.data is not None else base_config.get("data", "data/yolo/dataset.yaml")
    merged["resume"] = bool(args.resume)
    return merged


def main() -> None:
    """Train a YOLO model when the user explicitly invokes this script."""
    args = parse_args()
    logger = setup_logger("train_yolo")
    config = merge_runtime_config(load_yaml(args.config), args)

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("PyTorch is required before running training.") from exc

    if str(config.get("device", "0")) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Stop here and do not proceed to training.")

    try:
        from ultralytics import YOLO  # type: ignore

        logger.info("Launching training with model=%s data=%s", config["model"], config["data"])
        model = YOLO(config["model"])
        model.train(
            data=config["data"],
            imgsz=config.get("imgsz", 640),
            batch=config.get("batch", 8),
            epochs=config.get("epochs", 100),
            workers=config.get("workers", 2),
            device=config.get("device", 0),
            amp=config.get("amp", True),
            cache=config.get("cache", False),
            patience=config.get("patience", 20),
            pretrained=config.get("pretrained", True),
            project=config.get("project", "outputs/yolo"),
            name=config.get("name", "lesion_localization"),
            seed=config.get("seed", 42),
            save_period=config.get("save_period", -1),
            exist_ok=config.get("exist_ok", True),
            resume=config.get("resume", False),
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or "cuda" in message and "memory" in message:
            logger.error(
                "CUDA OOM detected. Try batch fallback 8 -> 6 -> 4 -> 2, and if needed reduce imgsz 640 -> 512."
            )
        raise


if __name__ == "__main__":
    main()

