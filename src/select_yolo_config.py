"""Select a safe Ultralytics YOLO training configuration for mid-range GPUs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils.common import setup_logger
from utils.config import save_yaml


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Select a safe YOLO training config.")
    parser.add_argument("--gpu_vram_gb", type=float, required=True, help="Available GPU VRAM in GB.")
    parser.add_argument("--ram_gb", type=float, required=True, help="Available system RAM in GB.")
    parser.add_argument("--task", choices=["detect", "segment"], required=True, help="YOLO task.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/yolo_train.yaml"),
        help="Output YAML config file.",
    )
    parser.add_argument("--prefer-family", choices=["auto", "yolo11", "yolo26"], default="auto", help="Model family.")
    return parser.parse_args()


def version_tuple(version_text: str) -> tuple[int, ...]:
    """Parse a dotted version string into integer parts."""
    parts: list[int] = []
    for part in version_text.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def detect_model_family(prefer_family: str) -> str:
    """Select YOLO11 or YOLO26 based on user preference and installed ultralytics version."""
    if prefer_family in {"yolo11", "yolo26"}:
        return prefer_family

    try:
        import ultralytics  # type: ignore

        if version_tuple(ultralytics.__version__) >= (8, 4, 0):
            return "yolo26"
    except Exception:
        pass
    return "yolo11"


def estimate_vram_usage_gb(task: str, batch: int, imgsz: int, amp: bool = True) -> float:
    """Use a conservative heuristic to estimate VRAM pressure for a nano model."""
    base_per_image = 0.55 if task == "detect" else 0.72
    scale = (imgsz / 640.0) ** 2
    amp_factor = 0.88 if amp else 1.0
    return 1.35 + (base_per_image * batch * scale * amp_factor)


def choose_training_config(gpu_vram_gb: float, ram_gb: float, task: str, prefer_family: str) -> dict[str, Any]:
    """Choose a safe training configuration for the available hardware."""
    family = detect_model_family(prefer_family)
    model = f"{family}n-seg" if task == "segment" else f"{family}n"
    imgsz = 640
    batch_candidates = [8, 6, 4, 2]
    amp = True
    cache = False
    workers = 2 if ram_gb <= 16 else 4

    safe_vram_budget = gpu_vram_gb * 0.82
    batch = batch_candidates[-1]
    selected_estimate = 0.0
    for candidate in batch_candidates:
        estimate = estimate_vram_usage_gb(task=task, batch=candidate, imgsz=imgsz, amp=amp)
        if estimate <= safe_vram_budget:
            batch = candidate
            selected_estimate = estimate
            break
    else:
        imgsz = 512
        for candidate in batch_candidates:
            estimate = estimate_vram_usage_gb(task=task, batch=candidate, imgsz=imgsz, amp=amp)
            if estimate <= safe_vram_budget:
                batch = candidate
                selected_estimate = estimate
                break
        else:
            batch = 2
            selected_estimate = estimate_vram_usage_gb(task=task, batch=batch, imgsz=imgsz, amp=amp)

    return {
        "task": task,
        "data": "khoanhvungla/dataset.yaml",
        "model": model,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "epochs": 100,
        "amp": amp,
        "cache": cache,
        "patience": 20,
        "pretrained": True,
        "device": 0,
        "project": "outputs/yolo",
        "name": "lesion_localization",
        "seed": 42,
        "save_period": -1,
        "exist_ok": True,
        "oom_estimate_gb": round(selected_estimate, 2),
    }


def main() -> None:
    """Generate and save the selected YOLO config."""
    args = parse_args()
    logger = setup_logger("select_yolo_config")
    config = choose_training_config(
        gpu_vram_gb=args.gpu_vram_gb,
        ram_gb=args.ram_gb,
        task=args.task,
        prefer_family=args.prefer_family,
    )
    oom_estimate = config.pop("oom_estimate_gb")
    save_yaml(args.output, config, header_comment=f"task: {config['task']}")
    logger.info(
        "Saved config to %s | model=%s | imgsz=%s | batch=%s | estimated_vram_gb=%s",
        args.output,
        config["model"],
        config["imgsz"],
        config["batch"],
        oom_estimate,
    )


if __name__ == "__main__":
    main()

