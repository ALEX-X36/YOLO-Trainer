"""
YOLO Trainer — Utility Functions
Shared helpers for file I/O, YAML handling, image statistics, and directory management.
"""

import os
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


def read_yaml(path: str) -> dict:
    """Read a YAML file and return its contents as a dict. Returns empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def write_yaml(path: str, data: dict) -> bool:
    """Write a dict to a YAML file. Returns True on success."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception:
        return False


def count_images(dir_path: str) -> int:
    """Count image files (jpg, jpeg, png, bmp, tiff, webp) recursively in a directory."""
    if not os.path.isdir(dir_path):
        return 0
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    count = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                count += 1
    return count


def count_labels(dir_path: str) -> int:
    """Count YOLO label files (.txt) recursively in a directory."""
    if not os.path.isdir(dir_path):
        return 0
    count = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith(".txt"):
                count += 1
    return count


def format_metric_name(key: str) -> str:
    """Convert Ultralytics metric key to a human-readable label."""
    mapping = {
        "metrics/mAP50(B)": "mAP@50",
        "metrics/mAP50-95(B)": "mAP@50-95",
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "train/box_loss": "Box Loss",
        "train/cls_loss": "Cls Loss",
        "train/dfl_loss": "DFL Loss",
        "val/box_loss": "Val Box Loss",
        "val/cls_loss": "Val Cls Loss",
        "val/dfl_loss": "Val DFL Loss",
    }
    return mapping.get(key, key)


def ensure_dir(path: str) -> str:
    """Ensure a directory exists; create if necessary. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def clean_old_outputs(max_age_days: int = 30) -> int:
    """Remove training outputs older than max_age_days. Returns count of removed runs."""
    if not os.path.isdir(ensure_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))):
        return 0
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    return _clean_dir_by_age(outputs_dir, max_age_days)


def _clean_dir_by_age(directory: str, max_age_days: int) -> int:
    """Remove subdirectories older than max_age_days. Returns count removed."""
    import time
    cutoff = time.time() - (max_age_days * 86400)
    removed = 0
    try:
        for entry in os.scandir(directory):
            if entry.is_dir():
                try:
                    mtime = entry.stat().st_mtime
                    if mtime < cutoff:
                        shutil.rmtree(entry.path)
                        removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    return removed


def generate_run_name(dataset_name: str, model_name: str) -> str:
    """Generate a unique run name: dataset_model_YYYYMMDD_HHMMSS."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset = dataset_name.replace(" ", "_").replace("/", "_")
    safe_model = model_name.replace(" ", "_")
    return f"{safe_dataset}_{safe_model}_{timestamp}"


def list_subdirs(directory: str) -> list[str]:
    """List subdirectory names in a directory, sorted by modification time descending."""
    if not os.path.isdir(directory):
        return []
    entries = []
    try:
        for entry in os.scandir(directory):
            if entry.is_dir():
                entries.append((entry.name, entry.stat().st_mtime))
    except OSError:
        return []
    entries.sort(key=lambda x: x[1], reverse=True)
    return [e[0] for e in entries]


def get_dir_size_mb(directory: str) -> float:
    """Return total size of a directory in megabytes."""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    continue
    except OSError:
        pass
    return total / (1024 * 1024)


def format_size(size_mb: float) -> str:
    """Format a size in MB to a human-readable string."""
    if size_mb < 1:
        return f"{size_mb * 1024:.0f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb / 1024:.1f} GB"
