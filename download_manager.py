"""
YOLO Trainer — Download Manager
Handles pre-downloading of YOLO pretrained model weights with progress tracking.
Communicates real-time download progress via a thread-safe queue.
"""

import os
import queue
import threading
import time
import glob
from typing import Optional

import torch
from ultralytics import YOLO

from config import MODEL_VARIANTS, MODEL_SIZES
from utils import format_size


def _get_torch_hub_checkpoints() -> str:
    """Get the torch hub checkpoints directory, creating it if needed."""
    hub_dir = torch.hub.get_dir()
    checkpoints = os.path.join(hub_dir, "checkpoints")
    os.makedirs(checkpoints, exist_ok=True)
    return checkpoints


def find_model_file(model_id: str) -> Optional[str]:
    """Search for a downloaded YOLO model in all known cache locations.

    Args:
        model_id: Model filename (e.g., 'yolov8n.pt')

    Returns:
        Absolute path to the cached model file, or None if not found.
    """
    # 1. Torch hub checkpoints (primary cache)
    checkpoints = _get_torch_hub_checkpoints()
    model_path = os.path.join(checkpoints, model_id)
    if os.path.isfile(model_path):
        return model_path

    # 2. Ultralytics cache in user home
    ultralytics_cache = os.path.join(os.path.expanduser("~"), ".cache", "ultralytics")
    if os.path.isdir(ultralytics_cache):
        for root, _dirs, files in os.walk(ultralytics_cache):
            if model_id in files:
                return os.path.join(root, model_id)

    # 3. Project root directory (current working dir)
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_id)
    if os.path.isfile(local_path):
        return local_path

    return None


def check_all_models() -> dict:
    """Check download status for all model variants defined in config.

    Returns:
        dict: {model_name: {"downloaded": bool, "path": str|None,
                            "size_mb": float|None, "model_id": str,
                            "expected_mb": float|None}}
    """
    result = {}
    for name, model_id in MODEL_VARIANTS.items():
        path = find_model_file(model_id)
        if path and os.path.isfile(path):
            actual_size = os.path.getsize(path) / (1024 * 1024)
            result[name] = {
                "downloaded": True,
                "path": path,
                "size_mb": round(actual_size, 1),
                "model_id": model_id,
                "expected_mb": MODEL_SIZES.get(model_id),
            }
        else:
            result[name] = {
                "downloaded": False,
                "path": None,
                "size_mb": None,
                "model_id": model_id,
                "expected_mb": MODEL_SIZES.get(model_id),
            }
    return result


def download_model(model_name: str, model_id: str, progress_queue: queue.Queue):
    """Download a YOLO pretrained model in a background thread.

    Uses Ultralytics' built-in download (via YOLO constructor) and monitors
    the cache file for progress updates.

    Args:
        model_name: Display name (e.g., 'YOLOv8n')
        model_id: Model filename (e.g., 'yolov8n.pt')
        progress_queue: Queue to report progress events.
            Event types:
                {"type": "start", "model_name": str, "model_id": str}
                {"type": "progress", "percent": float, "downloaded_mb": float,
                 "total_mb": float|None, "speed_mbps": float, "eta_s": float}
                {"type": "complete", "path": str, "size_mb": float,
                 "elapsed_s": float, "speed_mbps": float}
                {"type": "already_cached", "path": str}
                {"type": "error", "message": str}
    """
    # Check if already cached
    existing = find_model_file(model_id)
    if existing and os.path.isfile(existing):
        size_mb = os.path.getsize(existing) / (1024 * 1024)
        progress_queue.put({
            "type": "already_cached",
            "model_name": model_name,
            "model_id": model_id,
            "path": existing,
            "size_mb": round(size_mb, 1),
        })
        return

    progress_queue.put({
        "type": "start",
        "model_name": model_name,
        "model_id": model_id,
    })

    start_time = time.time()
    checkpoints = _get_torch_hub_checkpoints()
    target_path = os.path.join(checkpoints, model_id)

    # Shared result container (mutated by download thread)
    result = {"path": None, "error": None}

    def _do_download():
        try:
            model = YOLO(model_id)
            # Find where ultralytics put it
            result["path"] = find_model_file(model_id)
        except Exception as e:
            result["error"] = str(e)

    dl_thread = threading.Thread(target=_do_download, daemon=True)
    dl_thread.start()

    # Poll file size while download is in progress
    expected_size = MODEL_SIZES.get(model_id)
    last_size = 0
    last_time = start_time

    while dl_thread.is_alive():
        dl_thread.join(1.0)

        # Try to read current file size
        current_path = target_path
        tmp_path = target_path + ".part"

        if os.path.isfile(tmp_path):
            current_path = tmp_path
        elif not os.path.isfile(target_path):
            continue

        try:
            current_size = os.path.getsize(current_path)
        except OSError:
            continue

        now = time.time()
        elapsed = now - start_time
        size_mb = current_size / (1024 * 1024)

        # Calculate speed
        if now - last_time > 0:
            instant_speed = (current_size - last_size) / (now - last_time) / (1024 * 1024)
        else:
            instant_speed = 0
        avg_speed = size_mb / elapsed if elapsed > 0 else 0

        # Calculate ETA
        if expected_size:
            percent = min(current_size / (expected_size * 1024 * 1024) * 100, 99.9)
            remaining = expected_size - size_mb
            eta = remaining / avg_speed if avg_speed > 0 else 0
            total_mb = expected_size
        else:
            percent = 0
            eta = 0
            total_mb = None

        last_size = current_size
        last_time = now

        progress_queue.put({
            "type": "progress",
            "model_name": model_name,
            "percent": round(percent, 1),
            "downloaded_mb": round(size_mb, 1),
            "total_mb": total_mb,
            "speed_mbps": round(instant_speed if instant_speed > 0 else avg_speed, 1),
            "eta_s": round(eta, 0),
        })

    # Done — check result
    if result["error"]:
        progress_queue.put({
            "type": "error",
            "model_name": model_name,
            "model_id": model_id,
            "message": f"❌ {model_name} 下载失败: {result['error']}",
        })
        return

    elapsed = time.time() - start_time
    final_path = result["path"] or target_path

    if os.path.isfile(final_path):
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        avg_speed = size_mb / elapsed if elapsed > 0 else 0
        progress_queue.put({
            "type": "complete",
            "model_name": model_name,
            "model_id": model_id,
            "path": final_path,
            "size_mb": round(size_mb, 1),
            "elapsed_s": round(elapsed, 1),
            "speed_mbps": round(avg_speed, 1),
        })
    else:
        progress_queue.put({
            "type": "complete",
            "model_name": model_name,
            "model_id": model_id,
            "path": None,
            "size_mb": 0,
            "elapsed_s": round(elapsed, 1),
            "speed_mbps": 0,
        })
