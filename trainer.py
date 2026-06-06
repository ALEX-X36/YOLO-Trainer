"""
YOLO Trainer — Training Engine
Manages YOLO model training in a background thread with start/stop control.
Communicates real-time progress via a thread-safe queue.
"""

import os
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Optional

import torch
from ultralytics import YOLO


@dataclass
class TrainingState:
    """State object shared between the training thread and the Gradio UI."""

    thread: Optional[threading.Thread] = None
    stop_flag: threading.Event = field(default_factory=threading.Event)
    model: Optional[YOLO] = None
    log_queue: queue.Queue = field(default_factory=queue.Queue)
    current_epoch: int = 0
    total_epochs: int = 0
    metrics: dict = field(default_factory=dict)
    status: str = "idle"  # idle | running | completed | error | stopped
    error_message: str = ""
    run_name: str = ""
    output_dir: str = ""


def run_training(params: dict, state: TrainingState):
    """Execute YOLO training in the current (background) thread.

    Args:
        params: Training configuration dict with keys:
            model_variant, data_yaml, epochs, batch, imgsz, lr0, lrf,
            optimizer, weight_decay, momentum, warmup_epochs,
            warmup_momentum, warmup_bias_lr, patience, close_mosaic,
            workers, seed, dropout, cos_lr, amp, multi_scale,
            pretrained, resume, project, name
        state: Shared TrainingState for communication.
    """
    try:
        state.status = "running"
        state.stop_flag.clear()
        state.current_epoch = 0
        state.metrics = {}
        state.error_message = ""

        model_id = params.get("model_variant", "yolov8n.pt")
        data_yaml = params.get("data_yaml", "")
        epochs = int(params.get("epochs", 100))
        batch = int(params.get("batch", 16))
        imgsz = int(params.get("imgsz", 640))
        lr0 = float(params.get("lr0", 0.01))
        lrf = float(params.get("lrf", 0.01))
        optimizer = params.get("optimizer", "auto")
        weight_decay = float(params.get("weight_decay", 0.0005))
        momentum = float(params.get("momentum", 0.937))
        warmup_epochs = int(params.get("warmup_epochs", 3))
        warmup_momentum = float(params.get("warmup_momentum", 0.8))
        warmup_bias_lr = float(params.get("warmup_bias_lr", 0.1))
        patience = int(params.get("patience", 50))
        close_mosaic = int(params.get("close_mosaic", 10))
        workers = int(params.get("workers", 8))
        seed = int(params.get("seed", 0))
        dropout = float(params.get("dropout", 0.0))
        cos_lr = bool(params.get("cos_lr", False))
        amp = bool(params.get("amp", True))
        multi_scale = bool(params.get("multi_scale", False))
        pretrained = bool(params.get("pretrained", True))
        resume = bool(params.get("resume", False))
        project = params.get("project", "outputs")
        name = params.get("name", "train")

        state.total_epochs = epochs
        state.run_name = name
        state.output_dir = os.path.join(project, name)

        state.log_queue.put({
            "type": "log",
            "message": f"🚀 开始训练 | 模型: {model_id} | 数据集: {data_yaml} | Epochs: {epochs} | Batch: {batch} | 图片尺寸: {imgsz}"
        })

        # Load model
        if pretrained and not resume:
            state.log_queue.put({"type": "log", "message": f"📥 加载预训练模型: {model_id}"})
            model = YOLO(model_id)
        else:
            state.log_queue.put({"type": "log", "message": f"📥 创建模型（从头训练）: {model_id}"})
            # Create model config path from the model name
            model = YOLO(model_id)  # Ultralytics auto-downloads if needed

        state.model = model

        # Register callbacks
        from callbacks import create_log_callback, setup_log_capture, remove_log_capture

        callbacks = create_log_callback(state.log_queue)
        log_handler = setup_log_capture(state.log_queue)

        try:
            # Build training arguments
            train_args = {
                "data": data_yaml,
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "lr0": lr0,
                "lrf": lrf,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "warmup_epochs": warmup_epochs,
                "warmup_momentum": warmup_momentum,
                "warmup_bias_lr": warmup_bias_lr,
                "patience": patience,
                "close_mosaic": close_mosaic,
                "workers": workers,
                "seed": seed,
                "dropout": dropout,
                "cos_lr": cos_lr,
                "amp": amp,
                "multi_scale": multi_scale,
                "project": project,
                "name": name,
                "exist_ok": True,
                "verbose": True,
            }

            state.log_queue.put({"type": "log", "message": f"⚙️  训练参数: {train_args}"})

            # Start training
            results = model.train(**train_args)

            # Check if stopped by user
            if state.stop_flag.is_set():
                state.status = "stopped"
                state.log_queue.put({"type": "log", "message": "⏹️ 训练已被用户停止"})
                state.log_queue.put({"type": "end", "reason": "stopped", "final_metrics": {}})
                return

            # Training completed
            state.status = "completed"

            # Extract final metrics from results
            final_metrics = {}
            if hasattr(results, "results_dict") and results.results_dict:
                final_metrics = results.results_dict
            state.metrics = final_metrics

            state.log_queue.put({"type": "log", "message": "✅ 训练完成！"})
            state.log_queue.put({
                "type": "end",
                "reason": "completed",
                "final_metrics": final_metrics,
            })

        finally:
            # Clean up
            remove_log_capture(log_handler)

    except Exception as e:
        state.status = "error"
        state.error_message = str(e)
        trace = traceback.format_exc()
        state.log_queue.put({"type": "log", "message": f"❌ 训练出错: {str(e)}"})
        state.log_queue.put({"type": "log", "message": trace})
        state.log_queue.put({"type": "error", "message": str(e)})


def stop_training(state: TrainingState):
    """Signal the training thread to stop at the next epoch boundary."""
    state.stop_flag.set()
    if state.model is not None:
        try:
            state.model.stop_training = True
        except Exception:
            pass
    state.log_queue.put({"type": "log", "message": "⏹️ 正在停止训练（当前epoch完成后停止）..."})


def get_device_info() -> str:
    """Return a human-readable string describing available compute devices."""
    parts = []
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem_total = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
            parts.append(f"GPU {i}: {name} ({mem_total:.1f} GB)")
    else:
        parts.append("CPU (无GPU)")
        try:
            import psutil
            mem = psutil.virtual_memory()
            parts.append(f"系统内存: {mem.total / (1024**3):.1f} GB")
        except ImportError:
            pass
    return " | ".join(parts)


def parse_training_results(output_dir: str) -> dict:
    """Parse training results from the Ultralytics output directory.

    Returns a dict with metrics summary, plot paths, and training history.
    """
    result = {
        "metrics": {},
        "plots": {},
        "csv_path": None,
        "best_pt": None,
        "last_pt": None,
    }

    if not os.path.isdir(output_dir):
        return result

    # Find results.csv
    csv_path = os.path.join(output_dir, "results.csv")
    if os.path.isfile(csv_path):
        result["csv_path"] = csv_path

    # Find weight files
    weights_dir = os.path.join(output_dir, "weights")
    if os.path.isdir(weights_dir):
        best_pt = os.path.join(weights_dir, "best.pt")
        last_pt = os.path.join(weights_dir, "last.pt")
        if os.path.isfile(best_pt):
            result["best_pt"] = best_pt
        if os.path.isfile(last_pt):
            result["last_pt"] = last_pt

    # Find plot images
    for fname in ["results.png", "confusion_matrix.png", "labels.jpg",
                   "train_batch0.jpg", "val_batch0_pred.jpg",
                   "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            name = fname.replace(".png", "").replace(".jpg", "")
            result["plots"][name] = fpath

    # Parse last line of results.csv for final metrics
    if result["csv_path"]:
        try:
            import pandas as pd
            df = pd.read_csv(result["csv_path"])
            if not df.empty:
                # Clean column names (strip whitespace)
                df.columns = df.columns.str.strip()
                last_row = df.iloc[-1].to_dict()
                result["metrics"] = last_row
        except Exception:
            pass

    return result
