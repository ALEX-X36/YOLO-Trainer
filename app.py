"""
YOLO 鱼类识别训练器 — Main Entry Point
Launches the Gradio WebUI for YOLO model training.

Usage:
    python app.py
    python app.py --port 7860 --share
"""

import os
import sys
import queue
import threading
import time
import argparse
from datetime import datetime

import gradio as gr
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for thread safety

# Ensure the Trainer directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_VARIANTS,
    EXPORT_FORMATS,
    IMAGE_SIZE_CHOICES,
    OPTIMIZER_CHOICES,
    OUTPUTS_DIR,
    EXPORTS_DIR,
    DEFAULT_TRAINING_PARAMS,
    UPLOAD_REQUIREMENTS,
)
from utils import ensure_dir, generate_run_name, list_subdirs, format_size, get_dir_size_mb
from dataset_manager import (
    validate_zip_structure,
    extract_dataset,
    list_datasets,
    get_dataset_info,
    delete_dataset,
    get_dataset_data_yaml_path,
)
from download_manager import (
    check_all_models,
    download_model,
    find_model_file,
)
from trainer import (
    TrainingState,
    run_training,
    stop_training,
    get_device_info,
    parse_training_results,
)


# ---- Global training state (persists across requests) ----
_active_training: TrainingState | None = None
_ui_refresh_active = False

# ---- Global download state ----
_download_queue: queue.Queue = queue.Queue()
_download_active = False
_current_model_name = ""
_current_model_id = ""
_download_log_lines: list[str] = []
_download_thread: threading.Thread | None = None


# ================================================================
# CALLBACK HANDLERS
# ================================================================

def handle_upload(zip_file, dataset_name):
    """Handle dataset ZIP upload and validation."""
    if zip_file is None:
        return "⚠️ 请先选择 ZIP 文件", gr.update(), gr.update()

    if not dataset_name or not dataset_name.strip():
        dataset_name = os.path.splitext(os.path.basename(zip_file.name))[0]

    # Validate
    zip_path = zip_file.name
    validation = validate_zip_structure(zip_path)

    if not validation["valid"]:
        errors = "\n".join(f"- {e}" for e in validation["errors"])
        status = f"❌ **验证失败**\n\n{errors}"
        return status, gr.update(), gr.update()

    # Extract
    try:
        data_yaml_path = extract_dataset(zip_path, dataset_name.strip())
        stats = validation["stats"]
        status = (
            f"✅ **数据集 '{dataset_name}' 上传成功！**\n\n"
            f"- 图片总数: {stats['image_count']} (训练: {stats['train_images']}, 验证: {stats['val_images']})\n"
            f"- 类别数: {stats['class_count']}\n"
            f"- 类别: {', '.join(stats['class_names'][:20])}\n"
            f"- data.yaml: {data_yaml_path}"
        )
    except Exception as e:
        status = f"❌ **解压失败**: {str(e)}"

    # Refresh dataset list
    datasets = list_datasets()
    return status, gr.update(choices=datasets), gr.update()


def handle_refresh_datasets():
    """Refresh the dataset dropdown list."""
    datasets = list_datasets()
    return gr.update(choices=datasets)


def handle_select_dataset(dataset_name):
    """Show detailed info for a selected dataset."""
    if not dataset_name:
        return "选择数据集查看详情..."
    info = get_dataset_info(dataset_name)
    if "error" in info:
        return f"❌ {info['error']}"

    return (
        f"### 📊 {info['name']}\n\n"
        f"| 属性 | 值 |\n|------|------|\n"
        f"| 图片总数 | {info['total_images']} |\n"
        f"| 训练集 | {info['train_images']} |\n"
        f"| 验证集 | {info['val_images']} |\n"
        f"| 类别数 | {info['class_count']} |\n"
        f"| 类别 | {', '.join(info['class_names'][:30])} |\n"
        f"| 大小 | {format_size(info['size_mb'])} |\n"
        f"| 路径 | {info['path']} |\n"
    )


def handle_delete_dataset(dataset_name):
    """Delete a dataset."""
    if not dataset_name:
        return "⚠️ 请先选择要删除的数据集", gr.update()
    if delete_dataset(dataset_name):
        datasets = list_datasets()
        return f"✅ 数据集 '{dataset_name}' 已删除", gr.update(choices=datasets)
    return f"❌ 删除 '{dataset_name}' 失败", gr.update()


def handle_start_training(
    dataset_selector, model_selector,
    epochs_slider, batch_slider, imgsz_dropdown, optimizer_dropdown,
    lr0_number, lrf_number, weight_decay_number, momentum_number,
    warmup_slider, patience_slider, close_mosaic_number,
    workers_slider, seed_number,
    pretrained_cb, amp_cb, cos_lr_cb, multi_scale_cb, resume_cb,
    dropout_number, warmup_momentum_number, warmup_bias_lr_number,
):
    """Start YOLO training in a background thread."""
    global _active_training

    # Validate inputs
    if _active_training is not None and _active_training.status == "running":
        return (
            gr.update(),
            "**状态**: 🔴 训练已在运行中！请先停止当前训练。",
        )

    if not dataset_selector:
        return (
            gr.update(),
            "**状态**: ⚠️ 请先选择训练数据集",
        )

    data_yaml = get_dataset_data_yaml_path(dataset_selector)
    if not data_yaml:
        return (
            gr.update(),
            f"**状态**: ⚠️ 数据集 '{dataset_selector}' 的 data.yaml 不存在",
        )

    model_id = MODEL_VARIANTS.get(model_selector, "yolov8n.pt")
    run_name = generate_run_name(dataset_selector, model_selector)

    # Build params
    params = {
        "model_variant": model_id,
        "data_yaml": data_yaml,
        "epochs": int(epochs_slider),
        "batch": int(batch_slider),
        "imgsz": int(imgsz_dropdown),
        "lr0": float(lr0_number),
        "lrf": float(lrf_number),
        "optimizer": optimizer_dropdown,
        "weight_decay": float(weight_decay_number),
        "momentum": float(momentum_number),
        "warmup_epochs": int(warmup_slider),
        "warmup_momentum": float(warmup_momentum_number),
        "warmup_bias_lr": float(warmup_bias_lr_number),
        "patience": int(patience_slider),
        "close_mosaic": int(close_mosaic_number),
        "workers": int(workers_slider),
        "seed": int(seed_number),
        "dropout": float(dropout_number),
        "cos_lr": bool(cos_lr_cb),
        "amp": bool(amp_cb),
        "multi_scale": bool(multi_scale_cb),
        "pretrained": bool(pretrained_cb),
        "resume": bool(resume_cb),
        "project": "outputs",
        "name": run_name,
    }

    # Create training state
    _active_training = TrainingState()
    _active_training.log_queue = queue.Queue()

    # Start training thread
    thread = threading.Thread(
        target=run_training,
        args=(params, _active_training),
        daemon=True,
    )
    _active_training.thread = thread
    thread.start()

    status_msg = (
        f"**状态**: 🔵 正在训练...\n\n"
        f"模型: {model_selector} | 数据集: {dataset_selector}\n"
        f"Epochs: {epochs_slider} | Batch: {batch_slider} | 图片尺寸: {imgsz_dropdown}"
    )

    return (
        gr.update(choices=list_datasets()),
        status_msg,
    )


def handle_stop_training():
    """Stop the currently running training."""
    global _active_training
    if _active_training is not None and _active_training.status == "running":
        stop_training(_active_training)
        return "**状态**: 🟡 正在停止训练（当前epoch完成后停止）..."
    return "**状态**: ℹ️ 当前没有正在运行的训练"


def handle_monitor_refresh(log_area_value, epoch_progress_value):
    """Poll the training queue and return updated UI values.

    Called every 1 second by Gradio's timer.
    Returns updates for: log_area, epoch_progress, epoch_text, status_text,
      metric_*, gpu_info, training_status
    """
    global _active_training

    if _active_training is None:
        return (
            log_area_value, epoch_progress_value,
            "等待训练开始...", "就绪",
            "—", "—", "—", "—", "—", "—", "—",
            get_device_info(),
            "**状态**: 🟡 就绪",
        )

    # Drain the log queue
    new_lines = []
    new_epoch = epoch_progress_value
    epoch_text_val = "等待训练开始..."
    status_text_val = "就绪"
    map50 = "—"
    map50_95 = "—"
    precision = "—"
    recall = "—"
    box_loss = "—"
    cls_loss = "—"
    dfl_loss = "—"
    training_status_val = "**状态**: 🟡 就绪"

    try:
        while True:
            event = _active_training.log_queue.get_nowait()
            if event["type"] == "log":
                new_lines.append(event["message"])
            elif event["type"] == "start":
                new_epoch = 0
                epoch_text_val = f"0 / {event['total_epochs']}"
                status_text_val = "运行中"
                new_lines.append(f"🏁 训练开始 — 总计 {event['total_epochs']} epochs")
            elif event["type"] == "epoch":
                new_epoch = event["epoch"]
                epoch_text_val = f"{event['epoch']} / {event['total']}"
                status_text_val = "运行中"
                # Extract metric values
                m = event.get("metrics", {})
                map50_val = m.get("metrics/mAP50(B)", m.get("val/box_loss", None))
                map50_95_val = m.get("metrics/mAP50-95(B)", None)
                prec_val = m.get("metrics/precision(B)", None)
                rec_val = m.get("metrics/recall(B)", None)
                box_l = m.get("train/box_loss", m.get("val/box_loss", None))
                cls_l = m.get("train/cls_loss", m.get("val/cls_loss", None))
                dfl_l = m.get("train/dfl_loss", m.get("val/dfl_loss", None))

                if map50_val is not None:
                    map50 = f"{map50_val:.4f}"
                if map50_95_val is not None:
                    map50_95 = f"{map50_95_val:.4f}"
                if prec_val is not None:
                    precision = f"{prec_val:.4f}"
                if rec_val is not None:
                    recall = f"{rec_val:.4f}"
                if box_l is not None:
                    box_loss = f"{box_l:.4f}"
                if cls_l is not None:
                    cls_loss = f"{cls_l:.4f}"
                if dfl_l is not None:
                    dfl_loss = f"{dfl_l:.4f}"

                # Format summary
                parts = []
                for k, v in m.items():
                    short_k = k.split("/")[-1] if "/" in k else k
                    parts.append(f"{short_k}: {v}")
                if parts:
                    new_lines.append(f"📊 Epoch {event['epoch']}/{event['total']} — {', '.join(parts[:6])}")
            elif event["type"] == "end":
                status_text_val = "已完成" if event["reason"] == "completed" else "已停止"
                epoch_text_val = f"{new_epoch} / {_active_training.total_epochs}"
                new_lines.append(f"🏁 训练{status_text_val}")
            elif event["type"] == "error":
                status_text_val = "错误"
                new_lines.append(f"❌ 错误: {event['message']}")
    except queue.Empty:
        pass

    # Update log area
    if new_lines:
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        new_content = "\n".join(f"{timestamp}{line}" for line in new_lines)
        if log_area_value:
            # Keep last ~500 lines
            combined = log_area_value + "\n" + new_content
            lines = combined.split("\n")
            if len(lines) > 500:
                combined = "\n".join(lines[-500:])
            log_area_value = combined
        else:
            log_area_value = new_content

    # Determine status based on training state
    if _active_training:
        if _active_training.status == "running":
            training_status_val = "**状态**: 🔵 训练中..."
            status_text_val = "运行中"
            epoch_text_val = f"{new_epoch} / {_active_training.total_epochs}"
        elif _active_training.status == "completed":
            training_status_val = "**状态**: 🟢 训练完成"
            status_text_val = "已完成"
            epoch_text_val = f"{_active_training.total_epochs} / {_active_training.total_epochs}"
        elif _active_training.status == "stopped":
            training_status_val = "**状态**: 🟠 训练已停止"
            status_text_val = "已停止"
        elif _active_training.status == "error":
            training_status_val = f"**状态**: 🔴 训练出错: {_active_training.error_message}"
            status_text_val = "错误"

    gpu_str = get_device_info()

    return (
        log_area_value, new_epoch,
        epoch_text_val, status_text_val,
        map50, map50_95, precision, recall,
        box_loss, cls_loss, dfl_loss,
        gpu_str,
        training_status_val,
    )


def handle_refresh_runs():
    """Refresh the training runs dropdown."""
    runs = list_subdirs(OUTPUTS_DIR)
    return gr.update(choices=runs)


def handle_select_run(run_name):
    """Display results for a selected training run."""
    if not run_name:
        return "请选择一个训练运行...", None, None, None, None

    output_dir = os.path.join(OUTPUTS_DIR, run_name)
    results = parse_training_results(output_dir)

    # Build info markdown
    md = f"### 📊 训练运行: {run_name}\n\n"

    # Metrics table
    if results["metrics"]:
        md += "| 指标 | 值 |\n|------|------|\n"
        for key, value in list(results["metrics"].items())[:15]:
            try:
                md += f"| {key} | {float(value):.4f} |\n"
            except (ValueError, TypeError):
                md += f"| {key} | {value} |\n"

    if results["best_pt"]:
        md += f"\n**最佳权重**: `{results['best_pt']}`\n"
    if results["last_pt"]:
        md += f"\n**最后权重**: `{results['last_pt']}`\n"

    # Plot images
    plot_images = {}
    plot_keys = ["results", "confusion_matrix", "PR_curve", "F1_curve"]
    for key in plot_keys:
        if key in results["plots"]:
            plot_images[key] = results["plots"][key]

    return (
        md,
        plot_images.get("results"),
        plot_images.get("confusion_matrix"),
        plot_images.get("PR_curve"),
        plot_images.get("F1_curve"),
    )


def handle_refresh_model_status():
    """Refresh the model download status display."""
    status = check_all_models()
    existing = sum(1 for m in status.values() if m["downloaded"])
    total = len(status)

    md = f"### 📊 模型缓存状态 ({existing}/{total} 已下载)\n\n"
    md += "| 模型 | 文件 | 预期大小 | 状态 | 实际大小 |\n"
    md += "|------|------|----------|------|----------|\n"

    # Collect undownloaded model choices
    undownloaded = []

    for name, info in status.items():
        if info["downloaded"]:
            state_badge = "✅ 已下载"
            actual = f"{info['size_mb']:.1f} MB"
        else:
            state_badge = "❌ 未下载"
            actual = "—"
            undownloaded.append(name)

        expected = f"{info['expected_mb']:.0f} MB" if info["expected_mb"] else "?"
        md += f"| {name} | `{info['model_id']}` | {expected} | {state_badge} | {actual} |\n"

    return md, gr.update(choices=undownloaded)


def handle_download_model(model_name: str):
    """Start downloading a single model."""
    global _download_active, _current_model_name, _current_model_id
    global _download_log_lines, _download_thread, _download_queue

    if _download_active:
        return (
            gr.update(),
            gr.update(),
            f"⚠️ 正在下载 {_current_model_name}，请等待完成后再试。\n\n{''.join(_download_log_lines[-5:])}",
        )

    if not model_name:
        return (
            gr.update(),
            gr.update(),
            "⚠️ 请先选择要下载的模型",
        )

    from config import MODEL_VARIANTS
    model_id = MODEL_VARIANTS.get(model_name)
    if not model_id:
        return (
            gr.update(),
            gr.update(),
            f"⚠️ 未知模型: {model_name}",
        )

    _download_queue = queue.Queue()
    _download_active = True
    _current_model_name = model_name
    _current_model_id = model_id
    _download_log_lines = []

    _download_thread = threading.Thread(
        target=download_model,
        args=(model_name, model_id, _download_queue),
        daemon=True,
    )
    _download_thread.start()

    _download_log_lines.append(f"📥 开始下载 {model_name} ({model_id})...\n")

    return (
        gr.update(value=0),
        f"🔄 正在下载 {model_name}...",
        "".join(_download_log_lines),
    )


def handle_download_all():
    """Start downloading all undownloaded models sequentially."""
    global _download_active, _current_model_name, _current_model_id
    global _download_log_lines, _download_thread, _download_queue

    if _download_active:
        return (
            gr.update(),
            gr.update(),
            f"⚠️ 正在下载 {_current_model_name}，请等待完成后再试。\n\n{''.join(_download_log_lines[-5:])}",
        )

    status = check_all_models()
    undownloaded = [(name, info["model_id"]) for name, info in status.items() if not info["downloaded"]]

    if not undownloaded:
        return (
            gr.update(),
            "✅ 所有模型已下载",
            "🎉 全部 10 个模型已缓存，无需下载。\n",
        )

    _download_queue = queue.Queue()
    _download_active = True
    _download_log_lines = []
    _download_log_lines.append(f"📥 准备下载 {len(undownloaded)} 个模型...\n")

    # Start downloading first model
    name, model_id = undownloaded[0]
    _current_model_name = name
    _current_model_id = model_id
    _pending = undownloaded[1:]  # Store remaining for the monitor to pick up

    # We use a list to hold state across the chain of threads
    pending_holder = _pending

    def _chain_downloads(name, model_id, remaining):
        global _download_active, _current_model_name, _current_model_id, _download_log_lines, _download_thread
        q = _download_queue

        download_model(name, model_id, q)

        if remaining:
            next_name, next_id = remaining[0]
            _current_model_name = next_name
            _current_model_id = next_id
            _download_log_lines.append(f"\n📥 开始下载 {next_name} ({next_id})...\n")
            _chain_downloads(next_name, next_id, remaining[1:])
        else:
            _download_active = False
            _download_log_lines.append("\n🎉 所有模型下载完成！\n")

    _download_thread = threading.Thread(
        target=_chain_downloads,
        args=(name, model_id, pending_holder),
        daemon=True,
    )
    _download_thread = None  # Will be set by the chain

    # Actually let's simplify - just download first, timer will handle chain
    _download_thread = threading.Thread(
        target=download_model,
        args=(name, model_id, _download_queue),
        daemon=True,
    )
    _download_thread.start()

    # Store pending list as attribute on the queue for the monitor to access
    _download_queue.pending = pending_holder  # type: ignore[attr-defined]

    return (
        gr.update(value=0),
        f"🔄 正在下载 1/{len(undownloaded)}: {name}...",
        "".join(_download_log_lines),
    )


def handle_download_monitor(progress_val, status_val, log_val):
    """Poll the download queue and update UI. Called by timer every 1 second."""
    global _download_active, _download_thread, _current_model_name, _download_log_lines

    if not _download_active:
        return progress_val, status_val, log_val

    # Drain the queue
    try:
        while True:
            event = _download_queue.get_nowait()
            etype = event.get("type", "")

            if etype == "start":
                _download_log_lines.append(
                    f"📥 开始下载 {event['model_name']} ({event['model_id']})...\n"
                )

            elif etype == "progress":
                progress_val = event["percent"] / 100.0
                total_str = f"/{event['total_mb']:.0f} MB" if event["total_mb"] else ""
                status_val = (
                    f"🔄 下载 {event['model_name']}: {event['downloaded_mb']:.1f}{total_str} MB "
                    f"({event['speed_mbps']:.1f} MB/s, ETA {event['eta_s']:.0f}s)"
                )

            elif etype == "already_cached":
                progress_val = 1.0
                status_val = f"✅ {event['model_name']} 已缓存 ({event['size_mb']:.1f} MB)"
                _download_log_lines.append(
                    f"✅ {event['model_name']} 已在缓存中 ({event['size_mb']:.1f} MB)\n"
                )
                _download_active = False

            elif etype == "complete":
                progress_val = 1.0
                elapsed = event.get("elapsed_s", 0)
                speed = event.get("speed_mbps", 0)
                size = event.get("size_mb", 0)
                status_val = f"✅ {event['model_name']} 下载完成 ({size:.1f} MB, {elapsed:.1f}s, {speed:.1f} MB/s)"
                _download_log_lines.append(
                    f"✅ {event['model_name']} 完成 — {size:.1f} MB, {elapsed:.1f}s, {speed:.1f} MB/s\n"
                )

                # Check for pending downloads (download all mode)
                pending = getattr(_download_queue, "pending", [])
                if pending:
                    next_name, next_id = pending[0]
                    _download_queue.pending = pending[1:]
                    _current_model_name = next_name
                    _download_log_lines.append(f"📥 开始下载 {next_name} ({next_id})...\n")
                    remaining = len(pending)
                    status_val += f"\n🔄 队列中还有 {remaining} 个模型..."
                    _download_thread = threading.Thread(
                        target=download_model,
                        args=(next_name, next_id, _download_queue),
                        daemon=True,
                    )
                    _download_thread.start()
                    progress_val = 0.0
                else:
                    _download_active = False
                    _download_log_lines.append("\n🎉 全部下载完成！\n")

            elif etype == "error":
                status_val = f"❌ {event['model_name']} 失败: {event['message']}"
                _download_log_lines.append(f"{event['message']}\n")
                _download_active = False

    except queue.Empty:
        pass

    # Trim log lines
    log_val = "".join(_download_log_lines[-50:])

    return progress_val, status_val, log_val


def handle_export_model(run_name, export_format):
    """Export a trained model to the selected format."""
    if not run_name:
        return "⚠️ 请先选择一个训练运行"

    output_dir = os.path.join(OUTPUTS_DIR, run_name)
    best_pt = os.path.join(output_dir, "weights", "best.pt")

    if not os.path.isfile(best_pt):
        return f"❌ 未找到模型权重文件: {best_pt}"

    # Map format string to Ultralytics format arg
    format_map = {
        "PyTorch (.pt)": None,  # Already have .pt — just copy
        "ONNX (.onnx)": "onnx",
        "TensorRT (.engine)": "engine",
        "OpenVINO": "openvino",
        "CoreML": "coreml",
        "TFLite": "tflite",
    }

    fmt = format_map.get(export_format)

    try:
        from ultralytics import YOLO
        model = YOLO(best_pt)

        ensure_dir(EXPORTS_DIR)
        export_name = f"{run_name}"

        if fmt is None:
            # Copy the .pt file
            import shutil
            dst = os.path.join(EXPORTS_DIR, f"{export_name}.pt")
            shutil.copy2(best_pt, dst)
            return f"✅ 模型已复制到: `{dst}`"
        else:
            # Export
            export_path = model.export(format=fmt, imgsz=640)
            return f"✅ 模型已导出: `{export_path}`"

    except Exception as e:
        return f"❌ 导出失败: {str(e)}\n\n💡 提示：某些格式需要额外依赖（如 onnx, openvino 等）"


# ================================================================
# DESKTOP CSS
# ================================================================

DESKTOP_CSS = """
/* Full-width desktop layout */
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 12px 24px !important;
}
body { margin: 0; padding: 0; }

/* Header area */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0;
    border-bottom: 1px solid var(--border-color-primary);
    margin-bottom: 20px;
}
.app-header h1 { margin: 0; font-size: 26px; }

/* Metric cards row */
.metrics-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.metrics-row > * {
    flex: 1;
    min-width: 120px;
}

/* Log area styling */
.log-area textarea {
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace !important;
    font-size: 13px !important;
    line-height: 1.4 !important;
    background: #1a1a2e !important;
    color: #e0e0e0 !important;
    border-color: #333 !important;
}

/* Tab content padding */
.tabs > .tabitem > .tabitem {
    padding-top: 16px;
}

/* Section divider */
.section-title {
    font-size: 16px;
    font-weight: 600;
    padding: 8px 0 4px 0;
    margin: 16px 0 8px 0;
    border-bottom: 2px solid var(--primary-400);
    display: inline-block;
}

/* Two-column panel layout */
.panel-left  { min-width: 280px; }
.panel-right { flex: 1; min-width: 400px; }

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 14px;
}
"""

# ================================================================
# MAIN APP SETUP
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO 鱼类识别训练器")
    parser.add_argument("--port", type=int, default=7860, help="WebUI 端口 (默认: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")
    parser.add_argument("--share", action="store_true", help="生成公开分享链接")
    args = parser.parse_args()

    # Ensure directories exist
    for d in ["datasets", "outputs", "exports", "logs"]:
        ensure_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), d))

    print(f"🐟 YOLO 鱼类识别训练器")
    print(f"   设备: {get_device_info()}")
    print(f"   启动 WebUI 在 http://{args.host}:{args.port}")
    print()

    demo = _build_wired_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=DESKTOP_CSS,
    )


def _build_wired_app() -> gr.Blocks:
    """Build the complete desktop-optimized Gradio application."""

    with gr.Blocks(title="YOLO 鱼类识别训练器") as app:

        # ================================================================
        # HEADER — full width with title + device info
        # ================================================================
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                gr.Markdown(
                    """# 🐟 YOLO 鱼类识别训练器
                    上传鱼类数据集，训练YOLOv8/YOLOv11模型，为SeeFish应用提供识别能力。"""
                )
            with gr.Column(scale=1):
                device_info = gr.Textbox(
                    label="🖥️ 设备信息",
                    value=get_device_info(),
                    interactive=False,
                    show_label=True,
                )

        # Hidden state
        training_state = gr.State({
            "running": False, "status": "idle", "run_name": "", "output_dir": "",
        })

        # ================================================================
        # TAB 1: DATASET MANAGEMENT — two-panel desktop layout
        # ================================================================
        with gr.Tab("📦 数据集管理"):
            with gr.Row(equal_height=False):
                # ---- LEFT PANEL: Upload ----
                with gr.Column(scale=2, elem_classes=["panel-left"]):
                    gr.Markdown("### 📋 上传新数据集")
                    zip_file = gr.File(
                        label="选择数据集 ZIP 文件",
                        file_types=[".zip"],
                        file_count="single",
                        height=80,
                    )
                    with gr.Row():
                        dataset_name_input = gr.Textbox(
                            label="数据集名称",
                            placeholder="例如: fish_detection_v1",
                            scale=2,
                        )
                        upload_btn = gr.Button("📤 上传并验证", variant="primary", scale=1)

                    upload_status = gr.Markdown("", container=True)

                    with gr.Accordion("📋 上传格式要求", open=False):
                        gr.Markdown(UPLOAD_REQUIREMENTS)

                # ---- RIGHT PANEL: Browse & manage ----
                with gr.Column(scale=3, elem_classes=["panel-right"]):
                    gr.Markdown("### 📚 已有数据集")
                    with gr.Row():
                        dataset_list_dd = gr.Dropdown(
                            label="选择数据集",
                            choices=list_datasets(),
                            interactive=True,
                            scale=4,
                        )
                        refresh_list_btn = gr.Button("🔄 刷新", variant="secondary", size="sm", scale=1)
                        delete_btn = gr.Button("🗑️ 删除", variant="stop", size="sm", scale=1)

                    delete_confirm = gr.Markdown("")
                    dataset_info_md = gr.Markdown("选择数据集查看详情...", container=True)

            # Wire dataset events
            upload_btn.click(
                fn=handle_upload,
                inputs=[zip_file, dataset_name_input],
                outputs=[upload_status, dataset_list_dd, dataset_info_md],
            )
            refresh_list_btn.click(
                fn=handle_refresh_datasets, inputs=[], outputs=[dataset_list_dd],
            )
            dataset_list_dd.change(
                fn=handle_select_dataset, inputs=[dataset_list_dd], outputs=[dataset_info_md],
            )
            delete_btn.click(
                fn=handle_delete_dataset, inputs=[dataset_list_dd],
                outputs=[delete_confirm, dataset_list_dd],
            )

        # ================================================================
        # TAB 2: MODEL DOWNLOAD — pre-download pretrained weights
        # ================================================================
        with gr.Tab("📥 模型下载"):
            with gr.Row():
                with gr.Column(scale=2):
                    model_status_md = gr.Markdown(
                        "点击 🔄 刷新状态 查看模型缓存情况...",
                        container=True,
                    )
                with gr.Column(scale=1, min_width=220):
                    gr.Markdown("#### 🎯 操作")
                    refresh_models_btn = gr.Button(
                        "🔄 刷新状态", variant="secondary", size="lg",
                    )
                    model_selector_dl = gr.Dropdown(
                        label="选择要下载的模型",
                        choices=list(MODEL_VARIANTS.keys()),
                        interactive=True,
                    )
                    with gr.Row():
                        download_btn = gr.Button(
                            "📥 下载选中模型", variant="primary", size="lg", scale=2,
                        )
                        download_all_btn = gr.Button(
                            "📥 下载全部", variant="secondary", size="lg", scale=1,
                        )

            # Download progress
            download_progress = gr.Slider(
                label="下载进度",
                minimum=0,
                maximum=1.0,
                value=0,
                step=0.01,
                interactive=False,
            )
            download_status = gr.Textbox(
                label="下载状态",
                value="就绪",
                interactive=False,
            )

            gr.Markdown("#### 📝 下载日志")
            download_log = gr.Textbox(
                label="",
                value="",
                interactive=False,
                lines=12,
                max_lines=500,
                elem_classes=["log-area"],
                autoscroll=True,
                show_label=False,
            )

            # Timer for progress polling
            dl_timer = gr.Timer(value=1, active=True)

            # Wire download events
            refresh_models_btn.click(
                fn=handle_refresh_model_status,
                inputs=[],
                outputs=[model_status_md, model_selector_dl],
            )
            download_btn.click(
                fn=handle_download_model,
                inputs=[model_selector_dl],
                outputs=[download_progress, download_status, download_log],
            )
            download_all_btn.click(
                fn=handle_download_all,
                inputs=[],
                outputs=[download_progress, download_status, download_log],
            )
            dl_timer.tick(
                fn=handle_download_monitor,
                inputs=[download_progress, download_status, download_log],
                outputs=[download_progress, download_status, download_log],
            )

        # ================================================================
        # TAB 3: TRAINING CONFIGURATION — compact multi-column grid
        # ================================================================
        with gr.Tab("⚙️ 训练配置"):
            # Row 1: Dataset + Model selectors
            gr.Markdown('<span class="section-title">🎯 基本设置</span>')
            with gr.Row():
                dataset_selector = gr.Dropdown(
                    label="📦 训练数据集", choices=list_datasets(),
                    interactive=True, scale=1,
                )
                model_selector = gr.Dropdown(
                    label="🤖 YOLO 模型", choices=list(MODEL_VARIANTS.keys()),
                    value="YOLOv8n", interactive=True, scale=1,
                )

            # Row 2-4: Hyperparameters in 4-column grid
            gr.Markdown('<span class="section-title">🔧 超参数</span>')
            with gr.Row():
                epochs_slider = gr.Slider(label="Epochs", minimum=1, maximum=500, value=100, step=1)
                batch_slider = gr.Slider(label="Batch Size", minimum=1, maximum=128, value=16, step=1)
                imgsz_dropdown = gr.Dropdown(label="Image Size", choices=IMAGE_SIZE_CHOICES, value=640)
                optimizer_dropdown = gr.Dropdown(label="Optimizer", choices=OPTIMIZER_CHOICES, value="auto")

            with gr.Row():
                lr0_number = gr.Number(label="LR (初始)", value=0.01, minimum=0.0001, maximum=0.1, step=0.001)
                lrf_number = gr.Number(label="LR (最终因子)", value=0.01, minimum=0.001, maximum=0.1, step=0.001)
                weight_decay_number = gr.Number(label="Weight Decay", value=0.0005, minimum=0, maximum=0.01, step=0.0001)
                momentum_number = gr.Number(label="Momentum", value=0.937, minimum=0.5, maximum=0.999, step=0.001)

            with gr.Row():
                warmup_slider = gr.Slider(label="Warmup Epochs", minimum=0, maximum=20, value=3, step=1)
                patience_slider = gr.Slider(label="Patience", minimum=5, maximum=200, value=50, step=5)
                close_mosaic_number = gr.Number(label="Close Mosaic", value=10, minimum=0, maximum=50, step=1)
                workers_slider = gr.Slider(label="Workers", minimum=1, maximum=32, value=8, step=1)

            with gr.Row():
                seed_number = gr.Number(label="Seed", value=0, minimum=0, maximum=9999, step=1)
                dropout_number = gr.Number(label="Dropout", value=0.0, minimum=0.0, maximum=0.5, step=0.05)
                warmup_momentum_number = gr.Number(label="Warmup Momentum", value=0.8, minimum=0.0, maximum=1.0, step=0.1)
                warmup_bias_lr_number = gr.Number(label="Warmup Bias LR", value=0.1, minimum=0.0, maximum=1.0, step=0.01)

            # Advanced options — checkboxes in one row
            gr.Markdown('<span class="section-title">🎛️ 训练选项</span>')
            with gr.Row():
                pretrained_cb = gr.Checkbox(label="预训练权重", value=True)
                amp_cb = gr.Checkbox(label="混合精度 (AMP)", value=True)
                cos_lr_cb = gr.Checkbox(label="余弦学习率", value=False)
                multi_scale_cb = gr.Checkbox(label="多尺度训练", value=False)
                resume_cb = gr.Checkbox(label="从检查点恢复", value=False)

            # Action buttons
            gr.Markdown('<span class="section-title">🚀 训练控制</span>')
            with gr.Row(equal_height=True):
                start_btn = gr.Button("▶️ 开始训练", variant="primary", size="lg", scale=3)
                stop_btn = gr.Button("⏹️ 停止训练", variant="stop", size="lg", scale=1)
                training_status = gr.Markdown("**🟡 就绪**")

            # Wire training events
            start_btn.click(
                fn=handle_start_training,
                inputs=[
                    dataset_selector, model_selector,
                    epochs_slider, batch_slider, imgsz_dropdown, optimizer_dropdown,
                    lr0_number, lrf_number, weight_decay_number, momentum_number,
                    warmup_slider, patience_slider, close_mosaic_number,
                    workers_slider, seed_number,
                    pretrained_cb, amp_cb, cos_lr_cb, multi_scale_cb, resume_cb,
                    dropout_number, warmup_momentum_number, warmup_bias_lr_number,
                ],
                outputs=[dataset_selector, training_status],
            )
            stop_btn.click(
                fn=handle_stop_training, inputs=[], outputs=[training_status],
            )

        # ================================================================
        # TAB 4: TRAINING MONITOR — dashboard + log side-by-side
        # ================================================================
        with gr.Tab("📊 训练监控"):
            # Top: progress bar full-width
            epoch_progress = gr.Slider(
                label="Epoch 进度", minimum=0, maximum=100, value=0, step=1,
                interactive=False,
            )
            with gr.Row():
                epoch_text = gr.Textbox(label="当前进度", value="等待训练开始...", interactive=False, scale=2)
                monitor_status_text = gr.Textbox(label="状态", value="就绪", interactive=False, scale=1)

            # Metrics dashboard (left) + Log area (right)
            with gr.Row(equal_height=False):
                # LEFT: metrics cards in 4x2 grid
                with gr.Column(scale=1, min_width=280):
                    gr.Markdown("#### 📈 实时指标")
                    with gr.Row():
                        metric_mAP50 = gr.Textbox(label="mAP@50", value="—", interactive=False)
                        metric_mAP50_95 = gr.Textbox(label="mAP@50-95", value="—", interactive=False)
                    with gr.Row():
                        metric_precision = gr.Textbox(label="Precision", value="—", interactive=False)
                        metric_recall = gr.Textbox(label="Recall", value="—", interactive=False)
                    with gr.Row():
                        metric_box_loss = gr.Textbox(label="Box Loss", value="—", interactive=False)
                        metric_cls_loss = gr.Textbox(label="Cls Loss", value="—", interactive=False)
                    with gr.Row():
                        metric_dfl_loss = gr.Textbox(label="DFL Loss", value="—", interactive=False)
                        monitor_gpu_info = gr.Textbox(label="GPU 信息", value=get_device_info(), interactive=False)

                # RIGHT: log area
                with gr.Column(scale=2, min_width=400):
                    gr.Markdown("#### 📝 训练日志")
                    log_area = gr.Textbox(
                        label="",
                        value="",
                        interactive=False,
                        lines=28,
                        max_lines=1000,
                        elem_classes=["log-area"],
                        autoscroll=True,
                        show_label=False,
                    )

            # Timer polling
            monitor_timer = gr.Timer(value=1, active=True)
            monitor_timer.tick(
                fn=handle_monitor_refresh,
                inputs=[log_area, epoch_progress],
                outputs=[
                    log_area, epoch_progress, epoch_text, monitor_status_text,
                    metric_mAP50, metric_mAP50_95, metric_precision, metric_recall,
                    metric_box_loss, metric_cls_loss, metric_dfl_loss,
                    monitor_gpu_info, training_status,
                ],
            )

        # ================================================================
        # TAB 5: RESULTS & EXPORT — side panel + 2x2 plot grid
        # ================================================================
        with gr.Tab("📈 结果与导出"):
            # Run selector bar
            with gr.Row():
                run_selector = gr.Dropdown(
                    label="选择训练运行",
                    choices=list_subdirs(OUTPUTS_DIR),
                    interactive=True, scale=4,
                )
                refresh_runs_btn = gr.Button("🔄 刷新", variant="secondary", size="sm", scale=1)

            with gr.Row(equal_height=False):
                # LEFT: run info + export
                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("#### 📊 运行详情")
                    run_info = gr.Markdown("请选择一个训练运行...", container=True)

                    gr.Markdown("#### 📤 导出模型")
                    export_format = gr.Dropdown(
                        label="导出格式", choices=EXPORT_FORMATS, value=EXPORT_FORMATS[0],
                    )
                    export_btn = gr.Button("📤 导出模型", variant="primary", size="lg")
                    export_status = gr.Markdown("")

                # RIGHT: 2x2 plot grid
                with gr.Column(scale=2, min_width=500):
                    gr.Markdown("#### 📈 训练曲线")
                    with gr.Row():
                        results_plot = gr.Image(label="训练结果总览", interactive=False, scale=1)
                        confusion_matrix = gr.Image(label="混淆矩阵", interactive=False, scale=1)
                    with gr.Row():
                        pr_curve = gr.Image(label="PR 曲线", interactive=False, scale=1)
                        f1_curve = gr.Image(label="F1 曲线", interactive=False, scale=1)

            # Wire results events
            refresh_runs_btn.click(
                fn=handle_refresh_runs, inputs=[], outputs=[run_selector],
            )
            run_selector.change(
                fn=handle_select_run, inputs=[run_selector],
                outputs=[run_info, results_plot, confusion_matrix, pr_curve, f1_curve],
            )
            export_btn.click(
                fn=handle_export_model, inputs=[run_selector, export_format],
                outputs=[export_status],
            )

        # ---- App load: refresh dataset, run, and model status lists ----
        app.load(
            fn=lambda: (
                gr.update(choices=list_datasets()),
                gr.update(choices=list_subdirs(OUTPUTS_DIR)),
                *handle_refresh_model_status(),
            ),
            inputs=[],
            outputs=[dataset_list_dd, run_selector, model_status_md, model_selector_dl],
        )

    return app


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
