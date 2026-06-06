"""
YOLO Trainer — UI Components
Builds the complete Gradio WebUI with 4 tabs:
  1. Dataset Management
  2. Training Configuration
  3. Training Monitor
  4. Results & Export
"""

import gradio as gr

from config import (
    MODEL_VARIANTS,
    OPTIMIZER_CHOICES,
    IMAGE_SIZE_CHOICES,
    EXPORT_FORMATS,
    DEFAULT_TRAINING_PARAMS,
    UPLOAD_REQUIREMENTS,
    CUSTOM_CSS,
)


def build_ui() -> gr.Blocks:
    """Construct and return the complete Gradio Blocks application."""

    with gr.Blocks(title="YOLO 鱼类识别训练器") as app:

        # ---- Header ----
        gr.Markdown(
            """
            # 🐟 YOLO 鱼类识别训练器
            上传鱼类数据集，训练YOLOv8/YOLOv11模型，为SeeFish应用提供识别能力。
            """
        )

        # ---- Device Info Bar ----
        device_info = gr.Textbox(
            label="🖥️ 设备信息",
            value="正在检测设备...",
            interactive=False,
            elem_id="device-info",
        )

        # Shared state
        training_state = gr.State({
            "running": False,
            "status": "idle",  # idle | running | completed | error | stopped
            "run_name": "",
            "output_dir": "",
        })

        # =========================================================
        # TAB 1: DATASET MANAGEMENT
        # =========================================================
        with gr.Tab("📦 数据集管理") as tab_dataset:
            build_dataset_tab()

        # =========================================================
        # TAB 2: TRAINING CONFIGURATION
        # =========================================================
        with gr.Tab("⚙️ 训练配置") as tab_training:
            config_components = build_training_tab()

        # =========================================================
        # TAB 3: TRAINING MONITOR
        # =========================================================
        with gr.Tab("📊 训练监控") as tab_monitor:
            monitor_components = build_monitor_tab()

        # =========================================================
        # TAB 4: RESULTS & EXPORT
        # =========================================================
        with gr.Tab("📈 结果与导出") as tab_results:
            results_components = build_results_tab()

        # Return all components needed for wiring (returned via app)
        app._components = {
            "device_info": device_info,
            "training_state": training_state,
            "config": config_components,
            "monitor": monitor_components,
            "results": results_components,
        }

    return app


# ================================================================
# TAB 1: DATASET MANAGEMENT
# ================================================================

def build_dataset_tab():
    """Build the Dataset Management tab components."""

    gr.Markdown("### 📋 上传数据集")

    with gr.Row():
        with gr.Column(scale=2):
            zip_file = gr.File(
                label="选择数据集 ZIP 文件",
                file_types=[".zip"],
                file_count="single",
            )
        with gr.Column(scale=1):
            dataset_name_input = gr.Textbox(
                label="数据集名称",
                placeholder="例如: fish_detection_v1",
            )
            upload_btn = gr.Button("📤 上传并验证", variant="primary", size="lg")

    with gr.Accordion("📋 上传格式要求", open=False):
        gr.Markdown(UPLOAD_REQUIREMENTS)

    upload_status = gr.Markdown("")

    gr.Markdown("### 📚 已有数据集")

    dataset_list = gr.Dropdown(
        label="选择数据集",
        choices=[],
        interactive=True,
    )

    refresh_list_btn = gr.Button("🔄 刷新列表", variant="secondary", size="sm")

    dataset_info = gr.Markdown("选择数据集查看详情...")

    with gr.Row():
        delete_btn = gr.Button("🗑️ 删除选中数据集", variant="stop", size="sm")
        delete_confirm = gr.Markdown("")

    # Store components for wiring
    return {
        "zip_file": zip_file,
        "dataset_name_input": dataset_name_input,
        "upload_btn": upload_btn,
        "upload_status": upload_status,
        "dataset_list": dataset_list,
        "refresh_list_btn": refresh_list_btn,
        "dataset_info": dataset_info,
        "delete_btn": delete_btn,
        "delete_confirm": delete_confirm,
    }


# ================================================================
# TAB 2: TRAINING CONFIGURATION
# ================================================================

def build_training_tab():
    """Build the Training Configuration tab components."""

    gr.Markdown("### 🎯 训练设置")

    with gr.Row():
        dataset_selector = gr.Dropdown(
            label="📦 训练数据集",
            choices=[],
            interactive=True,
            scale=1,
        )
        model_selector = gr.Dropdown(
            label="🤖 YOLO 模型",
            choices=list(MODEL_VARIANTS.keys()),
            value="YOLOv8n",
            interactive=True,
            scale=1,
        )

    gr.Markdown("### 🔧 超参数")

    with gr.Row():
        epochs_slider = gr.Slider(
            label="Epochs (训练轮数)",
            minimum=1, maximum=500, value=DEFAULT_TRAINING_PARAMS["epochs"], step=1,
        )
        batch_slider = gr.Slider(
            label="Batch Size (批次大小)",
            minimum=1, maximum=128, value=DEFAULT_TRAINING_PARAMS["batch"], step=1,
        )

    with gr.Row():
        imgsz_dropdown = gr.Dropdown(
            label="Image Size (图片尺寸)",
            choices=IMAGE_SIZE_CHOICES,
            value=DEFAULT_TRAINING_PARAMS["imgsz"],
        )
        optimizer_dropdown = gr.Dropdown(
            label="Optimizer (优化器)",
            choices=OPTIMIZER_CHOICES,
            value=DEFAULT_TRAINING_PARAMS["optimizer"],
        )

    with gr.Row():
        lr0_number = gr.Number(
            label="Learning Rate lr0 (初始学习率)",
            value=DEFAULT_TRAINING_PARAMS["lr0"],
            minimum=0.0001, maximum=0.1, step=0.001,
        )
        lrf_number = gr.Number(
            label="Learning Rate lrf (最终学习率因子)",
            value=DEFAULT_TRAINING_PARAMS["lrf"],
            minimum=0.001, maximum=0.1, step=0.001,
        )

    with gr.Row():
        weight_decay_number = gr.Number(
            label="Weight Decay (权重衰减)",
            value=DEFAULT_TRAINING_PARAMS["weight_decay"],
            minimum=0, maximum=0.01, step=0.0001,
        )
        momentum_number = gr.Number(
            label="Momentum (动量)",
            value=DEFAULT_TRAINING_PARAMS["momentum"],
            minimum=0.5, maximum=0.999, step=0.001,
        )

    with gr.Row():
        warmup_slider = gr.Slider(
            label="Warmup Epochs (预热轮数)",
            minimum=0, maximum=20, value=DEFAULT_TRAINING_PARAMS["warmup_epochs"], step=1,
        )
        patience_slider = gr.Slider(
            label="Patience (早停耐心值)",
            minimum=5, maximum=200, value=DEFAULT_TRAINING_PARAMS["patience"], step=5,
        )

    with gr.Row():
        close_mosaic_number = gr.Number(
            label="Close Mosaic (关闭马赛克增强)",
            value=DEFAULT_TRAINING_PARAMS["close_mosaic"],
            minimum=0, maximum=50, step=1,
        )
        workers_slider = gr.Slider(
            label="Workers (数据加载线程数)",
            minimum=1, maximum=32, value=DEFAULT_TRAINING_PARAMS["workers"], step=1,
        )
        seed_number = gr.Number(
            label="Seed (随机种子)",
            value=DEFAULT_TRAINING_PARAMS["seed"],
            minimum=0, maximum=9999, step=1,
        )

    gr.Markdown("### 🎛️ 高级选项")

    with gr.Accordion("高级训练选项", open=False):
        with gr.Row():
            pretrained_cb = gr.Checkbox(
                label="使用预训练权重 (Pretrained)",
                value=DEFAULT_TRAINING_PARAMS["pretrained"],
            )
            amp_cb = gr.Checkbox(
                label="混合精度训练 (AMP)",
                value=DEFAULT_TRAINING_PARAMS["amp"],
            )
            cos_lr_cb = gr.Checkbox(
                label="余弦学习率调度 (CosLR)",
                value=DEFAULT_TRAINING_PARAMS["cos_lr"],
            )
            multi_scale_cb = gr.Checkbox(
                label="多尺度训练 (Multi-Scale)",
                value=DEFAULT_TRAINING_PARAMS["multi_scale"],
            )

        with gr.Row():
            resume_cb = gr.Checkbox(
                label="从检查点恢复 (Resume)",
                value=DEFAULT_TRAINING_PARAMS["resume"],
            )
            dropout_number = gr.Number(
                label="Dropout (丢弃率)",
                value=DEFAULT_TRAINING_PARAMS["dropout"],
                minimum=0.0, maximum=0.5, step=0.05,
            )
            warmup_momentum_number = gr.Number(
                label="Warmup Momentum",
                value=DEFAULT_TRAINING_PARAMS["warmup_momentum"],
                minimum=0.0, maximum=1.0, step=0.1,
            )
            warmup_bias_lr_number = gr.Number(
                label="Warmup Bias LR",
                value=DEFAULT_TRAINING_PARAMS["warmup_bias_lr"],
                minimum=0.0, maximum=1.0, step=0.01,
            )

    gr.Markdown("### 🚀 训练控制")

    with gr.Row():
        start_btn = gr.Button("▶️ 开始训练", variant="primary", size="lg", scale=2)
        stop_btn = gr.Button("⏹️ 停止训练", variant="stop", size="lg", scale=1)

    training_status = gr.Markdown("**状态**: 🟡 就绪")

    return {
        "dataset_selector": dataset_selector,
        "model_selector": model_selector,
        "epochs_slider": epochs_slider,
        "batch_slider": batch_slider,
        "imgsz_dropdown": imgsz_dropdown,
        "optimizer_dropdown": optimizer_dropdown,
        "lr0_number": lr0_number,
        "lrf_number": lrf_number,
        "weight_decay_number": weight_decay_number,
        "momentum_number": momentum_number,
        "warmup_slider": warmup_slider,
        "patience_slider": patience_slider,
        "close_mosaic_number": close_mosaic_number,
        "workers_slider": workers_slider,
        "seed_number": seed_number,
        "pretrained_cb": pretrained_cb,
        "amp_cb": amp_cb,
        "cos_lr_cb": cos_lr_cb,
        "multi_scale_cb": multi_scale_cb,
        "resume_cb": resume_cb,
        "dropout_number": dropout_number,
        "warmup_momentum_number": warmup_momentum_number,
        "warmup_bias_lr_number": warmup_bias_lr_number,
        "start_btn": start_btn,
        "stop_btn": stop_btn,
        "training_status": training_status,
    }


# ================================================================
# TAB 3: TRAINING MONITOR
# ================================================================

def build_monitor_tab():
    """Build the Training Monitor tab components."""

    gr.Markdown("### 📊 训练进度")

    with gr.Row():
        progress_bar = gr.Progress(track_tqdm=False)

    with gr.Row():
        epoch_progress = gr.Slider(
            label="Epoch 进度",
            minimum=0, maximum=100, value=0, step=1,
            interactive=False,
        )

    with gr.Row():
        epoch_text = gr.Textbox(label="当前进度", value="等待训练开始...", interactive=False, scale=2)
        status_text = gr.Textbox(label="状态", value="就绪", interactive=False, scale=1)

    gr.Markdown("### 📈 实时指标")

    with gr.Row():
        metric_mAP50 = gr.Textbox(label="mAP@50", value="—", interactive=False, elem_classes=["metric-card"])
        metric_mAP50_95 = gr.Textbox(label="mAP@50-95", value="—", interactive=False)
        metric_precision = gr.Textbox(label="Precision", value="—", interactive=False)
        metric_recall = gr.Textbox(label="Recall", value="—", interactive=False)

    with gr.Row():
        metric_box_loss = gr.Textbox(label="Box Loss", value="—", interactive=False)
        metric_cls_loss = gr.Textbox(label="Cls Loss", value="—", interactive=False)
        metric_dfl_loss = gr.Textbox(label="DFL Loss", value="—", interactive=False)
        gpu_info = gr.Textbox(label="GPU 信息", value="检测中...", interactive=False)

    gr.Markdown("### 📝 训练日志")

    log_area = gr.Textbox(
        label="实时日志",
        value="",
        interactive=False,
        lines=20,
        max_lines=500,
        elem_classes=["log-area"],
        autoscroll=True,
    )

    return {
        "epoch_progress": epoch_progress,
        "epoch_text": epoch_text,
        "status_text": status_text,
        "metric_mAP50": metric_mAP50,
        "metric_mAP50_95": metric_mAP50_95,
        "metric_precision": metric_precision,
        "metric_recall": metric_recall,
        "metric_box_loss": metric_box_loss,
        "metric_cls_loss": metric_cls_loss,
        "metric_dfl_loss": metric_dfl_loss,
        "gpu_info": gpu_info,
        "log_area": log_area,
    }


# ================================================================
# TAB 4: RESULTS & EXPORT
# ================================================================

def build_results_tab():
    """Build the Results & Export tab components."""

    gr.Markdown("### 📊 训练结果")

    with gr.Row():
        run_selector = gr.Dropdown(
            label="选择训练运行",
            choices=[],
            interactive=True,
        )
        refresh_runs_btn = gr.Button("🔄 刷新", variant="secondary", size="sm")

    run_info = gr.Markdown("请选择一个训练运行...")

    gr.Markdown("### 📈 训练曲线")

    with gr.Row():
        results_plot = gr.Image(label="训练结果总览", interactive=False, scale=1)
        confusion_matrix = gr.Image(label="混淆矩阵", interactive=False, scale=1)

    with gr.Row():
        pr_curve = gr.Image(label="PR 曲线", interactive=False, scale=1)
        f1_curve = gr.Image(label="F1 曲线", interactive=False, scale=1)

    gr.Markdown("### 📤 导出模型")

    with gr.Row():
        export_format = gr.Dropdown(
            label="导出格式",
            choices=EXPORT_FORMATS,
            value=EXPORT_FORMATS[0],
        )
        export_btn = gr.Button("📤 导出模型", variant="primary", size="lg")
        export_status = gr.Markdown("")

    return {
        "run_selector": run_selector,
        "refresh_runs_btn": refresh_runs_btn,
        "run_info": run_info,
        "results_plot": results_plot,
        "confusion_matrix": confusion_matrix,
        "pr_curve": pr_curve,
        "f1_curve": f1_curve,
        "export_format": export_format,
        "export_btn": export_btn,
        "export_status": export_status,
    }
