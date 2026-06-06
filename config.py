"""
YOLO Trainer — Configuration
Defines all constants, model variants, hyperparameter presets, and UI choices.
"""

import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ---- Model Variants ----
MODEL_VARIANTS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLOv8l": "yolov8l.pt",
    "YOLOv8x": "yolov8x.pt",
    "YOLOv11n": "yolo11n.pt",
    "YOLOv11s": "yolo11s.pt",
    "YOLOv11m": "yolo11m.pt",
    "YOLOv11l": "yolo11l.pt",
    "YOLOv11x": "yolo11x.pt",
}

# ---- Optimizer Choices ----
OPTIMIZER_CHOICES = ["auto", "SGD", "Adam", "AdamW", "RMSProp"]

# ---- Image Size Choices ----
IMAGE_SIZE_CHOICES = [320, 416, 512, 640, 800, 1024]

# ---- Pretrained Model Sizes (approximate, in MB) ----
MODEL_SIZES = {
    "yolov8n.pt": 6.2,
    "yolov8s.pt": 21.5,
    "yolov8m.pt": 49.7,
    "yolov8l.pt": 83.7,
    "yolov8x.pt": 130.5,
    "yolo11n.pt": 5.3,
    "yolo11s.pt": 18.4,
    "yolo11m.pt": 38.8,
    "yolo11l.pt": 66.5,
    "yolo11x.pt": 109.4,
}

# ---- Export Format Choices ----
EXPORT_FORMATS = [
    "PyTorch (.pt)",
    "ONNX (.onnx)",
    "TensorRT (.engine)",
    "OpenVINO",
    "CoreML",
    "TFLite",
]

# ---- Default Training Parameters ----
DEFAULT_TRAINING_PARAMS = {
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "lr0": 0.01,
    "lrf": 0.01,
    "optimizer": "auto",
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "patience": 50,
    "close_mosaic": 10,
    "workers": 8,
    "seed": 0,
    "dropout": 0.0,
    "cos_lr": False,
    "amp": True,
    "multi_scale": False,
    "pretrained": True,
    "resume": False,
}

# ---- Data Annotation Explanation ----
UPLOAD_REQUIREMENTS = """
### 📋 数据集上传要求

ZIP 文件必须包含以下结构：

```
my_dataset.zip
├── data.yaml          # 数据集配置文件
├── images/
│   ├── train/         # 训练图片
│   └── val/           # 验证图片
└── labels/
    ├── train/         # YOLO格式标注 (.txt)
    └── val/           # YOLO格式标注 (.txt)
```

**data.yaml 格式示例：**
```yaml
path: ./
train: images/train
val: images/val
nc: 2
names:
  0: fish_a
  1: fish_b
```

**YOLO标注格式 (.txt)：** 每行：`class_id cx cy w h`（均为归一化坐标，范围0-1）
"""

# ---- Custom CSS ----
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.log-area textarea {
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace !important;
    font-size: 13px !important;
    line-height: 1.4 !important;
}
.status-idle { color: #6b7280; }
.status-running { color: #3b82f6; }
.status-completed { color: #22c55e; }
.status-error { color: #ef4444; }
.metric-card {
    text-align: center;
    padding: 12px;
    border-radius: 8px;
    background: var(--background-fill-secondary);
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--primary-400);
}
.metric-label {
    font-size: 13px;
    color: var(--body-text-color-subdued);
    margin-top: 4px;
}
.tab-header {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border-color-primary);
}
"""
